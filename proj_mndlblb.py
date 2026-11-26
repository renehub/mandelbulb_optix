#!/usr/bin/env python3
"""
Real-time Mandelbulb ray marcher using NVIDIA OptiX.

This is the full project version: three shadow-casting directional lights,
checkered floor toggle, fractal shininess, and live power / alpha-offset /
beta-offset sliders.  Each fractal parameter can be animated independently
(ping-pong) with its own min/max boundaries and its own animation speed.
Every light has an on/off checkbox and independent RGB controls; only the key
light is enabled by default.  Each light row also has its own Orbit checkbox
and orbit-speed slider.  The fractal material has a colour tint, an optional
glass-refraction mode with adjustable IOR, and the ambient fill has its own
RGB colour.  The control legend lives inside the bottom-left panel, and H
hides/shows all control panels.  OptiX traverses a bounding sphere and the
closest-hit program ray marches the Mandelbulb distance field.

Requirements
------------
    pip install warp-optix pygame-ce pillow numpy

Run
---
    ./.venv/bin/python proj_mndlblb.py

Controls
--------
    Left-drag        orbit around the Mandelbulb
    Right-drag       pan the orbit target
    Mouse wheel      zoom in/out
    Arrow keys       orbit (works while paused)
    + / -            zoom in / out
    Quality slider   render quality, bottom-left
    Shininess slider control the fractal material, bottom-left
    Glass checkbox   switch the fractal to refractive glass, bottom-left
    IOR slider       glass index of refraction, bottom-left
    Lights panel     key/fill/rim checkboxes + RGB, top-left
    Orbit checkbox   per-light orbit toggle
    S sliders        per-light orbit speed, in each light row
    Lighting colours ambient + fractal RGB tint, top-left
    Fractal sliders  power, alpha offset, beta offset, bottom-right
    Anim checkboxes  enable parameter ping-pong; reveals per-slider min/max + speed
    F                toggle the checkered floor
    H                show/hide all control panels
    R                reset camera and light orbit, restart scene clock
    Space            pause/resume
    S                save a screenshot at the current quality
    Esc              quit

The window is resizable; the renderer and camera aspect follow the window
size.  The default quality is 4/10 for interactive use and 10/10 for --still.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

try:
    import warp as wp
    import warp_optix as wo
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    sys.stderr.write(
        f"\nThis interpreter ({sys.executable}) is missing {missing!r}.\n\n"
        "Use the project environment, which already has the OptiX stack:\n"
        "    ./.venv/bin/python proj_mndlblb.py\n\n"
        "Or install the dependencies into the current environment:\n"
        f"    {sys.executable} -m pip install warp-optix pygame-ce pillow\n\n"
        "Note: the PyPI package named 'warp' is not NVIDIA Warp. The correct\n"
        "distribution, 'warp-lang', is installed automatically by 'warp-optix'.\n\n"
    )
    raise SystemExit(1) from None


# ---------------------------------------------------------------------------
# Quality presets
# ---------------------------------------------------------------------------
class QualityPreset(NamedTuple):
    samples: int
    steps: int
    iterations: int


QUALITY_PRESETS: tuple[QualityPreset, ...] = (
    QualityPreset(1, 64, 8),
    QualityPreset(1, 80, 10),
    QualityPreset(1, 112, 12),
    QualityPreset(2, 128, 12),
    QualityPreset(2, 160, 14),
    QualityPreset(3, 192, 16),
    QualityPreset(4, 224, 18),
    QualityPreset(6, 256, 20),
    QualityPreset(8, 320, 24),
    QualityPreset(12, 384, 28),
)
DEFAULT_QUALITY_INDEX = 3
MAX_QUALITY_INDEX = len(QUALITY_PRESETS) - 1


# ---------------------------------------------------------------------------
# Scene and camera constants
# ---------------------------------------------------------------------------
# The classic power-8 bulb fits inside R = 1.3, but variable power and angle
# offsets need a looser bound.  R = 2.0 is the escape radius and safely covers
# the full slider range.
BOUND_RADIUS = 2.0
BOUND_RADIUS_SQ = BOUND_RADIUS * BOUND_RADIUS
BOUND_HALF = BOUND_RADIUS
FLOOR_Y = -2.1
FLOOR_CELL = 1.2
TEAPOT_SCALE = 0.72


def _mandelbulb_bound_radius(power: float) -> float:
    # Solve r**power - r = escape_radius for the first-iteration escape bound.
    # Angle offsets only rotate the result, so this is conservative for all
    # alpha/beta values.  A small margin protects against float rounding.
    lo, hi = 1.0, BOUND_RADIUS
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if mid**power - mid < BOUND_RADIUS:
            lo = mid
        else:
            hi = mid
    return min(BOUND_RADIUS, hi * 1.2)

POWER_MIN = 2.0
POWER_MAX = 16.0
POWER_DEFAULT = 8.0
ANGLE_MIN = -180.0
ANGLE_MAX = 180.0

CAMERA_TARGET = np.array([0.0, -0.30, 0.0], dtype=np.float32)
CAMERA_YAW = 0.55
CAMERA_PITCH = 0.20
CAMERA_DISTANCE = 4.9
CAMERA_FOV_DEG = 40.0
CAMERA_MIN_DISTANCE = 2.6
CAMERA_MAX_DISTANCE = 14.0
CAMERA_MIN_PITCH = 0.02
CAMERA_MAX_PITCH = 1.45


def _norm3(x: float, y: float, z: float) -> tuple[float, float, float]:
    inv = 1.0 / math.sqrt(x * x + y * y + z * z)
    return (x * inv, y * inv, z * inv)


def _cross3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


_KEY_LIGHT_DIR_PY = _norm3(-0.52, 0.74, 0.45)
_FILL_LIGHT_DIR_PY = _norm3(0.66, 0.16, 0.42)
_RIM_LIGHT_DIR_PY = _norm3(0.10, 0.38, -0.90)
_LIGHT_DIRS_PY = (_KEY_LIGHT_DIR_PY, _FILL_LIGHT_DIR_PY, _RIM_LIGHT_DIR_PY)
LIGHT_NAMES = ("Key", "Fill", "Rim")
LIGHT_DEFAULT_COLORS = (
    (1.00, 0.78, 0.56),
    (0.34, 0.56, 1.00),
    (0.26, 1.00, 0.86),
)
LIGHT_DEFAULT_ENABLED = (True, False, False)
DEFAULT_LIGHTS = tuple(
    (enabled, color)
    for enabled, color in zip(LIGHT_DEFAULT_ENABLED, LIGHT_DEFAULT_COLORS)
)
FRACTAL_COLOR_DEFAULT = (1.0, 1.0, 1.0)
AMBIENT_COLOR_DEFAULT = (1.0, 1.0, 1.0)
GLASS_IOR_MIN = 1.0
GLASS_IOR_MAX = 2.0
GLASS_IOR_DEFAULT = 1.45
_SKY_KEY_DIR_PY = _norm3(-0.52, 0.68, 0.42)
_SKY_KEY_TANGENT_PY = _norm3(*_cross3(_SKY_KEY_DIR_PY, (0.0, 1.0, 0.0)))
_SKY_KEY_BI_PY = _cross3(_SKY_KEY_TANGENT_PY, _SKY_KEY_DIR_PY)
_SKY_STRIP_DIR_PY = _norm3(0.70, 0.32, -0.45)
_SKY_STRIP_TANGENT_PY = _norm3(*_cross3(_SKY_STRIP_DIR_PY, (0.0, 1.0, 0.0)))
_SKY_STRIP_BI_PY = _cross3(_SKY_STRIP_TANGENT_PY, _SKY_STRIP_DIR_PY)

SKY_KEY_DIR = wp.constant(wp.vec3(*_SKY_KEY_DIR_PY))
SKY_KEY_TANGENT = wp.constant(wp.vec3(*_SKY_KEY_TANGENT_PY))
SKY_KEY_BI = wp.constant(wp.vec3(*_SKY_KEY_BI_PY))
SKY_STRIP_DIR = wp.constant(wp.vec3(*_SKY_STRIP_DIR_PY))
SKY_STRIP_TANGENT = wp.constant(wp.vec3(*_SKY_STRIP_TANGENT_PY))
SKY_STRIP_BI = wp.constant(wp.vec3(*_SKY_STRIP_BI_PY))


def _smoothstep_py(edge0: float, edge1: float, x: float) -> float:
    t = min(max((x - edge0) / (edge1 - edge0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _softbox_py(
    direction: tuple[float, float, float],
    center: tuple[float, float, float],
    tangent: tuple[float, float, float],
    bitangent: tuple[float, float, float],
    half_width: float,
    half_height: float,
) -> float:
    d_dot_c = direction[0] * center[0] + direction[1] * center[1] + direction[2] * center[2]
    if d_dot_c <= 0.65:
        return 0.0
    inv = 1.0 / max(d_dot_c, 0.035)
    u = (direction[0] * tangent[0] + direction[1] * tangent[1] + direction[2] * tangent[2]) * inv
    v = (direction[0] * bitangent[0] + direction[1] * bitangent[1] + direction[2] * bitangent[2]) * inv
    fu = 1.0 - _smoothstep_py(half_width * 0.82, half_width, abs(u))
    fv = 1.0 - _smoothstep_py(half_height * 0.82, half_height, abs(v))
    return fu * fv


def _eval_sky_color_py(direction: tuple[float, float, float]) -> tuple[float, float, float]:
    up = min(max(direction[1], 0.0), 1.0)
    t = up ** 0.55
    r = 0.012 + (0.001 - 0.012) * t
    g = 0.018 + (0.002 - 0.018) * t
    b = 0.028 + (0.006 - 0.028) * t

    sb_k = _softbox_py(direction, _SKY_KEY_DIR_PY, _SKY_KEY_TANGENT_PY, _SKY_KEY_BI_PY, 0.40, 0.22)
    r += sb_k * 6.5
    g += sb_k * 4.4
    b += sb_k * 2.5

    dk = max(
        direction[0] * _SKY_KEY_DIR_PY[0] + direction[1] * _SKY_KEY_DIR_PY[1] + direction[2] * _SKY_KEY_DIR_PY[2],
        0.0,
    )
    glow = dk ** 18.0
    r += glow * 0.30
    g += glow * 0.18
    b += glow * 0.075

    sb_s = _softbox_py(direction, _SKY_STRIP_DIR_PY, _SKY_STRIP_TANGENT_PY, _SKY_STRIP_BI_PY, 0.12, 0.55)
    r += sb_s * 0.35
    g += sb_s * 1.0
    b += sb_s * 2.2
    return (r, g, b)


_FLOOR_SKY_UP = _eval_sky_color_py((0.0, 1.0, 0.0))


# ---------------------------------------------------------------------------
# Warp / OptiX types
# ---------------------------------------------------------------------------
@wp.struct
class Camera:
    position: wp.vec3
    forward: wp.vec3
    right: wp.vec3
    up: wp.vec3
    tan_half_fov: float
    aspect: float


@wp.struct
class SceneFrame:
    w2l_row0: wp.vec3
    w2l_row1: wp.vec3
    w2l_row2: wp.vec3
    l2w_row0: wp.vec3
    l2w_row1: wp.vec3
    l2w_row2: wp.vec3
    light_local0: wp.vec3
    light_local1: wp.vec3
    light_local2: wp.vec3
    light_world0: wp.vec3
    light_world1: wp.vec3
    light_world2: wp.vec3
    light_color0: wp.vec3
    light_color1: wp.vec3
    light_color2: wp.vec3
    fractal_color: wp.vec3
    ambient_color: wp.vec3
    glass_enabled: int
    glass_ior: float
    teapot_active: int
    teapot_pos: wp.vec3
    teapot_yaw: float
    floor_n_local: wp.vec3
    floor_sky_up: wp.vec3
    fractal_power: float
    fractal_alpha: float
    fractal_beta: float
    bound_radius: float
    bound_radius_sq: float
    alpha8_cos: float
    alpha8_sin: float
    beta8_cos: float
    beta8_sin: float


@wp.struct
class Payload:
    color: wp.vec3


def make_scene_frame(
    spin: float,
    tilt: float,
    power: float = POWER_DEFAULT,
    alpha: float = 0.0,
    beta: float = 0.0,
    lights=DEFAULT_LIGHTS,
    fractal_rgb=FRACTAL_COLOR_DEFAULT,
    ambient_rgb=AMBIENT_COLOR_DEFAULT,
    light_angles=(0.0, 0.0, 0.0),
    glass_enabled: bool = False,
    glass_ior: float = GLASS_IOR_DEFAULT,
    teapot_active: bool = False,
    teapot_pos=(0.0, 0.0, 0.0),
    teapot_yaw: float = 0.0,
) -> SceneFrame:
    ca = math.cos(spin)
    sa = math.sin(spin)
    ct = math.cos(tilt)
    st = math.sin(tilt)

    r0 = (ca, sa * st, -sa * ct)
    r1 = (0.0, ct, st)
    r2 = (sa, -ca * st, ca * ct)

    frame = SceneFrame()
    frame.w2l_row0 = wp.vec3(r0)
    frame.w2l_row1 = wp.vec3(r1)
    frame.w2l_row2 = wp.vec3(r2)
    frame.l2w_row0 = wp.vec3(ca, 0.0, sa)
    frame.l2w_row1 = wp.vec3(sa * st, ct, -ca * st)
    frame.l2w_row2 = wp.vec3(-sa * ct, st, ca * ct)

    locals_ = []
    world_dirs = []
    colors = []
    for index, (light_dir, light_state) in enumerate(zip(_LIGHT_DIRS_PY, lights)):
        enabled, rgb = light_state
        lx, ly, lz = light_dir
        angle = float(light_angles[index]) if light_angles is not None else 0.0
        lca = math.cos(angle)
        lsa = math.sin(angle)
        wx = lx * lca + lz * lsa
        wz = lz * lca - lx * lsa
        world = (wx, ly, wz)
        world_dirs.append(world)
        local = (
            r0[0] * wx + r0[1] * ly + r0[2] * wz,
            r1[0] * wx + r1[1] * ly + r1[2] * wz,
            r2[0] * wx + r2[1] * ly + r2[2] * wz,
        )
        locals_.append(local)
        if enabled:
            colors.append(
                (
                    max(0.0, float(rgb[0])),
                    max(0.0, float(rgb[1])),
                    max(0.0, float(rgb[2])),
                )
            )
        else:
            colors.append((0.0, 0.0, 0.0))

    frame.light_local0 = wp.vec3(*locals_[0])
    frame.light_local1 = wp.vec3(*locals_[1])
    frame.light_local2 = wp.vec3(*locals_[2])
    frame.light_world0 = wp.vec3(*world_dirs[0])
    frame.light_world1 = wp.vec3(*world_dirs[1])
    frame.light_world2 = wp.vec3(*world_dirs[2])
    frame.light_color0 = wp.vec3(*colors[0]) * 4.5
    frame.light_color1 = wp.vec3(*colors[1]) * 4.5
    frame.light_color2 = wp.vec3(*colors[2]) * 4.5
    frame.fractal_color = wp.vec3(
        min(max(float(fractal_rgb[0]), 0.0), 1.0),
        min(max(float(fractal_rgb[1]), 0.0), 1.0),
        min(max(float(fractal_rgb[2]), 0.0), 1.0),
    )
    frame.ambient_color = wp.vec3(
        min(max(float(ambient_rgb[0]), 0.0), 1.0),
        min(max(float(ambient_rgb[1]), 0.0), 1.0),
        min(max(float(ambient_rgb[2]), 0.0), 1.0),
    )
    frame.glass_enabled = 1 if glass_enabled else 0
    frame.glass_ior = min(max(float(glass_ior), GLASS_IOR_MIN), GLASS_IOR_MAX)
    frame.teapot_active = 1 if teapot_active else 0
    frame.teapot_pos = wp.vec3(
        float(teapot_pos[0]),
        float(teapot_pos[1]),
        float(teapot_pos[2]),
    )
    frame.teapot_yaw = float(teapot_yaw)
    frame.floor_n_local = wp.vec3(r0[1], r1[1], r2[1])
    frame.floor_sky_up = wp.vec3(_FLOOR_SKY_UP)
    frame.fractal_power = float(power)
    frame.fractal_alpha = float(alpha)
    frame.fractal_beta = float(beta)
    bound_radius = _mandelbulb_bound_radius(float(power))
    frame.bound_radius = bound_radius
    frame.bound_radius_sq = bound_radius * bound_radius
    frame.alpha8_cos = math.cos(8.0 * float(alpha))
    frame.alpha8_sin = math.sin(8.0 * float(alpha))
    frame.beta8_cos = math.cos(8.0 * float(beta))
    frame.beta8_sin = math.sin(8.0 * float(beta))
    return frame


# ---------------------------------------------------------------------------
# Math, rotation, and studio environment
# ---------------------------------------------------------------------------
@wp.func
def hash_u32(x: wp.uint32) -> wp.uint32:
    x = (x ^ wp.uint32(61)) ^ (x >> wp.uint32(16))
    x *= wp.uint32(9)
    x = x ^ (x >> wp.uint32(4))
    x *= wp.uint32(0x27D4EB2F)
    x = x ^ (x >> wp.uint32(15))
    return x


@wp.func
def rand01(seed: wp.uint32) -> float:
    return float(hash_u32(seed)) * (1.0 / 4294967295.0)


@wp.func
def fract(x: float) -> float:
    return x - wp.floor(x)


@wp.func
def softbox(
    direction: wp.vec3,
    center: wp.vec3,
    tangent: wp.vec3,
    bitangent: wp.vec3,
    half_width: float,
    half_height: float,
) -> float:
    d_dot_c = wp.dot(direction, center)
    if d_dot_c <= 0.65:
        return 0.0
    inv_denom = 1.0 / wp.max(d_dot_c, 0.035)
    u = wp.dot(direction, tangent) * inv_denom
    v = wp.dot(direction, bitangent) * inv_denom
    fu = 1.0 - wp.smoothstep(half_width * 0.82, half_width, wp.abs(u))
    fv = 1.0 - wp.smoothstep(half_height * 0.82, half_height, wp.abs(v))
    return fu * fv


@wp.func
def sky_color(direction: wp.vec3) -> wp.vec3:
    up = wp.clamp(direction[1], 0.0, 1.0)
    color = wp.lerp(
        wp.vec3(0.012, 0.018, 0.028),
        wp.vec3(0.001, 0.002, 0.006),
        wp.pow(up, 0.55),
    )

    # Warm key softbox; mostly visible as a reflection on the fractal.
    color += softbox(direction, SKY_KEY_DIR, SKY_KEY_TANGENT, SKY_KEY_BI, 0.40, 0.22) * wp.vec3(6.5, 4.4, 2.5)

    dk = wp.dot(direction, SKY_KEY_DIR)
    if dk > 0.0:
        dk2 = dk * dk
        dk4 = dk2 * dk2
        dk8 = dk4 * dk4
        dk16 = dk8 * dk8
        color += (dk16 * dk2) * wp.vec3(0.30, 0.18, 0.075)

    # Cool rim strip on the other side.
    color += softbox(direction, SKY_STRIP_DIR, SKY_STRIP_TANGENT, SKY_STRIP_BI, 0.12, 0.55) * wp.vec3(0.35, 1.0, 2.2)
    return color


@wp.func
def background_color(direction: wp.vec3, view_dir: wp.vec3) -> wp.vec3:
    color = sky_color(direction)
    centered = wp.dot(direction, view_dir)
    if centered > 0.0:
        c2 = centered * centered
        c4 = c2 * c2
        c8 = c4 * c4
        c16 = c8 * c8
        color += (c16 * c8 * centered) * wp.vec3(0.055, 0.080, 0.125)
    downward = wp.clamp(-direction[1], 0.0, 1.0)
    color += (downward * downward) * wp.vec3(0.008, 0.006, 0.005)
    return color


@wp.func
def world_to_local_vec(v: wp.vec3, frame: SceneFrame) -> wp.vec3:
    return wp.vec3(
        wp.dot(frame.w2l_row0, v),
        wp.dot(frame.w2l_row1, v),
        wp.dot(frame.w2l_row2, v),
    )


@wp.func
def local_to_world_vec(v: wp.vec3, frame: SceneFrame) -> wp.vec3:
    return wp.vec3(
        wp.dot(frame.l2w_row0, v),
        wp.dot(frame.l2w_row1, v),
        wp.dot(frame.l2w_row2, v),
    )


# ---------------------------------------------------------------------------
# Mandelbulb distance estimator (classic power 8, trig-free)
# ---------------------------------------------------------------------------
@wp.func
def mandelbulb_de_fast(
    p_local: wp.vec3,
    max_iterations: int,
    alpha8_cos: float,
    alpha8_sin: float,
    beta8_cos: float,
    beta8_sin: float,
) -> float:
    # Local world uses Y up; the classic Mandelbulb formula uses Z as its
    # symmetry axis, so swap Y and Z before evaluating the formula.
    c = wp.vec3(p_local[0], p_local[2], p_local[1])
    z = c
    dr = float(1.0)
    r = float(0.0)

    for _ in range(max_iterations):
        x = z[0]
        y = z[1]
        zz = z[2]
        x2 = x * x
        y2 = y * y
        z2 = zz * zz
        w = x2 + y2
        r2 = w + z2
        if r2 > 4.0:
            r = wp.sqrt(r2)
            break
        if r2 < 1.0e-18:
            r = 1.0e-9
            dr = 1.0
            z = c
            continue

        r = wp.sqrt(r2)
        r4 = r2 * r2
        r8 = r4 * r4
        r7 = r4 * r2 * r
        dr = r7 * 8.0 * dr + 1.0

        # Exact trigonometric expansion of (cos(8*theta), sin(8*theta))
        # and (cos(8*phi), sin(8*phi)) via 3 complex squarings.
        inv_r2 = 1.0 / r2
        rxy = wp.sqrt(w)
        ct2 = (z2 - w) * inv_r2
        st2 = (2.0 * zz * rxy) * inv_r2
        ct4 = ct2 * ct2 - st2 * st2
        st4 = 2.0 * ct2 * st2
        ct8_raw = ct4 * ct4 - st4 * st4
        st8_raw = 2.0 * ct4 * st4
        ct8 = ct8_raw * beta8_cos - st8_raw * beta8_sin
        st8 = st8_raw * beta8_cos + ct8_raw * beta8_sin

        if w > 1.0e-24:
            inv_w = 1.0 / w
            c2 = (x2 - y2) * inv_w
            s2 = (2.0 * x * y) * inv_w
            c4 = c2 * c2 - s2 * s2
            s4 = 2.0 * c2 * s2
            c8_raw = c4 * c4 - s4 * s4
            s8_raw = 2.0 * c4 * s4
        else:
            c8_raw = 1.0
            s8_raw = 0.0

        c8 = c8_raw * alpha8_cos - s8_raw * alpha8_sin
        s8 = s8_raw * alpha8_cos + c8_raw * alpha8_sin

        z = wp.vec3(st8 * c8, st8 * s8, ct8) * r8 + c

    return wp.max(0.0, 0.5 * wp.log(wp.max(r, 1.0e-9)) * r / dr)


@wp.func
def mandelbulb_de_generic(
    p_local: wp.vec3,
    max_iterations: int,
    power: float,
    alpha: float,
    beta: float,
) -> float:
    # General power / angle-offset Mandelbulb used by the parameter sliders.
    c = wp.vec3(p_local[0], p_local[2], p_local[1])
    z = c
    dr = float(1.0)
    r = float(0.0)

    for _ in range(max_iterations):
        r2 = wp.dot(z, z)
        if r2 > 4.0:
            r = wp.sqrt(r2)
            break
        if r2 < 1.0e-18:
            r = 1.0e-9
            dr = 1.0
            z = c
            continue

        r = wp.sqrt(r2)
        theta = wp.acos(wp.clamp(z[2] / r, -1.0, 1.0)) + beta
        phi = wp.atan2(z[1], z[0]) + alpha
        rp = wp.pow(r, power - 1.0)
        dr = rp * power * dr + 1.0

        theta *= power
        phi *= power
        st = wp.sin(theta)
        ct = wp.cos(theta)
        sp = wp.sin(phi)
        cp = wp.cos(phi)
        z = wp.vec3(st * cp, st * sp, ct) * (rp * r) + c

    return wp.max(0.0, 0.5 * wp.log(wp.max(r, 1.0e-9)) * r / dr)


@wp.func
def mandelbulb_de(
    p_local: wp.vec3,
    max_iterations: int,
    power: float,
    alpha: float,
    beta: float,
    alpha8_cos: float,
    alpha8_sin: float,
    beta8_cos: float,
    beta8_sin: float,
) -> float:
    # The power-8 path stays trig-free for any alpha/beta offset.
    if wp.abs(power - 8.0) < 1.0e-5:
        return mandelbulb_de_fast(
            p_local,
            max_iterations,
            alpha8_cos,
            alpha8_sin,
            beta8_cos,
            beta8_sin,
        )
    return mandelbulb_de_generic(p_local, max_iterations, power, alpha, beta)


@wp.func
def march_mandelbulb(
    ro_local: wp.vec3,
    rd_local: wp.vec3,
    t_enter: float,
    t_exit: float,
    max_steps: int,
    max_iterations: int,
    power: float,
    alpha: float,
    beta: float,
    alpha8_cos: float,
    alpha8_sin: float,
    beta8_cos: float,
    beta8_sin: float,
) -> wp.vec2:
    t = t_enter + 0.0005
    for _ in range(max_steps):
        p = ro_local + rd_local * t
        d = mandelbulb_de(
            p,
            max_iterations,
            power,
            alpha,
            beta,
            alpha8_cos,
            alpha8_sin,
            beta8_cos,
            beta8_sin,
        )
        if d < 0.0004 + 0.00055 * t:
            return wp.vec2(t, d)
        t += d * 0.85
        if t >= t_exit:
            break
    return wp.vec2(-1.0, 0.0)


@wp.func
def mandelbulb_shadow(
    p_local: wp.vec3,
    n_local: wp.vec3,
    light_dir_local: wp.vec3,
    max_steps: int,
    max_iterations: int,
    power: float,
    alpha: float,
    beta: float,
    bound_radius: float,
    bound_radius_sq: float,
    alpha8_cos: float,
    alpha8_sin: float,
    beta8_cos: float,
    beta8_sin: float,
) -> float:
    ro = p_local + n_local * 0.003

    # Intersect shadow ray against the dynamic Mandelbulb bounding sphere.
    t_c = -wp.dot(ro, light_dir_local)
    closest = ro + light_dir_local * t_c
    d2 = wp.dot(closest, closest)
    if d2 > bound_radius_sq:
        return 1.0
    half_chord = wp.sqrt(wp.max(bound_radius_sq - d2, 0.0))
    t_start = wp.max(0.01, t_c - half_chord)
    t_end = wp.min(3.0, t_c + half_chord)
    if t_start >= t_end:
        return 1.0

    t = t_start
    for _ in range(max_steps):
        d = mandelbulb_de(
            ro + light_dir_local * t,
            max_iterations,
            power,
            alpha,
            beta,
            alpha8_cos,
            alpha8_sin,
            beta8_cos,
            beta8_sin,
        )
        if d < 0.0015 + 0.0005 * t:
            return 0.0
        t += wp.max(d * 0.9, 0.0015)
        if t > t_end:
            break
    return 1.0


@wp.func
def mandelbulb_normal(
    p_local: wp.vec3,
    d0: float,
    max_iterations: int,
    power: float,
    alpha: float,
    beta: float,
    alpha8_cos: float,
    alpha8_sin: float,
    beta8_cos: float,
    beta8_sin: float,
) -> wp.vec3:
    h = 0.0015
    gx = mandelbulb_de(p_local + wp.vec3(h, 0.0, 0.0), max_iterations, power, alpha, beta, alpha8_cos, alpha8_sin, beta8_cos, beta8_sin) - d0
    gy = mandelbulb_de(p_local + wp.vec3(0.0, h, 0.0), max_iterations, power, alpha, beta, alpha8_cos, alpha8_sin, beta8_cos, beta8_sin) - d0
    gz = mandelbulb_de(p_local + wp.vec3(0.0, 0.0, h), max_iterations, power, alpha, beta, alpha8_cos, alpha8_sin, beta8_cos, beta8_sin) - d0
    n = wp.vec3(gx, gy, gz)
    length = wp.length(n)
    if length < 1.0e-9:
        return wp.normalize(p_local + wp.vec3(0.0, 1.0e-6, 0.0))
    return n / length


@wp.func
def mandelbulb_normal_wide(
    p_local: wp.vec3,
    d0: float,
    max_iterations: int,
    power: float,
    alpha: float,
    beta: float,
    alpha8_cos: float,
    alpha8_sin: float,
    beta8_cos: float,
    beta8_sin: float,
) -> wp.vec3:
    h = 0.015
    gx = mandelbulb_de(p_local + wp.vec3(h, 0.0, 0.0), max_iterations, power, alpha, beta, alpha8_cos, alpha8_sin, beta8_cos, beta8_sin) - d0
    gy = mandelbulb_de(p_local + wp.vec3(0.0, h, 0.0), max_iterations, power, alpha, beta, alpha8_cos, alpha8_sin, beta8_cos, beta8_sin) - d0
    gz = mandelbulb_de(p_local + wp.vec3(0.0, 0.0, h), max_iterations, power, alpha, beta, alpha8_cos, alpha8_sin, beta8_cos, beta8_sin) - d0
    n = wp.vec3(gx, gy, gz)
    length = wp.length(n)
    if length < 1.0e-9:
        return wp.normalize(p_local + wp.vec3(0.0, 1.0e-6, 0.0))
    return n / length


@wp.func
def cavity_ao(
    p_local: wp.vec3,
    n_local: wp.vec3,
    max_iterations: int,
    power: float,
    alpha: float,
    beta: float,
    alpha8_cos: float,
    alpha8_sin: float,
    beta8_cos: float,
    beta8_sin: float,
) -> float:
    d1 = mandelbulb_de(p_local + n_local * 0.025, max_iterations, power, alpha, beta, alpha8_cos, alpha8_sin, beta8_cos, beta8_sin)
    d2 = mandelbulb_de(p_local + n_local * 0.075, max_iterations, power, alpha, beta, alpha8_cos, alpha8_sin, beta8_cos, beta8_sin)
    d3 = mandelbulb_de(p_local + n_local * 0.18, max_iterations, power, alpha, beta, alpha8_cos, alpha8_sin, beta8_cos, beta8_sin)
    occ = 1.0 - wp.clamp(d1 / 0.025, 0.0, 1.0)
    occ += 0.65 * (1.0 - wp.clamp(d2 / 0.075, 0.0, 1.0))
    occ += 0.35 * (1.0 - wp.clamp(d3 / 0.18, 0.0, 1.0))
    return wp.clamp(1.0 - 0.50 * occ, 0.15, 1.0)


# ---------------------------------------------------------------------------
# Shading
# ---------------------------------------------------------------------------
@wp.func
def fresnel_schlick(cos_theta: float, f0: wp.vec3) -> wp.vec3:
    t = 1.0 - wp.clamp(cos_theta, 0.0, 1.0)
    t2 = t * t
    f = t2 * t2 * t
    return f0 + (wp.vec3(1.0, 1.0, 1.0) - f0) * f


@wp.func
def specular_ggx(n: wp.vec3, v: wp.vec3, l: wp.vec3, roughness: float, f0: wp.vec3) -> wp.vec3:
    h = wp.normalize(v + l)
    ndl = wp.max(wp.dot(n, l), 0.0)
    ndv = wp.max(wp.dot(n, v), 0.001)
    ndh = wp.max(wp.dot(n, h), 0.0)
    if ndl <= 0.0:
        return wp.vec3(0.0)

    a = roughness * roughness
    a2 = a * a
    denom = ndh * ndh * (a2 - 1.0) + 1.0
    d = a2 / (3.14159265359 * denom * denom)
    k = (roughness + 1.0) * (roughness + 1.0) * 0.125
    g = (ndl / (ndl * (1.0 - k) + k)) * (ndv / (ndv * (1.0 - k) + k))
    f = fresnel_schlick(wp.max(wp.dot(h, v), 0.0), f0)
    return d * g * f / (4.0 * ndv)


@wp.func
def direct_light_term(
    n_world: wp.vec3,
    v: wp.vec3,
    albedo: wp.vec3,
    roughness: float,
    f0: wp.vec3,
    light_dir: wp.vec3,
    radiance: wp.vec3,
    ao: float,
) -> wp.vec3:
    ndl = wp.max(wp.dot(n_world, light_dir), 0.0)
    diffuse = wp.cw_mul(albedo, radiance) * (ndl / 3.14159265359)
    spec = wp.cw_mul(specular_ggx(n_world, v, light_dir, roughness, f0), radiance) * ndl
    return (diffuse + spec) * (0.72 + 0.28 * ao)


@wp.func
def fractal_light_term(
    n_world: wp.vec3,
    v: wp.vec3,
    albedo: wp.vec3,
    roughness: float,
    f0: wp.vec3,
    ao: float,
    shine: float,
    light_dir: wp.vec3,
    light_color: wp.vec3,
    visibility: float,
) -> wp.vec3:
    if visibility <= 0.0 or wp.dot(light_color, light_color) < 1.0e-12:
        return wp.vec3(0.0)
    ndl = wp.max(wp.dot(n_world, light_dir), 0.0)
    if ndl <= 0.0:
        return wp.vec3(0.0)
    diffuse = wp.cw_mul(albedo, light_color) * (ndl / 3.14159265359)
    spec = wp.cw_mul(specular_ggx(n_world, v, light_dir, roughness, f0), light_color) * ndl
    spec_gain = 0.15 + 4.0 * shine
    diffuse_gain = 1.0 - 0.35 * shine
    result = (diffuse * diffuse_gain + spec * spec_gain) * visibility * (0.72 + 0.28 * ao)
    if shine > 0.0:
        sheen = specular_ggx(n_world, v, light_dir, 0.24, wp.vec3(0.10, 0.10, 0.10))
        result += wp.cw_mul(sheen, light_color) * ndl * visibility * (1.8 * shine)
    return result


@wp.func
def shade_mandelbulb(
    p_local: wp.vec3,
    n_local: wp.vec3,
    n_world: wp.vec3,
    rd_world: wp.vec3,
    max_iterations: int,
    visibility0: float,
    visibility1: float,
    visibility2: float,
    shininess: float,
    power: float,
    alpha: float,
    beta: float,
    alpha8_cos: float,
    alpha8_sin: float,
    beta8_cos: float,
    beta8_sin: float,
    light_color0: wp.vec3,
    light_color1: wp.vec3,
    light_color2: wp.vec3,
    fractal_color: wp.vec3,
    ambient_color: wp.vec3,
    light_dir0: wp.vec3,
    light_dir1: wp.vec3,
    light_dir2: wp.vec3,
) -> wp.vec3:
    v = -rd_world
    cos_theta = wp.max(wp.dot(n_world, v), 0.0)
    ao = cavity_ao(
        p_local,
        n_local,
        max_iterations,
        power,
        alpha,
        beta,
        alpha8_cos,
        alpha8_sin,
        beta8_cos,
        beta8_sin,
    )

    # A dark bronze body with a warm metallic specular response.  The small
    # normal-driven blend keeps the intricate surface from reading as one
    # flat colour.
    cool_body = wp.vec3(0.075, 0.145, 0.155)
    warm_body = wp.vec3(0.300, 0.145, 0.045)
    blend = wp.clamp(0.5 + 0.5 * n_world[1], 0.0, 1.0)
    albedo = wp.lerp(cool_body, warm_body, blend)
    f0 = wp.lerp(wp.vec3(0.05, 0.07, 0.065), wp.vec3(0.28, 0.16, 0.045), blend)
    shine = wp.clamp(shininess, 0.0, 1.0)
    roughness = wp.max(0.10, 0.90 - 0.70 * shine)
    f0 = f0 * (0.6 + 0.9 * shine)
    albedo = wp.cw_mul(albedo, fractal_color)
    f0 = wp.cw_mul(f0, fractal_color)

    color = wp.vec3(0.0)
    color += fractal_light_term(n_world, v, albedo, roughness, f0, ao, shine, light_dir0, light_color0, visibility0)
    color += fractal_light_term(n_world, v, albedo, roughness, f0, ao, shine, light_dir1, light_color1, visibility1)
    color += fractal_light_term(n_world, v, albedo, roughness, f0, ao, shine, light_dir2, light_color2, visibility2)

    # Environment reflection plus a sky ambient fill.
    reflected = wp.normalize(rd_world - 2.0 * wp.dot(rd_world, n_world) * n_world)
    env = sky_color(reflected)
    fresnel = fresnel_schlick(cos_theta, f0)
    gloss = 0.20 + 6.0 * shine
    color += wp.cw_mul(env, fresnel) * (0.24 + 0.50 * ao) * gloss
    color += wp.cw_mul(wp.cw_mul(sky_color(n_world), albedo), ambient_color) * (1.35 * ao)

    # A soft sky bounce keeps shadowed areas readable.
    color += wp.cw_mul(wp.cw_mul(albedo, wp.vec3(0.22, 0.27, 0.32)), ambient_color) * (0.60 + 0.40 * ao)

    omc = 1.0 - cos_theta
    omc2 = omc * omc
    rim = omc2 * omc2
    color += rim * shine * wp.vec3(0.55, 0.68, 0.85) * 0.35
    return color


@wp.func
def floor_intersection(ro: wp.vec3, rd: wp.vec3) -> float:
    if wp.abs(rd[1]) < 1.0e-8:
        return -1.0
    t = (FLOOR_Y - ro[1]) / rd[1]
    if t > 0.001:
        return t
    return -1.0


@wp.func
def shade_floor(
    p_world: wp.vec3,
    rd_world: wp.vec3,
    t_floor: float,
    camera: Camera,
    frame: SceneFrame,
    max_steps: int,
    max_iterations: int,
) -> wp.vec3:
    n = wp.vec3(0.0, 1.0, 0.0)
    v = -rd_world

    cell_x = wp.floor(p_world[0] / FLOOR_CELL)
    cell_z = wp.floor(p_world[2] / FLOOR_CELL)
    checker = 0.5 + 0.5 * wp.cos(3.14159265359 * (cell_x + cell_z))
    dark = wp.vec3(0.040, 0.050, 0.062)
    light = wp.vec3(0.245, 0.230, 0.195)
    albedo = wp.lerp(dark, light, checker)
    checker_fade = wp.exp(-t_floor * 0.035)
    albedo = wp.lerp((dark + light) * 0.5, albedo, checker_fade)

    f0 = wp.vec3(0.040, 0.040, 0.040)
    roughness = 0.34

    visibility0 = 1.0
    visibility1 = 1.0
    visibility2 = 1.0
    # Only the central floor area needs shadow marches; the checker fades out
    # at distance anyway.
    if p_world[0] * p_world[0] + p_world[2] * p_world[2] < 30.0:
        shadow_steps = max_steps // 3
        if shadow_steps < 12:
            shadow_steps = 12
        if shadow_steps > 64:
            shadow_steps = 64
        p_local = world_to_local_vec(p_world, frame)
        if wp.dot(frame.light_color0, frame.light_color0) > 1.0e-12 and wp.dot(n, frame.light_world0) > 0.0:
            visibility0 = mandelbulb_shadow(
                p_local, frame.floor_n_local, frame.light_local0,
                shadow_steps, max_iterations,
                frame.fractal_power, frame.fractal_alpha, frame.fractal_beta,
                BOUND_RADIUS, BOUND_RADIUS_SQ,
                frame.alpha8_cos, frame.alpha8_sin,
                frame.beta8_cos, frame.beta8_sin,
            )
        if wp.dot(frame.light_color1, frame.light_color1) > 1.0e-12 and wp.dot(n, frame.light_world1) > 0.0:
            visibility1 = mandelbulb_shadow(
                p_local, frame.floor_n_local, frame.light_local1,
                shadow_steps, max_iterations,
                frame.fractal_power, frame.fractal_alpha, frame.fractal_beta,
                BOUND_RADIUS, BOUND_RADIUS_SQ,
                frame.alpha8_cos, frame.alpha8_sin,
                frame.beta8_cos, frame.beta8_sin,
            )
        if wp.dot(frame.light_color2, frame.light_color2) > 1.0e-12 and wp.dot(n, frame.light_world2) > 0.0:
            visibility2 = mandelbulb_shadow(
                p_local, frame.floor_n_local, frame.light_local2,
                shadow_steps, max_iterations,
                frame.fractal_power, frame.fractal_alpha, frame.fractal_beta,
                BOUND_RADIUS, BOUND_RADIUS_SQ,
                frame.alpha8_cos, frame.alpha8_sin,
                frame.beta8_cos, frame.beta8_sin,
            )

    if frame.glass_enabled != 0:
        visibility0 = 1.0 - (1.0 - visibility0) * 0.45
        visibility1 = 1.0 - (1.0 - visibility1) * 0.45
        visibility2 = 1.0 - (1.0 - visibility2) * 0.45

    color = wp.vec3(0.0)
    if wp.dot(frame.light_color0, frame.light_color0) > 1.0e-12:
        color += direct_light_term(n, v, albedo, roughness, f0, frame.light_world0, frame.light_color0, 1.0) * visibility0
    if wp.dot(frame.light_color1, frame.light_color1) > 1.0e-12:
        color += direct_light_term(n, v, albedo, roughness, f0, frame.light_world1, frame.light_color1, 1.0) * visibility1
    if wp.dot(frame.light_color2, frame.light_color2) > 1.0e-12:
        color += direct_light_term(n, v, albedo, roughness, f0, frame.light_world2, frame.light_color2, 1.0) * visibility2

    floor_ambient = frame.floor_sky_up * 1.15 + wp.vec3(0.075, 0.095, 0.120)
    color += wp.cw_mul(albedo, wp.cw_mul(floor_ambient, frame.ambient_color))
    cos_theta = wp.max(-rd_world[1], 0.0)
    reflected = wp.vec3(rd_world[0], -rd_world[1], rd_world[2])
    color += wp.cw_mul(sky_color(reflected), fresnel_schlick(cos_theta, f0)) * 0.28

    if frame.teapot_active != 0:
        dx = p_world[0] - frame.teapot_pos[0]
        dz = p_world[2] - frame.teapot_pos[2]
        height = wp.max(0.0, frame.teapot_pos[1] - FLOOR_Y)
        radius = 0.36 + 0.22 * height
        dist = wp.sqrt(dx * dx + dz * dz)
        blob = 1.0 - wp.smoothstep(radius * 0.35, radius, dist)
        color = color * (1.0 - (0.42 * blob) / (1.0 + 0.8 * height))

    fog = wp.smoothstep(11.0, 32.0, t_floor)
    color = wp.lerp(color, background_color(rd_world, camera.forward), fog * 0.92)
    return color


@wp.func
def teapot_ellipsoid(p: wp.vec3, center: wp.vec3, radii: wp.vec3) -> float:
    q = p - center
    q = wp.vec3(q[0] / radii[0], q[1] / radii[1], q[2] / radii[2])
    return (wp.length(q) - 1.0) * wp.min(radii[0], wp.min(radii[1], radii[2]))


@wp.func
def teapot_capsule(p: wp.vec3, a: wp.vec3, b: wp.vec3, radius: float) -> float:
    pa = p - a
    ba = b - a
    h = wp.clamp(wp.dot(pa, ba) / wp.max(wp.dot(ba, ba), 1.0e-9), 0.0, 1.0)
    return wp.length(pa - ba * h) - radius


@wp.func
def teapot_torus_xy(p: wp.vec3, center: wp.vec3, major: float, minor: float) -> float:
    q = p - center
    return wp.length(wp.vec2(wp.length(wp.vec2(q[0], q[1])) - major, q[2])) - minor


@wp.func
def teapot_sdf_local(p: wp.vec3) -> float:
    body = teapot_ellipsoid(p, wp.vec3(0.0, 0.36, 0.0), wp.vec3(0.38, 0.31, 0.38))
    base = teapot_ellipsoid(p, wp.vec3(0.0, 0.10, 0.0), wp.vec3(0.31, 0.10, 0.31))
    lid = teapot_ellipsoid(p, wp.vec3(0.0, 0.67, 0.0), wp.vec3(0.24, 0.06, 0.24))
    knob = teapot_ellipsoid(p, wp.vec3(0.0, 0.74, 0.0), wp.vec3(0.06, 0.055, 0.06))
    spout = teapot_capsule(
        p,
        wp.vec3(0.28, 0.36, 0.0),
        wp.vec3(0.72, 0.56, 0.0),
        0.075,
    )
    spout_tip = teapot_ellipsoid(p, wp.vec3(0.72, 0.56, 0.0), wp.vec3(0.08, 0.065, 0.08))
    handle = teapot_torus_xy(p, wp.vec3(-0.34, 0.36, 0.0), 0.27, 0.055)

    d = wp.min(body, base)
    d = wp.min(d, lid)
    d = wp.min(d, knob)
    d = wp.min(d, spout)
    d = wp.min(d, spout_tip)
    d = wp.min(d, handle)
    return d


@wp.func
def world_to_teapot(p_world: wp.vec3, frame: SceneFrame) -> wp.vec3:
    d = p_world - frame.teapot_pos
    ca = wp.cos(frame.teapot_yaw)
    sa = wp.sin(frame.teapot_yaw)
    return wp.vec3(d[0] * ca - d[2] * sa, d[1], d[0] * sa + d[2] * ca)


@wp.func
def teapot_dir_to_world(v: wp.vec3, frame: SceneFrame) -> wp.vec3:
    ca = wp.cos(frame.teapot_yaw)
    sa = wp.sin(frame.teapot_yaw)
    return wp.vec3(v[0] * ca + v[2] * sa, v[1], -v[0] * sa + v[2] * ca)


@wp.func
def teapot_sdf_world(p_world: wp.vec3, frame: SceneFrame) -> float:
    p_local = world_to_teapot(p_world, frame) / TEAPOT_SCALE
    return teapot_sdf_local(p_local) * TEAPOT_SCALE


@wp.func
def teapot_normal_world(p_world: wp.vec3, frame: SceneFrame) -> wp.vec3:
    p_local = world_to_teapot(p_world, frame) / TEAPOT_SCALE
    e = 0.0025
    n_local = wp.vec3(
        teapot_sdf_local(p_local + wp.vec3(e, 0.0, 0.0))
        - teapot_sdf_local(p_local - wp.vec3(e, 0.0, 0.0)),
        teapot_sdf_local(p_local + wp.vec3(0.0, e, 0.0))
        - teapot_sdf_local(p_local - wp.vec3(0.0, e, 0.0)),
        teapot_sdf_local(p_local + wp.vec3(0.0, 0.0, e))
        - teapot_sdf_local(p_local - wp.vec3(0.0, 0.0, e)),
    )
    return teapot_dir_to_world(wp.normalize(n_local), frame)


@wp.func
def teapot_intersect(
    ro: wp.vec3,
    rd: wp.vec3,
    frame: SceneFrame,
    max_steps: int,
) -> wp.vec2:
    if frame.teapot_active == 0:
        return wp.vec2(-1.0, 0.0)

    center = frame.teapot_pos + wp.vec3(0.0, 0.40 * TEAPOT_SCALE, 0.0)
    bounds_radius = 0.95 * TEAPOT_SCALE
    oc = ro - center
    b = wp.dot(oc, rd)
    c = wp.dot(oc, oc) - bounds_radius * bounds_radius
    disc = b * b - c
    if disc < 0.0:
        return wp.vec2(-1.0, 0.0)
    root = wp.sqrt(disc)
    t_start = wp.max(-b - root, 0.001)
    t_end = -b + root
    if t_start >= t_end:
        return wp.vec2(-1.0, 0.0)

    steps = max_steps // 4
    if steps < 24:
        steps = 24
    if steps > 64:
        steps = 64

    t = t_start
    for _ in range(steps):
        d = teapot_sdf_world(ro + rd * t, frame)
        if d < 0.0015 + 0.0005 * t:
            return wp.vec2(t, d)
        t += wp.max(d * 0.8, 0.0015)
        if t > t_end:
            break
    return wp.vec2(-1.0, 0.0)


@wp.func
def shade_teapot(
    p_world: wp.vec3,
    n_world: wp.vec3,
    rd_world: wp.vec3,
    frame: SceneFrame,
) -> wp.vec3:
    n = n_world
    if wp.dot(n, rd_world) > 0.0:
        n = -n
    v = -rd_world
    cos_theta = wp.max(wp.dot(n, v), 0.0)

    blend = wp.clamp(0.5 + 0.5 * n[1], 0.0, 1.0)
    albedo = wp.lerp(
        wp.vec3(0.020, 0.080, 0.450),
        wp.vec3(0.060, 0.300, 0.950),
        blend,
    )
    f0 = wp.vec3(0.07, 0.13, 0.28)
    roughness = 0.30

    color = wp.vec3(0.0)
    color += direct_light_term(n, v, albedo, roughness, f0, frame.light_world0, frame.light_color0, 1.0)
    color += direct_light_term(n, v, albedo, roughness, f0, frame.light_world1, frame.light_color1, 1.0)
    color += direct_light_term(n, v, albedo, roughness, f0, frame.light_world2, frame.light_color2, 1.0)
    color += wp.cw_mul(sky_color(n), albedo) * 0.90
    color += wp.cw_mul(albedo, frame.ambient_color) * 0.18

    reflected = wp.normalize(rd_world - 2.0 * wp.dot(rd_world, n) * n)
    fresnel = fresnel_schlick(cos_theta, f0)
    color += wp.cw_mul(sky_color(reflected), fresnel) * 0.28

    omc = 1.0 - cos_theta
    color += (omc * omc * omc * omc) * wp.vec3(0.15, 0.35, 0.85) * 0.45
    return color


@wp.func
def scene_ray_color(
    origin: wp.vec3,
    direction: wp.vec3,
    camera: Camera,
    frame: SceneFrame,
    floor_enabled: int,
    max_steps: int,
    max_iterations: int,
) -> wp.vec3:
    t_floor = floor_intersection(origin, direction)
    teapot_hit = teapot_intersect(origin, direction, frame, max_steps)
    teapot_t = teapot_hit[0]
    if teapot_t > 0.0 and (floor_enabled == 0 or t_floor < 0.0 or teapot_t < t_floor):
        p_teapot = origin + direction * teapot_t
        n_teapot = teapot_normal_world(p_teapot, frame)
        return shade_teapot(p_teapot, n_teapot, direction, frame)
    if floor_enabled != 0 and t_floor > 0.001:
        return shade_floor(
            origin + direction * t_floor,
            direction,
            t_floor,
            camera,
            frame,
            max_steps,
            max_iterations,
        )
    return background_color(direction, camera.forward)


@wp.func
def glass_light_term(
    n_world: wp.vec3,
    v: wp.vec3,
    light_dir: wp.vec3,
    light_color: wp.vec3,
    visibility: float,
    roughness: float,
    f0: wp.vec3,
) -> wp.vec3:
    if visibility <= 0.0 or wp.dot(light_color, light_color) < 1.0e-12:
        return wp.vec3(0.0)
    ndl = wp.max(wp.dot(n_world, light_dir), 0.0)
    if ndl <= 0.0:
        return wp.vec3(0.0)
    return wp.cw_mul(specular_ggx(n_world, v, light_dir, roughness, f0), light_color) * ndl * visibility


@wp.func
def shade_glass(
    p_world: wp.vec3,
    n_world: wp.vec3,
    n_refract_world: wp.vec3,
    rd_world: wp.vec3,
    visibility0: float,
    visibility1: float,
    visibility2: float,
    shininess: float,
    camera: Camera,
    frame: SceneFrame,
    floor_enabled: int,
    max_steps: int,
    max_iterations: int,
) -> wp.vec3:
    n = n_world
    if wp.dot(n, rd_world) > 0.0:
        n = -n
    n_refract = n_refract_world
    if wp.dot(n_refract, rd_world) > 0.0:
        n_refract = -n_refract
    v = -rd_world
    cos_i = wp.clamp(wp.dot(v, n), 0.0, 1.0)
    cos_i_refract = wp.clamp(wp.dot(v, n_refract), 0.0, 1.0)

    ior = wp.max(frame.glass_ior, 1.0)
    eta = 1.0 / ior
    k = 1.0 - eta * eta * (1.0 - cos_i_refract * cos_i_refract)

    reflect_dir = wp.normalize(rd_world - 2.0 * wp.dot(rd_world, n) * n)
    reflection = scene_ray_color(
        p_world + n * 0.002,
        reflect_dir,
        camera,
        frame,
        floor_enabled,
        max_steps,
        max_iterations,
    )

    if k > 0.0:
        refract_dir = wp.normalize(
            eta * rd_world + (eta * cos_i_refract - wp.sqrt(k)) * n_refract
        )
    else:
        refract_dir = reflect_dir
    transmission = scene_ray_color(
        p_world - n_refract * 0.002,
        refract_dir,
        camera,
        frame,
        floor_enabled,
        max_steps,
        max_iterations,
    )
    straight = scene_ray_color(
        p_world - n_refract * 0.002,
        rd_world,
        camera,
        frame,
        floor_enabled,
        max_steps,
        max_iterations,
    )
    transmission = wp.lerp(transmission, straight, 0.45)

    f0 = ((ior - 1.0) / (ior + 1.0)) * ((ior - 1.0) / (ior + 1.0))
    fresnel = f0 + (1.0 - f0) * wp.pow(1.0 - cos_i, 5.0)
    tint = wp.lerp(wp.vec3(1.0), frame.fractal_color, 0.62)
    transmission = wp.cw_mul(transmission, tint)

    shine = wp.clamp(shininess, 0.0, 1.0)
    roughness = wp.max(0.04, 0.42 - 0.34 * shine)
    spec_f0 = wp.vec3(0.04, 0.04, 0.04) * (0.7 + 0.8 * shine)
    specular = wp.vec3(0.0)
    specular += glass_light_term(n, v, frame.light_world0, frame.light_color0, visibility0, roughness, spec_f0)
    specular += glass_light_term(n, v, frame.light_world1, frame.light_color1, visibility1, roughness, spec_f0)
    specular += glass_light_term(n, v, frame.light_world2, frame.light_color2, visibility2, roughness, spec_f0)

    color = wp.lerp(transmission, reflection, fresnel)
    color += specular * (0.8 + 1.4 * shine)
    return color


# ---------------------------------------------------------------------------
# OptiX programs
# ---------------------------------------------------------------------------
@wp.kernel(module="unique")
def raygen(
    scene: wp.uint64,
    camera: Camera,
    frame: SceneFrame,
    samples: int,
    max_steps: int,
    max_iterations: int,
    fractal_shininess: float,
    floor_enabled: int,
    image: wp.array2d[wp.uint32],
):
    y, x = wp.tid()
    height = image.shape[0]
    width = image.shape[1]

    seed = hash_u32(wp.uint32(y * width + x) ^ wp.uint32(0x9E3779B9))
    pixel_color = wp.vec3(0.0)

    inv_w = 2.0 / float(width)
    inv_h = 2.0 / float(height)
    right_scaled = camera.right * (camera.tan_half_fov * camera.aspect)
    up_scaled = camera.up * camera.tan_half_fov

    for sample_index in range(samples):
        # R2 low-discrepancy subpixel jitter.
        r2_x = fract(0.5 + float(sample_index) * 0.7548776662466927)
        r2_y = fract(0.5 + float(sample_index) * 0.5698402909980532)
        ndc_x = (float(x) + r2_x) * inv_w - 1.0
        ndc_y = 1.0 - (float(y) + r2_y) * inv_h
        direction = wp.normalize(
            camera.forward + right_scaled * ndc_x + up_scaled * ndc_y
        )

        payload = Payload()
        ro = camera.position
        ray_b = wp.dot(ro, direction)
        ray_c = wp.dot(ro, ro) - frame.bound_radius_sq
        bound_disc = ray_b * ray_b - ray_c
        if bound_disc < 0.0 or (-ray_b + wp.sqrt(bound_disc)) < 0.001:
            payload.color = scene_ray_color(
                ro,
                direction,
                camera,
                frame,
                floor_enabled,
                max_steps,
                max_iterations,
            )
        else:
            wp.optix_trace(
                scene,
                ro,
                direction,
                1.0e6,
                payload,
                0.001,
            )
        pixel_color += payload.color

    # Vignette + sample normalization + exposure.
    ndc_x = (float(x) + 0.5) * inv_w - 1.0
    ndc_y = (float(y) + 0.5) * inv_h - 1.0
    vignette = 1.0 - 0.32 * wp.smoothstep(0.55, 1.45, wp.sqrt(ndc_x * ndc_x + ndc_y * ndc_y))
    pixel_color *= (1.05 / float(samples)) * vignette

    # ACES filmic approximation.
    aces = wp.vec3(
        wp.clamp(
            pixel_color[0] * (2.51 * pixel_color[0] + 0.03)
            / (pixel_color[0] * (2.43 * pixel_color[0] + 0.59) + 0.14),
            0.0,
            1.0,
        ),
        wp.clamp(
            pixel_color[1] * (2.51 * pixel_color[1] + 0.03)
            / (pixel_color[1] * (2.43 * pixel_color[1] + 0.59) + 0.14),
            0.0,
            1.0,
        ),
        wp.clamp(
            pixel_color[2] * (2.51 * pixel_color[2] + 0.03)
            / (pixel_color[2] * (2.43 * pixel_color[2] + 0.59) + 0.14),
            0.0,
            1.0,
        ),
    )
    inv_gamma = 1.0 / 2.2
    display = wp.vec3(
        wp.pow(aces[0], inv_gamma),
        wp.pow(aces[1], inv_gamma),
        wp.pow(aces[2], inv_gamma),
    )

    dither = (rand01(hash_u32(seed ^ wp.uint32(0x27D4EB2F))) - 0.5) * 0.35 / 255.0
    r = wp.uint32(wp.clamp(display[0] + dither, 0.0, 1.0) * 255.0 + 0.5)
    g = wp.uint32(wp.clamp(display[1] + dither, 0.0, 1.0) * 255.0 + 0.5)
    b = wp.uint32(wp.clamp(display[2] + dither, 0.0, 1.0) * 255.0 + 0.5)
    image[y, x] = (wp.uint32(255) << wp.uint32(24)) | (b << wp.uint32(16)) | (g << wp.uint32(8)) | r


@wp.kernel(module="unique")
def bounding_box_intersection():
    ro = wp.optix_get_object_ray_origin()
    rd = wp.optix_get_object_ray_direction()
    ray_tmin = wp.optix_get_ray_tmin()
    ray_tmax = wp.optix_get_ray_tmax()

    a = wp.dot(rd, rd)
    if a > 0.0:
        t_c = -wp.dot(ro, rd) / a
        closest = ro + rd * t_c
        d2 = wp.dot(closest, closest)
        if d2 <= BOUND_RADIUS_SQ:
            half_chord = wp.sqrt((BOUND_RADIUS_SQ - d2) / a)
            t0 = t_c - half_chord
            t1 = t_c + half_chord
            if t1 >= ray_tmin and t0 <= ray_tmax:
                entry = wp.max(t0, ray_tmin)
                if entry <= ray_tmax:
                    wp.optix_report_intersection(entry, wp.uint32(0), wp.optix_float_as_uint32(t1))


@wp.kernel(module="unique")
def mandelbulb_closest_hit(
    camera: Camera,
    frame: SceneFrame,
    max_steps: int,
    max_iterations: int,
    fractal_shininess: float,
    floor_enabled: int,
):
    t_enter = wp.optix_get_ray_tmax()
    t_exit = wp.optix_uint32_as_float(wp.optix_get_attribute(0))
    ro = wp.optix_get_world_ray_origin()
    rd = wp.normalize(wp.optix_get_world_ray_direction())

    ro_local = world_to_local_vec(ro, frame)
    rd_local = world_to_local_vec(rd, frame)

    hit = wp.vec2(-1.0, 0.0)
    teapot_hit = teapot_intersect(ro, rd, frame, max_steps)
    teapot_t = teapot_hit[0]
    if teapot_t <= 0.0 or teapot_t > t_enter:
        if teapot_t > 0.0:
            t_exit = wp.min(t_exit, teapot_t)
        hit = march_mandelbulb(
            ro_local,
            rd_local,
            t_enter,
            t_exit,
            max_steps,
            max_iterations,
            frame.fractal_power,
            frame.fractal_alpha,
            frame.fractal_beta,
            frame.alpha8_cos,
            frame.alpha8_sin,
            frame.beta8_cos,
            frame.beta8_sin,
        )
    t_hit = hit[0]

    payload = Payload()
    t_floor = -1.0
    teapot_first = 0
    if t_hit > 0.0 and teapot_t > 0.0 and teapot_t < t_hit:
        teapot_first = 1
    if t_hit < 0.0:
        t_floor = floor_intersection(ro, rd)
        if teapot_t > 0.0 and (floor_enabled == 0 or t_floor < 0.0 or teapot_t < t_floor):
            teapot_first = 1

    if teapot_first != 0:
        p_teapot = ro + rd * teapot_t
        n_teapot = teapot_normal_world(p_teapot, frame)
        payload.color = shade_teapot(p_teapot, n_teapot, rd, frame)
    elif t_hit < 0.0:
        if floor_enabled != 0 and t_floor > 0.0:
            payload.color = shade_floor(
                ro + rd * t_floor,
                rd,
                t_floor,
                camera,
                frame,
                max_steps,
                max_iterations,
            )
        else:
            payload.color = background_color(rd, camera.forward)
    else:
        p_local = ro_local + rd_local * t_hit
        n_local = mandelbulb_normal(
            p_local,
            hit[1],
            max_iterations,
            frame.fractal_power,
            frame.fractal_alpha,
            frame.fractal_beta,
            frame.alpha8_cos,
            frame.alpha8_sin,
            frame.beta8_cos,
            frame.beta8_sin,
        )
        n_world = wp.normalize(local_to_world_vec(n_local, frame))
        if wp.dot(n_world, rd) > 0.0:
            n_world = -n_world

        visibility0 = 0.0
        visibility1 = 0.0
        visibility2 = 0.0
        shadow_steps = max_steps // 2
        if shadow_steps < 16:
            shadow_steps = 16
        if shadow_steps > 96:
            shadow_steps = 96
        if wp.dot(frame.light_color0, frame.light_color0) > 1.0e-12 and wp.dot(n_world, frame.light_world0) > 0.0:
            visibility0 = mandelbulb_shadow(
                p_local, n_local, frame.light_local0,
                shadow_steps, max_iterations,
                frame.fractal_power, frame.fractal_alpha, frame.fractal_beta,
                BOUND_RADIUS, BOUND_RADIUS_SQ,
                frame.alpha8_cos, frame.alpha8_sin,
                frame.beta8_cos, frame.beta8_sin,
            )
        if wp.dot(frame.light_color1, frame.light_color1) > 1.0e-12 and wp.dot(n_world, frame.light_world1) > 0.0:
            visibility1 = mandelbulb_shadow(
                p_local, n_local, frame.light_local1,
                shadow_steps, max_iterations,
                frame.fractal_power, frame.fractal_alpha, frame.fractal_beta,
                BOUND_RADIUS, BOUND_RADIUS_SQ,
                frame.alpha8_cos, frame.alpha8_sin,
                frame.beta8_cos, frame.beta8_sin,
            )
        if wp.dot(frame.light_color2, frame.light_color2) > 1.0e-12 and wp.dot(n_world, frame.light_world2) > 0.0:
            visibility2 = mandelbulb_shadow(
                p_local, n_local, frame.light_local2,
                shadow_steps, max_iterations,
                frame.fractal_power, frame.fractal_alpha, frame.fractal_beta,
                BOUND_RADIUS, BOUND_RADIUS_SQ,
                frame.alpha8_cos, frame.alpha8_sin,
                frame.beta8_cos, frame.beta8_sin,
            )
        p_world = ro + rd * t_hit
        if frame.glass_enabled != 0:
            n_refract_local = mandelbulb_normal_wide(
                p_local,
                hit[1],
                max_iterations,
                frame.fractal_power,
                frame.fractal_alpha,
                frame.fractal_beta,
                frame.alpha8_cos,
                frame.alpha8_sin,
                frame.beta8_cos,
                frame.beta8_sin,
            )
            n_refract_world = wp.normalize(local_to_world_vec(n_refract_local, frame))
            if wp.dot(n_refract_world, rd) > 0.0:
                n_refract_world = -n_refract_world
            payload.color = shade_glass(
                p_world,
                n_world,
                n_refract_world,
                rd,
                visibility0,
                visibility1,
                visibility2,
                fractal_shininess,
                camera,
                frame,
                floor_enabled,
                max_steps,
                max_iterations,
            )
        else:
            payload.color = shade_mandelbulb(
                p_local,
                n_local,
                n_world,
                rd,
                max_iterations,
                visibility0,
                visibility1,
                visibility2,
                fractal_shininess,
                frame.fractal_power,
                frame.fractal_alpha,
                frame.fractal_beta,
                frame.alpha8_cos,
                frame.alpha8_sin,
                frame.beta8_cos,
                frame.beta8_sin,
                frame.light_color0,
                frame.light_color1,
                frame.light_color2,
                frame.fractal_color,
                frame.ambient_color,
                frame.light_world0,
                frame.light_world1,
                frame.light_world2,
            )
    wp.optix_store_payload(payload)


@wp.kernel(module="unique")
def miss(
    camera: Camera,
    frame: SceneFrame,
    max_steps: int,
    max_iterations: int,
    floor_enabled: int,
):
    ro = wp.optix_get_world_ray_origin()
    rd = wp.normalize(wp.optix_get_world_ray_direction())
    payload = Payload()
    t_floor = floor_intersection(ro, rd)
    teapot_hit = teapot_intersect(ro, rd, frame, max_steps)
    if teapot_hit[0] > 0.0 and (floor_enabled == 0 or t_floor < 0.0 or teapot_hit[0] < t_floor):
        p_teapot = ro + rd * teapot_hit[0]
        n_teapot = teapot_normal_world(p_teapot, frame)
        payload.color = shade_teapot(p_teapot, n_teapot, rd, frame)
    elif floor_enabled != 0 and t_floor > 0.0:
        payload.color = shade_floor(
            ro + rd * t_floor,
            rd,
            t_floor,
            camera,
            frame,
            max_steps,
            max_iterations,
        )
    else:
        payload.color = background_color(rd, camera.forward)

    wp.optix_store_payload(payload)


# ---------------------------------------------------------------------------
# Camera rig
# ---------------------------------------------------------------------------
def make_camera(
    width: int,
    height: int,
    target: np.ndarray | None = None,
    yaw: float = CAMERA_YAW,
    pitch: float = CAMERA_PITCH,
    distance: float = CAMERA_DISTANCE,
) -> Camera:
    if target is None:
        target = CAMERA_TARGET

    cos_pitch = math.cos(pitch)
    position = target + distance * np.array(
        [cos_pitch * math.sin(yaw), math.sin(pitch), cos_pitch * math.cos(yaw)],
        dtype=np.float32,
    )
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    cam = Camera()
    cam.position = wp.vec3(position)
    cam.forward = wp.vec3(forward)
    cam.right = wp.vec3(right)
    cam.up = wp.vec3(up)
    cam.tan_half_fov = math.tan(math.radians(CAMERA_FOV_DEG) * 0.5)
    cam.aspect = float(width) / float(height)
    return cam


class OrbitCamera:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.target = CAMERA_TARGET.copy()
        self.yaw = CAMERA_YAW
        self.pitch = CAMERA_PITCH
        self.distance = CAMERA_DISTANCE

    def reset(self):
        self.target = CAMERA_TARGET.copy()
        self.yaw = CAMERA_YAW
        self.pitch = CAMERA_PITCH
        self.distance = CAMERA_DISTANCE

    def resize(self, width: int, height: int):
        self.width = max(1, int(width))
        self.height = max(1, int(height))

    def rotate(self, delta_yaw: float, delta_pitch: float):
        self.yaw += delta_yaw
        self.pitch = min(max(self.pitch + delta_pitch, CAMERA_MIN_PITCH), CAMERA_MAX_PITCH)

    def orbit(self, dx: float, dy: float):
        self.rotate(-dx * 0.006, dy * 0.005)

    def zoom(self, steps: float):
        self.distance = min(
            max(self.distance * math.exp(-0.12 * steps), CAMERA_MIN_DISTANCE),
            CAMERA_MAX_DISTANCE,
        )

    def pan(self, dx: float, dy: float):
        right = np.array([math.cos(self.yaw), 0.0, -math.sin(self.yaw)], dtype=np.float32)
        forward = np.array([-math.sin(self.yaw), 0.0, -math.cos(self.yaw)], dtype=np.float32)
        scale = self.distance * 0.0012
        self.target -= right * dx * scale
        self.target -= forward * dy * scale
        self.target[0] = min(max(self.target[0], -3.0), 3.0)
        self.target[2] = min(max(self.target[2], -3.0), 3.0)

    def camera(self) -> Camera:
        return make_camera(self.width, self.height, self.target, self.yaw, self.pitch, self.distance)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
class MandelbulbRenderer:
    def __init__(
        self,
        width: int,
        height: int,
        quality_index: int = DEFAULT_QUALITY_INDEX,
        device: str = "cuda",
    ):
        self.width = width
        self.height = height
        self.quality_index = min(max(int(quality_index), 0), MAX_QUALITY_INDEX)
        self.device = device
        self.spin = 0.55
        self.tilt = 0.50

        with wp.ScopedDevice(device):
            bounds = wp.array(
                [[-BOUND_HALF, -BOUND_HALF, -BOUND_HALF, BOUND_HALF, BOUND_HALF, BOUND_HALF]],
                dtype=wp.float32,
            )
            self.box = wo.Accel([wo.aabbs(bounds=bounds)], compact=True)
            self.sbt = wo.SbtLayout(
                miss=miss,
                hit_groups={
                    "mandelbulb": wo.HitGroup(
                        intersection=bounding_box_intersection,
                        closest_hit=mandelbulb_closest_hit,
                    )
                },
            )
            self.pipeline = wo.Pipeline(
                raygen,
                sbt=self.sbt,
                payload_words=3,
                attribute_words=1,
                max_trace_depth=1,
            )
            self.image = wp.empty((height, width), dtype=wp.uint32)
            self.camera = make_camera(width, height)

        self.host_image = wp.empty((height, width), dtype=wp.uint32, device="cpu", pinned=True)
        self._host_np = self.host_image.numpy()

    def set_quality(self, index: int):
        self.quality_index = min(max(int(index), 0), MAX_QUALITY_INDEX)

    def resize(self, width: int, height: int):
        width = max(1, int(width))
        height = max(1, int(height))
        if width == self.width and height == self.height:
            return
        self.width = width
        self.height = height
        with wp.ScopedDevice(self.device):
            self.image = wp.empty((height, width), dtype=wp.uint32)
        self.host_image = wp.empty((height, width), dtype=wp.uint32, device="cpu", pinned=True)
        self._host_np = self.host_image.numpy()
        self.camera = make_camera(width, height)

    def update(self, t: float):
        self.spin = 0.55 + 0.18 * t
        self.tilt = 0.45 + 0.18 * math.sin(0.33 * t)

    def render_still(
        self,
        t: float,
        floor_enabled: bool = True,
        fractal_shininess: float = 0.4,
        power: float = POWER_DEFAULT,
        alpha_deg: float = 0.0,
        beta_deg: float = 0.0,
        lights=DEFAULT_LIGHTS,
        fractal_rgb=FRACTAL_COLOR_DEFAULT,
        ambient_rgb=AMBIENT_COLOR_DEFAULT,
        light_angles=(0.0, 0.0, 0.0),
        glass_enabled: bool = False,
        glass_ior: float = GLASS_IOR_DEFAULT,
        teapot_active: bool = False,
        teapot_pos=(0.0, 0.0, 0.0),
        teapot_yaw: float = 0.0,
    ) -> np.ndarray:
        self.update(t)
        return self.render(
            floor_enabled,
            fractal_shininess,
            power,
            alpha_deg,
            beta_deg,
            lights,
            fractal_rgb,
            ambient_rgb,
            light_angles,
            glass_enabled,
            glass_ior,
            teapot_active,
            teapot_pos,
            teapot_yaw,
        )

    def render(
        self,
        floor_enabled: bool = True,
        fractal_shininess: float = 0.4,
        power: float = POWER_DEFAULT,
        alpha_deg: float = 0.0,
        beta_deg: float = 0.0,
        lights=DEFAULT_LIGHTS,
        fractal_rgb=FRACTAL_COLOR_DEFAULT,
        ambient_rgb=AMBIENT_COLOR_DEFAULT,
        light_angles=(0.0, 0.0, 0.0),
        glass_enabled: bool = False,
        glass_ior: float = GLASS_IOR_DEFAULT,
        teapot_active: bool = False,
        teapot_pos=(0.0, 0.0, 0.0),
        teapot_yaw: float = 0.0,
    ) -> np.ndarray:
        quality = QUALITY_PRESETS[self.quality_index]
        frame = make_scene_frame(
            self.spin,
            self.tilt,
            power,
            math.radians(alpha_deg),
            math.radians(beta_deg),
            lights,
            fractal_rgb,
            ambient_rgb,
            light_angles,
            glass_enabled,
            glass_ior,
            teapot_active,
            teapot_pos,
            teapot_yaw,
        )
        with wp.ScopedDevice(self.device):
            wo.launch(
                self.pipeline,
                dim=(self.height, self.width),
                inputs=[
                    self.box,
                    self.camera,
                    frame,
                    int(quality.samples),
                    int(quality.steps),
                    int(quality.iterations),
                    float(fractal_shininess),
                    int(floor_enabled),
                    self.image,
                ],
            )
            wp.copy(self.host_image, self.image)
            wp.synchronize_device(self.device)
        return self._host_np


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
def _pack_rgb(packed: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            (packed & 0xFF).astype(np.uint8),
            ((packed >> 8) & 0xFF).astype(np.uint8),
            ((packed >> 16) & 0xFF).astype(np.uint8),
        ),
        axis=-1,
    )


_TEXT_CACHE = {}


def _render_text(font, text: str, color):
    key = (id(font), text, color)
    surface = _TEXT_CACHE.get(key)
    if surface is None:
        surface = font.render(text, True, color)
        if len(_TEXT_CACHE) >= 2048:
            _TEXT_CACHE.clear()
        _TEXT_CACHE[key] = surface
    return surface


def save_png(packed: np.ndarray, path: Path):
    from PIL import Image

    Image.fromarray(_pack_rgb(packed)).save(path)
    print(f"Saved {path}")


class QualitySlider:
    HELP_LINES = (
        "Drag: orbit   Right-drag: pan   Wheel: zoom",
        "Arrows: orbit   +/-: zoom   R: reset   Space: pause",
        "F: floor   H: panels   S: save   Esc: quit",
        "Animate: checkbox + own min/max/speed",
    )

    def __init__(
        self,
        index: int,
        floor_enabled: bool = True,
        shininess: float = 0.4,
        glass_enabled: bool = False,
        glass_ior: float = GLASS_IOR_DEFAULT,
    ):
        self.index = min(max(index, 0), MAX_QUALITY_INDEX)
        self.floor_enabled = bool(floor_enabled)
        self.shininess = min(max(float(shininess), 0.0), 1.0)
        self.glass_enabled = bool(glass_enabled)
        self.glass_ior = min(max(float(glass_ior), GLASS_IOR_MIN), GLASS_IOR_MAX)
        self.panel_rect = None
        self.track_rect = None
        self.shine_rect = None
        self.glass_rect = None
        self.glass_ior_rect = None
        self.floor_rect = None
        self._panel_surface = None
        self._static = None

    def layout(self, width: int, height: int):
        panel_w = min(430, max(260, width - 28))
        panel_h = 204
        panel_x = 14
        panel_y = height - panel_h - 14
        self.panel_rect = (panel_x, panel_y, panel_w, panel_h)
        track_x = panel_x + 14
        track_w = panel_w - 28
        self.track_rect = (track_x, panel_y + 32 - 8, track_w, 16)
        material_gap = 22
        material_w = max(72, (track_w - material_gap) // 2)
        glass_x = track_x + material_w + material_gap
        self.shine_rect = (track_x, panel_y + 85 - 8, material_w, 16)
        self.glass_ior_rect = (glass_x, panel_y + 85 - 8, material_w, 16)
        self.glass_rect = (glass_x, panel_y + 60, 16, 16)
        self.floor_rect = (panel_x + 14, panel_y + 98, 18, 18)
        self._panel_surface = None

    def panel_hit(self, pos: tuple[int, int]) -> bool:
        if self.panel_rect is None:
            return False
        x, y, w, h = self.panel_rect
        return x <= pos[0] <= x + w and y <= pos[1] <= y + h

    @staticmethod
    def _rect_hit(rect, pos: tuple[int, int]) -> bool:
        if rect is None:
            return False
        x, y, w, h = rect
        return x - 10 <= pos[0] <= x + w + 10 and y <= pos[1] <= y + h

    def hit(self, pos: tuple[int, int]) -> bool:
        return self._rect_hit(self.track_rect, pos)

    def hit_shininess(self, pos: tuple[int, int]) -> bool:
        return self._rect_hit(self.shine_rect, pos)

    def hit_glass(self, pos: tuple[int, int]) -> bool:
        if self.glass_rect is None:
            return False
        x, y, w, h = self.glass_rect
        return x - 5 <= pos[0] <= x + w + 5 and y <= pos[1] <= y + h

    def hit_glass_ior(self, pos: tuple[int, int]) -> bool:
        return self._rect_hit(self.glass_ior_rect, pos)

    def hit_floor(self, pos: tuple[int, int]) -> bool:
        if self.floor_rect is None:
            return False
        x, y, w, h = self.floor_rect
        return x - 4 <= pos[0] <= x + w + 4 and y - 4 <= pos[1] <= y + h + 4

    @staticmethod
    def _value_from_x(rect, x_pos: float, vmin: float, vmax: float) -> float:
        x, _, w, _ = rect
        t = min(max((x_pos - x) / max(1.0, float(w)), 0.0), 1.0)
        return vmin + t * (vmax - vmin)

    def set_from_x(self, x_pos: float) -> bool:
        if self.track_rect is None:
            return False
        t = self._value_from_x(self.track_rect, x_pos, 0.0, 1.0)
        index = int(round(t * (len(QUALITY_PRESETS) - 1)))
        changed = index != self.index
        self.index = index
        return changed

    def set_shininess_from_x(self, x_pos: float) -> bool:
        if self.shine_rect is None:
            return False
        value = self._value_from_x(self.shine_rect, x_pos, 0.0, 1.0)
        changed = abs(value - self.shininess) > 1.0e-4
        self.shininess = value
        return changed

    def set_glass_ior_from_x(self, x_pos: float) -> bool:
        if self.glass_ior_rect is None:
            return False
        value = self._value_from_x(self.glass_ior_rect, x_pos, GLASS_IOR_MIN, GLASS_IOR_MAX)
        value = round(value, 2)
        changed = abs(value - self.glass_ior) > 1.0e-4
        self.glass_ior = value
        return changed

    def draw(self, screen, font, small_font, width: int):
        import pygame

        if self.panel_rect is None:
            self.layout(width, screen.get_height())
        x, y, w, h = self.panel_rect

        if self._panel_surface is None:
            panel = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(panel, (5, 9, 14, 170), panel.get_rect(), border_radius=10)
            self._panel_surface = panel

        if self._static is None:
            self._static = {
                "quality": font.render("Render quality", True, (230, 237, 247)),
                "shininess": font.render("Shininess", True, (230, 237, 247)),
                "glass": font.render("Glass", True, (230, 237, 247)),
                "floor": small_font.render("Checkered floor", True, (220, 230, 242)),
                "floor_key": small_font.render("F", True, (124, 226, 185)),
                "help": [small_font.render(line, True, (166, 180, 198)) for line in self.HELP_LINES],
            }

        screen.blit(self._panel_surface, (x, y))

        def draw_track(rect, t, accent, knob_color=(235, 244, 250), knob_dark=(20, 30, 38)):
            tx, ty, tw, th = rect
            cy = ty + th // 2
            pygame.draw.rect(screen, (36, 44, 57), (tx, cy - 2, tw, 4), border_radius=2)
            fill = int(tw * min(max(t, 0.0), 1.0))
            if fill > 0:
                pygame.draw.rect(screen, accent, (tx, cy - 2, fill, 4), border_radius=2)
            pygame.draw.circle(screen, knob_color, (tx + fill, cy), 7)
            pygame.draw.circle(screen, knob_dark, (tx + fill, cy), 4)

        static = self._static
        screen.blit(static["quality"], (x + 14, y + 8))
        q = QUALITY_PRESETS[self.index]
        qv = _render_text(font, f"{self.index + 1}/{len(QUALITY_PRESETS)}", (124, 226, 185))
        qd = _render_text(
            small_font,
            f"{q.samples} spp  ·  {q.steps} steps  ·  {q.iterations} iterations",
            (172, 184, 201),
        )
        screen.blit(qv, (x + w - qv.get_width() - 14, y + 8))
        draw_track(self.track_rect, self.index / max(1, len(QUALITY_PRESETS) - 1), (93, 202, 169))
        screen.blit(qd, (x + 14, y + 41))
        pygame.draw.line(screen, (33, 42, 53), (x + 14, y + 57), (x + w - 14, y + 57), 1)

        screen.blit(static["shininess"], (x + 14, y + 62))
        shine_pct = int(round(self.shininess * 100.0))
        sv = _render_text(font, f"{shine_pct}%", (238, 185, 103))
        screen.blit(sv, (self.shine_rect[0] + self.shine_rect[2] - sv.get_width(), y + 62))
        draw_track(self.shine_rect, self.shininess, (224, 173, 94), knob_color=(245, 243, 236), knob_dark=(54, 40, 24))

        gx, gy, gw, gh = self.glass_rect
        pygame.draw.rect(screen, (48, 60, 74), (gx, gy, gw, gh), border_radius=4)
        pygame.draw.rect(screen, (118, 136, 158), (gx, gy, gw, gh), 1, border_radius=4)
        if self.glass_enabled:
            pygame.draw.rect(screen, (93, 202, 169), (gx + 1, gy + 1, gw - 2, gh - 2), border_radius=3)
            pygame.draw.rect(screen, (155, 230, 202), (gx, gy, gw, gh), 1, border_radius=4)
            pygame.draw.lines(
                screen,
                (12, 32, 30),
                False,
                [(gx + 3, gy + 8), (gx + 7, gy + 12), (gx + 13, gy + 4)],
                2,
            )
        screen.blit(static["glass"], (gx + 20, y + 62))
        ior_text = _render_text(
            font,
            f"{self.glass_ior:.2f}",
            (115, 205, 255) if self.glass_enabled else (130, 144, 162),
        )
        screen.blit(
            ior_text,
            (self.glass_ior_rect[0] + self.glass_ior_rect[2] - ior_text.get_width(), y + 62),
        )
        ior_t = (self.glass_ior - GLASS_IOR_MIN) / (GLASS_IOR_MAX - GLASS_IOR_MIN)
        draw_track(
            self.glass_ior_rect,
            ior_t,
            (115, 205, 255) if self.glass_enabled else (58, 68, 84),
        )

        bx, by, bw, bh = self.floor_rect
        pygame.draw.rect(screen, (59, 72, 88), (bx, by, bw, bh), border_radius=4)
        if self.floor_enabled:
            pygame.draw.rect(screen, (93, 202, 169), (bx + 1, by + 1, bw - 2, bh - 2), border_radius=3)
            pygame.draw.lines(
                screen,
                (12, 32, 30),
                False,
                [(bx + 3, by + 9), (bx + 7, by + 13), (bx + 15, by + 4)],
                2,
            )
        screen.blit(static["floor"], (x + 40, y + 99))
        screen.blit(static["floor_key"], (x + w - 24, y + 99))

        pygame.draw.line(screen, (33, 42, 53), (x + 14, y + 120), (x + w - 14, y + 120), 1)
        for i, line in enumerate(static["help"]):
            screen.blit(line, (x + 14, y + 128 + i * 19))

class LightsPanel:
    ROW_H = 48
    HEADER_H = 26
    BOTTOM = 10
    LIGHT_COUNT = 3
    ORBIT_SPEED_MIN = 0.0
    ORBIT_SPEED_MAX = 2.0
    ORBIT_SPEED_DEFAULT = 0.35

    def __init__(
        self,
        colors=None,
        enabled=None,
        ambient_rgb=AMBIENT_COLOR_DEFAULT,
        fractal_rgb=FRACTAL_COLOR_DEFAULT,
    ):
        colors = LIGHT_DEFAULT_COLORS if colors is None else colors
        enabled = LIGHT_DEFAULT_ENABLED if enabled is None else enabled
        self.colors = [
            [min(max(float(channel), 0.0), 1.0) for channel in color]
            for color in colors
        ]
        self.enabled = [bool(value) for value in enabled]
        self.ambient_rgb = [min(max(float(channel), 0.0), 1.0) for channel in ambient_rgb]
        self.fractal_rgb = [min(max(float(channel), 0.0), 1.0) for channel in fractal_rgb]
        self.orbit_enabled = [False for _ in range(self.LIGHT_COUNT)]
        self.orbit_speeds = [self.ORBIT_SPEED_DEFAULT for _ in range(self.LIGHT_COUNT)]
        self.orbit_angles = [0.0 for _ in range(self.LIGHT_COUNT)]
        self.panel_rect = None
        self.rows = []
        self.ambient_row = None
        self.fractal_row = None
        self._panel_surface = None
        self._static = None

    def layout(self, width: int, height: int):
        panel_w = min(300, max(210, (width - 42) // 2))
        panel_h = self.HEADER_H + (self.LIGHT_COUNT + 2) * self.ROW_H + self.BOTTOM
        panel_x = 14
        panel_y = 14
        self.panel_rect = (panel_x, panel_y, panel_w, panel_h)

        self.rows = []
        for i in range(self.LIGHT_COUNT):
            top = panel_y + self.HEADER_H + i * self.ROW_H
            self.rows.append(
                {
                    "top": top,
                    "check": (panel_x + 14, top + 2, 16, 16),
                    "orbit_check": (panel_x + panel_w - 96, top + 2, 16, 16),
                    "swatch": (panel_x + panel_w - 52, top + 4, 38, 12),
                }
            )

        ambient_top = panel_y + self.HEADER_H + self.LIGHT_COUNT * self.ROW_H
        self.ambient_row = {
            "top": ambient_top,
            "swatch": (panel_x + panel_w - 52, ambient_top + 4, 38, 12),
        }
        fractal_top = ambient_top + self.ROW_H
        self.fractal_row = {
            "top": fractal_top,
            "swatch": (panel_x + panel_w - 52, fractal_top + 4, 38, 12),
        }
        self._update_tracks()
        self._panel_surface = None

    def _update_tracks(self):
        if self.panel_rect is None:
            return
        x, _, w, _ = self.panel_rect
        track_x = x + 14
        track_w = w - 28

        light_gap = 6
        light_col_w = max(28, (track_w - 3 * light_gap) // 4)
        for row in self.rows:
            rects = {}
            for j, channel in enumerate(("r", "g", "b", "s")):
                xx = track_x + j * (light_col_w + light_gap)
                rects[channel] = (xx, row["top"] + 27, light_col_w, 14)
            row["color_rects"] = rects
            row["col_w"] = light_col_w

        extra_gap = 8
        extra_col_w = max(30, (track_w - 2 * extra_gap) // 3)
        for row in (self.ambient_row, self.fractal_row):
            if row is None:
                continue
            rects = {}
            for j, channel in enumerate(("r", "g", "b")):
                xx = track_x + j * (extra_col_w + extra_gap)
                rects[channel] = (xx, row["top"] + 27, extra_col_w, 14)
            row["color_rects"] = rects
            row["col_w"] = extra_col_w

    @staticmethod
    def _rect_hit(rect, pos: tuple[int, int], pad: int = 10) -> bool:
        if rect is None:
            return False
        x, y, w, h = rect
        return x - pad <= pos[0] <= x + w + pad and y - 2 <= pos[1] <= y + h + 2

    def panel_hit(self, pos: tuple[int, int]) -> bool:
        if self.panel_rect is None:
            return False
        x, y, w, h = self.panel_rect
        return x <= pos[0] <= x + w and y <= pos[1] <= y + h

    def hit_toggle(self, index: int, pos: tuple[int, int]) -> bool:
        return self._rect_hit(self.rows[index]["check"], pos, pad=5)

    def hit_color(self, index: int, channel: str, pos: tuple[int, int]) -> bool:
        if not 0 <= index < len(self.rows):
            return False
        return self._rect_hit(self.rows[index]["color_rects"][channel], pos, pad=3)

    def hit_ambient(self, channel: str, pos: tuple[int, int]) -> bool:
        if self.ambient_row is None:
            return False
        return self._rect_hit(self.ambient_row["color_rects"][channel], pos, pad=3)

    def hit_fractal(self, channel: str, pos: tuple[int, int]) -> bool:
        if self.fractal_row is None:
            return False
        return self._rect_hit(self.fractal_row["color_rects"][channel], pos, pad=3)

    def hit_orbit(self, index: int, pos: tuple[int, int]) -> bool:
        if not 0 <= index < len(self.rows):
            return False
        return self._rect_hit(self.rows[index]["orbit_check"], pos, pad=5)

    def hit_light_speed(self, index: int, pos: tuple[int, int]) -> bool:
        if not 0 <= index < len(self.rows):
            return False
        return self._rect_hit(self.rows[index]["color_rects"]["s"], pos, pad=3)

    def toggle_enabled(self, index: int) -> bool:
        self.enabled[index] = not self.enabled[index]
        return True

    def toggle_orbit(self, index: int) -> bool:
        self.orbit_enabled[index] = not self.orbit_enabled[index]
        return True

    def reset_orbit(self):
        self.orbit_angles = [0.0 for _ in range(self.LIGHT_COUNT)]

    def advance_orbit(self, dt: float) -> bool:
        if dt <= 0.0:
            return False
        changed = False
        for i, speed in enumerate(self.orbit_speeds):
            if not self.orbit_enabled[i] or speed <= 0.0:
                continue
            self.orbit_angles[i] = (self.orbit_angles[i] + speed * dt) % (2.0 * math.pi)
            changed = True
        return changed

    def orbit_state(self):
        return tuple(self.orbit_angles)

    @staticmethod
    def _rgb_from_x(rect, x_pos: float, old: float) -> tuple[float, bool]:
        x, _, w, _ = rect
        t = min(max((x_pos - x) / max(1.0, float(w)), 0.0), 1.0)
        value = round(t * 100.0) / 100.0
        return value, abs(value - old) > 1.0e-4

    def _set_light_color_from_x(self, index: int, channel: str, x_pos: float) -> bool:
        rect = self.rows[index]["color_rects"][channel]
        channel_index = {"r": 0, "g": 1, "b": 2}[channel]
        value, changed = self._rgb_from_x(rect, x_pos, self.colors[index][channel_index])
        self.colors[index][channel_index] = value
        return changed

    def _set_extra_from_x(self, row, rgb, channel: str, x_pos: float) -> bool:
        if row is None:
            return False
        rect = row["color_rects"][channel]
        channel_index = {"r": 0, "g": 1, "b": 2}[channel]
        value, changed = self._rgb_from_x(rect, x_pos, rgb[channel_index])
        rgb[channel_index] = value
        return changed

    def set_color_from_x(self, index: int, channel: str, x_pos: float) -> bool:
        return self._set_light_color_from_x(index, channel, x_pos)

    def set_ambient_from_x(self, channel: str, x_pos: float) -> bool:
        return self._set_extra_from_x(self.ambient_row, self.ambient_rgb, channel, x_pos)

    def set_fractal_from_x(self, channel: str, x_pos: float) -> bool:
        return self._set_extra_from_x(self.fractal_row, self.fractal_rgb, channel, x_pos)

    def set_light_speed_from_x(self, index: int, x_pos: float) -> bool:
        rect = self.rows[index]["color_rects"]["s"]
        x, _, w, _ = rect
        t = min(max((x_pos - x) / max(1.0, float(w)), 0.0), 1.0)
        speed = round(self.ORBIT_SPEED_MIN + t * (self.ORBIT_SPEED_MAX - self.ORBIT_SPEED_MIN), 3)
        changed = abs(speed - self.orbit_speeds[index]) > 1.0e-4
        self.orbit_speeds[index] = speed
        return changed

    def state(self):
        return tuple(
            (bool(self.enabled[i]), tuple(self.colors[i]))
            for i in range(self.LIGHT_COUNT)
        )

    def ambient_state(self):
        return tuple(self.ambient_rgb)

    def fractal_state(self):
        return tuple(self.fractal_rgb)

    def draw(self, screen, font, small_font, width: int):
        import pygame

        if self.panel_rect is None:
            self.layout(width, screen.get_height())
        x, y, w, h = self.panel_rect

        if self._panel_surface is None:
            panel = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(panel, (5, 9, 14, 170), panel.get_rect(), border_radius=10)
            self._panel_surface = panel

        if self._static is None:
            self._static = {
                "title": font.render("Lighting & Color", True, (230, 237, 247)),
                "hint": small_font.render("S = orbit speed", True, (132, 149, 168)),
                "orbit": small_font.render("Orbit", True, (214, 226, 240)),
                "names_on": [font.render(name, True, (224, 233, 244)) for name in LIGHT_NAMES],
                "names_off": [font.render(name, True, (112, 126, 143)) for name in LIGHT_NAMES],
                "ambient": font.render("Ambient", True, (224, 233, 244)),
                "fractal": font.render("Fractal", True, (224, 233, 244)),
                "rgb": [
                    small_font.render("R", True, (235, 105, 98)),
                    small_font.render("G", True, (105, 220, 132)),
                    small_font.render("B", True, (110, 155, 255)),
                ],
                "speed_on": small_font.render("S", True, (245, 184, 96)),
                "speed_off": small_font.render("S", True, (112, 126, 143)),
            }

        screen.blit(self._panel_surface, (x, y))
        screen.blit(self._static["title"], (x + 14, y + 7))
        if w >= 250:
            hint = self._static["hint"]
            screen.blit(hint, (x + w - hint.get_width() - 14, y + 9))

        def draw_checkbox(rect, checked, accent):
            bx, by, bw, bh = rect
            pygame.draw.rect(screen, (48, 60, 74), (bx, by, bw, bh), border_radius=4)
            pygame.draw.rect(screen, (118, 136, 158), (bx, by, bw, bh), 1, border_radius=4)
            if checked:
                pygame.draw.rect(screen, accent, (bx + 1, by + 1, bw - 2, bh - 2), border_radius=3)
                pygame.draw.rect(screen, (155, 230, 202), (bx, by, bw, bh), 1, border_radius=4)
                pygame.draw.lines(
                    screen,
                    (12, 32, 30),
                    False,
                    [(bx + 3, by + 8), (bx + 7, by + 12), (bx + 13, by + 4)],
                    2,
                )

        orbit_label = self._static["orbit"]

        def draw_track(rect, t, accent):
            tx, ty, tw, th = rect
            cy = ty + th // 2
            pygame.draw.rect(screen, (36, 44, 57), (tx, cy - 2, tw, 4), border_radius=2)
            fill = int(tw * min(max(t, 0.0), 1.0))
            if fill > 0:
                pygame.draw.rect(screen, accent, (tx, cy - 2, fill, 4), border_radius=2)
            pygame.draw.circle(screen, (240, 246, 252), (tx + fill, cy), 6)
            pygame.draw.circle(screen, (22, 31, 42), (tx + fill, cy), 3)

        accents = ((231, 91, 86), (92, 208, 120), (103, 143, 255))
        for i, row in enumerate(self.rows):
            top = row["top"]
            enabled = self.enabled[i]

            draw_checkbox(row["check"], enabled, (93, 202, 169))
            name = self._static["names_on" if enabled else "names_off"][i]
            screen.blit(name, (x + 40, top - 2))

            orbit_check = row["orbit_check"]
            draw_checkbox(orbit_check, self.orbit_enabled[i], (93, 202, 169))
            screen.blit(orbit_label, (orbit_check[0] - 6 - orbit_label.get_width(), top + 2))

            sx, sy, sw, sh = row["swatch"]
            swatch = tuple(int(round(c * 255.0)) for c in self.colors[i])
            if not enabled:
                swatch = tuple(int(round(c * 0.42)) for c in swatch)
            pygame.draw.rect(screen, swatch, (sx, sy, sw, sh), border_radius=3)
            pygame.draw.rect(screen, (80, 95, 114), (sx, sy, sw, sh), 1, border_radius=3)

            for j, channel in enumerate(("r", "g", "b", "s")):
                rect = row["color_rects"][channel]
                col_w = row["col_w"]
                rx = rect[0]
                if channel == "s":
                    active = enabled and self.orbit_enabled[i]
                    label_surf = self._static["speed_on" if active else "speed_off"]
                    speed = self.orbit_speeds[i]
                    value_t = (speed - self.ORBIT_SPEED_MIN) / (self.ORBIT_SPEED_MAX - self.ORBIT_SPEED_MIN)
                    value_text = f"{speed:.2f}"
                    accent = (245, 184, 96) if active else (58, 68, 84)
                else:
                    active = enabled
                    label_surf = self._static["rgb"][j]
                    value_t = self.colors[i][j]
                    value_text = str(int(round(self.colors[i][j] * 100.0)))
                    accent = accents[j] if enabled else (58, 68, 84)
                screen.blit(label_surf, (rx, top + 18))
                vtxt = _render_text(
                    small_font,
                    value_text,
                    (214, 226, 240) if active else (130, 144, 162),
                )
                screen.blit(vtxt, (rx + col_w - vtxt.get_width(), top + 18))
                draw_track(rect, value_t, accent)

        def draw_channel_sliders(row, rgb, enabled=True):
            for j, channel in enumerate(("r", "g", "b")):
                rect = row["color_rects"][channel]
                col_w = row["col_w"]
                rx = rect[0]
                screen.blit(self._static["rgb"][j], (rx, row["top"] + 18))
                value = int(round(rgb[j] * 100.0))
                vtxt = _render_text(small_font, str(value), (214, 226, 240) if enabled else (130, 144, 162))
                screen.blit(vtxt, (rx + col_w - vtxt.get_width(), row["top"] + 18))
                draw_track(rect, rgb[j], accents[j] if enabled else (58, 68, 84))

        if self.ambient_row is not None:
            ambient_top = self.ambient_row["top"]
            pygame.draw.line(screen, (33, 42, 53), (x + 14, ambient_top - 6), (x + w - 14, ambient_top - 6), 1)

            for row, name_surf, rgb in (
                (self.ambient_row, self._static["ambient"], self.ambient_rgb),
                (self.fractal_row, self._static["fractal"], self.fractal_rgb),
            ):
                top = row["top"]
                screen.blit(name_surf, (x + 14, top - 2))
                sx, sy, sw, sh = row["swatch"]
                swatch = tuple(int(round(c * 255.0)) for c in rgb)
                pygame.draw.rect(screen, swatch, (sx, sy, sw, sh), border_radius=3)
                pygame.draw.rect(screen, (80, 95, 114), (sx, sy, sw, sh), 1, border_radius=3)
                draw_channel_sliders(row, rgb, True)

class FractalParamsPanel:
    PARAMS = (
        ("power", "Power", POWER_MIN, POWER_MAX, (245, 184, 96)),
        ("alpha", "Alpha offset", ANGLE_MIN, ANGLE_MAX, (114, 218, 157)),
        ("beta", "Beta offset", ANGLE_MIN, ANGLE_MAX, (150, 166, 255)),
    )
    TITLE_H = 28
    ROW_H = 36
    BOUND_H = 24
    SPEED_H = 22
    BOTTOM = 10
    SPEED_MIN = 0.01
    SPEED_MAX = 0.8
    SPEED_DEFAULT = 0.15

    def __init__(self, power: float = POWER_DEFAULT, alpha: float = 0.0, beta: float = 0.0):
        self.power = min(max(float(power), POWER_MIN), POWER_MAX)
        self.alpha = min(max(float(alpha), ANGLE_MIN), ANGLE_MAX)
        self.beta = min(max(float(beta), ANGLE_MIN), ANGLE_MAX)
        self.anim = {}
        for key, _label, vmin, vmax, _accent in self.PARAMS:
            value = getattr(self, key)
            span = max(vmax - vmin, 1.0e-9)
            self.anim[key] = {
                "enabled": False,
                "lo": vmin,
                "hi": vmax,
                "phase": min(max((value - vmin) / span, 0.0), 1.0),
                "dir": 1,
                "speed": self.SPEED_DEFAULT,
            }
        self.panel_rect = None
        self.rows = {}
        self._panel_surface = None
        self._width = 0
        self._height = 0

    def _meta(self, key: str):
        for meta in self.PARAMS:
            if meta[0] == key:
                return meta
        raise KeyError(key)

    def _value(self, key: str) -> float:
        return getattr(self, key)

    def _set_value(self, key: str, value: float):
        setattr(self, key, value)

    @staticmethod
    def _angle_text(value: float) -> str:
        iv = int(round(value))
        return "0°" if iv == 0 else f"{iv:+d}°"

    def _fmt(self, key: str, value: float) -> str:
        if key == "power":
            return f"{value:.1f}"
        return self._angle_text(value)

    def _round_value(self, key: str, value: float) -> float:
        _key, _label, vmin, vmax, _accent = self._meta(key)
        value = min(max(value, vmin), vmax)
        if key == "power":
            return round(value, 1)
        return float(round(value))

    def _norm(self, key: str, value: float) -> float:
        _key, _label, vmin, vmax, _accent = self._meta(key)
        span = max(vmax - vmin, 1.0e-9)
        return min(max((value - vmin) / span, 0.0), 1.0)

    def _any_animation(self) -> bool:
        return any(st["enabled"] for st in self.anim.values())

    def layout(self, width: int, height: int):
        self._width = width
        self._height = height
        enabled_count = sum(1 for st in self.anim.values() if st["enabled"])
        panel_h = (
            self.TITLE_H
            + len(self.PARAMS) * self.ROW_H
            + enabled_count * (self.BOUND_H + self.SPEED_H)
            + self.BOTTOM
        )

        if width >= 780:
            panel_w = 300
            panel_x = width - panel_w - 14
            panel_y = height - panel_h - 14
        else:
            # Keep the fractal controls on the right, below the FPS badge.
            # The lights panel uses the mirror position on the left.
            panel_w = min(300, max(210, (width - 42) // 2))
            panel_x = width - panel_w - 14
            panel_y = 48
        panel_y = max(14, panel_y)
        self.panel_rect = (panel_x, panel_y, panel_w, panel_h)

        self.rows = {}
        cursor = panel_y + self.TITLE_H
        for key, label, _vmin, _vmax, accent in self.PARAMS:
            row = {
                "top": cursor,
                "label": label,
                "accent": accent,
                "check": (panel_x + 14, cursor, 16, 16),
                "track": (panel_x + 42, cursor + 20 - 8, panel_w - 56, 16),
            }
            self.rows[key] = row
            cursor += self.ROW_H

            if self.anim[key]["enabled"]:
                gap = 10
                half_w = (panel_w - 28 - gap) // 2
                min_x = panel_x + 14
                max_x = min_x + half_w + gap
                label_w = 20
                value_w = 38
                track_w = max(30, half_w - label_w - value_w)
                track_y = cursor + 12 - 7
                row["bound_top"] = cursor
                row["half_w"] = half_w
                row["min_x"] = min_x
                row["max_x"] = max_x
                row["min_track"] = (min_x + label_w, track_y, track_w, 14)
                row["max_track"] = (max_x + label_w, track_y, track_w, 14)
                cursor += self.BOUND_H

                speed_label_w = 44
                speed_value_w = 48
                speed_track_x = min_x + speed_label_w
                speed_track_w = max(40, panel_w - 28 - speed_label_w - speed_value_w)
                row["speed_top"] = cursor
                row["speed_track"] = (speed_track_x, cursor + 11 - 7, speed_track_w, 14)
                cursor += self.SPEED_H

        self.panel_rect = (panel_x, panel_y, panel_w, cursor - panel_y + self.BOTTOM)
        self._panel_surface = None

    @staticmethod
    def _rect_hit(rect, pos: tuple[int, int]) -> bool:
        if rect is None:
            return False
        x, y, w, h = rect
        return x - 10 <= pos[0] <= x + w + 10 and y <= pos[1] <= y + h

    def panel_hit(self, pos: tuple[int, int]) -> bool:
        if self.panel_rect is None:
            return False
        x, y, w, h = self.panel_rect
        return x <= pos[0] <= x + w and y <= pos[1] <= y + h

    def hit_checkbox(self, key: str, pos: tuple[int, int]) -> bool:
        row = self.rows.get(key)
        if not row:
            return False
        x, y, w, h = row["check"]
        return x - 2 <= pos[0] <= x + w + 2 and y - 2 <= pos[1] <= y + h + 2

    def hit_track(self, key: str, pos: tuple[int, int]) -> bool:
        row = self.rows.get(key)
        if not row:
            return False
        return self._rect_hit(row["track"], pos)

    def hit_bound(self, key: str, which: str, pos: tuple[int, int]) -> bool:
        row = self.rows.get(key)
        if not row or "min_track" not in row:
            return False
        return self._rect_hit(row["min_track" if which == "min" else "max_track"], pos)

    def hit_speed(self, key: str, pos: tuple[int, int]) -> bool:
        row = self.rows.get(key)
        if not row or "speed_track" not in row:
            return False
        return self._rect_hit(row["speed_track"], pos)

    @staticmethod
    def _value_from_x(rect, x_pos: float, vmin: float, vmax: float) -> float:
        x, _, w, _ = rect
        t = min(max((x_pos - x) / max(1.0, float(w)), 0.0), 1.0)
        return vmin + t * (vmax - vmin)

    def set_value_from_x(self, key: str, x_pos: float) -> bool:
        row = self.rows.get(key)
        if not row:
            return False
        _key, _label, vmin, vmax, _accent = self._meta(key)
        value = self._round_value(key, self._value_from_x(row["track"], x_pos, vmin, vmax))
        changed = abs(value - self._value(key)) > 1.0e-4
        self._set_value(key, value)
        return changed

    def disable_animation(self, key: str) -> bool:
        st = self.anim.get(key)
        if st is None or not st["enabled"]:
            return False
        st["enabled"] = False
        self.layout(self._width, self._height)
        return True

    def toggle_animation(self, key: str) -> bool:
        st = self.anim.get(key)
        if st is None:
            return False
        st["enabled"] = not st["enabled"]
        changed = False
        if st["enabled"]:
            lo, hi = st["lo"], st["hi"]
            old = self._value(key)
            value = min(max(old, lo), hi)
            if abs(value - old) > 1.0e-6:
                self._set_value(key, value)
                changed = True
            span = hi - lo
            phase = 0.0 if span <= 1.0e-9 else min(max((value - lo) / span, 0.0), 1.0)
            st["phase"] = phase
            st["dir"] = -1 if phase >= 0.999 else 1
        self.layout(self._width, self._height)
        return changed

    def set_bound_from_x(self, key: str, which: str, x_pos: float) -> bool:
        row = self.rows.get(key)
        st = self.anim.get(key)
        if row is None or st is None or "min_track" not in row:
            return False
        _key, _label, vmin, vmax, _accent = self._meta(key)
        rect = row["min_track" if which == "min" else "max_track"]
        value = self._round_value(key, self._value_from_x(rect, x_pos, vmin, vmax))
        if which == "min":
            value = min(value, st["hi"])
            st["lo"] = value
        else:
            value = max(value, st["lo"])
            st["hi"] = value

        changed = False
        if st["enabled"]:
            lo, hi = st["lo"], st["hi"]
            old = self._value(key)
            current = min(max(old, lo), hi)
            if abs(current - old) > 1.0e-6:
                self._set_value(key, current)
                changed = True
            span = hi - lo
            phase = 0.0 if span <= 1.0e-9 else (current - lo) / span
            st["phase"] = min(max(phase, 0.0), 1.0)
            if st["phase"] >= 0.999:
                st["dir"] = -1
            elif st["phase"] <= 0.001:
                st["dir"] = 1
        return changed

    def set_speed_from_x(self, key: str, x_pos: float) -> bool:
        row = self.rows.get(key)
        st = self.anim.get(key)
        if row is None or st is None or "speed_track" not in row:
            return False
        speed = self._value_from_x(row["speed_track"], x_pos, self.SPEED_MIN, self.SPEED_MAX)
        speed = round(speed, 3)
        changed = abs(speed - st["speed"]) > 1.0e-4
        st["speed"] = speed
        return changed

    def advance_animation(self, dt: float) -> bool:
        if dt <= 0.0 or not self._any_animation():
            return False
        changed = False
        for key, st in self.anim.items():
            if not st["enabled"]:
                continue
            lo, hi = st["lo"], st["hi"]
            span = hi - lo
            speed = st["speed"]
            if span <= 1.0e-9 or speed <= 0.0:
                continue
            phase = st["phase"] + st["dir"] * speed * dt
            if phase >= 1.0:
                phase = 1.0
                st["dir"] = -1
            elif phase <= 0.0:
                phase = 0.0
                st["dir"] = 1
            st["phase"] = phase
            value = lo + phase * span
            if abs(value - self._value(key)) > 1.0e-6:
                self._set_value(key, value)
                changed = True
        return changed

    def draw(self, screen, font, small_font, width: int):
        import pygame

        if self.panel_rect is None:
            self.layout(width, screen.get_height())
        x, y, w, h = self.panel_rect

        if self._panel_surface is None:
            panel = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(panel, (5, 9, 14, 170), panel.get_rect(), border_radius=10)
            self._panel_surface = panel
        screen.blit(self._panel_surface, (x, y))

        title = _render_text(font, "Fractal parameters", (230, 237, 247))
        screen.blit(title, (x + 14, y + 7))

        def draw_checkbox(rect, enabled, accent):
            bx, by, bw, bh = rect
            pygame.draw.rect(screen, (62, 75, 91), (bx, by, bw, bh), border_radius=3)
            if enabled:
                pygame.draw.rect(screen, accent, (bx + 1, by + 1, bw - 2, bh - 2), border_radius=2)
                pygame.draw.lines(
                    screen,
                    (14, 24, 32),
                    False,
                    [(bx + 3, by + 8), (bx + 7, by + 12), (bx + 13, by + 4)],
                    2,
                )

        def draw_track(rect, t, accent, centered=False):
            tx, ty, tw, th = rect
            cy = ty + th // 2
            pygame.draw.rect(screen, (36, 44, 57), (tx, cy - 2, tw, 4), border_radius=2)
            knob_x = tx + int(round(tw * min(max(t, 0.0), 1.0)))
            if centered:
                center_x = tx + tw // 2
                pygame.draw.line(screen, (66, 78, 95), (center_x, cy - 5), (center_x, cy + 5), 1)
                x0, x1 = min(center_x, knob_x), max(center_x, knob_x)
                if x1 > x0:
                    pygame.draw.rect(screen, accent, (x0, cy - 2, x1 - x0, 4), border_radius=2)
            elif knob_x > tx:
                pygame.draw.rect(screen, accent, (tx, cy - 2, knob_x - tx, 4), border_radius=2)
            pygame.draw.circle(screen, (240, 246, 252), (knob_x, cy), 7)
            pygame.draw.circle(screen, (24, 35, 48), (knob_x, cy), 4)

        for key, label, _vmin, _vmax, accent in self.PARAMS:
            row = self.rows[key]
            st = self.anim[key]
            top = row["top"]
            draw_checkbox(row["check"], st["enabled"], accent)

            label_surf = _render_text(font, label, (218, 228, 240))
            value_surf = _render_text(font, self._fmt(key, self._value(key)), accent)
            screen.blit(label_surf, (x + 38, top))
            screen.blit(value_surf, (x + w - value_surf.get_width() - 14, top))

            tx, ty, tw, th = row["track"]
            cy = ty + th // 2
            pygame.draw.rect(screen, (36, 44, 57), (tx, cy - 2, tw, 4), border_radius=2)
            value_t = self._norm(key, self._value(key))
            if st["enabled"]:
                lo_t = self._norm(key, st["lo"])
                hi_t = self._norm(key, st["hi"])
                band_x = tx + int(round(tw * lo_t))
                band_w = max(1, int(round(tw * (hi_t - lo_t))))
                pygame.draw.rect(screen, (48, 62, 82), (band_x, cy - 3, band_w, 6), border_radius=3)
                pygame.draw.line(screen, accent, (band_x, cy - 5), (band_x, cy + 5), 1)
                end_x = band_x + band_w
                pygame.draw.line(screen, accent, (end_x, cy - 5), (end_x, cy + 5), 1)
                knob_x = tx + int(round(tw * value_t))
                pygame.draw.circle(screen, (240, 246, 252), (knob_x, cy), 7)
                pygame.draw.circle(screen, (24, 35, 48), (knob_x, cy), 4)
            else:
                centered = key != "power"
                draw_track(row["track"], value_t, accent, centered)

            if st["enabled"]:
                bound_top = row["bound_top"]
                half_w = row["half_w"]
                min_x = row["min_x"]
                max_x = row["max_x"]
                min_label = _render_text(small_font, "min", (160, 174, 194))
                max_label = _render_text(small_font, "max", (160, 174, 194))
                min_val = _render_text(small_font, self._fmt(key, st["lo"]), (220, 232, 244))
                max_val = _render_text(small_font, self._fmt(key, st["hi"]), (220, 232, 244))
                screen.blit(min_label, (min_x, bound_top - 1))
                screen.blit(max_label, (max_x, bound_top - 1))
                screen.blit(min_val, (min_x + half_w - min_val.get_width(), bound_top - 1))
                screen.blit(max_val, (max_x + half_w - max_val.get_width(), bound_top - 1))
                lo_t = self._norm(key, st["lo"])
                hi_t = self._norm(key, st["hi"])
                draw_track(row["min_track"], lo_t, accent)
                draw_track(row["max_track"], hi_t, accent)

                speed_top = row["speed_top"]
                speed_label = _render_text(small_font, "Speed", (160, 174, 194))
                speed_val = _render_text(small_font, f"{st['speed']:.2f}/s", (115, 205, 255))
                screen.blit(speed_label, (min_x, speed_top - 1))
                screen.blit(speed_val, (x + w - 14 - speed_val.get_width(), speed_top - 1))
                speed_t = (st["speed"] - self.SPEED_MIN) / (self.SPEED_MAX - self.SPEED_MIN)
                draw_track(row["speed_track"], speed_t, (115, 205, 255))

class TeapotEasterEgg:
    DURATION_FRAMES = 240
    ORBIT_RADIUS = 2.2
    ORBIT_PERIOD_FRAMES = 150.0
    BOUNCE_PERIOD_FRAMES = 58.0
    BOUNCE_HEIGHT = 0.90

    def __init__(self):
        self.active = False
        self.frames_left = 0
        self.age = 0
        self.pos = (0.0, FLOOR_Y, 0.0)
        self.yaw = 0.0

    def spawn(self):
        self.active = True
        self.frames_left = self.DURATION_FRAMES
        self.age = 0
        self._update_state()

    def _update_state(self):
        angle = self.age * (2.0 * math.pi / self.ORBIT_PERIOD_FRAMES)
        bounce = abs(math.sin(self.age * (2.0 * math.pi / self.BOUNCE_PERIOD_FRAMES))) * self.BOUNCE_HEIGHT
        self.pos = (
            math.cos(angle) * self.ORBIT_RADIUS,
            FLOOR_Y + bounce,
            math.sin(angle) * self.ORBIT_RADIUS,
        )
        self.yaw = angle + math.pi * 0.5

    def update(self):
        if not self.active:
            return
        self.age += 1
        self.frames_left -= 1
        if self.frames_left < 0:
            self.active = False
            return
        self._update_state()

    def state(self):
        return bool(self.active), self.pos, self.yaw


def run_interactive(args) -> int:
    import pygame

    pygame.init()
    pygame.display.set_caption("NVIDIA OptiX - Mandelbulb + Checkered Floor")
    width, height = max(480, int(args.width)), max(560, int(args.height))
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    clock = pygame.time.Clock()

    quality_index = args.quality - 1 if args.quality is not None else DEFAULT_QUALITY_INDEX
    floor_enabled = not args.no_floor
    renderer = MandelbulbRenderer(width, height, quality_index, device=args.device)
    camera = OrbitCamera(width, height)
    renderer.camera = camera.camera()
    slider = QualitySlider(
        renderer.quality_index,
        floor_enabled,
        args.shininess / 100.0,
        args.glass,
        args.ior,
    )
    slider.layout(width, height)
    lights = LightsPanel(
        colors=(
            (args.light_r, args.light_g, args.light_b),
            (args.fill_r, args.fill_g, args.fill_b),
            (args.rim_r, args.rim_g, args.rim_b),
        ),
        enabled=(True, args.fill_light, args.rim_light),
        ambient_rgb=(args.ambient_r, args.ambient_g, args.ambient_b),
        fractal_rgb=(args.fractal_r, args.fractal_g, args.fractal_b),
    )
    lights.layout(width, height)
    params = FractalParamsPanel(args.power, args.alpha, args.beta)
    params.layout(width, height)
    easter_egg = TeapotEasterEgg()

    slider_font = pygame.font.Font(None, 19)
    slider_small = pygame.font.Font(None, 15)

    fps_font = pygame.font.Font(None, 23)

    def make_fps_overlay(fps: float):
        if fps >= 50.0:
            color = (128, 235, 165)
        elif fps >= 30.0:
            color = (245, 206, 115)
        else:
            color = (245, 130, 120)
        text = fps_font.render(f"{fps:4.1f} FPS", True, color)
        shadow = fps_font.render(f"{fps:4.1f} FPS", True, (0, 0, 0))
        panel = pygame.Surface((text.get_width() + 20, text.get_height() + 10), pygame.SRCALPHA)
        pygame.draw.rect(panel, (5, 9, 14, 140), panel.get_rect(), border_radius=8)
        return panel, shadow, text

    fps_overlay = make_fps_overlay(0.0)

    start = time.perf_counter()
    paused = False
    paused_time = 0.0
    dragging = None
    frames = 0
    pixels = None
    scene_dirty = True
    panels_visible = True

    running = True
    while running:
        pending_size = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                pending_size = (max(480, int(event.w)), max(560, int(event.h)))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if paused:
                        start = time.perf_counter() - paused_time
                        paused = False
                    else:
                        paused_time = time.perf_counter() - start
                        paused = True
                    scene_dirty = True
                elif event.key == pygame.K_r:
                    start = time.perf_counter()
                    paused_time = 0.0
                    paused = False
                    camera.reset()
                    lights.reset_orbit()
                    scene_dirty = True
                elif event.key == pygame.K_s:
                    t_save = paused_time if paused else time.perf_counter() - start
                    teapot_active, teapot_pos, teapot_yaw = easter_egg.state()
                    renderer.camera = camera.camera()
                    save_png(
                        renderer.render_still(
                            t_save,
                            slider.floor_enabled,
                            slider.shininess,
                            params.power,
                            params.alpha,
                            params.beta,
                            lights.state(),
                            lights.fractal_state(),
                            lights.ambient_state(),
                            lights.orbit_state(),
                            slider.glass_enabled,
                            slider.glass_ior,
                            teapot_active,
                            teapot_pos,
                            teapot_yaw,
                        ),
                        Path("proj_mndlblb.png"),
                    )
                elif event.key == pygame.K_f:
                    slider.floor_enabled = not slider.floor_enabled
                    scene_dirty = True
                elif event.key == pygame.K_h:
                    panels_visible = not panels_visible
                    dragging = None
                elif event.key == pygame.K_b and (event.mod & pygame.KMOD_SHIFT):
                    easter_egg.spawn()
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    camera.zoom(1.0)
                    scene_dirty = True
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    camera.zoom(-1.0)
                    scene_dirty = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == pygame.BUTTON_LEFT:
                    if panels_visible and lights.panel_hit(event.pos):
                        handled = False
                        for i in range(len(lights.rows)):
                            if handled:
                                break
                            if lights.hit_orbit(i, event.pos):
                                lights.toggle_orbit(i)
                                dragging = None
                                handled = True
                                break
                            if lights.hit_toggle(i, event.pos):
                                lights.toggle_enabled(i)
                                dragging = None
                                scene_dirty = True
                                handled = True
                                break
                            channel_hit = False
                            for channel in ("r", "g", "b"):
                                if lights.hit_color(i, channel, event.pos):
                                    dragging = f"lights_{i}_{channel}"
                                    if lights.set_color_from_x(i, channel, float(event.pos[0])):
                                        scene_dirty = True
                                    channel_hit = True
                                    break
                            if channel_hit:
                                handled = True
                                break
                            if lights.hit_light_speed(i, event.pos):
                                dragging = f"orbit_speed_{i}"
                                lights.set_light_speed_from_x(i, float(event.pos[0]))
                                handled = True
                                break
                        if not handled:
                            for channel in ("r", "g", "b"):
                                if lights.hit_ambient(channel, event.pos):
                                    dragging = "ambient_" + channel
                                    if lights.set_ambient_from_x(channel, float(event.pos[0])):
                                        scene_dirty = True
                                    handled = True
                                    break
                        if not handled:
                            for channel in ("r", "g", "b"):
                                if lights.hit_fractal(channel, event.pos):
                                    dragging = "fractal_" + channel
                                    if lights.set_fractal_from_x(channel, float(event.pos[0])):
                                        scene_dirty = True
                                    handled = True
                                    break
                        if not handled:
                            dragging = None
                    elif panels_visible and params.panel_hit(event.pos):
                        handled = False
                        for key in params.anim:
                            if params.hit_checkbox(key, event.pos):
                                if params.toggle_animation(key):
                                    scene_dirty = True
                                dragging = None
                                handled = True
                                break
                            if params.hit_track(key, event.pos):
                                if params.disable_animation(key):
                                    scene_dirty = True
                                dragging = key
                                if params.set_value_from_x(key, float(event.pos[0])):
                                    scene_dirty = True
                                handled = True
                                break
                            if params.hit_bound(key, "min", event.pos):
                                dragging = "min_" + key
                                if params.set_bound_from_x(key, "min", float(event.pos[0])):
                                    scene_dirty = True
                                handled = True
                                break
                            if params.hit_bound(key, "max", event.pos):
                                dragging = "max_" + key
                                if params.set_bound_from_x(key, "max", float(event.pos[0])):
                                    scene_dirty = True
                                handled = True
                                break
                            if params.hit_speed(key, event.pos):
                                dragging = "speed_" + key
                                params.set_speed_from_x(key, float(event.pos[0]))
                                handled = True
                                break
                        if not handled:
                            dragging = None
                    elif panels_visible and slider.panel_hit(event.pos):
                        if slider.hit_floor(event.pos):
                            slider.floor_enabled = not slider.floor_enabled
                            dragging = None
                            scene_dirty = True
                        elif slider.hit_glass(event.pos):
                            slider.glass_enabled = not slider.glass_enabled
                            dragging = None
                            scene_dirty = True
                        elif slider.hit_shininess(event.pos):
                            dragging = "shininess"
                            if slider.set_shininess_from_x(float(event.pos[0])):
                                scene_dirty = True
                        elif slider.hit_glass_ior(event.pos):
                            dragging = "glass_ior"
                            if slider.set_glass_ior_from_x(float(event.pos[0])):
                                scene_dirty = True
                        elif slider.hit(event.pos):
                            dragging = "quality"
                            if slider.set_from_x(float(event.pos[0])):
                                renderer.set_quality(slider.index)
                                scene_dirty = True
                        else:
                            dragging = None
                    else:
                        dragging = "orbit"
                elif event.button == pygame.BUTTON_RIGHT:
                    dragging = "pan"
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in (pygame.BUTTON_LEFT, pygame.BUTTON_RIGHT):
                    dragging = None
            elif event.type == pygame.MOUSEMOTION:
                if dragging == "orbit":
                    camera.orbit(float(event.rel[0]), float(event.rel[1]))
                    scene_dirty = True
                elif dragging == "pan":
                    camera.pan(float(event.rel[0]), float(event.rel[1]))
                    scene_dirty = True
                elif dragging == "quality":
                    if slider.set_from_x(float(event.pos[0])):
                        renderer.set_quality(slider.index)
                        scene_dirty = True
                elif dragging == "shininess":
                    if slider.set_shininess_from_x(float(event.pos[0])):
                        scene_dirty = True
                elif dragging == "glass_ior":
                    if slider.set_glass_ior_from_x(float(event.pos[0])):
                        scene_dirty = True
                elif dragging is not None and dragging.startswith("lights_"):
                    _, light_index, channel = dragging.split("_")
                    if lights.set_color_from_x(int(light_index), channel, float(event.pos[0])):
                        scene_dirty = True
                elif dragging is not None and dragging.startswith("ambient_"):
                    channel = dragging.split("_", 1)[1]
                    if lights.set_ambient_from_x(channel, float(event.pos[0])):
                        scene_dirty = True
                elif dragging is not None and dragging.startswith("fractal_"):
                    channel = dragging.split("_", 1)[1]
                    if lights.set_fractal_from_x(channel, float(event.pos[0])):
                        scene_dirty = True
                elif dragging is not None and dragging.startswith("orbit_speed_"):
                    lights.set_light_speed_from_x(int(dragging.split("_")[-1]), float(event.pos[0]))
                elif dragging in ("power", "alpha", "beta"):
                    if params.set_value_from_x(dragging, float(event.pos[0])):
                        scene_dirty = True
                elif dragging is not None and (dragging.startswith("min_") or dragging.startswith("max_")):
                    which, key = dragging.split("_", 1)
                    if params.set_bound_from_x(key, which, float(event.pos[0])):
                        scene_dirty = True
                elif dragging is not None and dragging.startswith("speed_"):
                    params.set_speed_from_x(dragging[6:], float(event.pos[0]))
            elif event.type == pygame.MOUSEWHEEL:
                camera.zoom(float(event.y))
                scene_dirty = True
            elif event.type == pygame.WINDOWFOCUSLOST:
                dragging = None

        if pending_size is not None:
            width, height = pending_size
            screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
            renderer.resize(width, height)
            camera.resize(width, height)
            slider.layout(width, height)
            lights.layout(width, height)
            params.layout(width, height)
            scene_dirty = True

        mouse = pygame.mouse.get_pressed()
        if dragging is not None and dragging != "pan" and not mouse[0]:
            dragging = None
        elif dragging == "pan" and not mouse[2]:
            dragging = None

        dt = min(clock.get_time() / 1000.0, 0.05)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            camera.rotate(1.35 * dt, 0.0)
            scene_dirty = True
        if keys[pygame.K_RIGHT]:
            camera.rotate(-1.35 * dt, 0.0)
            scene_dirty = True
        if keys[pygame.K_UP]:
            camera.rotate(0.0, 0.9 * dt)
            scene_dirty = True
        if keys[pygame.K_DOWN]:
            camera.rotate(0.0, -0.9 * dt)
            scene_dirty = True

        if not paused:
            params.advance_animation(dt)
            lights.advance_orbit(dt)

        teapot_was_active = easter_egg.active
        easter_egg.update()
        if teapot_was_active and not easter_egg.active:
            scene_dirty = True

        now = paused_time if paused else time.perf_counter() - start
        if not paused or scene_dirty or pixels is None or easter_egg.active:
            if not paused:
                renderer.update(now)
            renderer.camera = camera.camera()
            teapot_active, teapot_pos, teapot_yaw = easter_egg.state()
            pixels = renderer.render(
                slider.floor_enabled,
                slider.shininess,
                params.power,
                params.alpha,
                params.beta,
                lights.state(),
                lights.fractal_state(),
                lights.ambient_state(),
                lights.orbit_state(),
                slider.glass_enabled,
                slider.glass_ior,
                teapot_active,
                teapot_pos,
                teapot_yaw,
            )
            scene_dirty = False

        frame_surface = pygame.image.frombuffer(pixels, (width, height), "RGBX")
        screen.blit(frame_surface, (0, 0))

        panel, shadow, text = fps_overlay
        if width >= 220:
            panel_x = width - panel.get_width() - 14
            panel_y = 12
            screen.blit(panel, (panel_x, panel_y))
            screen.blit(shadow, (panel_x + 11, panel_y + 6))
            screen.blit(text, (panel_x + 10, panel_y + 5))

        if panels_visible:
            lights.draw(screen, slider_font, slider_small, width)
            slider.draw(screen, slider_font, slider_small, width)
            params.draw(screen, slider_font, slider_small, width)
        pygame.display.flip()

        frames += 1
        if args.frames and frames >= args.frames:
            running = False

        if frames % 15 == 0:
            fps = clock.get_fps()
            fps_overlay = make_fps_overlay(fps)
            state = "PAUSED" if paused else f"{fps:4.0f} fps"
            floor_state = "floor on" if slider.floor_enabled else "floor off"
            pygame.display.set_caption(
                f"NVIDIA OptiX - Mandelbulb + Floor  |  {state}  |  quality {slider.index + 1}/10  |  {floor_state}"
            )
        clock.tick(args.fps)

    pygame.quit()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hardware ray-marched Mandelbulb with a selectable checkered floor, using NVIDIA OptiX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument(
        "--quality",
        type=int,
        default=None,
        help="render quality 1-10 (default: 4 interactive, 10 for --still)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--frames", type=int, default=0, help="quit after N frames (0 = run until closed)")
    parser.add_argument("--still", type=Path, default=None, help="render one still and exit")
    parser.add_argument("--time", type=float, default=1.1, help="animation time used for --still")
    parser.add_argument("--no-floor", action="store_true", help="start with the checkered floor hidden")
    parser.add_argument(
        "--shininess",
        type=int,
        default=40,
        help="fractal shininess from 0 (matte) to 100 (glossy)",
    )
    parser.add_argument("--power", type=float, default=POWER_DEFAULT, help="Mandelbulb power")
    parser.add_argument("--alpha", type=float, default=0.0, help="alpha angle offset in degrees")
    parser.add_argument("--beta", type=float, default=0.0, help="beta angle offset in degrees")
    parser.add_argument("--light-r", type=float, default=LIGHT_DEFAULT_COLORS[0][0], help="key light red channel, 0..1")
    parser.add_argument("--light-g", type=float, default=LIGHT_DEFAULT_COLORS[0][1], help="key light green channel, 0..1")
    parser.add_argument("--light-b", type=float, default=LIGHT_DEFAULT_COLORS[0][2], help="key light blue channel, 0..1")
    parser.add_argument("--fill-r", type=float, default=LIGHT_DEFAULT_COLORS[1][0], help="fill light red channel, 0..1")
    parser.add_argument("--fill-g", type=float, default=LIGHT_DEFAULT_COLORS[1][1], help="fill light green channel, 0..1")
    parser.add_argument("--fill-b", type=float, default=LIGHT_DEFAULT_COLORS[1][2], help="fill light blue channel, 0..1")
    parser.add_argument("--rim-r", type=float, default=LIGHT_DEFAULT_COLORS[2][0], help="rim light red channel, 0..1")
    parser.add_argument("--rim-g", type=float, default=LIGHT_DEFAULT_COLORS[2][1], help="rim light green channel, 0..1")
    parser.add_argument("--rim-b", type=float, default=LIGHT_DEFAULT_COLORS[2][2], help="rim light blue channel, 0..1")
    parser.add_argument("--fill-light", action="store_true", help="start with the fill light enabled")
    parser.add_argument("--rim-light", action="store_true", help="start with the rim light enabled")
    parser.add_argument("--fractal-r", type=float, default=FRACTAL_COLOR_DEFAULT[0], help="fractal colour tint red channel, 0..1")
    parser.add_argument("--fractal-g", type=float, default=FRACTAL_COLOR_DEFAULT[1], help="fractal colour tint green channel, 0..1")
    parser.add_argument("--fractal-b", type=float, default=FRACTAL_COLOR_DEFAULT[2], help="fractal colour tint blue channel, 0..1")
    parser.add_argument("--ambient-r", type=float, default=AMBIENT_COLOR_DEFAULT[0], help="ambient light red channel, 0..1")
    parser.add_argument("--ambient-g", type=float, default=AMBIENT_COLOR_DEFAULT[1], help="ambient light green channel, 0..1")
    parser.add_argument("--ambient-b", type=float, default=AMBIENT_COLOR_DEFAULT[2], help="ambient light blue channel, 0..1")
    parser.add_argument("--glass", action="store_true", help="start with glass refraction enabled")
    parser.add_argument(
        "--ior",
        type=float,
        default=GLASS_IOR_DEFAULT,
        help=f"glass index of refraction, {GLASS_IOR_MIN:g}..{GLASS_IOR_MAX:g}",
    )
    args = parser.parse_args()

    if args.width < 1 or args.height < 1:
        parser.error("--width and --height must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.frames < 0:
        parser.error("--frames must be zero or greater")
    if args.quality is not None and not 1 <= args.quality <= len(QUALITY_PRESETS):
        parser.error(f"--quality must be between 1 and {len(QUALITY_PRESETS)}")
    if not 0 <= args.shininess <= 100:
        parser.error("--shininess must be between 0 and 100")
    if not POWER_MIN <= args.power <= POWER_MAX:
        parser.error(f"--power must be between {POWER_MIN} and {POWER_MAX}")
    if not ANGLE_MIN <= args.alpha <= ANGLE_MAX:
        parser.error(f"--alpha must be between {ANGLE_MIN:g} and {ANGLE_MAX:g}")
    if not ANGLE_MIN <= args.beta <= ANGLE_MAX:
        parser.error(f"--beta must be between {ANGLE_MIN:g} and {ANGLE_MAX:g}")
    if not GLASS_IOR_MIN <= args.ior <= GLASS_IOR_MAX:
        parser.error(f"--ior must be between {GLASS_IOR_MIN:g} and {GLASS_IOR_MAX:g}")
    for prefix in ("light", "fill", "rim", "fractal", "ambient"):
        for channel in ("r", "g", "b"):
            value = getattr(args, f"{prefix}_{channel}")
            if not 0.0 <= value <= 1.0:
                parser.error(f"--{prefix}-{channel} must be between 0 and 1")

    if args.still is not None:
        quality = args.quality if args.quality is not None else len(QUALITY_PRESETS)
        renderer = MandelbulbRenderer(
            args.width,
            args.height,
            quality_index=quality - 1,
            device=args.device,
        )
        t0 = time.perf_counter()
        renderer.update(args.time)
        light_state = (
            (True, (args.light_r, args.light_g, args.light_b)),
            (args.fill_light, (args.fill_r, args.fill_g, args.fill_b)),
            (args.rim_light, (args.rim_r, args.rim_g, args.rim_b)),
        )
        pixels = renderer.render(
            not args.no_floor,
            args.shininess / 100.0,
            args.power,
            args.alpha,
            args.beta,
            light_state,
            (args.fractal_r, args.fractal_g, args.fractal_b),
            (args.ambient_r, args.ambient_g, args.ambient_b),
            (0.0, 0.0, 0.0),
            args.glass,
            args.ior,
        )
        save_png(pixels, args.still)
        print(f"Rendered in {time.perf_counter() - t0:.2f}s")
        return 0

    return run_interactive(args)


if __name__ == "__main__":
    raise SystemExit(main())
