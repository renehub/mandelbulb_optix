# Mandelbulb OptiX

Real-time Mandelbulb ray marcher with a checkered floor, shadowed directional
lights, live material controls, optional glass refraction, and independently
animated fractal parameters.

The renderer uses NVIDIA OptiX through Warp. It needs a CUDA-capable NVIDIA
GPU with an installed OptiX runtime.

#### Requirements

```bash
python -m venv .venv
source .venv/bin/activate
pip install warp-optix pygame-ce pillow numpy
```

#### Run

```bash
python proj_mndlblb.py
```

Render a single still:

```bash
python proj_mndlblb.py --still image.png --width 1440 --height 900
```

#### Keyboard and mouse controls

| Input | Action |
|---|---|
| Left-drag scene | Orbit the camera around the Mandelbulb |
| Right-drag scene | Pan the orbit target |
| Mouse wheel | Zoom in/out |
| Arrow keys | Orbit the camera; works while paused |
| `+` / `-` | Zoom in/out |
| `Space` | Pause/resume the scene, parameter animations, and light orbit |
| `F` | Toggle the checkered floor |
| `H` | Show/hide all control panels |
| `R` | Reset the camera and light-orbit angles; restart the scene animation clock |
| `S` | Save a screenshot as `proj_mndlblb.png` |
| `Esc` | Quit |

Sliders are set by dragging. Checkboxes are clicked to toggle.

#### Control panels

##### Top-left: Lighting & Color

Each **Key**, **Fill**, and **Rim** light row contains:

| Control | Description |
|---|---|
| On/off checkbox | Enables that light and its shadow contribution |
| R / G / B sliders | Light color, `0–100%` |
| Orbit checkbox | Makes only that light rotate around the Mandelbulb |
| S speed slider | Orbit speed for that light, `0.00–2.00 rad/s` |

Defaults:

- **Key**: enabled, `100 / 78 / 56`
- **Fill**: disabled, `34 / 56 / 100`
- **Rim**: disabled, `26 / 100 / 86`
- All Orbit checkboxes are off by default.
- All orbit speeds default to `0.35 rad/s`.
- Setting an orbit speed to `0.00` keeps that light stationary.

The orbit takes place around the vertical axis through the object. Orbit
speed sliders are dimmed while their light's Orbit checkbox is off, but they
can still be preset.

Below the lights:

| Control | Description |
|---|---|
| Ambient R / G / B | Ambient/sky-fill color, `0–100%` |
| Fractal R / G / B | Fractal material color tint, `0–100%` |

For Ambient and Fractal, `100 / 100 / 100` leaves the current appearance
unchanged. The ambient tint affects ambient illumination on the fractal and
floor; the procedural sky background stays fixed.

##### Bottom-left: Render quality

| Control | Description |
|---|---|
| Render quality | Preset `1–10`; higher is slower and cleaner |
| Shininess | Material glossiness, `0–100%` |
| Glass | Switch the fractal to refractive glass |
| Glass IOR | Index of refraction, `1.00–2.00`; default `1.45` |
| Checkered floor | Floor visibility checkbox, same as the `F` key |

Glass mode approximates air-to-glass refraction using the refracted floor/sky
color and a Fresnel reflection. It does not trace the fractal's far side, but
it does tint the transmission with the Fractal RGB color and softens the
fractal's floor shadow.

Quality presets:

| Quality | Samples | Steps | Iterations |
|---:|---:|---:|---:|
| 1 | 1 | 64 | 8 |
| 2 | 1 | 80 | 10 |
| 3 | 1 | 112 | 12 |
| 4 | 2 | 128 | 12 |
| 5 | 2 | 160 | 14 |
| 6 | 3 | 192 | 16 |
| 7 | 4 | 224 | 18 |
| 8 | 6 | 256 | 20 |
| 9 | 8 | 320 | 24 |
| 10 | 12 | 384 | 28 |

Quality `4` is the interactive default. Quality `10` is the default for
`--still`.

##### Bottom-right: Fractal parameters

| Parameter | Range |
|---|---|
| Power | `2.0–16.0` |
| Alpha offset | `-180°–+180°` |
| Beta offset | `-180°–+180°` |

Each parameter has an animation checkbox. Enabling it reveals:

- **Min**: lower animation boundary
- **Max**: upper animation boundary
- **Speed**: `0.01–0.80/s`

The animation ping-pongs between its min and max. Dragging a parameter's main
slider manually disables that parameter's animation.

##### Top-right: FPS meter

Displays the current frame rate. It remains visible when panels are hidden.

#### CLI options

| Option | Default | Description |
|---|---|---|
| `--width` | `960` | Window/render width |
| `--height` | `600` | Window/render height |
| `--quality` | default varies | Quality `1–10`; `4` interactive, `10` for `--still` |
| `--device` | `cuda` | Warp device |
| `--fps` | `60` | Interactive frame-rate cap |
| `--frames` | `0` | Quit after N frames; `0` runs until closed |
| `--still PATH` | none | Render one still and exit |
| `--time` | `1.1` | Animation time used for `--still` |
| `--no-floor` | floor on | Start with the checkered floor hidden |
| `--shininess` | `40` | Fractal shininess, `0–100` |
| `--power` | `8.0` | Mandelbulb power |
| `--alpha` | `0.0` | Alpha angle offset in degrees |
| `--beta` | `0.0` | Beta angle offset in degrees |
| `--light-r/g/b` | `1.0 / 0.78 / 0.56` | Key light color, normalized `0–1` |
| `--fill-r/g/b` | `0.34 / 0.56 / 1.0` | Fill light color, normalized `0–1` |
| `--rim-r/g/b` | `0.26 / 1.0 / 0.86` | Rim light color, normalized `0–1` |
| `--fill-light` | off | Start with the fill light enabled |
| `--rim-light` | off | Start with the rim light enabled |
| `--fractal-r/g/b` | `1.0 / 1.0 / 1.0` | Fractal color tint, normalized `0–1` |
| `--ambient-r/g/b` | `1.0 / 1.0 / 1.0` | Ambient light color, normalized `0–1` |
| `--glass` | off | Start with glass refraction enabled |
| `--ior` | `1.45` | Glass index of refraction, `1.0–2.0` |
| `-h`, `--help` | | Show CLI help |

Notes:

- CLI color values are normalized (`0.0–1.0`); the UI displays them as
  percentages (`0–100%`).
- Interactive animation, per-light Orbit selection, and panel visibility are
  controlled from the UI. The `--still` path does not animate or apply orbit
  angles.
- The interactive window can be resized, with a minimum of `480 × 560`.
- The `S` key saves a screenshot. The separate `S` label inside each light row
  is the per-light orbit-speed slider.
