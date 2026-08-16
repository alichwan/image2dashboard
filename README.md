# image2dashboard

Analyze real-time camera images in terms of color composition.

`main.py` opens a webcam feed and shows it as a 2x2 dashboard: the
original frame (with a live CMY readout, timeseries chart and main-color
swatch overlaid) plus three quadrants that each emphasize one subtractive
ink channel — Cyan, Magenta and Yellow (K is ignored).

![Example dashboard](imgs/example.png)

## How it works

- The video feed itself is read and displayed every loop iteration, so it
  stays smooth.
- The actual color analysis (mean Cyan/Magenta/Yellow percentage of the
  frame) only runs every `--interval` seconds — a configurable throttle so
  you can trade responsiveness for CPU cost.
- Each analysis tick appends a sample to a rolling history, drawn as a
  small line chart per channel (window length set by `--history`).
- C/M/Y are computed as the plain RGB complement, expressed as a
  percentage and ignoring K: `C = 100 * (255 - R) / 255` (same for
  M/Y from G/B).
- An optional per-frame denoise filter (Gaussian, median or bilateral)
  can be applied before display/analysis to reduce sensor noise.

## Installation

This project uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Usage

```bash
uv run main.py
```

Press `q` in the window to quit.

### Options

| Flag | Default | Description |
|---|---|---|
| `--camera` | `0` | Camera device index |
| `--interval` | `1.0` | Seconds between color analyses |
| `--denoise` | `gaussian` | Denoise filter: `none`, `gaussian`, `median`, `bilateral` |
| `--denoise-strength` | `5` | Kernel size / filter diameter for the denoise filter |
| `--history` | `60` | Number of past samples kept in the CMY timeseries chart |

Example: a lighter-weight, less frequent analysis with a longer trend
window:

```bash
uv run main.py --interval 2.0 --history 30 --denoise none
```
