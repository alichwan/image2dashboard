"""Real-time camera color dashboard.

Streams the webcam feed continuously and, every `interval` seconds
(configurable, so the analysis cost can be tuned), measures the cyan,
magenta and yellow percentage (CMYK with K ignored) of the current frame
and derives the resulting combined "main color".
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np


def denoise_frame(frame: np.ndarray, method: str, strength: int) -> np.ndarray:
    """Smooth sensor/low-light noise out of a frame before display or analysis."""
    if method == "none":
        return frame
    if method == "gaussian":
        k = strength | 1  # kernel size must be odd
        return cv2.GaussianBlur(frame, (k, k), 0)
    if method == "median":
        k = strength | 1
        return cv2.medianBlur(frame, k)
    if method == "bilateral":
        return cv2.bilateralFilter(frame, d=strength, sigmaColor=75, sigmaSpace=75)
    raise ValueError(f"Unknown denoise method: {method}")


def analyze_colors(frame: np.ndarray) -> tuple[float, float, float]:
    """Return the mean (cyan, magenta, yellow) percentage of a BGR frame (K ignored)."""
    b, g, r = cv2.mean(frame)[:3]
    c = 100.0 * (255 - r) / 255
    m = 100.0 * (255 - g) / 255
    y = 100.0 * (255 - b) / 255
    return c, m, y


def cmy_to_bgr(c: float, m: float, y: float) -> tuple[int, int, int]:
    """Convert CMY percentages (K ignored) back to a displayable BGR pixel."""
    r = 255 * (1 - c / 100)
    g = 255 * (1 - m / 100)
    b = 255 * (1 - y / 100)
    return round(b), round(g), round(r)


def emphasize_ink(frame: np.ndarray, channel: str) -> np.ndarray:
    """Show only the given subtractive ink's concentration ('c', 'm' or 'y'), rest zeroed."""
    b, g, r = cv2.split(frame)
    zeros = np.zeros_like(b)
    if channel == "c":  # cyan = complement of red
        inv = cv2.bitwise_not(r)
        return cv2.merge([inv, inv, zeros])
    if channel == "m":  # magenta = complement of green
        inv = cv2.bitwise_not(g)
        return cv2.merge([inv, zeros, inv])
    if channel == "y":  # yellow = complement of blue
        inv = cv2.bitwise_not(b)
        return cv2.merge([zeros, inv, inv])
    raise ValueError(f"Unknown channel: {channel}")


def label(img: np.ndarray, text: str) -> np.ndarray:
    """Stamp a small caption in the top-left corner of a quadrant."""
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_timeseries(frame: np.ndarray, x: int, y: int, w: int, h: int,
                     histories: list[deque], colors: list[tuple[int, int, int]],
                     y_max: float = 100.0) -> np.ndarray:
    """Draw overlaid line charts (one per history, 0..y_max on the y-axis) in a rect."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 90, 90), 1)
    for hist, color in zip(histories, colors):
        n = len(hist)
        if n < 2:
            continue
        points = []
        for i, value in enumerate(hist):
            px = x + int(i * (w - 1) / (n - 1))
            frac = min(max(value / y_max, 0.0), 1.0)
            py = y + h - 1 - int(frac * (h - 1))
            points.append((px, py))
        for p0, p1 in zip(points, points[1:]):
            cv2.line(frame, p0, p1, color, 1, cv2.LINE_AA)
    return frame


def build_grid(frame: np.ndarray, c_hist: deque, m_hist: deque, y_hist: deque) -> np.ndarray:
    """Arrange the original frame plus isolated C/M/Y ink channels into a 2x2 grid."""
    h, w = frame.shape[:2]
    half_w, half_h = w // 2, h // 2

    original = draw_dashboard(frame.copy(), c_hist, m_hist, y_hist)
    cyan_only = emphasize_ink(frame, "c")
    magenta_only = emphasize_ink(frame, "m")
    yellow_only = emphasize_ink(frame, "y")

    quadrants = []
    for img, text in ((original, "Original"), (cyan_only, "Cyan"),
                       (magenta_only, "Magenta"), (yellow_only, "Yellow")):
        small = cv2.resize(img, (half_w, half_h), interpolation=cv2.INTER_AREA)
        quadrants.append(label(small, text))

    top = np.hstack(quadrants[0:2])
    bottom = np.hstack(quadrants[2:4])
    return np.vstack([top, bottom])


def draw_dashboard(frame: np.ndarray, c_hist: deque, m_hist: deque, y_hist: deque) -> np.ndarray:
    """Overlay a CMY (%) timeseries chart and the combined main-color swatch onto the frame."""
    c = c_hist[-1] if c_hist else 0.0
    m = m_hist[-1] if m_hist else 0.0
    y = y_hist[-1] if y_hist else 0.0

    panel_w = 620
    panel_h = 210
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    channels = [("C", c, (255, 255, 0)), ("M", m, (255, 0, 255)), ("Y", y, (0, 255, 255))]
    for i, (chan_label, value, color) in enumerate(channels):
        x_pos = 5 + i * 70
        cv2.putText(frame, f"{chan_label} {value:4.0f}%", (x_pos, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    draw_timeseries(frame, 5, 25, panel_w - 10, 90,
                     [c_hist, m_hist, y_hist],
                     [(255, 255, 0), (255, 0, 255), (0, 255, 255)])

    swatch_y = 125
    cv2.putText(frame, "Main color", (5, swatch_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (5, swatch_y), (panel_w - 10, swatch_y + 30),
                  cmy_to_bgr(c, m, y), -1)
    return frame


def run(camera_index: int = 0, interval: float = 1.0,
        denoise: str = "gaussian", denoise_strength: int = 5, history: int = 60) -> None:
    """Show the live camera feed, refreshing color stats every `interval` seconds."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    c_hist: deque = deque(maxlen=history)
    m_hist: deque = deque(maxlen=history)
    y_hist: deque = deque(maxlen=history)
    last_analysis = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # frame = denoise_frame(frame, denoise, denoise_strength)

            now = time.time()
            if now - last_analysis >= interval:
                c, m, y = analyze_colors(frame)
                c_hist.append(c)
                m_hist.append(m)
                y_hist.append(y)
                last_analysis = now
                print(f"C={c:.1f}% M={m:.1f}% Y={y:.1f}% -> main color BGR{cmy_to_bgr(c, m, y)}")

            grid = build_grid(frame, c_hist, m_hist, y_hist)
            cv2.imshow("image2dashboard", grid)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--interval", type=float, default=1.0,
                         help="Seconds between color analyses (default: 1.0)")
    parser.add_argument("--denoise", choices=["none", "gaussian", "median", "bilateral"],
                         default="gaussian", help="Noise-reduction filter applied per frame (default: gaussian)")
    parser.add_argument("--denoise-strength", type=int, default=5,
                         help="Kernel size / filter diameter for the denoise filter (default: 5)")
    parser.add_argument("--history", type=int, default=60,
                         help="Number of past samples to keep in the CMY timeseries chart (default: 60)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(camera_index=args.camera, interval=args.interval,
        denoise=args.denoise, denoise_strength=args.denoise_strength, history=args.history)
