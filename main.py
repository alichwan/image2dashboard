"""Real-time camera color dashboard.

Streams the webcam feed continuously and, every `interval` seconds
(configurable, so the analysis cost can be tuned), measures the cyan,
magenta and yellow percentage (CMYK with K ignored) of the current frame
and derives the resulting combined "main color".
"""

import argparse
import time

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


def build_grid(frame: np.ndarray, c: float, m: float, y: float) -> np.ndarray:
    """Arrange the original frame plus isolated C/M/Y ink channels into a 2x2 grid."""
    h, w = frame.shape[:2]
    half_w, half_h = w // 2, h // 2

    original = draw_dashboard(frame.copy(), c, m, y)
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


def draw_dashboard(frame: np.ndarray, c: float, m: float, y: float) -> np.ndarray:
    """Overlay CMY (%) bars and the combined main-color swatch onto the frame."""
    panel_w = 220
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, 165), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    bar_max = panel_w - 90
    channels = [("C", c, (255, 255, 0)), ("M", m, (255, 0, 255)), ("Y", y, (0, 255, 255))]
    for i, (chan_label, value, color) in enumerate(channels):
        y_pos = 25 + i * 30
        cv2.putText(frame, f"{chan_label} {value:5.1f}%", (5, y_pos + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        bar_len = int(bar_max * (value / 100))
        cv2.rectangle(frame, (75, y_pos - 8), (75 + bar_len, y_pos + 2), color, -1)

    swatch_y = 25 + 3 * 30 + 10
    cv2.putText(frame, "Main color", (5, swatch_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (5, swatch_y), (panel_w - 10, swatch_y + 30),
                  cmy_to_bgr(c, m, y), -1)
    return frame


def run(camera_index: int = 0, interval: float = 1.0,
        denoise: str = "gaussian", denoise_strength: int = 5) -> None:
    """Show the live camera feed, refreshing color stats every `interval` seconds."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    c = m = y = 0.0
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
                last_analysis = now
                print(f"C={c:.1f}% M={m:.1f}% Y={y:.1f}% -> main color BGR{cmy_to_bgr(c, m, y)}")

            grid = build_grid(frame, c, m, y)
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(camera_index=args.camera, interval=args.interval,
        denoise=args.denoise, denoise_strength=args.denoise_strength)
