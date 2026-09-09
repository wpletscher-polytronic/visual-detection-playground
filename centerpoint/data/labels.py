"""The one place YOLO label files are parsed.

Three consumers need this — analyze_labels.py, data/dataset.py, and later the eval
harness converting ground truth for metrics. They must never disagree about what a
label file means, so the parse lives here and nowhere else.
"""

import numpy as np


def load_yolo_labels(path, img_w, img_h):
    """Read one YOLO label file. Returns (N, 3) float array of (cx, cy, r) in pixels.

    A YOLO line is `cls cx cy w h` with every coordinate normalised to 0..1, so the
    image size is required to recover pixels — and w is normalised by width while h is
    normalised by height, which only coincide on square images. Convert to pixels
    first, then take the radius, or non-square images silently produce wrong radii.

    r = max(w, h) / 2 is an assumption, not ground truth: the labels are boxes, and a
    box around a circle gives the diameter on its longer side. See PLAN.md — the boxes
    in this dataset are measurably non-square (aspect p50 1.21, p95 2.0), so this
    systematically overestimates. Kept because it is what 8_yolo26_detection.py already
    does, which keeps the two models comparable.

    An empty label file is legal (an image with no holes) and returns shape (0, 3),
    never None — callers should not have to special-case it.

    Centres must be strictly inside the image, matching codec/encode.py: one sitting at
    exactly the width has no cell to floor into. The bound is checked on the float32
    value because that is what data/dataset.py converts to before encoding — a float64
    centre a hair inside the edge can round up to exactly the width and pass otherwise.
    """
    rows = []
    with open(path) as handle:
        for line_no, line in enumerate(handle, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"{path}:{line_no}: expected 5 fields, got {len(parts)}")

            values = [float(part) for part in parts]
            # Checked before anything else: nan fails every comparison below, so a nan
            # width would slip past `w <= 0.0` and only surface inside encode.
            if not np.isfinite(values).all():
                raise ValueError(f"{path}:{line_no}: non-finite value in {line.strip()!r}")

            _cls, cx, cy, w, h = values
            if w <= 0.0 or h <= 0.0:
                raise ValueError(f"{path}:{line_no}: non-positive box {w}x{h}")

            centre_x, centre_y = np.float32(cx * img_w), np.float32(cy * img_h)
            if not (0.0 <= centre_x < img_w and 0.0 <= centre_y < img_h):
                raise ValueError(f"{path}:{line_no}: centre {cx},{cy} is {centre_x},{centre_y} px, "
                                 f"not strictly inside {img_w}x{img_h}")

            rows.append((cx * img_w, cy * img_h, max(w * img_w, h * img_h) / 2))

    return np.array(rows, dtype=np.float64).reshape(-1, 3)
