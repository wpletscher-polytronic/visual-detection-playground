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
    """
    rows = []
    with open(path) as handle:
        for line_no, line in enumerate(handle, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"{path}:{line_no}: expected 5 fields, got {len(parts)}")

            _cls, cx, cy, w, h = (float(p) for p in parts)
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                raise ValueError(f"{path}:{line_no}: centre {cx},{cy} outside 0..1")
            if w <= 0.0 or h <= 0.0:
                raise ValueError(f"{path}:{line_no}: non-positive box {w}x{h}")

            rows.append((cx * img_w, cy * img_h, max(w * img_w, h * img_h) / 2))

    return np.array(rows, dtype=np.float64).reshape(-1, 3)
