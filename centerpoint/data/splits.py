"""Pairing images with labels, and the integrity checks that go with it.

Both analyze_labels.py and data/dataset.py need to walk a split and know which image
belongs to which label file. Doing that twice would let them drift apart about what a
valid split even is — the same reason load_yolo_labels lives alone in data/labels.py.

Two entry points, layered:
- list_split() is the primitive: pairing plus integrity, no file contents read.
- load_split_geometry() adds sizes and parsed holes on top, for callers that want the
  geometry without decoding images.
"""

import glob
import os

import numpy as np
from PIL import Image

from centerpoint.data.labels import load_yolo_labels

IMAGE_EXTS = ('.jpg', '.jpeg', '.png')


def find_image(images_dir, stem):
    """Locate the image matching a label stem, whatever extension it uses."""
    for ext in IMAGE_EXTS:
        path = os.path.join(images_dir, stem + ext)
        if os.path.isfile(path):
            return path
    return None


def list_split(root, split):
    """Return [(stem, image_path, label_path)] for one split, asserting the pairing holds.

    Both directions are checked, because they fail differently. A label with no image is
    loud — something would try to read a missing file. An image with no label is silent:
    it would be trained on as if it genuinely contained no holes, quietly teaching the
    model that targets are empty. The second is the dangerous one.
    """
    label_dir = os.path.join(root, split, 'labels')
    images_dir = os.path.join(root, split, 'images')
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"missing split directory: {label_dir}")

    entries = []
    seen_images = set()
    for label_path in sorted(glob.glob(os.path.join(label_dir, '*.txt'))):
        stem = os.path.splitext(os.path.basename(label_path))[0]
        image_path = find_image(images_dir, stem)
        if image_path is None:
            raise FileNotFoundError(f"label without image: {label_path}")
        seen_images.add(os.path.basename(image_path))
        entries.append((stem, image_path, label_path))

    on_disk = {os.path.basename(p) for ext in IMAGE_EXTS
               for p in glob.glob(os.path.join(images_dir, '*' + ext))}
    orphans = on_disk - seen_images
    assert not orphans, \
        f"{split}: {len(orphans)} image(s) with no label file, e.g. {sorted(orphans)[:3]}"

    return entries


def load_split_geometry(root, split):
    """Return [(stem, image_path, holes, (width, height))] with holes in pixels.

    Sizes come from the image headers via PIL, which does not decode the pixels — reading
    3000 headers is seconds, reading 3000 JPEGs is not. Callers that decode the image
    anyway (the torch Dataset) should use list_split and take the size from the decoded
    array instead.

    Sizes are read rather than assumed: "all images are 640x640" is a fact about one
    Roboflow export, not about the format.
    """
    entries = []
    for stem, image_path, label_path in list_split(root, split):
        width, height = Image.open(image_path).size
        holes = load_yolo_labels(label_path, width, height)

        # Parsed count must match the file, or the parser is silently dropping lines.
        with open(label_path) as handle:
            n_lines = sum(1 for line in handle if line.split())
        assert len(holes) == n_lines, f"{label_path}: parsed {len(holes)} of {n_lines} lines"

        if len(holes):
            assert np.isfinite(holes).all(), f"{label_path}: non-finite value"
            assert (holes[:, 2] > 0).all(), f"{label_path}: non-positive radius"
            assert (holes[:, 0] <= width).all() and (holes[:, 1] <= height).all(), \
                f"{label_path}: centre outside image"

        entries.append((stem, image_path, holes, (width, height)))

    return entries
