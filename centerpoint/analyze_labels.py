"""STEP 1 — measure the data before choosing stride and sigma. Labels only, no model.

Numpy and cv2 only, no torch and no GPU. Paths come from paths.py, image/label
pairing and parsing from data/splits.py and data/labels.py.

What this decides, and why it has to come first:
- output stride (4 vs 2), from how small the holes are in heatmap cells
- the sigma range worth trying
- the exact pixel separations that tests/test_encode.py::test_merge_threshold asserts
- whether the tiny boxes at the bottom of the radius distribution are real holes

Two different merge limits are reported, and the distinction matters. cell_collisions is
the HARD limit: two centres in one output cell cannot both be represented, full stop.
separation_at_risk is the SOFT limit: two centres closer than roughly sigma flatten into
one peak even when they land in different cells. The soft limit binds first and by a wide
margin, so reading only the collision number badly understates the risk.

Run either way, from anywhere:
    python -m centerpoint.analyze_labels [dataset_root]
    python centerpoint/analyze_labels.py [dataset_root]
Defaults to config.DATASET. Pass another root to compare datasets — every run writes to
outputs/debug/label_stats/<dataset>/, so runs never overwrite each other.
"""

import json
import os
import random
import sys

import cv2
import numpy as np


if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from centerpoint.paths import DATASET, debug_dir  # noqa: E402
from centerpoint.data.splits import load_split_geometry          # noqa: E402



SPLITS = ('train', 'valid', 'test')
STRIDES = (4, 2)
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)

# Neighbour distances to report as "at risk", in output cells. A Gaussian with sigma of
# ~1 cell has essentially merged by 1-2 cells of separation, so these bracket the range
# where two real holes stop being two peaks.
RISK_CELLS = (1.0, 2.0, 3.0)

OVERLAY_UPSCALE = 3   # 1-2 px circles are invisible at native size; this makes them legible


def percentiles(values):
    """Percentile dict for one distribution, or None if there is nothing to describe."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {f'p{p}': float(v) for p, v in zip(PERCENTILES, np.percentile(values, PERCENTILES))}


def nearest_neighbour_distances(holes):
    """Distance from each hole to its closest neighbour in the same image, in pixels.

    A hole in a single-hole image has no neighbour and gets inf, so it still counts in the
    denominator of any risk fraction — dropping those holes instead would inflate every
    "at risk" number.
    """
    if len(holes) < 2:
        return np.full(len(holes), np.inf)
    centres = holes[:, :2]
    dist = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=1)


def cell_collisions(per_image, stride):
    """Holes whose centre floors into an already-occupied output cell — the hard limit.

    Mirrors the CenterNet collision experiment (614 of 860001 COCO objects, 0.07%).
    """
    lost = total = 0
    for holes in per_image:
        cells = {(int(x // stride), int(y // stride)) for x, y, _ in holes}
        total += len(holes)
        lost += len(holes) - len(cells)
    return {'colliding': lost, 'total': total,
            'fraction': (lost / total) if total else 0.0}


def separation_at_risk(nn_pixels, stride, cells):
    """Fraction of holes whose nearest neighbour is within `cells` output cells.

    The soft merge limit. Unlike a collision this does not make the hole impossible to
    represent — it makes it likely to be absorbed into its neighbour's peak.
    """
    nn_pixels = np.asarray(nn_pixels)
    if not len(nn_pixels):
        return 0.0
    return float(np.mean(nn_pixels < cells * stride))


def print_percentiles(label, stats, scale=1.0):
    if stats is None:
        print(f"  {label:<28s} (empty)")
        return
    cols = '  '.join(f'{stats["p" + str(p)] / scale:6.2f}' for p in PERCENTILES)
    print(f"  {label:<28s} {cols}")


def pick_overlay_images(entries, nn_per_image):
    """Choose five images worth looking at, each for a stated reason.

    Deliberately not five random ones: the parse is most likely to look fine on an easy
    image and wrong on a hard one, and the hard cases are also what the stride decision
    hinges on.
    """
    usable = [(i, e) for i, e in enumerate(entries) if len(e[2])]
    if not usable:
        return []

    picks = {}
    picks['densest'] = max(usable, key=lambda ie: len(ie[1][2]))[1]

    tight = [(nn_per_image[i].min(), e) for i, e in usable if np.isfinite(nn_per_image[i]).any()]
    if tight:
        picks['tightest-pair'] = min(tight, key=lambda te: te[0])[1]

    with_several = [e for _, e in usable if len(e[2]) >= 3]
    if with_several:
        picks['smallest-holes'] = min(with_several, key=lambda e: np.median(e[2][:, 2]))
        picks['largest-holes'] = max(with_several, key=lambda e: np.median(e[2][:, 2]))

    picks['random'] = random.Random(0).choice(usable)[1]
    return list(picks.items())


def overlay(reason, entry, out_dir, upscale=OVERLAY_UPSCALE):
    """Draw the parsed circles over the source image, upscaled so 1 px radii are visible."""
    stem, image_path, holes, _size = entry
    img = cv2.imread(image_path)
    if img is None:
        raise IOError(f"could not read {image_path}")

    img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_NEAREST)
    for x, y, r in holes:
        centre = (int(round(x * upscale)), int(round(y * upscale)))
        cv2.circle(img, centre, max(int(round(r * upscale)), 1), (0, 255, 0), 1)
        cv2.circle(img, centre, 1, (0, 0, 255), -1)   # centre, so offset errors are visible

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{reason}_{stem}.png')
    cv2.imwrite(out_path, img)
    return out_path, len(holes)


def analyse_split(root, split):
    entries = load_split_geometry(root, split)
    per_image = [e[2] for e in entries]
    all_holes = np.concatenate(per_image) if per_image else np.zeros((0, 3))

    sizes = {e[3] for e in entries}
    nn_per_image = [nearest_neighbour_distances(h) for h in per_image]
    nn = np.concatenate(nn_per_image) if nn_per_image else np.zeros(0)

    print(f"\n=== {split} ===")
    print(f"  images {len(entries)}   holes {len(all_holes)}   "
          f"holes/image {len(all_holes) / max(len(entries), 1):.1f}")
    print(f"  image sizes: {sorted(sizes)}" if len(sizes) <= 4
          else f"  image sizes: {len(sizes)} distinct")

    header = '  '.join(f'{"p" + str(p):>6s}' for p in PERCENTILES)
    print(f"\n  {'':<28s} {header}")
    radius = percentiles(all_holes[:, 2]) if len(all_holes) else None
    print_percentiles('radius px', radius)
    for stride in STRIDES:
        print_percentiles(f'radius / stride {stride} (cells)', radius, scale=stride)

    nn_stats = percentiles(nn)
    print_percentiles('nearest neighbour px', nn_stats)
    for stride in STRIDES:
        print_percentiles(f'  / stride {stride} (cells)', nn_stats, scale=stride)

    result = {'images': len(entries), 'holes': int(len(all_holes)),
              'image_sizes': sorted(str(s) for s in sizes),
              'radius_px': radius, 'nn_distance_px': nn_stats, 'strides': {}}

    print()
    for stride in STRIDES:
        collisions = cell_collisions(per_image, stride)
        risk = {str(c): separation_at_risk(nn, stride, c) for c in RISK_CELLS}
        print(f"  stride {stride}: hard limit  {collisions['colliding']:5d} / "
              f"{collisions['total']} holes collide in one cell = "
              f"{100 * collisions['fraction']:.2f}%")
        soft = '   '.join(f'<{c:g} cells {100 * risk[str(c)]:5.2f}%' for c in RISK_CELLS)
        print(f"            soft limit  neighbour {soft}")
        result['strides'][str(stride)] = {'collisions': collisions, 'at_risk': risk}

    return result, entries, nn_per_image


def main(root=DATASET):
    root = os.path.abspath(root)
    out_dir = debug_dir('label_stats', root)
    print(f"dataset: {root}")
    print("CenterNet on COCO for reference: 614 / 860001 collisions = 0.07%")

    report = {'dataset_root': root}
    train_entries = train_nn = None
    for split in SPLITS:
        report[split], entries, nn_per_image = analyse_split(root, split)
        if split == 'train':
            train_entries, train_nn = entries, nn_per_image

    print("\n=== overlays ===")
    overlays = {}
    for reason, entry in pick_overlay_images(train_entries, train_nn):
        path, n = overlay(reason, entry, out_dir)
        overlays[reason] = path
        print(f"  {reason:<16s} {n:3d} holes  ->  {path}")
    report['overlays'] = overlays

    stats_path = os.path.join(out_dir, 'stats.json')
    with open(stats_path, 'w') as handle:
        json.dump(report, handle, indent=2)
    print(f"\nwrote {stats_path}")

    print("\nNow look at the overlays before trusting any number above: every green circle "
          "must sit on a hole,\nand every red centre dot must be in the middle of one. "
          "Then decide stride and sigma (PLAN.md).")
    return report


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else DATASET)
