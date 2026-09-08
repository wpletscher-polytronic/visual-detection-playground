"""Render the encode -> decode round trip on real images, so it can be looked at.

A development tool, not part of the pipeline. Everything it writes lands in
outputs/debug/codec_roundtrip/ and can be deleted at any time. Re-run after changing SIGMA_SCALE,
SIGMA_MIN, STRIDE or SCORE_THRESHOLD: the tests say the codec is self-consistent, they do
not say the targets are sensible for this data.

Decoding an encoded heatmap is the strongest check available before a model exists. Any
hole that fails to come back out is one the architecture cannot represent, whatever the
model later learns.

Run either way, from anywhere:
    python -m centerpoint.inspect_codec [dataset_root]
    python centerpoint/inspect_codec.py [dataset_root]
"""

import os
import sys

import cv2
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from centerpoint.analyze_labels import (nearest_neighbour_distances,  # noqa: E402
                                        pick_overlay_images)
from centerpoint.codec.decode import SCORE_THRESHOLD, decode  # noqa: E402
from centerpoint.codec.encode import (SIGMA_MIN, SIGMA_SCALE,  # noqa: E402
                                      encode, sigma_for)
from centerpoint.params import STRIDE  # noqa: E402
from centerpoint.paths import DATASET, debug_dir  # noqa: E402
from centerpoint.data.splits import load_split_geometry  # noqa: E402

SPLIT = 'train'
BLEND = 0.45        # heatmap weight in the overlay
GT_COLOUR = (0, 255, 0)         # ground truth circles
DET_COLOUR = (0, 0, 255)        # decoded circles, drawn slightly larger so both show


def colourise(heatmap, size):
    """Heatmap -> a full-size BGR image. Nearest-neighbour, so cells stay square blocks."""
    big = cv2.resize(heatmap, (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap((big * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def label(img, text):
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    return img


def closest_pair(holes):
    """Indices of the two closest hole centres, or None with fewer than two holes."""
    if len(holes) < 2:
        return None
    i = int(np.argmin(nearest_neighbour_distances(holes)))
    d = np.linalg.norm(holes[:, :2] - holes[i, :2], axis=1)
    d[i] = np.inf
    return i, int(np.argmin(d))


def closest_pair_valley(holes, heatmap):
    """Lowest heatmap value on the straight line between the two closest hole centres.

    1.0 means the two peaks are touching with nothing between them to separate them, and
    no peak-finder can split them. Lower means there is a dip for the model to reproduce.
    Returns (distance in px, valley) or None when there are fewer than two holes.
    """
    pair = closest_pair(holes)
    if pair is None:
        return None
    i, j = pair
    centres = holes[:, :2]
    d = np.linalg.norm(centres - centres[i], axis=1)

    steps = max(int(d[j]), 2)
    ts = np.linspace(0, 1, steps + 1)
    xs = np.clip((centres[i, 0] + ts * (centres[j, 0] - centres[i, 0])) / STRIDE, 0,
                 heatmap.shape[1] - 1).astype(int)
    ys = np.clip((centres[i, 1] + ts * (centres[j, 1] - centres[i, 1])) / STRIDE, 0,
                 heatmap.shape[0] - 1).astype(int)
    return float(d[j]), float(heatmap[ys, xs].min())


def draw_circles(img, circles, colour, pad=0):
    for x, y, r in circles[:, :3]:
        cv2.circle(img, (int(round(x)), int(round(y))), max(int(round(r)) + pad, 1), colour, 1)
    return img


def worst_centre_error(holes, detections):
    """Largest distance from a decoded centre to the nearest true centre, in pixels."""
    if not len(detections) or not len(holes):
        return 0.0
    gaps = np.linalg.norm(detections[:, None, :2] - holes[None, :, :2], axis=-1)
    return float(gaps.min(axis=1).max())


def render(reason, entry, out_dir):
    stem, image_path, holes, (w, _h) = entry
    targets = encode(holes, w)
    heatmap = targets['heatmap'][0]
    detections = decode(targets['heatmap'], targets['offset'], targets['radius'])

    img = cv2.imread(image_path)
    heat = colourise(heatmap, w)
    blend = cv2.addWeighted(img, 1 - BLEND, heat, BLEND, 0)

    circles = draw_circles(img.copy(), holes, GT_COLOUR)
    circles = draw_circles(circles, detections, DET_COLOUR, pad=2)

    panel = np.hstack([label(img.copy(), 'source'),
                       label(heat.copy(), f'encoded {heatmap.shape[0]}x{heatmap.shape[1]}'),
                       label(blend.copy(), 'blend'),
                       label(circles.copy(), 'green truth / red decoded (+2px)')])
    cv2.imwrite(os.path.join(out_dir, f'{reason}_{stem}.png'), panel)

    peaks = int((heatmap == 1.0).sum())
    errors = worst_centre_error(holes, detections)
    return peaks, len(detections), errors, closest_pair_valley(holes, heatmap)


def main(root=DATASET):
    root = os.path.abspath(root)
    out_dir = debug_dir('codec_roundtrip', root)
    print(f"dataset: {root}")
    print(f"stride {STRIDE}   sigma = max({SIGMA_SCALE} * radius_cells, {SIGMA_MIN})")
    print(f"  a 1 px radius hole gets sigma {sigma_for(1.0 / STRIDE):.2f}, "
          f"a 17 px one {sigma_for(17.0 / STRIDE):.2f}")
    print(f"decode score threshold {SCORE_THRESHOLD}")

    entries = load_split_geometry(root, SPLIT)
    nn_per_image = [nearest_neighbour_distances(e[2]) for e in entries]

    print()
    print("  case             holes  peaks  decoded  worst err   closest pair   valley")
    for reason, entry in pick_overlay_images(entries, nn_per_image):
        peaks, n_decoded, err, valley = render(reason, entry, out_dir)
        n = len(entry[2])
        pair = f"{valley[0]:6.1f} px" if valley else "     n/a"
        # 1.00 means the two peaks touch with no dip between them: nothing to split on.
        depth = f"{valley[1]:6.2f}" if valley else "   n/a"
        print(f"  {reason:<16s} {n:5d}  {peaks:5d}  {n_decoded:7d}  {err:8.1e}   {pair}   {depth}")

    print(f"\nwrote {out_dir}")
    robustness_sweep(entries)
    merge_rate(entries)

    print("Look for: every green circle ringed in red, and no red ring on a digit or a "
          "ring line.")
    print("A green circle with no red ring is a hole the codec cannot represent — "
          "expect exactly one per same-cell collision.")


# ---------------------------------------------------------------------------
# Robustness: what decode does with imperfect maps.
#
# Everything above feeds decode a perfect heatmap, where the peak is always in the exact
# cell encode wrote the geometry into. A trained model will not do that. These sweeps
# degrade the maps the way a model would and report what survives.
# ---------------------------------------------------------------------------

JITTER_CELLS = (0, 1, 2)
NOISE_LEVELS = (0.0, 0.02, 0.05, 0.10)
SPREAD_CELLS = (0, 1, 2)


def shift_peaks(targets, cells):
    """Move every heatmap peak sideways, leaving the geometry maps where they were.

    The pessimistic simulation of a near miss: geometry is zero outside the true centre
    cell, and nothing supervised it there, so a model's value is whatever the convolution
    happens to emit. Zero stands in for "arbitrary".
    """
    shifted = dict(targets)
    shifted['heatmap'] = np.roll(targets['heatmap'], cells, axis=2)
    return shifted


def spread_geometry(targets, cells):
    """Copy each centre's offset and radius outward into a (2*cells+1)^2 block.

    Simulates a model whose geometry predictions are locally consistent rather than a
    single supervised pixel. If this rescues a shifted peak, the fix for the near-miss
    problem is to supervise geometry over a neighbourhood.
    """
    if cells == 0:
        return targets
    spread = {k: v.copy() for k, v in targets.items()}
    n = targets['mask'].shape[-1]
    rows, cols = np.nonzero(targets['mask'][0])
    for row, col in zip(rows, cols):
        r0, r1 = max(0, row - cells), min(n, row + cells + 1)
        c0, c1 = max(0, col - cells), min(n, col + cells + 1)
        spread['offset'][0, r0:r1, c0:c1] = targets['offset'][0, row, col]
        spread['offset'][1, r0:r1, c0:c1] = targets['offset'][1, row, col]
        spread['radius'][0, r0:r1, c0:c1] = targets['radius'][0, row, col]
    return spread


def add_noise(targets, sigma, seed=0):
    """Additive i.i.d. Gaussian noise on the heatmap PROBABILITIES, clipped to [0, 1].

    Assumptions worth naming, because they bound what the result means:
    - noise is added to probabilities, not to logits, and is spatially uncorrelated; a
      real model's error is neither
    - the same sigma is used at both strides, so a finer grid sees the same per-cell
      amplitude over four times as many cells
    - clipping at 1.0 creates flat plateaus near peaks, and decode keeps ties, so some
      extra detections come from clipping rather than from the noise itself. Measured on
      the densest v30 image at sigma 0.10: 199 detections clipped vs 187 unclipped.
    """
    if sigma == 0:
        return targets
    rng = np.random.default_rng(seed)
    noisy = dict(targets)
    noisy['heatmap'] = np.clip(
        targets['heatmap'] + rng.normal(0, sigma, targets['heatmap'].shape), 0, 1
    ).astype(np.float32)
    return noisy


def errors_against_truth(detections, holes):
    """Median centre and radius error, matching each true hole to its nearest detection."""
    if not len(detections) or not len(holes):
        return float('nan'), float('nan')
    gaps = np.linalg.norm(detections[:, None, :2] - holes[None, :, :2], axis=-1)
    nearest = gaps.argmin(axis=0)
    centre = np.linalg.norm(detections[nearest, :2] - holes[:, :2], axis=1)
    radius = np.abs(detections[nearest, 2] - holes[:, 2])
    return float(np.median(centre)), float(np.median(radius))


def sweep(holes, img_size, stride, make_maps):
    """Decode once per perturbation setting and report what came out."""
    targets = encode(holes, img_size, stride=stride)
    got = decode(*[make_maps(targets)[k] for k in ('heatmap', 'offset', 'radius')],
                 stride=stride)
    centre, radius = errors_against_truth(got, holes)
    return len(got), centre, radius


def robustness_sweep(entries):
    """Print how decode degrades as the maps get less ideal, at stride 4 and stride 2."""
    holes = max((e[2] for e in entries), key=len)          # the densest image
    img_size = entries[0][3][0]
    print(f"\n\n=== robustness: {len(holes)} holes, the densest image in the split ===")

    print("\n  peak shifted sideways by whole cells (geometry maps left in place)")
    print("  stride  shift   decoded   median centre err   median radius err")
    for stride in (4, 2):
        for cells in JITTER_CELLS:
            n, c, r = sweep(holes, img_size, stride,
                            lambda t, k=cells: shift_peaks(t, k))
            print(f"  {stride:6d}  {cells:5d}   {n:7d}   {c:15.2f} px   {r:14.2f} px")

    print("\n  same 1-cell shift, but geometry spread over a neighbourhood first")
    print("  stride  spread  decoded   median centre err   median radius err")
    for stride in (4, 2):
        for cells in SPREAD_CELLS:
            n, c, r = sweep(holes, img_size, stride,
                            lambda t, k=cells: shift_peaks(spread_geometry(t, k), 1))
            print(f"  {stride:6d}  {cells:5d}   {n:7d}   {c:15.2f} px   {r:14.2f} px")

    print("\n  additive noise on the heatmap")
    print("  stride  sigma   decoded   median centre err   median radius err")
    for stride in (4, 2):
        for sigma in NOISE_LEVELS:
            n, c, r = sweep(holes, img_size, stride,
                            lambda t, s=sigma: add_noise(t, s))
            print(f"  {stride:6d}  {sigma:5.2f}   {n:7d}   {c:15.2f} px   {r:14.2f} px")


def merge_rate(entries, strides=(4, 2), jitter=0.01, seed=0):
    """Measured merge rate over the whole split, not a proximity estimate.

    Encode the real labels, break exact ties with a small multiplicative jitter (a trained
    model never ties), decode, and compare against the peaks the encoder produced. The
    shortfall is what a 3x3 window actually suppresses — which depends on grid alignment
    and relative peak heights, and so cannot be read off a distance threshold.

    Still an upper bound on quality: the heatmap shape is ideal apart from the tie-break.
    """
    print()
    print("  measured merge rate (ideal heatmaps, ties broken by 1% jitter)")
    print("  stride   holes    peaks   detections   collision   merge")
    for stride in strides:
        rng = np.random.default_rng(seed)
        n_holes = n_peaks = n_det = 0
        for _stem, _path, holes, (w, _h) in entries:
            if not len(holes):
                continue
            t = encode(holes, w, stride=stride)
            hm = (t['heatmap'] * (1 + jitter * rng.random(t['heatmap'].shape))
                  ).astype(np.float32)
            n_holes += len(holes)
            n_peaks += int((t['heatmap'][0] == 1.0).sum())
            n_det += len(decode(hm, t['offset'], t['radius'], stride=stride))
        coll, merged = n_holes - n_peaks, n_peaks - n_det
        print(f"  {stride:6d}  {n_holes:6d}   {n_peaks:6d}   {n_det:10d}   "
              f"{100*coll/n_holes:6.2f}%  {100*merged/n_holes:6.2f}%")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else DATASET)
