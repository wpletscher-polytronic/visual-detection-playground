"""Verification for codec/decode.py."""

import numpy as np
import pytest

from centerpoint.codec.decode import decode, local_maxima
from centerpoint.codec.encode import STRIDE, encode

IMG = 640


def test_flat_background_produces_no_peaks():
    """Every cell in a constant region equals its own neighbourhood maximum, so without
    the threshold an empty 320x320 target reports ~66,000 detections."""
    flat = np.zeros((320, 320), np.float32)
    assert local_maxima(flat, threshold=0.0).sum() == 0

    flat_but_bright = np.full((320, 320), 0.2, np.float32)
    assert local_maxima(flat_but_bright, threshold=0.3).sum() == 0


def test_local_maxima_finds_the_bumps_and_nothing_else():
    hm = np.zeros((10, 10), np.float32)
    hm[3, 3] = 1.0
    hm[3, 4] = 0.6      # on the slope, its neighbour is higher
    hm[7, 8] = 0.9
    found = set(zip(*np.nonzero(local_maxima(hm, threshold=0.3))))
    assert found == {(3, 3), (7, 8)}


def test_adjacent_equal_peaks_are_both_kept():
    """Two holes one cell apart give two touching cells at 1.0. A strict `>` comparison
    would find neither greater than the other and lose both."""
    hm = np.zeros((10, 10), np.float32)
    hm[5, 5] = hm[5, 6] = 1.0
    assert local_maxima(hm, threshold=0.3).sum() == 2


def test_roundtrip_recovers_centres_and_radii():
    holes = np.array([[100.3, 200.7, 5.0], [300.0, 150.0, 8.0], [420.9, 480.1, 12.0]])
    t = encode(holes, IMG)
    got = decode(t['heatmap'], t['offset'], t['radius'])

    assert len(got) == len(holes)
    # decode returns score-sorted; sort both by x to compare like for like
    got = got[np.argsort(got[:, 0])]
    want = holes[np.argsort(holes[:, 0])]
    assert got[:, 0] == pytest.approx(want[:, 0], abs=1e-4)
    assert got[:, 1] == pytest.approx(want[:, 1], abs=1e-4)
    assert got[:, 2] == pytest.approx(want[:, 2], abs=1e-4)


def test_output_is_in_image_pixels_not_cells():
    """The likeliest unit bug in the project: forgetting to multiply by stride."""
    t = encode(np.array([[100.0, 200.0, 6.0]]), IMG)
    x, y, r, _ = decode(t['heatmap'], t['offset'], t['radius'])[0]
    assert (x, y) == pytest.approx((100.0, 200.0), abs=1e-4)
    assert (x, y) != pytest.approx((100.0 / STRIDE, 200.0 / STRIDE))


def test_scores_are_sorted_descending_and_in_range():
    rng = np.random.default_rng(0)
    holes = np.stack([rng.uniform(0, IMG, 30), rng.uniform(0, IMG, 30),
                      rng.uniform(2, 15, 30)], 1)
    t = encode(holes, IMG)
    got = decode(t['heatmap'], t['offset'], t['radius'])
    scores = got[:, 3]
    assert (np.diff(scores) <= 0).all()
    assert (scores >= 0).all() and (scores <= 1).all()


def test_sorting_keeps_each_row_intact():
    """Sorting must move whole detections, not just the score column: the x, y, r and
    score in a row have to stay the same hole."""
    holes = np.array([[100.0, 100.0, 4.0], [300.0, 300.0, 12.0], [500.0, 200.0, 8.0]])
    t = encode(holes, IMG)
    got = decode(t['heatmap'], t['offset'], t['radius'])

    for x, y, r, _score in got:
        match = np.isclose(holes[:, 0], x, atol=1e-3) & np.isclose(holes[:, 1], y, atol=1e-3)
        assert match.sum() == 1, f"({x}, {y}) is not one of the input holes"
        assert holes[match][0, 2] == pytest.approx(r, abs=1e-3), "radius came from another row"


def test_threshold_filters():
    hm = np.zeros((1, 10, 10), np.float32)
    hm[0, 3, 3], hm[0, 7, 7] = 0.9, 0.4
    zeros = np.zeros((2, 10, 10), np.float32)
    rad = np.zeros((1, 10, 10), np.float32)
    assert len(decode(hm, zeros, rad, threshold=0.3)) == 2
    assert len(decode(hm, zeros, rad, threshold=0.5)) == 1
    assert len(decode(hm, zeros, rad, threshold=0.95)) == 0


def test_no_holes_gives_an_empty_result_not_a_crash():
    t = encode(np.zeros((0, 3)), IMG)
    got = decode(t['heatmap'], t['offset'], t['radius'])
    assert got.shape == (0, 4)


def test_overlapping_detections_both_survive():
    """The project's whole premise: no suppression by overlap. Two holes 6 px apart have
    heavily overlapping circles, and both must come out."""
    holes = np.array([[100.0, 100.0, 8.0], [106.0, 100.0, 8.0]])
    t = encode(holes, IMG)
    got = decode(t['heatmap'], t['offset'], t['radius'])
    assert len(got) == 2
    separation = abs(got[0, 0] - got[1, 0])
    assert separation < got[0, 2] + got[1, 2], "circles should overlap, yet both kept"


def test_dense_real_scale_image_decodes_every_hole():
    rng = np.random.default_rng(1)
    holes = np.stack([rng.uniform(0, IMG, 125), rng.uniform(0, IMG, 125),
                      rng.uniform(2, 15, 125)], 1)
    t = encode(holes, IMG)
    peaks = int((t['heatmap'][0] == 1.0).sum())     # fewer than 125 if any cells collided
    assert len(decode(t['heatmap'], t['offset'], t['radius'])) == peaks
