"""Verification for codec/decode.py."""

import numpy as np
import pytest

from centerpoint.codec.decode import decode, local_maxima
from centerpoint.codec.encode import encode
from centerpoint.params import IMG_SIZE as IMG, STRIDE

CELL = float(STRIDE)          # one cell, in pixels


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


@pytest.mark.skipif(STRIDE == 1, reason="cells and pixels coincide, so nothing to confuse")
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
    gap = 3 * CELL                                  # far enough apart to stay two peaks
    holes = np.array([[100.0, 100.0, gap], [100.0 + gap, 100.0, gap]])
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


# --- behaviour under imperfect maps, which the round-trip tests above cannot see ---

def test_peak_one_cell_off_destroys_the_radius():
    """The near-miss problem, pinned. offset and radius are written ONLY at the true
    centre cell, so a peak one cell away reads geometry that was never supervised. The
    heatmap can be nearly right and the decoded radius still be nonsense."""
    cx, cy, r = 100.0, 200.0, 9.0
    t = encode(np.array([[cx, cy, r]]), IMG)
    col, row = int(cx / STRIDE), int(cy / STRIDE)

    assert t['radius'][0, row, col] == pytest.approx(r / STRIDE)
    assert t['radius'][0, row, col + 1] == 0.0, "the neighbour was never supervised"

    # Roll the whole bump one cell right: a controlled displacement, where the peak moves
    # and its shape does not. Poking individual cells instead leaves the Gaussian ring
    # around the true centre higher than the poked values, which tests something else
    # (an irregular heatmap) rather than a clean displacement.
    shifted = np.roll(t['heatmap'], 1, axis=2)
    got = decode(shifted, t['offset'], t['radius'])

    assert len(got) == 1
    assert got[0, 2] == pytest.approx(0.0), "radius decodes to zero, not to r"


def test_copying_geometry_outward_rescues_radius_but_not_centre():
    """A partial fix, and the trap in it.

    Copying each centre's radius into a 3x3 block does make the radius survive a one-cell
    peak error. Copying the *offset* does not work the same way: an offset is measured
    from its own cell, so cell col+1 needs `x - (col+1)`, not `x - col`. Copying the
    original value decodes a centre one whole cell out — worse than not spreading, since
    without it the offset reads 0 and lands on the cell centre.

    Any real implementation therefore needs signed offsets recomputed per cell, plus an
    assignment rule where neighbourhoods overlap. Not done here; this pins the trap."""
    cx, cy, r = 100.5, 200.0, 9.0
    t = encode(np.array([[cx, cy, r]]), IMG)
    col, row = int(cx / STRIDE), int(cy / STRIDE)

    plain = decode(np.roll(t['heatmap'], 1, axis=2), t['offset'], t['radius'])

    for key, ch in (('radius', 0), ('offset', 0), ('offset', 1)):
        t[key][ch, row - 1:row + 2, col - 1:col + 2] = t[key][ch, row, col]
    copied = decode(np.roll(t['heatmap'], 1, axis=2), t['offset'], t['radius'])

    assert plain[0, 2] == pytest.approx(0.0), "without spreading the radius is lost"
    assert copied[0, 2] == pytest.approx(r, abs=1e-3), "copying rescues the radius"

    assert abs(copied[0, 0] - cx) > abs(plain[0, 0] - cx), (
        "copied offsets should make the centre WORSE, not better")


@pytest.mark.parametrize('cells, expected', [(1, 1), (2, 2), (3, 2)])
def test_unequal_adjacent_peaks_merge_at_one_cell(cells, expected):
    """The realistic merge limit. Ties are kept, but a trained model will not tie: the
    3x3 window then suppresses the lower of any pair within one cell. This, not the
    equal-peak case, is what sets the minimum resolvable separation."""
    gap = cells * STRIDE
    holes = np.array([[200.0, 200.0, 6.0], [200.0 + gap, 200.0, 6.0]])
    t = encode(holes, IMG)
    hm = t['heatmap'].copy()
    hm[0, int(200.0 / STRIDE), int((200.0 + gap) / STRIDE)] *= 0.95
    assert len(decode(hm, t['offset'], t['radius'])) == expected
