"""Verification for codec/encode.py.

Everything here runs on hand-written points, no images and no files, so a failure points
at the encoder and nothing else.
"""

import numpy as np
import pytest

from centerpoint.codec.encode import STRIDE, encode, sigma_for

IMG = 640
N = IMG // STRIDE


def test_shapes_and_dtypes():
    t = encode(np.array([[100.0, 200.0, 5.0]]), IMG)
    assert t['heatmap'].shape == (1, N, N)
    assert t['offset'].shape == (2, N, N)
    assert t['radius'].shape == (1, N, N)
    assert t['mask'].shape == (1, N, N)
    assert all(v.dtype == np.float32 for v in t.values())


def test_peak_lands_on_the_correct_cell():
    cx, cy = 100.0, 200.0
    hm = encode(np.array([[cx, cy, 5.0]]), IMG)['heatmap'][0]
    assert np.unravel_index(hm.argmax(), hm.shape) == (int(cy / STRIDE), int(cx / STRIDE))
    assert hm[int(cy / STRIDE), int(cx / STRIDE)] == 1.0     # exactly, not approximately


def test_heatmap_stays_in_range():
    rng = np.random.default_rng(0)
    holes = np.stack([rng.uniform(0, IMG, 40), rng.uniform(0, IMG, 40),
                      rng.uniform(1, 17, 40)], 1)
    hm = encode(holes, IMG)['heatmap']
    assert hm.min() >= 0.0 and hm.max() <= 1.0


def test_offset_recovers_the_sub_cell_remainder():
    cx, cy = 101.3, 200.7                      # deliberately not on a cell boundary
    t = encode(np.array([[cx, cy, 5.0]]), IMG)
    ix, iy = int(cx / STRIDE), int(cy / STRIDE)
    assert t['offset'][0, iy, ix] == pytest.approx(cx / STRIDE - ix, abs=1e-6)
    assert t['offset'][1, iy, ix] == pytest.approx(cy / STRIDE - iy, abs=1e-6)
    assert 0.0 <= t['offset'][0, iy, ix] < 1.0
    assert 0.0 <= t['offset'][1, iy, ix] < 1.0


def test_radius_is_stored_in_cells():
    cx, cy, r = 100.0, 200.0, 5.0
    t = encode(np.array([[cx, cy, r]]), IMG)
    # Arrays are indexed [row, col] = [y, x]. Writing these the wrong way round is the
    # single easiest bug to introduce here, so spell both out.
    ix, iy = int(cx / STRIDE), int(cy / STRIDE)
    assert (ix, iy) == (50, 100)
    assert t['radius'][0, iy, ix] == pytest.approx(r / STRIDE)
    assert t['radius'][0, ix, iy] == 0.0, "transposed index must NOT also hold the value"


def test_mask_marks_only_centre_cells():
    holes = np.array([[100.0, 200.0, 5.0], [300.0, 400.0, 8.0]])
    t = encode(holes, IMG)
    assert t['mask'].sum() == 2
    # offset and radius must be untouched everywhere the mask is 0, or the loss would be
    # trained against zeros it should never have seen.
    assert (t['offset'] * (1 - t['mask'])).sum() == 0
    assert (t['radius'] * (1 - t['mask'])).sum() == 0


def test_peak_count_equals_hole_count_when_cells_differ():
    holes = np.array([[100.0, 100.0, 5.0], [200.0, 200.0, 5.0], [300.0, 300.0, 5.0]])
    hm = encode(holes, IMG)['heatmap'][0]
    assert (hm == 1.0).sum() == 3


def test_two_centres_in_one_cell_collide_to_a_single_peak():
    """The hard limit. Measured at 0.01% of holes at stride 2 — real but negligible."""
    holes = np.array([[100.0, 100.0, 5.0], [100.5, 100.5, 5.0]])   # same cell at stride 2
    t = encode(holes, IMG)
    assert (t['heatmap'][0] == 1.0).sum() == 1
    assert t['mask'].sum() == 1


def test_overlapping_gaussians_take_maximum_not_sum():
    holes = np.array([[100.0, 100.0, 5.0], [102.0, 100.0, 5.0]])   # adjacent cells
    hm = encode(holes, IMG)['heatmap']
    assert hm.max() <= 1.0, "summing would push overlapping bumps above 1.0"


@pytest.mark.parametrize('gap_cells, expect_valley', [
    (1, False),    # adjacent cells: two 1.0s, but nothing between them to separate them
    (2, True),     # one cell between the peaks, and it dips
    (3, True),
    (5, True),
])
def test_separating_valley_needs_two_cells_of_gap(gap_cells, expect_valley):
    """What actually limits resolving two close holes — and it is NOT the encoder.

    Element-wise maximum means two centres in different cells always yield two cells at
    exactly 1.0, at any separation. So nothing merges here. What varies is whether there
    is a *dip between* them for a model to reproduce and a peak-finder to split on.

    At one cell of gap the peaks are adjacent with no cell in between, so the target is a
    flat two-cell plateau. Whether that decodes as one detection or two is decided by the
    plateau rule in codec/decode.py, not here. Roughly 1.4% of real holes at stride 2 sit
    in that regime.
    """
    x0 = 100.0
    holes = np.array([[x0, 100.0, 5.0], [x0 + gap_cells * STRIDE, 100.0, 5.0]])
    hm = encode(holes, IMG)['heatmap'][0]
    iy, ix0 = int(100 / STRIDE), int(x0 / STRIDE)

    assert (hm == 1.0).sum() == 2, "distinct cells must always give two peaks"
    between = hm[iy, ix0 + 1:ix0 + gap_cells]
    assert (len(between) > 0 and between.min() < 1.0) == expect_valley


@pytest.mark.parametrize('cx, cy', [
    (-5.0, 100.0),      # well off the left edge
    (-0.01, 100.0),     # barely off it: int() would truncate this to cell 0 and store a
                        # negative offset, so it must not slip through
    (100.0, -1.0),
    (float(IMG), 100.0),      # exactly on the far edge -> cell index n, one past the grid
    (IMG + 10.0, 100.0),
])
def test_out_of_frame_holes_raise_rather_than_being_dropped(cx, cy):
    """Loud, not silent. Whatever moves a hole out of frame must filter it itself —
    a hole quietly discarded here is training signal lost with no error to notice."""
    with pytest.raises(AssertionError):
        encode(np.array([[cx, cy, 5.0]]), IMG)


def test_every_written_offset_is_in_unit_range():
    """The invariant the bug above broke, asserted over the whole grid rather than at one
    hand-picked cell."""
    rng = np.random.default_rng(1)
    holes = np.stack([rng.uniform(0, IMG, 200), rng.uniform(0, IMG, 200),
                      rng.uniform(1, 17, 200)], 1)
    t = encode(holes, IMG)
    written = t['offset'][:, t['mask'][0] == 1.0]
    assert written.size > 0
    assert (written >= 0.0).all() and (written < 1.0).all()


def test_corners_clip_without_error():
    for cx, cy in [(0.0, 0.0), (IMG - 1.0, 0.0), (0.0, IMG - 1.0), (IMG - 1.0, IMG - 1.0)]:
        t = encode(np.array([[cx, cy, 17.0]]), IMG)      # large radius -> window overhangs
        assert t['mask'].sum() == 1
        assert t['heatmap'].max() == 1.0


def test_no_holes_gives_all_zero_targets():
    t = encode(np.zeros((0, 3)), IMG)
    assert all(v.shape[-2:] == (N, N) for v in t.values())
    assert all(v.sum() == 0 for v in t.values())


def test_sigma_never_falls_below_one_cell():
    assert sigma_for(0.0) == 1.0
    assert sigma_for(0.5) == 1.0        # a 1 px radius hole at stride 2
    assert sigma_for(10.0) > 1.0        # large holes scale up


def test_img_size_must_divide_by_stride():
    with pytest.raises(AssertionError):
        encode(np.zeros((0, 3)), 641)
