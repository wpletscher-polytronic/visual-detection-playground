"""Synthetic tensors only — no images, no model, no dataset.

The loss is the piece where a bug still trains, so these tests pin semantics, not just
shapes: what counts as a positive, what the Gaussian shoulder is worth, and what the
regression terms divide by.
"""

import math

import numpy as np
import pytest
import torch

from centerpoint.codec.encode import encode
from centerpoint.model.losses import (ALPHA, BETA, LAMBDA_OFFSET, LAMBDA_RADIUS,
                                      detection_loss, focal, masked_l1)

IMG = 64
STRIDE = 4
GRID = IMG // STRIDE
HOLE = (30.0, 30.0, 6.0)


def targets_for(holes, dtype=torch.float32):
    """encode's output as a batch of one."""
    array = np.asarray(holes, dtype=np.float32).reshape(-1, 3)
    encoded = encode(array, IMG, stride=STRIDE)
    return {k: torch.from_numpy(v)[None].to(dtype) for k, v in encoded.items()}


def constant(value, channels=1, grad=False):
    return torch.full((1, channels, GRID, GRID), value, requires_grad=grad)


# --- what counts as a positive -------------------------------------------------------

def test_positives_are_exactly_the_centre_mask():
    """focal's `target == 1` and encode's mask must mark the same cells, or the heatmap
    and regression terms end up normalised by different counts."""
    targets = targets_for([HOLE, (12.0, 44.0, 3.0)])
    assert targets['heatmap'].eq(1.0).sum() == targets['mask'].sum() == 2


def test_near_perfect_prediction_is_near_zero():
    targets = targets_for([HOLE])
    logits = torch.where(targets['heatmap'].eq(1.0), 10.0, -10.0)
    assert focal(logits, targets['heatmap']).item() < 1e-3


def test_inverted_prediction_is_large():
    targets = targets_for([HOLE])
    confident = torch.where(targets['heatmap'].eq(1.0), 10.0, -10.0)
    good = focal(confident, targets['heatmap']).item()
    bad = focal(-confident, targets['heatmap']).item()
    assert bad > 100.0
    assert bad > 100 * good


# --- the Gaussian shoulder is a negative, not a target -------------------------------

def test_shoulder_penalty_is_reduced_by_one_minus_y_to_the_beta():
    """The defining property. Same wrong prediction at two cells, one on a shoulder and
    one on empty background: the ratio of their penalties is the penalty-reduction term.

    Each penalty is read as a difference against the baseline, so the baseline is made
    near-zero (centres predicted correctly) rather than ~10. focal computes in float32
    whatever it is handed, so a float64 target would buy no precision back.
    """
    targets = targets_for([HOLE])
    heat = targets['heatmap']
    shoulder = tuple(((heat > 0.5) & (heat < 0.99)).nonzero()[0].tolist())
    background = tuple(heat.eq(0.0).nonzero()[0].tolist())

    base = torch.where(heat.eq(1.0), 10.0, -10.0)
    baseline = focal(base, heat)

    def penalty_at(index):
        hot = base.clone()
        hot[index] = 2.0
        return (focal(hot, heat) - baseline).item()

    expected = (1.0 - heat[shoulder].item()) ** BETA
    assert penalty_at(shoulder) == pytest.approx(penalty_at(background) * expected, rel=1e-5)


def test_predicting_zero_on_the_shoulder_beats_reproducing_the_gaussian():
    """What separates this loss from BCE on soft targets. Reproducing the bump — the
    prediction BCE would drive towards — must score worse than peaks-only."""
    targets = targets_for([HOLE])
    heat = targets['heatmap']
    reproduce_the_bump = torch.logit(heat.clamp(1e-6, 1 - 1e-6))
    peaks_only = torch.where(heat.eq(1.0), 10.0, -10.0)
    assert focal(peaks_only, heat).item() < focal(reproduce_the_bump, heat).item()


# --- numerics ------------------------------------------------------------------------

def test_focal_reads_logits_not_probabilities():
    """All-zero input means p = 0.5, not p = 0. Analytic value on an empty target: every
    cell is a negative at full weight, contributing 0.5^ALPHA * log(0.5)."""
    empty = torch.zeros(1, 1, GRID, GRID)
    expected = GRID * GRID * 0.5 ** ALPHA * math.log(2.0)
    assert focal(torch.zeros_like(empty), empty).item() == pytest.approx(expected, rel=1e-5)


def test_empty_target_is_finite_and_negative_only():
    """An image with no holes: num_pos clamps to 1 rather than dividing by zero."""
    empty = torch.zeros(1, 1, GRID, GRID)
    loss = focal(torch.full_like(empty, -6.0), empty)
    assert torch.isfinite(loss) and 0.0 < loss.item() < 1.0


@pytest.mark.parametrize('value', [-1e4, -30.0, 30.0, 1e4])
def test_extreme_logits_stay_finite_forwards_and_backwards(value):
    targets = targets_for([HOLE])
    logits = constant(value, grad=True)
    loss = focal(logits, targets['heatmap'])
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


# --- regression normalisation --------------------------------------------------------

CENTRES = ((2, 3), (5, 7))


def two_centre_mask():
    mask = torch.zeros(1, 1, GRID, GRID)
    for row, col in CENTRES:
        mask[0, 0, row, col] = 1.0
    return mask


def test_offset_components_sum_per_centre_rather_than_averaging():
    """0.3 and 0.4 wrong at each of two centres is 0.7 per centre. Dividing by the scalar
    element count would give 0.35, and by the cell count 0.0055."""
    pred = torch.zeros(1, 2, GRID, GRID)
    for row, col in CENTRES:
        pred[0, 0, row, col] = 0.3
        pred[0, 1, row, col] = -0.4
    assert masked_l1(pred, torch.zeros_like(pred), two_centre_mask()).item() == pytest.approx(0.7)


def test_radius_uses_the_same_centre_denominator():
    pred = torch.zeros(1, 1, GRID, GRID)
    pred[0, 0, 2, 3] = 1.0
    pred[0, 0, 5, 7] = 3.0
    assert masked_l1(pred, torch.zeros_like(pred), two_centre_mask()).item() == pytest.approx(2.0)


def test_values_outside_the_mask_do_not_move_the_loss():
    mask = two_centre_mask()
    target = torch.zeros(1, 2, GRID, GRID)
    noisy = torch.randn_like(target) * 100.0
    for row, col in CENTRES:
        noisy[:, :, row, col] = 0.0
    assert masked_l1(noisy, target, mask).item() == pytest.approx(0.0)


def test_no_gradient_outside_the_mask():
    mask = two_centre_mask()
    pred = torch.randn(1, 2, GRID, GRID, requires_grad=True)
    masked_l1(pred, torch.zeros(1, 2, GRID, GRID), mask).backward()
    supervised = mask.expand_as(pred).bool()
    assert (pred.grad[~supervised] == 0).all()
    assert (pred.grad[supervised] != 0).all()


def test_empty_mask_is_zero_and_leaves_no_gradient():
    """A batch with no holes must not divide by zero or push NaN into the optimiser."""
    pred = torch.randn(1, 2, GRID, GRID, requires_grad=True)
    loss = masked_l1(pred, torch.zeros(1, 2, GRID, GRID), torch.zeros(1, 1, GRID, GRID))
    loss.backward()
    assert loss.item() == 0.0
    assert torch.isfinite(pred.grad).all() and (pred.grad == 0).all()


# --- low precision, as AMP will hand it over -----------------------------------------

LOW_PRECISION = [torch.float16, torch.bfloat16]


def mask_with(count):
    mask = torch.zeros(1, 1, GRID, GRID)
    mask.view(-1)[:count] = 1.0
    return mask


@pytest.mark.parametrize('dtype', LOW_PRECISION)
def test_masked_l1_does_not_overflow_before_normalising(dtype):
    """100 errors of 1000 sum to 100,000, past float16's 65,504 ceiling, for a mean of
    only 1000. Summing in the input dtype returns inf from finite inputs.

    Only float16 can actually overflow here — bfloat16 carries float32's exponent range.
    It is parametrised anyway to pin the same contract for both.
    """
    mask = mask_with(100)
    pred = (mask * 1000.0).to(dtype)
    target = torch.zeros(1, 1, GRID, GRID, dtype=dtype)
    assert masked_l1(pred, target, mask.to(dtype)).item() == pytest.approx(1000.0)


@pytest.mark.parametrize('dtype', LOW_PRECISION)
def test_masked_l1_low_precision_gradients_are_finite(dtype):
    mask = mask_with(100).to(dtype)
    pred = torch.full((1, 1, GRID, GRID), 1000.0, dtype=dtype, requires_grad=True)
    masked_l1(pred, torch.zeros(1, 1, GRID, GRID, dtype=dtype), mask).backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize('dtype', LOW_PRECISION)
def test_focal_in_low_precision_tracks_float32(dtype):
    """Only the inputs are rounded; the arithmetic is not. Measured relative error on
    this case: 4e-6 casting, against 6e-4 (float16) and 2e-3 (bfloat16) without. The
    threshold sits between the two, so it fails if the cast is ever dropped."""
    targets = targets_for([HOLE])
    logits = torch.where(targets['heatmap'].eq(1.0), 4.0, -3.0)
    reference = focal(logits, targets['heatmap']).item()
    low = focal(logits.to(dtype), targets['heatmap'].to(dtype)).item()
    assert low == pytest.approx(reference, rel=1e-4)


@pytest.mark.parametrize('dtype', LOW_PRECISION)
def test_focal_low_precision_gradients_are_finite(dtype):
    targets = targets_for([HOLE], dtype=dtype)
    logits = torch.full((1, 1, GRID, GRID), -8.0, dtype=dtype, requires_grad=True)
    focal(logits, targets['heatmap']).backward()
    assert torch.isfinite(logits.grad).all()


# --- the combined loss ---------------------------------------------------------------

def outputs_for(grad=False):
    # Not radius 1.5: that is exactly HOLE's target (r / STRIDE), and L1's subgradient at
    # zero error is zero, so the backward test would pass or fail on the fixture value.
    return {'heatmap': constant(-2.0, 1, grad=grad),
            'offset': constant(0.25, 2, grad=grad),
            'radius': constant(0.5, 1, grad=grad)}


def test_detection_loss_applies_the_lambdas():
    targets = targets_for([HOLE])
    parts = detection_loss(outputs_for(), targets)
    expected = (parts['heatmap'] + LAMBDA_OFFSET * parts['offset']
                + LAMBDA_RADIUS * parts['radius'])
    assert parts['total'].item() == pytest.approx(expected.item())


def test_detection_loss_backward_reaches_all_three_heads():
    targets = targets_for([HOLE])
    outputs = outputs_for(grad=True)
    detection_loss(outputs, targets)['total'].backward()
    for name, tensor in outputs.items():
        assert torch.isfinite(tensor.grad).all(), name
        assert tensor.grad.abs().sum() > 0, name


def test_zero_hole_image_gives_a_finite_total():
    targets = targets_for([])
    parts = detection_loss(outputs_for(), targets)
    assert torch.isfinite(parts['total'])
    assert parts['offset'].item() == 0.0 and parts['radius'].item() == 0.0
