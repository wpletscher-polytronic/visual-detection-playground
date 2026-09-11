"""The training script's pure logic: the schedule, the argument guards, the preview pick.

No model and no real dataset — the training loop itself is exercised by actually running
the script, not by mocking a forward pass into submission.
"""

import numpy as np
import pytest

from centerpoint.metrics import CANDIDATE_FLOOR
from centerpoint.train import (LEARNING_RATE, MIN_LEARNING_RATE, SELECTION_METRIC,
                               learning_rate_at, parse_args, preview_indices)


# --- the schedule -------------------------------------------------------------------------

def test_warmup_rises_linearly_from_almost_zero_to_the_base_rate():
    rates = [learning_rate_at(step, 1000, base_lr=1.0, warmup_steps=100) for step in range(1, 101)]
    assert rates[0] == pytest.approx(0.01)
    assert rates[-1] == pytest.approx(1.0)
    assert np.allclose(np.diff(rates), 0.01)


def test_the_rate_never_exceeds_the_base_rate():
    """A cosine that overshoots at the handover would undo the warmup's whole purpose."""
    rates = [learning_rate_at(step, 2000, base_lr=1e-3, warmup_steps=500) for step in range(1, 2001)]
    assert max(rates) == pytest.approx(1e-3)


def test_the_cosine_decays_monotonically_after_warmup():
    rates = [learning_rate_at(step, 2000, base_lr=1e-3, warmup_steps=500)
             for step in range(500, 2001)]
    assert np.all(np.diff(rates) <= 1e-15), "the post-warmup schedule must not rise"


def test_the_final_step_lands_on_the_floor():
    assert learning_rate_at(2000, 2000, base_lr=1e-3, warmup_steps=500) \
        == pytest.approx(MIN_LEARNING_RATE)


def test_the_schedule_stays_finite_and_positive_everywhere():
    for step in range(1, 3001):                       # deliberately past total_steps
        rate = learning_rate_at(step, 2000, base_lr=1e-3, warmup_steps=500)
        assert np.isfinite(rate) and rate > 0


@pytest.mark.parametrize('total_steps', [1, 2, 7, 499])
def test_a_run_shorter_than_the_warmup_still_schedules(total_steps):
    """A one-epoch smoke test has fewer steps than the 500-step warmup. Clamped, not
    divided by a warmup longer than the run."""
    rates = [learning_rate_at(step, total_steps, base_lr=1e-3, warmup_steps=500)
             for step in range(1, total_steps + 1)]
    assert all(np.isfinite(rate) and rate > 0 for rate in rates)
    assert max(rates) <= 1e-3 + 1e-15


def test_zero_warmup_starts_on_the_cosine():
    assert learning_rate_at(1, 100, base_lr=1.0, warmup_steps=0) == pytest.approx(1.0, abs=1e-3)


# --- argument guards ------------------------------------------------------------------------

@pytest.mark.parametrize('argv', [['--epochs', '0'], ['--batch', '0'], ['--lr', '0'],
                                  ['--lr', '-1'], ['--weight-decay', '-0.1'],
                                  ['--warmup', '-1'], ['--workers', '-1'],
                                  ['--operating-threshold', '0.01']])
def test_invalid_settings_are_rejected_before_anything_loads(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_an_operating_threshold_below_the_candidate_floor_is_refused():
    """Reporting P/R below the floor would read off a curve whose tail was never generated."""
    with pytest.raises(SystemExit):
        parse_args(['--operating-threshold', str(CANDIDATE_FLOOR / 2)])
    parse_args(['--operating-threshold', str(CANDIDATE_FLOOR)])


def test_defaults_are_the_documented_recipe():
    args = parse_args([])
    assert (args.epochs, args.batch, args.lr) == (50, 8, LEARNING_RATE)
    assert (args.weight_decay, args.warmup, args.seed) == (1e-4, 500, 0)
    assert args.limit == 0, "a real run must not be truncated by default"


def test_the_selection_metric_is_ap_not_f1_at_an_arbitrary_threshold():
    """Selecting on F1 at 0.3 would bake a pre-model threshold into which weights survive."""
    assert SELECTION_METRIC == 'AP_center_2px'


# --- the preview pick -----------------------------------------------------------------------

class FakeDataset:
    """Just enough of a Dataset to exercise the selection rule."""

    def __init__(self, hole_counts):
        self.hole_counts = hole_counts

    def __len__(self):
        return len(self.hole_counts)

    def __getitem__(self, index):
        return {'holes': np.zeros((self.hole_counts[index], 3), dtype=np.float32)}


def test_previews_skip_images_with_no_holes():
    """An empty target panel says nothing about whether the heatmap is forming."""
    assert preview_indices(FakeDataset([0, 0, 3, 0, 2, 5]), count=2) == [2, 4]


def test_previews_are_the_same_every_run():
    dataset = FakeDataset([0, 1, 2, 3, 4])
    assert preview_indices(dataset, count=3) == preview_indices(dataset, count=3)


def test_previews_cope_with_a_split_that_has_no_holes_at_all():
    assert preview_indices(FakeDataset([0, 0, 0]), count=4) == []


def test_previews_stop_at_the_requested_count():
    assert len(preview_indices(FakeDataset([1] * 20), count=4)) == 4
