"""The overfit gate's pure logic: matching, counting, and the pass/fail rule.

No model, no dataset and no device — everything here is numpy in, verdict out. The point
is that a failing run cannot be explained away by the scorer being wrong, and that the
thresholds are asserted rather than trusted.
"""

import itertools
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from centerpoint.metrics import assign_optimal, centre_criterion
from centerpoint.overfit import (GATE, chunks, gate_verdict, parse_args,
                                 preflight, score_image, select_indices, source_stem)

TOLERANCE = GATE['centre_tolerance_px']

# The subprocess regression test starts a fresh interpreter, which has none of pytest's
# path setup, so it has to be told where the package lives.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def match_by_distance(gt_xy, det_xy, tolerance):
    """Pairs with their distances, composed from the two functions metrics.py exports.

    metrics.py deliberately does not ship this: it is the shape the gate wants, not a
    third matching rule, and a convenience wrapper with one caller does not belong in the
    evaluation contract. score_image composes the same two calls inline.
    """
    feasible, distances = centre_criterion(gt_xy, det_xy, tolerance)
    return [(i, j, float(distances[i, j])) for i, j in assign_optimal(feasible, distances)]


def holes(*rows):
    """Ground truth in the (n, 3) shape the Dataset hands over."""
    return np.array(rows, dtype=np.float32).reshape(-1, 3)


def detections(*rows):
    """Detections in the (m, 4) shape decode returns; the score never affects matching."""
    return np.array([(x, y, r, 0.9) for x, y, r in rows], dtype=np.float32).reshape(-1, 4)


def greedy_match(gt_xy, det_xy, tolerance):
    """Naive greedy nearest-neighbour, present only so a test can show it losing.

    Repeatedly takes the globally closest feasible pair and removes both from play. This
    is the obvious implementation and it is what match_by_distance must not be.
    """
    gaps = np.linalg.norm(np.asarray(gt_xy)[:, None] - np.asarray(det_xy)[None, :], axis=-1)
    pairs = sorted((gaps[i, j], i, j)
                   for i in range(len(gt_xy)) for j in range(len(det_xy))
                   if gaps[i, j] <= tolerance)

    taken_gt, taken_det, matched = set(), set(), []
    for distance, i, j in pairs:
        if i in taken_gt or j in taken_det:
            continue
        taken_gt.add(i)
        taken_det.add(j)
        matched.append((i, j, distance))
    return matched


# --- matching -----------------------------------------------------------------------------

def test_exact_predictions_match_one_to_one():
    gt = [(10.0, 10.0), (40.0, 80.0)]
    matched = match_by_distance(gt, gt, TOLERANCE)
    assert [(i, j) for i, j, _d in matched] == [(0, 0), (1, 1)]
    assert all(d == pytest.approx(0.0) for _i, _j, d in matched)


def test_a_detection_beyond_the_tolerance_matches_nothing():
    matched = match_by_distance([(10.0, 10.0)], [(10.0, 10.0 + TOLERANCE + 0.01)], TOLERANCE)
    assert matched == []


def test_the_tolerance_boundary_is_inclusive():
    matched = match_by_distance([(10.0, 10.0)], [(10.0, 10.0 + TOLERANCE)], TOLERANCE)
    assert len(matched) == 1


def test_each_detection_is_used_at_most_once():
    """Two holes competing for one detection: exactly one of them may claim it."""
    matched = match_by_distance([(10.0, 10.0), (10.5, 10.0)], [(10.2, 10.0)], TOLERANCE)
    assert len(matched) == 1


def test_greedy_nearest_neighbour_loses_a_valid_assignment():
    """The reason this module does not use greedy matching, as a concrete configuration.

    On a line with a 1 px tolerance: hole A at 0.0 can only reach detection P at 0.6, and
    hole B at 1.05 can reach P (0.45 away) or Q at 2.05 (1.0 away, exactly at tolerance).
    Greedy takes the globally closest pair, B-P, which strands A. A valid assignment of
    both exists and costs more in total, which is precisely what greedy will not consider.
    """
    gt = [(0.0, 0.0), (1.05, 0.0)]
    det = [(0.6, 0.0), (2.05, 0.0)]

    assert len(greedy_match(gt, det, 1.0)) == 1
    matched = match_by_distance(gt, det, 1.0)
    assert [(i, j) for i, j, _d in matched] == [(0, 0), (1, 1)]


def test_minimum_total_distance_among_the_largest_assignments():
    """Both pairings match everything; the cheaper one must be the one reported."""
    gt = [(0.0, 0.0), (0.0, 1.0)]
    det = [(0.0, 0.1), (0.0, 1.1)]
    matched = match_by_distance(gt, det, 5.0)
    assert [(i, j) for i, j, _d in matched] == [(0, 0), (1, 1)]
    assert sum(d for _i, _j, d in matched) == pytest.approx(0.2, abs=1e-6)


@pytest.mark.parametrize('gt, det', [([], [(1.0, 1.0)]), ([(1.0, 1.0)], []), ([], [])])
def test_an_empty_side_matches_nothing_without_raising(gt, det):
    assert match_by_distance(gt, det, TOLERANCE) == []


# --- per-image scoring ----------------------------------------------------------------------

def test_counts_and_errors_for_a_clean_image():
    report = score_image(holes((10.0, 10.0, 5.0), (40.0, 80.0, 6.0)),
                         detections((10.5, 10.0, 5.25), (40.0, 80.0, 7.0)))
    assert (report['n_gt'], report['n_detected'], report['n_matched']) == (2, 2, 2)
    assert report['unmatched_gt'] == [] and report['extra_detections'] == []
    assert report['centre_errors'] == pytest.approx([0.5, 0.0], abs=1e-5)
    assert report['radius_errors'] == pytest.approx([0.25, 1.0], abs=1e-5)
    assert report['max_radius_error'] == pytest.approx(1.0, abs=1e-5)


def test_a_missed_hole_is_reported_by_index():
    report = score_image(holes((10.0, 10.0, 5.0), (400.0, 400.0, 5.0)),
                         detections((10.0, 10.0, 5.0)))
    assert report['n_matched'] == 1
    assert report['unmatched_gt'] == [1]
    assert report['extra_detections'] == []


def test_a_spurious_detection_is_reported_by_index():
    report = score_image(holes((10.0, 10.0, 5.0)),
                         detections((10.0, 10.0, 5.0), (300.0, 300.0, 4.0)))
    assert report['n_matched'] == 1
    assert report['unmatched_gt'] == []
    assert report['extra_detections'] == [1]


def test_an_image_with_no_detections_still_scores():
    report = score_image(holes((10.0, 10.0, 5.0)), detections())
    assert report['n_detected'] == 0 and report['unmatched_gt'] == [0]
    assert np.isnan(report['median_centre_error'])


# --- the verdict ------------------------------------------------------------------------------

def clean_reports(count=3):
    """Reports that pass every criterion, as the baseline the failure tests perturb."""
    return [(f'img{i}', score_image(holes((10.0, 10.0, 5.0), (40.0, 80.0, 6.0)),
                                    detections((10.2, 10.0, 5.1), (40.0, 80.2, 6.2))))
            for i in range(count)]


def test_a_memorised_set_passes():
    passed, aggregates, failures = gate_verdict(clean_reports())
    assert passed and failures == []
    assert aggregates['n_matches'] == 6
    assert aggregates['median_centre_error'] < GATE['median_centre_error_px']


def test_one_bad_image_fails_the_gate_however_good_the_medians_are():
    """The whole point of scoring per image: seven perfect ones must not average away one
    that decoded the wrong number of holes."""
    reports = clean_reports(7)
    reports.append(('img7', score_image(holes((10.0, 10.0, 5.0), (400.0, 400.0, 5.0)),
                                        detections((10.0, 10.0, 5.0)))))

    passed, aggregates, failures = gate_verdict(reports)
    assert not passed
    assert aggregates['median_centre_error'] < GATE['median_centre_error_px']
    assert any('img7' in f and 'decoded 1' in f for f in failures)
    assert any('img7' in f and 'no detection within' in f for f in failures)


def test_an_extra_detection_fails_even_when_every_hole_matched():
    reports = clean_reports(1)
    reports.append(('busy', score_image(holes((10.0, 10.0, 5.0)),
                                        detections((10.0, 10.0, 5.0), (300.0, 300.0, 5.0)))))
    passed, _aggregates, failures = gate_verdict(reports)
    assert not passed
    assert any('busy' in f and 'matching no hole' in f for f in failures)


def test_centre_error_exactly_at_the_threshold_is_not_below_it():
    """The criterion is 'below 1 px', so 1.0 px must fail rather than squeak through."""
    offset = GATE['median_centre_error_px']
    reports = [('a', score_image(holes((10.0, 10.0, 5.0)),
                                 detections((10.0 + offset, 10.0, 5.0))))]
    passed, aggregates, failures = gate_verdict(reports)
    assert aggregates['median_centre_error'] == pytest.approx(offset, abs=1e-5)
    assert not passed and any('median centre error' in f for f in failures)


def test_radius_error_beyond_the_worst_case_limit_fails():
    """Counts and centres are perfect; only the radius is wrong. Nothing else may mask it."""
    gt_radius = 5.0
    reports = [('a', score_image(holes((10.0, 10.0, gt_radius)),
                                 detections((10.0, 10.0, gt_radius + 3.5))))]
    passed, _aggregates, failures = gate_verdict(reports)
    assert not passed and any('worst radius error' in f for f in failures)


def test_a_negative_matched_radius_fails():
    """detections_from clamps at inference, so this is a guard against it being bypassed."""
    reports = [('a', score_image(holes((10.0, 10.0, 0.5)), detections((10.0, 10.0, -0.5))))]
    passed, _aggregates, failures = gate_verdict(reports)
    assert not passed and any('non-finite or negative' in f for f in failures)


def test_no_matches_at_all_is_a_failure_not_a_nan_pass():
    reports = [('a', score_image(holes((10.0, 10.0, 5.0)), detections()))]
    passed, aggregates, failures = gate_verdict(reports)
    assert not passed
    assert np.isnan(aggregates['median_centre_error'])
    assert any('no matches at all' in f for f in failures)


# --- selection and batching -------------------------------------------------------------------

def test_source_stem_strips_the_roboflow_suffix():
    assert source_stem('10_v2_jpg.rf.0ad28bcf2a55') == '10_v2_jpg'
    assert source_stem('no_suffix_here') == 'no_suffix_here'


def test_selection_takes_the_first_entry_of_each_distinct_source():
    entries = [(f'{stem}.rf.{i}', 'img', 'lbl') for i, stem in
               enumerate(['a', 'a', 'b', 'c', 'c', 'c', 'd'])]
    assert select_indices(entries, count=3) == [0, 2, 3]


def test_selection_raises_rather_than_returning_a_short_set():
    entries = [('a.rf.0', 'img', 'lbl'), ('a.rf.1', 'img', 'lbl')]
    with pytest.raises(RuntimeError, match='only 1 distinct'):
        select_indices(entries, count=8)


@pytest.mark.parametrize('size, expected', [
    (8, [[0, 1, 2, 3, 4, 5, 6, 7]]),
    (4, [[0, 1, 2, 3], [4, 5, 6, 7]]),
    (2, [[0, 1], [2, 3], [4, 5], [6, 7]]),
])
def test_the_memory_fallback_covers_the_same_eight_images(size, expected):
    """Whatever the batch size, every image is still trained on and none is duplicated."""
    assert chunks(8, size) == expected
    assert sorted(i for rows in chunks(8, size) for i in rows) == list(range(8))


# --- the matcher against an independent oracle ------------------------------------------------

def oracle_objective(gt_xy, det_xy, tolerance):
    """(cardinality, total distance) of the best assignment, by exhaustive search.

    Shares no code with match_by_distance on purpose, so agreement is evidence rather than
    a tautology. Exponential, hence the tiny sizes it is used on.
    """
    gt_xy, det_xy = np.asarray(gt_xy, float), np.asarray(det_xy, float)
    n_gt, n_det = len(gt_xy), len(det_xy)
    if not n_gt or not n_det:
        return 0, 0.0

    gaps = np.linalg.norm(gt_xy[:, None] - det_xy[None, :], axis=-1)
    for size in range(min(n_gt, n_det), 0, -1):
        best = None
        for rows in itertools.combinations(range(n_gt), size):
            for cols in itertools.permutations(range(n_det), size):
                pairs = list(zip(rows, cols))
                if all(gaps[i, j] <= tolerance for i, j in pairs):
                    total = sum(gaps[i, j] for i, j in pairs)
                    if best is None or total < best:
                        best = total
        if best is not None:
            return size, best
    return 0, 0.0


def matcher_objective(gt_xy, det_xy, tolerance):
    matches = match_by_distance(gt_xy, det_xy, tolerance)
    return len(matches), sum(d for _i, _j, d in matches)


def assert_agrees_with_oracle(gt_xy, det_xy, tolerance):
    """Objective values, not the pairs: an optimal assignment need not be unique."""
    got_size, got_cost = matcher_objective(gt_xy, det_xy, tolerance)
    want_size, want_cost = oracle_objective(gt_xy, det_xy, tolerance)
    assert got_size == want_size, f"cardinality {got_size} != {want_size}"
    assert got_cost == pytest.approx(want_cost, abs=1e-9), f"total {got_cost} != {want_cost}"


def test_the_reported_hang_case_is_matched_correctly():
    """The exact configuration that made the previous residual-graph matcher spin forever.

    Two detections share a coordinate, which gives the residual graph zero-cost cycles, and
    the old predecessor walk could close into one. Five matches exist and must all be found.
    """
    gt = np.array([[0.0, 5.0], [3.0, 9.0], [9.0, 4.0], [1.0, 5.0], [9.0, 8.0]])
    det = np.array([[8.0, 1.0], [1.0, 6.0], [2.0, 4.0], [1.0, 6.0], [7.0, 8.0]])

    matches = match_by_distance(gt, det, 10.0)
    assert len(matches) == 5
    assert sorted(j for _i, j, _d in matches) == [0, 1, 2, 3, 4]
    assert sum(d for _i, _j, d in matches) == pytest.approx(11.5962560604, abs=1e-8)
    assert_agrees_with_oracle(gt, det, 10.0)


def test_the_hang_case_terminates_in_a_fresh_process():
    """A regression of this class hangs rather than fails, which would wedge the whole
    suite. A subprocess with a timeout turns the hang back into an ordinary test failure."""
    source = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {REPO_ROOT!r})
        import numpy as np
        from centerpoint.metrics import assign_optimal, centre_criterion
        gt = np.array([[0.,5.],[3.,9.],[9.,4.],[1.,5.],[9.,8.]])
        det = np.array([[8.,1.],[1.,6.],[2.,4.],[1.,6.],[7.,8.]])
        assert len(assign_optimal(*centre_criterion(gt, det, 10.0))) == 5
    """)
    try:
        done = subprocess.run([sys.executable, '-c', source], timeout=60,
                              capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        pytest.fail("assign_optimal did not terminate within 60 s")
    assert done.returncode == 0, done.stderr


def test_duplicate_detection_coordinates_are_still_two_detections():
    """The trap in the hang case, isolated: identical coordinates are not one detection."""
    matches = match_by_distance([(0.0, 0.0), (0.5, 0.0)], [(0.2, 0.0), (0.2, 0.0)], 2.0)
    assert len(matches) == 2


def test_every_point_identical_still_matches_every_pair():
    """Every distance ties at zero. It must still match, not stall on the tie."""
    matches = match_by_distance([(3.0, 3.0)] * 4, [(3.0, 3.0)] * 4, 1.0)
    assert len(matches) == 4
    assert sorted(j for _i, j, _d in matches) == [0, 1, 2, 3]


@pytest.mark.parametrize('size', [1, 2, 5, 12, 30])
def test_massively_tied_costs_terminate_and_match_everything(size):
    """Where a predecessor cycle would actually show up, at sizes the oracle cannot reach.

    Every point coincides, so every cost ties and the residual graph is nothing but
    zero-cost cycles. The augmenting-tree invariant is what keeps this terminating; the
    assertion inside _hungarian fires if a future edit breaks it.
    """
    points = [(2.0, 2.0)] * size
    matches = match_by_distance(points, points, 1.0)
    assert len(matches) == size
    assert sorted(j for _i, j, _d in matches) == list(range(size))
    assert all(d == 0.0 for _i, _j, d in matches)


def test_two_distinct_cost_levels_at_scale_still_assign_optimally():
    """Ties everywhere plus a single cheaper column, so the optimum is unique in cost but
    wildly ambiguous in assignment. Cardinality and total are checked without an oracle."""
    gt = [(0.0, float(i)) for i in range(20)]
    det = [(0.0, float(i) + 0.5) for i in range(20)]
    matches = match_by_distance(gt, det, 0.5)
    assert len(matches) == 20
    assert sum(d for _i, _j, d in matches) == pytest.approx(10.0)


def test_a_non_finite_detection_centre_is_unmatchable_rather_than_an_error():
    """The offset head is linear and nothing guarantees a finite centre, so nan must not
    take the scorer down or poison a distance comparison."""
    matches = match_by_distance([(10.0, 10.0), (20.0, 20.0)],
                                [(float('nan'), 10.0), (20.0, 20.0)], 2.0)
    assert [(i, j) for i, j, _d in matches] == [(1, 1)]


@pytest.mark.parametrize('seed', range(40))
def test_random_cases_agree_with_the_exhaustive_oracle(seed):
    """Deliberately degenerate random geometry: coordinates come from a coarse integer grid
    so duplicates and ties are common, sizes differ, and the tolerance is often small
    enough to leave the feasible graph sparse or empty."""
    rng = np.random.default_rng(seed)
    for _case in range(25):
        gt = rng.integers(0, 5, size=(int(rng.integers(0, 6)), 2)).astype(float)
        det = rng.integers(0, 5, size=(int(rng.integers(0, 6)), 2)).astype(float)
        assert_agrees_with_oracle(gt, det, float(rng.choice([0.0, 1.0, 2.0, 3.0, 10.0])))


def test_those_random_cases_actually_defeat_greedy():
    """Guards the guard. If greedy never lost on this distribution, the oracle comparison
    above would not be exercising the part of the problem that makes greedy wrong."""
    rng = np.random.default_rng(0)
    greedy_losses = 0
    for _case in range(400):
        gt = rng.integers(0, 5, size=(int(rng.integers(2, 6)), 2)).astype(float)
        det = rng.integers(0, 5, size=(int(rng.integers(2, 6)), 2)).astype(float)
        if len(greedy_match(gt, det, 2.0)) < len(match_by_distance(gt, det, 2.0)):
            greedy_losses += 1
    assert greedy_losses > 0, "greedy never lost, so these cases prove less than claimed"


# --- preflight ---------------------------------------------------------------------------------

def ideal_selection(count=3):
    """Holes the codec can represent exactly: far apart and comfortably inside the image."""
    return [holes((100.0 + 40 * i, 100.0, 6.0), (300.0, 300.0 + 40 * i, 5.0))
            for i in range(count)]


def test_preflight_passes_a_representable_selection():
    assert preflight(ideal_selection(), ['a', 'b', 'c'], count=3) == []


def test_preflight_rejects_two_holes_sharing_one_stride_cell():
    """The hard limit: one cell holds one peak, so the target itself cannot carry both and
    the count criterion is unreachable however well the model trains."""
    crowded = holes((100.0, 100.0, 5.0), (101.0, 100.0, 5.0))
    problems = preflight([crowded], ['crowded'], count=1)
    assert any('share a stride-4 cell' in problem for problem in problems)


def test_preflight_rejects_an_image_with_no_holes():
    problems = preflight([holes()], ['empty'], count=1)
    assert any('contain no holes' in problem for problem in problems)


def test_preflight_requires_a_multi_hole_image():
    """One hole per image would never test separating neighbouring peaks."""
    singles = [holes((100.0 + 40 * i, 100.0, 5.0)) for i in range(3)]
    problems = preflight(singles, ['a', 'b', 'c'], count=3)
    assert any('more than one hole' in problem for problem in problems)


def test_preflight_rejects_the_wrong_number_of_images():
    problems = preflight(ideal_selection(2), ['a', 'b'], count=8)
    assert any('expected 8' in problem for problem in problems)


def test_preflight_rejects_a_repeated_image_id():
    problems = preflight(ideal_selection(2), ['same', 'same'], count=2)
    assert any('repeats an image id' in problem for problem in problems)


# --- argument validation --------------------------------------------------------------------

@pytest.mark.parametrize('argv', [['--steps', '0'], ['--batch', '0'], ['--batch', '9'],
                                  ['--lr', '0'], ['--lr', '-1e-3']])
def test_invalid_settings_are_rejected_before_anything_runs(argv):
    """--steps 0 would otherwise report an untrained model as a gate result."""
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_defaults_are_the_documented_recipe():
    args = parse_args([])
    assert (args.steps, args.batch, args.lr, args.seed) == (400, 8, 1e-3, 0)
