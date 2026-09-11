"""The evaluation contract, on synthetic geometry only. No model, no dataset, no device.

Two things these tests exist to pin down. First, that the AP integration is the convention
it claims to be — checked against hand-calculated examples and an independent, deliberately
slow reference, not against the implementation itself. Second, that the two assignment
rules stay distinguishable: they are allowed to disagree, and a change that quietly made
them agree would be a change to the contract.
"""

import warnings

import numpy as np
import pytest

from centerpoint.metrics import (CANDIDATE_FLOOR, CENTRE_TOLERANCES_PX,
                                 PRIMARY_CENTRE_TOLERANCE_PX,
                                 RECALL_POINTS, assign_by_confidence, assign_optimal,
                                 average_precision, centre_criterion,
                                 validate_generation_threshold, circle_iou,
                                 circle_iou_criterion, counting_scores, evaluate_split,
                                 geometry_errors, label_detections, prepare_detections,
                                 relative_criterion)


def holes(*rows):
    return np.array(rows, dtype=np.float64).reshape(-1, 3)


def detections(*rows):
    """(x, y, r, score) as decode returns them."""
    return np.array(rows, dtype=np.float64).reshape(-1, 4)


def reference_average_precision(labels, n_ground_truth, points=RECALL_POINTS):
    """The 101-point envelope written the slow, obvious way, as an independent check.

    Plain python loops over every recall query and every later point. Shares no array
    trickery with the implementation, so agreement is evidence rather than a restatement.
    """
    if n_ground_truth == 0:
        return float('nan')
    if not labels:
        return 0.0

    ordered = sorted(range(len(labels)), key=lambda k: (-labels[k][0], k))
    precisions, recalls = [], []
    true_positives = false_positives = 0
    for k in ordered:
        if labels[k][1]:
            true_positives += 1
        else:
            false_positives += 1
        precisions.append(true_positives / (true_positives + false_positives))
        recalls.append(true_positives / n_ground_truth)

    total = 0.0
    for step in range(points):
        query = step / (points - 1)
        reachable = [precisions[k] for k in range(len(recalls)) if recalls[k] >= query]
        total += max(reachable) if reachable else 0.0
    return total / points


# --- circle IoU --------------------------------------------------------------------------

def test_identical_circles_overlap_completely():
    assert circle_iou(holes((5.0, 5.0, 3.0)), detections((5.0, 5.0, 3.0, 1.0)))[0, 0] \
        == pytest.approx(1.0)


def test_disjoint_circles_do_not_overlap():
    assert circle_iou(holes((0.0, 0.0, 1.0)), detections((5.0, 0.0, 1.0, 1.0)))[0, 0] == 0.0


def test_touching_circles_do_not_overlap():
    """Exactly tangent: the intersection is a point, so the area is zero, not a sliver."""
    assert circle_iou(holes((0.0, 0.0, 2.0)), detections((5.0, 0.0, 3.0, 1.0)))[0, 0] == 0.0


def test_one_circle_wholly_inside_another_is_the_area_ratio():
    """Concentric, radii 2 and 4: the small disc is the whole intersection."""
    assert circle_iou(holes((0.0, 0.0, 4.0)), detections((0.0, 0.0, 2.0, 1.0)))[0, 0] \
        == pytest.approx(4.0 / 16.0)


def test_internally_tangent_circles_are_still_contained():
    """d == |r1 - r2| is the boundary between the contained and lens regimes."""
    assert circle_iou(holes((0.0, 0.0, 4.0)), detections((2.0, 0.0, 2.0, 1.0)))[0, 0] \
        == pytest.approx(4.0 / 16.0)


def test_the_lens_case_matches_the_hand_calculation():
    """Two unit circles one radius apart. Intersection 2r^2*acos(d/2r) - (d/2)*sqrt(4r^2-d^2)
    = 2*pi/3 - sqrt(3)/2, union 2*pi minus that."""
    intersection = 2 * np.pi / 3 - np.sqrt(3) / 2
    expected = intersection / (2 * np.pi - intersection)
    assert circle_iou(holes((0.0, 0.0, 1.0)), detections((1.0, 0.0, 1.0, 1.0)))[0, 0] \
        == pytest.approx(expected)
    assert expected == pytest.approx(0.24300, abs=1e-5)


def test_circle_iou_agrees_with_monte_carlo_sampling():
    """An independent estimate of the same quantity, to catch a wrong closed form."""
    rng = np.random.default_rng(0)
    for _case in range(12):
        gt = holes((0.0, 0.0, float(rng.uniform(1, 4))))
        det = detections((float(rng.uniform(-4, 4)), float(rng.uniform(-4, 4)),
                          float(rng.uniform(1, 4)), 1.0))

        # Wide enough to contain both discs whole: a centre up to 4 from the origin with a
        # radius up to 4 reaches 8, and a box that clipped either one would undercount the
        # union and quietly inflate the sampled IoU.
        span = 20.0
        points = rng.uniform(-span / 2, span / 2, size=(400_000, 2))
        in_gt = np.linalg.norm(points - gt[0, :2], axis=1) <= gt[0, 2]
        in_det = np.linalg.norm(points - det[0, :2], axis=1) <= det[0, 2]
        sampled = (in_gt & in_det).sum() / max((in_gt | in_det).sum(), 1)

        assert circle_iou(gt, det)[0, 0] == pytest.approx(sampled, abs=0.01)


def test_a_non_positive_predicted_radius_gives_no_overlap_rather_than_an_error():
    """The radius head is linear, so a negative prediction can reach the metric."""
    assert circle_iou(holes((0.0, 0.0, 2.0)), detections((0.0, 0.0, -1.0, 1.0)))[0, 0] == 0.0
    assert circle_iou(holes((0.0, 0.0, 2.0)), detections((0.0, 0.0, 0.0, 1.0)))[0, 0] == 0.0


# --- the criteria ------------------------------------------------------------------------

def test_centre_criterion_is_inclusive_at_the_tolerance():
    feasible, cost = centre_criterion(holes((0.0, 0.0, 1.0)), detections((2.0, 0.0, 1.0, 1.0)), 2.0)
    assert feasible[0, 0] and cost[0, 0] == pytest.approx(2.0)


def test_relative_criterion_scales_with_the_ground_truth_radius():
    """The same 2 px error is inside half a radius for a 5 px hole and outside it for 2 px."""
    big, small = holes((0.0, 0.0, 5.0)), holes((0.0, 0.0, 2.0))
    detection = detections((2.0, 0.0, 1.0, 1.0))
    assert relative_criterion(big, detection, 0.5)[0][0, 0]
    assert not relative_criterion(small, detection, 0.5)[0][0, 0]


def test_the_circle_iou_threshold_is_inclusive():
    """AP_circle_50 means IoU of at least 0.50, matching the inclusive distance tolerance.

    Two equal circles whose overlap is exactly the threshold: the separation is solved for
    rather than guessed, so the pair sits on the boundary and not near it.
    """
    radius, target = 1.0, 0.5

    def overlap_at(distance):
        return circle_iou(holes((0.0, 0.0, radius)),
                          detections((distance, 0.0, radius, 1.0)))[0, 0]

    low, high = 0.0, 2 * radius
    for _step in range(200):                      # bisection; overlap falls with distance
        middle = (low + high) / 2
        if overlap_at(middle) > target:
            low = middle
        else:
            high = middle

    assert overlap_at(low) == pytest.approx(target, abs=1e-12)
    assert circle_iou_criterion(holes((0.0, 0.0, radius)),
                                detections((low, 0.0, radius, 1.0)), target)[0][0, 0], \
        "a pair exactly at the threshold must be feasible"


def test_circle_iou_criterion_costs_one_minus_overlap():
    """Minimising cost has to mean maximising overlap, or the assignment optimises backwards."""
    feasible, cost = circle_iou_criterion(holes((0.0, 0.0, 3.0)),
                                          detections((0.0, 0.0, 3.0, 1.0)), 0.5)
    assert feasible[0, 0] and cost[0, 0] == pytest.approx(0.0)


def test_relative_criterion_costs_radii_not_pixels():
    """Cost must be normalised like the feasibility test, not left in absolute pixels."""
    _feasible, cost = relative_criterion(holes((0.0, 0.0, 4.0)),
                                         detections((2.0, 0.0, 4.0, 1.0)), 1.0)
    assert cost[0, 0] == pytest.approx(0.5)


def test_relative_assignment_prefers_the_smaller_relative_error():
    """Where absolute and relative cost disagree, and the reason both halves are normalised.

    A 2 px hole missed by 1.0 px is half a radius out; a 10 px hole missed by 1.5 px is
    0.15 of one. Absolute distance would pair the detections the other way round, judging
    which pairs are allowed by scale but which to prefer by raw pixels.
    """
    gt = holes((0.0, 0.0, 2.0), (0.0, 100.0, 10.0))
    det = detections((1.0, 0.0, 2.0, 0.9), (1.5, 100.0, 10.0, 0.9))

    feasible, cost = relative_criterion(gt, det, 1.0)
    assert cost[0, 0] == pytest.approx(0.5)
    assert cost[1, 1] == pytest.approx(0.15)
    assert assign_optimal(feasible, cost) == [(0, 0), (1, 1)]


def test_a_non_positive_ground_truth_radius_is_infeasible_not_a_division_error():
    """Our parser rejects these, so this guards a future caller rather than today's data.

    The division still happens and still yields nan, which is fine: `feasible` is the sole
    authority on which cells mean anything, and every assignment rule masks by it before
    reading a cost. What must not happen is a raise or a warning escaping.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        feasible, cost = relative_criterion(holes((0.0, 0.0, 0.0)),
                                            detections((0.0, 0.0, 1.0, 0.9)), 1.0)

    assert not feasible.any()
    assert cost.shape == (1, 1)
    assert assign_optimal(feasible, cost) == [], "an infeasible cost must never be matched"


def test_a_non_finite_centre_is_infeasible_under_every_criterion():
    gt = holes((10.0, 10.0, 5.0))
    broken = detections((float('nan'), 10.0, 5.0, 0.9))
    assert not centre_criterion(gt, broken, 2.0)[0].any()
    assert not relative_criterion(gt, broken, 1.0)[0].any()
    assert not circle_iou_criterion(gt, broken, 0.5)[0].any()


# --- the two assignment rules -------------------------------------------------------------

def confident_case():
    """Where the two rules must disagree, on a line with a 1 px tolerance.

    Hole A at 0.0 can only reach detection P at 0.6. Hole B at 1.05 can reach P (0.45 away)
    or Q at 2.05 (exactly 1.0 away). P is the more confident detection and takes its
    cheapest hole, B, stranding A and making Q a false positive. The optimal rule pairs
    both. Neither is wrong; they answer different questions.
    """
    return (holes((0.0, 0.0, 1.0), (1.05, 0.0, 1.0)),
            detections((0.6, 0.0, 1.0, 0.9), (2.05, 0.0, 1.0, 0.8)))


def test_optimal_assignment_finds_both_pairs():
    gt, det = confident_case()
    feasible, cost = centre_criterion(gt, det, 1.0)
    assert assign_optimal(feasible, cost) == [(0, 0), (1, 1)]


def test_confidence_assignment_finds_only_one():
    gt, det = confident_case()
    feasible, cost = centre_criterion(gt, det, 1.0)
    pairs, order = assign_by_confidence(feasible, cost, det[:, 3])
    assert pairs == [(1, 0)]
    assert list(order) == [0, 1], "detections must be considered in descending score order"


def test_confidence_assignment_breaks_score_ties_stably():
    """Equal scores keep decode's own order, so a run is reproducible."""
    gt = holes((0.0, 0.0, 1.0))
    det = detections((0.1, 0.0, 1.0, 0.5), (0.2, 0.0, 1.0, 0.5))
    pairs, order = assign_by_confidence(*centre_criterion(gt, det, 2.0), det[:, 3])
    assert list(order) == [0, 1] and pairs == [(0, 0)]


def test_a_second_detection_on_a_claimed_hole_is_a_false_positive():
    """What makes decode's local-maximum suppression part of the measurement."""
    labels = label_detections(holes((10.0, 10.0, 5.0)),
                              detections((10.0, 10.0, 5.0, 0.9), (10.0, 10.0, 5.0, 0.8)),
                              lambda g, d: centre_criterion(g, d, 2.0))
    assert labels == [(0.9, True), (0.8, False)]


# --- average precision --------------------------------------------------------------------

def test_perfect_detection_scores_one():
    labels = [(0.9, True), (0.8, True)]
    assert average_precision(labels, 2) == pytest.approx(1.0)


def test_only_false_positives_score_zero():
    assert average_precision([(0.9, False), (0.8, False)], 2) == pytest.approx(0.0)


def test_no_detections_at_all_scores_zero_when_holes_exist():
    assert average_precision([], 5) == pytest.approx(0.0)


def test_no_ground_truth_anywhere_is_undefined_not_perfect():
    """A split with no holes must not read as a perfect score just because nothing was missed."""
    assert np.isnan(average_precision([(0.9, False)], 0))
    assert np.isnan(average_precision([], 0))


def test_the_hand_calculated_interleaved_case():
    """Two holes; detections TP, FP, TP by descending score.

    Cumulative precision 1.0, 0.5, 2/3 at recalls 0.5, 0.5, 1.0. The envelope is 1.0 up to
    recall 0.5 and 2/3 beyond it, so 51 of the 101 queries read 1.0 and 50 read 2/3.
    """
    labels = [(0.9, True), (0.8, False), (0.7, True)]
    expected = (51 * 1.0 + 50 * (2.0 / 3.0)) / 101
    assert average_precision(labels, 2) == pytest.approx(expected)
    assert expected == pytest.approx(0.834984, abs=1e-6)


def test_the_hand_calculated_half_recall_case():
    """One of two holes found, with no false positives: the envelope is 1.0 up to recall
    0.5 and undefined above it, which contributes zero. 51 of 101 queries."""
    assert average_precision([(0.9, True)], 2) == pytest.approx(51 / 101)


def test_a_missed_hole_costs_more_than_a_false_positive_at_the_tail():
    """Sanity on the shape of the metric, not just its value."""
    missed = average_precision([(0.9, True)], 2)
    spurious = average_precision([(0.9, True), (0.1, False)], 1)
    assert missed < spurious


@pytest.mark.parametrize('seed', range(30))
def test_average_precision_agrees_with_the_slow_reference(seed):
    """Random label sequences, including duplicate scores and all-true/all-false runs.

    True positives are capped at n_ground_truth because one-to-one matching cannot produce
    more of them, and average_precision now rejects label sets that claim otherwise. An
    uncapped generator was fabricating sequences no matcher could ever emit.
    """
    rng = np.random.default_rng(seed)
    count = int(rng.integers(0, 25))
    scores = [float(rng.choice([0.95, 0.8, 0.8, 0.5, 0.2, 0.05])) for _ in range(count)]
    flags = [bool(rng.integers(0, 2)) for _ in range(count)]
    n_ground_truth = int(rng.integers(1, 12))

    budget = n_ground_truth
    capped = []
    for flag in flags:
        keep = flag and budget > 0
        budget -= int(keep)
        capped.append(keep)
    labels = list(zip(scores, capped))

    assert average_precision(labels, n_ground_truth) == \
        pytest.approx(reference_average_precision(labels, n_ground_truth), abs=1e-12)


def test_more_true_positives_than_holes_is_rejected():
    """One-to-one matching cannot emit this, so it means the caller pooled labels against
    the wrong ground-truth count — a silent AP inflation if it were allowed through."""
    with pytest.raises(ValueError, match='More true positives'):
        average_precision([(0.9, True), (0.8, True)], 1)


def test_recall_beyond_the_ground_truth_count_cannot_occur():
    """One-to-one matching bounds true positives by the hole count, so recall stops at 1.0
    and AP stays in [0, 1] even when the model floods the image."""
    labels = [(0.9, True)] + [(0.5, False)] * 50
    assert 0.0 <= average_precision(labels, 1) <= 1.0


# --- candidate policy ----------------------------------------------------------------------

def test_detections_below_the_candidate_floor_are_dropped_and_counted():
    kept, notes = prepare_detections(
        detections((1.0, 1.0, 2.0, 0.9), (2.0, 2.0, 2.0, 0.01)), candidate_floor=0.05)
    assert len(kept) == 1 and notes['n_below_floor'] == 1


def test_the_floor_is_inclusive():
    kept, _notes = prepare_detections(detections((1.0, 1.0, 2.0, 0.05)), candidate_floor=0.05)
    assert len(kept) == 1


def test_a_non_finite_score_is_dropped_and_counted():
    kept, notes = prepare_detections(detections((1.0, 1.0, 2.0, float('nan')),
                                                (2.0, 2.0, 2.0, 0.9)))
    assert len(kept) == 1 and notes['n_non_finite_score'] == 1


def test_the_cap_keeps_the_most_confident_and_records_that_it_bit():
    many = detections(*[(float(i), 0.0, 1.0, i / 100.0) for i in range(10)])
    kept, notes = prepare_detections(many, candidate_floor=0.0, cap=4)
    assert len(kept) == 4 and notes['n_capped'] == 6
    assert sorted(kept[:, 3]) == pytest.approx([0.06, 0.07, 0.08, 0.09])


def test_an_empty_detection_array_survives_the_policy():
    kept, notes = prepare_detections(detections())
    assert len(kept) == 0 and notes['n_raw'] == 0


# --- the integration contract ------------------------------------------------------------

def test_candidates_generated_above_the_floor_are_refused():
    """The one upstream mistake the data cannot reveal: detections cut at the operating
    threshold look exactly like a model that never predicted anything weaker."""
    with pytest.raises(ValueError, match='must be <='):
        validate_generation_threshold(0.3, candidate_floor=0.05)


def test_an_unstated_generation_threshold_is_refused():
    with pytest.raises(ValueError, match='must explicitly state|must be finite'):
        validate_generation_threshold(None)
    with pytest.raises(ValueError, match='must explicitly state|must be finite'):
        validate_generation_threshold(float('nan'))


def test_generating_at_or_below_the_floor_is_accepted():
    validate_generation_threshold(0.05, candidate_floor=0.05)
    validate_generation_threshold(0.0, candidate_floor=0.05)


def test_evaluate_split_refuses_pre_thresholded_predictions():
    samples = [(holes((10.0, 10.0, 3.0)), detections((10.0, 10.0, 3.0, 0.9)))]
    with pytest.raises(ValueError, match='must be <='):
        evaluate_split(samples, generation_threshold=0.3)


def test_evaluate_split_records_the_generation_threshold():
    results = evaluate_split([(holes((10.0, 10.0, 3.0)), detections((10.0, 10.0, 3.0, 0.9)))], generation_threshold=CANDIDATE_FLOOR)
    contract = results['contract']
    assert contract['generation_threshold'] <= contract['candidate_floor']


# --- counting and geometry ------------------------------------------------------------------

def primary(gt, det):
    return centre_criterion(gt, det, PRIMARY_CENTRE_TOLERANCE_PX)


def test_an_empty_image_with_no_detections_is_not_a_failure():
    """The gate's positive-only rule must not travel: this is a correct result."""
    scores = counting_scores([(holes(), detections())], primary, operating_threshold=0.3)
    assert scores['n_ground_truth'] == 0 and scores['n_predicted'] == 0
    assert scores['n_matched'] == 0
    assert np.isnan(scores['precision']) and np.isnan(scores['recall'])


def test_an_empty_image_with_a_detection_costs_precision_only():
    samples = [(holes(), detections((5.0, 5.0, 2.0, 0.9))),
               (holes((10.0, 10.0, 3.0)), detections((10.0, 10.0, 3.0, 0.9)))]
    scores = counting_scores(samples, primary, operating_threshold=0.3)
    assert scores['recall'] == pytest.approx(1.0)
    assert scores['precision'] == pytest.approx(0.5)


def test_detections_below_the_operating_threshold_do_not_count():
    samples = [(holes((10.0, 10.0, 3.0)), detections((10.0, 10.0, 3.0, 0.2)))]
    scores = counting_scores(samples, primary, operating_threshold=0.3)
    assert scores['n_predicted'] == 0 and scores['recall'] == pytest.approx(0.0)


def test_counting_reports_both_kinds_of_leftover():
    samples = [(holes((10.0, 10.0, 3.0), (400.0, 400.0, 3.0)),
                detections((10.0, 10.0, 3.0, 0.9), (200.0, 200.0, 3.0, 0.9)))]
    scores = counting_scores(samples, primary, operating_threshold=0.3)
    assert scores['n_unmatched_ground_truth'] == 1
    assert scores['n_unmatched_detections'] == 1


def test_geometry_reports_spread_and_signed_bias():
    """Radii predicted 1 px large every time, so the bias is +1 and the absolute error 1."""
    samples = [(holes((10.0, 10.0, 3.0), (100.0, 100.0, 5.0)),
                detections((10.5, 10.0, 4.0, 0.9), (100.0, 100.0, 6.0, 0.9)))]
    stats = geometry_errors(samples, primary, operating_threshold=0.3)
    assert stats['n_matched'] == 2
    assert stats['centre_error_median'] == pytest.approx(0.25)
    assert stats['centre_error_max'] == pytest.approx(0.5)
    assert stats['radius_bias_mean'] == pytest.approx(1.0)
    assert stats['radius_error_median'] == pytest.approx(1.0)


def test_geometry_on_no_matches_is_nan_not_zero():
    """Zero error would read as a perfect model that simply found nothing."""
    stats = geometry_errors([(holes((10.0, 10.0, 3.0)), detections())], primary)
    assert stats['n_matched'] == 0
    assert np.isnan(stats['centre_error_median']) and np.isnan(stats['radius_bias_mean'])


# --- the reported bundle ---------------------------------------------------------------------

def test_the_metric_names_state_criterion_threshold_and_rule():
    results = evaluate_split([(holes((10.0, 10.0, 3.0)), detections((10.0, 10.0, 3.0, 0.9)))], generation_threshold=CANDIDATE_FLOOR)

    for tolerance in CENTRE_TOLERANCES_PX:
        assert f'AP_center_{tolerance:g}px' in results
    assert 'AP_center_mean_over_1_2_3_5px' in results
    assert 'AP_circle_50' in results and 'AP_circle_25' in results
    assert 'AP_center_relative_0.5r' in results
    assert 'F1_center_2px_optimal' in results
    assert not any(key.lower().startswith('map') for key in results), \
        "nothing here is box mAP and nothing may be named as if it were"


def test_the_contract_metadata_records_the_policy():
    results = evaluate_split([(holes((10.0, 10.0, 3.0)), detections((10.0, 10.0, 3.0, 0.9)))], generation_threshold=CANDIDATE_FLOOR)
    contract = results['contract']
    assert contract['ap_assignment'] == 'confidence'
    assert contract['counting_assignment'] == 'optimal'
    assert contract['primary_metric'] == 'AP_center_2px'
    assert contract['candidate_floor'] < contract['operating_threshold']
    assert contract['recall_points'] == 101
    assert contract['detection_cap_applied'] is False


def test_a_perfect_split_scores_one_everywhere():
    samples = [(holes((10.0, 10.0, 3.0), (100.0, 100.0, 4.0)),
                detections((10.0, 10.0, 3.0, 0.9), (100.0, 100.0, 4.0, 0.8)))]
    results = evaluate_split(samples, generation_threshold=CANDIDATE_FLOOR)
    assert results['AP_center_2px'] == pytest.approx(1.0)
    assert results['AP_circle_50'] == pytest.approx(1.0)
    assert results['F1_center_2px_optimal'] == pytest.approx(1.0)
    assert results['geometry']['centre_error_max'] == pytest.approx(0.0)


def test_a_split_of_empty_images_reports_undefined_rather_than_perfect():
    results = evaluate_split([(holes(), detections()), (holes(), detections())], generation_threshold=CANDIDATE_FLOOR)
    assert np.isnan(results['AP_center_2px'])
    assert results['contract']['n_ground_truth'] == 0
    assert results['contract']['n_images_without_holes'] == 2


def test_the_candidate_floor_admits_detections_the_operating_threshold_would_hide():
    """The reason AP is generated from a floor: a 0.1-score true positive is part of the
    curve even though precision and recall at 0.3 never see it."""
    samples = [(holes((10.0, 10.0, 3.0), (100.0, 100.0, 3.0)),
                detections((10.0, 10.0, 3.0, 0.9), (100.0, 100.0, 3.0, 0.1)))]
    results = evaluate_split(samples, generation_threshold=CANDIDATE_FLOOR)
    assert results['AP_center_2px'] == pytest.approx(1.0)
    assert results['recall_center_2px_optimal'] == pytest.approx(0.5)
    assert results['candidates']['n_below_floor'] == 0


def test_a_stricter_tolerance_never_scores_higher():
    """Monotonicity across the sweep, which a sign error in the criterion would break."""
    rng = np.random.default_rng(3)
    samples = []
    for _image in range(6):
        truth = holes(*[(float(x), float(y), 4.0)
                        for x, y in rng.integers(20, 600, size=(5, 2))])
        noisy = detections(*[(row[0] + rng.normal(0, 1.5), row[1] + rng.normal(0, 1.5),
                              4.0, float(rng.uniform(0.3, 1.0))) for row in truth])
        samples.append((truth, noisy))

    results = evaluate_split(samples, generation_threshold=CANDIDATE_FLOOR)
    scores = [results[f'AP_center_{t:g}px'] for t in CENTRE_TOLERANCES_PX]
    assert scores == sorted(scores), f"AP fell as the tolerance loosened: {scores}"
