"""The evaluation contract. One matching implementation, several named metrics.

Nothing here imports a model or a dataset: it takes arrays in and returns numbers, so the
tests run on synthetic geometry and the same code scores CenterPointNet and the YOLO26
baseline without either knowing about the other.

THE CONTRACT, stated once because several incompatible conventions share these names.

A CRITERION decides which (hole, detection) pairs may be paired at all, and how good a
pairing is. Three are defined, and they are separate metrics rather than variants of one:

    centre distance     paired if the centres are within a fixed pixel tolerance.
                        Measures localisation in sensor units, which is closest to what
                        shot scoring actually needs. Says nothing about radius.
    radius-relative     paired if centre distance <= alpha * ground-truth radius.
                        Normalises by object scale, and inherits the noise in our derived
                        radii into the pairing decision itself.
    circle IoU          paired if the discs overlap by more than a threshold. The only one
                        that judges centre and radius together, and therefore the only one
                        that cannot say which of the two went wrong.

An ASSIGNMENT RULE turns a criterion into pairs. Two are defined, and every metric name
below says which one produced it, because they do not agree:

    optimal             maximum cardinality first, then minimum total cost. The honest
                        answer to "how many holes did it find", used for P/R/F1 and for
                        every geometry statistic.
    confidence          detections in descending score order, each claiming the best
                        unclaimed hole. Used for AP, because AP is defined by a ranking
                        sweep and the labels have to be stable as the sweep proceeds.

Both rules share the criterion, the feasibility test and the cost matrix. Only the
selection differs.

THE INTEGRATION CONTRACT, which is the easiest thing here to get wrong. AP is the area
under a precision-recall curve, and that curve only exists if the low-confidence tail of
the predictions is present. Nothing in this module can recover a detection that was already
thrown away upstream, so the caller must generate candidates down to CANDIDATE_FLOOR:

    CenterPointNet   detections_from(..., threshold=CANDIDATE_FLOOR) — NOT the 0.3 default
                     that codec/decode.py and the overfit gate use.
    YOLO26           its own confidence floor set to CANDIDATE_FLOOR, same policy.

Generating at the operating threshold instead does not raise anything; it silently reports
a smaller AP, because the whole tail of the curve is simply missing. evaluate_split takes
`generation_threshold` and refuses to score predictions that were filtered above the floor,
so the mistake has to be declared before it can be made.

WHAT IS NOT HERE. The overfit gate's rule that an image with no matches is a failure does
not apply: an image with no holes and no detections is a correct result and contributes
nothing to either precision or recall. See centerpoint/overfit.py for that gate.
"""

import numpy as np
from functools import partial


# Detections are generated down to this score and no lower. Deliberately far below any
# operating threshold: AP is an area under a curve that only exists if the low-confidence
# tail is present, and starting the sweep at the operating threshold silently truncates it.
CANDIDATE_FLOOR = 0.05

# Where precision and recall are read off for the single-number report. Inherited from
# codec/decode.py and PROVISIONAL: it was chosen before any model existed. Pick a better
# one from validation results, never from test.
OPERATING_THRESHOLD = 0.3

PRIMARY_CENTRE_TOLERANCE_PX = 2.0
CENTRE_TOLERANCES_PX = (1.0, 2.0, 3.0, 5.0)
RADIUS_RELATIVE_ALPHAS = (0.5, 1.0)
CIRCLE_IOU_THRESHOLDS = (0.5, 0.25)

# COCO's convention: sample the precision envelope at 101 evenly spaced recall values.
RECALL_POINTS = 101

# Per-image ceiling on candidates, so one pathological heatmap cannot dominate the sweep.
# Chosen well above the densest image in either dataset (~200 holes); if it ever binds, the
# result records that it did rather than truncating in silence.
DETECTION_CAP = 2000





# --- pairwise geometry -----------------------------------------------------------------------

def centre_distances(holes, detections):
    """(n, m) matrix of centre-to-centre distances in pixels."""
    holes, detections = np.asarray(holes, float), np.asarray(detections, float)
    if not len(holes) or not len(detections):
        return np.zeros((len(holes), len(detections)))
    
    hole_centres , detection_centres = holes[:, :2] , detections[:, :2]
    differences = hole_centres[:, None, :] - detection_centres[None, :, :]
    distances = np.linalg.norm(differences, axis=-1)
    return distances




def circle_iou(holes, detections):
    """(n, m) matrix of intersection-over-union between the ground-truth and predicted discs.

    The closed form for two circles, as CircleNet uses. Three regimes: disjoint gives 0,
    one disc wholly inside the other gives the area ratio, and the general case is the sum
    of two circular segments. A non-positive radius on either side gives 0 rather than a
    division error — the radius head is linear and may predict one.
    """
    holes, detections = np.asarray(holes, float), np.asarray(detections, float)
    if not len(holes) or not len(detections):
        return np.zeros((len(holes), len(detections)))

    distances = centre_distances(holes, detections)
    r_holes = holes[:, None, 2]
    r_detections = detections[None, :, 2]

    distances_sq = distances ** 2
    r_holes_sq = r_holes ** 2
    r_detections_sq = r_detections ** 2

    with np.errstate(invalid='ignore', divide='ignore'):
        cos_holes = (distances_sq + r_holes_sq - r_detections_sq) / (2 * distances * r_holes)
        cos_detections = (distances_sq + r_detections_sq - r_holes_sq) / (2 * distances * r_detections)
        cos_holes = np.clip(cos_holes, -1, 1)
        cos_detections = np.clip(cos_detections, -1, 1)

        triangle_term = (-distances + r_holes + r_detections) * (distances - r_holes + r_detections) * (distances + r_holes - r_detections) * (distances + r_holes + r_detections) # Heron-derived term for triangles
        lens_area = r_holes_sq * np.arccos(cos_holes) + r_detections_sq * np.arccos(cos_detections) - 0.5 * np.sqrt(np.clip(triangle_term, 0, None))

    smaller_r = np.minimum(r_holes, r_detections)
    contained_areas = np.pi * smaller_r ** 2
    is_contained = distances <= np.abs(r_holes - r_detections)
    is_disjoint = distances >= r_holes + r_detections
    intersection_areas = np.where(is_contained, contained_areas, lens_area)
    intersection_areas = np.where(is_disjoint, 0.0, intersection_areas)

    hole_areas = np.pi * r_holes_sq
    detection_areas = np.pi * r_detections_sq
    union_areas = hole_areas + detection_areas - intersection_areas

    usable = np.isfinite(intersection_areas) & (union_areas > 0) & (r_holes > 0) & (r_detections > 0)
    iou = np.divide(intersection_areas, union_areas, out=np.zeros_like(union_areas), where=usable)
    return iou




def centre_criterion(holes, detections, tolerance=PRIMARY_CENTRE_TOLERANCE_PX):
    """Fixed pixel tolerance. Returns (feasible, cost) with cost in pixels, lower better."""
    distances = centre_distances(holes, detections)
    feasible = np.isfinite(distances) & (distances <= tolerance)
    return feasible, distances




def relative_criterion(holes, detections, alpha):
    """Tolerance proportional to each hole's own radius, so scale cancels."""
    holes = np.asarray(holes, float)
    distances = centre_distances(holes, detections)

    if not distances.size:
        return distances.astype(bool), distances

    r_holes = holes[:, None, 2]
    usable = np.isfinite(r_holes) & (r_holes > 0) & np.isfinite(distances)
    feasible = usable & (distances <= alpha * r_holes)

    with np.errstate(invalid='ignore', divide='ignore'):
        relative_distances = np.divide(distances, r_holes)

    return feasible, relative_distances




def circle_iou_criterion(holes, detections, threshold):
    """Pair circles by IoU. Higher overlap gives lower assignment cost."""
    overlap = circle_iou(holes, detections)
    feasible = overlap >= threshold
    cost = 1.0 - overlap
    return feasible, cost





# --- assignment --------------------------------------------------------------------------------

def assign_optimal(feasible, cost):
    """Maximum-cardinality assignment, then minimum cost among those assignments."""
    feasible = np.asarray(feasible, bool)
    cost = np.asarray(cost, float)

    n_holes, n_detections = feasible.shape
    if not feasible.any():
        return []

    matrix_size = max(n_holes, n_detections)

    max_feasible_cost = float(cost[feasible].max())
    penalty_cost = (matrix_size + 1) * (max_feasible_cost + 1.0)

    padded_costs = np.full((matrix_size, matrix_size), penalty_cost)
    np.copyto(padded_costs[:n_holes, :n_detections], cost, where=feasible)

    assigned = _hungarian(padded_costs)

    matches = [
        (int(hole_idx), int(detection_idx))
        for detection_idx, hole_idx in enumerate(assigned)
        if hole_idx < n_holes and detection_idx < n_detections and feasible[hole_idx, detection_idx]
    ]
    return sorted(matches)




def assign_by_confidence(feasible, cost, scores):
    """Match detections in descending confidence order to the best available hole."""
    feasible = np.asarray(feasible, bool)
    cost = np.asarray(cost, float)
    scores = np.asarray(scores, float)

    detection_order = np.argsort(-scores, kind='stable')

    claimed_holes = set()
    matches = []

    for detection_idx in detection_order:
        candidate_holes = [hole_idx for hole_idx in range(feasible.shape[0]) if feasible[hole_idx, detection_idx] and hole_idx not in claimed_holes]

        if not candidate_holes:
            continue

        best_hole = min(candidate_holes, key=lambda hole_idx: (cost[hole_idx, detection_idx], hole_idx))
        claimed_holes.add(best_hole)
        matches.append((best_hole, int(detection_idx)))

    return matches, detection_order




def _hungarian(cost):
    """Minimum-total-cost perfect assignment on a square matrix. Returns column -> row.

    The O(n^3) shortest-augmenting-path form with dual potentials, which is the textbook
    Hungarian/Jonker-Volgenant algorithm.

    WHY IT TERMINATES, precisely. It is not the potentials: they keep reduced costs
    non-negative, which is what makes the per-phase search a valid Dijkstra, but a
    non-negative graph can still hold zero-cost cycles and that alone would not save the
    reconstruction. The guarantee is the augmenting TREE that `used` and `came_from`
    maintain together:

      - a column is marked `used` before it is scanned, and once marked it is never
        rescanned, so each phase visits strictly new columns and ends within n of them;
      - `came_from[j]` is only ever written for a column that is NOT yet used, and is set
        to the column being scanned, which IS already used.

    So every predecessor link points from a later-visited column to an earlier-visited one.
    The chain therefore strictly retreats through visit order and must reach the sentinel
    column 0, whatever the costs — ties, zeros and duplicate columns included. The
    check in the reconstruction loop states exactly that, so a future edit that breaks the
    invariant fails loudly instead of hanging. It raises rather than asserts, because the
    guard has to survive `python -O`.

    Row and column 0 are the algorithm's sentinels, so everything is 1-indexed against a
    cost matrix that is not. Verified against an exhaustive oracle in the tests.
    """
    n = len(cost)
    potential_row = np.zeros(n + 1)
    potential_col = np.zeros(n + 1)
    column_row = np.zeros(n + 1, dtype=int)     # column j -> assigned row, 0 for none
    came_from = np.zeros(n + 1, dtype=int)      # column j -> previous column on the path

    for row in range(1, n + 1):
        column_row[0] = row
        column = 0
        best = np.full(n + 1, np.inf)           # cheapest reduced cost reaching each column
        used = np.zeros(n + 1, dtype=bool)
        visit_order = np.full(n + 1, -1)        # per phase; stale values would misjudge it
        visited = 0

        while column_row[column]:
            used[column] = True
            visit_order[column] = visited
            visited += 1
            current_row = column_row[column]

            free = ~used[1:]
            reduced = cost[current_row - 1] - potential_row[current_row] - potential_col[1:]
            improved = free & (reduced < best[1:])
            best[1:][improved] = reduced[improved]
            came_from[1:][improved] = column

            reachable = np.where(free, best[1:], np.inf)
            next_column = int(np.argmin(reachable)) + 1
            delta = reachable[next_column - 1]

            # Shift the potentials so the chosen column's reduced cost becomes zero. The
            # rows behind `used` columns are distinct by construction — a column holds at
            # most one row — so this indexed update cannot double-count.
            visited_columns = np.flatnonzero(used)
            potential_row[column_row[visited_columns]] += delta
            potential_col[visited_columns] -= delta
            best[~used] -= delta

            column = next_column

        # The free column that ended the search is the last one reached, so it closes the
        # visit order the reconstruction below walks back down.
        visit_order[column] = visited

        while column:
            previous = came_from[column]
            # Raised, not asserted: `python -O` strips assertions, and this guard is what
            # stands between a broken invariant and an infinite loop.
            if not 0 <= visit_order[previous] < visit_order[column]:
                raise RuntimeError(
                    "augmenting chain left the visited tree; this is the invariant that "
                    "makes termination independent of the costs")
            column_row[column] = column_row[previous]
            column = previous

    return column_row[1:] - 1





# --- candidate generation ------------------------------------------------------------------------

def validate_generation_threshold(generation_threshold, candidate_floor=CANDIDATE_FLOOR):
    """Validate that detections were generated low enough for AP evaluation."""
    if generation_threshold is None:
        raise ValueError("generation_threshold must explicitly state the threshold used to generate detections.")

    if not np.isfinite(generation_threshold):
        raise ValueError(f"generation_threshold must be finite, got {generation_threshold}.")

    if not np.isfinite(candidate_floor):
        raise ValueError(f"candidate_floor must be finite, got {candidate_floor}.")

    if generation_threshold > candidate_floor:
        raise ValueError(
            f"generation_threshold ({generation_threshold}) must be <= "
            f"candidate_floor ({candidate_floor}). Regenerate detections at a lower "
            "threshold; otherwise the AP curve is truncated."
        )




def prepare_detections(detections, candidate_floor=CANDIDATE_FLOOR, cap=DETECTION_CAP):
    """Filter one image's detections to the candidate set used for evaluation."""
    detections = np.asarray(detections, float).reshape(-1, 4)
    n_raw = len(detections)

    finite_score = np.isfinite(detections[:, 3])
    n_non_finite_score = int((~finite_score).sum())
    detections = detections[finite_score]

    above_floor = detections[:, 3] >= candidate_floor
    n_below_floor = int((~above_floor).sum())
    detections = detections[above_floor]

    n_capped = max(len(detections) - cap, 0)
    if n_capped:
        top_indices = np.argsort(-detections[:, 3], kind='stable')[:cap]
        detections = detections[np.sort(top_indices)]  # preserve original order

    stats = {
        'n_raw': n_raw,
        'n_non_finite_score': n_non_finite_score,
        'n_below_floor': n_below_floor,
        'n_capped': n_capped,
        'n_kept': len(detections),
    }

    return detections, stats





# --- average precision -----------------------------------------------------------------------------

def label_detections(holes, detections, criterion):
    """Confidence-ordered TP/FP labels for one image. Returns [(score, is_true_positive)]."""
    holes = np.asarray(holes, float).reshape(-1, 3)
    detections = np.asarray(detections, float).reshape(-1, 4)

    if not len(detections):
        return []

    feasible, cost = criterion(holes, detections)
    matches, detection_order = assign_by_confidence(feasible, cost, detections[:, 3])
    is_true_positive = np.zeros(len(detections), dtype=bool)

    if matches:
        matched_detection_indices = np.asarray(matches, dtype=int)[:, 1]
        is_true_positive[matched_detection_indices] = True

    scores = detections[:, 3]
    return list(zip(scores[detection_order].tolist(), is_true_positive[detection_order].tolist()))




def average_precision(labels, n_ground_truth, recall_points=RECALL_POINTS):
    """AP over the whole split from pooled (score, is_tp) labels."""

    if n_ground_truth < 0:
        raise ValueError("n_ground_truth cannot be negative.")

    if recall_points < 2:
        raise ValueError("recall_points must be at least 2.")

    if n_ground_truth == 0: 
        return float('nan')

    if not labels:
        return 0.0

    scores = np.asarray([score for score, _ in labels], dtype=float)
    is_true_positive = np.asarray([is_tp for _, is_tp in labels], dtype=bool)

    detection_order = np.argsort(-scores, kind='stable')
    is_true_positive = is_true_positive[detection_order]

    cumulative_tp = np.cumsum(is_true_positive)
    cumulative_predictions = np.arange(1, len(labels) + 1)

    if cumulative_tp[-1] > n_ground_truth:
        raise ValueError("More true positives than ground-truth objects.")

    precision = cumulative_tp / cumulative_predictions
    recall = cumulative_tp / n_ground_truth
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]

    recall_queries = np.arange(recall_points) / (recall_points - 1)
    query_indices = np.searchsorted(recall, recall_queries, side='left')

    sampled_precision = np.zeros(recall_points, dtype=float)
    reachable = query_indices < len(precision_envelope)
    sampled_precision[reachable] = precision_envelope[query_indices[reachable]]

    return float(sampled_precision.mean())





# --- the reported bundle ---------------------------------------------------------------------------

def counting_scores(samples, criterion, operating_threshold=OPERATING_THRESHOLD):
    """Precision, recall and F1 at one operating threshold, under optimal assignment."""

    n_matched = n_predicted = n_ground_truth = 0

    for holes, detections in samples:
        holes = np.asarray(holes, float).reshape(-1, 3)
        detections = np.asarray(detections, float).reshape(-1, 4)
        detections = detections[detections[:, 3] >= operating_threshold] 

        n_ground_truth += len(holes)
        n_predicted += len(detections)

        if len(holes) and len(detections):
            feasible, cost = criterion(holes, detections)
            n_matched += len(assign_optimal(feasible, cost))

    precision = n_matched / n_predicted if n_predicted else float('nan')
    recall = n_matched / n_ground_truth if n_ground_truth else float('nan')

    f1_denominator = n_predicted + n_ground_truth
    f1 = 2 * n_matched / f1_denominator if f1_denominator else float('nan')

    return {'precision': precision, 'recall': recall, 'f1': f1,
            'n_matched': n_matched, 'n_predicted': n_predicted, 'n_ground_truth': n_ground_truth,
            'n_unmatched_ground_truth': n_ground_truth - n_matched,
            'n_unmatched_detections': n_predicted - n_matched,
            'assignment': 'optimal', 'operating_threshold': operating_threshold}




def geometry_errors(samples, criterion, operating_threshold=OPERATING_THRESHOLD):
    """Centre and radius error over matched pairs, under optimal assignment."""

    centre_errors, radius_errors, signed_radius_errors = [], [], []

    for holes, detections in samples:
        holes = np.asarray(holes, float).reshape(-1, 3)
        detections = np.asarray(detections, float).reshape(-1, 4)
        detections = detections[detections[:, 3] >= operating_threshold]

        if not len(holes) or not len(detections):
            continue

        feasible, cost = criterion(holes, detections)
        distances = centre_distances(holes, detections)
        for hole_idx, detection_idx in assign_optimal(feasible, cost):
            centre_errors.append(float(distances[hole_idx, detection_idx]))
            radius_difference = float(detections[detection_idx, 2] - holes[hole_idx, 2])
            signed_radius_errors.append(radius_difference)
            radius_errors.append(abs(radius_difference))

    def summarize_errors(values, prefix):
        if len(values) == 0:
            return {f'{prefix}_median': float('nan'),
                    f'{prefix}_p95': float('nan'),
                    f'{prefix}_max': float('nan')}
        return {f'{prefix}_median': float(np.median(values)),
                f'{prefix}_p95': float(np.percentile(values, 95)),
                f'{prefix}_max': float(np.max(values))}

    return {'n_matched': len(centre_errors),
            **summarize_errors(centre_errors, 'centre_error'),
            **summarize_errors(radius_errors, 'radius_error'),
            'radius_bias_mean': (float(np.mean(signed_radius_errors))if signed_radius_errors else float('nan')),
            'radius_bias_median': (float(np.median(signed_radius_errors))if signed_radius_errors else float('nan')),
            'assignment': 'optimal', 'operating_threshold': operating_threshold}




def _split_ap(samples, n_ground_truth, criterion):
    labels = []
    for holes, detections in samples:
        labels.extend(label_detections(holes, detections, criterion))
    return average_precision(labels, n_ground_truth)



def evaluate_split(samples, candidate_floor=CANDIDATE_FLOOR, operating_threshold=OPERATING_THRESHOLD, cap=DETECTION_CAP, generation_threshold=None):
    """Every named metric for one split. samples is [(holes, raw_detections)]."""

    validate_generation_threshold(generation_threshold, candidate_floor)

    if not np.isfinite(operating_threshold):
        raise ValueError(f"operating_threshold must be finite, got {operating_threshold}.")
    if operating_threshold < candidate_floor:
        raise ValueError(f"operating_threshold ({operating_threshold}) must be >= candidate_floor ({candidate_floor}).")
    if PRIMARY_CENTRE_TOLERANCE_PX not in CENTRE_TOLERANCES_PX:
        raise ValueError("PRIMARY_CENTRE_TOLERANCE_PX must be included in CENTRE_TOLERANCES_PX.")

    prepared, notes = [], []

    for holes, detections in samples:
        kept, note = prepare_detections(detections, candidate_floor, cap)
        holes = np.asarray(holes, float).reshape(-1, 3)
        prepared.append((holes, kept))
        notes.append(note)

    n_ground_truth = sum(len(holes) for holes, _ in prepared)

    primary_tag = f'{PRIMARY_CENTRE_TOLERANCE_PX:g}px'

    results = {
        'contract': {
            'candidate_floor': candidate_floor,
            'generation_threshold': generation_threshold,
            'operating_threshold': operating_threshold,
            'detection_cap': cap,
            'detection_cap_applied': any(note['n_capped'] for note in notes),
            'recall_points': RECALL_POINTS,
            'ap_assignment': 'confidence',
            'counting_assignment': 'optimal',
            'primary_metric': f'AP_center_{primary_tag}',
            'n_images': len(prepared),
            'n_ground_truth': n_ground_truth,
            'n_images_without_holes': sum(1 for holes, _ in prepared if not len(holes)),
        },
        'candidates': {
            'n_raw': sum(note['n_raw'] for note in notes),
            'n_non_finite_score': sum(note['n_non_finite_score'] for note in notes),
            'n_below_floor': sum(note['n_below_floor'] for note in notes),
            'n_capped': sum(note['n_capped'] for note in notes),
            'n_kept': sum(note['n_kept'] for note in notes),
        },
    }

    centre_aps = []

    for tolerance in CENTRE_TOLERANCES_PX:
        criterion = partial(centre_criterion, tolerance=tolerance)
        ap = _split_ap(prepared, n_ground_truth, criterion)
        results[f'AP_center_{tolerance:g}px'] = ap
        centre_aps.append(ap)

    thresholds = '_'.join(f'{t:g}' for t in CENTRE_TOLERANCES_PX)
    results[f'AP_center_mean_over_{thresholds}px'] = float(np.mean(centre_aps))

    for alpha in RADIUS_RELATIVE_ALPHAS:
        criterion = partial(relative_criterion, alpha=alpha)
        results[f'AP_center_relative_{alpha:g}r'] = _split_ap(prepared, n_ground_truth, criterion)

    for threshold in CIRCLE_IOU_THRESHOLDS:
        criterion = partial(circle_iou_criterion, threshold=threshold)
        results[f'AP_circle_{int(threshold * 100)}'] = _split_ap(prepared, n_ground_truth, criterion)

    primary = partial(centre_criterion, tolerance=PRIMARY_CENTRE_TOLERANCE_PX)

    counts = counting_scores(prepared, primary, operating_threshold)

    results[f'precision_center_{primary_tag}_optimal'] = counts['precision']
    results[f'recall_center_{primary_tag}_optimal'] = counts['recall']
    results[f'F1_center_{primary_tag}_optimal'] = counts['f1']
    results['counts'] = counts
    results['geometry'] = geometry_errors(prepared, primary, operating_threshold)

    return results
