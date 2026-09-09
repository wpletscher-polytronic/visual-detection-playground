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

WHAT IS NOT HERE. The overfit gate's rule that an image with no matches is a failure does
not apply: an image with no holes and no detections is a correct result and contributes
nothing to either precision or recall. See centerpoint/overfit.py for that gate.
"""

import numpy as np

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
    return np.linalg.norm(holes[:, None, :2] - detections[None, :, :2], axis=-1)


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

    distance = centre_distances(holes, detections)
    r_gt = holes[:, None, 2]
    r_det = detections[None, :, 2]

    with np.errstate(invalid='ignore', divide='ignore'):
        # Clipped because a distance a hair outside the valid range is a rounding artefact,
        # not a signal, and arccos would return nan for it.
        cos_gt = np.clip((distance ** 2 + r_gt ** 2 - r_det ** 2) / (2 * distance * r_gt), -1, 1)
        cos_det = np.clip((distance ** 2 + r_det ** 2 - r_gt ** 2) / (2 * distance * r_det), -1, 1)
        lens = (r_gt ** 2 * np.arccos(cos_gt) + r_det ** 2 * np.arccos(cos_det)
                - 0.5 * np.sqrt(np.clip((-distance + r_gt + r_det) * (distance + r_gt - r_det)
                                        * (distance - r_gt + r_det) * (distance + r_gt + r_det),
                                        0, None)))

    smaller = np.minimum(r_gt, r_det)
    contained = np.pi * smaller ** 2
    intersection = np.where(distance <= np.abs(r_gt - r_det), contained, lens)
    intersection = np.where(distance >= r_gt + r_det, 0.0, intersection)

    union = np.pi * r_gt ** 2 + np.pi * r_det ** 2 - intersection
    usable = np.isfinite(intersection) & (union > 0) & (r_gt > 0) & (r_det > 0)
    return np.where(usable, np.divide(intersection, union, out=np.zeros_like(union),
                                      where=usable), 0.0)


def centre_criterion(holes, detections, tolerance=PRIMARY_CENTRE_TOLERANCE_PX):
    """Fixed pixel tolerance. Returns (feasible, cost) with cost in pixels, lower better."""
    distance = centre_distances(holes, detections)
    # A non-finite centre is unmatchable rather than an error: nothing upstream of the
    # linear offset head guarantees a finite coordinate.
    feasible = np.isfinite(distance) & (distance <= tolerance)
    return feasible, np.where(np.isfinite(distance), distance, 0.0)


def relative_criterion(holes, detections, alpha=RADIUS_RELATIVE_ALPHAS[0]):
    """Tolerance proportional to each hole's own radius, so scale cancels."""
    holes = np.asarray(holes, float)
    distance = centre_distances(holes, detections)
    if not distance.size:
        return distance.astype(bool), distance
    tolerance = alpha * holes[:, None, 2]
    feasible = np.isfinite(distance) & (distance <= tolerance)
    return feasible, np.where(np.isfinite(distance), distance, 0.0)


def circle_iou_criterion(holes, detections, threshold=CIRCLE_IOU_THRESHOLDS[0]):
    """Disc overlap. Cost is 1 - IoU so that minimising total cost maximises total overlap."""
    overlap = circle_iou(holes, detections)
    return overlap > threshold, 1.0 - overlap


# --- assignment --------------------------------------------------------------------------------

def assign_optimal(feasible, cost):
    """Maximum-cardinality assignment, minimum total cost among those. Returns [(i, j)].

    The two objectives become one by padding to a square matrix and charging every
    infeasible pair UNMATCHED instead of a cost. UNMATCHED exceeds the largest total a full
    set of feasible pairs could reach, so converting any infeasible slot into a feasible one
    always lowers the total: cardinality is decided outright and cost only breaks ties.
    """
    feasible = np.asarray(feasible, bool)
    cost = np.asarray(cost, float)
    n_gt, n_det = feasible.shape
    if not n_gt or not n_det or not feasible.any():
        return []

    size = max(n_gt, n_det)
    worst = float(cost[feasible].max())
    unmatched = (size + 1) * (worst + 1.0)

    padded = np.full((size, size), unmatched)
    np.copyto(padded[:n_gt, :n_det], cost, where=feasible)

    assigned = _hungarian(padded)
    return sorted((int(i), int(j)) for j, i in enumerate(assigned)
                  if i < n_gt and j < n_det and feasible[i, j])


def assign_by_confidence(feasible, cost, scores):
    """Detections in descending score order, each taking its cheapest unclaimed hole.

    The AP rule. Not the same answer as assign_optimal — an early confident detection can
    take a hole that a later one needed — and that is intended: AP is defined by a ranking
    sweep, so a detection's label must be fixed by the time the sweep passes it rather than
    revised when a lower-ranked detection appears.

    Returns [(i, j)] and the order the detections were considered in, since the caller
    needs that order to build the curve.
    """
    feasible = np.asarray(feasible, bool)
    cost = np.asarray(cost, float)
    scores = np.asarray(scores, float)

    # Stable, so equal scores keep the order decode returned and the result is reproducible.
    order = np.argsort(-scores, kind='stable')

    claimed = set()
    pairs = []
    for j in order:
        candidates = [i for i in range(feasible.shape[0]) if feasible[i, j] and i not in claimed]
        if not candidates:
            continue
        best = min(candidates, key=lambda i: (cost[i, j], i))
        claimed.add(best)
        pairs.append((best, int(j)))
    return pairs, order


def match_by_distance(gt_xy, det_xy, tolerance):
    """Optimal one-to-one matching on centre distance. Returns [(gt, det, distance)].

    The convenience form the overfit gate uses, kept here so there is exactly one matcher
    in the project. gt_xy and det_xy may be the full (n, 3) / (m, 4) arrays or bare
    coordinate pairs; only the first two columns are read.
    """
    gt_xy = np.asarray(gt_xy, float).reshape(-1, 2) if len(gt_xy) == 0 else np.asarray(gt_xy, float)
    det_xy = np.asarray(det_xy, float).reshape(-1, 2) if len(det_xy) == 0 else np.asarray(det_xy, float)
    if not len(gt_xy) or not len(det_xy):
        return []

    distance = centre_distances(gt_xy, det_xy)
    feasible = np.isfinite(distance) & (distance <= tolerance)
    return [(i, j, float(distance[i, j])) for i, j in assign_optimal(feasible, distance)]


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
    assertion in the reconstruction loop states exactly that, so a future edit that breaks
    the invariant fails loudly instead of hanging.

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
            assert 0 <= visit_order[previous] < visit_order[column], (
                "augmenting chain left the visited tree; this is the invariant that makes "
                "termination independent of the costs")
            column_row[column] = column_row[previous]
            column = previous

    return column_row[1:] - 1


# --- candidate generation ------------------------------------------------------------------------

def prepare_detections(detections, candidate_floor=CANDIDATE_FLOOR, cap=DETECTION_CAP):
    """Apply the candidate policy to one image. Returns (detections, notes).

    The same policy must run for both models, or the comparison is between two different
    candidate sets rather than two models. Everything it discards it counts, so a silent
    truncation cannot be mistaken for a model that simply predicted less.
    """
    detections = np.asarray(detections, float).reshape(-1, 4)
    notes = {'n_raw': int(len(detections)), 'n_non_finite_score': 0,
             'n_below_floor': 0, 'n_capped': 0}
    if not len(detections):
        return detections, notes

    finite = np.isfinite(detections[:, 3])
    notes['n_non_finite_score'] = int((~finite).sum())
    detections = detections[finite]

    keep = detections[:, 3] >= candidate_floor
    notes['n_below_floor'] = int((~keep).sum())
    detections = detections[keep]

    if len(detections) > cap:
        order = np.argsort(-detections[:, 3], kind='stable')[:cap]
        notes['n_capped'] = int(len(detections) - cap)
        detections = detections[np.sort(order)]
    return detections, notes


# --- average precision -----------------------------------------------------------------------------

def label_detections(holes, detections, criterion):
    """Confidence-ordered TP/FP labels for one image. Returns [(score, is_true_positive)].

    A detection that lands on a hole another detection already claimed is a false positive,
    which is what makes decode's local-maximum suppression part of the measurement rather
    than incidental to it.
    """
    holes = np.asarray(holes, float).reshape(-1, 3)
    detections = np.asarray(detections, float).reshape(-1, 4)
    if not len(detections):
        return []
    if not len(holes):
        return [(float(score), False) for score in detections[:, 3]]

    feasible, cost = criterion(holes, detections)
    pairs, order = assign_by_confidence(feasible, cost, detections[:, 3])
    true_positives = {j for _i, j in pairs}
    return [(float(detections[j, 3]), j in true_positives) for j in order]


def average_precision(labels, n_ground_truth, recall_points=RECALL_POINTS):
    """AP over the whole split from pooled (score, is_tp) labels. COCO's 101-point envelope.

    labels is pooled across every image; per-image AP averaged afterwards is a different
    and worse-behaved quantity, so it is not offered.

    Ties in score are broken by the order the caller pooled them in, which is deterministic
    because label_detections walks a stable sort. The interpolation takes precision to be
    non-increasing in recall — p_interp(r) = max{p(r') : r' >= r} — then reads it at 101
    evenly spaced recalls and averages. Recall never reaching 1 is not padded: the missing
    points contribute zero, which is what makes AP punish missed holes.
    """
    if n_ground_truth == 0:
        # No holes anywhere means recall is undefined, not zero. The caller decides whether
        # such a split is worth reporting; it must not silently read as a perfect score.
        return float('nan')
    if not labels:
        return 0.0

    ordered = sorted(range(len(labels)), key=lambda k: (-labels[k][0], k))
    is_tp = np.array([labels[k][1] for k in ordered], dtype=float)

    true_positives = np.cumsum(is_tp)
    false_positives = np.cumsum(1.0 - is_tp)
    precision = true_positives / np.maximum(true_positives + false_positives, 1e-12)
    recall = true_positives / n_ground_truth

    # Right-to-left running maximum: the envelope, not the raw sawtooth.
    envelope = np.maximum.accumulate(precision[::-1])[::-1]

    # arange/(n-1), not linspace: linspace(0, 1, 101)[70] is 0.7000000000000001, so a
    # recall of exactly 0.7 fell a bit-width below its own query point and the whole tail
    # of the curve read as unreachable. Integer division lands on the same double that
    # tp / n_ground_truth produces for the divisors that actually occur.
    queries = np.arange(recall_points) / (recall_points - 1)
    # searchsorted rather than a loop: the first index whose recall reaches each query.
    indices = np.searchsorted(recall, queries, side='left')
    sampled = np.where(indices < len(envelope), envelope[np.minimum(indices, len(envelope) - 1)], 0.0)
    return float(sampled.mean())


# --- the reported bundle ---------------------------------------------------------------------------

def counting_scores(samples, criterion, operating_threshold=OPERATING_THRESHOLD):
    """Precision, recall and F1 at one operating threshold, under optimal assignment.

    samples is [(holes, detections)] with detections already through prepare_detections.
    Images with no holes and no detections are counted and contribute nothing — they are a
    correct result, not a failure.
    """
    matched = predicted = actual = 0
    for holes, detections in samples:
        holes = np.asarray(holes, float).reshape(-1, 3)
        detections = np.asarray(detections, float).reshape(-1, 4)
        detections = detections[detections[:, 3] >= operating_threshold] if len(detections) \
            else detections

        actual += len(holes)
        predicted += len(detections)
        if len(holes) and len(detections):
            feasible, cost = criterion(holes, detections)
            matched += len(assign_optimal(feasible, cost))

    precision = matched / predicted if predicted else float('nan')
    recall = matched / actual if actual else float('nan')
    if predicted and actual and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0 if (predicted or actual) else float('nan')

    return {'precision': precision, 'recall': recall, 'f1': f1,
            'n_matched': matched, 'n_predicted': predicted, 'n_ground_truth': actual,
            'n_unmatched_ground_truth': actual - matched,
            'n_unmatched_detections': predicted - matched,
            'assignment': 'optimal', 'operating_threshold': operating_threshold}


def geometry_errors(samples, criterion, operating_threshold=OPERATING_THRESHOLD):
    """Centre and radius error over matched pairs, under optimal assignment.

    Reported with the counts beside them, because a model that matches three easy holes and
    misses forty looks excellent on medians alone.

    `radius_bias` is signed and measured against OUR derived annotation radii, which come
    from max(w, h) / 2 on measurably non-square boxes. Using the same convention on both
    sides gives a common target; it does not make the model's error relative to a physical
    hole radius cancel.
    """
    centre, radius, signed = [], [], []
    for holes, detections in samples:
        holes = np.asarray(holes, float).reshape(-1, 3)
        detections = np.asarray(detections, float).reshape(-1, 4)
        detections = detections[detections[:, 3] >= operating_threshold] if len(detections) \
            else detections
        if not len(holes) or not len(detections):
            continue

        feasible, cost = criterion(holes, detections)
        distance = centre_distances(holes, detections)
        for i, j in assign_optimal(feasible, cost):
            centre.append(float(distance[i, j]))
            signed.append(float(detections[j, 2] - holes[i, 2]))
            radius.append(abs(float(detections[j, 2] - holes[i, 2])))

    def spread(values, name):
        if not values:
            return {f'{name}_median': float('nan'), f'{name}_p95': float('nan'),
                    f'{name}_max': float('nan')}
        return {f'{name}_median': float(np.median(values)),
                f'{name}_p95': float(np.percentile(values, 95)),
                f'{name}_max': float(np.max(values))}

    return {'n_matched': len(centre),
            **spread(centre, 'centre_error'),
            **spread(radius, 'radius_error'),
            'radius_bias_mean': float(np.mean(signed)) if signed else float('nan'),
            'radius_bias_median': float(np.median(signed)) if signed else float('nan'),
            'assignment': 'optimal', 'operating_threshold': operating_threshold}


def evaluate_split(samples, candidate_floor=CANDIDATE_FLOOR,
                   operating_threshold=OPERATING_THRESHOLD, cap=DETECTION_CAP):
    """Every named metric for one split. samples is [(holes, raw_detections)].

    Names carry their criterion, their threshold and, where it matters, their assignment
    rule. Nothing here is called mAP: an average over our distance thresholds is not the
    box-IoU quantity that name refers to, and the key states the exact set it averaged.
    """
    prepared, notes = [], []
    for holes, detections in samples:
        kept, note = prepare_detections(detections, candidate_floor, cap)
        prepared.append((np.asarray(holes, float).reshape(-1, 3), kept))
        notes.append(note)

    n_ground_truth = sum(len(holes) for holes, _ in prepared)
    results = {
        'contract': {
            'candidate_floor': candidate_floor,
            'operating_threshold': operating_threshold,
            'operating_threshold_status': 'provisional, inherited from decode',
            'detection_cap': cap,
            'detection_cap_bound': any(note['n_capped'] for note in notes),
            'recall_points': RECALL_POINTS,
            'ap_assignment': 'confidence',
            'counting_assignment': 'optimal',
            'primary_metric': f'AP_center_{PRIMARY_CENTRE_TOLERANCE_PX:g}px',
            'n_images': len(prepared),
            'n_ground_truth': n_ground_truth,
            'n_images_without_holes': sum(1 for holes, _ in prepared if not len(holes)),
        },
        'candidates': {
            'n_raw': sum(note['n_raw'] for note in notes),
            'n_below_floor': sum(note['n_below_floor'] for note in notes),
            'n_non_finite_score': sum(note['n_non_finite_score'] for note in notes),
            'n_capped': sum(note['n_capped'] for note in notes),
        },
    }

    def ap_for(criterion):
        labels = [label for holes, detections in prepared
                  for label in label_detections(holes, detections, criterion)]
        return average_precision(labels, n_ground_truth)

    for tolerance in CENTRE_TOLERANCES_PX:
        def criterion(holes, detections, tolerance=tolerance):
            return centre_criterion(holes, detections, tolerance)
        results[f'AP_center_{tolerance:g}px'] = ap_for(criterion)

    thresholds = '_'.join(f'{t:g}' for t in CENTRE_TOLERANCES_PX)
    results[f'AP_center_mean_over_{thresholds}px'] = float(np.mean(
        [results[f'AP_center_{t:g}px'] for t in CENTRE_TOLERANCES_PX]))

    for alpha in RADIUS_RELATIVE_ALPHAS:
        def criterion(holes, detections, alpha=alpha):
            return relative_criterion(holes, detections, alpha)
        results[f'AP_center_relative_{alpha:g}r'] = ap_for(criterion)

    for threshold in CIRCLE_IOU_THRESHOLDS:
        def criterion(holes, detections, threshold=threshold):
            return circle_iou_criterion(holes, detections, threshold)
        results[f'AP_circle_{int(threshold * 100)}'] = ap_for(criterion)

    def primary(holes, detections):
        return centre_criterion(holes, detections, PRIMARY_CENTRE_TOLERANCE_PX)

    tag = f'{PRIMARY_CENTRE_TOLERANCE_PX:g}px'
    counts = counting_scores(prepared, primary, operating_threshold)
    results[f'precision_center_{tag}_optimal'] = counts['precision']
    results[f'recall_center_{tag}_optimal'] = counts['recall']
    results[f'F1_center_{tag}_optimal'] = counts['f1']
    results['counts'] = counts
    results['geometry'] = geometry_errors(prepared, primary, operating_threshold)
    return results
