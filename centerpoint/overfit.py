"""STEP 6 gate — can the whole pipeline memorise eight images?

A diagnostic, not the Step 7 trainer. Deliberately one file with no scheduler, no
augmentation, no resume and no experiment manager: everything here exists to answer one
question, and anything reusable belongs in train.py once this has passed.

What a pass means: images, encode, model, loss, optimiser, decode and the coordinate
conventions between them are wired together correctly. It says nothing about
generalisation, and a low loss on its own says nothing at all — the gate is decoded
geometry, checked per image.

The seed makes a rerun repeatable on this machine and this build. It is not a promise of
bitwise determinism: a different device, driver or torch version can reorder reductions
and move the last digits.

Run either way, from anywhere:
    python -m centerpoint.overfit [--batch 8] [--steps 400] [--dataset ROOT]
    python centerpoint/overfit.py

Exit codes: 0 gate passed, 1 gate failed, 2 out of memory, 3 the selection failed
preflight before any training happened.
"""

import argparse
import copy
import json
import os
import platform
import sys
import time

import cv2
import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from centerpoint.codec.decode import SCORE_THRESHOLD, decode                # noqa: E402
from centerpoint.codec.encode import encode                                 # noqa: E402
from centerpoint.data.dataset import BulletHoleDataset, collate             # noqa: E402
from centerpoint.inspect_codec import colourise, draw_circles, label        # noqa: E402
from centerpoint.metrics import assign_optimal, centre_criterion            # noqa: E402
from centerpoint.model.detector import CenterPointNet, detections_from      # noqa: E402
from centerpoint.model.losses import (ALPHA, BETA, LAMBDA_OFFSET,           # noqa: E402
                                      LAMBDA_RADIUS, detection_loss)
from centerpoint.params import IMG_SIZE, STRIDE, resolve_device             # noqa: E402
from centerpoint.paths import DATASET, debug_dir                            # noqa: E402

SPLIT = 'train'
N_IMAGES = 8

# Diagnostic starting points, not tuned settings. If the gate fails, the failure is
# evidence about one configuration — report it and change one thing, do not sweep.
STEPS = 400
BATCH = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
SEED = 0
LOG_EVERY = 25

# Provisional engineering targets for memorisation. They are not the project's evaluation
# metric — metrics.py is where that goes — and must not be relaxed to make a run pass.
GATE = {
    'centre_tolerance_px': 2.0,     # a detection further than this matches nothing
    'median_centre_error_px': 1.0,  # strictly below
    'median_radius_error_px': 1.5,  # strictly below
    'max_radius_error_px': 3.0,     # at most
}

GT_COLOUR = (0, 255, 0)
DET_COLOUR = (0, 0, 255)


# --- selection -----------------------------------------------------------------------------

def source_stem(image_id):
    """The Roboflow export appends '.rf.<hash>' to each augmented copy of a source photo.

    Stripping it is a heuristic for not picking the same photograph eight times, not a
    guarantee of independence: two exports of one target under different names
    ('..._v1', '..._v2') stay distinct here, and may well be the same physical target.
    """
    return image_id.split('.rf.')[0]


def select_indices(entries, count=N_IMAGES):
    """Indices of the first `count` entries with distinct source stems, in split order.

    Deterministic by construction: list_split sorts by label filename and this walks it
    once, so the same command always trains on the same eight images.
    """
    chosen, seen = [], set()
    for index, (image_id, _image_path, _label_path) in enumerate(entries):
        stem = source_stem(image_id)
        if stem in seen:
            continue
        seen.add(stem)
        chosen.append(index)
        if len(chosen) == count:
            break
    if len(chosen) < count:
        raise RuntimeError(f"only {len(chosen)} distinct source stems in {SPLIT}")
    return chosen


# --- per-image and aggregate scoring ---------------------------------------------------------

def score_image(holes, detections, tolerance=GATE['centre_tolerance_px']):
    """Compare one image's decoded detections against its ground truth.

    holes is the (n, 3) array the Dataset returns, detections the (m, 4) decode produces.
    """
    feasible, distances = centre_criterion(holes, detections, tolerance)
    matches = [(i, j, float(distances[i, j])) for i, j in assign_optimal(feasible, distances)]

    matched_gt = {i for i, _j, _d in matches}
    matched_det = {j for _i, j, _d in matches}

    centre_errors = [d for _i, _j, d in matches]
    radius_errors = [abs(float(detections[j, 2]) - float(holes[i, 2])) for i, j, _d in matches]

    return {
        'n_gt': int(len(holes)),
        'n_detected': int(len(detections)),
        'n_matched': len(matches),
        'unmatched_gt': [i for i in range(len(holes)) if i not in matched_gt],
        'extra_detections': [j for j in range(len(detections)) if j not in matched_det],
        'centre_errors': centre_errors,
        'radius_errors': radius_errors,
        'matched_radii': [float(detections[j, 2]) for _i, j, _d in matches],
        # Reported even when holes went unmatched, so a partial failure still says how far
        # off the part that did match was.
        'median_centre_error': float(np.median(centre_errors)) if centre_errors else float('nan'),
        'max_centre_error': max(centre_errors, default=float('nan')),
        'median_radius_error': float(np.median(radius_errors)) if radius_errors else float('nan'),
        'max_radius_error': max(radius_errors, default=float('nan')),
    }


def gate_verdict(reports, gate=GATE):
    """(passed, aggregates, failures) over the per-image reports of the whole set.

    reports is [(image_id, score_image(...))]. An aggregate never rescues a per-image
    failure: medians are taken across every match in the set, and an image that decoded
    the wrong number of holes fails on its own line.

    Treating "no matches at all" as a failure is right for this gate, where every image is
    a positive example. It is NOT a rule for the eventual eval harness, where an image with
    no holes and no detections is a correct result.
    """
    failures = []
    for image_id, report in reports:
        if report['n_detected'] != report['n_gt']:
            failures.append(f"{image_id}: decoded {report['n_detected']} where ground truth "
                            f"has {report['n_gt']}")
        if report['unmatched_gt']:
            failures.append(f"{image_id}: {len(report['unmatched_gt'])} hole(s) with no "
                            f"detection within {gate['centre_tolerance_px']} px")
        if report['extra_detections']:
            failures.append(f"{image_id}: {len(report['extra_detections'])} detection(s) "
                            f"matching no hole")

    centre = [e for _id, report in reports for e in report['centre_errors']]
    radius = [e for _id, report in reports for e in report['radius_errors']]
    radii = [v for _id, report in reports for v in report['matched_radii']]

    aggregates = {
        'n_matches': len(centre),
        'median_centre_error': float(np.median(centre)) if centre else float('nan'),
        'max_centre_error': float(np.max(centre)) if centre else float('nan'),
        'median_radius_error': float(np.median(radius)) if radius else float('nan'),
        'max_radius_error': float(np.max(radius)) if radius else float('nan'),
    }

    if not centre:
        failures.append("no matches at all, so no error statistic is defined")
    else:
        if not aggregates['median_centre_error'] < gate['median_centre_error_px']:
            failures.append(f"median centre error {aggregates['median_centre_error']:.3f} px "
                            f"is not below {gate['median_centre_error_px']} px")
        if not aggregates['median_radius_error'] < gate['median_radius_error_px']:
            failures.append(f"median radius error {aggregates['median_radius_error']:.3f} px "
                            f"is not below {gate['median_radius_error_px']} px")
        if not aggregates['max_radius_error'] <= gate['max_radius_error_px']:
            failures.append(f"worst radius error {aggregates['max_radius_error']:.3f} px "
                            f"exceeds {gate['max_radius_error_px']} px")
        unusable = [v for v in radii if not np.isfinite(v) or v < 0]
        if unusable:
            failures.append(f"{len(unusable)} matched radius value(s) non-finite or negative")

    return not failures, aggregates, failures


# --- preflight -------------------------------------------------------------------------------

def preflight(holes, image_ids, img_size=IMG_SIZE, stride=STRIDE, threshold=SCORE_THRESHOLD,
              count=N_IMAGES, gate=GATE):
    """Is this selection even winnable by a perfect model? Returns a list of reasons it is not.

    Runs the real encoder and decoder on the ground truth, so the answer is about the codec
    as configured rather than about a proximity estimate. Two centres flooring into one
    stride cell leave one peak for two holes, and no model can then decode the right count
    — that is a property of the selection and the stride, and finding it out after 400
    steps would waste the run and look like a training failure.

    Checks nothing about training. It only rules out a selection that is impossible before
    the model gets blamed for it.
    """
    problems = []
    if len(holes) != count:
        problems.append(f"selected {len(holes)} images, expected {count}")
    if len(set(image_ids)) != len(image_ids):
        problems.append("the selection repeats an image id")

    empty = [image_ids[i] for i, h in enumerate(holes) if not len(h)]
    if empty:
        problems.append(f"{len(empty)} selected image(s) contain no holes, so the gate would "
                        f"prove nothing: {empty[:3]}")
    if sum(1 for h in holes if len(h) > 1) == 0:
        problems.append("no selected image has more than one hole, so nothing tests "
                        "separating neighbouring peaks")

    for image_id, image_holes in zip(image_ids, holes):
        if not len(image_holes):
            continue
        targets = encode(image_holes.astype(np.float32), img_size, stride=stride)
        peaks = int((targets['heatmap'][0] == 1.0).sum())
        if peaks != len(image_holes):
            problems.append(f"{image_id}: {len(image_holes) - peaks} hole(s) share a stride-"
                            f"{stride} cell with another, so the target itself cannot hold "
                            f"{len(image_holes)} peaks")

        ideal = decode(targets['heatmap'], targets['offset'], targets['radius'],
                       stride=stride, threshold=threshold)
        report = score_image(image_holes, ideal, tolerance=gate['centre_tolerance_px'])
        if report['n_detected'] != report['n_gt'] or report['unmatched_gt']:
            problems.append(f"{image_id}: ideal encode/decode returns "
                            f"{report['n_detected']} of {report['n_gt']} holes at threshold "
                            f"{threshold}, so a perfect model could not pass either")
        elif report['max_centre_error'] > gate['median_centre_error_px']:
            problems.append(f"{image_id}: ideal encode/decode is already "
                            f"{report['max_centre_error']:.3f} px off, above the "
                            f"{gate['median_centre_error_px']} px the gate asks of the model")

    return problems


# --- training ---------------------------------------------------------------------------------

def load_fixed_batch(root, device, stride):
    """The eight samples, encoded once and parked on the device. Nothing here is random.

    One collate call rather than a DataLoader: the set is fixed, unshuffled and small
    enough to hold, so a loader would only add a reshuffle nobody wants. `image_path` is
    carried alongside because collate has no reason to and the renders need it.
    """
    dataset = BulletHoleDataset(root, SPLIT)
    indices = select_indices(dataset.entries)

    batch = collate([dataset[i] for i in indices], stride=stride)
    batch['image'] = batch['image'].to(device)
    batch['targets'] = {key: value.to(device) for key, value in batch['targets'].items()}
    batch['image_path'] = [dataset.entries[i][1] for i in indices]
    return batch


def chunks(count, batch_size):
    """Index slices of the fixed set, in order. One slice is one optimiser step."""
    return [list(range(start, min(start + batch_size, count)))
            for start in range(0, count, batch_size)]


def train(net, batch, steps, batch_size, learning_rate, log):
    """Run the optimiser and return the logged loss history. Stops on a non-finite loss.

    Every logged loss is PRE-update: it is the value the parameters had when the step
    began, so the record labelled `steps` describes the model one update before the one
    that gets saved. main() logs a separate post-training loss for the saved parameters.

    A smaller batch_size is the documented out-of-memory fallback, not gradient
    accumulation: each slice is its own optimiser step, and BatchNorm sees exactly the
    images in that slice. Accumulating would not change that, so it is not offered.
    """
    optimiser = torch.optim.AdamW(net.parameters(), lr=learning_rate,
                                  weight_decay=WEIGHT_DECAY)
    slices = chunks(batch['image'].shape[0], batch_size)
    history = []

    net.train()
    for step in range(1, steps + 1):
        rows = slices[(step - 1) % len(slices)]
        targets = {key: value[rows] for key, value in batch['targets'].items()}

        losses = detection_loss(net(batch['image'][rows]), targets)
        if not torch.isfinite(losses['total']):
            raise RuntimeError(
                f"non-finite loss at step {step} on rows {rows}: "
                + ", ".join(f"{k}={v.item()}" for k, v in losses.items())
                + ". Nothing is sanitised here — inspect that slice's inputs and targets "
                  "before touching the learning rate.")

        optimiser.zero_grad(set_to_none=True)
        losses['total'].backward()

        # A finite loss does not imply finite gradients: a saturated sigmoid or a zero
        # denominator can leave the forward value clean and the backward pass full of nan.
        grad_norm = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(p.grad.detach().float())
                         for p in net.parameters() if p.grad is not None]))
        if not torch.isfinite(grad_norm):
            raise RuntimeError(
                f"non-finite gradient at step {step} on rows {rows} while the loss was "
                f"{losses['total'].item():.6f}. The optimiser step was NOT taken. Look for "
                f"the first parameter whose grad is non-finite rather than clipping.")

        optimiser.step()

        if step == 1 or step % LOG_EVERY == 0 or step == steps:
            record = {'step': step, 'when': 'pre-update', 'rows': rows,
                      **{key: float(value.item()) for key, value in losses.items()}}
            record['radius_weighted'] = LAMBDA_RADIUS * record['radius']
            record['grad_norm'] = float(grad_norm.item())
            history.append(record)
            log(f"  {step:5d}  {record['heatmap']:10.4f}  {record['offset']:8.4f}  "
                f"{record['radius']:8.4f}  {record['radius_weighted']:10.4f}  "
                f"{record['total']:9.4f}  {record['grad_norm']:9.3f}")
    return history


@torch.no_grad()
def evaluation_loss(net, batch):
    """The loss of the parameters that actually get saved, in eval mode over the whole set.

    Eval mode so BatchNorm uses its running statistics and this forward pass does not
    update them. That makes the number a different quantity from the training log, not a
    continuation of it: it is comparable across runs, not against the pre-update values.
    """
    net.eval()
    losses = detection_loss(net(batch['image']), batch['targets'])
    record = {'step': 'final', 'when': 'post-update, eval mode',
              **{key: float(value.item()) for key, value in losses.items()}}
    record['radius_weighted'] = LAMBDA_RADIUS * record['radius']
    return record


@torch.no_grad()
def predict(net, batch, stride, batch_size=BATCH, threshold=SCORE_THRESHOLD):
    """Decoded detections and sigmoid heatmaps for every image, in eval mode, in order.

    Chunked by the same batch_size the fallback uses: a machine that could not train on
    eight at once will not manage eight at once here either. Eval mode makes the split
    free of consequence — BatchNorm reads running statistics, so a chunk's neighbours
    cannot change its result.
    """
    net.eval()
    detections, heatmaps = [], []
    for rows in chunks(batch['image'].shape[0], batch_size):
        outputs = net(batch['image'][rows])
        detections.extend(detections_from(outputs, stride=stride, index=i, threshold=threshold)
                          for i in range(len(rows)))
        heatmaps.append(torch.sigmoid(outputs['heatmap']).float().cpu().numpy())
    return detections, np.concatenate(heatmaps)


@torch.no_grad()
def predict_in_train_mode(net, batch, stride, batch_size=BATCH, threshold=SCORE_THRESHOLD):
    """The same predictions with BatchNorm on batch statistics, using a throwaway copy.

    Chunked by the TRAINING slices, not by convenience: batch statistics depend on which
    images share a forward pass, so running all eight together after training on twos would
    compare against a batch composition that never occurred.

    no_grad() does not stop BatchNorm updating its running statistics — only eval mode or
    not calling forward does. Running this on `net` would therefore corrupt the very
    statistics the eval-mode numbers were produced with, so it runs on a deep copy and the
    trained model is left exactly as it was.
    """
    clone = copy.deepcopy(net)
    clone.train()
    detections = []
    for rows in chunks(batch['image'].shape[0], batch_size):
        outputs = clone(batch['image'][rows])
        detections.extend(detections_from(outputs, stride=stride, index=i, threshold=threshold)
                          for i in range(len(rows)))
    return detections


# --- output -------------------------------------------------------------------------------------

def render(image_path, gt_heatmap, pred_heatmap, holes, detections, out_path):
    """Four panels: source, encoded target, predicted heatmap, and both sets of circles."""
    img = cv2.imread(image_path)
    if img is None:
        raise IOError(f"could not read {image_path}")
    size = img.shape[1]

    circles = draw_circles(img.copy(), holes, GT_COLOUR)
    circles = draw_circles(circles, detections, DET_COLOUR, pad=2)

    panel = np.hstack([
        label(img.copy(), 'source'),
        label(colourise(gt_heatmap, size), 'target heatmap'),
        label(colourise(pred_heatmap, size), 'predicted, sigmoid'),
        label(circles, 'green truth / red decoded (+2px)'),
    ])
    if not cv2.imwrite(out_path, panel):
        raise IOError(f"could not write {out_path}")


def is_out_of_memory(error):
    """torch raises OutOfMemoryError on some backends and a plain RuntimeError on others."""
    return isinstance(error, getattr(torch, 'OutOfMemoryError', ())) \
        or 'out of memory' in str(error).lower()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', default=DATASET)
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--batch', type=int, default=BATCH,
                        help='out-of-memory fallback; the eight images do not change')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--tag', default='', help='suffix on the run directory name')
    args = parser.parse_args(argv)

    # Checked here rather than left to fail later: --steps 0 would otherwise produce an
    # empty history and an untrained model reported as a gate result.
    if args.steps < 1:
        parser.error(f"--steps must be at least 1, got {args.steps}")
    if not 1 <= args.batch <= N_IMAGES:
        parser.error(f"--batch must be between 1 and {N_IMAGES}, got {args.batch}")
    if not args.lr > 0 or not np.isfinite(args.lr):
        parser.error(f"--lr must be finite and positive, got {args.lr}")
    return args


def main(argv=None):
    args = parse_args(argv)

    root = os.path.abspath(args.dataset)
    run_id = time.strftime('%Y%m%d-%H%M%S') + (f'-{args.tag}' if args.tag else '')
    run_dir = os.path.join(debug_dir('overfit', root), run_id)
    os.makedirs(run_dir)                     # never exist_ok: a run must not overwrite one

    lines = []

    def log(text=''):
        # flush: a whole run's output is well under the block buffer python uses when
        # stdout is a file, so without this a redirected run shows nothing until it exits.
        print(text, flush=True)
        lines.append(text)

    def save_log():
        with open(os.path.join(run_dir, 'log.txt'), 'w') as handle:
            handle.write('\n'.join(lines) + '\n')

    try:
        return _run(args, root, run_id, run_dir, log)
    except BaseException as error:           # noqa: BLE001 - re-raised below
        # The log is the only record of how far a crashed run got. Written, then the
        # exception continues on its way: a failed run must not exit 0.
        log(f"\nunhandled {type(error).__name__}: {error}")
        raise
    finally:
        save_log()


def _run(args, root, run_id, run_dir, log):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device()

    log(f"overfit-{N_IMAGES} gate")
    log(f"  dataset     {root}   split {SPLIT}")
    log(f"  device      {device}   torch {torch.__version__}   {platform.platform()}")
    log(f"  geometry    stride {STRIDE}   image {IMG_SIZE}   threshold {SCORE_THRESHOLD}")
    log(f"  optimiser   AdamW lr={args.lr} weight_decay={WEIGHT_DECAY}, no schedule")
    log(f"  run         {args.steps} steps, batch {args.batch}, seed {args.seed}, float32")
    log(f"  output      {run_dir}")

    batch = load_fixed_batch(root, device, STRIDE)
    holes = [sample.numpy() for sample in batch['holes']]
    selection = [{'image_id': image_id, 'source_stem': source_stem(image_id),
                  'n_holes': int(len(sample))}
                 for image_id, sample in zip(batch['image_id'], holes)]

    log()
    log("selected images - first distinct source stems in split order")
    log("  holes  source stem, then image id")
    for item in selection:
        log(f"  {item['n_holes']:5d}  {item['source_stem']}")
        log(f"         {item['image_id']}")

    def write_run_json(**extra):
        with open(os.path.join(run_dir, 'run.json'), 'w') as handle:
            json.dump({'run_id': run_id,
                       'config': {'dataset': root, 'split': SPLIT, 'steps': args.steps,
                                  'batch': args.batch, 'lr': args.lr,
                                  'weight_decay': WEIGHT_DECAY, 'seed': args.seed,
                                  'stride': STRIDE, 'img_size': IMG_SIZE,
                                  'score_threshold': SCORE_THRESHOLD, 'device': str(device),
                                  'torch': torch.__version__,
                                  'platform': platform.platform()},
                       'gate': GATE, 'selection': selection, **extra},
                      handle, indent=2)

    problems = preflight(holes, batch['image_id'])
    log()
    if problems:
        log("PREFLIGHT FAILED - this selection cannot pass even with a perfect model")
        for problem in problems:
            log(f"  {problem}")
        log("Nothing was trained. Fix the selection rule or the stride; do not hand-pick "
            "different images to make the gate pass.")
        write_run_json(passed=False, preflight=problems)
        return 3
    log(f"preflight ok - ideal encode/decode recovers every hole in all {N_IMAGES} images "
        f"at stride {STRIDE}, threshold {SCORE_THRESHOLD}")

    net = CenterPointNet(out_stride=STRIDE, pretrained=True, skips=True).to(device)

    log()
    log(f"training - terms unweighted; total = heatmap + 1.0*offset + {LAMBDA_RADIUS}*radius")
    log("  every row is PRE-update: the loss of the parameters that step started with")
    log("   step     heatmap    offset    radius  0.1*radius      total  grad norm")
    try:
        history = train(net, batch, args.steps, args.batch, args.lr, log)
    except RuntimeError as error:
        if not is_out_of_memory(error):
            raise
        log(f"\nout of memory at batch {args.batch}: {error}")
        log("Re-run the whole gate at a smaller batch, e.g. --batch 4. Do not resume - the "
            "configuration changed, so it is a different experiment.")
        write_run_json(passed=False, out_of_memory=True)
        return 2

    final_loss = evaluation_loss(net, batch)
    log(f"  final  {final_loss['heatmap']:10.4f}  {final_loss['offset']:8.4f}  "
        f"{final_loss['radius']:8.4f}  {final_loss['radius_weighted']:10.4f}  "
        f"{final_loss['total']:9.4f}   (post-update, eval mode, all {N_IMAGES})")
    log("  the final row is the saved parameters under running BatchNorm statistics, so it "
        "is a different quantity from the rows above, not their continuation")

    detections, heatmaps = predict(net, batch, STRIDE, batch_size=args.batch)
    gt_heatmaps = batch['targets']['heatmap'].cpu().numpy()

    reports = [(image_id, score_image(holes[i], detections[i]))
               for i, image_id in enumerate(batch['image_id'])]
    passed, aggregates, failures = gate_verdict(reports)

    log()
    log("per image, eval mode")
    log("   gt  det  matched   median centre    max centre  median radius    max radius")
    for image_id, report in reports:
        log(f"  {report['n_gt']:3d}  {report['n_detected']:3d}  {report['n_matched']:7d}   "
            f"{report['median_centre_error']:10.3f} px  {report['max_centre_error']:8.3f} px  "
            f"{report['median_radius_error']:10.3f} px  {report['max_radius_error']:8.3f} px")
        log(f"       {image_id}")
        if report['unmatched_gt'] or report['extra_detections']:
            log(f"       unmatched ground truth {report['unmatched_gt']}, "
                f"extra detections {report['extra_detections']}")

    log()
    log(f"aggregate over {aggregates['n_matches']} matches")
    log(f"  centre error   median {aggregates['median_centre_error']:.3f} px   "
        f"max {aggregates['max_centre_error']:.3f} px")
    log(f"  radius error   median {aggregates['median_radius_error']:.3f} px   "
        f"max {aggregates['max_radius_error']:.3f} px")

    train_slices = chunks(len(holes), args.batch)
    train_mode = predict_in_train_mode(net, batch, STRIDE, batch_size=args.batch)
    train_reports = [score_image(holes[i], train_mode[i]) for i in range(len(holes))]
    log()
    log("BatchNorm - eval mode against train mode on the same images, throwaway copy")
    log(f"  train-mode batch statistics come from the training slices {train_slices}")
    log("  eval det  train det  train median centre   image")
    for i, image_id in enumerate(batch['image_id']):
        log(f"  {len(detections[i]):8d}  {len(train_mode[i]):9d}  "
            f"{train_reports[i]['median_centre_error']:16.3f} px   {image_id[:34]}")

    # Saved before rendering: a cv2 failure must not cost the verdict, the metrics or the
    # weights that took 400 steps to produce.
    torch.save({'state_dict': {k: v.cpu() for k, v in net.state_dict().items()},
                'out_stride': net.out_stride, 'img_size': IMG_SIZE,
                'pretrained': True, 'skips': True,
                'dataset_root': root, 'split': SPLIT, 'image_ids': batch['image_id'],
                'steps': args.steps, 'batch': args.batch, 'lr': args.lr,
                'weight_decay': WEIGHT_DECAY, 'seed': args.seed,
                'loss': {'focal_alpha': ALPHA, 'focal_beta': BETA,
                         'lambda_offset': LAMBDA_OFFSET, 'lambda_radius': LAMBDA_RADIUS},
                'score_threshold': SCORE_THRESHOLD, 'gate': GATE, 'passed': passed},
               os.path.join(run_dir, 'checkpoint.pt'))
    write_run_json(passed=passed, failures=failures, preflight=[],
                   losses=history, final_loss=final_loss, aggregates=aggregates,
                   train_mode_slices=train_slices,
                   per_image=[{'image_id': image_id, **report}
                              for image_id, report in reports])

    render_errors = []
    for i, image_id in enumerate(batch['image_id']):
        try:
            render(batch['image_path'][i], gt_heatmaps[i, 0], heatmaps[i, 0],
                   holes[i], detections[i], os.path.join(run_dir, f'{i}_{image_id}.png'))
        except (IOError, cv2.error) as error:
            render_errors.append(f"{image_id}: {error}")
    if render_errors:
        log()
        log(f"{len(render_errors)} render(s) failed; the verdict and checkpoint are unaffected")
        for problem in render_errors:
            log(f"  {problem}")
        write_run_json(passed=passed, failures=failures, preflight=[],
                       losses=history, final_loss=final_loss, aggregates=aggregates,
                       train_mode_slices=train_slices, render_errors=render_errors,
                       per_image=[{'image_id': image_id, **report}
                                  for image_id, report in reports])

    log()
    if passed:
        log(f"GATE PASSED - {N_IMAGES}/{N_IMAGES} images within the provisional criteria.")
    else:
        log("GATE FAILED")
        for failure in failures:
            log(f"  {failure}")
    log(f"wrote {run_dir}")
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
