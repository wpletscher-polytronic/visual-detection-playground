# centerpoint

CenterNet-style detector predicting (center, radius) per bullet hole.
Plan, phases and open decisions: `../PLAN.md`.

## Layout

- `data/` — dataset, YOLO-box → (center, radius) conversion, augmentation
- `codec/` — encode (points → heatmap/offset/radius targets) and decode
  (heatmap → detections). Exact inverses of each other, so they live together
  and are tested as a round-trip.
- `model/` — backbone, neck/decoder, heads, losses. Everything that is an `nn.Module`.
- `tests/` — run `python -m pytest` from repo root
- `outputs/` — checkpoints, heatmap dumps, eval results (gitignored)

Flat modules at this level, added as the phases need them: config, metrics,
train, evaluate, visualize.

## Data

Reads `datasets/bullet_rchsr/clean/` in place — same images and splits as the YOLO26
baseline, so the comparison stays apples-to-apples. Regenerate it with
`Playground/src/prepare_bullet_rchsr_dataset.py`, run from repo root.
The path is set in one place so this folder stays self-contained.

Qualitative checks run against `datasets/examples/` (five unlabelled images, visual only).
