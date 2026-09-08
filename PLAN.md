# Center-Point Bullet-Hole Detection — Plan

Goal: heatmap detector predicting (center, radius) per hole.
Approach: own implementation on a torchvision backbone. New code, copied math.
Learning project, no deadline.

Success criterion: matches the YOLO26 baseline within a reasonable margin on the same
split, is fully understood line by line, and carries no AGPL dependency.
Explicitly NOT "beats YOLO on dense clusters" — see Motivation.

## Motivation
- Learning. The point is to understand every part, not to ship the best number.
- Licensing. `ultralytics` is **AGPL-3.0** (verified in the installed metadata). Copyleft
  reaches a whole network-facing service, so shipping it commercially needs a paid licence.
  torchvision is BSD; CenterNet and CircleNet are MIT. An own implementation sidesteps this.
  Consequence: never paste AGPL code in, and keep the MIT attribution notice on anything
  derived from the reference implementations.
- NOT because NMS demonstrably fails. That diagnosis was dropped unmeasured (see Phases).
  And YOLO26 is NMS-free by default anyway — its one-to-one head emits final detections
  directly — so avoiding NMS is no advantage over this particular baseline.

## Environment (verified, not assumed)
- torch 2.14.0+xpu, torchvision 0.29.0. `torch.cuda.is_available()` is **False**.
- Device string is `"xpu"`, not `"cuda"`. Write device-agnostic; never hardcode `.cuda()`.
- Local Intel iGPU: overfit tests and encode/decode debugging only.
- Colab: full runs. Budget ~2-4 h/run (YOLO26 fine-tune was ~1 h, 73 s/epoch x 48).
- ResNet-18 has `IMAGENET1K_V1` only — no V2. Use `.DEFAULT`.
- Data: `bullet_holes/clean_v30` for bring-up (easy, YOLO26 hit 99.3% there, so a low
  score means a bug), `bullet_rchsr/clean` as the target set and the only source of
  stride/sigma — never tune on v30. Both are placeholders for our own data later.
- No new labeling required. Boxes -> `(cx, cy)`, `r = max(w,h)/2`.

## Implementation tiers
Rule: write it yourself if a bug would still train. Delegate it if a bug would crash.

Tier 1 — use as-is, never implement:
- torchvision ResNet-18 + ImageNet weights (only pretrained component in the project)
- albumentations for augmentation coordinate math
- TensorBoard, checkpointing, config

Tier 2 — write every line yourself (~250 lines, the whole intellectual content):
- target encoding, losses, decode, heads and decoder, the eval matching criterion.
- Per-file TODOs in `centerpoint/` carry the signatures, invariants and open choices.

Tier 3 — delegate:
- dataset wrapper, training loop, eval harness plumbing, visualisation, Colab notebook

## Phases

0. ~~Baseline NMS diagnosis~~ — **dropped**, unmeasured. Motivation is learning plus
   licensing, not a demonstrated NMS failure. Consequence: we never established how much
   of the cluster failure NMS caused, so no claim may be made about it.
1. ~~**Measure the data first.**~~ **DONE** — `centerpoint/analyze_labels.py`,
   results in `centerpoint/outputs/debug/label_stats/<dataset>/stats.json`.
   See "Step 1 results" below.
2. ~~Config.~~ **DONE** — split into `paths.py` (where things are) and `params.py`
   (values two modules must agree on: STRIDE, IMG_SIZE). The torch Dataset moved to 5.5,
   because encode and decode need no images.
3. ~~Target encoding.~~ **DONE** — `codec/encode.py`.
4. ~~Decode.~~ **DONE** — `codec/decode.py`. 34 tests, green at strides 1 to 20.
   Steps 1-4 are numpy and cv2 only — no torch, no GPU.
5. Model: backbone + neck + heads. Shapes and sanity only. See "Step 5 design" below.
5.5 Dataset + collate, once training needs batches of real images.
5.7 ~~**Perturbation tests before any training.**~~ **DONE** — see results below. Every check so far uses ideal targets,
   where the peak is always in the right cell. Feed decode heatmaps with unequal peaks,
   sub-cell shifts and noise, and see what survives. This is the only way to learn what
   the decoder does with realistic input, and it settles the stride question properly.
6. Losses, then **overfit 8 images to near-zero loss**. Hard gate.
7. First full training run. Point metrics only, no radius yet.
8. Radius head.
9. Eval harness + YOLO26 comparison — a **signal**, not an oracle. A large gap is a reason
   to go looking, but initialisation, training recipe, capacity, augmentation and label
   noise can all explain one. It does not prove an encode/decode defect.
10. Capacity ladder, in this order: **neck design -> stride -> input resolution -> backbone
    size**. A bigger backbone will not fix small holes — ResNet-18/34/50 share an identical
    downsampling schedule, so capacity and resolution are orthogonal. One change at a time.
11. Write results into `INFO.md` section 9.

Steps 1-4 are one or two sittings and need no GPU.

## Optional / future — Boundary refinement (do not build yet)

Gated on the core center+radius detector working and being measured first.

Idea: per hole, refine the circle to the real torn boundary by seeded region growing —
seed = detected center, growth capped by that hole's own predicted circle, each hole
grown independently so refined boundaries may overlap.

### Verified
- Adams & Bischof, "Seeded Region Growing", IEEE PAMI 16, 641-647, 1994. Citation correct.
- Liu et al., "Overlapping Bullet Hole Detection Based on Improved Watershed", ICSIP 2023,
  pp. 136-140. Exists; contents not retrieved.
- Independent-growth-allows-overlap vs watershed-forces-disjoint is a real, documented
  distinction, not a rationalisation. Watershed partitions by construction.
- `cv2.floodFill`'s mask argument does encode the circular constraint directly.

### Challenges — read before committing to this
- **The scoring rule may not want shape at all.** ISSF scores with a plug gauge of nominal
  calibre seated in the hole; a shot reaches a ring when its *centre* is within
  ring radius + 2.25 mm (4.5 mm pellet). CMP and NSRA are gauge-based too. So the official
  rule is literally center + fixed known radius, and explicitly ignores the ragged tear.
  Refining to the true boundary moves *away* from the rule. Settle which rule set the
  internship targets use before building anything here.
- **The boundary is unobservable exactly where it is wanted.** Where two holes merge, the
  paper between them is gone — there is no image evidence of hole A's edge inside the
  overlap. Growth just floods to the circle cap, so in the overlap region the output is
  100% the predicted circle and 0% refinement. The stated motivation names the one region
  the method cannot help with.
- **The cap contradicts the motivation.** Growth can only ever shrink the circle, never
  exceed it. Torn paper between close holes typically bulges *outside* a clean circle —
  the constraint clips precisely the deviation being chased. This is erosion-only refinement.
- **Unfalsifiable with current labels.** Boxes only, no polygon/mask GT. Nothing to score a
  refined boundary against without new annotation.
- **Polarity returns.** `floodFill` grows on raw intensity via loDiff/upDiff — the exact
  assumption that broke scripts 1-3 (holes are not consistently darker or lighter). Any
  growth criterion must run on the contrast residual (script 4) or a learned feature.
- **Wrong term to optimise.** Radius GT is `max(w,h)/2` from boxes and is coarse. Refining
  shape while the scale estimate is sloppy is polish on top of noise.
- **No downstream consumer for the overlap.** Scoring needs per-hole center and size.
  Overlapping masks are philosophically correct but currently change no output.

### Alternatives, if shape does turn out to matter
- SAM point prompt — independent masks, overlap by construction, zero training. But
  documented point-prompt coordinate bias on small objects, leaking into adjacent regions
  near edges, high sensitivity to exact prompt location, SAM2 weak on fine detail.
  Cheaper first probe than region growing; apply the circle as a post-hoc mask.
- Mask R-CNN style per-ROI masks — overlap by construction, but needs mask annotations.
- BCNet, "Occlusion-Aware Instance Segmentation with Overlapping BiLayers" (arXiv 2103.12340)
  — explicitly models overlapping object pairs in two layers.
- Amodal instance segmentation — the correct frame for the invisible part. AISDiff
  (arXiv 2409.18256), ShapeMoE (arXiv 2508.01664), AURA (ICCV 2025). All need amodal GT and
  a learned shape prior; classical region growing cannot do this.
- Concave-point detection + ellipse fitting (arXiv 2008.00997) — classical, works on the
  outer contour of a merged blob. Better classical fit than region growing, but partitions.

### Open questions before this phase is worth starting
- Which scoring rule set applies? If gauge-based, is this phase needed at all?
- Measure first: on ~20 hard cases, how far does the fitted circle actually sit from a
  hand-drawn boundary, and would any of those deviations change a ring call?
- Is refinement wanted on the outer (visible, ring-facing) edge only? That is a much
  smaller, better-posed problem than full boundary recovery.

## Step 1 results (measured, train split: 3101 images, 58533 holes)

Reproduce with `python -m centerpoint.analyze_labels`.

- Every image is already 640x640. No resize or letterbox decision needed.
- Radius px, p1/p5/p25/p50/p75/p95/p99: `1.0 2.0 3.0 5.5 7.75 12.0 17.0`
  - at stride 4 that is `0.25 0.50 0.75 1.38 1.94 3.00 4.25` cells
  - at stride 2, `0.50 1.00 1.50 2.75 3.88 6.00 8.50` cells
- Nearest-neighbour centre distance px, p1/p5/p25/p50/p75: `3.6 5.4 12.1 20.1 34.2`
- Hard limit (two centres in one cell): stride 4 = **0.19%**, stride 2 = **0.01%**.
  CenterNet on COCO was 0.07%.
- Soft limit (neighbour within N cells, where peaks flatten into one):
  - stride 4: <1 cell 1.45%, <2 cells **11.74%**, <3 cells 24.41%
  - stride 2: <1 cell 0.03%, <2 cells 1.45%, <3 cells 6.23%
- Boxes are not square: aspect p50 1.21, p95 2.0. `r = max(w,h)/2` overestimates.
- valid and test agree with train, so the splits are consistent.

What this means
- The collision number understates the risk, but the 2-cell figure overstates it. A 3x3
  peak window only suppresses a neighbour within **1 cell**: that is 1.45% at stride 4 and
  0.03% at stride 2. The 11.74%-within-2-cells figure is holes whose target valley is
  shallow (0.73), which a model *may* fail to reproduce — a hypothesis, not a measurement.
  Establish it with the perturbation tests (phase 5.7) before treating it as fact.
- Holes are small: a quarter are under 0.75 cells radius at stride 4. Sigma would have to
  be ~1 cell, which a heatmap can barely represent.
- Evidence leans stride 2. Cost is 4x head memory (320x320 vs 160x160). Still your call.
- Radius p1 is 1 px and p5 is 2 px — a 2 px-diameter hole at 640x640 may be annotation
  noise. Overlays confirm the parse is right, so these are real labels, not a bug.

## Decided
- Success is parity with the baseline plus full understanding and no AGPL, not a win.
- Heatmap/CenterNet-style, not P2PNet set prediction. Debuggable, and has a radius head.
- Own implementation. Read the references, fork nothing.
- Pretrained ImageNet backbone; never train a backbone from scratch on 3.3k images.
- ResNet-18 first for iteration speed and comparability, not because it is best.
- Modernize the scaffolding (AMP, AdamW + cosine, albumentations, no DCN compilation).
  Do NOT modernize the math until a faithful baseline trains.

## Decided at Step 3 (STRIDE and IMG_SIZE in `params.py`, the rest in `codec/encode.py`)
- **Stride 2.** Holes with a neighbour within 2 cells: 11.74% at stride 4, 1.45% at
  stride 2. Radius p25 goes 0.75 -> 1.50 cells. Costs 4x head memory (320x320 grid).
- **`sigma = max(0.5 * radius_cells, 1.0)`.** Follows hole size but never below one cell:
  a sub-cell Gaussian is a lone lit pixel with no falloff, and the focal loss `(1-Y)^beta`
  term exists precisely to exploit that falloff. The 0.5 is sweepable; the floor is not.
- **Radius stored in cells** (`r / stride`), so centres, offsets and radii share one
  coordinate system and decode is a single multiply by stride for all three.
- **Splat is windowed to 3 sigma.** Measured: full-grid costs 44.8 ms/image at stride 2
  (139 s/epoch), windowed 0.22 ms. Same result to within 5e-3 (pure truncation).

## The merge limit is a DECODE problem, not an encode one
Measured, and it corrects the original brief. Element-wise maximum means two centres in
different cells always produce two cells at exactly 1.0, at any separation — nothing
merges in the encoder. What varies is whether a dip exists *between* them:
- same cell: one peak. Unavoidable, 0.01% of holes at stride 2.
- 1 cell apart: two 1.0 cells, **no cell between them**, i.e. a flat 2-cell plateau.
- 2+ cells apart: a real valley (0.73 at 2 cells, 0.28 at 4).

So ~1.4% of holes at stride 2 land in the plateau regime. **Decided: ties are kept**
(`==`, not `>`) — a strict comparison would find neither of two equal adjacent cells
greater than the other and drop both. Verified: 1.5, 2 and 3 cells apart all decode as two.

Caveat for inference. That holds for *encoded* targets, where both cells are exactly 1.0.
A trained model will rarely tie, so one of two adjacent peaks will be marginally lower and
lose the 3x3 comparison. The max-pool window is measured in cells, so the minimum
resolvable separation is +/- 1 cell — another reason the window size is a parameter to
think about, not just the stride.

## Open decisions — yours, not mine
- Radius representation for the head: linear in cells (decided at Step 3) or log(radius).
  Log makes errors proportional, so a 1 px error on a 5 px hole weighs as much as a 20 px
  error on a 100 px one. Note the usual justification is wrong under L1: d|x|/dx is +/-1
  regardless of magnitude, so large holes do *not* dominate the gradient. Changes decode.
- Sigma when the stride changes. Note the algebra first: physical sigma is
  `max(0.5 * r_px, stride)`, so the *scaled* term has no stride in it. Halving the stride
  narrows only the holes sitting at the floor — 76% of them at stride 4, but 35% at
  stride 2. "Halving stride halves the bumps" is false for the majority.
- Radius loss: L1 (CircleNet), smooth-L1, or a cIoU/gCIoU regression loss.
- Match threshold for the point metric: absolute px, fraction of GT radius, or k-NN
  normalised (nAP). This defines what "correct" means for the whole project.
- Ellipse vs circle for perspective-distorted holes.
- Whether to filter tiny boxes (p1 radius 1 px, p5 2 px).

## Step 5 design (measured, before writing any of it)

- **Bilinear upsample + 3x3 conv, not ConvTranspose2d.** Transposed convolution makes
  periodic checkerboard artefacts when the kernel is not divisible by the stride (Odena
  et al., Distill 2016). CenterNet dodges it with kernel 4 stride 2. For us the risk is
  not cosmetic: decode finds local maxima, so a periodic artefact manufactures spurious
  peaks at regular intervals.
- **Taper the neck width per level.** One 3x3 conv 64->64 at 320x320 is 3.77 GMACs, about
  a whole ResNet-18 stage (~3.7 GMACs). At 32 channels it is 0.94. The finest level wants
  position, not semantics.
- **Radius head is linear, never ReLU.** A ReLU unit that goes negative outputs zero and
  gets zero gradient forever — a dead radius that cannot recover. CenterNet uses a linear
  size head with L1 for this reason.
- **Sigmoid on the heatmap lives in exactly one place**, the layer or the loss. In both, it
  trains plausibly and is quietly wrong.
- **Expose the stem** (after relu, before maxpool). ResNet's finest residual output is
  stride 4, so stride 2 needs it. Taking it one line later silently gives stride 4 back
  with no error.
- **Develop at stride 4, switch to 2 for real runs.** The stride-2 output map is 4x the
  activations (210 MB vs 52 MB, 64 channels at batch 8) and the debug loop runs dozens of
  times. Same pattern as developing on v30 and targeting rchsr. Verified switchable: the
  codec is green at strides 1 to 20.
- **The loss balance shifts with stride.** Stride 4 -> 2 takes positives to negatives from
  roughly 1:2,600 to 1:10,200 at the same hole count. If training behaves differently after
  a stride change, check the loss-term magnitudes before suspecting the architecture.
- **BatchNorm behaves differently in train and eval mode**, and ResNet-18 is full of it.
  Eight correlated images give unreliable running statistics, so the overfit gate can look
  good in train mode and much worse in eval. Check both. Freezing the pretrained BN
  statistics is an experiment to run if needed, not a default. Gradient accumulation does
  not help — BN never sees the larger batch.
- Backbone forward at 640x640 on this machine's XPU: 32 ms.

## The near-miss geometry problem (found in review, not yet addressed)

offset and radius are supervised **only at the exact ground-truth centre cell**; every
other cell holds zero and receives no gradient for that object. So if the model's peak
lands one cell off, decode reads geometry from a cell that was never trained.

Measured on a hole at (100.3, 200.7) with r = 9, forcing the peak one cell sideways:

    decoded  [100.0  198.0   0.0]
    truth    [100.3  200.7   9.0]
    centre error 2.72 px, radius error 9.00 px

The radius decodes to **zero**. A 100% error from a heatmap that was nearly right.

This is inherent to the CenterNet arrangement, not a bug in our code — the reference
supervises regression at centre locations too. What matters is that **no test we have can
see it**: the exact round trip, the 34 unit tests and inspect_codec all use ideal targets
where the peak is always in the correct cell.

### Measured by the phase 5.7 sweep, and there is a clean fix

Densest v30 image, 13 holes, whole-bump shifts:

    shift    stride 4 radius err    stride 2 radius err
      0            0.00 px               0.00 px
      1           11.75 px              11.75 px
      2           11.75 px              11.75 px

In this synthetic test a finer stride halves the centre error (3.04 -> 1.41 px) and
leaves the radius equally destroyed. It does not follow that stride never helps in
practice: how often a trained model misses by a whole cell, and whether that rate depends
on stride, is unmeasured.

Spreading each centre's offset and radius into a 3x3 block before decoding:

    spread   stride 4 radius err    stride 2 radius err
      0           11.75 px              11.75 px
      1            0.00 px               0.00 px

**It fixes the radius. It does NOT fix the centre**, and copying offsets makes the centre
worse: 1.50 px error without spreading, 2.00 px with. An offset is measured from its own
cell, so cell `col+1` needs `x - (col+1)`, not the copied `x - col`. Getting this right
means recomputing signed offsets per cell — the current targets are unsigned [0, 1) — plus
an assignment rule where neighbourhoods from nearby holes overlap, and a matching change
to the loss mask and its normalisation.

So: a promising later ablation, not a design change to slip in. Pinned in
tests/test_decode.py::test_copying_geometry_outward_rescues_radius_but_not_centre.

Until then: when training starts, always inspect predicted peak location and sampled
geometry together. A good heatmap does not imply good decoded circles.

## Phase 5.7 results — the stride question, settled with evidence

**Merge rate, measured by decoding rather than estimated from proximity.** Encode the real
labels, break exact ties with 1% multiplicative jitter (a trained model never ties), decode
the whole train split, compare against the peaks the encoder produced:

    dataset  stride   collision   merge
    v30        4        0.05%     0.30%
    v30        2        0.02%     0.00%
    rchsr      4        0.19%     4.37%
    rchsr      2        0.01%     0.61%

Stride 2 recovers about **3.9 points** on rchsr (4.56% lost -> 0.62%) and 0.3 on v30.
Not the 11.74% the plan first claimed, nor the 6.23% proximity estimate that replaced it:
a Euclidean distance threshold is not the suppression condition, which depends on grid
alignment and relative peak heights. Reproduce with `merge_rate()` in inspect_codec.py.

Still an upper bound on quality — the heatmaps are ideal apart from the tie-break.

**A cost we had not considered: on this noise model, the finer grid produces more false
peaks.** Additive i.i.d. Gaussian noise on the heatmap *probabilities*, clipped to [0, 1],
same sigma at both strides, 13 real holes on the densest v30 image:

    sigma    stride 4 detections   stride 2 detections
     0.02           13                    14
     0.05           13                    17
     0.10           47                   199

The ratio is not a constant 4x — it is 1.1x at sigma 0.02 and 4.2x at 0.10 — and the
result is specific to this noise model. Real model error is neither i.i.d. nor applied to
probabilities, and clipping at 1.0 creates plateaus that decode keeps as ties: 199
detections clipped versus 187 unclipped at sigma 0.10.

Supported conclusion: calibrate SCORE_THRESHOLD separately per stride and watch false
positives, not only recall. NOT supported: raising the stride-2 threshold pre-emptively
before seeing a trained model's precision-recall behaviour.

Net: stride 2 is still justified on rchsr and marginal on v30, and developing at stride 4
remains the right call for iteration speed — but on measured merge rates, not on an
unmeasured rescue rate.

## Risks / difficulties
- Heatmaps relocate the merge failure, they do not remove it. Two centers in one stride-4
  cell yield one detection; overlapping same-class Gaussians combine by element-wise max,
  so peaks closer than ~sigma flatten into one. Better trade for circular objects, not a fix.
- Silent failures are the real cost. A wrong sigma or an off-by-one in the stride-4
  coordinate mapping trains happily and produces a plausible, wrong heatmap. Mitigations:
  the overfit-8 gate, and dumping the predicted heatmap as an image every epoch.
- Do not copy `gaussian_radius` blindly. The version in CircleNet (and most CenterNet
  forks) has the quadratic-formula denominator as `2` instead of `2a`, making r2 ~4x too
  large. Inherited from CornerNet 2018, fixed upstream, still live downstream. It still
  trains, which is exactly why nobody noticed — and why Tier 2 gets written, not copied.
- Small holes: ~10 px at stride 4 is ~2.5 heatmap px, comparable to sigma. The sub-pixel
  offset head is load-bearing.
- Radius GT is derived from boxes, not annotated. It inherits box sloppiness.
- Perspective makes holes elliptical; a circle then fits worse than a box.
- No pretrained checkpoint exists for this head combination. Only the backbone transfers.
- Biggest time sink is the eval harness and the fair baseline comparison, not the model.
- Focal loss hyperparameters may need retuning away from COCO defaults (one dense class).

## Read, don't fork
- CircleNet — hrlblab/CircleNet, MIT. Exactly this architecture (heatmap + offset + radius).
  Read `losses.py`, `decode.py`, `utils/image.py`. Install path is py3.7 / torch 1.11 /
  gcc-6 / compile DCNv2 — do not go there.
- CenterNet — xingyizhou/CenterNet. Original. Last push 2023.
- mmdetection CenterNet configs — COCO-pretrained ResNet-18 checkpoints (25.9 box AP)
  if a warmer start is ever wanted. Repo frozen since Aug 2024, ~1960 open issues.
- HRNet / HigherHRNet — the principled version of "keep stride-4 detail". Phase 9.
- APGCC (arXiv 2405.10589), PET (arXiv 2308.13814), STEERER (arXiv 2308.10468) — only if
  the heatmap route stalls on scale variance.
- torchvision `ops.deform_conv2d` — DCNv2 built in, no compilation, if ever wanted.

## Corrections to the original brief
- "No off-the-shelf radius-regression head exists" — false. CircleNet is CenterNet plus a
  1-channel radius head, L1 loss, lambda=0.1, public code, MIT.
- "No established evaluation metric fits" — false. cIoU has a closed form and drops into
  mAP unchanged; gCIoU extends it; nAP covers point-only localisation.
- P2PNet has no heatmap and no size output. It is Hungarian set prediction over dense
  point proposals. The "visually debuggable heatmap" upside applies to CenterNet only.
- Dense small circular object detection is well-solved in digital pathology. The novelty
  here is the application, not the method — which means copy rather than invent.
- Newer backbones do handle small objects (Swin/PVT pyramids, HRNet, DINOv3 dense
  features). ResNet-18 is chosen for iteration speed and baseline comparability, not
  because the alternatives cannot see small holes.
