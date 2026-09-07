# Center-Point Bullet-Hole Detection — Plan

Goal: heatmap detector predicting (center, radius) per hole, benchmarked head-to-head
against the YOLO26 baseline on the same split.
Approach: own implementation on a torchvision backbone. New code, copied math.
Learning project, no deadline.

## Environment (verified, not assumed)
- torch 2.14.0+xpu, torchvision 0.29.0. `torch.cuda.is_available()` is **False**.
- Device string is `"xpu"`, not `"cuda"`. Write device-agnostic; never hardcode `.cuda()`.
- Local Intel iGPU: overfit tests and encode/decode debugging only.
- Colab: full runs. Budget ~2-4 h/run (YOLO26 fine-tune was ~1 h, 73 s/epoch x 48).
- ResNet-18 has `IMAGENET1K_V1` only — no V2. Use `.DEFAULT`.
- Data: reuse `bullet_rchsr` clean split unchanged. ~3.3k images, ~60k hole instances.
- No new labeling required. Boxes -> `(cx, cy)`, `r = max(w,h)/2`.

## Implementation tiers
Rule: write it yourself if a bug would still train. Delegate it if a bug would crash.

Tier 1 — use as-is, never implement:
- torchvision ResNet-18 + ImageNet weights (only pretrained component in the project)
- albumentations for augmentation coordinate math
- TensorBoard, checkpointing, config

Tier 2 — write every line yourself (~250 lines, the whole intellectual content):
- target encoding: Gaussian splat + sub-pixel offset + radius targets
- losses: penalty-reduced focal (alpha=2, beta=4) + masked L1
- decode: 3x3 max-pool peaks, top-k, threshold, offset, radius
- heads (3x conv-relu-conv) and the stride-32 -> stride-4 decoder
- the eval matching criterion (defines what "correct" means; ~20 lines)

Tier 3 — delegate:
- dataset wrapper, training loop, eval harness plumbing, visualisation, Colab notebook

## Phases
0. Baseline diagnosis — raise `IOU` 0.30 -> 0.7 in `8_yolo26_detection.py`, dump pre-NMS
   boxes on cluster cases, record how much failure NMS actually causes.
1. Scaffold + failing unit tests. Tier 3 written, Tier 2 stubbed with equations in docstrings.
2. Target encoding. Render encoded heatmap over source image; verify by eye.
3. Model: ResNet-18 + plain transposed-conv decoder + heatmap/offset heads. Shapes only.
4. Losses. **Overfit 8 images to near-zero loss.** Hard gate — nothing downstream works if this doesn't.
5. Decode + visualisation. Confirm the 8 overfit images round-trip to the right points.
6. First full training run. Point metrics only, no radius yet.
7. Radius head. L1, lambda ~0.1. Switch on cIoU-AP.
8. Eval harness + YOLO26 comparison, overall and on the dense/overlapping subset.
9. Backbone ladder: plain decoder -> U-Net skips from layer1/2/3 -> HRNet-W18 (timm).
   Three points on a curve, one change at a time.
10. Write results into `INFO.md` section 9.

Phases 1-5 are one or two sittings and need no meaningful GPU time.

## Unit tests to write first (they are the learning device)
- encode a known point/radius -> argmax at that pixel, value exactly 1.0
- encode -> decode round-trips to sub-pixel tolerance
- two points 3 px apart survive as two peaks; 1 px apart merge
  (this pins the sigma/stride trade-off numerically instead of by intuition)
- focal loss ~0 on a perfect prediction, large on an inverted one

## Decided
- Heatmap/CenterNet-style, not P2PNet set prediction. Debuggable, and has a radius head.
- Own implementation. Read the references, fork nothing.
- Pretrained ImageNet backbone; never train a backbone from scratch on 3.3k images.
- ResNet-18 first for iteration speed and comparability, not because it is best.
- Modernize the scaffolding (AMP, AdamW + cosine, albumentations, no DCN compilation).
  Do NOT modernize the math until a faithful baseline trains.

## Open decisions — yours, not mine
- Heatmap sigma: CenterNet's IoU-derived `gaussian_radius`, or simply proportional to
  hole radius. The latter is simpler and better matched to circular objects — but decide
  it deliberately and write down why.
- Radius representation: raw px, log-radius, or fraction of image size.
- Radius loss: L1 (CircleNet), smooth-L1, or a cIoU/gCIoU regression loss.
- Output stride: 4 (standard) vs 2 (better peak separation, 4x memory).
- Match threshold for the point metric: absolute px, fraction of GT radius, or k-NN
  normalised (nAP). This defines what "correct" means for the whole project.
- Ellipse vs circle for perspective-distorted holes.
- Input handling: full 640px images, or crops at native resolution.

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
