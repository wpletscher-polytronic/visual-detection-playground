"""STEP 5c — three prediction heads on the stride-4 feature map. The simplest file here.

Classes
- Head(in_ch, out_ch)
    conv3x3 -> relu -> conv1x1. That is all CenterNet uses.

- CenterPointHeads(in_ch)
    heatmap  1 ch, sigmoid  -> [0, 1] objectness per cell
    offset   2 ch, linear   -> sub-pixel remainder
    radius   1 ch           ** OPEN: relu / softplus / exp, tied to the radius
                               representation chosen in codec/encode.py — they must match **
    forward(x) -> dict with those three keys

Initialisation — do not skip this
- Bias the final heatmap conv to -log((1 - p) / p) with p ~ 0.01, so the model starts out
  predicting "almost everything is background". Without it the first steps produce a huge
  focal loss from ~16k confidently-wrong cells and training starts unstable. This is a
  RetinaNet/CenterNet detail that is easy to miss and hard to diagnose afterwards.
- Other heads: default init is fine.

Invariants
- heatmap output in [0, 1]; radius output strictly > 0
- all three heads read the SAME feature map — one shared trunk, three cheap tails
- head count and channel widths do not depend on the backbone

Verify by
- a shape test: (B, 3, 512, 512) in -> heatmap (B, 1, 128, 128), offset (B, 2, ...),
  radius (B, 1, ...)
- at random init the heatmap should be near p, not near 0.5. If it is 0.5, the bias init
  did not take.
"""
