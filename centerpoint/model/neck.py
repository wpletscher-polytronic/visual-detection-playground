"""STEP 5b — stride 32 back up to stride 4. Where small-hole recall is decided.

Build the plain version first so the skip version can be MEASURED against it. Two points
on a curve, not one guess.

Classes
- PlainDecoder(in_ch=512, out_ch=64)
    Three ConvTranspose2d blocks, 32 -> 16 -> 8 -> 4, each conv-bn-relu.
    Uses C5 only. Discards all high-resolution detail and re-hallucinates it. This is the
    literal CenterNet-ResNet design, and it is the weak point for ~10 px objects.
    forward(feats) -> (out_ch, H/4, W/4)

- SkipDecoder(out_ch=64)
    Same upsampling path, but concatenate (or add) C4, C3, C2 at their matching
    resolutions, 1x1 conv to reduce channels after each merge. U-Net shaped.
    Detail that C2 already holds is reused instead of reconstructed.
    forward(feats) -> (out_ch, H/4, W/4)

Invariants
- both return the SAME output shape, so swapping one for the other changes nothing
  downstream — that is what makes the comparison clean
- output stride is exactly config.STRIDE
- reads channel counts from the backbone rather than hardcoding 512/256/128/64

Later (phase 10): HRNet-W18 via timm replaces this entirely — it keeps stride-4 resolution
throughout instead of destroying and rebuilding it. Same output contract.
"""
