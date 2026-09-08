"""STEP 5.5 — augmentation. Needs an explicit circle-geometry contract, not a library promise.

"Keypoint-aware" only guarantees the centre transforms correctly. The radius is ours to
check, and not every transform preserves a circle at all:

    uniform scale        circle stays a circle, radius scales      safe
    flip, 90 deg rotate  circle unchanged                          safe
    arbitrary rotation   circle unchanged                          safe
    non-uniform scale    circle becomes an ellipse                 NOT safe
    perspective / shear  circle becomes a conic                    NOT safe

Start with appearance-only changes (brightness, contrast, blur, noise) plus the safe
geometry, and verify the radius transform explicitly rather than assuming albumentations
handles it.

Normalisation: use the pretrained weights' mean and std, but do NOT apply torchvision's
full classification transform — it resizes and centre-crops, which would silently move
every annotation.

Anything that can push a hole out of frame must drop it before calling codec.encode,
which asserts rather than dropping. That is deliberate: a silently discarded hole is lost
training signal with no error.
"""
