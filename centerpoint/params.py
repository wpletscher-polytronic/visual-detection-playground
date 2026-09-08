"""Values more than one module has to agree on. Imported, never run.

Anything only one module uses stays in that module: sigma in codec/encode.py, the score
threshold in codec/decode.py, the percentile list in analyze_labels.py, lr in train.py.
The knob belongs next to the code it controls.

What lives here is the opposite case — a value where disagreement is silent. If encode
floored centres at stride 2 and decode scaled by 4, every coordinate would be doubled and
nothing would raise.
"""

# codec/encode.py, codec/decode.py, model/neck.py and data/dataset.py must all agree.
#
# 4 during development: a 160x160 grid instead of 320x320, so the debug loop is cheaper.
# Measured merge loss on the target dataset is 4.56% at stride 4 against 0.62% at stride 2
# (PLAN.md, phase 5.7), so real runs belong at 2 — as a SEPARATE experiment with its own
# checkpoints and results, not by flipping this on an already-trained model.
STRIDE = 4

# Both datasets are natively 640x640, so nothing is resized. Downscaling would drag the
# p5 hole radius from 2.0 px to 1.6 px, which is signal we cannot spare.
IMG_SIZE = 640


def resolve_device():
    """'xpu', 'cuda' or 'cpu'. Resolved rather than set, but global to a run like the rest.

    This machine runs the Intel XPU build of torch, where torch.cuda.is_available() is
    False and torch.xpu.is_available() is True. Colab will be the opposite, so any code
    calling .cuda() directly breaks on one of the two.

    torch is imported lazily so the numpy-only scripts do not pay for it.
    """
    import torch

    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
