"""Backbone + neck + heads, and the one place torch output becomes codec input.

The model emits the same three maps codec/encode.py builds from ground truth, so
codec/decode.py reads either without knowing which produced it.
"""

import torch
import torch.nn as nn

from centerpoint.codec.decode import SCORE_THRESHOLD, decode
from centerpoint.model.backbone import Backbone
from centerpoint.model.heads import CenterPointHeads
from centerpoint.model.neck import Neck
from centerpoint.params import STRIDE


class CenterPointNet(nn.Module):
    """image -> {'heatmap' (logits), 'offset', 'radius'} at out_stride."""

    def __init__(self, out_stride=STRIDE, pretrained=True, skips=True):
        super().__init__()
        self.out_stride = out_stride
        self.backbone = Backbone(pretrained=pretrained)
        self.neck = Neck(self.backbone.channels, out_stride=out_stride, skips=skips)
        self.heads = CenterPointHeads(self.neck.out_channels)

    def forward(self, images):
        # Off-size input misaligns the neck's ladder; neck.py has the failure modes.
        height, width = images.shape[-2:]
        assert height % 32 == 0 and width % 32 == 0, (
            f"input {height}x{width} must be divisible by 32")
        return self.heads(self.neck(self.backbone(images)))


def detections_from(outputs, stride, index=0, threshold=SCORE_THRESHOLD):
    """One image's model outputs -> the (m, 4) array decode returns.

    `stride` has no default: falling back to params.STRIDE would silently decode a
    stride-2 model at stride 4. Pass net.out_stride.

    Outside forward because training needs the raw predictions, not because sigmoid and
    clamp are non-differentiable — they are. Sigmoid runs once here (the heads emit
    logits, the loss applies its own), clamp makes the circle drawable, float32 because
    decode is numpy and bfloat16 has no numpy dtype.
    """
    with torch.no_grad():
        heatmap = torch.sigmoid(outputs['heatmap'][index]).float().cpu().numpy()
        offset = outputs['offset'][index].float().cpu().numpy()
        radius = outputs['radius'][index].clamp(min=0.0).float().cpu().numpy()
    return decode(heatmap, offset, radius, stride=stride, threshold=threshold)
