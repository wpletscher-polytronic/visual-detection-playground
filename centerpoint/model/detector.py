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
        # The neck upsamples by exactly 2 per step, which only lines up with the backbone
        # levels when both sides divide by 32. Off-size input either raises inside the
        # neck (with skips) or silently returns the wrong output stride (without them).
        height, width = images.shape[-2:]
        assert height % 32 == 0 and width % 32 == 0, (
            f"input {height}x{width} must be divisible by 32")
        return self.heads(self.neck(self.backbone(images)))


def detections_from(outputs, index=0, stride=STRIDE, threshold=SCORE_THRESHOLD):
    """One image's model outputs -> the (m, 4) array decode returns.

    Deliberately NOT part of forward. Four things happen here that must not sit in the
    differentiable path: the heatmap sigmoid is applied (exactly once — the heads emit
    logits and the loss will apply its own), the radius is clamped, gradients are dropped,
    and the tensors cross to numpy because decode is numpy.

    The radius clamp is the negative-radius policy. The head is linear so it can and does
    emit negatives — about 25% of cells at random init — and keeping it linear is what
    stops a ReLU unit dying. Training therefore sees the raw prediction and a signed L1
    error, while a caller asking for detections gets a circle it can actually draw.
    """
    with torch.no_grad():
        heatmap = torch.sigmoid(outputs['heatmap'][index]).detach().cpu().numpy()
        offset = outputs['offset'][index].detach().cpu().numpy()
        radius = outputs['radius'][index].clamp(min=0.0).detach().cpu().numpy()
    return decode(heatmap, offset, radius, stride=stride, threshold=threshold)
