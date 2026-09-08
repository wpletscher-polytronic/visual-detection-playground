"""Climb from stride 32 back to the output stride. Where small-hole recall is decided.

Each step upsamples, optionally adds the matching backbone level, then convolves: "what"
from above fused with "where" from the side, since upsampling cannot recreate detail the
backbone discarded.

Input sides must divide by 32, or the backbone levels round independently and the ladder
breaks — loudly with skips, SILENTLY without them (600 px gives 152x152 where 150x150 was
meant). CenterPointNet asserts it.

`skips=False` is the C5-only ablation: not parameter-matched, and it does not shrink the
backbone, which still computes every level.

Why bilinear-and-convolve rather than ConvTranspose2d, and why the widths taper:
PLAN.md, Step 5 design.
"""

import torch.nn as nn
import torch.nn.functional as F

from centerpoint.params import STRIDE

# Which backbone level is fused in at each stride on the way up.
SKIP_SOURCE = {16: 'C4', 8: 'C3', 4: 'C2', 2: 'stem'}

# Output width per step. Tapered because a 3x3 conv 64->64 at 320x320 costs about as much
# as a whole ResNet-18 stage, so width is cheaper at the coarse end.
WIDTH = {16: 128, 8: 96, 4: 64, 2: 32}



def conv_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )



class Neck(nn.Module):
    """C5 -> out_stride, optionally fusing a skip at each level.

    backbone_channels: the dict Backbone exposes, so widths are read and not assumed.
    """

    def __init__(self, backbone_channels, out_stride=STRIDE, skips=True):
        super().__init__()
        assert out_stride in WIDTH, f"out_stride must be one of {sorted(WIDTH)}"
        self.out_stride = out_stride
        self.skips = skips

        self.steps = [s for s in (16, 8, 4, 2) if s >= out_stride]

        width_in = WIDTH[self.steps[0]]
        self.input_proj = nn.Conv2d(backbone_channels['C5'], width_in, 1)

        self.lateral = nn.ModuleDict()      # 1x1, skip level -> current width
        self.smooth = nn.ModuleDict()       # 3x3, current width -> this step's width
        for stride in self.steps:
            if skips:
                skip_channels = backbone_channels[SKIP_SOURCE[stride]]
                self.lateral[str(stride)] = nn.Conv2d(skip_channels, width_in, 1)
            self.smooth[str(stride)] = conv_bn_relu(width_in, WIDTH[stride])
            width_in = WIDTH[stride]

        self.out_channels = width_in


    def forward(self, features):
        x = self.input_proj(features['C5'])
        for stride in self.steps:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            if self.skips:
                # Added, not concatenated: the lateral has already matched the widths.
                x = x + self.lateral[str(stride)](features[SKIP_SOURCE[stride]])
            x = self.smooth[str(stride)](x)
        return x
