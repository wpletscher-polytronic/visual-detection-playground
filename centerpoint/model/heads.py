"""Three prediction heads on the neck output.

The output contract, spelled out because getting it wrong still trains:

    heatmap   LOGITS. Sigmoid lives in the loss, and once at inference in
              detector.detections_from. Applying it here too squashes everything into
              (0.500, 0.731): every cell past the 0.3 threshold, prior init erased.
    offset    unrestricted linear; encode's targets are unsigned [0, 1) today.
    radius    linear, in cells. Not ReLU, which risks a unit that outputs zero for every
              input and stops learning. A negative radius is therefore possible, and
              clamping belongs to detections_from so training sees the raw prediction.

The loss is not BCE on these soft targets — see model/losses.py.
"""

import math

import torch.nn as nn

# Start the heatmap at "holes are rare" rather than "50% everywhere", easing the
# foreground/background imbalance early. RetinaNet's prior probability initialisation.
HEATMAP_PRIOR = 0.01




class Head(nn.Module):
    """conv3x3 -> ReLU -> conv1x1. The ReLU is what makes the second conv worth having:
    two stacked linear convolutions would collapse into one."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)




class CenterPointHeads(nn.Module):
    """heatmap (1), offset (2) and radius (1), all read from the same neck output."""

    def __init__(self, in_channels):
        super().__init__()
        self.heatmap = Head(in_channels, 1)
        self.offset = Head(in_channels, 2)
        self.radius = Head(in_channels, 1)

        # sigmoid(b) = p  ->  b = -log((1 - p) / p).  p = 0.01 gives about -4.595.
        prior_bias = -math.log((1 - HEATMAP_PRIOR) / HEATMAP_PRIOR)
        nn.init.constant_(self.heatmap.block[-1].bias, prior_bias)

    def forward(self, features):
        return {'heatmap': self.heatmap(features),      # logits
                'offset': self.offset(features),
                'radius': self.radius(features)}
