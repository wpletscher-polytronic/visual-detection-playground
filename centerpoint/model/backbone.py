"""ImageNet-pretrained feature extractor. The only pretrained part of the project.

The backbone trades position for meaning: each stage halves the resolution and doubles the
channels, so a cell at stride 32 describes a large region richly and a cell at stride 2 a
tiny one crudely. Growing the receptive field is what lets the network use "this dark blob
sits inside a printed ring" rather than judging pixels alone. The neck buys position back.

Backbone size and output resolution are separate levers, not substitutes. ResNet-18/34/50
share a downsampling schedule, so a larger one yields no finer maps — but deeper features
can still encode sub-cell information in their channels. What upsampling cannot do is
recover information genuinely discarded. Which lever to pull is empirical, PLAN.md phase 10.
"""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18



class Backbone(nn.Module):
    """ResNet-18 without its classifier. forward(x) -> feature maps at strides 2 to 32."""

    def __init__(self, pretrained=True):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)

        # maxpool stays out of `stem`: folding it in would silently make stem stride 4,
        # and it is the only stride-2 tensor a stride-2 head could skip from.
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.pool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        self.channels = {'stem': 64, 'C2': 64, 'C3': 128, 'C4': 256, 'C5': 512}


    def forward(self, x):
        stem = self.stem(x)                 # stride 2
        c2 = self.layer1(self.pool(stem))   # stride 4
        c3 = self.layer2(c2)                # stride 8
        c4 = self.layer3(c3)                # stride 16
        c5 = self.layer4(c4)                # stride 32
        return {'stem': stem, 'C2': c2, 'C3': c3, 'C4': c4, 'C5': c5}
