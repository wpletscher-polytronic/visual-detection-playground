"""STEP 5a — ImageNet-pretrained feature extractor. The only pretrained part of the project.

Functions
- build_backbone(name='resnet18', pretrained=True) -> nn.Module
    torchvision resnet18 with ResNet18_Weights.DEFAULT (note: ResNet-18 has V1 weights
    only, there is no IMAGENET1K_V2 for it). Drop avgpool and fc — classification is not
    wanted, only the feature hierarchy.
    forward(x) -> dict {'C2', 'C3', 'C4', 'C5'}

Reference shapes, 512x512 input (verified on this machine)
    C2  (64, 128, 128)  stride 4     <- layer1, where small holes still live
    C3 (128,  64,  64)  stride 8     <- layer2
    C4 (256,  32,  32)  stride 16    <- layer3
    C5 (512,  16,  16)  stride 32    <- layer4

Invariants
- channel counts and strides above are exact for resnet18/34; resnet50+ is 4x wider
  (256/512/1024/2048) because of the bottleneck expansion. The neck must read them from
  the backbone, not hardcode them.
- input normalised with ImageNet mean/std, or the pretrained weights are wasted
- swapping the backbone must not require touching the heads

Why resnet18: fastest debug loop, and every CenterNet reference uses it so published
numbers are comparable. Not because it is the best — see PLAN.md phase 10.
"""
