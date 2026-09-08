"""Shapes, initialisation and gradient flow. No pretrained weights, so this stays offline.

The neck is not built yet, so the heads are fed a backbone level directly. That is enough
to exercise every gradient path the real model will have.
"""

import math

import numpy as np
import pytest
import torch

from centerpoint.model.backbone import Backbone
from centerpoint.model.detector import CenterPointNet, detections_from
from centerpoint.model.heads import HEATMAP_PRIOR, CenterPointHeads
from centerpoint.model.neck import Neck
from centerpoint.params import IMG_SIZE
BATCH = 2
# Strides are ratios, so a small input asserts them just as well as 640 and keeps the
# suite fast on CPU. Must stay divisible by 32 so every level lands on a whole number.
SIZE = 128


@pytest.fixture(autouse=True)
def _deterministic():
    """Several tests draw random inputs and assert on the outcome. Without a fixed seed
    they pass or fail depending on the draw, which is worse than having no test."""
    torch.manual_seed(0)


@pytest.fixture(scope='module')
def backbone():
    return Backbone(pretrained=False)


def test_backbone_returns_every_level_at_the_right_stride(backbone):
    feats = backbone(torch.zeros(1, 3, SIZE, SIZE))
    strides = {name: SIZE // f.shape[-1] for name, f in feats.items()}
    assert strides == {'stem': 2, 'C2': 4, 'C3': 8, 'C4': 16, 'C5': 32}


def test_stem_is_taken_before_the_maxpool(backbone):
    """The trap: reading it one line later gives stride 4, and a stride-2 head would then
    behave exactly like a stride-4 one with no error anywhere."""
    feats = backbone(torch.zeros(1, 3, SIZE, SIZE))
    assert feats['stem'].shape[-1] == SIZE // 2
    assert feats['stem'].shape[-1] == 2 * feats['C2'].shape[-1]


def test_declared_channel_widths_match_the_real_ones(backbone):
    """The widths are stated, not probed, so something has to check they are true."""
    feats = backbone(torch.zeros(1, 3, SIZE, SIZE))
    assert backbone.channels == {name: f.shape[1] for name, f in feats.items()}
    assert backbone.channels == {'stem': 64, 'C2': 64, 'C3': 128, 'C4': 256, 'C5': 512}


def test_construction_leaves_batchnorm_untouched():
    """Regression: the constructor used to probe channel widths with a dummy forward.
    no_grad() stops autograd but not BatchNorm, which updates running statistics whenever
    the module is in training mode — and a fresh module is. One all-zeros batch pulled
    every running_var to 0.9 of its value, shifting eval-mode output at C5 by ~73%."""
    fresh = Backbone(pretrained=False)
    batchnorms = [m for m in fresh.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert batchnorms, "expected BatchNorm layers in the backbone"
    for bn in batchnorms:
        assert bn.num_batches_tracked.item() == 0
        assert torch.equal(bn.running_var, torch.ones_like(bn.running_var))
        assert torch.equal(bn.running_mean, torch.zeros_like(bn.running_mean))


def test_heads_shapes():
    heads = CenterPointHeads(64)
    out = heads(torch.zeros(BATCH, 64, 32, 32))
    assert out['heatmap'].shape == (BATCH, 1, 32, 32)
    assert out['offset'].shape == (BATCH, 2, 32, 32)
    assert out['radius'].shape == (BATCH, 1, 32, 32)


def test_heatmap_bias_is_the_prior():
    """Assert the bias itself, not the output of a random network."""
    heads = CenterPointHeads(64)
    expected = -math.log((1 - HEATMAP_PRIOR) / HEATMAP_PRIOR)
    assert expected == pytest.approx(-4.595, abs=1e-3)
    assert heads.heatmap.block[-1].bias.item() == pytest.approx(expected, abs=1e-5)
    assert torch.sigmoid(torch.tensor(expected)).item() == pytest.approx(HEATMAP_PRIOR, abs=1e-6)


def test_heatmap_head_emits_logits_not_probabilities():
    """If a sigmoid ever appears inside the head, this catches it: logits are unbounded."""
    heads = CenterPointHeads(64)
    out = heads(torch.randn(1, 64, 16, 16) * 10)
    assert out['heatmap'].min() < 0.0, "a probability could never be negative"


def test_gradients_reach_the_backbone_and_all_three_heads(backbone):
    """Forward -> dummy loss -> backward. Not the real loss; this only proves nothing is
    detached and every parameter is on the graph."""
    heads = CenterPointHeads(backbone.channels['C2'])
    backbone.zero_grad(set_to_none=True)
    heads.zero_grad(set_to_none=True)

    out = heads(backbone(torch.randn(1, 3, SIZE, SIZE))['C2'])
    loss = sum(v.square().mean() for v in out.values())
    loss.backward()

    assert backbone.stem[0].weight.grad is not None, "no gradient reached the backbone"
    assert backbone.layer1[0].conv1.weight.grad is not None
    for name in ('heatmap', 'offset', 'radius'):
        grad = getattr(heads, name).block[0].weight.grad
        assert grad is not None and grad.abs().sum() > 0, f"{name} head got no gradient"


def test_batchnorm_runs_in_both_modes(backbone):
    """Outputs are not expected to match — BN uses batch statistics in train mode and
    running statistics in eval. The point is that eval mode works at all, since the
    overfit gate can look fine in one and bad in the other."""
    x = torch.randn(BATCH, 3, SIZE, SIZE)
    backbone.train()
    trained = backbone(x)['C5']
    backbone.eval()
    with torch.no_grad():
        evaluated = backbone(x)['C5']
    assert trained.shape == evaluated.shape
    assert torch.isfinite(evaluated).all()


# --- neck, composition, and the boundary to the numpy codec ---

def test_neck_reaches_the_output_stride(backbone):
    for out_stride in (4, 2):
        neck = Neck(backbone.channels, out_stride=out_stride)
        out = neck(backbone(torch.zeros(1, 3, SIZE, SIZE)))
        assert out.shape[-1] == SIZE // out_stride
        assert out.shape[1] == neck.out_channels


def test_skip_and_no_skip_share_the_same_ladder(backbone):
    """An architectural comparison, NOT a parameter-matched one: same widths, steps and
    output shape, but the skip version necessarily carries extra lateral convolutions.
    Any measured difference includes that extra capacity, not only the fusion."""
    feats = backbone(torch.zeros(1, 3, SIZE, SIZE))
    with_skips = Neck(backbone.channels, out_stride=4, skips=True)
    without = Neck(backbone.channels, out_stride=4, skips=False)

    assert with_skips.out_channels == without.out_channels
    assert with_skips.steps == without.steps
    assert with_skips(feats).shape == without(feats).shape
    assert len(without.lateral) == 0, "no-skip must not read the finer levels at all"
    assert len(with_skips.lateral) == len(with_skips.steps)


def test_detector_output_shapes():
    net = CenterPointNet(out_stride=4, pretrained=False)
    out = net(torch.zeros(BATCH, 3, SIZE, SIZE))
    cells = SIZE // 4
    assert out['heatmap'].shape == (BATCH, 1, cells, cells)
    assert out['offset'].shape == (BATCH, 2, cells, cells)
    assert out['radius'].shape == (BATCH, 1, cells, cells)


def test_every_parameter_receives_a_gradient():
    """Stronger than spot-checking a few tensors: every parameter in the model must be on
    the graph. A None gradient means something is disconnected."""
    net = CenterPointNet(out_stride=4, pretrained=False)
    loss = sum(v.square().mean() for v in net(torch.randn(1, 3, SIZE, SIZE)).values())
    loss.backward()

    missing = [n for n, p in net.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing[:5]}"

    non_finite = [n for n, p in net.named_parameters() if not torch.isfinite(p.grad).all()]
    assert not non_finite, f"non-finite gradient in: {non_finite[:5]}"

    # and each of the three heads must be driven, not just present on the graph
    for head in ('heatmap', 'offset', 'radius'):
        grads = [p.grad.abs().sum().item()
                 for n, p in net.named_parameters() if n.startswith(f'heads.{head}.')]
        assert sum(grads) > 0, f"{head} head received only zero gradients"

    # so must the neck's own layers, including the laterals that carry the skips
    for prefix in ('neck.input_proj', 'neck.lateral', 'neck.smooth', 'backbone.layer4'):
        total = sum(p.grad.abs().sum().item()
                    for n, p in net.named_parameters() if n.startswith(prefix))
        assert total > 0, f"{prefix} received only zero gradients"


def test_untrained_model_output_is_decodable():
    """The contract between model and codec, checked before any training exists. The
    detections are meaningless; the point is that the shapes, dtype and units line up."""
    net = CenterPointNet(out_stride=4, pretrained=False).eval()
    out = net(torch.randn(1, 3, SIZE, SIZE))
    dets = detections_from(out, stride=4)

    assert dets.ndim == 2 and dets.shape[1] == 4
    assert dets.dtype == np.float32
    if len(dets):                          # an empty result is entirely valid
        assert (dets[:, 3] >= 0).all() and (dets[:, 3] <= 1).all()
        assert (np.diff(dets[:, 3]) <= 0).all()


def test_inference_clamps_negative_radii_but_the_head_stays_linear():
    """The head must be able to emit negatives — that is what keeps the unit alive — so
    the clamp belongs at the inference boundary, not in the model."""
    outputs = {'heatmap': torch.full((1, 1, 8, 8), 5.0),
               'offset': torch.zeros(1, 2, 8, 8),
               'radius': torch.full((1, 1, 8, 8), -3.0)}
    outputs['heatmap'][0, 0, 4, 4] = 9.0

    dets = detections_from(outputs, stride=4)
    assert len(dets) and (dets[:, 2] >= 0).all(), "inference must not return negative radii"
    assert outputs['radius'].min().item() == pytest.approx(-3.0), "raw prediction untouched"

    # Forced rather than sampled: zero the last conv's weights and push its bias negative,
    # so the output is the bias itself. A random init only emits negatives about seven
    # times in eight, which is a property of the draw, not of the head.
    net = CenterPointNet(out_stride=4, pretrained=False).eval()
    torch.nn.init.zeros_(net.heads.radius.block[-1].weight)
    torch.nn.init.constant_(net.heads.radius.block[-1].bias, -5.0)
    with torch.no_grad():
        raw = net(torch.zeros(1, 3, SIZE, SIZE))['radius']
    expected = torch.full_like(raw, -5.0)
    assert torch.allclose(raw, expected), "nothing may clamp the radius head's output"


def test_detections_from_does_not_touch_autograd():
    net = CenterPointNet(out_stride=4, pretrained=False)
    out = net(torch.randn(1, 3, SIZE, SIZE))
    assert out['heatmap'].requires_grad, "forward must stay differentiable"
    detections_from(out, stride=4)
    assert out['heatmap'].requires_grad, "decoding must not detach the graph in place"
    assert out['heatmap'].grad_fn is not None


@pytest.mark.parametrize('skips', [True, False])
def test_detector_rejects_sizes_the_ladder_cannot_handle(skips):
    """Off-size input raises inside the neck when skips are on, but WITHOUT them it
    silently returns the wrong output stride — 600 px gives 152x152 where 150x150 was
    meant. The detector asserts up front so both cases fail the same way."""
    net = CenterPointNet(out_stride=4, pretrained=False, skips=skips)
    with pytest.raises(AssertionError):
        net(torch.zeros(1, 3, 600, 600))


def test_input_size_from_params_is_accepted():
    net = CenterPointNet(out_stride=4, pretrained=False)
    assert IMG_SIZE % 32 == 0, "the pipeline size must satisfy the neck's contract"
