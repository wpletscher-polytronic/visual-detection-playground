"""Penalty-reduced focal loss for the heatmap, masked L1 for offset and radius.

The Gaussian is not a regression target. From CenterNet eq. (1):

    positive   a cell where Y == 1, exactly
    negative   every other cell, including one at Y = 0.9, whose penalty is scaled
               by (1 - Y)^BETA

So a cell beside a centre is a negative the model is barely punished for missing, and
sigma sets how much slack it gets. BCE on the same targets would instead train the model
to output 0.9 there — a different objective.

`Y == 1` is an exact float comparison. It holds because encode centres each Gaussian on
the integer cell, and is the first thing to break if a target map is ever resampled.

Both functions compute in float32 whatever they are handed: in float16 the regression sum
overflows before it is normalised, 100 errors of 1000 giving inf for a mean of 1000.
"""

import torch
import torch.nn.functional as F

# CenterNet's COCO values, unchanged for the first baseline. ALPHA down-weights cells the
# model already has right, BETA sets how fast the penalty reduction falls off.
ALPHA = 2.0
BETA = 4.0

# CircleNet's weights. The terms are not naturally comparable: an offset target lies in
# [0, 1) while a radius target is unbounded in cells.
LAMBDA_OFFSET = 1.0
LAMBDA_RADIUS = 0.1




def focal(logits, target):
    """Penalty-reduced focal loss over a batch of heatmaps.

    logits are pre-sigmoid, as the heatmap head emits them, and target is encode's
    heatmap. Working on logits rather than probabilities keeps log(p) finite once p rounds
    to 0 or 1 — which is exactly where a confidently wrong cell needs its gradient.
    """
    logits = logits.float()
    target = target.float()

    positive = target.eq(1.0).float()
    negative = 1.0 - positive

    p = torch.sigmoid(logits)
    q = torch.sigmoid(-logits)      # 1 - p, which cancels to exactly 0 at large logits
    log_p = F.logsigmoid(logits)
    log_q = F.logsigmoid(-logits)

    pos_loss = positive * q.pow(ALPHA) * log_p
    # `negative` is redundant while BETA > 0: (1 - target) is already 0 at a positive. It
    # keeps BETA = 0 usable as an ablation instead of pushing every centre down.
    neg_loss = negative * (1.0 - target).pow(BETA) * p.pow(ALPHA) * log_q

    # Centres, not cells: a cell-count denominator would rescale the loss with the stride.
    # Clamped so an image with no holes gives a finite negative-only loss.
    num_pos = positive.sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos




def masked_l1(pred, target, mask):
    """Mean absolute error over supervised centre cells only.

    mask is (b, 1, h, w) and broadcasts across pred's channels, so the denominator counts
    centres once however many values each supervises: offset's two components sum into one
    error per centre. Dividing by the element count would average them over every cell.
    """
    mask = mask.float()
    error = (pred.float() - target.float()).abs() * mask
    return error.sum() / mask.sum().clamp(min=1.0)




def detection_loss(outputs, targets):
    """CenterPointNet output against encode targets. Backward on ['total'], log the rest.

    outputs['heatmap'] stays logits; focal applies the sigmoid itself.
    """
    heatmap = focal(outputs['heatmap'], targets['heatmap'])
    offset = masked_l1(outputs['offset'], targets['offset'], targets['mask'])
    radius = masked_l1(outputs['radius'], targets['radius'], targets['mask'])
    return {'total': heatmap + LAMBDA_OFFSET * offset + LAMBDA_RADIUS * radius,
            'heatmap': heatmap, 'offset': offset, 'radius': radius}
