"""STEP 6 — penalty-reduced focal loss for the heatmap, masked L1 for offset and radius.

Read this before writing it: the Gaussian is NOT a regression target.

From the CenterNet paper, eq. (1), and its reference implementation (`_neg_loss` in
CenterNet/src/lib/models/losses.py):

    positives = cells where Y == 1        exactly, nothing else
    negatives = every other cell, INCLUDING a cell at Y = 0.9
    negative penalty is scaled by (1 - Y)^beta

So a cell next to a centre is a negative whose penalty is nearly cancelled — the model is
trained to predict 0 there and barely punished when it does not. The bump is a
penalty-reduction map, not something to reproduce. Sigma controls how much slack a near
miss gets, not the shape the network should output.

Functions
- focal(pred, target) -> scalar
    pos: (1 - pred)^alpha * log(pred)                       where target == 1
    neg: (1 - target)^beta * pred^alpha * log(1 - pred)     elsewhere
    alpha=2, beta=4. Normalise by the number of positives, not by cell count, or the loss
    scale moves with the stride. Clamp pred away from 0 and 1 before the log.

- masked_l1(pred, target, mask) -> scalar
    Offset and radius are defined only where mask is 1. Divide by mask.sum(), never by the
    element count, or ~102,000 zeros drown the ~19 real values.

Invariants
- focal ~0 on a perfect prediction, large on an inverted one
- masked_l1 ignores everything outside the mask: changing values there must not move it
- the loss is finite when an image has zero holes (mask.sum() == 0 -> guard the divide)

Watch the stride. Stride 4 -> 2 takes positives-to-negatives from about 1:2,600 to
1:10,200 at the same hole count. If training behaves differently after a stride change,
compare the loss-term magnitudes before suspecting the architecture.
"""
