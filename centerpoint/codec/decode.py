"""Model output -> detections. Inverse of codec/encode.py.

A cell is a detection if it is the largest in its own 3x3 neighbourhood. That is what
replaces box-IoU suppression, and why two touching holes stay two detections.
"""

import numpy as np

from centerpoint.codec.encode import STRIDE

SCORE_THRESHOLD = 0.3




def local_maxima(heatmap, threshold):
    """Cells that are the largest in their own 3x3 neighbourhood, as a boolean mask.

    The threshold is not optional: in a flat region every cell equals its own
    neighbourhood maximum, so an empty target yields ~66,000 peaks without it.
    """
    height, width = heatmap.shape
    padded = np.pad(heatmap, 1, constant_values=-np.inf)

    neighbourhood_max = np.full_like(heatmap, -np.inf)
    for row_shift in range(3):
        for col_shift in range(3):
            shifted = padded[row_shift:row_shift + height, col_shift:col_shift + width]
            np.maximum(neighbourhood_max, shifted, out=neighbourhood_max)

    # Ties are kept. Two holes one cell apart produce two adjacent cells at exactly 1.0;
    # a strict `>` would find neither greater than the other and drop both.
    return (heatmap == neighbourhood_max) & (heatmap > threshold)





def decode(heatmap, offset, radius, stride=STRIDE, threshold=SCORE_THRESHOLD):
    """Predicted maps -> (m, 4) of (x, y, r, score) in input pixels, best score first.

    Takes the (1,n,n), (2,n,n) and (1,n,n) maps encode produces. m may be 0.
    """
    heatmap = heatmap[0]
    is_peak = local_maxima(heatmap, threshold)
    rows, cols = np.nonzero(is_peak)

    scores = heatmap[rows, cols]

    # Adding the offset before scaling recovers the sub-pixel precision that encode lost
    # when it floored the centre into a cell.
    x = (cols + offset[0, rows, cols]) * stride
    y = (rows + offset[1, rows, cols]) * stride
    r = radius[0, rows, cols] * stride

    detections = np.stack([x, y, r, scores], axis=1).astype(np.float32)

    # Descending, because the eval metric matches greedily from the most confident down.
    return detections[np.argsort(-detections[:, 3])]
