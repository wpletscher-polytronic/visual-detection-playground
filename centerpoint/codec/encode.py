"""Ground-truth geometry -> training targets. Inverse of codec/decode.py.

offset and radius are only meaningful where mask is 1.
"""

import numpy as np

STRIDE = 2          # 320x320 grid for a 640 px image
SIGMA_SCALE = 0.5   # sigma = SIGMA_SCALE * radius_cells, floored at SIGMA_MIN
SIGMA_MIN = 1.0     # below one cell the bump is a lone pixel with no gradient to descend
TRUNCATE = 3.0      # splat this many sigma; past it the bump is under 0.011




def sigma_for(radius_cells):
    return max(SIGMA_SCALE * radius_cells, SIGMA_MIN)




def gaussian(dist_sq, sigma):
    """Bump shape; edit here to change it. 1.0 at d=0, 0.61 at sigma, 0.011 at 3*sigma."""
    return np.exp(-dist_sq / (2 * sigma ** 2))




def encode(holes, img_size, stride=STRIDE):
    """Build the training targets for one image.

    holes is (N, 3) of (cx, cy, r) in pixels. Returns float32 heatmap (1,n,n),
    offset (2,n,n), radius (1,n,n) and mask (1,n,n), with n = img_size // stride.

    Raises if a centre falls outside the image: whatever moved it must drop it, since
    dropping it here would lose training signal with no error.
    """
    assert (img_size % stride == 0), (f"img_size {img_size} not divisible by stride {stride}")
    assert (holes[:, :2] >= 0).all() and (holes[:, :2] < img_size).all(), ("hole centres must lie inside the image; filter them before encoding")

    n = img_size // stride
    heatmap = np.zeros((1, n, n), np.float32)
    offset = np.zeros((2, n, n), np.float32)
    radius = np.zeros((1, n, n), np.float32)
    mask = np.zeros((1, n, n), np.float32)

    for cx, cy, r in holes:
        x, y = cx / stride, cy / stride
        col, row = int(x), int(y)          # centres are non-negative, so int is floor

        sigma = sigma_for(r / stride)
        reach = int(TRUNCATE * sigma) + 1
        x0, x1 = max(0, col - reach), min(n, col + reach + 1)
        y0, y1 = max(0, row - reach), min(n, row + reach + 1)

        dx = np.arange(x0 - col, x1 - col, dtype=np.float32)
        dy = np.arange(y0 - row, y1 - row, dtype=np.float32)
        dist_sq = dy[:, None] ** 2 + dx ** 2
        bump = gaussian(dist_sq, sigma)

        # Maximum, not sum: two overlapping holes must stay two peaks.
        np.maximum(heatmap[0, y0:y1, x0:x1], bump, out=heatmap[0, y0:y1, x0:x1])

        offset[0, row, col] = x - col
        offset[1, row, col] = y - row
        radius[0, row, col] = r / stride
        mask[0, row, col] = 1.0

    return {'heatmap': heatmap, 'offset': offset, 'radius': radius, 'mask': mask}
