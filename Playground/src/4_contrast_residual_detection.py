import cv2
import os
import numpy as np

IMAGES_DIR = 'Playground/data/images'
EXAMPLES_DIR = IMAGES_DIR + '/examples'
OUTPUT_DIR = 'Playground/output/contrast_residual_detection'

# All lengths are fractions of min(height, width): the target fills most of the
# frame in every example, so this is what makes one setting work across
# resolutions from 355px to 1000px. Values come from a multi-start coordinate
# ascent on the ground truth above (see the note at the bottom of INFO.md).
BACKGROUND_SCALE = 0.13    # median window; must be wider than a hole
STROKE_SCALE = 0.014       # opening radius; must exceed the widest print stroke
CLOSE_SCALE = 0.008        # bridges the ragged rim of a hole into one blob
MIN_RADIUS_SCALE = 0.004
MAX_RADIUS_SCALE = 0.055

THRESHOLD_FRACTION = 0.55  # of Otsu, to keep low-contrast holes on the bull
SPLIT_AREA_FACTOR = 0.4    # of max hole area, above which we try to split
PEAK_FRACTION = 0.70       # distance-transform peak height for a split marker
MIN_CIRCULARITY = 0.45
MIN_SOLIDITY = 0.72


def save(stage, img_name, img):
    os.makedirs(OUTPUT_DIR + '/' + stage, exist_ok=True)
    cv2.imwrite(OUTPUT_DIR + '/' + stage + '/' + img_name, img)


def odd(value, minimum=3):
    return max(minimum, int(value) | 1)


def ellipse(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def residual(img, scale):
    """Local-contrast residual that is blind to polarity.

    A hole takes its colour from whatever sits behind the target, so within a
    single image it can be darker than the paper *and* lighter than the bull.
    A median over a window wider than a hole reconstructs the printed
    background as if the hole were not there, so the absolute difference
    responds to "differs from its local surroundings" in either direction.

    Median rather than a morphological top-hat/black-hat pair: the hat
    transforms need a structuring element bigger than a hole, but in easy.png
    the rings sit closer together than a hole is wide, so such an element also
    erases the gaps between rings and lights up the whole target face.
    """
    background = cv2.medianBlur(img, odd(scale * BACKGROUND_SCALE))
    return cv2.absdiff(img, background)


def split_blob(mask):
    """Separate touching shots inside one connected component.

    Distance-transform peaks give one marker per shot and watershed grows them
    back out. PEAK_FRACTION is high so that a single ragged hole keeps exactly
    one maximum rather than being torn into pieces.
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return [mask]

    _, peaks = cv2.threshold(dist, PEAK_FRACTION * dist.max(), 255, cv2.THRESH_BINARY)
    peaks = np.uint8(peaks)

    count, markers = cv2.connectedComponents(peaks)
    if count <= 2:  # background + one peak -> nothing to split
        return [mask]

    # Watershed needs a 3-channel image, and the pixels it must assign set to 0.
    markers = markers + 1
    markers[cv2.subtract(mask, peaks) == 255] = 0
    cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), markers)

    return [np.uint8(markers == label) * 255 for label in range(2, count + 1)]


def accept(contour, min_area, max_area):
    """Shape gate: keeps ragged holes, rejects leftover print fragments."""
    area = cv2.contourArea(contour)
    if not (min_area < area < max_area):
        return False

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False

    if 4 * np.pi * area / (perimeter ** 2) < MIN_CIRCULARITY:
        return False

    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area == 0 or area / hull_area < MIN_SOLIDITY:
        return False

    return True


def find_holes(img_gray, img_name=None):
    """Return [(x, y, radius)] for every detected hole."""
    scale = min(img_gray.shape)
    min_area = np.pi * (scale * MIN_RADIUS_SCALE) ** 2
    max_area = np.pi * (scale * MAX_RADIUS_SCALE) ** 2

    img_residual = residual(img_gray, scale)

    # Otsu sits too high when a few high-contrast holes dominate the histogram,
    # which loses the faint ones on the black bull, so cut at a fraction of it.
    otsu, _ = cv2.threshold(img_residual, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(img_residual, max(6.0, otsu * THRESHOLD_FRACTION),
                            255, cv2.THRESH_BINARY)

    # Ring lines and digit strokes are narrower than a hole, so an opening at
    # stroke scale removes them; the closing then seals a hole's ragged rim.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse(odd(scale * STROKE_SCALE)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ellipse(odd(scale * CLOSE_SCALE)))

    # A hole photographs as a bright torn rim around a differently-coloured
    # centre, so the mask is often an annulus. Fill it to get one solid blob.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, contours, -1, 255, -1)

    if img_name is not None:
        save('residual', img_name, img_residual)
        save('mask', img_name, mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    candidates = []
    for contour in contours:
        if cv2.contourArea(contour) <= SPLIT_AREA_FACTOR * max_area:
            candidates.append(contour)
            continue

        component = np.zeros_like(mask)
        cv2.drawContours(component, [contour], -1, 255, -1)
        for piece in split_blob(component):
            pieces, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            candidates.extend(pieces)

    holes = []
    for contour in candidates:
        if not accept(contour, min_area, max_area):
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        holes.append((x, y, radius))

    return holes


def detect(img_name):
    img = cv2.imread(IMAGES_DIR + '/examples/' + img_name)

    if img is None:
        print(f"Skipping {img_name} — could not load as image")
        return

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    holes = find_holes(img_gray, img_name)
    print(f"{img_name:36s} {len(holes):2d} holes")

    for x, y, radius in holes:
        cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    save('labelled', img_name, img)


for img_name in sorted(os.listdir(EXAMPLES_DIR)):
    detect(img_name)
