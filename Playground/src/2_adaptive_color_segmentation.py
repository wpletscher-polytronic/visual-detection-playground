import cv2
import os
import numpy as np

IMAGES_DIR = 'Playground/data/images'
EXAMPLES_DIR = IMAGES_DIR + '/examples'
OUTPUT_DIR = 'Playground/output/adaptive_color_segmentation'


def segment(img_name):
    img = cv2.imread(IMAGES_DIR + '/examples/' + img_name)

    if img is None:
        print(f"Skipping {img_name} — could not load as image")
        return

    # Contrast normalization before thresholding
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)

    os.makedirs(OUTPUT_DIR + '/clahe', exist_ok=True)
    cv2.imwrite(OUTPUT_DIR + '/clahe/' + img_name, img_clahe)

    # Adaptive threshold (Otsu) instead of fixed inRange values
    _, img_thresholded = cv2.threshold(
        img_clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    os.makedirs(OUTPUT_DIR + '/thresholded', exist_ok=True)
    cv2.imwrite(OUTPUT_DIR + '/thresholded/' + img_name, img_thresholded)

    # --- CHANGE: much smaller kernel, used only for minor speckle cleanup ---
    # A 10x10 opening was eating through the ring/crosshair line strokes and
    # fragmenting the target blob. Holes are background notches *inside* the
    # blob, not isolated foreground specks, so opening was never the right
    # tool to isolate them — hierarchy (below) does that job instead.
    kernel = np.ones((3, 3), np.uint8)
    img_opening = cv2.morphologyEx(img_thresholded, cv2.MORPH_OPEN, kernel)

    os.makedirs(OUTPUT_DIR + '/kerneled', exist_ok=True)
    cv2.imwrite(OUTPUT_DIR + '/kerneled/' + img_name, img_opening)

    contours, hierarchy = cv2.findContours(img_opening.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print(len(contours))

    # --- CHANGE: use hierarchy to keep only "hole" contours ---
    # hierarchy[0][i] = [next, previous, first_child, parent]
    # A contour with parent != -1 is nested inside another contour, i.e. it's
    # a background-colored region enclosed within the white target blob —
    # exactly what a bullet hole looks like in this mask.
    MIN_AREA = 20
    MIN_CIRCULARITY = 0.6

    hole_count = 0

    if hierarchy is not None:
        hierarchy = hierarchy[0]  # unwrap the extra dimension OpenCV adds

        for i, contour in enumerate(contours):
            parent = hierarchy[i][3]
            if parent == -1:
                continue  # no parent -> this is an outer/top-level contour, not a hole

            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if area < MIN_AREA or perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < MIN_CIRCULARITY:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(img, center, radius, (0, 255, 0), 2)

            hole_count += 1

    print(f"  -> {hole_count} candidate holes after hierarchy + shape filtering")

    os.makedirs(OUTPUT_DIR + '/labelled', exist_ok=True)
    cv2.imwrite(OUTPUT_DIR + '/labelled/' + img_name, img)


for img_name in os.listdir(EXAMPLES_DIR):
    segment(img_name)