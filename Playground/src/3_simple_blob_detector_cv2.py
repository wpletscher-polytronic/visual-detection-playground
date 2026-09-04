import cv2
import os
import numpy as np

IMAGES_DIR = 'Playground/data/images'
EXAMPLES_DIR = IMAGES_DIR + '/examples'
OUTPUT_DIR = 'Playground/output/blob_detection'


def get_detector():
    params = cv2.SimpleBlobDetector_Params()

    # blob size — tune these to your image resolution / actual hole size in pixels
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 5000

    params.filterByCircularity = True
    params.minCircularity = 0.7

    params.filterByConvexity = True
    params.minConvexity = 0.8

    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    # holes are dark against a lighter target — blobColor=0 keeps only dark blobs.
    # SimpleBlobDetector does its own internal multi-level thresholding, so no
    # separate cv2.inRange/threshold step is needed before calling detect().
    params.filterByColor = True
    params.blobColor = 255

    return cv2.SimpleBlobDetector_create(params)


def segment(img_name, detector):
    img = cv2.imread(IMAGES_DIR + '/examples/' + img_name)

    if img is None:
        print(f"Skipping {img_name} — could not load as image")
        return

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    os.makedirs(OUTPUT_DIR + '/gray', exist_ok=True)
    cv2.imwrite(OUTPUT_DIR + '/gray/' + img_name, img_gray)

    keypoints = detector.detect(img_gray)
    print(len(keypoints))

    for kp in keypoints:
        center = (int(kp.pt[0]), int(kp.pt[1]))
        radius = int(kp.size / 2)
        cv2.circle(img, center, radius, (0, 255, 0), 2)

    os.makedirs(OUTPUT_DIR + '/labelled', exist_ok=True)
    cv2.imwrite(OUTPUT_DIR + '/labelled/' + img_name, img)


detector = get_detector()

for img_name in os.listdir(EXAMPLES_DIR):
    segment(img_name, detector)