import cv2
import os
import numpy as np
import base64
import json
import urllib.request
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

IMAGES_DIR = 'Playground/data/images'
EXAMPLES_DIR = IMAGES_DIR + '/examples'
OUTPUT_DIR = 'Playground/output/roboflow_segmentation'

# "bullet" segmentation by Roboflow Universe — returns polygons, not boxes.
API_KEY = os.environ['ROBOFLOW_API_KEY']
MODEL_ID = 'bullet-ihpm1/2'

# This model is much less confident than the detection one: at 0.40 it returns
# nothing at all for easy.png, so the threshold sits far lower here.
CONFIDENCE = 0.10
OVERLAP = 0.30

# It also segments the target face itself, which we are not looking for.
HOLE_CLASS = 'bullet'


def save(stage, img_name, img):
    os.makedirs(OUTPUT_DIR + '/' + stage, exist_ok=True)
    cv2.imwrite(OUTPUT_DIR + '/' + stage + '/' + img_name, img)


def find_holes(img_name, shape):
    """Return [(x, y, radius)] plus the filled mask the polygons describe.

    The model gives one polygon per hole, so the circle comes from the polygon's
    minimum enclosing circle — the same step script 4 applies to its contours.
    """
    with open(IMAGES_DIR + '/examples/' + img_name, 'rb') as handle:
        payload = base64.b64encode(handle.read())

    query = urllib.parse.urlencode({
        'api_key': API_KEY,
        'confidence': CONFIDENCE,
        'overlap': OVERLAP,
    })
    request = urllib.request.Request(
        f'https://serverless.roboflow.com/{MODEL_ID}?{query}',
        data=payload,
        headers={'Content-Type': 'application/x-www-form-urlencoded'},
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.loads(response.read().decode('utf-8'))

    holes = []
    mask = np.zeros(shape[:2], np.uint8)

    for prediction in result['predictions']:
        if prediction['class'] != HOLE_CLASS:
            continue

        points = np.array([[p['x'], p['y']] for p in prediction['points']], np.int32)
        cv2.fillPoly(mask, [points], 255)

        (x, y), radius = cv2.minEnclosingCircle(points)
        holes.append((x, y, radius))

    return holes, mask


def detect(img_name):
    img = cv2.imread(IMAGES_DIR + '/examples/' + img_name)
    holes, mask = find_holes(img_name, img.shape)
    print(f"{img_name:36s} {len(holes):2d} holes")

    save('mask', img_name, mask)

    for x, y, radius in holes:
        cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    save('labelled', img_name, img)


for img_name in sorted(os.listdir(EXAMPLES_DIR)):
    detect(img_name)
