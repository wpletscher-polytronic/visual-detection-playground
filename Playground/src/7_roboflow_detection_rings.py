import cv2
import os
import base64
import json
import urllib.request
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

IMAGES_DIR = 'Playground/data/images'
EXAMPLES_DIR = IMAGES_DIR + '/examples'
OUTPUT_DIR = 'Playground/output/roboflow_detection_rings'

# "Bullet hole object detection" by Project bat — the Butt et al. 2023 model,
# the best published result in this niche (YOLOv8m, 96.7% mAP50).
#   https://universe.roboflow.com/project-bat-bullet-hole-detection/bullet-hole-object-detection
API_KEY = os.environ['ROBOFLOW_API_KEY']
MODEL_ID = 'bullet-hole-object-detection/12'

CONFIDENCE = 0.40   # drop predictions below this score
OVERLAP = 0.30      # server-side NMS IoU threshold

# This model classifies each hole by the ring it landed in — Bullet_1 to
# Bullet_10 — and also detects the target face itself, which we skip.
HOLE_CLASS_PREFIX = 'Bullet'


def save(stage, img_name, img):
    os.makedirs(OUTPUT_DIR + '/' + stage, exist_ok=True)
    cv2.imwrite(OUTPUT_DIR + '/' + stage + '/' + img_name, img)


def find_holes(img_name):
    """Return [(x, y, radius)] from the hosted model, in original pixel coords.

    Uses urllib rather than the inference-sdk so the script needs nothing beyond
    what the other scripts already use — that package has no Python 3.14 wheel.
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

    return [(p['x'], p['y'], max(p['width'], p['height']) / 2)
            for p in result['predictions']
            if p['class'].startswith(HOLE_CLASS_PREFIX)]


def detect(img_name):
    img = cv2.imread(IMAGES_DIR + '/examples/' + img_name)
    holes = find_holes(img_name)
    print(f"{img_name:36s} {len(holes):2d} holes")

    for x, y, radius in holes:
        cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    save('labelled', img_name, img)


for img_name in sorted(os.listdir(EXAMPLES_DIR)):
    detect(img_name)
