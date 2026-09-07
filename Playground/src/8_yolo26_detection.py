import cv2
import os
from ultralytics import YOLO

EXAMPLES_DIR = 'datasets/examples'
OUTPUT_DIR = 'Playground/output/yolo26_detection_rchsr_colab'

# Our own model — YOLO26s fine-tuned on cdmstrong/bullet-rchsr via
# prepare_bullet_rchsr_dataset.py + train_yolo26.py (Colab, run_rchsr_colab).
# Best at epoch 40/48, patience=8 genuinely triggered on a real plateau this
# time — mAP50 0.580, mAP50-95 0.188, notably lower than v30's 0.993/0.586
# despite far more data, consistent with this dataset being much denser
# (~19 boxes/image vs v30's ~5) and more heterogeneous (scraped stock photos,
# watermarks, many countries) — harder to fit perfectly, which is the same
# trade we were making for hopefully-better generalization to novel images.
WEIGHTS = 'Playground/output/yolo26_training/run_rchsr_colab/weights/best.pt'

CONFIDENCE = 0.40   # drop predictions below this score
IOU = 0.30          # NMS IoU threshold

model = YOLO(WEIGHTS)


def save(stage, img_name, img):
    os.makedirs(OUTPUT_DIR + '/' + stage, exist_ok=True)
    cv2.imwrite(OUTPUT_DIR + '/' + stage + '/' + img_name, img)


def find_holes(img_name):
    """Return [(x, y, radius)] from our local model, in original pixel coords."""
    result = model.predict(EXAMPLES_DIR + '/' + img_name,
                           conf=CONFIDENCE, iou=IOU, verbose=False)[0]

    holes = []
    for box in result.boxes.xywh:
        x, y, w, h = box.tolist()
        holes.append((x, y, max(w, h) / 2))
    return holes


def detect(img_name):
    img = cv2.imread(EXAMPLES_DIR + '/' + img_name)
    holes = find_holes(img_name)
    print(f"{img_name:36s} {len(holes):2d} holes")

    for x, y, radius in holes:
        cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    save('labelled', img_name, img)


for img_name in sorted(os.listdir(EXAMPLES_DIR)):
    detect(img_name)
