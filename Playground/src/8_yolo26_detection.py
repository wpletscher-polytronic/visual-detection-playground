import cv2
import os
from ultralytics import YOLO

IMAGES_DIR = 'Playground/data/images'
EXAMPLES_DIR = IMAGES_DIR + '/examples'
OUTPUT_DIR = 'Playground/output/yolo26_detection_v30_colab'

# Our own model — YOLO26s fine-tuned on the cleaned Project bat dataset via
# prepare_bullet_dataset.py + train_yolo26.py. Trained on Colab (see
# run_v30_colab), not locally — the local run_v30 was stopped early at epoch
# 16/20 once Colab proved faster, so it's a strictly worse, superseded partial
# run; this Colab checkpoint is the properly patience-converged one (best at
# epoch 36/44, higher mAP50-95 than either earlier run).
WEIGHTS = 'Playground/output/yolo26_training/run_v30_colab/weights/best.pt'

CONFIDENCE = 0.40   # drop predictions below this score
IOU = 0.30          # NMS IoU threshold

model = YOLO(WEIGHTS)


def save(stage, img_name, img):
    os.makedirs(OUTPUT_DIR + '/' + stage, exist_ok=True)
    cv2.imwrite(OUTPUT_DIR + '/' + stage + '/' + img_name, img)


def find_holes(img_name):
    """Return [(x, y, radius)] from our local model, in original pixel coords."""
    result = model.predict(IMAGES_DIR + '/examples/' + img_name,
                           conf=CONFIDENCE, iou=IOU, verbose=False)[0]

    holes = []
    for box in result.boxes.xywh:
        x, y, w, h = box.tolist()
        holes.append((x, y, max(w, h) / 2))
    return holes


def detect(img_name):
    img = cv2.imread(IMAGES_DIR + '/examples/' + img_name)
    holes = find_holes(img_name)
    print(f"{img_name:36s} {len(holes):2d} holes")

    for x, y, radius in holes:
        cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    save('labelled', img_name, img)


for img_name in sorted(os.listdir(EXAMPLES_DIR)):
    detect(img_name)
