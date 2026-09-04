import os
from ultralytics import YOLO

# Points at the checkpoint from the interrupted run, not the base model — the
# checkpoint carries the full trainer state (optimizer, LR schedule position,
# epoch counter), so this continues at epoch 31 instead of restarting at 1.
# resume=True also reads args.yaml from the same run/ folder automatically, so
# data/epochs/batch/device etc. don't need to be repeated here.
OUTPUT_DIR = os.path.abspath('Playground/output/yolo26_training')
LAST_CHECKPOINT = f'{OUTPUT_DIR}/run/weights/last.pt'

model = YOLO(LAST_CHECKPOINT)
model.train(resume=True)
