import os
from ultralytics import YOLO

DATA_YAML = 'Playground/data/datasets/bullet_holes/clean/data.yaml'
# Ultralytics silently nests a relative `project=` path under its own global
# runs/ folder (~/AppData/Roaming/Ultralytics/settings.json's runs_dir),
# instead of the path you actually gave it — abspath sidesteps that entirely.
OUTPUT_DIR = os.path.abspath('Playground/output/yolo26_training')

# Fine-tuning a pretrained checkpoint, not training from scratch — 342 training
# images is plenty for that, nowhere near enough for the latter. Kept locally
# instead of letting ultralytics re-download it into the repo root each time.
BASE_MODEL = 'Playground/data/models/yolo26s.pt'

EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 16
DEVICE = 'xpu'   # the Intel integrated GPU; confirmed available via torch.xpu

# 0 sidesteps a Windows multiprocessing gotcha: a >0 worker count spawns
# dataloader subprocesses, which on Windows requires this script to be guarded
# by `if __name__ == '__main__':`. 393 images is small enough that dataloading
# was never going to be the bottleneck anyway.
WORKERS = 0

model = YOLO(BASE_MODEL)

model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    workers=WORKERS,
    project=OUTPUT_DIR,
    name='run',
)
