import os
from ultralytics import YOLO

# v30: 928 source images (2784 after augmenting train only) vs v12's 131 —
# see RESEARCH.md / the redness analysis for why this version specifically.
DATA_YAML = 'Playground/data/datasets/bullet_holes/clean_v30/data.yaml'
# Ultralytics silently nests a relative `project=` path under its own global
# runs/ folder (~/AppData/Roaming/Ultralytics/settings.json's runs_dir),
# instead of the path you actually gave it — abspath sidesteps that entirely.
OUTPUT_DIR = os.path.abspath('Playground/output/yolo26_training')

# Fine-tuning a pretrained checkpoint, not training from scratch. Kept locally
# instead of letting ultralytics re-download it into the repo root each time.
BASE_MODEL = 'Playground/data/models/yolo26s.pt'

# v12's 342 train images -> ~75s/epoch; v30's 2784 (~8x more) -> ~590s/epoch,
# so 100 epochs here would be ~16 hours. The v12 run's own results.csv also
# showed its best epoch was 21 of 100 — the rest just overfit — so a much
# smaller budget plus real early stopping (patience, unset last time and thus
# effectively 100) is both faster and the better call, not just a compromise.
EPOCHS = 20
PATIENCE = 8
IMAGE_SIZE = 640
BATCH_SIZE = 16
DEVICE = 'xpu'   # the Intel integrated GPU; confirmed available via torch.xpu

# 0 sidesteps a Windows multiprocessing gotcha: a >0 worker count spawns
# dataloader subprocesses, which on Windows requires this script to be guarded
# by `if __name__ == '__main__':`.
WORKERS = 0

model = YOLO(BASE_MODEL)

model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    patience=PATIENCE,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    workers=WORKERS,
    project=OUTPUT_DIR,
    name='run_v30',
)
