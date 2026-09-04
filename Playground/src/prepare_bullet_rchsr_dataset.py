import hashlib
import os
import shutil
import yaml
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()

# "bullet" by cdmstrong — a much more general dataset than Project bat: 3362
# images, single "Bullet Hole" class already (no ring-score/Target classes to
# collapse), one version, no augmentation multiplier applied at export.
#   https://universe.roboflow.com/cdmstrong-iocyj/bullet-rchsr
#
# Uses Roboflow's own train/valid/test split as-is, trusting the author rather
# than reassigning it ourselves. We tried redistributing to a cleaner 80/10/10
# and found real leakage from Roboflow's own augmented variants sharing a
# filename prefix (293 of 658 source-photo groups spanned multiple splits) —
# fixed that, but then found some duplicate/near-duplicate photos don't even
# share a prefix, which filename-based grouping can't catch at all. Properly
# solving that needs perceptual image-similarity hashing, well beyond what
# this script does; reverted to the simple, trust-the-source approach instead.
API_KEY = os.environ['ROBOFLOW_API_KEY']
WORKSPACE = 'cdmstrong-iocyj'
PROJECT = 'bullet-rchsr'
VERSION = 1

RAW_DIR = 'Playground/data/datasets/bullet_rchsr/raw'
CLEAN_DIR = 'Playground/data/datasets/bullet_rchsr/clean'
SPLITS = ('train', 'valid', 'test')

# Already just one class, so nothing needs excluding — kept as a set (not
# skipped entirely) so this stays the same shape as prepare_bullet_dataset.py.
NON_HOLE_CLASSES = set()
HOLE_CLASS_NAME = 'bullet_hole'


def download():
    if os.path.isdir(RAW_DIR) and os.listdir(RAW_DIR):
        print(f"Raw dataset already present at {RAW_DIR}, skipping download.")
        return
    Roboflow(api_key=API_KEY).workspace(WORKSPACE).project(PROJECT) \
        .version(VERSION).download('yolov8', location=RAW_DIR)


def file_hash(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def find_duplicates():
    """Map content hash -> [(split, filename), ...] for every hash seen twice.

    Byte-identical only. Roboflow's own augmented copies are pixel-different
    on purpose and are correctly left alone here.
    """
    seen = {}
    for split in SPLITS:
        img_dir = os.path.join(RAW_DIR, split, 'images')
        if not os.path.isdir(img_dir):
            continue
        for name in os.listdir(img_dir):
            digest = file_hash(os.path.join(img_dir, name))
            seen.setdefault(digest, []).append((split, name))
    return {h: locs for h, locs in seen.items() if len(locs) > 1}


def label_path(root, split, image_name):
    stem = os.path.splitext(image_name)[0]
    return os.path.join(root, split, 'labels', stem + '.txt')


def copy_clean_split(skip):
    """Copy every non-duplicate image + its label into CLEAN_DIR, unchanged."""
    for split in SPLITS:
        img_dir = os.path.join(RAW_DIR, split, 'images')
        if not os.path.isdir(img_dir):
            continue
        os.makedirs(os.path.join(CLEAN_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(CLEAN_DIR, split, 'labels'), exist_ok=True)

        for name in os.listdir(img_dir):
            if (split, name) in skip:
                continue
            shutil.copy2(os.path.join(img_dir, name),
                        os.path.join(CLEAN_DIR, split, 'images', name))

            src_label = label_path(RAW_DIR, split, name)
            dst_label = label_path(CLEAN_DIR, split, name)
            if os.path.isfile(src_label):
                shutil.copy2(src_label, dst_label)
            else:
                open(dst_label, 'w').close()  # image had no boxes at all


def rewrite_labels(class_names):
    """Collapse every hole class to id 0, drop any excluded class's lines."""
    old_to_new = {i: (None if name in NON_HOLE_CLASSES else 0)
                  for i, name in enumerate(class_names)}

    empty_count = 0
    for split in SPLITS:
        label_dir = os.path.join(CLEAN_DIR, split, 'labels')
        if not os.path.isdir(label_dir):
            continue

        for name in os.listdir(label_dir):
            path = os.path.join(label_dir, name)
            with open(path) as handle:
                lines = handle.readlines()

            kept = []
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                new_id = old_to_new.get(int(parts[0]))
                if new_id is None:
                    continue  # excluded class — drop it
                kept.append(' '.join([str(new_id), *parts[1:]]))

            if not kept:
                empty_count += 1
            with open(path, 'w') as handle:
                handle.write('\n'.join(kept) + ('\n' if kept else ''))

    if empty_count:
        print(f"{empty_count} image(s) ended up with zero boxes"
              f" after dropping excluded-class labels.")


def write_data_yaml():
    """Rewrite data.yaml for the single-class dataset. Returns the old class list."""
    with open(os.path.join(RAW_DIR, 'data.yaml')) as handle:
        data = yaml.safe_load(handle)

    class_names = data['names']
    data['nc'] = 1
    data['names'] = [HOLE_CLASS_NAME]
    for split in SPLITS:
        key = 'val' if split == 'valid' else split
        if key in data:
            data[key] = f'../{split}/images'

    with open(os.path.join(CLEAN_DIR, 'data.yaml'), 'w') as handle:
        yaml.safe_dump(data, handle, sort_keys=False)

    return class_names


def dedupe_plan(duplicates):
    """Decide which copy of each duplicate to keep.

    A duplicate that spans splits is a leakage risk (the model would train on
    an image it is later "tested" against), so the held-out splits win: keep
    the test/valid copy and drop the train copy, not the other way round.
    """
    priority = ('test', 'valid', 'train')
    skip = set()

    for digest, locs in duplicates.items():
        ranked = sorted(locs, key=lambda loc: priority.index(loc[0]))
        keep, *drop = ranked
        skip.update(drop)
        if len({split for split, _ in locs}) > 1:
            print(f"  cross-split duplicate: keeping {keep}, dropping {drop}")

    return skip


download()

duplicates = find_duplicates()
print(f"Found {len(duplicates)} duplicate group(s) among the raw images.")
skip = dedupe_plan(duplicates)
print(f"Dropping {len(skip)} duplicate file(s); copying the rest to {CLEAN_DIR}.")

copy_clean_split(skip)
class_names = write_data_yaml()
rewrite_labels(class_names)

print(f"Clean single-class dataset ready at {CLEAN_DIR}")
