"""STEP 2b — torch Dataset over datasets/bullet_rchsr/clean. Same splits as the baseline.

Functions / classes
- do NOT re-implement path handling or parsing. Both already exist:
    data/splits.py  list_split(root, split) -> [(stem, image_path, label_path)], with the
                    image/label integrity assertions already done
    data/labels.py  load_yolo_labels(path, w, h) -> (N, 3) of (cx, cy, r) in pixels
  Use list_split here, not load_split_geometry: this class decodes the image anyway, so
  take (w, h) from the decoded array rather than re-reading the header with PIL.

- class BulletHoleDataset(torch.utils.data.Dataset)
    __init__(root, split, img_size, transforms=None)
        split in {'train', 'valid', 'test'} — note the folder is 'valid', the yaml key 'val'.
    __len__() -> int
    __getitem__(i) -> dict:
        'image'     float tensor (3, H, W), ImageNet-normalised
        'holes'     float tensor (N, 3) = (cx, cy, r) in INPUT-IMAGE pixels
        'image_id'  str, for traceability in visualisations
      Deliberately returns raw geometry, NOT encoded targets. Encoding happens in
      codec/encode.py so it can be unit-tested without touching the Dataset.

- collate(batch) -> dict
    N varies per image, so holes cannot stack. Keep a list, or pad with a validity mask.

Invariants
- len(images) == len(labels), filenames pair up by stem
- after resize and augmentation, every centre is still inside the image; radii scale with it
- N == 0 is legal (an image with no holes) and must not crash collate or the loss
- no image is silently dropped: a load failure raises, never returns None

Verify by
- one test asserting shapes, dtypes and coordinate ranges on a single sample
- drawing circles from the Dataset output and confirming they match the analyze_labels
  overlays exactly. If they differ, the resize maths is wrong.
"""
