"""Images and labels as tensors. Training targets are built in collate, not here.

__getitem__ returns raw geometry in input-image pixels, so codec/encode.py stays testable
without a Dataset and the stride stays with the entry point instead of the loader.

Pairing and parsing are not repeated here: data/splits.py owns the integrity checks and
data/labels.py owns the label format.
"""

import numpy as np
import torch
from PIL import Image

from centerpoint.codec.encode import encode
from centerpoint.data.labels import load_yolo_labels
from centerpoint.data.splits import list_split

# The statistics the torchvision weights were trained with. Stated literally so importing
# this module never has to touch a checkpoint.
IMAGENET_MEAN = np.float32([0.485, 0.456, 0.406])
IMAGENET_STD = np.float32([0.229, 0.224, 0.225])




class BulletHoleDataset(torch.utils.data.Dataset):
    """One split of a YOLO-layout dataset. No augmentation, no resize."""

    def __init__(self, root, split):
        self.entries = list_split(root, split)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        stem, image_path, label_path = self.entries[index]

        with Image.open(image_path) as handle:
            # convert('RGB') expands greyscale and drops alpha, and fixes the channel
            # order the pretrained backbone expects — cv2 would hand back BGR here.
            pixels = np.asarray(handle.convert('RGB'), dtype=np.float32) / 255.0

        height, width = pixels.shape[:2]
        normalised = (pixels - IMAGENET_MEAN) / IMAGENET_STD
        holes = load_yolo_labels(label_path, width, height)

        return {'image': torch.from_numpy(np.ascontiguousarray(normalised.transpose(2, 0, 1))),
                'holes': torch.from_numpy(holes.astype(np.float32)),
                'image_id': stem}




def collate(batch, stride):
    """Stack a batch and encode its targets. Bind stride with functools.partial.

    stride is an argument rather than params.STRIDE so a run's entry point owns it, the
    same way the model owns out_stride — the two must match for decode to read back.

    'holes' stays a list because N varies per image, and (0, 3) for an image with no
    holes is legal and must survive to the loss.
    """
    images = torch.stack([sample['image'] for sample in batch])

    height, width = images.shape[-2:]
    assert height == width, f"encode builds a square grid; got {height}x{width}"

    encoded = [encode(sample['holes'].numpy(), width, stride=stride) for sample in batch]
    targets = {key: torch.from_numpy(np.stack([target[key] for target in encoded]))
               for key in ('heatmap', 'offset', 'radius', 'mask')}

    return {'image': images,
            'targets': targets,
            'holes': [sample['holes'] for sample in batch],
            'image_id': [sample['image_id'] for sample in batch]}
