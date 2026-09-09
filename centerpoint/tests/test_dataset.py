"""Dataset and collate against fixtures written to a temp directory.

Small synthetic images rather than the real dataset: these tests must pass on a machine
that has never run prepare_bullet_*.py.
"""

from functools import partial

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from centerpoint.data.dataset import (IMAGENET_MEAN, IMAGENET_STD, BulletHoleDataset,
                                      collate)

IMG = 64
STRIDE = 4
GRID = IMG // STRIDE


def write_sample(root, stem, holes, colour=(0, 0, 0), split='train', size=IMG):
    """One image plus its YOLO label file. Coordinates are exact binary fractions of
    `size`, so the pixel -> normalised -> pixel round trip loses nothing."""
    images = root / split / 'images'
    labels = root / split / 'labels'
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    Image.fromarray(np.full((size, size, 3), colour, np.uint8)).save(images / f'{stem}.png')
    lines = [f"0 {cx / size} {cy / size} {2 * r / size} {2 * r / size}" for cx, cy, r in holes]
    (labels / f'{stem}.txt').write_text('\n'.join(lines))


def dataset_with(root, samples, colour=(0, 0, 0), split='train'):
    for stem, holes in samples:
        write_sample(root, stem, holes, colour=colour, split=split)
    return BulletHoleDataset(root, split)


# --- the image ------------------------------------------------------------------------

def test_channels_are_rgb_not_bgr(tmp_path):
    """Pure red: only channel 0 sits above its mean. BGR would put it in channel 2."""
    sample = dataset_with(tmp_path, [('a', [])], colour=(255, 0, 0))[0]
    red, green, blue = sample['image'][:, 0, 0]
    assert red > 0 and green < 0 and blue < 0


def test_normalisation_matches_imagenet_statistics(tmp_path):
    """A different value per channel, so a mean/std applied in the wrong order shows up."""
    colour = (255, 128, 0)
    sample = dataset_with(tmp_path, [('a', [])], colour=colour)[0]
    expected = (np.float32(colour) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    assert sample['image'][:, 5, 7].numpy() == pytest.approx(expected, rel=1e-6)


def test_image_is_channels_first_float32(tmp_path):
    sample = dataset_with(tmp_path, [('a', [])])[0]
    assert sample['image'].shape == (3, IMG, IMG)
    assert sample['image'].dtype == torch.float32


# --- the geometry ---------------------------------------------------------------------

def test_holes_survive_the_yolo_round_trip(tmp_path):
    """Written in pixels, stored normalised, read back in pixels. A wrong image size or a
    swapped axis here would silently move every centre."""
    written = [(30.0, 20.0, 6.0), (12.0, 44.0, 3.0)]
    sample = dataset_with(tmp_path, [('a', written)])[0]
    assert sample['holes'].numpy() == pytest.approx(np.array(written, np.float32))


def test_empty_label_file_gives_a_zero_by_three_tensor(tmp_path):
    sample = dataset_with(tmp_path, [('a', [])])[0]
    assert sample['holes'].shape == (0, 3)
    assert sample['holes'].dtype == torch.float32


def test_length_and_image_id_follow_the_split(tmp_path):
    dataset = dataset_with(tmp_path, [('a', []), ('b', []), ('c', [])])
    assert len(dataset) == 3
    assert sorted(dataset[i]['image_id'] for i in range(3)) == ['a', 'b', 'c']


# --- label contracts the encoder depends on ---------------------------------------------

def write_raw_label(root, stem, lines, split='train', size=IMG):
    """A label file written verbatim, for values write_sample could not express."""
    write_sample(root, stem, [], split=split, size=size)
    (root / split / 'labels' / f'{stem}.txt').write_text('\n'.join(lines))
    return BulletHoleDataset(root, split)


def test_centre_on_the_far_edge_is_rejected(tmp_path):
    """cx == 1.0 is one pixel past the last column. encode has no cell for it."""
    dataset = write_raw_label(tmp_path, 'a', ['0 1.0 0.5 0.05 0.05'])
    with pytest.raises(ValueError, match='not strictly inside'):
        dataset[0]


def test_centre_that_rounds_up_in_float32_is_rejected(tmp_path):
    """0.99999999 * 64 is inside the image in float64 and exactly 64.0 in float32, which
    is the dtype the Dataset hands to encode. Checking in float64 alone would let it by."""
    normalised = 0.99999999
    assert normalised * IMG < IMG and np.float32(normalised * IMG) == IMG
    dataset = write_raw_label(tmp_path, 'a', [f'0 {normalised} 0.5 0.05 0.05'])
    with pytest.raises(ValueError, match='not strictly inside'):
        dataset[0]


@pytest.mark.parametrize('line', ['0 0.5 0.5 nan 0.05',
                                  '0 nan 0.5 0.05 0.05',
                                  '0 0.5 0.5 inf 0.05'])
def test_non_finite_values_are_rejected_with_path_and_line(tmp_path, line):
    """A nan width fails every `w <= 0.0` comparison, so without this it reaches encode."""
    dataset = write_raw_label(tmp_path, 'a', ['0 0.25 0.25 0.05 0.05', line])
    with pytest.raises(ValueError, match=r'a\.txt:2: non-finite'):
        dataset[0]


def test_a_centre_just_inside_the_edge_is_still_accepted(tmp_path):
    """The bound is strict, not a blanket rejection of the last pixel."""
    dataset = write_raw_label(tmp_path, 'a', [f'0 {63.5 / IMG} {63.5 / IMG} 0.05 0.05'])
    assert dataset[0]['holes'][0, :2].numpy() == pytest.approx([63.5, 63.5])


# --- collate --------------------------------------------------------------------------

def batch_of(tmp_path, samples):
    dataset = dataset_with(tmp_path, samples)
    return collate([dataset[i] for i in range(len(dataset))], stride=STRIDE)


def test_collate_target_shapes_and_dtypes(tmp_path):
    targets = batch_of(tmp_path, [('a', [(30.0, 20.0, 6.0)]), ('b', [])])['targets']
    assert targets['heatmap'].shape == (2, 1, GRID, GRID)
    assert targets['offset'].shape == (2, 2, GRID, GRID)
    assert targets['radius'].shape == (2, 1, GRID, GRID)
    assert targets['mask'].shape == (2, 1, GRID, GRID)
    for name, tensor in targets.items():
        assert tensor.dtype == torch.float32, name


def test_collate_encodes_each_sample_at_its_own_centre(tmp_path):
    """Also pins row/column order: cy indexes the row, cx the column."""
    targets = batch_of(tmp_path, [('a', [(30.0, 20.0, 6.0)]), ('b', [])])['targets']
    assert targets['heatmap'][0, 0, 20 // STRIDE, 30 // STRIDE] == 1.0
    assert targets['mask'][0].sum() == 1
    assert targets['mask'][1].sum() == 0


def test_collate_encodes_fractional_offsets_and_radius(tmp_path):
    """Sub-cell geometry end to end, which is the whole reason the offset head exists.
    At stride 4 a centre at (30.5, 20.25) floors into row 5, column 7, leaving offsets
    (0.625, 0.0625), and r = 6.5 px is 1.625 cells."""
    batch = batch_of(tmp_path, [('a', [(30.5, 20.25, 6.5)])])
    targets = batch['targets']
    assert batch['holes'][0].numpy() == pytest.approx(np.float32([[30.5, 20.25, 6.5]]))
    assert targets['heatmap'][0, 0, 5, 7].item() == 1.0
    assert targets['offset'][0, 0, 5, 7].item() == pytest.approx(0.625)
    assert targets['offset'][0, 1, 5, 7].item() == pytest.approx(0.0625)
    assert targets['radius'][0, 0, 5, 7].item() == pytest.approx(1.625)


def test_collate_keeps_variable_hole_counts_as_a_list(tmp_path):
    batch = batch_of(tmp_path, [('a', [(30.0, 20.0, 6.0), (12.0, 44.0, 3.0)]),
                                ('b', []),
                                ('c', [(8.0, 8.0, 2.0)])])
    assert [tuple(holes.shape) for holes in batch['holes']] == [(2, 3), (0, 3), (1, 3)]


def test_collate_preserves_raw_geometry_alongside_the_targets(tmp_path):
    written = [(30.0, 20.0, 6.0)]
    batch = batch_of(tmp_path, [('a', written)])
    assert batch['holes'][0].numpy() == pytest.approx(np.array(written, np.float32))
    assert batch['image_id'] == ['a']


# --- through a DataLoader ---------------------------------------------------------------

def loader_for(tmp_path, count, **kwargs):
    samples = [(f's{i}', [(8.0 + 8 * j, 12.0 + 4 * j, 3.0) for j in range(i % 3)])
               for i in range(count)]
    dataset = dataset_with(tmp_path, samples)
    return DataLoader(dataset, collate_fn=partial(collate, stride=STRIDE), **kwargs)


def test_dataloader_produces_a_complete_batch(tmp_path):
    batch = next(iter(loader_for(tmp_path, 5, batch_size=2)))
    assert batch['image'].shape == (2, 3, IMG, IMG)
    assert batch['targets']['heatmap'].shape == (2, 1, GRID, GRID)
    assert len(batch['holes']) == 2 and len(batch['image_id']) == 2


def test_dataloader_with_worker_processes(tmp_path):
    """Windows spawns workers rather than forking, so the Dataset and the partial-bound
    collate both have to pickle. This is the cheapest place to find out that they do not."""
    batches = list(loader_for(tmp_path, 4, batch_size=2, num_workers=2))
    assert sum(batch['image'].shape[0] for batch in batches) == 4
