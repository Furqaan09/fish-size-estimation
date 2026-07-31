import torch
from pathlib import Path
from src.dataset import FishDataset
from tests.conftest import needs_data

SEG = Path("data/Segmentation")


@needs_data
def test_dataset_load():
    ds = FishDataset(SEG / "test.csv", SEG)
    assert len(ds) > 0


@needs_data
def test_image_tensor_shape_and_range():
    ds = FishDataset(SEG / "test.csv", SEG)
    image, _ = ds[0]
    assert image.shape == (3, 256, 448)
    assert image.min() >= 0.0
    assert image.max() <= 1.0


@needs_data
def test_mask_is_binary_integers():
    ds = FishDataset(SEG / "test.csv", SEG)
    _, mask = ds[0]
    assert mask.shape == (256, 448)
    assert mask.dtype == torch.long
    assert set(torch.unique(mask).tolist()) <= {0, 1}


@needs_data
def test_image_and_mask_align():
    ds = FishDataset(SEG / "test.csv", SEG)
    image, mask = ds[0]
    assert image.shape[1:] == mask.shape
