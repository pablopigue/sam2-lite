"""FrameDataset must preprocess exactly like SAM 2 (checked against SAM 2's own loader)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from sam2.utils.misc import load_video_frames

from sam2lite.data.frames import FrameDataset

AUGMENT = {
    "crop_scale_min": 0.6,
    "hflip_p": 0.5,
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.2,
}
CPU = torch.device("cpu")


@pytest.fixture
def video_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    for i in range(2):  # small 16:9 frames with random content
        pixels = rng.integers(0, 256, size=(48, 85, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(tmp_path / f"{i:05d}.jpg")
    return tmp_path


def test_matches_sam2_preprocessing_without_augmentation(video_dir: Path) -> None:
    sam2_images, _, _ = load_video_frames(
        str(video_dir), image_size=1024, offload_video_to_cpu=True, compute_device=CPU
    )
    dataset = FrameDataset(sorted(str(p) for p in video_dir.glob("*.jpg")), image_size=1024)
    for i in range(len(dataset)):
        assert torch.equal(dataset[i], sam2_images[i])


def test_augmented_sample_shape_and_reproducibility(video_dir: Path) -> None:
    dataset = FrameDataset([str(video_dir / "00000.jpg")], image_size=1024, augment=AUGMENT)
    torch.manual_seed(0)
    first = dataset[0]
    torch.manual_seed(0)
    again = dataset[0]
    assert first.shape == (3, 1024, 1024) and first.dtype == torch.float32
    assert torch.equal(first, again)  # same seed -> same augmentation
