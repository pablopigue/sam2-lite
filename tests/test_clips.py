"""MemoryClipDataset on a synthetic DAVIS layout (no dataset needed)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sam2lite.data.clips import MemoryClipDataset, prepare_mask
from sam2lite.data.davis import save_annotation

SIZE = 64  # small model image size keeps the test fast
PALETTE = [0, 0, 0, 128, 0, 0, 0, 128, 0] + [0] * (256 * 3 - 9)


@pytest.fixture
def davis(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    for video, n_frames in [("short", 3), ("long", 10)]:
        jpg = tmp_path / "JPEGImages" / "480p" / video
        png = tmp_path / "Annotations" / "480p" / video
        jpg.mkdir(parents=True)
        for i in range(n_frames):
            pixels = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(jpg / f"{i:05d}.jpg")
            ids = np.zeros((24, 32), dtype=np.uint8)
            ids[2:8, 2:10] = 1  # object 1 on the left
            ids[10:20, 20:30] = 2  # object 2 on the right
            save_annotation(png / f"{i:05d}.png", ids, PALETTE)
    return tmp_path


def test_train_clips_shapes_and_skips_short_videos(davis: Path) -> None:
    ds = MemoryClipDataset(davis, ["short", "long"], clip_len=4, image_size=SIZE, stride=3)
    assert {video for video, _, _ in ds.items} == {"long"}  # "short" has fewer than 4 frames
    assert [start for _, start, _ in ds.items] == [0, 3, 6]
    frames, mask = ds[0]
    assert frames.shape == (4, 3, SIZE, SIZE) and mask.shape == (1, 1, SIZE, SIZE)
    assert set(mask.unique().tolist()) <= {0.0, 1.0} and mask.sum() > 0


def test_val_clips_enumerate_every_object(davis: Path) -> None:
    ds = MemoryClipDataset(davis, ["long"], clip_len=4, image_size=SIZE, train=False)
    assert ds.items == [("long", 0, 1), ("long", 0, 2)]
    _, mask_1 = ds[0]
    _, mask_2 = ds[1]
    assert mask_1[..., :, : SIZE // 2].sum() > 0 and mask_1[..., :, SIZE // 2 :].sum() == 0
    assert mask_2[..., :, SIZE // 2 :].sum() > 0


def test_flip_is_applied_to_frames_and_mask_together(davis: Path) -> None:
    ds = MemoryClipDataset(davis, ["long"], clip_len=4, image_size=SIZE, train=False)
    frames, mask = ds[0]
    flipped = MemoryClipDataset(davis, ["long"], clip_len=4, image_size=SIZE, hflip_p=1.0)
    flipped.items = ds.items  # same clip and object, but with the train-time flip
    f_frames, f_mask = flipped[0]
    torch.testing.assert_close(f_frames, frames.flip(-1))
    torch.testing.assert_close(f_mask, mask.flip(-1))


def test_prepare_mask_matches_add_new_mask_rule() -> None:
    mask = np.zeros((48, 64), dtype=bool)
    mask[10:30, 20:40] = True
    out = prepare_mask(mask, SIZE)
    assert out.shape == (1, 1, SIZE, SIZE) and set(out.unique().tolist()) == {0.0, 1.0}
