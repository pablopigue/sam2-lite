"""Video I/O and mask drawing of the demo (synthetic videos, no model or dataset needed)."""

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from sam2lite.demo import (
    PALETTE,
    overlay_mask,
    overlay_masks,
    read_video,
    write_mp4,
)


def make_video(path: Path, seconds: float, fps: int, size: tuple[int, int] = (64, 48)) -> None:
    """MPEG-4 Part 2 video (OpenCV can write it) whose frame i has brightness 8 * i (mod 256)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(int(seconds * fps)):
        writer.write(np.full((size[1], size[0], 3), (8 * i) % 256, dtype=np.uint8))
    writer.release()


def test_read_video_cuts_and_subsamples(tmp_path: Path) -> None:
    path = tmp_path / "in.mp4"
    make_video(path, seconds=3, fps=30)
    frames, fps = read_video(str(path), max_seconds=1, target_fps=10)
    assert fps == pytest.approx(10)  # step = 30 / 10 = 3
    assert len(frames) == 10  # first second only: 30 frames, every 3rd
    assert frames[0].shape == (48, 64, 3) and frames[0].dtype == np.uint8
    assert abs(frames[1].mean() - 24) < 6  # source frame 3 (= 8 * 3), up to codec loss


def test_read_video_downscales_keeping_aspect_ratio(tmp_path: Path) -> None:
    path = tmp_path / "big.mp4"
    make_video(path, seconds=0.5, fps=10, size=(320, 180))
    frames, _ = read_video(str(path), max_seconds=1, target_fps=10, max_side=160)
    assert frames[0].shape == (90, 160, 3)  # 320x180 -> 160x90
    small, _ = read_video(str(path), max_seconds=1, target_fps=10, max_side=1000)
    assert small[0].shape == (180, 320, 3)  # never upscaled


def test_read_video_rejects_unreadable_files(tmp_path: Path) -> None:
    path = tmp_path / "not_a_video.mp4"
    path.write_text("hello")
    with pytest.raises(ValueError):
        read_video(str(path), max_seconds=1, target_fps=10)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the system ffmpeg")
def test_write_mp4_is_h264_with_odd_sizes(tmp_path: Path) -> None:
    path = tmp_path / "out.mp4"
    frames = [np.zeros((47, 63, 3), dtype=np.uint8)] * 5  # odd sizes must be padded
    write_mp4(frames, fps=8, path=str(path))
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,pix_fmt", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert probe.stdout.strip() == "h264,yuv420p"  # what browsers play


def test_overlay_mask_blends_only_inside_the_mask() -> None:
    frame = np.full((2, 2, 3), 100, dtype=np.uint8)
    mask = np.array([[True, False], [False, False]])
    out = overlay_mask(frame, mask, color=(200, 0, 0))
    assert out[0, 0].tolist() == [150, 50, 50]  # 0.5 * 100 + 0.5 * color
    assert out[1, 1].tolist() == [100, 100, 100]
    assert frame[0, 0].tolist() == [100, 100, 100]  # input untouched


def test_overlay_masks_uses_one_colour_per_object() -> None:
    frame = np.zeros((1, 2, 3), dtype=np.uint8)
    masks = {1: np.array([[True, False]]), 2: np.array([[False, True]])}
    out = overlay_masks(frame, masks)
    assert out[0, 0].tolist() == [c // 2 for c in PALETTE[0]]  # 0.5 * colour on black
    assert out[0, 1].tolist() == [c // 2 for c in PALETTE[1]]
