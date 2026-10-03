"""Click-to-track logic shared by the Gradio app and the API (no UI code here).

read_video -> load_tracker -> track (one click on the first frame) -> overlay_mask -> write_mp4.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download
from omegaconf import DictConfig
from sam2.sam2_video_predictor import SAM2VideoPredictor

from sam2lite.export.bundle import load_bundle


def read_video(
    path: str, max_seconds: float, target_fps: float, max_side: int | None = None
) -> tuple[list[np.ndarray], float]:
    """RGB frames (H, W, 3, uint8) of the first `max_seconds`, subsampled to ~`target_fps`.

    With `max_side`, larger frames are downscaled (aspect ratio kept) so a 4K upload neither
    fills the memory nor produces a huge output video; the model resizes to 576/1024 anyway.

    Returns the frames and their actual frame rate (source fps / step).
    """
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError(f"cannot open video {path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0  # some containers do not report it
    step = max(1, round(source_fps / target_fps))
    max_frames = int(max_seconds * source_fps)
    frames, index = [], 0
    while index < max_frames:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        if index % step == 0:
            frames.append(cv2.cvtColor(_fit(frame_bgr, max_side), cv2.COLOR_BGR2RGB))
        index += 1
    capture.release()
    if not frames:
        raise ValueError(f"no frames could be read from {path}")
    return frames, source_fps / step


def _fit(frame: np.ndarray, max_side: int | None) -> np.ndarray:
    """Downscale so the longer side is at most `max_side` (INTER_AREA: best for shrinking)."""
    height, width = frame.shape[:2]
    if max_side is None or max(height, width) <= max_side:
        return frame
    scale = max_side / max(height, width)
    size = (round(width * scale), round(height * scale))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def load_tracker(cfg: DictConfig, model: str) -> SAM2VideoPredictor:
    """sam2-lite or sam2-lite-mobile (ONNX encoder), from the local bundle or the Hub."""
    spec = cfg.models[model]
    torch.set_num_threads(cfg.threads)  # before building: the ONNX session copies this value
    if Path(cfg.local_bundle).exists():
        bundle_dir, onnx_dir = Path(cfg.local_bundle), Path(cfg.local_onnx_dir)
    else:  # private repo: the token comes from HF_TOKEN (Space secret or .env), never the code
        bundle_dir = onnx_dir = Path(
            snapshot_download(cfg.repo_id, token=os.environ.get("HF_TOKEN"))
        )
    return load_bundle(bundle_dir, image_size=spec.image_size, onnx=str(onnx_dir / spec.onnx))


@torch.inference_mode()
def track(
    predictor: SAM2VideoPredictor, frames: list[np.ndarray], point_xy: tuple[float, float]
) -> list[np.ndarray]:
    """Boolean mask (H, W) per frame for the object under `point_xy` (pixels, first frame)."""
    with tempfile.TemporaryDirectory() as tmp:
        # SAM 2 reads an .mp4 only through decord (not installed) or a folder of JPEGs.
        for i, frame in enumerate(frames):
            cv2.imwrite(f"{tmp}/{i:05d}.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        state = predictor.init_state(video_path=tmp)
    predictor.add_new_points_or_box(
        state,
        frame_idx=0,
        obj_id=1,
        points=[list(point_xy)],
        labels=[1],  # 1 = foreground
    )
    masks = [np.zeros(frames[0].shape[:2], dtype=bool)] * len(frames)
    for frame_idx, _, mask_logits in predictor.propagate_in_video(state):
        masks[frame_idx] = (mask_logits[0, 0] > 0).cpu().numpy()
    return masks


def overlay_mask(
    frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """Copy of `frame` (H, W, 3, uint8) with `color` alpha-blended (0.5) on the boolean mask."""
    out = frame.copy()
    # A convex combination of two values in [0, 255] stays in [0, 255].
    out[mask] = (0.5 * out[mask] + 0.5 * np.array(color)).astype(np.uint8)
    return out


def write_mp4(frames: list[np.ndarray], fps: float, path: str) -> None:
    """H.264 mp4 that browsers can play, encoded by the system ffmpeg (libx264).

    OpenCV's pip wheels cannot encode H.264 (only MPEG-4 Part 2, which browsers do not play),
    so raw RGB frames are piped to ffmpeg.
    """
    height, width = frames[0].shape[:2]
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps:.3f}",
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",  # yuv420p: the format browsers decode
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",   # yuv420p needs even width and height
        "-movflags", "+faststart",                 # metadata first: playback starts at once
        path,
    ]  # fmt: skip
    result = subprocess.run(command, input=np.stack(frames).tobytes(), capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace')}")
