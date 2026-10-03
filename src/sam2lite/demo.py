"""Click-to-track logic shared by the Gradio app and the API (no UI code here).

read_video -> load_tracker -> track (clicks on the first frame) -> overlay_masks -> write_mp4.

A click is (obj_id, x, y, label): label 1 = "part of this object", 0 = "not part of it". Several
clicks with the same obj_id refine one object; different obj_ids are different objects. The app
uses one positive click (simplest and fastest, D-045); the API keeps the general form.
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

Click = tuple[int, int, int, int]  # (obj_id, x, y, label), x and y in first-frame pixels
# One colour per object (RGB), distinguishable on most backgrounds.
PALETTE = [(255, 64, 64), (64, 160, 255), (255, 200, 0), (180, 90, 255), (0, 210, 140)]


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


def load_tracker(cfg: DictConfig, model: str, device: str = "cpu") -> SAM2VideoPredictor:
    """sam2-lite or sam2-lite-mobile, from the local bundle or the Hub.

    On CPU the encoder runs with ONNX Runtime (D-040); on GPU (the ZeroGPU Space) everything runs
    in PyTorch, since the ONNX session is CPU-only.
    """
    spec = cfg.models[model]
    torch.set_num_threads(cfg.threads)  # before building: the ONNX session copies this value
    if Path(cfg.local_bundle).exists():
        bundle_dir, onnx_dir = Path(cfg.local_bundle), Path(cfg.local_onnx_dir)
    else:  # private repo: the token comes from HF_TOKEN (Space secret or .env), never the code
        bundle_dir = onnx_dir = Path(
            snapshot_download(cfg.repo_id, token=os.environ.get("HF_TOKEN"))
        )
    onnx = str(onnx_dir / spec.onnx) if device == "cpu" else None
    return load_bundle(bundle_dir, image_size=spec.image_size, onnx=onnx, device=device)


def _init_state(predictor: SAM2VideoPredictor, frames: list[np.ndarray]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        # SAM 2 reads an .mp4 only through decord (not installed) or a folder of JPEGs.
        for i, frame in enumerate(frames):
            cv2.imwrite(f"{tmp}/{i:05d}.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return predictor.init_state(video_path=tmp)


def _add_clicks(predictor: SAM2VideoPredictor, state: dict, clicks: list[Click]) -> dict:
    """Add every object's clicks on frame 0; return {obj_id: mask} for frame 0."""
    out: dict[int, np.ndarray] = {}
    for obj_id in sorted({c[0] for c in clicks}):
        points = [[x, y] for o, x, y, _ in clicks if o == obj_id]
        labels = [label for o, _, _, label in clicks if o == obj_id]
        _, obj_ids, mask_logits = predictor.add_new_points_or_box(
            state, frame_idx=0, obj_id=obj_id, points=points, labels=labels
        )
        # Each call returns frame 0's masks for all objects added so far.
        out = {o: (mask_logits[i, 0] > 0).cpu().numpy() for i, o in enumerate(obj_ids)}
    return out


@torch.inference_mode()
def track(
    predictor: SAM2VideoPredictor, frames: list[np.ndarray], clicks: list[Click]
) -> list[dict[int, np.ndarray]]:
    """{obj_id: boolean mask (H, W)} per frame, for every object defined by `clicks`."""
    state = _init_state(predictor, frames)
    _add_clicks(predictor, state, clicks)
    masks: list[dict[int, np.ndarray]] = [{} for _ in frames]
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        masks[frame_idx] = {o: (mask_logits[i, 0] > 0).cpu().numpy() for i, o in enumerate(obj_ids)}
    return masks


def overlay_mask(
    frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """Copy of `frame` (H, W, 3, uint8) with `color` alpha-blended (0.5) on the boolean mask."""
    out = frame.copy()
    # A convex combination of two values in [0, 255] stays in [0, 255].
    out[mask] = (0.5 * out[mask] + 0.5 * np.array(color)).astype(np.uint8)
    return out


def overlay_masks(frame: np.ndarray, masks: dict[int, np.ndarray]) -> np.ndarray:
    """`frame` with each object's mask in its own colour (PALETTE[obj_id - 1])."""
    for obj_id, mask in sorted(masks.items()):
        frame = overlay_mask(frame, mask, PALETTE[(obj_id - 1) % len(PALETTE)])
    return frame


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
