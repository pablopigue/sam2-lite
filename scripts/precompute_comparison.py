"""Precompute the teacher-vs-student video for the demo's hard example (honest failure case).

SAM 2.1 Hiera-T (teacher) and sam2-lite (student, 1024 px) track the same frames from the same
clicks; the two results are written side by side, labelled, as a small H.264 mp4 for git.
Runs locally: it needs the teacher checkpoint, which the Space does not ship.

Example:
    uv run python scripts/precompute_comparison.py
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from sam2.build_sam import build_sam2_video_predictor

from sam2lite.demo import overlay_masks, read_video, track, write_mp4
from sam2lite.export.bundle import load_bundle

TEACHER_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
TEACHER_CKPT = "checkpoints/sam2.1_hiera_tiny.pt"
PANEL_WIDTH = 512  # two panels -> 1024 px wide (and < 1 MB)
MAX_BYTES = 1_000_000  # pre-commit blocks files over 1 MB


def label(frame: np.ndarray, text: str) -> np.ndarray:
    """Write `text` on a dark band at the top-left corner."""
    out = frame.copy()
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    (w, h), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(out, (0, 0), (w + 16, h + 16), (0, 0, 0), -1)
    cv2.putText(out, text, (8, h + 8), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def main() -> None:
    cfg = OmegaConf.load("configs/app.yaml")
    example = next(v for v in cfg.examples.videos if v.get("hard"))
    frames, fps = read_video(
        f"{cfg.examples.dir}/{example.name}.mp4", cfg.max_seconds, cfg.target_fps, cfg.max_side
    )
    clicks = [tuple(c) for c in example.clicks]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {
        "SAM 2.1 Hiera-T (teacher)": build_sam2_video_predictor(
            TEACHER_CONFIG, TEACHER_CKPT, device=device
        ),
        "sam2-lite (student)": load_bundle(cfg.local_bundle, image_size=1024, device=device),
    }
    panels = []
    for name, predictor in models.items():
        # bf16 on GPU, as in the official evaluation; full precision on CPU.
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            masks = track(predictor, frames, clicks)
        lost = np.mean([not m or max(x.mean() for x in m.values()) < 0.002 for m in masks])
        print(f"{name}: object lost in {lost:.0%} of {len(frames)} frames")
        height = round(frames[0].shape[0] * PANEL_WIDTH / frames[0].shape[1])
        panels.append(
            [
                label(cv2.resize(overlay_masks(f, m), (PANEL_WIDTH, height)), name)
                for f, m in zip(frames, masks, strict=True)
            ]
        )
    out = Path(cfg.examples.dir) / f"{example.name}_teacher_vs_student.mp4"
    write_mp4([np.hstack(pair) for pair in zip(*panels, strict=True)], fps, str(out))
    size = out.stat().st_size
    if size > MAX_BYTES:
        raise RuntimeError(f"{out} is {size} bytes (> {MAX_BYTES}): lower PANEL_WIDTH")
    print(f"{out}: {size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
