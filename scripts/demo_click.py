"""Segment and track one object in a video from a single click, using SAM 2.1.

Example:
    uv run python scripts/demo_click.py \
        --frames data/raw/DAVIS/JPEGImages/480p/blackswan --point 335 293
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"  # resolved inside the installed sam2 package
CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"


def overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a copy of `frame_bgr` (H, W, 3, uint8) with the boolean `mask` (H, W) drawn on it."""
    alpha = 0.5
    color = np.array([0, 0, 255])  # red in BGR
    out = np.copy(frame_bgr)
    # Alpha blending on the masked pixels only; a convex combination stays within [0, 255].
    out[mask] = ((1 - alpha) * out[mask] + alpha * color).astype(np.uint8)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=Path, required=True, help="Folder with 00000.jpg, ...")
    parser.add_argument(
        "--point", type=float, nargs=2, required=True, metavar=("X", "Y"), help="Click (pixels)"
    )
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame where the click is")
    parser.add_argument("--out", type=Path, default=None, help="Output .mp4 path")
    parser.add_argument("--fps", type=float, default=24.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = args.out or Path("outputs") / f"demo_click_{args.frames.name}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    predictor = build_sam2_video_predictor(CONFIG, CHECKPOINT, device=device)
    frame_paths = sorted(args.frames.glob("*.jpg"))

    # bf16 autocast on GPU, as in the official SAM 2 examples; full precision on CPU.
    with (
        torch.inference_mode(),
        torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"),
    ):
        # Loads every frame (resized to 1024x1024 and normalized) and encodes frame 0.
        state = predictor.init_state(video_path=str(args.frames))
        predictor.add_new_points_or_box(
            state,
            frame_idx=args.frame_idx,
            obj_id=1,
            points=np.array([args.point], dtype=np.float32),  # (x, y) in original pixels
            labels=np.array([1], dtype=np.int32),  # 1 = positive click, 0 = negative
        )
        # Masks come out as logits at the original video resolution: foreground is > 0.
        masks = {
            frame_idx: (mask_logits[0, 0] > 0).cpu().numpy()
            for frame_idx, _, mask_logits in predictor.propagate_in_video(state)
        }

    height, width = cv2.imread(str(frame_paths[0])).shape[:2]
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )
    for frame_idx, path in enumerate(frame_paths):
        frame = cv2.imread(str(path))
        if frame_idx in masks:
            frame = overlay_mask(frame, masks[frame_idx])
        writer.write(frame)
    writer.release()
    print(f"Tracked {len(masks)}/{len(frame_paths)} frames -> {out_path}")


if __name__ == "__main__":
    main()
