"""Side-by-side video for one DAVIS video: frame | teacher masks | student masks.

Output goes to outputs/compare/ (git-ignored): DAVIS frames must not be redistributed.

Example:
    uv run python scripts/side_by_side.py --video india
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from sam2lite.data.davis import frame_paths, load_annotation

ALPHA = 0.5


def overlay(frame_bgr: np.ndarray, ids: np.ndarray, palette: list[int]) -> np.ndarray:
    """Blend each object's palette colour over its pixels (same colours as the GT)."""
    colours = np.array(palette, dtype=np.float32).reshape(-1, 3)[:, ::-1]  # RGB -> BGR
    out = frame_bgr.astype(np.float32)
    mask = ids > 0
    out[mask] = (1 - ALPHA) * out[mask] + ALPHA * colours[ids[mask]]
    return out.astype(np.uint8)


def label(image: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True)
    parser.add_argument("--teacher", type=Path, default=Path("outputs/vos/teacher_val"))
    parser.add_argument("--student", type=Path, default=Path("outputs/vos/student_val"))
    parser.add_argument("--davis-root", type=Path, default=Path("data/raw/DAVIS"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/compare"))
    parser.add_argument("--fps", type=float, default=12.0)
    args = parser.parse_args()

    frames = frame_paths(args.davis_root, args.video)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.video}_teacher_vs_student.mp4"
    writer = None
    for path in frames:
        frame = cv2.imread(str(path))
        t_ids, palette = load_annotation(args.teacher / args.video / f"{path.stem}.png")
        s_ids, _ = load_annotation(args.student / args.video / f"{path.stem}.png")
        panel = np.hstack(
            [
                label(frame.copy(), path.stem),
                label(overlay(frame, t_ids, palette), "teacher"),
                label(overlay(frame, s_ids, palette), "student"),
            ]
        )
        if writer is None:
            height, width = panel.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (width, height))
        writer.write(panel)
    if writer is not None:
        writer.release()
    print(f"{len(frames)} frames -> {out_path}")


if __name__ == "__main__":
    main()
