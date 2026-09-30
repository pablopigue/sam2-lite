"""Summarise a DAVIS 2017 split: frames, objects and the first frame where each object appears.

Example:
    uv run python scripts/davis_summary.py --split val
"""

import argparse
from pathlib import Path

import numpy as np

from sam2lite.data.davis import annotation_paths, frame_paths, list_videos, load_annotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--davis-root", type=Path, default=Path("data/raw/DAVIS"))
    parser.add_argument("--split", choices=["train", "val"], default="val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos = list_videos(args.davis_root, args.split)
    total_frames = total_objects = 0
    late_objects: list[str] = []  # objects that do not appear in frame 0
    all_values: set[int] = set()

    print(f"{'video':<20} {'frames':>6} {'objects':>7}  first frame per object")
    for video in videos:
        frames = frame_paths(args.davis_root, video)
        annotations = annotation_paths(args.davis_root, video)
        if len(frames) != len(annotations):
            raise ValueError(f"{video}: {len(frames)} frames but {len(annotations)} annotations")

        first_seen: dict[int, int] = {}
        for frame_idx, path in enumerate(annotations):
            values = np.unique(load_annotation(path)[0])
            all_values.update(int(v) for v in values)
            for object_id in values[values > 0]:
                first_seen.setdefault(int(object_id), frame_idx)

        total_frames += len(frames)
        total_objects += len(first_seen)
        late_objects += [f"{video}:{o}@{f}" for o, f in first_seen.items() if f > 0]
        print(f"{video:<20} {len(frames):>6} {len(first_seen):>7}  {first_seen}")

    print(f"\n{args.split}: {len(videos)} videos, {total_frames} frames, {total_objects} objects")
    print(f"Pixel values found in annotations: {sorted(all_values)}")
    print(f"Objects not present in frame 0: {late_objects or 'none'}")


if __name__ == "__main__":
    main()
