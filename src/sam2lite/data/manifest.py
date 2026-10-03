"""Frame manifest for distillation: one row per frame, split into train/val BY VIDEO."""

import csv
import random
from pathlib import Path

from sam2lite.data.davis import frame_paths

FIELDS = ["video", "frame", "path", "split"]


def split_videos(videos: list[str], n_val: int, seed: int) -> dict[str, str]:
    """Map every video to "train" or "val"; all frames of a video share its split."""
    if not 0 < n_val < len(videos):
        raise ValueError(f"n_val={n_val} must be between 1 and {len(videos) - 1}")
    val = set(random.Random(seed).sample(sorted(videos), n_val))
    return {video: "val" if video in val else "train" for video in videos}


def build_rows(davis_root: Path, splits: dict[str, str]) -> list[dict[str, str]]:
    rows = []
    for video in sorted(splits):
        frames = frame_paths(davis_root, video)
        if not frames:
            raise FileNotFoundError(f"No frames found for video {video!r} under {davis_root}")
        for path in frames:
            row = {"video": video, "frame": path.stem, "path": str(path), "split": splits[video]}
            rows.append(row)
    return rows


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path, split: str) -> list[dict[str, str]]:
    """Rows of one split ("train" or "val")."""
    with path.open(newline="") as f:
        return [row for row in csv.DictReader(f) if row["split"] == split]
