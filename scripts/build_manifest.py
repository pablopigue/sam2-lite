"""Write data/manifest.csv: every DAVIS train frame, with a train/val split made BY VIDEO.

Example:
    uv run python scripts/build_manifest.py --config configs/data/davis_distill.yaml
"""

import argparse
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf

from sam2lite.data.davis import list_videos
from sam2lite.data.manifest import build_rows, split_videos, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/data/davis_distill.yaml"))
    cfg = OmegaConf.load(parser.parse_args().config)

    davis_root = Path(cfg.davis_root)
    videos = list_videos(davis_root, cfg.source_split)
    splits = split_videos(videos, cfg.val_videos, cfg.seed)
    rows = build_rows(davis_root, splits)
    write_manifest(rows, Path(cfg.manifest))

    frames_per_split = Counter(row["split"] for row in rows)
    val_videos = sorted(v for v, s in splits.items() if s == "val")
    print(f"{len(videos)} videos, {len(rows)} frames -> {cfg.manifest}")
    print(f"train: {len(videos) - len(val_videos)} videos, {frames_per_split['train']} frames")
    print(f"val:   {len(val_videos)} videos, {frames_per_split['val']} frames {val_videos}")


if __name__ == "__main__":
    main()
