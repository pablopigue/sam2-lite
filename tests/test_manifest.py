"""Manifest split on a synthetic DAVIS layout (no dataset needed)."""

from pathlib import Path

import pytest

from sam2lite.data.manifest import build_rows, read_manifest, split_videos, write_manifest

VIDEOS = [f"video{i}" for i in range(10)]


def test_split_is_by_video_and_deterministic() -> None:
    splits = split_videos(VIDEOS, n_val=3, seed=0)
    assert set(splits) == set(VIDEOS)
    assert sum(s == "val" for s in splits.values()) == 3
    assert split_videos(VIDEOS, n_val=3, seed=0) == splits  # same seed, same split
    assert split_videos(list(reversed(VIDEOS)), n_val=3, seed=0) == splits  # order-independent


def test_split_rejects_invalid_sizes() -> None:
    for n_val in (0, len(VIDEOS)):
        with pytest.raises(ValueError):
            split_videos(VIDEOS, n_val=n_val, seed=0)


def test_manifest_round_trip_has_no_leakage(tmp_path: Path) -> None:
    root = tmp_path / "DAVIS"
    for video in VIDEOS:
        frames_dir = root / "JPEGImages" / "480p" / video
        frames_dir.mkdir(parents=True)
        for i in range(4):
            (frames_dir / f"{i:05d}.jpg").touch()

    splits = split_videos(VIDEOS, n_val=2, seed=1)
    write_manifest(build_rows(root, splits), tmp_path / "manifest.csv")
    train = read_manifest(tmp_path / "manifest.csv", "train")
    val = read_manifest(tmp_path / "manifest.csv", "val")

    assert len(train) + len(val) == 4 * len(VIDEOS)
    assert not {r["video"] for r in train} & {r["video"] for r in val}  # no video in both
