"""J&F on DAVIS 2017 with SAM 2's evaluator (vendored in third_party/sav_benchmark.py).

Standard semi-supervised protocol: first and last frames are skipped, J and F are averaged per
object over frames, then over all objects; J&F = (J + F) / 2. Scores are in [0, 100].

Example:
    uv run python -m sam2lite.eval.jf --pred-dir outputs/vos/teacher_val
"""

import argparse
from pathlib import Path

from sam2lite.data.davis import RESOLUTION, annotation_paths, list_videos
from sam2lite.eval.third_party.sav_benchmark import benchmark


def check_predictions(davis_root: Path, pred_dir: Path, videos: list[str]) -> None:
    """Fail loudly if any video or frame is missing: the evaluator would silently skip it."""
    for video in videos:
        expected = [p.name for p in annotation_paths(davis_root, video)]
        found = {p.name for p in (pred_dir / video).glob("*.png")}
        missing = [name for name in expected if name not in found]
        if missing:
            raise FileNotFoundError(f"{video}: {len(missing)} predicted masks missing")
    extra = {p.name for p in pred_dir.iterdir() if p.is_dir()} - set(videos)
    if extra:
        raise ValueError(f"Unexpected videos in {pred_dir} (not in the split): {sorted(extra)}")


def evaluate(
    davis_root: Path, pred_dir: Path, videos: list[str], num_processes: int = 8
) -> dict[str, float]:
    """Return {"J&F", "J", "F"} averaged over all objects of `videos`, in [0, 100]."""
    check_predictions(davis_root, pred_dir, videos)
    gt_root = davis_root / "Annotations" / RESOLUTION
    # strict=False because gt_root also holds the train videos; check_predictions covers the rest.
    jf, j, f, _ = benchmark(
        [str(gt_root)], [str(pred_dir)], strict=False, num_processes=num_processes, verbose=False
    )
    return {"J&F": float(jf[0]), "J": float(j[0]), "F": float(f[0])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--davis-root", type=Path, default=Path("data/raw/DAVIS"))
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--num-processes", type=int, default=8)
    args = parser.parse_args()

    videos = list_videos(args.davis_root, args.split)
    scores = evaluate(args.davis_root, args.pred_dir, videos, args.num_processes)
    summary = "  ".join(f"{name}={value:.1f}" for name, value in scores.items())
    print(f"{args.split} ({len(videos)} videos): {summary}")
    print(f"Per-object results: {args.pred_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
