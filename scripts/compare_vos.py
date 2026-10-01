"""Compare teacher vs student per object from the evaluator's results.csv files.

Example:
    uv run python scripts/compare_vos.py \
        --teacher outputs/vos/teacher_val --student outputs/vos/student_val --top 10
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from sam2lite.data.davis import annotation_paths, load_annotation


def read_results(pred_dir: Path) -> dict[tuple[str, int], tuple[float, float, float]]:
    """{(video, object_id): (J&F, J, F)} from the evaluator's results.csv (global row skipped)."""
    rows = {}
    with (pred_dir / "results.csv").open() as f:
        for row in csv.reader(f):
            cells = [c.strip() for c in row]
            if cells[0] in ("sequence", "Global score"):
                continue
            rows[(cells[0], int(cells[1]))] = (float(cells[2]), float(cells[3]), float(cells[4]))
    return rows


def object_sizes(davis_root: Path, videos: set[str]) -> dict[tuple[str, int], float]:
    """Mean fraction of the image covered by each object over the frames where it appears."""
    sizes: dict[tuple[str, int], list[float]] = {}
    for video in videos:
        for path in annotation_paths(davis_root, video):
            ids, _ = load_annotation(path)
            for object_id in np.unique(ids[ids > 0]):
                fraction = float((ids == object_id).mean())
                sizes.setdefault((video, int(object_id)), []).append(fraction)
    return {key: float(np.mean(v)) for key, v in sizes.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teacher", type=Path, default=Path("outputs/vos/teacher_val"))
    parser.add_argument("--student", type=Path, default=Path("outputs/vos/student_val"))
    parser.add_argument("--davis-root", type=Path, default=Path("data/raw/DAVIS"))
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    teacher, student = read_results(args.teacher), read_results(args.student)
    if set(teacher) != set(student):
        raise ValueError("teacher and student results cover different objects")
    sizes = object_sizes(args.davis_root, {video for video, _ in teacher})

    rows = []
    for key in sorted(teacher):
        (t_jf, t_j, t_f), (s_jf, s_j, s_f) = teacher[key], student[key]
        rows.append((key, s_jf - t_jf, s_j - t_j, s_f - t_f, t_jf, s_jf, sizes[key]))
    rows.sort(key=lambda r: r[1])  # biggest J&F drop first

    d_jf, d_j, d_f = (np.array([r[i] for r in rows]) for i in (1, 2, 3))
    print(
        f"{len(rows)} objects | mean delta (student - teacher): "
        f"J&F {d_jf.mean():+.1f}  J {d_j.mean():+.1f}  F {d_f.mean():+.1f}"
    )
    print(
        f"student >= teacher on {(d_jf >= 0).sum()} objects; drop > 10 points on "
        f"{(d_jf < -10).sum()}"
    )

    size = np.array([r[6] for r in rows])
    small = size < np.median(size)
    print(
        f"mean delta J&F on small objects (< median size {np.median(size):.1%}): "
        f"{d_jf[small].mean():+.1f} | on large objects: {d_jf[~small].mean():+.1f}"
    )
    print(f"correlation(object size, delta J&F) = {np.corrcoef(size, d_jf)[0, 1]:+.2f}\n")

    print(
        f"{'video':<20} {'obj':>3} {'dJ&F':>6} {'dJ':>6} {'dF':>6} "
        f"{'teacher':>8} {'student':>8} {'size':>6}"
    )
    for (video, obj), djf, dj, df, tjf, sjf, sz in rows[: args.top]:
        print(
            f"{video:<20} {obj:>3} {djf:>+6.1f} {dj:>+6.1f} {df:>+6.1f} "
            f"{tjf:>8.1f} {sjf:>8.1f} {sz:>6.1%}"
        )


if __name__ == "__main__":
    main()
