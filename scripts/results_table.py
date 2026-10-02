"""Print the README results table straight from MLflow runs (no number is copied by hand).

Example:
    uv run python scripts/results_table.py \
        --teacher-eval b44efdd1 --student-eval c27ca8b7 \
        --teacher-latency <run> --student-latency <run>
"""

import argparse
import os

import mlflow

from sam2lite.tracking import DEFAULT_TRACKING_URI


def get_run(prefix: str) -> mlflow.entities.Run:
    """Run whose id starts with `prefix` (8 characters are enough), across all experiments."""
    runs = mlflow.search_runs(search_all_experiments=True, output_format="list")
    matches = [r for r in runs if r.info.run_id.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"{len(matches)} runs match {prefix!r}")
    run = matches[0]
    if run.data.tags.get("git_dirty") != "False":
        raise ValueError(f"run {prefix} was produced with uncommitted changes: not reportable")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("teacher-eval", "student-eval", "teacher-latency", "student-latency"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))

    t_eval, s_eval = get_run(args.teacher_eval).data, get_run(args.student_eval).data
    t_lat, s_lat = get_run(args.teacher_latency).data, get_run(args.student_latency).data
    tm, sm = t_lat.metrics, s_lat.metrics

    def ms(key: str) -> str:
        return f"{tm[key]:.0f} | {sm[key]:.0f} | {tm[key] / sm[key]:.2f}×"

    rows = [
        (
            "J&F (DAVIS 2017 val)",
            f"{t_eval.metrics['JF']:.1f}",
            f"{s_eval.metrics['JF']:.1f}",
            f"{s_eval.metrics['JF'] - t_eval.metrics['JF']:+.1f}",
        ),
        (
            "J",
            f"{t_eval.metrics['J']:.1f}",
            f"{s_eval.metrics['J']:.1f}",
            f"{s_eval.metrics['J'] - t_eval.metrics['J']:+.1f}",
        ),
        (
            "F",
            f"{t_eval.metrics['F']:.1f}",
            f"{s_eval.metrics['F']:.1f}",
            f"{s_eval.metrics['F'] - t_eval.metrics['F']:+.1f}",
        ),
        (
            "Encoder params (M)",
            f"{tm['encoder_params_M']:.1f}",
            f"{sm['encoder_params_M']:.1f}",
            f"{tm['encoder_params_M'] / sm['encoder_params_M']:.1f}× fewer",
        ),
        (
            "Encoder size, fp32 (MB)",
            f"{tm['encoder_size_mb']:.0f}",
            f"{sm['encoder_size_mb']:.0f}",
            f"{tm['encoder_size_mb'] / sm['encoder_size_mb']:.1f}× smaller",
        ),
        (
            "Whole model size, fp32 (MB)",
            f"{tm['model_size_mb']:.0f}",
            f"{sm['model_size_mb']:.0f}",
            f"{tm['model_size_mb'] / sm['model_size_mb']:.1f}× smaller",
        ),
    ]
    latency_rows = [
        ("Encoder, CPU 6 threads (ms/frame)", "cpu6t_encoder_median_ms"),
        ("Full pipeline, CPU 6 threads (ms/frame)", "cpu6t_pipeline_median_ms"),
        ("Encoder, CPU 2 threads (ms/frame)", "cpu2t_encoder_median_ms"),
        ("Full pipeline, CPU 2 threads (ms/frame)", "cpu2t_pipeline_median_ms"),
        ("Full pipeline, GPU bf16 (ms/frame)", "cuda_pipeline_median_ms"),
    ]

    print("| | SAM 2.1 Hiera-T (teacher) | sam2-lite (student) | Change |")
    print("|---|---|---|---|")
    for name, t, s, change in rows:
        print(f"| {name} | {t} | {s} | {change} |")
    for name, key in latency_rows:
        print(f"| {name} | {ms(key)} |")
    print(
        f"\nLatency: median of {t_lat.params.get('iters.cpu')} (CPU) / "
        f"{t_lat.params.get('iters.cuda')} (GPU) calls after warmup, measured on "
        f"{t_lat.tags.get('cpu')} / {t_lat.tags.get('gpu')}, torch {t_lat.tags.get('torch')}."
    )
    print(
        f"MLflow runs: eval {args.teacher_eval} / {args.student_eval}, "
        f"latency {args.teacher_latency} / {args.student_latency}."
    )


if __name__ == "__main__":
    main()
