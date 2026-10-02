"""Where does a SAM 2.1 video frame spend its time on CPU? Per-component breakdown + top ops.

Same workload and config as bench/latency.py (frames of `cows` inside propagate_in_video, after
a warmup that fills the memory bank). Forward hooks time every top-level module of the predictor;
whatever is left of the frame time (interpolations, Python glue) is reported as "other".

Example:
    uv run python -m sam2lite.bench.profile_pipeline model.name=student \
        model.ckpt=checkpoints/distill/night1/best.pt
"""

import time
from collections import defaultdict
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from sam2lite.bench.latency import load_config, pipeline_workload
from sam2lite.eval.run_vos import build_predictor
from sam2lite.tracking import start_run

COMPONENTS = [
    "image_encoder",
    "memory_attention",
    "sam_mask_decoder",
    "memory_encoder",
    "sam_prompt_encoder",
    "obj_ptr_proj",
    "mask_downsample",
]


def time_module(module: torch.nn.Module, key: str, times: dict[str, float]) -> None:
    """Accumulate the wall time of every forward call of `module` into times[key] (ms)."""
    starts: list[float] = []
    module.register_forward_pre_hook(lambda m, i, s=starts: s.append(time.perf_counter()))

    def post(m, i, o, s=starts) -> None:
        times[key] += (time.perf_counter() - s.pop()) * 1000

    module.register_forward_hook(post)


def add_timing_hooks(predictor: torch.nn.Module, times: dict[str, float]) -> None:
    """Top-level components (they do not overlap, so their times add up to the frame)."""
    for name in COMPONENTS:
        time_module(getattr(predictor, name), name, times)


def add_memory_attention_hooks(
    predictor: torch.nn.Module, inner: dict[str, float], tokens: list[int]
) -> None:
    """Inside memory_attention, summed over its layers: self-attn, cross-attn, MLP.

    Also records how many memory tokens each call attends to (spatial memory + object pointers).
    """
    for layer in predictor.memory_attention.layers:
        time_module(layer.self_attn, "self_attn", inner)
        time_module(layer.cross_attn_image, "cross_attn", inner)
        time_module(layer.linear1, "mlp", inner)
        time_module(layer.linear2, "mlp", inner)

    def count(m, args, kwargs) -> None:
        # SAM 2 passes the memory sequence-first, [N_tokens, B, C]; the module transposes it
        # internally when batch_first=True.
        memory = kwargs["memory"] if "memory" in kwargs else args[1]
        tokens.append(memory.shape[0])

    predictor.memory_attention.register_forward_pre_hook(count, with_kwargs=True)


@torch.inference_mode()
def profile_threads(cfg, threads: int) -> tuple[dict[str, float], str]:
    """Mean ms per frame of each component (+ "other" and "frame") and a top-ops table."""
    torch.set_num_threads(threads)
    predictor = build_predictor(cfg, "cpu")
    times: dict[str, float] = defaultdict(float)
    inner: dict[str, float] = defaultdict(float)
    tokens: list[int] = []
    add_timing_hooks(predictor, times)
    add_memory_attention_hooks(predictor, inner, tokens)
    next_frame = pipeline_workload(predictor, Path(cfg.davis_root), cfg.pipeline_video)

    for _ in range(cfg.warmup):  # fills the 7-frame memory bank, one-off costs
        next_frame()
    per_frame = []
    for _ in range(cfg.profile_frames):
        times.clear()
        inner.clear()
        tokens.clear()
        start = time.perf_counter()
        next_frame()
        frame_ms = (time.perf_counter() - start) * 1000
        row = {**times, "frame": frame_ms, "other": frame_ms - sum(times.values())}
        row |= {f"memory_attention.{k}": v for k, v in inner.items()}
        row["memory_tokens"] = float(tokens[-1])
        per_frame.append(row)
    means = {k: float(np.mean([f.get(k, 0.0) for f in per_frame])) for k in per_frame[0]}

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(cfg.profile_op_frames):
            next_frame()
    ops = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15)
    return means, ops


def main() -> None:
    cfg = load_config()
    cfg.profile_frames = cfg.get("profile_frames", 15)
    cfg.profile_op_frames = cfg.get("profile_op_frames", 3)
    if cfg.model.name == "student" and not cfg.model.ckpt:
        raise ValueError("model.name=student needs model.ckpt=<distillation checkpoint>")

    with start_run("profiling", run_name=cfg.model.name, cfg=cfg):
        report = [f"Model: {cfg.model.name} | video: {cfg.pipeline_video} | CPU fp32"]
        for threads in cfg.cpu_threads:
            means, ops = profile_threads(cfg, threads)
            frame = means.pop("frame")
            memory_tokens = means.pop("memory_tokens")
            inside = {k: means.pop(k) for k in list(means) if k.startswith("memory_attention.")}
            n = cfg.profile_frames
            report.append(f"\n{threads} threads: {frame:.0f} ms/frame (mean of {n} frames)")
            for name, ms in sorted(means.items(), key=lambda kv: -kv[1]):
                report.append(f"  {name:<20} {ms:8.1f} ms  {100 * ms / frame:5.1f} %")
                mlflow.log_metric(f"cpu{threads}t_{name}_ms", ms)
                mlflow.log_metric(f"cpu{threads}t_{name}_pct", 100 * ms / frame)
            report.append(f"  inside memory_attention ({memory_tokens:.0f} memory tokens):")
            for name, ms in sorted(inside.items(), key=lambda kv: -kv[1]):
                report.append(f"    {name:<28} {ms:8.1f} ms  {100 * ms / frame:5.1f} %")
                mlflow.log_metric(f"cpu{threads}t_{name}_ms", ms)
            mlflow.log_metric(f"cpu{threads}t_frame_ms", frame)
            mlflow.log_metric("memory_tokens", memory_tokens)
            mlflow.log_text(ops, f"top_ops_cpu{threads}t.txt")
        text = "\n".join(report)
        print(text)
        mlflow.log_text(text, "breakdown.txt")


if __name__ == "__main__":
    main()
