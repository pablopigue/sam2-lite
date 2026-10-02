"""Per-frame latency of SAM 2.1: image encoder alone and full video pipeline, on GPU and CPU.

The full pipeline is timed one frame at a time inside `propagate_in_video` (image encoder +
memory attention + mask decoder + memory encoder), so the encoder-only number gives the
ceiling of what distilling the encoder can save (Amdahl's law).

Example:
    uv run python -m sam2lite.bench.latency --config configs/bench/latency.yaml
"""

import argparse
import platform
import time
from collections.abc import Callable
from pathlib import Path

import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sam2.sam2_video_predictor import SAM2VideoPredictor
from torch import nn

from sam2lite.data.davis import annotation_paths, frame_paths, load_annotation, split_objects
from sam2lite.eval.run_vos import build_predictor
from sam2lite.tracking import start_run


def time_calls(fn: Callable[[], object], device: str, warmup: int, iters: int) -> np.ndarray:
    """Call `fn` `warmup` times untimed, then `iters` times timed; return each call's ms."""
    on_gpu = device == "cuda"
    for _ in range(warmup):  # cuDNN autotuning, allocator warm-up, lazy init, memory bank fill
        fn()

    times_ms = []
    for _ in range(iters):
        if on_gpu:
            torch.cuda.synchronize()  # nothing from previous calls may leak into this timing
        start = time.perf_counter()
        fn()
        if on_gpu:
            torch.cuda.synchronize()  # CUDA calls return before the GPU finishes: wait for it
        times_ms.append((time.perf_counter() - start) * 1000)
    return np.array(times_ms)


def summarize(times_ms: np.ndarray) -> dict[str, float]:
    return {
        "median_ms": float(np.median(times_ms)),
        "p90_ms": float(np.percentile(times_ms, 90)),
        "n": float(len(times_ms)),
    }


def encoder_workload(encoder: nn.Module, device: str, image_size: int) -> Callable[[], object]:
    """One forward of an image encoder (teacher or student) on a single preprocessed-size frame."""
    x = torch.randn(1, 3, image_size, image_size, device=device)
    return lambda: encoder(x)


def pipeline_workload(
    predictor: SAM2VideoPredictor, davis_root: Path, video: str
) -> Callable[[], object]:
    """Each call tracks one more frame of `video` (all objects), after the frame-0 GT masks."""
    # Loading and preprocessing the frames happens here, outside the timed region.
    state = predictor.init_state(video_path=str(frame_paths(davis_root, video)[0].parent))
    first_ids, _ = load_annotation(annotation_paths(davis_root, video)[0])
    for object_id, mask in split_objects(first_ids).items():
        predictor.add_new_mask(state, frame_idx=0, obj_id=object_id, mask=mask)
    frames = predictor.propagate_in_video(state)
    next(frames)  # frame 0 is the conditioning frame: already computed by add_new_mask
    return lambda: next(frames)


def load_config() -> DictConfig:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/bench/latency.yaml"))
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    assert isinstance(cfg, DictConfig), f"{args.config} must be a YAML mapping"
    return cfg


def hardware_tags() -> dict[str, str]:
    """Latency is only meaningful together with the machine it was measured on."""
    tags = {"cpu": platform.processor() or platform.machine(), "torch": torch.__version__}
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.exists():
        lines = cpu_info.read_text().splitlines()
        names = [line for line in lines if line.startswith("model name")]
        if names:
            tags["cpu"] = names[0].split(":", 1)[1].strip()
    if torch.cuda.is_available():
        tags["gpu"] = torch.cuda.get_device_name(0)
    return tags


@torch.inference_mode()
def main() -> None:
    cfg = load_config()
    if cfg.model.name == "student":
        if not cfg.model.ckpt:
            raise ValueError("model.name=student needs model.ckpt=<distillation checkpoint>")
        cfg.student = OmegaConf.load(cfg.student_config)  # logged with the run's params
    suffix = f"_mem{cfg.memory_frames}" if cfg.get("memory_frames") else ""
    with start_run("latency", run_name=f"{cfg.model.name}{suffix}", cfg=cfg):
        mlflow.set_tags(hardware_tags())
        mlflow.log_metrics(size_metrics(cfg))
        results = benchmark(cfg)
        table = format_table(cfg, results)
        print(table)
        mlflow.log_text(table, "latency_table.txt")
        for device, threads, name, stats in results:
            key = device if threads is None else f"{device}{threads}t"  # e.g. cpu6t_encoder
            mlflow.log_metric(f"{key}_{name}_median_ms", stats["median_ms"])
            mlflow.log_metric(f"{key}_{name}_p90_ms", stats["p90_ms"])


def size_metrics(cfg: DictConfig) -> dict[str, float]:
    """Parameters and fp32 size of the image encoder and of the whole predictor (MB = 2**20 B)."""
    predictor = build_predictor(cfg, "cpu")

    def count(module: torch.nn.Module) -> tuple[float, float]:
        tensors = list(module.parameters()) + list(module.buffers())
        n_params = sum(p.numel() for p in module.parameters())
        n_bytes = sum(t.numel() * 4 for t in tensors)  # fp32 = 4 bytes per value
        return n_params / 1e6, n_bytes / 2**20

    enc_params, enc_mb = count(predictor.image_encoder)
    all_params, all_mb = count(predictor)
    print(
        f"Encoder: {enc_params:.2f} M params, {enc_mb:.1f} MB | "
        f"whole model: {all_params:.2f} M params, {all_mb:.1f} MB (fp32)"
    )
    return {
        "encoder_params_M": enc_params,
        "encoder_size_mb": enc_mb,
        "model_params_M": all_params,
        "model_size_mb": all_mb,
    }


def benchmark(cfg: DictConfig) -> list[tuple[str, int | None, str, dict[str, float]]]:
    results = []
    for device in cfg.devices:
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available: skipping GPU")
            continue
        thread_counts = list(cfg.cpu_threads) if device == "cpu" else [None]
        for threads in thread_counts:
            if threads is not None:
                torch.set_num_threads(threads)
            iters = cfg.iters[device]
            # bf16 on GPU (as in evaluation); fp32 on CPU (the deployment target).
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                for name, fn in build_workloads(cfg, device).items():
                    stats = summarize(time_calls(fn, device, cfg.warmup, iters))
                    results.append((device, threads, name, stats))
    return results


def build_workloads(cfg: DictConfig, device: str) -> dict[str, Callable[[], object]]:
    # Same predictor as the evaluation: with model.name=student, the distilled encoder replaces
    # Hiera and memory + decoder stay SAM 2.1-tiny's, so both models run the same workloads.
    predictor = build_predictor(cfg, device)
    print(f"Image encoder: {type(predictor.image_encoder.trunk).__name__}")
    return {
        "encoder": encoder_workload(predictor.image_encoder, device, cfg.image_size),
        "pipeline": pipeline_workload(predictor, Path(cfg.davis_root), cfg.pipeline_video),
    }


def format_table(cfg: DictConfig, results: list) -> str:
    lines = [
        f"\nModel: {cfg.model.name} | pipeline video: {cfg.pipeline_video}",
        f"{'device':<6} {'threads':>7} {'workload':<9} {'median ms':>10} {'p90 ms':>8} {'n':>4}",
    ]
    for device, threads, name, s in results:
        t = "-" if threads is None else str(threads)
        median, p90, n = s["median_ms"], s["p90_ms"], s["n"]
        lines.append(f"{device:<6} {t:>7} {name:<9} {median:>10.1f} {p90:>8.1f} {n:>4.0f}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
