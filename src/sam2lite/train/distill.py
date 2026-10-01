"""Distil SAM 2.1-tiny's image encoder (teacher, frozen) into the student encoder.

Loss: weighted per-level MSE on backbone_fpn (losses.py). AdamW, linear warmup + cosine
schedule, bf16 autocast, gradient accumulation, resumable checkpoints, MLflow logging.

Example:
    uv run python -m sam2lite.train.distill --config configs/train/smoke.yaml
"""

import argparse
import math
import random
import time
from collections.abc import Iterator
from pathlib import Path

import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sam2.build_sam import build_sam2
from sam2.modeling.backbones.image_encoder import ImageEncoder
from torch.utils.data import DataLoader

from sam2lite.data.frames import FrameDataset
from sam2lite.data.manifest import read_manifest
from sam2lite.models.student import build_student, set_train_mode
from sam2lite.tracking import start_run
from sam2lite.train.losses import distillation_loss


def train_step(
    teacher: ImageEncoder,
    student: ImageEncoder,
    micro_batches: list[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    weights: list[float],
    grad_clip: float,
) -> dict[str, float]:
    """One optimizer step over `micro_batches` (gradient accumulation); return mean losses."""
    accum = len(micro_batches)
    sums: dict[str, float] = {}
    for images in micro_batches:
        # Teacher: frozen and without autograd graph (its activations are not kept in VRAM).
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            t_fpn = teacher(images)["backbone_fpn"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s_fpn = student(images)["backbone_fpn"]
        total, per_level = distillation_loss(s_fpn, t_fpn, weights)
        # Divide by accum so the K accumulated gradients add up to the mean, like one big batch.
        (total / accum).backward()

        logged = {"loss": total.item()} | {f"loss_l{i}": x.item() for i, x in enumerate(per_level)}
        for key, value in logged.items():
            sums[key] = sums.get(key, 0.0) + value

    grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {key: value / accum for key, value in sums.items()} | {"grad_norm": grad_norm.item()}


@torch.no_grad()
def evaluate(
    teacher: ImageEncoder,
    student: ImageEncoder,
    loader: DataLoader,
    weights: list[float],
    freeze_bn: bool,
) -> dict[str, float]:
    """Mean distillation loss on the held-out videos (no augmentation, no gradients)."""
    student.eval()
    sums: dict[str, float] = {}
    for images in loader:
        images = images.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t_fpn = teacher(images)["backbone_fpn"]
            s_fpn = student(images)["backbone_fpn"]
        total, per_level = distillation_loss(s_fpn, t_fpn, weights)
        losses = {"loss": total.item()} | {f"loss_l{i}": x.item() for i, x in enumerate(per_level)}
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + value
    set_train_mode(student, freeze_bn)
    return {key: value / len(loader) for key, value in sums.items()}


def lr_factor(step: int, warmup_steps: int, total_steps: int, min_ratio: float) -> float:
    """Linear warmup from ~0 to 1, then half-cosine decay from 1 to `min_ratio`."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def build_teacher(cfg: DictConfig) -> ImageEncoder:
    """SAM 2.1-tiny image encoder, frozen: eval mode and no gradients, ever."""
    teacher = build_sam2(cfg.teacher.config, cfg.teacher.checkpoint, device="cuda").image_encoder
    teacher.eval().requires_grad_(False)
    return teacher


def build_loaders(cfg: DictConfig, start_step: int) -> tuple[Iterator, DataLoader]:
    manifest = Path(cfg.data.manifest)
    train_paths = [r["path"] for r in read_manifest(manifest, "train")]
    val_paths = [r["path"] for r in read_manifest(manifest, "val")]
    rng = random.Random(cfg.seed)  # fixed subsets for smoke tests
    if cfg.data.max_train_images:
        train_paths = rng.sample(train_paths, cfg.data.max_train_images)
    if cfg.data.max_val_images:
        val_paths = rng.sample(val_paths, cfg.data.max_val_images)

    common = {"batch_size": cfg.batch_size, "num_workers": cfg.data.num_workers, "pin_memory": True}
    train_set = FrameDataset(train_paths, cfg.data.image_size, augment=cfg.data.augment)
    val_set = FrameDataset(val_paths, cfg.data.image_size)
    # Seed depends on the start step: a resumed run gets a new, but reproducible, data order.
    generator = torch.Generator().manual_seed(cfg.seed + start_step)
    train = DataLoader(train_set, shuffle=True, drop_last=True, generator=generator, **common)
    val = DataLoader(val_set, shuffle=False, **common)
    print(f"train: {len(train_set)} images | val: {len(val_set)} images")
    return infinite(train), val


def infinite(loader: DataLoader) -> Iterator[torch.Tensor]:
    while True:
        yield from loader


def save_checkpoint(cfg: DictConfig, step: int, state: dict) -> None:
    ckpt_dir = Path(cfg.checkpoint.dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_dir / f"step_{step:07d}.pt")
    for old in sorted(ckpt_dir.glob("step_*.pt"))[: -cfg.checkpoint.keep_last]:
        old.unlink()  # keep only the last N checkpoints
    print(f"checkpoint saved: step {step}")


def latest_checkpoint(cfg: DictConfig) -> Path | None:
    checkpoints = sorted(Path(cfg.checkpoint.dir).glob("step_*.pt"))
    return checkpoints[-1] if checkpoints else None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config() -> DictConfig:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/train/smoke.yaml"))
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    assert isinstance(cfg, DictConfig), f"{args.config} must be a YAML mapping"
    OmegaConf.resolve(cfg)
    cfg.student = OmegaConf.load(cfg.student_config)  # logged with the run's params
    return cfg


def main() -> None:
    cfg = load_config()
    seed_everything(cfg.seed)
    torch.backends.cudnn.benchmark = True

    teacher = build_teacher(cfg)
    student = build_student(cfg.student).cuda()
    set_train_mode(student, cfg.freeze_bn)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    sched = cfg.schedule
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: lr_factor(s, sched.warmup_steps, sched.total_steps, sched.min_lr_ratio),
    )

    start_step, run_id, best_val = 0, None, math.inf
    ckpt_path = latest_checkpoint(cfg)
    if ckpt_path is not None:
        # Load on CPU: the RNG state must stay a CPU tensor; load_state_dict moves the rest.
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        student.load_state_dict(state["student"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        start_step, run_id = state["step"], state["mlflow_run_id"]
        best_val = state.get("best_val", math.inf)
        print(f"Resuming from {ckpt_path} (step {start_step})")

    train_iter, val_loader = build_loaders(cfg, start_step)
    weights = list(cfg.loss.weights)
    with start_run("distill", run_name=cfg.run_name, cfg=cfg, run_id=run_id) as run:
        t0, images_seen, accum = time.perf_counter(), 0, cfg.grad_accum
        for step in range(start_step, sched.total_steps):
            micro_batches = [next(train_iter).cuda(non_blocking=True) for _ in range(accum)]
            losses = train_step(teacher, student, micro_batches, optimizer, weights, cfg.grad_clip)
            scheduler.step()
            images_seen += cfg.batch_size * cfg.grad_accum

            done = step + 1
            if done % cfg.log_every == 0:
                elapsed = time.perf_counter() - t0
                metrics = {f"train/{k}": v for k, v in losses.items()} | {
                    "lr": scheduler.get_last_lr()[0],
                    "images_per_s": images_seen / elapsed,
                    "vram_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                }
                mlflow.log_metrics(metrics, step=done)
                print(
                    f"step {done}/{sched.total_steps} loss {losses['loss']:.5f} "
                    f"lr {metrics['lr']:.2e} {metrics['images_per_s']:.1f} img/s"
                )
            if done % cfg.eval_every == 0 or done == sched.total_steps:
                val = evaluate(teacher, student, val_loader, weights, cfg.freeze_bn)
                mlflow.log_metrics({f"val/{k}": v for k, v in val.items()}, step=done)
                print(f"step {done} val loss {val['loss']:.5f}")
                if val["loss"] < best_val:  # selected on the held-out TRAIN videos, never DAVIS val
                    best_val = val["loss"]
                    best = {"step": done, "val_loss": best_val, "student": student.state_dict()}
                    Path(cfg.checkpoint.dir).mkdir(parents=True, exist_ok=True)
                    torch.save(best, Path(cfg.checkpoint.dir) / "best.pt")
                    mlflow.log_metric("best_val_loss", best_val, step=done)
                    print(f"new best val loss: {best_val:.5f} (step {done}) -> best.pt")
            if done % cfg.checkpoint.every == 0 or done == sched.total_steps:
                save_checkpoint(
                    cfg,
                    done,
                    {
                        "step": done,
                        "student": student.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "torch_rng": torch.get_rng_state(),
                        "mlflow_run_id": run.info.run_id,
                        "best_val": best_val,
                        "config": OmegaConf.to_container(cfg, resolve=True),
                    },
                )


if __name__ == "__main__":
    main()
