"""Distil SAM 2.1's memory attention (7 memory frames) into a copy that uses K frames.

For every clip, the frozen SAM 2.1 tracker (night1 student encoder + original memory/decoder)
builds the memory bank frame by frame, exactly as at inference. At each frame t >= 1:
  teacher = original memory attention over the 7-frame memory      (no gradients)
  student = trainable copy over K frames of the SAME memory bank   (gradients)
  loss    = MSE(student output, teacher output)
The memory bank comes from the teacher's predictions ("teacher forcing").

Example:
    uv run python -m sam2lite.train.distill_memory --config configs/train/memory_k3.yaml
"""

import argparse
import copy
import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sam2.sam2_video_predictor import SAM2VideoPredictor
from torch.utils.data import DataLoader

from sam2lite.data.clips import MemoryClipDataset
from sam2lite.data.manifest import read_manifest
from sam2lite.eval.run_vos import build_predictor
from sam2lite.models.memory import limit_memory_frames
from sam2lite.tracking import start_run
from sam2lite.train.distill import (
    infinite,
    latest_checkpoint,
    lr_factor,
    save_checkpoint,
    seed_everything,
)


@contextmanager
def memory_variant(model: SAM2VideoPredictor, attention: torch.nn.Module, k: int) -> Iterator:
    """Temporarily run `model` with another memory attention over K memory frames."""
    saved = model.memory_attention, model.num_maskmem, model.maskmem_tpos_enc
    model.memory_attention = attention
    limit_memory_frames(model, k)  # keeps every frame's temporal encoding (models/memory.py)
    try:
        yield
    finally:
        model.memory_attention, model.num_maskmem, model.maskmem_tpos_enc = saved


def memory_conditioned(model, t, feats_t, pos_t, sizes, output_dict, num_frames) -> torch.Tensor:
    """Frame t's memory-conditioned features, with the same inputs as SAM 2's call in
    _track_step: lowest-resolution level only, memory bank = output_dict (frames < t)."""
    return model._prepare_memory_conditioned_features(
        frame_idx=t,
        is_init_cond_frame=False,
        current_vision_feats=feats_t[-1:],
        current_vision_pos_embeds=pos_t[-1:],
        feat_sizes=sizes[-1:],
        output_dict=output_dict,
        num_frames=num_frames,
    )


def clip_losses(
    model: SAM2VideoPredictor,
    student: torch.nn.Module,
    frames: torch.Tensor,
    mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Per-frame distillation losses (frames 1..L-1) of one clip; differentiable w.r.t. student."""
    num_frames = frames.shape[0]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        backbone_out = model.forward_image(frames)  # all frames at once (frozen encoder)
        _, feats, pos, sizes = model._prepare_backbone_features(backbone_out)

    output_dict: dict[str, dict] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    losses = []
    bf16 = {"device_type": "cuda", "dtype": torch.bfloat16}
    for t in range(num_frames):
        feats_t = [x[:, t : t + 1] for x in feats]  # frame t, every level: [HW, 1, C]
        pos_t = [x[:, t : t + 1] for x in pos]

        if t > 0:  # memory bank = frames < t
            args = (model, t, feats_t, pos_t, sizes, output_dict, num_frames)
            with torch.no_grad(), torch.autocast(**bf16):
                target = memory_conditioned(*args)  # teacher: original attention, 7 frames
            with memory_variant(model, student, k), torch.autocast(**bf16):
                pred = memory_conditioned(*args)  # student: trainable copy, K frames
            losses.append(F.mse_loss(pred.float(), target.float()))

        # Advance the tracking with the teacher, as at inference: mask, then memory encoding.
        with torch.no_grad(), torch.autocast(**bf16):
            out = model.track_step(
                frame_idx=t,
                is_init_cond_frame=(t == 0),
                current_vision_feats=feats_t,
                current_vision_pos_embeds=pos_t,
                feat_sizes=sizes,
                point_inputs=None,
                mask_inputs=mask if t == 0 else None,
                output_dict=output_dict,
                num_frames=num_frames,
            )
        output_dict["cond_frame_outputs" if t == 0 else "non_cond_frame_outputs"][t] = out
    return torch.stack(losses)


@torch.no_grad()
def evaluate(model, student, loader, k: int) -> dict[str, float]:
    """Mean frame loss on the val clips, plus the same for the untrained K-frame baseline."""
    original = model.memory_attention
    totals = {"loss": 0.0, "loss_untrained_k": 0.0}
    for frames, mask in loader:
        frames, mask = frames[0].cuda(), mask[0].cuda()
        totals["loss"] += clip_losses(model, student, frames, mask, k).mean().item()
        totals["loss_untrained_k"] += clip_losses(model, original, frames, mask, k).mean().item()
    return {key: value / len(loader) for key, value in totals.items()}


def build_loaders(cfg: DictConfig) -> tuple[Iterator, DataLoader]:
    manifest, root = Path(cfg.data.manifest), Path(cfg.data.davis_root)
    train_videos = sorted({r["video"] for r in read_manifest(manifest, "train")})
    val_videos = sorted({r["video"] for r in read_manifest(manifest, "val")})
    train_set = MemoryClipDataset(
        root, train_videos, cfg.clip_len, stride=cfg.data.stride, hflip_p=cfg.data.hflip_p
    )
    val_set = MemoryClipDataset(root, val_videos, cfg.clip_len, train=False)
    if cfg.data.max_train_clips:
        train_set.items = train_set.items[: cfg.data.max_train_clips]
    if cfg.data.max_val_clips:
        val_set.items = val_set.items[: cfg.data.max_val_clips]
    workers = cfg.data.num_workers
    generator = torch.Generator().manual_seed(cfg.seed)
    train = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=workers,
        generator=generator,
        pin_memory=True,
    )
    val = DataLoader(val_set, batch_size=1, num_workers=workers)
    print(f"train clips: {len(train_set)} | val clips: {len(val_set)}")
    return infinite(train), val


def load_config() -> DictConfig:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/train/memory_k3.yaml"))
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    assert isinstance(cfg, DictConfig)
    OmegaConf.resolve(cfg)
    return cfg


def main() -> None:
    cfg = load_config()
    seed_everything(cfg.seed)
    k = cfg.memory_frames
    model = build_predictor(cfg.predictor, "cuda")  # eval mode, all frozen
    model.requires_grad_(False)
    # eval(): dropout OFF, like the (deterministic) teacher, so the loss measures imitation and
    # not dropout noise.
    student = copy.deepcopy(model.memory_attention).eval().requires_grad_(True)

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    s = cfg.schedule
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: lr_factor(i, s.warmup_steps, s.total_steps, s.min_lr_ratio)
    )
    start_step, run_id, best_val = 0, None, math.inf
    ckpt_path = latest_checkpoint(cfg)
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        student.load_state_dict(state["memory_attention"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        start_step, run_id, best_val = state["step"], state["mlflow_run_id"], state["best_val"]
        print(f"Resuming from {ckpt_path} (step {start_step})")

    train_iter, val_loader = build_loaders(cfg)
    with start_run("distill-memory", run_name=cfg.run_name, cfg=cfg, run_id=run_id) as run:
        t0, clips_seen = time.perf_counter(), 0
        for step in range(start_step, s.total_steps):
            total = 0.0
            for _ in range(cfg.grad_accum):
                frames, mask = next(train_iter)
                loss = clip_losses(model, student, frames[0].cuda(), mask[0].cuda(), k).mean()
                (loss / cfg.grad_accum).backward()
                total += loss.item() / cfg.grad_accum
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            clips_seen += cfg.grad_accum

            done = step + 1
            if done % cfg.log_every == 0:
                rate = clips_seen / (time.perf_counter() - t0)
                mlflow.log_metrics(
                    {
                        "train/loss": total,
                        "train/grad_norm": grad_norm.item(),
                        "lr": scheduler.get_last_lr()[0],
                        "clips_per_s": rate,
                    },
                    step=done,
                )
                print(f"step {done}/{s.total_steps} loss {total:.6f} {rate:.2f} clips/s")
            if done % cfg.eval_every == 0 or done == s.total_steps:
                val = evaluate(model, student, val_loader, k)
                mlflow.log_metrics({f"val/{key}": v for key, v in val.items()}, step=done)
                print(
                    f"step {done} val loss {val['loss']:.6f} "
                    f"(untrained K={k}: {val['loss_untrained_k']:.6f})"
                )
                if val["loss"] < best_val:  # held-out TRAIN videos
                    best_val = val["loss"]
                    Path(cfg.checkpoint.dir).mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "step": done,
                            "val_loss": best_val,
                            "memory_frames": k,
                            "memory_attention": student.state_dict(),
                        },
                        Path(cfg.checkpoint.dir) / "best.pt",
                    )
                    mlflow.log_metric("best_val_loss", best_val, step=done)
            if done % cfg.checkpoint.every == 0 or done == s.total_steps:
                save_checkpoint(
                    cfg,
                    done,
                    {
                        "step": done,
                        "memory_attention": student.state_dict(),
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
