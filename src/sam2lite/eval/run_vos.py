"""Semi-supervised VOS on DAVIS 2017: first-frame GT masks -> propagate -> PNGs -> J&F -> MLflow.

Mirrors `vos_inference` in tools/vos_inference.py (SAM 2 repo): all objects of a video are
tracked together from their frame-0 masks, and predictions are saved as palette PNGs with the
same layout as the ground truth, so a standard J&F evaluator can score them.

Example:
    uv run python -m sam2lite.eval.run_vos --config configs/eval/davis_val.yaml \
        videos=[blackswan,india] out_dir=outputs/vos/smoke
"""

import argparse
import time
from pathlib import Path

import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

from sam2lite.data.davis import (
    annotation_paths,
    frame_paths,
    list_videos,
    load_annotation,
    merge_objects,
    save_annotation,
    split_objects,
)
from sam2lite.eval.jf import evaluate
from sam2lite.export.bundle import load_bundle
from sam2lite.export.ort_encoder import load_ort_encoder
from sam2lite.models.memory import limit_memory_frames
from sam2lite.models.student import load_student
from sam2lite.tracking import start_run


def build_predictor(cfg: DictConfig, device: str) -> SAM2VideoPredictor:
    """SAM 2.1 video predictor configured like the official VOS evaluation.

    With model.name == "student", the teacher's image encoder is replaced by the distilled
    student; memory attention, memory encoder and mask decoder stay SAM 2.1-tiny's.
    With `bundle`, the whole tracker comes from one bundle (export/bundle.py) instead.
    """
    if cfg.get("bundle"):
        return load_bundle(cfg.bundle, cfg.get("image_size"), cfg.model.get("onnx"), device)
    overrides = [f"++model.non_overlap_masks={str(cfg.non_overlap_masks).lower()}"]
    if cfg.get("image_size"):  # mobile model: SAM 2 derives the token grid etc. from image_size
        overrides.append(f"++model.image_size={cfg.image_size}")
    predictor = build_sam2_video_predictor(
        config_file=cfg.model.config,
        ckpt_path=cfg.model.checkpoint,
        device=device,
        apply_postprocessing=cfg.apply_postprocessing,
        hydra_overrides_extra=overrides,
    )
    if cfg.model.get("onnx"):  # The student encoder run by ONNX Runtime (CPU only)
        if cfg.model.name != "student" or device != "cpu":
            raise ValueError("model.onnx needs model.name=student and the CPU")
        predictor.image_encoder = load_ort_encoder(
            OmegaConf.load(cfg.student_config), cfg.model.onnx
        )
    elif cfg.model.name == "student":
        use_student_encoder(predictor, OmegaConf.load(cfg.student_config), cfg.model.ckpt, device)
    if cfg.get("memory_frames"):  # Fewer memory frames, no training
        limit_memory_frames(predictor, cfg.memory_frames)
    if cfg.get("memory_stride"):  # memory frames every r-th frame (SAM 2's own eval option)
        predictor.memory_temporal_stride_for_eval = cfg.memory_stride
    if cfg.get("memory_ckpt"):  # Memory attention fine-tuned for those memory frames
        load_memory_attention(predictor, cfg.memory_ckpt, cfg.get("memory_frames"))
    return predictor


def load_memory_attention(
    predictor: SAM2VideoPredictor, ckpt_path: str, memory_frames: int | None
) -> None:
    """Load a memory attention distilled by train/distill_memory.py (best.pt or step_*.pt).

    It was trained for a fixed number of memory frames: evaluating it with another number
    would be meaningless, so a mismatch is an error.
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    trained_k = state.get("memory_frames", state.get("config", {}).get("memory_frames"))
    if trained_k != memory_frames:
        raise ValueError(
            f"{ckpt_path} was trained for {trained_k} memory frames, "
            f"but memory_frames={memory_frames}"
        )
    predictor.memory_attention.load_state_dict(state["memory_attention"], strict=True)


def use_student_encoder(
    predictor: SAM2VideoPredictor, student_cfg: DictConfig, ckpt_path: str, device: str
) -> None:
    """Replace predictor.image_encoder by the distilled student loaded from `ckpt_path`.

    `ckpt_path` is a distillation checkpoint (step_*.pt or best.pt), see `load_student`.
    """
    predictor.image_encoder = load_student(student_cfg, ckpt_path).to(device)


@torch.inference_mode()
def predict_video(
    predictor: SAM2VideoPredictor,
    davis_root: Path,
    video: str,
    out_dir: Path,
    score_thresh: float,
) -> int:
    """Track every object of `video` from its frame-0 GT mask; return the number of frames."""
    frames = frame_paths(davis_root, video)
    first_ids, palette = load_annotation(annotation_paths(davis_root, video)[0])
    first_masks = split_objects(first_ids)
    if not first_masks:
        raise ValueError(f"{video}: no objects in the first-frame annotation")

    state = predictor.init_state(video_path=str(frames[0].parent))
    height, width = state["video_height"], state["video_width"]
    for object_id, mask in first_masks.items():
        predictor.add_new_mask(state, frame_idx=0, obj_id=object_id, mask=mask)

    for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(state):
        masks = {
            object_id: (mask_logits[i, 0] > score_thresh).cpu().numpy()
            for i, object_id in enumerate(object_ids)
        }
        ids = merge_objects(masks, height, width)
        save_annotation(out_dir / video / f"{frames[frame_idx].stem}.png", ids, palette)

    predictor.reset_state(state)
    return len(frames)


def load_config() -> DictConfig:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/eval/davis_val.yaml"))
    args, overrides = parser.parse_known_args()  # the rest are OmegaConf `key=value` overrides
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    assert isinstance(cfg, DictConfig), f"{args.config} must be a YAML mapping"
    OmegaConf.resolve(cfg)
    return cfg


def run_inference(cfg: DictConfig, videos: list[str], device: str) -> dict[str, float]:
    """Predict masks for `videos` into cfg.out_dir; return timing metrics."""
    davis_root, out_dir = Path(cfg.davis_root), Path(cfg.out_dir)
    predictor = build_predictor(cfg, device)
    # Record which encoder actually ran (teacher 27.2 M vs student 7.5 M): guards against
    # evaluating the teacher by mistake under a "student" label.
    if cfg.model.get("onnx"):  # the weights live in the ONNX session
        mlflow.set_tag("encoder_runtime", "onnxruntime")
        print(f"Image encoder: ONNX Runtime, {cfg.model.onnx}")
    else:
        encoder_params = sum(p.numel() for p in predictor.image_encoder.parameters())
        mlflow.set_tag("encoder_params_M", f"{encoder_params / 1e6:.2f}")
        print(
            f"Image encoder: {type(predictor.image_encoder.trunk).__name__}, "
            f"{encoder_params / 1e6:.2f} M params"
        )
    total_frames, start = 0, time.perf_counter()
    # bf16 autocast on GPU, as in the official evaluation; full precision on CPU.
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        for n, video in enumerate(videos, start=1):
            print(f"[{n}/{len(videos)}] {video}")
            total_frames += predict_video(predictor, davis_root, video, out_dir, cfg.score_thresh)
    elapsed = time.perf_counter() - start
    print(f"Inference: {len(videos)} videos, {total_frames} frames in {elapsed:.0f}s -> {out_dir}")
    return {"inference_seconds": elapsed, "frames": float(total_frames)}


def main() -> None:
    cfg = load_config()
    if cfg.model.name not in ("teacher", "student"):
        raise ValueError(f"model.name must be 'teacher' or 'student', got {cfg.model.name!r}")
    if cfg.model.name == "student" and not (
        cfg.model.ckpt or cfg.model.get("onnx") or cfg.get("bundle")
    ):
        raise ValueError("model.name=student needs model.ckpt=<distillation checkpoint>")
    print(OmegaConf.to_yaml(cfg))
    # The ONNX encoder runs on CPU, so the whole predictor does too.
    device = "cuda" if torch.cuda.is_available() and not cfg.model.get("onnx") else "cpu"
    davis_root, out_dir = Path(cfg.davis_root), Path(cfg.out_dir)
    videos = list(cfg.videos) if cfg.videos else list_videos(davis_root, cfg.split)

    suffix = f"_r{cfg.image_size}" if cfg.get("image_size") not in (None, 1024) else ""
    suffix += f"_mem{cfg.memory_frames}" if cfg.get("memory_frames") else ""
    suffix += f"_s{cfg.memory_stride}" if cfg.get("memory_stride") else ""
    suffix += "_ft" if cfg.get("memory_ckpt") else ""
    suffix += "_onnx" if cfg.model.get("onnx") else ""
    suffix += "_bundle" if cfg.get("bundle") else ""
    with start_run("vos-davis", run_name=f"{cfg.model.name}_{cfg.split}{suffix}", cfg=cfg):
        mlflow.set_tags({"device": device, "n_videos": str(len(videos))})
        if cfg.skip_inference:  # reuse predictions already in out_dir
            print(f"Skipping inference, evaluating existing predictions in {out_dir}")
        else:
            mlflow.log_metrics(run_inference(cfg, videos, device))

        scores = evaluate(davis_root, out_dir, videos)
        mlflow.log_metrics(scores)
        mlflow.log_artifact(str(out_dir / "results.csv"))  # per-object J and F
        summary = "  ".join(f"{name}={value:.1f}" for name, value in scores.items())
        print(f"{cfg.split} ({len(videos)} videos): {summary}")


if __name__ == "__main__":
    main()
