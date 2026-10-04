"""The final tracker as one self-contained bundle: `model.safetensors` + `config.yaml`.

`model.safetensors` is the whole predictor's state_dict (student encoder, fine-tuned memory
attention, the memory's temporal encoding cut to K frames, SAM 2.1's memory encoder and mask
decoder). safetensors only stores tensors, so loading it cannot run code (unlike pickle).
`config.yaml` holds everything needed to rebuild the architecture before loading the weights.
sam2-lite and sam2-lite-mobile share the bundle: only `image_size` changes (1024 / 576).

Example:
    uv run python -m sam2lite.export.bundle --config configs/export/bundle.yaml
"""

import argparse
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file, save_file
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

from sam2lite.export.ort_encoder import load_ort_encoder
from sam2lite.models.memory import limit_memory_frames
from sam2lite.models.student import build_student

WEIGHTS, CONFIG = "model.safetensors", "config.yaml"


def bundle_config(cfg: DictConfig) -> DictConfig:
    """What `load_bundle` needs (from a run_vos-style config), plus where the weights came from."""
    # Imported here: tracking pulls in MLflow, which loading a bundle never needs.
    from sam2lite.tracking import git_state

    student = OmegaConf.merge(OmegaConf.load(cfg.student_config), {"pretrained": False})
    return OmegaConf.create(
        {
            "sam2_config": cfg.model.config,  # resolved inside the installed sam2 package
            "student": student,  # inline: the bundle does not depend on this repo's configs/
            "memory_frames": cfg.memory_frames,
            "memory_stride": cfg.memory_stride,
            "image_size": cfg.get("image_size") or 1024,
            "apply_postprocessing": cfg.apply_postprocessing,
            "non_overlap_masks": cfg.non_overlap_masks,
            "source": {
                "sam2_checkpoint": cfg.model.checkpoint,
                "encoder_ckpt": cfg.model.ckpt,
                "memory_ckpt": cfg.memory_ckpt,
                **git_state(),
            },
        }
    )


def save_bundle(predictor: SAM2VideoPredictor, config: DictConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in predictor.state_dict().items()}
    save_file(state, out_dir / WEIGHTS)
    OmegaConf.save(config, out_dir / CONFIG)


def load_bundle(
    bundle_dir: str | Path,
    image_size: int | None = None,
    onnx: str | None = None,
    device: str = "cpu",
) -> SAM2VideoPredictor:
    """Rebuild the architecture from config.yaml, then load every weight with strict=True.

    `image_size` overrides the bundle's (576 = sam2-lite-mobile); `onnx` runs the encoder
    with ONNX Runtime (CPU only).
    """
    bundle_dir = Path(bundle_dir)
    cfg = OmegaConf.load(bundle_dir / CONFIG)
    image_size = image_size or cfg.image_size
    predictor = build_sam2_video_predictor(
        config_file=cfg.sam2_config,
        ckpt_path=None,  # every weight comes from the bundle
        device="cpu",
        apply_postprocessing=cfg.apply_postprocessing,
        hydra_overrides_extra=[
            f"++model.non_overlap_masks={str(cfg.non_overlap_masks).lower()}",
            f"++model.image_size={image_size}",
        ],
    )
    # Same architecture changes as when the weights were saved, so the keys and shapes match.
    predictor.image_encoder = build_student(cfg.student)
    limit_memory_frames(predictor, cfg.memory_frames)
    predictor.memory_temporal_stride_for_eval = cfg.memory_stride
    # strict=True: a missing or unexpected tensor is an error, never a silently random weight.
    predictor.load_state_dict(load_file(bundle_dir / WEIGHTS), strict=True)
    predictor = predictor.to(device).eval()
    if onnx:
        if device != "cpu":
            raise ValueError("onnx needs device='cpu'")
        predictor.image_encoder = load_ort_encoder(cfg.student, onnx)
    return predictor


def main() -> None:
    from sam2lite.eval.run_vos import build_predictor  # the exact predictor that was evaluated

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/export/bundle.yaml"))
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    assert isinstance(cfg, DictConfig)
    with torch.inference_mode():
        predictor = build_predictor(cfg, "cpu")
    out_dir = Path(cfg.out_dir)
    save_bundle(predictor, bundle_config(cfg), out_dir)
    size_mb = (out_dir / WEIGHTS).stat().st_size / 2**20
    print(f"Saved {out_dir / WEIGHTS} ({size_mb:.1f} MB) and {out_dir / CONFIG}")


if __name__ == "__main__":
    main()
