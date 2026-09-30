"""Semi-supervised VOS inference on DAVIS 2017: first-frame GT masks -> propagate -> PNGs.

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


def build_predictor(cfg: DictConfig, device: str) -> SAM2VideoPredictor:
    """SAM 2.1 video predictor configured like the official VOS evaluation."""
    overrides = [f"++model.non_overlap_masks={str(cfg.non_overlap_masks).lower()}"]
    return build_sam2_video_predictor(
        config_file=cfg.model.config,
        ckpt_path=cfg.model.checkpoint,
        device=device,
        apply_postprocessing=cfg.apply_postprocessing,
        hydra_overrides_extra=overrides,
    )


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
    OmegaConf.resolve(cfg)
    return cfg


def main() -> None:
    cfg = load_config()
    print(OmegaConf.to_yaml(cfg))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    davis_root, out_dir = Path(cfg.davis_root), Path(cfg.out_dir)
    videos = list(cfg.videos) if cfg.videos else list_videos(davis_root, cfg.split)

    predictor = build_predictor(cfg, device)
    total_frames, start = 0, time.perf_counter()
    # bf16 autocast on GPU, as in the official evaluation; full precision on CPU.
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        for n, video in enumerate(videos, start=1):
            print(f"[{n}/{len(videos)}] {video}")
            total_frames += predict_video(predictor, davis_root, video, out_dir, cfg.score_thresh)

    elapsed = time.perf_counter() - start
    print(f"Done: {len(videos)} videos, {total_frames} frames in {elapsed:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
