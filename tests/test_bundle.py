"""A saved bundle must load back into exactly the same tracker (random weights, no downloads)."""

import pytest
import torch
from omegaconf import OmegaConf
from sam2.build_sam import build_sam2_video_predictor

from sam2lite.export.bundle import bundle_config, load_bundle, save_bundle
from sam2lite.models.memory import limit_memory_frames
from sam2lite.models.student import TimmTrunk, build_student

CFG = OmegaConf.create(
    {
        "model": {
            "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "checkpoint": "unused.pt",
            "ckpt": "unused.pt",
        },
        "student_config": "configs/model/student_mnv4.yaml",
        "memory_frames": 3,
        "memory_stride": 4,
        "memory_ckpt": "unused.pt",
        "image_size": 1024,
        "apply_postprocessing": False,
        "non_overlap_masks": True,
    }
)


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """A random-weight tracker with the final architecture, saved as a bundle."""
    predictor = build_sam2_video_predictor(CFG.model.config, ckpt_path=None, device="cpu")
    config = bundle_config(CFG)
    predictor.image_encoder = build_student(config.student)
    limit_memory_frames(predictor, CFG.memory_frames)
    out_dir = tmp_path_factory.mktemp("bundle")
    save_bundle(predictor, config, out_dir)
    return predictor, out_dir


@pytest.mark.parametrize("image_size", [None, 576])  # sam2-lite and sam2-lite-mobile
def test_bundle_round_trip(saved, image_size) -> None:
    original, out_dir = saved
    loaded = load_bundle(out_dir, image_size=image_size)

    expected, got = original.state_dict(), loaded.state_dict()
    assert got.keys() == expected.keys()
    for name, tensor in expected.items():
        assert torch.equal(got[name], tensor), name
    assert isinstance(loaded.image_encoder.trunk, TimmTrunk)
    assert loaded.num_maskmem == CFG.memory_frames
    assert loaded.memory_temporal_stride_for_eval == CFG.memory_stride
    assert loaded.image_size == (image_size or 1024)
    assert not loaded.training
