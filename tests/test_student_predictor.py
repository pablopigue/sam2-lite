"""The video predictor must run the distilled student as its image encoder."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from sam2.build_sam import build_sam2_video_predictor

from sam2lite.eval.run_vos import use_student_encoder
from sam2lite.models.student import TimmTrunk, build_student

STUDENT_CONFIG = "configs/model/student_mnv4.yaml"


@pytest.fixture
def student_ckpt(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    """A fake distillation checkpoint with random (non-pretrained) student weights."""
    cfg = OmegaConf.merge(OmegaConf.load(STUDENT_CONFIG), {"pretrained": False})
    weights = build_student(cfg).state_dict()
    path = tmp_path / "best.pt"
    torch.save({"step": 1, "val_loss": 0.1, "student": weights}, path)
    return path, weights


def test_predictor_uses_student_with_checkpoint_weights(student_ckpt) -> None:
    path, weights = student_ckpt
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_t.yaml", ckpt_path=None, device="cpu"
    )
    use_student_encoder(predictor, OmegaConf.load(STUDENT_CONFIG), str(path), "cpu")

    encoder = predictor.image_encoder
    assert isinstance(encoder.trunk, TimmTrunk)  # not Hiera any more
    assert not encoder.training  # inference mode
    for name, tensor in encoder.state_dict().items():
        assert torch.equal(tensor, weights[name]), name  # the distilled weights
    assert next(encoder.parameters()).device.type == "cpu"
