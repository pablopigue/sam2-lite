"""The ONNX Runtime encoder must honour the same contract as the PyTorch student encoder.

A random (non-pretrained) student is exported at a small size, so no checkpoint or data is
needed and the test runs in CI.
"""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from sam2lite.export.onnx import export
from sam2lite.export.ort_encoder import OrtImageEncoder
from sam2lite.models.student import build_student

STUDENT_CONFIG = "configs/model/student_mnv4.yaml"
IMAGE_SIZE = 256  # small and fast; strides 4..32 still divide it


@pytest.fixture(scope="module")
def encoders(tmp_path_factory) -> tuple[torch.nn.Module, OrtImageEncoder]:
    cfg = OmegaConf.merge(OmegaConf.load(STUDENT_CONFIG), {"pretrained": False})
    student = build_student(cfg).eval()
    path = Path(tmp_path_factory.mktemp("onnx")) / "student.onnx"
    export(student, IMAGE_SIZE, path, opset=18)
    return student, OrtImageEncoder(str(path), student.neck.position_encoding)


@torch.inference_mode()
def test_ort_encoder_matches_pytorch_contract(encoders) -> None:
    student, ort_encoder = encoders
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    expected, got = student(x), ort_encoder(x)

    assert got.keys() == expected.keys()
    for key in ("backbone_fpn", "vision_pos_enc"):
        assert len(got[key]) == len(expected[key]), key
        for g, e in zip(got[key], expected[key], strict=True):
            assert g.shape == e.shape and g.dtype == e.dtype, key
    # Error relative to each level's scale, as in `make export` (random weights give levels with
    # max|x| from ~0.01 to ~1e3, so an element-wise tolerance fails on near-zero values).
    for g, e in zip(got["backbone_fpn"], expected["backbone_fpn"], strict=True):
        assert (g - e).abs().max() / e.abs().max() < 1e-4
    for g, e in zip(got["vision_pos_enc"], expected["vision_pos_enc"], strict=True):
        assert torch.equal(g, e)  # same module, same shapes -> identical
    assert got["vision_features"] is got["backbone_fpn"][-1]  # as in ImageEncoder.forward


def test_ort_encoder_rejects_other_sizes(encoders) -> None:
    _, ort_encoder = encoders
    with pytest.raises(ValueError, match="expects"):
        ort_encoder(torch.zeros(1, 3, 2 * IMAGE_SIZE, 2 * IMAGE_SIZE))
