"""Encoder contract: the student must return exactly what the teacher's image_encoder returns.

Same dict keys, same number of levels, same shapes and dtypes, for [B, 3, 1024, 1024] inputs.
If a change breaks this test, the change is wrong: SAM 2's memory and decoder consume these
outputs unchanged. Both models use random weights (no downloads, runs in CI).
"""

import copy

import pytest
import torch
from omegaconf import OmegaConf
from sam2.build_sam import build_sam2
from sam2.modeling.backbones.image_encoder import ImageEncoder

from sam2lite.models.student import build_student, scale_neck_init, set_train_mode

IMAGE_SIZE = 1024
STUDENT_CONFIG = "configs/model/student_mnv4.yaml"
TEACHER_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


@pytest.fixture(scope="module")
def teacher() -> ImageEncoder:
    return build_sam2(TEACHER_CONFIG, ckpt_path=None, device="cpu").image_encoder.eval()


@pytest.fixture(scope="module")
def student() -> ImageEncoder:
    cfg = OmegaConf.merge(OmegaConf.load(STUDENT_CONFIG), {"pretrained": False})
    return build_student(cfg).eval()


@pytest.mark.parametrize("batch_size", [1, 2])
def test_student_matches_teacher_contract(
    teacher: ImageEncoder, student: ImageEncoder, batch_size: int
) -> None:
    x = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.inference_mode():
        t_out, s_out = teacher(x), student(x)

    assert list(s_out) == list(t_out)
    for key in ("backbone_fpn", "vision_pos_enc"):
        assert len(s_out[key]) == len(t_out[key]), key
        for level, (s, t) in enumerate(zip(s_out[key], t_out[key], strict=True)):
            assert s.shape == t.shape, f"{key}[{level}]: {s.shape} != {t.shape}"
            assert s.dtype == t.dtype, f"{key}[{level}]: {s.dtype} != {t.dtype}"
    assert s_out["vision_features"] is s_out["backbone_fpn"][-1]


def test_student_positional_encoding_equals_teacher(
    teacher: ImageEncoder, student: ImageEncoder
) -> None:
    """Same fixed sine encoding: nothing to distill there."""
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.inference_mode():
        t_pos, s_pos = teacher(x)["vision_pos_enc"], student(x)["vision_pos_enc"]
    assert all(torch.equal(s, t) for s, t in zip(s_pos, t_pos, strict=True))


def test_scale_neck_init_shrinks_lateral_convs(student: ImageEncoder) -> None:
    """Lateral conv weights are scaled exactly and biases zeroed (see neck_init_scale)."""
    neck = copy.deepcopy(student.neck)
    before = [lateral.conv.weight.clone() for lateral in neck.convs]
    scale_neck_init(neck, 0.5)
    for lateral, weight in zip(neck.convs, before, strict=True):
        torch.testing.assert_close(lateral.conv.weight, weight * 0.5)
        assert torch.all(lateral.conv.bias == 0)


def test_frozen_bn_keeps_running_stats(student: ImageEncoder) -> None:
    """In train mode with freeze_bn, a forward pass must not touch BatchNorm statistics."""
    model = copy.deepcopy(student)
    set_train_mode(model, freeze_bn=True)
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    assert bns and model.training and not any(bn.training for bn in bns)
    before = [bn.running_var.clone() for bn in bns]
    with torch.no_grad():
        model(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert all(torch.equal(bn.running_var, b) for bn, b in zip(bns, before, strict=True))


def test_student_neck_matches_teacher_hyperparameters(
    teacher: ImageEncoder, student: ImageEncoder
) -> None:
    """Only the neck's input channels may differ from the teacher's."""
    assert student.scalp == teacher.scalp
    assert student.neck.d_model == teacher.neck.d_model
    assert student.neck.fpn_top_down_levels == teacher.neck.fpn_top_down_levels
    assert student.trunk.channel_list == student.neck.backbone_channel_list
