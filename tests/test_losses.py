"""distillation_loss on tensors whose MSE can be computed by hand."""

import pytest
import torch

from sam2lite.train.losses import distillation_loss

SHAPES = [(2, 4, 8, 8), (2, 4, 4, 4), (2, 4, 2, 2)]  # high to low resolution


def test_known_values() -> None:
    teacher = [torch.zeros(s) for s in SHAPES]
    student = [torch.full(s, float(level + 1)) for level, s in enumerate(SHAPES)]
    total, per_level = distillation_loss(student, teacher, weights=[1.0, 0.5, 2.0])
    assert [round(x.item(), 6) for x in per_level] == [1.0, 4.0, 9.0]
    assert total.item() == pytest.approx(21.0)
    assert total.dim() == 0


def test_loss_is_float32_with_bf16_inputs() -> None:
    teacher = [torch.zeros(s, dtype=torch.bfloat16) for s in SHAPES]
    student = [torch.ones(s, dtype=torch.bfloat16) for s in SHAPES]
    total, per_level = distillation_loss(student, teacher, weights=[1.0, 1.0, 1.0])
    assert total.dtype == torch.float32
    assert all(x.dtype == torch.float32 for x in per_level)


def test_gradients_reach_the_student_only() -> None:
    teacher = [torch.randn(s) for s in SHAPES]
    student = [torch.randn(s, requires_grad=True) for s in SHAPES]
    total, _ = distillation_loss(student, teacher, weights=[1.0, 1.0, 1.0])
    total.backward()
    assert all(s.grad is not None for s in student)
    assert all(t.grad is None for t in teacher)


def test_rejects_mismatched_levels() -> None:
    with pytest.raises(ValueError):
        distillation_loss([torch.zeros(SHAPES[0])], [torch.zeros(s) for s in SHAPES], [1.0])
