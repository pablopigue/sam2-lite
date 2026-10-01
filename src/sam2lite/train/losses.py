"""Feature distillation loss between the student's and the teacher's backbone_fpn levels."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def distillation_loss(
    student_fpn: Sequence[torch.Tensor],
    teacher_fpn: Sequence[torch.Tensor],
    weights: Sequence[float],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Weighted per-level MSE between student and teacher feature maps.

    Args:
        student_fpn: the student's backbone_fpn, one [B, C, H, W] tensor per level (needs grad).
        teacher_fpn: the teacher's backbone_fpn, same shapes (no grad).
        weights: one weight per level, from the training config.

    Returns:
        (total, per_level): total = sum_l weights[l] * MSE_l, a scalar float32 tensor;
        per_level = the unweighted MSE_l of each level (for logging).
    """
    if len(student_fpn) != len(teacher_fpn) or len(student_fpn) != len(weights):
        raise ValueError(
            f"got {len(student_fpn)} student levels, {len(teacher_fpn)} teacher levels "
            f"and {len(weights)} weights"
        )
    per_level = []
    for i in range(len(student_fpn)):
        # float32 before subtracting: squared differences of small features underflow in bf16
        per_level.append(F.mse_loss(student_fpn[i].float(), teacher_fpn[i].float()))

    total = torch.stack([w * mse for w, mse in zip(weights, per_level, strict=True)]).sum()
    return total, per_level
