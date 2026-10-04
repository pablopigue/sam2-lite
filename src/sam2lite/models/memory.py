"""Inference-time changes to SAM 2's memory.

SAM 2.1 attends to `num_maskmem = 7` memory frames: the conditioning frame (the one with the
input mask/clicks) plus the 6 most recent frames. Each memory frame gets a learned temporal
encoding `maskmem_tpos_enc[num_maskmem - t_pos - 1]` (sam2_base.py), where t_pos = 0 for the
conditioning frame and t_pos = num_maskmem - t_rel for a frame t_rel frames in the past. So:
  - a recent frame t_rel steps back always uses index t_rel - 1 (0 = the previous frame);
  - the conditioning frame uses the LAST index (num_maskmem - 1 = 6 in SAM 2.1).
"""

import torch
from sam2.modeling.sam2_base import SAM2Base


def limit_memory_frames(model: SAM2Base, num_frames: int) -> None:
    """Attend to `num_frames` memory frames (conditioning + num_frames - 1 most recent).

    Every memory frame must keep the temporal encoding it had during training.
    """
    if not 2 <= num_frames <= model.num_maskmem:
        raise ValueError(f"num_frames must be in [2, {model.num_maskmem}], got {num_frames}")
    table = model.maskmem_tpos_enc  # [num_maskmem, 1, 1, mem_dim], read before changing anything
    # New index t_rel - 1 (recent frames) = old index t_rel - 1  -> the first num_frames - 1 rows.
    # New last index (conditioning frame) = old last index        -> the last row.
    recent, conditioning = table[: num_frames - 1], table[-1:]
    model.maskmem_tpos_enc = torch.nn.Parameter(torch.cat([recent, conditioning], dim=0))
    model.num_maskmem = num_frames
