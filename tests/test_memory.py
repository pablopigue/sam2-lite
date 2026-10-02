"""limit_memory_frames must keep each memory frame's temporal encoding (no training, C1)."""

import pytest
import torch
from sam2.build_sam import build_sam2_video_predictor

from sam2lite.models.memory import limit_memory_frames


@pytest.fixture(scope="module")
def original_tpos() -> torch.Tensor:
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_t.yaml", ckpt_path=None, device="cpu"
    )
    return predictor.maskmem_tpos_enc.detach().clone()  # [7, 1, 1, 64]


def tpos_index(num_maskmem: int, t_pos: int) -> int:
    """The index SAM 2 uses (sam2_base.py): maskmem_tpos_enc[num_maskmem - t_pos - 1]."""
    return num_maskmem - t_pos - 1


@pytest.mark.parametrize("num_frames", [2, 3, 5, 7])
def test_memory_frames_keep_their_temporal_encoding(original_tpos, num_frames) -> None:
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_t.yaml", ckpt_path=None, device="cpu"
    )
    predictor.maskmem_tpos_enc.data.copy_(original_tpos)  # same random init in both
    limit_memory_frames(predictor, num_frames)

    assert predictor.num_maskmem == num_frames
    new = predictor.maskmem_tpos_enc
    assert new.shape == (num_frames, 1, 1, 64)
    # Conditioning frame (t_pos = 0) keeps the last original encoding.
    torch.testing.assert_close(new[tpos_index(num_frames, 0)], original_tpos[-1])
    # A frame t_rel steps back keeps the encoding it had with 7 memory frames.
    for t_rel in range(1, num_frames):
        t_pos_new, t_pos_old = num_frames - t_rel, 7 - t_rel
        torch.testing.assert_close(
            new[tpos_index(num_frames, t_pos_new)], original_tpos[tpos_index(7, t_pos_old)]
        )


def test_rejects_invalid_number_of_frames() -> None:
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_t.yaml", ckpt_path=None, device="cpu"
    )
    for bad in (1, 8):
        with pytest.raises(ValueError):
            limit_memory_frames(predictor, bad)
