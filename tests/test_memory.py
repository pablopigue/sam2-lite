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


def test_memory_variant_restores_the_teacher() -> None:
    """Inside: student attention + K frames. Outside: the original teacher, untouched."""
    import copy

    from sam2lite.train.distill_memory import memory_variant

    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_t.yaml", ckpt_path=None, device="cpu"
    )
    teacher, tpos = predictor.memory_attention, predictor.maskmem_tpos_enc
    student = copy.deepcopy(teacher)
    with memory_variant(predictor, student, 3):
        assert predictor.memory_attention is student and predictor.num_maskmem == 3
    assert predictor.memory_attention is teacher
    assert predictor.num_maskmem == 7 and predictor.maskmem_tpos_enc is tpos


def test_load_memory_attention_checks_memory_frames(tmp_path) -> None:
    """The fine-tuned weights are loaded, and only with the K they were trained for."""
    from sam2lite.eval.run_vos import load_memory_attention

    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_t.yaml", ckpt_path=None, device="cpu"
    )
    weights = {k: v + 1.0 for k, v in predictor.memory_attention.state_dict().items()}
    path = tmp_path / "best.pt"
    torch.save({"memory_frames": 3, "memory_attention": weights}, path)

    with pytest.raises(ValueError):
        load_memory_attention(predictor, str(path), memory_frames=7)
    load_memory_attention(predictor, str(path), memory_frames=3)
    for name, tensor in predictor.memory_attention.state_dict().items():
        assert torch.equal(tensor, weights[name]), name
