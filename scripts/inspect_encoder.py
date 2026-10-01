"""Print what the teacher's image encoder returns: the contract the student must reproduce.

Example:
    uv run python scripts/inspect_encoder.py
"""

import torch
from sam2.build_sam import build_sam2

CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"
IMAGE_SIZE = 1024


def describe(t: torch.Tensor) -> str:
    stride = IMAGE_SIZE // t.shape[-1]
    return f"shape={tuple(t.shape)}  dtype={t.dtype}  stride={stride}"


@torch.inference_mode()
def main() -> None:
    encoder = build_sam2(CONFIG, CHECKPOINT, device="cpu").image_encoder.eval()
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    print(f"input: shape={tuple(x.shape)}\n")

    neck = encoder.neck
    print(f"trunk.channel_list         = {encoder.trunk.channel_list}")
    print(f"neck.backbone_channel_list = {neck.backbone_channel_list}")
    print(f"neck.d_model               = {neck.d_model}")
    print(f"neck.fpn_top_down_levels   = {neck.fpn_top_down_levels}")
    print(f"scalp                      = {encoder.scalp}\n")

    print("trunk output (before the neck), one tensor per stage:")
    for i, t in enumerate(encoder.trunk(x)):
        print(f"  [{i}] {describe(t)}")

    out = encoder(x)
    print(f"\nimage_encoder output: dict with keys {list(out)}")
    for key in ("backbone_fpn", "vision_pos_enc"):
        print(f"  {key}: list of {len(out[key])} levels")
        for i, t in enumerate(out[key]):
            print(f"    [{i}] {describe(t)}")
    print(f"  vision_features: {describe(out['vision_features'])}")
    same = out["vision_features"] is out["backbone_fpn"][-1]
    print(f"  vision_features is backbone_fpn[-1]: {same}")


if __name__ == "__main__":
    main()
