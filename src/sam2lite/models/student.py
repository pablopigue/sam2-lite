"""Student image encoder: a timm backbone as the trunk + SAM 2's FpnNeck.

It is an `ImageEncoder` (the same class as the teacher's), so it returns the same dict:
backbone_fpn / vision_pos_enc / vision_features with identical shapes. The neck reuses the
teacher's hyperparameters, read from the installed SAM 2 config; only its input channels differ.
"""

from importlib.resources import files

import timm
import torch
from omegaconf import DictConfig, OmegaConf
from sam2.modeling.backbones.image_encoder import FpnNeck, ImageEncoder
from sam2.modeling.position_encoding import PositionEmbeddingSine
from torch import nn


class TimmTrunk(nn.Module):
    """timm backbone returning 4 feature maps, from high to low resolution (strides 4..32)."""

    def __init__(self, name: str, pretrained: bool, out_indices: list[int]) -> None:
        super().__init__()
        self.body = timm.create_model(
            name, pretrained=pretrained, features_only=True, out_indices=tuple(out_indices)
        )
        strides = self.body.feature_info.reduction()
        if strides != [4, 8, 16, 32]:
            raise ValueError(f"{name} with out_indices={out_indices} gives strides {strides}")
        # SAM 2's ImageEncoder asserts trunk.channel_list == neck.backbone_channel_list,
        # which is ordered from LOW to HIGH resolution (e.g. Hiera-T: [768, 384, 192, 96]).
        self.channel_list = self.body.feature_info.channels()[::-1]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.body(x)


def build_student(cfg: DictConfig) -> ImageEncoder:
    """Build the student encoder described by a configs/model/student_*.yaml config."""
    teacher_cfg = OmegaConf.load(str(files("sam2") / cfg.teacher_config))
    teacher_encoder = teacher_cfg.model.image_encoder
    neck_cfg, pos_cfg = teacher_encoder.neck, teacher_encoder.neck.position_encoding

    trunk = TimmTrunk(cfg.backbone, cfg.pretrained, list(cfg.out_indices))
    neck = FpnNeck(
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=pos_cfg.num_pos_feats,
            temperature=pos_cfg.temperature,
            normalize=pos_cfg.normalize,
            scale=pos_cfg.scale,
        ),
        d_model=neck_cfg.d_model,
        backbone_channel_list=trunk.channel_list,  # the only difference with the teacher
        fpn_top_down_levels=list(neck_cfg.fpn_top_down_levels),
        fpn_interp_model=neck_cfg.fpn_interp_model,
    )
    return ImageEncoder(trunk=trunk, neck=neck, scalp=teacher_encoder.scalp)
