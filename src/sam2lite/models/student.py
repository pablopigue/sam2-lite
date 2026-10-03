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
    scale_neck_init(neck, cfg.neck_init_scale)
    return ImageEncoder(trunk=trunk, neck=neck, scalp=teacher_encoder.scalp)


def load_student(student_cfg: DictConfig, ckpt_path: str) -> ImageEncoder:
    """Build the student and load a distillation checkpoint (step_*.pt or best.pt) on CPU.

    The checkpoint is a dict whose "student" key holds the student's state_dict. Returned in
    eval() mode: BatchNorm must use its stored statistics at inference, not per-batch ones.
    """
    # pretrained=False: the ImageNet weights would be overwritten anyway, so don't download them.
    cfg = OmegaConf.merge(student_cfg, {"pretrained": False})
    assert isinstance(cfg, DictConfig)
    student = build_student(cfg)
    # Load on CPU (the checkpoint may also hold CPU-only state); strict=True catches a checkpoint
    # from a different architecture instead of silently loading part of it.
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    student.load_state_dict(state["student"], strict=True)
    return student.eval()


def set_train_mode(model: nn.Module, freeze_bn: bool) -> None:
    """model.train(), optionally keeping every BatchNorm in eval mode (frozen statistics).

    With micro-batches of 2 images, BatchNorm's running statistics drift: channels that are ~0
    on the training images get running_var -> ~0, and in eval mode a rare image that activates
    them is amplified by 1/sqrt(eps) ~ 300x per layer. Frozen BN keeps the pretrained ImageNet
    statistics (its affine weight and bias still train) and makes train and eval behave alike.
    """
    model.train()
    if freeze_bn:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


@torch.no_grad()
def scale_neck_init(neck: FpnNeck, scale: float) -> None:
    """Shrink the new neck's initial outputs towards the teacher's feature scale.

    The timm features have std ~3-8 and the default conv init keeps that scale, so an untrained
    student outputs features ~100-1000x larger than the teacher's (E[x^2] ~0.002-0.09). Scaling
    the lateral convs down (not to exactly zero, so gradients still reach the backbone) makes
    training start near the "predict zeros" loss instead of far above it.
    """
    for lateral in neck.convs:
        lateral.conv.weight.mul_(scale)
        lateral.conv.bias.zero_()
