"""ONNX Runtime session as a drop-in `image_encoder` for SAM 2's predictor.

The ONNX file (see export/onnx.py) only computes the `backbone_fpn` levels. This module rebuilds
the rest of the encoder contract in PyTorch: `vision_features` is `backbone_fpn[-1]` (as in
`ImageEncoder.forward`) and `vision_pos_enc` comes from the student neck's own sinusoidal
position encoding, which depends only on each level's shape. CPU only: the deployment target.
"""

import onnxruntime as ort
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from sam2lite.models.student import build_student


class OrtImageEncoder(nn.Module):
    """Returns the same dict as `ImageEncoder.forward`, computed by an ONNX Runtime session."""

    def __init__(
        self, onnx_path: str, position_encoding: nn.Module, num_threads: int | None = None
    ) -> None:
        """`position_encoding` = the student neck's (a parameter-free sinusoidal module).

        `num_threads` = ONNX Runtime's intra-op threads; None = torch's current setting, so the
        benchmark's `torch.set_num_threads` also applies to the ONNX encoder.
        """
        super().__init__()
        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads or torch.get_num_threads()
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            onnx_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = tuple(self.session.get_inputs()[0].shape)  # static: (1, 3, H, W)
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.position_encoding = position_encoding

    def forward(self, sample: torch.Tensor) -> dict:
        if tuple(sample.shape) != self.input_shape:
            raise ValueError(
                f"ONNX encoder expects {self.input_shape}, got {tuple(sample.shape)}: "
                "export a file for this image_size (make export) or fix image_size"
            )
        if sample.device.type != "cpu":
            raise ValueError("OrtImageEncoder runs on CPU only")
        # ONNX Runtime returns new numpy arrays on every run: no memory shared between frames.
        outputs = self.session.run(self.output_names, {self.input_name: sample.numpy()})
        features = [torch.from_numpy(x) for x in outputs]
        pos = [self.position_encoding(x).to(x.dtype) for x in features]  # as FpnNeck does
        return {"vision_features": features[-1], "vision_pos_enc": pos, "backbone_fpn": features}


def load_ort_encoder(student_cfg: DictConfig, onnx_path: str) -> OrtImageEncoder:
    """OrtImageEncoder with the position encoding of the student described by `student_cfg`.

    The weights are in the ONNX file: a fresh, non-pretrained student only provides its
    position encoding (hyperparameters read from the teacher config); the rest is discarded.
    """
    cfg = OmegaConf.merge(student_cfg, {"pretrained": False})
    assert isinstance(cfg, DictConfig)
    return OrtImageEncoder(onnx_path, build_student(cfg).neck.position_encoding).eval()
