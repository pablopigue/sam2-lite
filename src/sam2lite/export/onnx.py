"""Export the distilled student encoder to ONNX and check it against PyTorch with ONNX Runtime.

The ONNX graph covers trunk + FPN neck and returns the 3 `backbone_fpn` levels (after `scalp`).
`vision_features` is just `backbone_fpn[-1]`, and `vision_pos_enc` is a sinusoidal encoding
that depends only on the feature shapes, not on the image, so both are rebuilt in PyTorch by
the wrapper that plugs the ONNX session into the predictor (next step). One file per input
resolution: shapes are static, which lets ONNX Runtime plan every kernel ahead of time.

Example:
    uv run python -m sam2lite.export.onnx --config configs/export/onnx.yaml 'image_sizes=[576]'
"""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import onnxruntime as ort
import torch
from omegaconf import DictConfig, OmegaConf
from sam2.modeling.backbones.image_encoder import ImageEncoder
from torch import nn

from sam2lite.data.davis import frame_paths
from sam2lite.data.frames import FrameDataset
from sam2lite.models.student import load_student
from sam2lite.tracking import start_run

INPUT_NAME = "image"


class FpnOnly(nn.Module):
    """ImageEncoder -> tuple of its backbone_fpn levels (ONNX outputs cannot be a dict)."""

    def __init__(self, encoder: ImageEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self.encoder(image)["backbone_fpn"])


def export(encoder: ImageEncoder, image_size: int, path: Path, opset: int) -> list[str]:
    """Write `encoder` as a single ONNX file for [1, 3, image_size, image_size] inputs."""
    model = FpnOnly(encoder).eval()
    example = torch.zeros(1, 3, image_size, image_size)
    with torch.no_grad():
        num_levels = len(model(example))
    output_names = [f"backbone_fpn_{i}" for i in range(num_levels)]
    torch.onnx.export(
        model,
        (example,),
        path,
        input_names=[INPUT_NAME],
        output_names=output_names,
        opset_version=opset,
        dynamo=True,  # torch.export-based exporter (the default since torch 2.9)
        external_data=False,  # one self-contained file (~30 MB, far below protobuf's 2 GB)
    )
    return output_names


@torch.inference_mode()
def check(encoder: ImageEncoder, path: Path, frames: torch.Tensor) -> dict[str, float]:
    """Run PyTorch and ONNX Runtime on the same frames; return the worst error per output.

    Relative error = max|onnx - torch| / max|torch|, so it does not depend on each FPN level's
    feature scale.
    """
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    output_names = [o.name for o in session.get_outputs()]
    worst = dict.fromkeys(output_names, 0.0)
    for frame in frames:
        x = frame[None]  # [1, 3, H, W]
        expected = encoder(x)["backbone_fpn"]
        got = session.run(output_names, {INPUT_NAME: x.numpy()})
        for name, ref, out in zip(output_names, expected, got, strict=True):
            ref = ref.numpy()
            if out.shape != ref.shape:
                raise ValueError(f"{name}: ONNX shape {out.shape} != PyTorch shape {ref.shape}")
            rel = float(np.abs(out - ref).max() / np.abs(ref).max())
            worst[name] = max(worst[name], rel)
    return worst


def load_config() -> DictConfig:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/export/onnx.yaml"))
    args, overrides = parser.parse_known_args()  # the rest are OmegaConf `key=value` overrides
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    assert isinstance(cfg, DictConfig), f"{args.config} must be a YAML mapping"
    OmegaConf.resolve(cfg)
    return cfg


def main() -> None:
    cfg = load_config()
    print(OmegaConf.to_yaml(cfg))
    torch.set_num_threads(6)  # same P-core setting as the latency benchmark
    encoder = load_student(OmegaConf.load(cfg.student_config), cfg.ckpt)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    davis_root = Path(cfg.check.davis_root)
    paths = [str(frame_paths(davis_root, video)[0]) for video in cfg.check.videos]

    failed = []
    with start_run("export_onnx", f"onnx_{Path(cfg.ckpt).parent.name}", cfg):
        for size in cfg.image_sizes:
            path = out_dir / f"student_encoder_r{size}.onnx"
            export(encoder, size, path, cfg.opset)
            dataset = FrameDataset(paths, image_size=size)
            frames = torch.stack([dataset[i] for i in range(len(dataset))])
            worst = check(encoder, path, frames)
            size_mb = path.stat().st_size / 2**20
            print(f"r{size}: {path} ({size_mb:.1f} MB)")
            for name, rel in worst.items():
                print(f"  {name}: max relative error {rel:.2e}")
            mlflow.log_metrics({f"r{size}_{name}_max_rel_err": rel for name, rel in worst.items()})
            mlflow.log_metric(f"r{size}_file_mb", size_mb)
            if max(worst.values()) > cfg.check.max_rel_error:
                failed.append(size)

    if failed:
        raise SystemExit(f"ONNX outputs differ from PyTorch beyond tolerance at sizes {failed}")
    print(f"OK: ONNX matches PyTorch within {cfg.check.max_rel_error:.0e} at every size.")


if __name__ == "__main__":
    main()
