---
license: cc-by-nc-4.0
library_name: pytorch
pipeline_tag: mask-generation
tags:
  - video-object-segmentation
  - sam2
  - knowledge-distillation
  - onnx
---

# sam2-lite

**SAM 2.1 video object segmentation $speedup_6t× faster on CPU**: SAM 2.1 Hiera-tiny with its image encoder distilled into a MobileNetV4 backbone and its memory attention distilled to attend to 3 memory frames instead of 7. Click an object in the first frame and it is tracked through the video.

Code, training and evaluation: $github

## Results

DAVIS 2017 val (30 videos, semi-supervised, J&F), per-frame latency of the full video pipeline (median, fp32, $cpu).

| | SAM 2.1 Hiera-T (teacher) | **sam2-lite** @1024 | **sam2-lite-mobile** @576 |
|---|---|---|---|
| J&F | $jf_teacher | $jf_lite | $jf_mobile |
| CPU 6 threads (ms/frame) | $t6_teacher | $t6_lite | – |
| CPU 2 threads (ms/frame) | $t2_teacher | $t2_lite | $t2_mobile |
| Model size, fp32 (MB) | $mb_teacher | $mb_lite | $mb_lite |

sam2-lite-mobile is the same weights at a 576×576 input; it was measured in a different session from the other latency numbers (~10 % run-to-run variation). 2 CPU threads ≈ a free Hugging Face CPU Space.

### ONNX Runtime encoder

The image encoder is also provided as ONNX (one file per resolution). PyTorch → ONNX Runtime, all measured in one later session (hence e.g. 892 instead of 920 ms for the same PyTorch model: ~5-10 % varies between sessions, so only same-session numbers are compared):

| | Encoder (ms) | Full pipeline (ms) |
|---|---|---|
| 1024, CPU 6 threads | $onnx_enc_1024_6t | $onnx_pipe_1024_6t |
| 1024, CPU 2 threads | $onnx_enc_1024_2t | $onnx_pipe_1024_2t |
| 576, CPU 6 threads | $onnx_enc_576_6t | $onnx_pipe_576_6t |
| 576, CPU 2 threads | $onnx_enc_576_2t | $onnx_pipe_576_2t |

The encoder is faster but the pipeline gains little: the memory attention, which still runs in PyTorch, dominates each frame. Quality is unchanged (5 val videos at 576: J&F $jf_torch5 with PyTorch, $jf_onnx5 with ONNX Runtime).

## Files

- `model.safetensors` + `config.yaml`: the whole tracker (student encoder, fine-tuned memory attention, SAM 2.1-tiny memory encoder and mask decoder) and what is needed to rebuild it.
- `student_encoder_r1024.onnx`, `student_encoder_r576.onnx`: the image encoder for ONNX Runtime (CPU).

## Usage

```bash
git clone $github && cd sam2-lite && uv sync
```

```python
from huggingface_hub import snapshot_download
from sam2lite.export.bundle import load_bundle

path = snapshot_download("$repo_id")
predictor = load_bundle(path)  # sam2-lite @1024
mobile = load_bundle(
    path, image_size=576, onnx=f"{path}/student_encoder_r576.onnx"
)  # sam2-lite-mobile, ONNX encoder
# Then use it like SAM 2's video predictor: init_state, add_new_points_or_box, propagate_in_video.
```

## Training

- **Image encoder:** MobileNetV4-Conv-Medium (timm, ImageNet-1k pretrained) + SAM 2's FPN neck, trained to reproduce the teacher encoder's multi-scale features (MSE per FPN level, FitNets-style), on ~3.8k unlabeled frames from 54 DAVIS 2017 train videos. The rest of SAM 2.1 is reused unchanged.
- **Memory attention:** a copy of SAM 2.1's, fine-tuned to reproduce the original's output (7 memory frames) while attending to 3 (stride 4), on DAVIS 2017 train clips.
- DAVIS 2017 val was used only for the final evaluation.

## Limitations

- Accuracy drops by $jf_drop J&F points at 1024; a few small or close-up objects explain most of the drop.
- sam2-lite-mobile misses its own quality target (J&F ≥ 80) while meeting its latency target (≤ 600 ms with 2 threads).
- Distilled on a small dataset (DAVIS 2017 train only); the experiments point at the data as the current bottleneck.
- Latency was measured on one laptop CPU; numbers on other hardware will differ.

## License

Released for **non-commercial research use only** (CC BY-NC 4.0):

- It contains SAM 2.1 weights (memory encoder and mask decoder, plus a **modified**, fine-tuned memory attention) by Meta Platforms, Inc., licensed under the Apache License 2.0 (see `LICENSE` and `NOTICE`).
- The encoder starts from timm weights pretrained on ImageNet-1k, whose terms allow only non-commercial research and educational use.
- It was distilled on DAVIS 2017 frames, licensed under CC BY-NC 4.0. No DAVIS videos, frames or annotations are included.

## Citations

```bibtex
@article{ravi2024sam2,
  title   = {SAM 2: Segment Anything in Images and Videos},
  author  = {Ravi, Nikhila and others},
  journal = {arXiv preprint arXiv:2408.00714},
  year    = {2024}
}
@article{pont2017davis,
  title   = {The 2017 DAVIS Challenge on Video Object Segmentation},
  author  = {Pont-Tuset, Jordi and Perazzi, Federico and Caelles, Sergi and Arbel{\'a}ez, Pablo and Sorkine-Hornung, Alexander and Van Gool, Luc},
  journal = {arXiv preprint arXiv:1704.00675},
  year    = {2017}
}
@inproceedings{perazzi2016davis,
  title     = {A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation},
  author    = {Perazzi, Federico and Pont-Tuset, Jordi and McWilliams, Brian and Van Gool, Luc and Gross, Markus and Sorkine-Hornung, Alexander},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016}
}
@article{qin2024mobilenetv4,
  title   = {MobileNetV4: Universal Models for the Mobile Ecosystem},
  author  = {Qin, Danfeng and others},
  journal = {arXiv preprint arXiv:2404.10518},
  year    = {2024}
}
```

MLflow runs behind every number: $runs_line
