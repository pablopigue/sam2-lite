"""Quick GPU probe: throughput and peak VRAM of the teacher image encoder (bf16, 1024x1024).

Used to size the distillation batch (Day 4). Not a rigorous latency benchmark: see
src/sam2lite/bench/latency.py for that.

Example:
    uv run python scripts/gpu_probe.py --batch-sizes 1 2 4
"""

import argparse
import time

import torch
from sam2.build_sam import build_sam2

CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"
IMAGE_SIZE = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--warmup", type=int, default=5, help="Untimed iterations per batch size")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations per batch size")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available(), "This probe needs a CUDA GPU"
    device = torch.device("cuda")

    encoder = build_sam2(CONFIG, CHECKPOINT, device="cuda").image_encoder.eval()
    params = sum(p.numel() for p in encoder.parameters())
    print(f"GPU: {torch.cuda.get_device_name(0)} | image_encoder params: {params / 1e6:.1f}M")
    print(f"{'batch':>5} | {'img/s':>7} | {'ms/img':>7} | {'peak VRAM (MiB)':>15}")

    for batch_size in args.batch_sizes:
        x = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(args.warmup):  # cuDNN autotuning, allocator warm-up, lazy init
                encoder(x)
            # CUDA kernels run asynchronously: wait for them before reading the clock.
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.iters):
                encoder(x)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

        images_per_s = batch_size * args.iters / elapsed
        peak_mib = torch.cuda.max_memory_allocated() / 2**20
        ms_per_image = 1000 / images_per_s
        print(f"{batch_size:>5} | {images_per_s:>7.1f} | {ms_per_image:>7.2f} | {peak_mib:>15.0f}")


if __name__ == "__main__":
    main()
