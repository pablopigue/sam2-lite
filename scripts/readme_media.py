"""Build the README GIFs from the demo's licensed example videos (CC BY, see app/examples).

assets/demo.gif       sam2-lite tracking the swans example from its preset click
assets/hard_case.gif  the teacher-vs-student comparison of the hard example (precomputed mp4)
GIFs are small (< 1 MB, pre-commit limit): reduced width and frame rate, one optimised palette.

Example:
    uv run python scripts/readme_media.py
"""

import subprocess
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

from sam2lite.demo import load_tracker, overlay_masks, read_video, track, write_mp4

OUT = Path("assets")
MAX_BYTES = 1_000_000


def to_gif(mp4: Path, gif: Path, width: int, fps: int, colors: int = 128) -> None:
    """Two-pass ffmpeg GIF: build one palette for the whole clip, then map every frame to it."""
    graph = (
        f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
        f"[a]palettegen=max_colors={colors}[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-filter_complex", graph, str(gif)],
        check=True,
    )
    size = gif.stat().st_size
    if size > MAX_BYTES:
        raise RuntimeError(f"{gif} is {size} bytes (> {MAX_BYTES}): lower width or fps")
    print(f"{gif}: {size / 1e3:.0f} KB")


def main() -> None:
    cfg = OmegaConf.load("configs/app.yaml")
    OUT.mkdir(exist_ok=True)
    swans = next(v for v in cfg.examples.videos if v.name == "swans")
    frames, fps = read_video(
        f"{cfg.examples.dir}/swans.mp4", cfg.max_seconds, cfg.target_fps, cfg.max_side
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"  # same weights; GPU only for speed
    predictor = load_tracker(cfg, "sam2-lite", device)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        masks = track(predictor, frames, [tuple(c) for c in swans.clicks])
    with tempfile.TemporaryDirectory() as tmp:
        mp4 = Path(tmp) / "demo.mp4"
        write_mp4([overlay_masks(f, m) for f, m in zip(frames, masks, strict=True)], fps, str(mp4))
        to_gif(mp4, OUT / "demo.gif", width=400, fps=6, colors=64)  # water: costly in GIF
    hard = next(v for v in cfg.examples.videos if v.get("hard"))
    comparison = Path(cfg.examples.dir) / f"{hard.name}_teacher_vs_student.mp4"
    to_gif(comparison, OUT / "hard_case.gif", width=560, fps=5, colors=64)


if __name__ == "__main__":
    main()
