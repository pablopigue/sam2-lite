"""Gradio demo: upload a short video, click an object in the first frame, get it tracked.

All the logic lives in sam2lite.demo; this file is only the user interface.

Run locally:
    make app        # http://127.0.0.1:7860
"""

import tempfile
import time

import gradio as gr
import numpy as np
from omegaconf import OmegaConf
from sam2.sam2_video_predictor import SAM2VideoPredictor

from sam2lite.demo import load_tracker, overlay_mask, read_video, track, write_mp4

CFG = OmegaConf.load("configs/app.yaml")
# Measured with 2 threads (docs D-043): only used to tell the user how long to wait.
SECONDS_PER_FRAME = {"sam2-lite-mobile": 0.35, "sam2-lite": 1.9}
_trackers: dict[str, SAM2VideoPredictor] = {}


def get_tracker(model: str) -> SAM2VideoPredictor:
    """Load each model once per process (a few seconds), then reuse it for every request."""
    if model not in _trackers:
        _trackers[model] = load_tracker(CFG, model)
    return _trackers[model]


def draw_click(frame: np.ndarray, point: tuple[int, int]) -> np.ndarray:
    """First frame with a green dot where the user clicked."""
    out = frame.copy()
    x, y = point
    radius = max(4, frame.shape[1] // 120)
    yy, xx = np.ogrid[: frame.shape[0], : frame.shape[1]]
    out[(xx - x) ** 2 + (yy - y) ** 2 <= radius**2] = (0, 255, 0)
    return out


def on_upload(video_path: str | None):
    """Read (cut + subsample) the video and show its first frame for the click."""
    if video_path is None:
        return None, None, None, "Upload a video to start."
    frames, fps = read_video(video_path, CFG.max_seconds, CFG.target_fps, CFG.max_side)
    info = (
        f"{len(frames)} frames at {fps:.1f} fps (first {CFG.max_seconds} s, subsampled). "
        "Click the object to track."
    )
    return frames[0], {"frames": frames, "fps": fps}, None, info


def on_click(video: dict | None, evt: gr.SelectData):
    """Remember the clicked pixel (x, y in the original frame) and mark it."""
    if video is None:
        return None, None
    point = (int(evt.index[0]), int(evt.index[1]))
    return draw_click(video["frames"][0], point), point


def on_track(video: dict | None, point: tuple[int, int] | None, model: str):
    if video is None or point is None:
        raise gr.Error("Upload a video and click the object first.")
    start = time.perf_counter()
    masks = track(get_tracker(model), video["frames"], point)
    frames = [overlay_mask(f, m) for f, m in zip(video["frames"], masks, strict=True)]
    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    write_mp4(frames, video["fps"], out_path)
    seconds = time.perf_counter() - start
    return out_path, f"{model}: {len(frames)} frames in {seconds:.0f} s."


def estimate(video: dict | None, model: str) -> str:
    if video is None:
        return ""
    n = len(video["frames"])
    return f"{model}: about {n * SECONDS_PER_FRAME[model]:.0f} s for {n} frames on 2 CPU threads."


with gr.Blocks(title="sam2-lite") as demo:
    gr.Markdown(
        "# sam2-lite\n"
        "SAM 2.1 video object segmentation, faster on CPU: a distilled MobileNetV4 image encoder "
        "and a memory attention that attends to 3 frames instead of 7. "
        "Upload a short video, click an object in the first frame and press **Track**. "
        "Unofficial project, not affiliated with Meta. Non-commercial research use only."
    )
    video_state, point_state = gr.State(), gr.State()
    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label="Video (first 10 s are used)", sources=["upload"])
            first_frame = gr.Image(label="First frame: click the object", interactive=False)
            model = gr.Radio(
                list(CFG.models),
                value=CFG.default_model,
                label="Model",
                info="sam2-lite-mobile (576 px): ~5x faster. sam2-lite (1024 px): more accurate.",
            )
            run = gr.Button("Track", variant="primary")
            status = gr.Markdown()
        with gr.Column():
            video_out = gr.Video(label="Tracked object", autoplay=True)
            result = gr.Markdown()

    video_in.change(on_upload, video_in, [first_frame, video_state, point_state, status])
    first_frame.select(on_click, video_state, [first_frame, point_state])
    first_frame.select(estimate, [video_state, model], status)
    model.change(estimate, [video_state, model], status)
    run.click(on_track, [video_state, point_state, model], [video_out, result])

if __name__ == "__main__":
    demo.launch()
