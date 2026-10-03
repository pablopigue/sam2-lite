"""Gradio demo: upload a short video, click an object in the first frame, get it tracked.

All the logic lives in sam2lite.demo; this file is only the user interface.

Run locally:
    make app        # http://127.0.0.1:7860
"""

import tempfile
import time
from pathlib import Path

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


S, LINKS = CFG.stats, CFG.links
HERO = f"""
<div id="hero">
  <h1>sam2-lite</h1>
  <p class="tagline">Click an object in the first frame and SAM 2.1 tracks it through the video,
  now {S.speedup_cpu} faster on CPU thanks to a distilled mobile encoder and a lighter memory.</p>
  <div class="stats">
    <span class="stat"><b>{S.speedup_cpu}</b> faster on CPU</span>
    <span class="stat">J&amp;F <b>{S.jf_lite}</b> (teacher {S.jf_teacher})</span>
    <span class="stat"><b>{S.encoder_params}</b>-param encoder</span>
    <span class="stat">mobile: <b>{S.mobile_ms} ms</b>/frame on 2 CPUs</span>
  </div>
  <div class="links"><a href="{LINKS.github}">Code</a>·<a href="{LINKS.model}">Model</a></div>
</div>
"""
HOW_IT_WORKS = f"""
**Teacher:** SAM 2.1 Hiera-tiny. **Student:** the same tracker with two parts replaced by
distillation, so the rest of SAM 2.1 is reused unchanged:

1. **Image encoder** → MobileNetV4 + SAM 2's FPN neck ({S.encoder_params} parameters), trained to
   reproduce the teacher's multi-scale features.
2. **Memory attention** → attends to 3 past frames instead of 7, fine-tuned to reproduce the
   original's output.

The encoder runs with ONNX Runtime. Results on DAVIS 2017 val: J&F **{S.jf_lite}** for sam2-lite
(teacher {S.jf_teacher}) and **{S.jf_mobile}** for sam2-lite-mobile. Details, training and every
number's MLflow run: see the [model card]({LINKS.model}) and the [code]({LINKS.github}).
"""
LIMITS = f"""
- Only the first **{CFG.max_seconds} s** are used, subsampled to about **{CFG.target_fps} fps**,
  with the longer side reduced to {CFG.max_side} px: this Space has 2 CPU cores.
- One object per run, selected with one click on the first frame.
- Small or thin objects and close-ups of several touching objects are the hardest cases.
- **License:** non-commercial research use only (CC BY-NC 4.0). Contains SAM 2.1 weights by Meta
  (Apache 2.0, memory attention modified); encoder pretrained on ImageNet-1k; distilled on
  DAVIS 2017 (CC BY-NC 4.0).
"""
FOOTER = (
    '<div id="footer">sam2-lite is an independent project; it is not affiliated with, endorsed '
    "by or sponsored by Meta. “SAM 2” refers to the original model by Meta FAIR.</div>"
)


def step(number: int, title: str) -> None:
    gr.HTML(f'<div class="step-title"><span class="num">{number}</span>{title}</div>')


with gr.Blocks(title="sam2-lite · fast video object tracking") as demo:
    gr.HTML(HERO)
    video_state, point_state = gr.State(), gr.State()
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                step(1, "Upload a short video")
                video_in = gr.Video(show_label=False, sources=["upload"], height=260)
            with gr.Group(elem_classes="card"):
                step(2, "Click the object to track")
                first_frame = gr.Image(show_label=False, interactive=False, height=300)
            with gr.Group(elem_classes="card"):
                step(3, "Choose a model and track")
                model = gr.Radio(
                    list(CFG.models),
                    value=CFG.default_model,
                    show_label=False,
                    info="mobile (576 px): ~5× faster · sam2-lite (1024 px): more accurate",
                )
                run = gr.Button("Track object", variant="primary", elem_id="track-btn")
                status = gr.Markdown("Upload a video to start.", elem_classes="status")
        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                step(4, "Result")
                video_out = gr.Video(show_label=False, autoplay=True, height=420)
                result = gr.Markdown(elem_classes="status")
            with gr.Group(elem_classes="card"):
                gr.HTML('<div class="section-title">How it works</div>')
                gr.Markdown(HOW_IT_WORKS, elem_classes="info")
            with gr.Group(elem_classes="card"):
                gr.HTML('<div class="section-title">Limits and license</div>')
                gr.Markdown(LIMITS, elem_classes="info")
    gr.HTML(FOOTER)

    video_in.change(on_upload, video_in, [first_frame, video_state, point_state, status])
    first_frame.select(on_click, video_state, [first_frame, point_state])
    first_frame.select(estimate, [video_state, model], status)
    model.change(estimate, [video_state, model], status)
    run.click(on_track, [video_state, point_state, model], [video_out, result])

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

if __name__ == "__main__":
    demo.launch(theme=THEME, css=Path(__file__).with_name("style.css").read_text())
