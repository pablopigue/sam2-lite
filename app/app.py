"""Gradio demo: upload a short video, click objects in the first frame, get them tracked.

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

from sam2lite.demo import (
    PALETTE,
    Click,
    load_tracker,
    overlay_masks,
    preview,
    read_video,
    track,
    validate_clicks,
    write_mp4,
)

CFG = OmegaConf.load("configs/app.yaml")
# Measured with 2 threads (docs D-043), one object: only used to tell the user how long to wait.
SECONDS_PER_FRAME = {"sam2-lite-mobile": 0.35, "sam2-lite": 1.9}
COLOR_NAMES = ["red", "blue", "yellow", "purple", "green"]  # names of demo.PALETTE
INCLUDE, EXCLUDE = "➕ Include", "➖ Exclude"
_trackers: dict[str, SAM2VideoPredictor] = {}


def get_tracker(model: str) -> SAM2VideoPredictor:
    """Load each model once per process (a few seconds), then reuse it for every request."""
    if model not in _trackers:
        _trackers[model] = load_tracker(CFG, model)
    return _trackers[model]


def draw_clicks(frame: np.ndarray, clicks: list[Click]) -> np.ndarray:
    """Dots in each object's colour: white ring = include, black ring = exclude."""
    out = frame.copy()
    radius = max(5, frame.shape[1] // 110)
    yy, xx = np.ogrid[: frame.shape[0], : frame.shape[1]]
    for obj_id, x, y, label in clicks:
        dist2 = (xx - x) ** 2 + (yy - y) ** 2
        out[dist2 <= (radius + 2) ** 2] = (255, 255, 255) if label == 1 else (0, 0, 0)
        out[dist2 <= radius**2] = PALETTE[(obj_id - 1) % len(PALETTE)]
    return out


def render_first(video: dict, clicks: list[Click], model: str) -> np.ndarray:
    """First frame with the live preview masks (objects with a positive click) and the dots."""
    first = video["frames"][0]
    positive = {c[0] for c in clicks if c[3] == 1}
    usable = [c for c in clicks if c[0] in positive]  # negatives alone select nothing
    masks = preview(get_tracker(model), first, usable) if usable else {}
    return draw_clicks(overlay_masks(first, masks), clicks)


def describe(clicks: list[Click], obj: int) -> str:
    color = COLOR_NAMES[(obj - 1) % len(COLOR_NAMES)]
    n_obj = len({c[0] for c in clicks} | {obj})
    n_clicks = sum(c[0] == obj for c in clicks)
    return (
        f"Editing **object {obj} ({color})**: {n_clicks} click(s) · "
        f"objects: {n_obj}/{CFG.max_objects}. Include = part of the object, exclude = not."
    )


def on_upload(video_path: str | None):
    """Read (cut + subsample) the video and show its first frame for the clicks."""
    if video_path is None:
        return None, None, [], 1, "Upload a video to start."
    frames, fps = read_video(video_path, CFG.max_seconds, CFG.target_fps, CFG.max_side)
    info = (
        f"{len(frames)} frames at {fps:.1f} fps (first {CFG.max_seconds} s, subsampled). "
        "Click the object to track."
    )
    return frames[0], {"frames": frames, "fps": fps}, [], 1, info


def on_click(video, clicks: list[Click], obj: int, mode: str, model: str, evt: gr.SelectData):
    """Add a click (x, y in the original frame) to the current object and refresh the preview."""
    if video is None:
        raise gr.Error("Upload a video first.")
    label = 1 if mode == INCLUDE else 0
    if label == 0 and not any(c[0] == obj and c[3] == 1 for c in clicks):
        gr.Warning("Start each object with an include click.")
        return gr.skip(), clicks, describe(clicks, obj)
    clicks = [*clicks, (obj, int(evt.index[0]), int(evt.index[1]), label)]
    return render_first(video, clicks, model), clicks, describe(clicks, obj)


def on_new_object(clicks: list[Click], obj: int):
    if not any(c[0] == obj for c in clicks):
        return obj, describe(clicks, obj)  # the current object is still empty: reuse it
    if obj >= CFG.max_objects:
        gr.Warning(f"At most {CFG.max_objects} objects on this CPU Space.")
        return obj, describe(clicks, obj)
    return obj + 1, describe(clicks, obj + 1)


def on_undo(video, clicks: list[Click], model: str):
    if video is None or not clicks:
        return gr.skip(), clicks, 1, "Nothing to undo."
    clicks = clicks[:-1]
    obj = clicks[-1][0] if clicks else 1
    return render_first(video, clicks, model), clicks, obj, describe(clicks, obj)


def on_clear(video):
    if video is None:
        return None, [], 1, "Upload a video to start."
    return video["frames"][0], [], 1, describe([], 1)


def on_model_change(video, clicks: list[Click], obj: int, model: str):
    """A different model can produce a different preview mask: recompute it."""
    if video is None:
        return gr.skip(), ""
    return render_first(video, clicks, model), describe(clicks, obj) + " " + estimate(
        video, clicks, model
    )


def on_track(video, clicks: list[Click], model: str):
    if video is None:
        raise gr.Error("Upload a video first.")
    try:
        validate_clicks(clicks, CFG.max_objects)
    except ValueError as error:
        raise gr.Error(str(error).capitalize() + ".") from error
    start = time.perf_counter()
    masks = track(get_tracker(model), video["frames"], clicks)
    frames = [overlay_masks(f, m) for f, m in zip(video["frames"], masks, strict=True)]
    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    write_mp4(frames, video["fps"], out_path)
    seconds = time.perf_counter() - start
    n_obj = len({c[0] for c in clicks})
    return out_path, f"{model}: {n_obj} object(s), {len(frames)} frames in {seconds:.0f} s."


def estimate(video, clicks: list[Click], model: str) -> str:
    if video is None:
        return ""
    n, n_obj = len(video["frames"]), max(1, len({c[0] for c in clicks}))
    seconds = n * SECONDS_PER_FRAME[model]
    return f"Tracking: about {seconds:.0f} s per object ({n_obj} now) with {model} on 2 CPUs."


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
- Up to {CFG.max_objects} objects, selected with clicks on the first frame; each extra object
  adds roughly the time of another run.
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
    video_state = gr.State()
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                step(1, "Upload a short video")
                video_in = gr.Video(show_label=False, sources=["upload"], height=260)
            with gr.Group(elem_classes="card"):
                step(2, "Click the object(s) to track")
                first_frame = gr.Image(show_label=False, interactive=False, height=300)
                with gr.Row():
                    mode = gr.Radio([INCLUDE, EXCLUDE], value=INCLUDE, show_label=False)
                with gr.Row():
                    new_obj = gr.Button("🆕 New object", size="sm")
                    undo = gr.Button("↩ Undo", size="sm")
                    clear = gr.Button("🗑 Clear", size="sm")
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

    clicks_state, obj_state = gr.State([]), gr.State(1)
    video_in.change(
        on_upload, video_in, [first_frame, video_state, clicks_state, obj_state, status]
    )
    first_frame.select(
        on_click,
        [video_state, clicks_state, obj_state, mode, model],
        [first_frame, clicks_state, status],
    )
    new_obj.click(on_new_object, [clicks_state, obj_state], [obj_state, status])
    undo.click(
        on_undo, [video_state, clicks_state, model], [first_frame, clicks_state, obj_state, status]
    )
    clear.click(on_clear, video_state, [first_frame, clicks_state, obj_state, status])
    model.change(
        on_model_change, [video_state, clicks_state, obj_state, model], [first_frame, status]
    )
    run.click(on_track, [video_state, clicks_state, model], [video_out, result])

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

if __name__ == "__main__":
    demo.launch(theme=THEME, css=Path(__file__).with_name("style.css").read_text())
