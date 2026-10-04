"""Gradio demo: upload a short video, click an object in the first frame, get it tracked.

All the logic lives in sam2lite.demo; this file is only the user interface.

Run locally:
    make app                          # CPU, ONNX Runtime encoder: http://127.0.0.1:7860
    SAM2LITE_DEVICE=cuda make app     # GPU, as in the ZeroGPU Space (D-046)
"""

import os
import re
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np
import spaces
import torch
from omegaconf import OmegaConf
from sam2.sam2_video_predictor import SAM2VideoPredictor

from sam2lite.demo import load_tracker, overlay_masks, read_video, track, write_mp4

CFG = OmegaConf.load("configs/app.yaml")
# Measured with 2 threads: only used to tell the user how long to wait.
SECONDS_PER_FRAME = {"sam2-lite-mobile": 0.35, "sam2-lite": 1.9}
EXAMPLES_DIR = Path(CFG.examples.dir)
# Preset click of each example video, by file name (Gradio copies examples to its cache, keeping
# the name): selecting an example leaves only the Track button to press.
EXAMPLE_CLICK = {v.name: tuple(v.clicks[0][1:3]) for v in CFG.examples.videos}
HARD = next(v for v in CFG.examples.videos if v.get("hard"))
# "cpu" by default (the measured deployment target); "cuda" for a local GPU or a ZeroGPU Space.
DEVICE = os.environ.get("SAM2LITE_DEVICE", "cpu")
_trackers: dict[str, SAM2VideoPredictor] = {}


def get_tracker(model: str) -> SAM2VideoPredictor:
    """Load each model once per process (a few seconds), then reuse it for every request."""
    if model not in _trackers:
        _trackers[model] = load_tracker(CFG, model, DEVICE)
    return _trackers[model]


if DEVICE == "cuda":  # ZeroGPU wants models placed on cuda at start-up, not inside a request
    for _name in CFG.models:
        get_tracker(_name)


def draw_click(frame: np.ndarray, point: tuple[int, int]) -> np.ndarray:
    """First frame with a green dot (white ring) where the user clicked."""
    out = frame.copy()
    x, y = point
    radius = max(5, frame.shape[1] // 110)
    yy, xx = np.ogrid[: frame.shape[0], : frame.shape[1]]
    dist2 = (xx - x) ** 2 + (yy - y) ** 2
    out[dist2 <= (radius + 2) ** 2] = (255, 255, 255)
    out[dist2 <= radius**2] = (0, 200, 0)
    return out


def estimate(video: dict | None, model: str) -> str:
    if video is None:
        return ""
    n, cpu = len(video["frames"]), SECONDS_PER_FRAME[model]
    if DEVICE == "cuda":
        return f"Ready · {n} frames · shared GPU (on a laptop CPU, 2 threads: ~{cpu:.2f} s/frame)."
    return f"Ready · {n} frames · ~{cpu:.2f} s per frame with {model} (2 CPUs)."


def on_upload(video_path: str | None, model: str):
    """Read (cut + subsample) the video, show its first frame and, for an example, its click."""
    if video_path is None:
        return None, None, None, "Upload a video to start."
    frames, fps = read_video(video_path, CFG.max_seconds, CFG.target_fps, CFG.max_side)
    video = {"frames": frames, "fps": fps}
    point = EXAMPLE_CLICK.get(Path(video_path).stem)
    if point is not None:
        return draw_click(frames[0], point), video, point, estimate(video, model)
    info = f"{len(frames)} frames at {fps:.1f} fps. Click the object to track."
    return frames[0], video, None, info


def on_click(video: dict | None, model: str, evt: gr.SelectData):
    """Remember the clicked pixel (x, y in the original frame) and mark it."""
    if video is None:
        raise gr.Error("Upload a video first.")
    point = (int(evt.index[0]), int(evt.index[1]))
    return draw_click(video["frames"][0], point), point, estimate(video, model)


@spaces.GPU(duration=60)
def run_tracking(frames: list[np.ndarray], point: tuple[int, int], model: str) -> list[dict]:
    # bf16 on GPU, as in the official evaluation; full precision (and ONNX Runtime) on CPU.
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        return track(get_tracker(model), frames, [(1, *point, 1)])  # one positive click


def on_track(video: dict | None, point: tuple[int, int] | None, model: str):
    if video is None or point is None:
        raise gr.Error("Upload a video and click the object first.")
    start = time.perf_counter()
    masks = run_tracking(video["frames"], point, model)  # drawing + encoding stay off the GPU
    frames = [overlay_masks(f, m) for f, m in zip(video["frames"], masks, strict=True)]
    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    write_mp4(frames, video["fps"], out_path)
    seconds = time.perf_counter() - start
    return out_path, f"{model}: {len(frames)} frames · {seconds / len(frames):.2f} s per frame."


def credits() -> str:
    """Example-video attribution, from the file written by scripts/fetch_examples.py."""
    text = (EXAMPLES_DIR / "ATTRIBUTION.md").read_text()
    return re.sub(r"^# .*\n", "", text).strip()  # drop the title: the card has its own


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
HARD_CASE = """
Same clip, same click, both at 1024 px. With a fast camera zoom-out the teacher keeps the train
(lost in 0 % of the frames) while sam2-lite loses it in 28 %: large, fast changes of scale
are where the smaller encoder still falls short, consistent with its −4.7 J&F on DAVIS.
"""
HARDWARE_NOTE = (
    "- Running on a GPU, so it responds quickly; the CPU latencies above were measured on a\n"
    "  laptop CPU, which is what sam2-lite is optimised for."
    if DEVICE == "cuda"
    else f"- Running on CPU with {CFG.threads} threads, the setting of the latency numbers above."
)
LIMITS = f"""
- Only the first **{CFG.max_seconds} s** are used, subsampled to about **{CFG.target_fps} fps**,
  with the longer side reduced to {CFG.max_side} px, to stay responsive on a 2-core CPU.
- One object per run, selected with one click on the first frame. A click can be ambiguous
  (part or whole: one car or the whole train); click the centre of the object.
- Small or thin objects, fast zooms and close-ups of touching objects are the hardest cases.
{HARDWARE_NOTE}
- **Model license:** CC BY-NC 4.0, non-commercial research use only. Contains SAM 2.1 weights by
  Meta (Apache 2.0, memory attention modified); encoder pretrained on ImageNet-1k; distilled on
  DAVIS 2017 (CC BY-NC 4.0). App code: Apache 2.0; example videos: CC BY (credits below).
"""
FOOTER = (
    '<div id="footer">sam2-lite is an independent project; it is not affiliated with, endorsed '
    "by or sponsored by Meta. “SAM 2” refers to the original model by Meta FAIR.</div>"
)


def step(number: int, title: str) -> None:
    gr.HTML(f'<div class="step-title"><span class="num">{number}</span>{title}</div>')


def section(title: str) -> None:
    gr.HTML(f'<div class="section-title">{title}</div>')


with gr.Blocks(title="sam2-lite · fast video object tracking") as demo:
    gr.HTML(HERO)
    video_state, point_state = gr.State(), gr.State()
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                step(1, "Upload a short video or pick an example")
                video_in = gr.Video(show_label=False, sources=["upload"], height=260)
                gr.Examples(
                    [[str(EXAMPLES_DIR / f"{v.name}.mp4")] for v in CFG.examples.videos],
                    inputs=video_in,
                    label="Examples (click preset: just press Track)",
                )
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
                section("How it works")
                gr.Markdown(HOW_IT_WORKS, elem_classes="info")
            with gr.Group(elem_classes="card"):
                section("Where the teacher still wins")
                gr.Video(
                    str(EXAMPLES_DIR / f"{HARD.name}_teacher_vs_student.mp4"),
                    show_label=False,
                    interactive=False,
                    loop=True,
                )
                gr.Markdown(HARD_CASE, elem_classes="info")
            with gr.Group(elem_classes="card"):
                section("Limits and license")
                gr.Markdown(LIMITS, elem_classes="info")
            with gr.Group(elem_classes="card"):
                section("Example video credits")
                gr.Markdown(credits(), elem_classes="info")
    gr.HTML(FOOTER)

    video_in.change(on_upload, [video_in, model], [first_frame, video_state, point_state, status])
    first_frame.select(on_click, [video_state, model], [first_frame, point_state, status])
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
