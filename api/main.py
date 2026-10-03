"""HTTP API: POST a short video and a click, get the tracked object back as an mp4.

Same logic as the Gradio app (sam2lite.demo), for programs instead of people. CPU only, with
the ONNX Runtime encoder, like the measured deployment target.

Run locally:
    make api                # page: http://127.0.0.1:8000/  ·  docs: http://127.0.0.1:8000/docs
    curl -F video=@app/examples/swans.mp4 -F x=660 -F y=225 \
        http://127.0.0.1:8000/track -o tracked.mp4
"""

import os
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from omegaconf import OmegaConf
from sam2.sam2_video_predictor import SAM2VideoPredictor

from sam2lite.demo import load_tracker, overlay_masks, read_video, track, write_mp4

CFG = OmegaConf.load(os.environ.get("SAM2LITE_APP_CONFIG", "configs/app.yaml"))
TRACKERS: dict[str, SAM2VideoPredictor] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load every model once at start-up (a few seconds each), not on every request."""
    for name in CFG.models:
        TRACKERS[name] = load_tracker(CFG, name, "cpu")
    yield
    TRACKERS.clear()


app = FastAPI(title="sam2-lite", version="0.1.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """A minimal web page that calls /track (upload, click, see the result)."""
    return FileResponse(Path(__file__).with_name("static") / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models": sorted(TRACKERS)}


def original_width(path: str) -> int:
    capture = cv2.VideoCapture(path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    capture.release()
    return width


@app.post("/track", response_class=Response, responses={200: {"content": {"video/mp4": {}}}})
def track_video(
    video: Annotated[UploadFile, File(description="Short video; only the first seconds are used")],
    x: Annotated[int, Form(description="Click x, in pixels of the original video")],
    y: Annotated[int, Form(description="Click y, in pixels of the original video")],
    model: Annotated[str, Form(description="sam2-lite-mobile or sam2-lite")] = CFG.default_model,
) -> Response:
    """Track the object under (x, y) on the first frame; return an H.264 mp4 with its mask.

    A plain `def` (not `async`): FastAPI runs it in a worker thread, so the CPU-heavy tracking
    does not block the server's event loop.
    """
    if model not in TRACKERS:
        raise HTTPException(422, f"unknown model {model!r}; choose one of {sorted(TRACKERS)}")
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / (video.filename or "upload.mp4"))
        Path(path).write_bytes(video.file.read())
        try:
            frames, fps = read_video(path, CFG.max_seconds, CFG.target_fps, CFG.max_side)
        except ValueError as error:  # not a readable video
            raise HTTPException(400, str(error)) from error
        width = original_width(path)
        # read_video may have downscaled the frames (max_side): move the click with them.
        scale = frames[0].shape[1] / width if width else 1.0
        px, py = round(x * scale), round(y * scale)
        height_px, width_px = frames[0].shape[:2]
        if not (0 <= px < width_px and 0 <= py < height_px):
            raise HTTPException(422, f"click ({x}, {y}) is outside the {width}-px-wide video")
        masks = track(TRACKERS[model], frames, [(1, px, py, 1)])  # one positive click
        out_path = str(Path(tmp) / "tracked.mp4")
        write_mp4([overlay_masks(f, m) for f, m in zip(frames, masks, strict=True)], fps, out_path)
        content = Path(out_path).read_bytes()
    seconds = time.perf_counter() - start
    headers = {
        "X-Model": model,
        "X-Frames": str(len(frames)),
        "X-Seconds-Per-Frame": f"{seconds / len(frames):.3f}",
    }
    return Response(content, media_type="video/mp4", headers=headers)
