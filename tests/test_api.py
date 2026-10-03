"""HTTP API contract with a fake tracker (no model, runs in seconds; the real model is tested by
running the API: `make api` / Docker)."""

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api_main


@pytest.fixture
def client(monkeypatch) -> TestClient:
    """The API with a fake loader and a fake tracker that masks the top-left quarter."""

    def fake_track(_, frames, clicks):
        height, width = frames[0].shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        mask[: height // 2, : width // 2] = True
        fake_track.clicks = clicks
        return [{1: mask} for _ in frames]

    monkeypatch.setattr(api_main, "load_tracker", lambda cfg, name, device: object())
    monkeypatch.setattr(api_main, "track", fake_track)
    with TestClient(api_main.app) as test_client:  # `with` runs the lifespan (model loading)
        test_client.fake_track = fake_track
        yield test_client


@pytest.fixture
def video(tmp_path: Path) -> Path:
    """1 s, 30 fps, 1708x960: wider than max_side (854), so the API must rescale the click."""
    path = tmp_path / "in.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1708, 960))
    for i in range(30):
        writer.write(np.full((960, 1708, 3), 8 * i, dtype=np.uint8))
    writer.release()
    return path


def post(client: TestClient, video: Path, **form) -> object:
    with video.open("rb") as f:
        return client.post("/track", files={"video": ("in.mp4", f, "video/mp4")}, data=form)


def test_index_serves_the_web_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"] and "/track" in response.text


def test_health_lists_the_loaded_models(client: TestClient) -> None:
    assert client.get("/health").json() == {
        "status": "ok",
        "models": sorted(api_main.CFG.models),
    }


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the system ffmpeg")
def test_track_returns_an_h264_video_and_rescales_the_click(
    client: TestClient, video: Path, tmp_path: Path
) -> None:
    response = post(client, video, x=1000, y=500)
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["x-model"] == api_main.CFG.default_model
    assert int(response.headers["x-frames"]) > 0
    assert client.fake_track.clicks == [(1, 500, 250, 1)]  # 1708 -> 854 px wide: halved
    out = tmp_path / "out.mp4"
    out.write_bytes(response.content)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert probe.stdout.strip() == "h264,854"


def test_track_rejects_a_click_outside_the_video(client: TestClient, video: Path) -> None:
    assert post(client, video, x=5000, y=10).status_code == 422


def test_track_rejects_an_unknown_model(client: TestClient, video: Path) -> None:
    assert post(client, video, x=10, y=10, model="nope").status_code == 422


def test_track_rejects_a_file_that_is_not_a_video(client: TestClient, tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_text("not a video")
    assert post(client, bad, x=10, y=10).status_code == 400
