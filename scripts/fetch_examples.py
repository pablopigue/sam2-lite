"""Download the demo's example videos from Wikimedia Commons, cut them and write the attribution.

Only CC0 and CC BY files are accepted (checked in the file's own metadata, not assumed). Each
video is cut to `seconds`, scaled to `height`, stripped of audio and re-encoded as H.264 small
enough for git (< 1 MB). CC BY requires credit, a licence link and noting the changes: all go
into app/examples/ATTRIBUTION.md.

Example:
    uv run python scripts/fetch_examples.py
"""

import json
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

from omegaconf import OmegaConf

API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia asks API clients to identify themselves.
USER_AGENT = "sam2-lite-demo/0.1 (https://github.com/pablopigue/sam2-lite)"
ALLOWED = re.compile(r"^(CC0|CC BY \d(\.\d)?)$")  # no NC, ND or SA variants
MAX_BYTES = 1_000_000  # pre-commit blocks files over 1 MB


def get_json(params: dict) -> dict:
    url = f"{API}?{urllib.parse.urlencode({**params, 'format': 'json'})}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def file_info(title: str) -> dict[str, str]:
    """Download URL, licence and author of a Commons file, straight from its metadata."""
    data = get_json(
        {
            "action": "query",
            "titles": f"File:{title}",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist",
        }
    )
    page = next(iter(data["query"]["pages"].values()))
    if "imageinfo" not in page:
        raise ValueError(f"File:{title} not found on Commons")
    info = page["imageinfo"][0]
    meta = info["extmetadata"]
    license_name = meta["LicenseShortName"]["value"]
    if not ALLOWED.match(license_name):
        raise ValueError(f"File:{title} is {license_name!r}: only CC0 / CC BY are allowed")
    return {
        "url": info["url"],
        "page": info["descriptionurl"],
        "license": license_name,
        "license_url": meta.get("LicenseUrl", {}).get("value", ""),
        "author": re.sub(r"<[^>]+>", "", meta["Artist"]["value"]).strip(),
    }


def download(url: str, path: Path) -> None:
    if path.exists():
        return  # raw files are cached in outputs/ (git-ignored)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response:
        path.write_bytes(response.read())


def encode(raw: Path, out: Path, start: float, seconds: float, height: int) -> None:
    """Cut + scale + H.264 without audio; raise the CRF until the file fits in MAX_BYTES."""
    for crf in (26, 29, 32, 35):  # higher CRF = smaller file, lower quality
        command = [
            "ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-t", str(seconds),
            "-i", str(raw), "-an", "-vf", f"scale=-2:{height}",
            "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(out),
        ]  # fmt: skip
        subprocess.run(command, check=True)
        if out.stat().st_size < MAX_BYTES:
            return
    raise RuntimeError(f"{out} is still over {MAX_BYTES} bytes at CRF 35: shorten it")


def main() -> None:
    cfg = OmegaConf.load("configs/app.yaml").examples
    out_dir, raw_dir = Path(cfg.dir), Path("outputs/examples_raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Example videos: attribution",
        "",
        f"From Wikimedia Commons. Changes for this demo: cut to at most {cfg.seconds} s, "
        f"scaled to {cfg.height} px high, audio removed, re-encoded as H.264.",
        "",
    ]
    for item in cfg.videos:
        info = file_info(item.title)
        raw = raw_dir / item.title.replace(" ", "_")
        download(info["url"], raw)
        out = out_dir / f"{item.name}.mp4"
        seconds = item.get("seconds", cfg.seconds)  # shorter where the source has a scene cut
        encode(raw, out, item.start, seconds, cfg.height)
        print(f"{out}: {out.stat().st_size / 1e3:.0f} KB, {info['license']}, {info['author']}")
        license_text = (
            f"[{info['license']}]({info['license_url']})"
            if info["license_url"]
            else info["license"]
        )
        lines.append(
            f'- `{out.name}`: "{item.title}" by {info["author"]}, {license_text}, '
            f"from {info['page']} (seconds {item.start}-{item.start + seconds:g})."
        )
    (out_dir / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
