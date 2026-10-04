"""Stage the demo as a Hugging Face Space, and create/update it with --push.

Staging (default) builds configs/publish/space.yaml's out_dir with the Space layout:
    README.md          Space config (YAML header) + short description + license of each part
    LICENSE, NOTICE    the code's Apache 2.0 license; NOTICE gives every part its license
    requirements.txt   Python packages;  packages.txt: apt packages (ffmpeg)
    app/               app.py, style.css, examples/ and a copy of the sam2lite package
    configs/app.yaml   the app's settings
The app runs as `python app/app.py` from the Space root, so `app/` is on sys.path: the package
copy lives there. Review the folder, then --push creates the private Space (if missing), adds
HF_TOKEN as a secret (only while the model repo is private) and SAM2LITE_DEVICE as a variable,
uploads, and requests the configured hardware (or restarts the Space if it was paused).

Example:
    uv run python scripts/build_space.py            # stage only
    uv run python scripts/build_space.py --push     # create/update the private Space
"""

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi
from omegaconf import DictConfig, OmegaConf

# sam2lite modules the demo imports (demo -> export.bundle -> export.ort_encoder -> models);
# training, evaluation, benchmarks and MLflow helpers stay out of the Space.
PACKAGE_FILES = [
    "__init__.py",
    "demo.py",
    "export/__init__.py",
    "export/bundle.py",
    "export/ort_encoder.py",
    "models/__init__.py",
    "models/memory.py",
    "models/student.py",
]


def readme(cfg: DictConfig) -> str:
    header = OmegaConf.to_yaml(cfg.readme)
    model = f"[{cfg.readme.models[0]}](https://huggingface.co/{cfg.readme.models[0]})"
    body = (
        "# sam2-lite demo\n\n"
        "Click an object in the first frame of a short video and it is tracked through the video "
        "with sam2-lite: SAM 2.1 Hiera-tiny with a distilled MobileNetV4 image encoder and a "
        f"memory attention that attends to 3 frames. Model: {model}.\n\n"
        "sam2-lite is an independent project; it is not affiliated with, endorsed by or "
        "sponsored by Meta.\n\n"
        # One license per part: the Space's license tag covers its code only.
        "## License\n\n"
        "- **Code:** Apache License 2.0 (`LICENSE`); `NOTICE` gives every part its license.\n"
        f"- **Model** ({model}): CC BY-NC 4.0, non-commercial research use only.\n"
        "- **Example videos** (`app/examples/`): CC BY 2.0 and CC BY 3.0, credits in "
        "`app/examples/ATTRIBUTION.md`.\n"
    )
    return f"---\n{header}---\n\n{body}"


def stage(cfg: DictConfig) -> Path:
    out = Path(cfg.out_dir)
    if out.exists():
        shutil.rmtree(out)  # rebuilt from scratch: no stale file can be uploaded
    (out / "app").mkdir(parents=True)
    (out / "README.md").write_text(readme(cfg))
    for name in ("LICENSE", "NOTICE"):
        shutil.copy2(name, out / name)
    (out / "requirements.txt").write_text("\n".join(cfg.requirements) + "\n")
    (out / "packages.txt").write_text("\n".join(cfg.packages) + "\n")
    for name in ("app.py", "style.css"):
        shutil.copy2(Path("app") / name, out / "app" / name)
    shutil.copytree("app/examples", out / "app/examples")
    (out / "configs").mkdir()
    shutil.copy2("configs/app.yaml", out / "configs/app.yaml")
    for rel in PACKAGE_FILES:
        target = out / "app/sam2lite" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path("src/sam2lite") / rel, target)
    return out


def read_token() -> str:
    """HF_TOKEN from the environment or .env (never printed)."""
    if not os.environ.get("HF_TOKEN") and Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if line.strip().startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is empty: add a write token to .env")
    return token


def push(cfg: DictConfig, out: Path) -> None:
    token = read_token()
    api = HfApi(token=token)
    if not api.repo_exists(cfg.repo_id, repo_type="space"):
        # Creating a ZeroGPU Space through the API can be refused (402) for free accounts that
        # can still create it on the website (D-046): then create it there and push again.
        # The hardware must be chosen at creation: the default, cpu-basic, needs PRO.
        api.create_repo(
            cfg.repo_id,
            repo_type="space",
            space_sdk="gradio",
            space_hardware=cfg.hardware,
            private=cfg.private,
        )
    if api.model_info(cfg.readme.models[0]).private:  # a public model needs no token
        api.add_space_secret(cfg.repo_id, "HF_TOKEN", token)  # lets the Space read the model
    for key, value in cfg.variables.items():
        api.add_space_variable(cfg.repo_id, key, str(value))
    api.upload_folder(
        repo_id=cfg.repo_id, repo_type="space", folder_path=out, commit_message="Update demo"
    )
    runtime = api.get_space_runtime(cfg.repo_id)
    if runtime.requested_hardware != cfg.hardware:
        api.request_space_hardware(cfg.repo_id, cfg.hardware)
    elif runtime.stage == "PAUSED":  # a new commit does not wake a paused Space up
        api.restart_space(cfg.repo_id)
    print(f"Pushed to https://huggingface.co/spaces/{cfg.repo_id} (private={cfg.private})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/publish/space.yaml"))
    parser.add_argument("--push", action="store_true", help="create/update the Space")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    assert isinstance(cfg, DictConfig)
    out = stage(cfg)
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        print(f"  {path.relative_to(out)}")
    print(f"Staged {out}.")
    if args.push:
        push(cfg, out)


if __name__ == "__main__":
    main()
