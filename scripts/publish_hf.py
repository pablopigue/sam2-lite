"""Stage the sam2-lite model repo for the Hugging Face Hub, and upload it with --push.

Staging (default) builds configs/publish/hf.yaml's out_dir: the bundle, the ONNX encoders,
LICENSE + NOTICE (Apache 2.0 section 4 for the SAM 2.1 weights) and a model card whose numbers
are read from MLflow runs (git_dirty=False only), never typed by hand. Review it, then --push.

Example:
    uv run python scripts/publish_hf.py            # stage only
    uv run python scripts/publish_hf.py --push     # create the repo (private) and upload
"""

import argparse
import os
import shutil
from pathlib import Path
from string import Template

import mlflow
from huggingface_hub import HfApi
from omegaconf import DictConfig, OmegaConf

from sam2lite.tracking import DEFAULT_TRACKING_URI, get_run


def card_values(cfg: DictConfig) -> dict[str, str]:
    """Every number of the model card, from the MLflow runs listed in the config."""
    runs = cfg.runs

    def metrics(prefix: str) -> dict[str, float]:
        return get_run(prefix).data.metrics

    ev = {k: metrics(v)["JF"] for k, v in runs.eval.items()}
    lat = {k: metrics(v) for k, v in runs.latency.items()}
    onnx = {k: metrics(v) for k, v in runs.onnx.items()}
    quality = {k: metrics(v)["JF"] for k, v in runs.onnx_quality.items()}

    def change(size: int, threads: int, workload: str) -> str:
        key = f"cpu{threads}t_{workload}_median_ms"
        before, after = onnx[f"torch_{size}"][key], onnx[f"onnx_{size}"][key]
        return f"{before:.0f} → {after:.0f} ({before / after:.2f}×)"

    t6 = "cpu6t_pipeline_median_ms"
    speedup = lat["teacher"][t6] / lat["sam2_lite"][t6]
    values = {
        "jf_teacher": f"{ev['teacher']:.1f}",
        "jf_lite": f"{ev['sam2_lite']:.1f}",
        "jf_mobile": f"{ev['mobile']:.1f}",
        "jf_drop": f"{ev['teacher'] - ev['sam2_lite']:.1f}",
        "t6_teacher": f"{lat['teacher']['cpu6t_pipeline_median_ms']:.0f}",
        "t6_lite": f"{lat['sam2_lite']['cpu6t_pipeline_median_ms']:.0f}",
        "t2_teacher": f"{lat['teacher']['cpu2t_pipeline_median_ms']:.0f}",
        "t2_lite": f"{lat['sam2_lite']['cpu2t_pipeline_median_ms']:.0f}",
        "t2_mobile": f"{lat['mobile']['cpu2t_pipeline_median_ms']:.0f}",
        "speedup_6t": f"{speedup:.2f}",
        "mb_teacher": f"{lat['teacher']['model_size_mb']:.0f}",
        "mb_lite": f"{lat['sam2_lite']['model_size_mb']:.0f}",
        "jf_torch5": f"{quality['torch']:.2f}",
        "jf_onnx5": f"{quality['onnx']:.2f}",
        "cpu": get_run(runs.latency.sam2_lite).data.tags.get("cpu", "unknown CPU"),
        "repo_id": cfg.repo_id,
        "github": cfg.github,
    }
    for size in (1024, 576):
        for threads in (6, 2):
            values[f"onnx_enc_{size}_{threads}t"] = change(size, threads, "encoder")
            values[f"onnx_pipe_{size}_{threads}t"] = change(size, threads, "pipeline")
    all_runs = [v for group in runs.values() for v in group.values()]
    values["runs_line"] = ", ".join(f"`{r}`" for r in all_runs)
    return values


def stage(cfg: DictConfig) -> Path:
    out_dir = Path(cfg.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)  # rebuilt from scratch: no stale file can be uploaded
    out_dir.mkdir(parents=True)
    for name in ("model.safetensors", "config.yaml"):
        shutil.copy2(Path(cfg.bundle_dir) / name, out_dir / name)
    for path in cfg.onnx:
        shutil.copy2(path, out_dir / Path(path).name)
    for name in ("LICENSE", "NOTICE"):
        shutil.copy2(name, out_dir / name)
    # substitute (not safe_substitute): a placeholder without a value is an error.
    card = Template(Path(cfg.template).read_text()).substitute(card_values(cfg))
    (out_dir / "README.md").write_text(card)
    return out_dir


def load_env_file(path: Path = Path(".env")) -> None:
    """KEY=VALUE lines from .env into os.environ (existing variables win). Values never printed."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def push(cfg: DictConfig, out_dir: Path) -> None:
    load_env_file()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is empty: add a write token to .env (see .env)")
    api = HfApi(token=token)
    api.create_repo(cfg.repo_id, repo_type="model", private=cfg.private, exist_ok=True)
    commit = api.upload_folder(
        repo_id=cfg.repo_id, folder_path=out_dir, commit_message="Upload sam2-lite"
    )
    print(f"Uploaded to https://huggingface.co/{cfg.repo_id} ({commit.oid[:8]})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/publish/hf.yaml"))
    parser.add_argument("--push", action="store_true", help="upload the staged folder")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    assert isinstance(cfg, DictConfig)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))

    out_dir = stage(cfg)
    for path in sorted(out_dir.iterdir()):
        print(f"  {path.name:<28} {path.stat().st_size / 2**20:8.1f} MB")
    print(f"Staged {out_dir}: review README.md before uploading.")
    if args.push:
        push(cfg, out_dir)


if __name__ == "__main__":
    main()
