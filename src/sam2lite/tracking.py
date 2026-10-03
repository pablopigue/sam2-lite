"""MLflow helpers: every run records its full config and the git commit that produced it."""

import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import mlflow
from omegaconf import DictConfig, OmegaConf

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"  # the Model Registry needs a database backend


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """{"model": {"name": "t"}} -> {"model.name": "t"} (MLflow params are flat)."""
    flat: dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, prefix=f"{name}."))
        else:
            flat[name] = value
    return flat


def git_state() -> dict[str, str]:
    """Current commit and whether the working tree has uncommitted changes."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout
    return {"git_commit": commit, "git_dirty": str(bool(status.strip()))}


@contextmanager
def start_run(
    experiment: str, run_name: str, cfg: DictConfig, run_id: str | None = None
) -> Iterator[mlflow.ActiveRun]:
    """Open an MLflow run that logs `cfg` (as params and as config.yaml) and the git state.

    With `run_id`, an existing run is resumed (e.g. training restarted from a checkpoint):
    its params are already logged, so only the git state of the resumed process is added.
    """
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(experiment)
    if run_id is not None:
        with mlflow.start_run(run_id=run_id) as run:
            mlflow.set_tags({f"resumed_{k}": v for k, v in git_state().items()})
            yield run
        return
    with mlflow.start_run(run_name=run_name) as run:
        container = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(container, dict)
        mlflow.log_params(flatten(container))
        mlflow.set_tags(git_state())
        mlflow.log_text(OmegaConf.to_yaml(cfg, resolve=True), "config.yaml")
        yield run


def get_run(prefix: str) -> mlflow.entities.Run:
    """Run whose id starts with `prefix` (8 characters are enough), across all experiments.

    Only runs from committed code (git_dirty=False) can be reported in the README or model card.
    """
    runs = mlflow.search_runs(search_all_experiments=True, output_format="list")
    matches = [r for r in runs if r.info.run_id.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"{len(matches)} runs match {prefix!r}")
    run = matches[0]
    if run.data.tags.get("git_dirty") != "False":
        raise ValueError(f"run {prefix} was produced with uncommitted changes: not reportable")
    return run
