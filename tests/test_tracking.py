"""MLflow helpers, against a throwaway SQLite database."""

from pathlib import Path

import mlflow
import pytest
from omegaconf import OmegaConf

from sam2lite.tracking import flatten, start_run


def test_flatten_nested_dict() -> None:
    assert flatten({"a": 1, "b": {"c": 2, "d": {"e": [3]}}}) == {"a": 1, "b.c": 2, "b.d.e": [3]}


def test_start_run_logs_config_and_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # artifacts are written relative to the working directory
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(
        "sam2lite.tracking.git_state", lambda: {"git_commit": "abc", "git_dirty": "False"}
    )
    cfg = OmegaConf.create({"model": {"name": "teacher"}, "split": "val"})

    with start_run("test-exp", "test-run", cfg) as run:
        mlflow.log_metric("JF", 89.1)  # MLflow metric names cannot contain "&"

    logged = mlflow.get_run(run.info.run_id).data
    assert logged.params == {"model.name": "teacher", "split": "val"}
    assert logged.metrics == {"JF": 89.1}
    assert logged.tags["git_commit"] == "abc"
