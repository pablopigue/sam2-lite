"""Register a distilled student encoder in the MLflow Model Registry.

The model is logged into its training run (so its lineage is the training config and git
commit), registered as a new version, tagged with its evaluation results and given an alias.

Example:
    uv run python scripts/register_model.py --ckpt checkpoints/distill/night1/best.pt \
        --train-run 44c5146a --eval-run c27ca8b7 --latency-run 575ec43c --alias champion
"""

import argparse
import os

import mlflow
import mlflow.pytorch
import torch
from omegaconf import DictConfig, OmegaConf

from sam2lite.models.student import build_student
from sam2lite.tracking import DEFAULT_TRACKING_URI

MODEL_NAME = "sam2-lite-student"


def get_run(client: mlflow.MlflowClient, prefix: str) -> mlflow.entities.Run:
    runs = mlflow.search_runs(search_all_experiments=True, output_format="list")
    matches = [r.info.run_id for r in runs if r.info.run_id.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"{len(matches)} runs match {prefix!r}")
    return client.get_run(matches[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True, help="distillation checkpoint (best.pt)")
    parser.add_argument("--student-config", default="configs/model/student_mnv4.yaml")
    parser.add_argument("--train-run", required=True, help="distill run that produced --ckpt")
    parser.add_argument("--eval-run", required=True, help="vos-davis run of --ckpt")
    parser.add_argument("--latency-run", required=True, help="latency run of --ckpt")
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    client = mlflow.MlflowClient()

    train_run = get_run(client, args.train_run)
    eval_run, latency_run = get_run(client, args.eval_run), get_run(client, args.latency_run)
    # Only report runs that are reproducible and that really used this checkpoint.
    for run in (eval_run, latency_run):
        if run.data.tags.get("git_dirty") != "False":
            raise ValueError(f"run {run.info.run_id[:8]} has git_dirty != False")
        if run.data.params.get("model.ckpt") != args.ckpt:
            raise ValueError(f"run {run.info.run_id[:8]} used {run.data.params.get('model.ckpt')}")

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.merge(OmegaConf.load(args.student_config), {"pretrained": False})
    assert isinstance(cfg, DictConfig)
    student = build_student(cfg)
    student.load_state_dict(state["student"], strict=True)
    student.eval()

    with mlflow.start_run(run_id=train_run.info.run_id):
        info = mlflow.pytorch.log_model(
            student,
            name="student_encoder",
            registered_model_name=MODEL_NAME,
            extra_files=[args.student_config],
        )
    version = info.registered_model_version

    tags = {
        "checkpoint": args.ckpt,
        "best_step": str(state["step"]),
        "jf_davis_val": f"{eval_run.data.metrics['JF']:.1f}",
        "j": f"{eval_run.data.metrics['J']:.1f}",
        "f": f"{eval_run.data.metrics['F']:.1f}",
        "cpu6t_pipeline_ms": f"{latency_run.data.metrics['cpu6t_pipeline_median_ms']:.0f}",
        "encoder_params_M": f"{latency_run.data.metrics['encoder_params_M']:.2f}",
        "eval_run": eval_run.info.run_id,
        "latency_run": latency_run.info.run_id,
    }
    for key, value in tags.items():
        client.set_model_version_tag(MODEL_NAME, version, key, value)
    client.set_registered_model_alias(MODEL_NAME, args.alias, version)
    print(
        f"Registered {MODEL_NAME} v{version} (alias '{args.alias}') "
        f"from training run {train_run.info.run_id[:8]}"
    )
    print(tags)


if __name__ == "__main__":
    main()
