"""Register the final sam2-lite tracker (full video predictor) in the MLflow Model Registry.

Unlike scripts/register_model.py (student encoder only), this logs the whole SAM 2.1 video
predictor as configured for deployment: distilled encoder + fine-tuned memory attention + SAM 2.1
memory encoder and mask decoder, with its number of memory frames and stride.

Example:
    uv run python scripts/register_tracker.py --train-run 517352cc --eval-run 62e59f5e \
        --latency-run <run> memory_frames=3 memory_stride=4 \
        memory_ckpt=checkpoints/memory/memory_k3/best.pt
"""

import argparse
import os

import mlflow
import mlflow.pytorch
from omegaconf import DictConfig, OmegaConf
from sam2.modeling.position_encoding import PositionEmbeddingSine

from sam2lite.eval.run_vos import build_predictor
from sam2lite.tracking import DEFAULT_TRACKING_URI

DEPLOY_KEYS = ("memory_frames", "memory_stride", "memory_ckpt", "image_size")


def get_run(client: mlflow.MlflowClient, prefix: str) -> mlflow.entities.Run:
    runs = mlflow.search_runs(search_all_experiments=True, output_format="list")
    matches = [r.info.run_id for r in runs if r.info.run_id.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"{len(matches)} runs match {prefix!r}")
    return client.get_run(matches[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/eval/davis_val.yaml")
    parser.add_argument("--train-run", required=True, help="run that trained the last component")
    parser.add_argument("--eval-run", required=True)
    parser.add_argument("--latency-run", required=True)
    parser.add_argument("--name", default="sam2-lite", help="registered model name")
    parser.add_argument("--alias", default="champion")
    args, overrides = parser.parse_known_args()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    client = mlflow.MlflowClient()

    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    cfg.model.name = "student"
    cfg.model.ckpt = cfg.model.ckpt or "checkpoints/distill/night1/best.pt"
    assert isinstance(cfg, DictConfig)

    eval_run, latency_run = get_run(client, args.eval_run), get_run(client, args.latency_run)
    # Both runs must be reproducible and must have used exactly this configuration.
    for run in (eval_run, latency_run):
        if run.data.tags.get("git_dirty") != "False":
            raise ValueError(f"run {run.info.run_id[:8]} has git_dirty != False")
        for key in ("model.ckpt", *DEPLOY_KEYS):
            expected = str(OmegaConf.select(cfg, key))
            if run.data.params.get(key) != expected:
                raise ValueError(
                    f"run {run.info.run_id[:8]}: {key}={run.data.params.get(key)}, "
                    f"expected {expected}"
                )

    predictor = build_predictor(cfg, "cpu").eval()
    # Positional-encoding caches (~106 MB) are recomputed on the first forward: don't store them.
    for module in predictor.modules():
        if isinstance(module, PositionEmbeddingSine):
            module.cache = {}

    with mlflow.start_run(run_id=get_run(client, args.train_run).info.run_id):
        info = mlflow.pytorch.log_model(
            predictor,
            name="sam2_lite_tracker",
            registered_model_name=args.name,
            serialization_format="pickle",  # see scripts/register_model.py
        )
    version = info.registered_model_version
    m_eval, m_lat = eval_run.data.metrics, latency_run.data.metrics
    tags = {
        "encoder_ckpt": cfg.model.ckpt,
        **{key: str(cfg[key]) for key in DEPLOY_KEYS},
        "jf_davis_val": f"{m_eval['JF']:.2f}",
        "j": f"{m_eval['J']:.2f}",
        "f": f"{m_eval['F']:.2f}",
        "eval_run": eval_run.info.run_id,
        "latency_run": latency_run.info.run_id,
    }
    for threads in (6, 2):  # whatever the latency run measured
        key = f"cpu{threads}t_pipeline_median_ms"
        if key in m_lat:
            tags[f"cpu{threads}t_pipeline_ms"] = f"{m_lat[key]:.0f}"
    for key, value in tags.items():
        client.set_model_version_tag(args.name, version, key, value)
    client.set_registered_model_alias(args.name, args.alias, version)
    print(f"Registered {args.name} v{version} (alias '{args.alias}')")
    print(tags)


if __name__ == "__main__":
    main()
