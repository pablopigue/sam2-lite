"""CI eval gate: J&F of the released tracker on a few DAVIS val videos must not regress.

Runs the deployed configuration (bundle + ONNX Runtime encoder, CPU) on configs/eval_gate.yaml's
`ci.videos` with the official protocol (first-frame GT masks, SAM 2's J&F evaluator), and fails
(exit code 1) if J&F < reference_jf - tolerance. The reference is the champion's score on the same
videos, from an MLflow run (docs D-050). No MLflow here: CI has no tracking database.

Example:
    uv run python scripts/eval_gate.py          # local: uses checkpoints/ if present
"""

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from omegaconf import DictConfig, OmegaConf

from sam2lite.eval.jf import evaluate
from sam2lite.eval.run_vos import build_predictor, predict_video


def model_files(gate: DictConfig) -> tuple[str, str]:
    """(bundle dir, ONNX file): local checkpoints if present, else the private Hub repo."""
    if Path(gate.local_bundle).exists() and Path(gate.local_onnx).exists():
        return gate.local_bundle, gate.local_onnx
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("no local checkpoints and no HF_TOKEN: add it as a GitHub Actions secret")
    path = snapshot_download(gate.model_repo, token=os.environ["HF_TOKEN"])
    return path, str(Path(path) / gate.onnx_file)


def passes(jf: float, reference: float, tolerance: float) -> bool:
    """The gate condition: no regression beyond `tolerance` J&F points."""
    return jf >= reference - tolerance


def write_summary(text: str) -> None:
    """Markdown summary shown on the GitHub Actions run page (no-op outside Actions)."""
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text + "\n")


def main() -> None:
    gate = OmegaConf.load("configs/eval_gate.yaml").ci
    bundle, onnx = model_files(gate)
    cfg = OmegaConf.merge(
        OmegaConf.load("configs/eval/davis_val.yaml"),
        {"model": {"name": "student", "onnx": onnx}, "bundle": bundle, "out_dir": "outputs/gate"},
    )
    assert isinstance(cfg, DictConfig)
    davis_root, out_dir = Path(cfg.davis_root), Path(cfg.out_dir)
    predictor = build_predictor(cfg, "cpu")  # ONNX Runtime encoder: CPU only
    for video in gate.videos:
        frames = predict_video(predictor, davis_root, video, out_dir, cfg.score_thresh)
        print(f"{video}: {frames} frames")
    scores = evaluate(davis_root, out_dir, list(gate.videos))

    threshold = gate.reference_jf - gate.tolerance
    ok = passes(scores["JF"], gate.reference_jf, gate.tolerance)
    verdict = "PASS" if ok else "FAIL"
    table = (
        f"### Eval gate: {verdict}\n\n"
        f"| | J&F | J | F |\n|---|---|---|---|\n"
        f"| sam2-lite (this commit) | **{scores['JF']:.2f}** | {scores['J']:.2f} | "
        f"{scores['F']:.2f} |\n"
        f"| threshold (champion - {gate.tolerance}) | {threshold:.2f} | | |\n"
        f"| teacher, for context | {gate.teacher_jf:.2f} | | |\n\n"
        f"Videos: {', '.join(gate.videos)} (DAVIS 2017 val), CPU + ONNX Runtime encoder."
    )
    print(table)
    write_summary(table)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
