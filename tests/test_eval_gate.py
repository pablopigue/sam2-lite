"""The CI eval gate's condition and its configuration (no model or dataset needed)."""

import importlib.util

import pytest
from omegaconf import OmegaConf

spec = importlib.util.spec_from_file_location("eval_gate", "scripts/eval_gate.py")
eval_gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_gate)


@pytest.mark.parametrize(
    ("jf", "expected"),
    [(81.8, True), (80.8, True), (80.79, False), (60.0, False)],  # reference 81.8, tolerance 1.0
)
def test_gate_fails_only_beyond_the_tolerance(jf: float, expected: bool) -> None:
    assert eval_gate.passes(jf, reference=81.8, tolerance=1.0) is expected


def test_gate_config_is_complete() -> None:
    gate = OmegaConf.load("configs/eval_gate.yaml").ci
    assert len(gate.videos) >= 1 and gate.tolerance > 0
    assert gate.reference_jf - gate.tolerance > 0
