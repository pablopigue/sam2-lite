"""Smoke tests: the environment is installed and both packages import."""

import sam2

import sam2lite


def test_sam2lite_exposes_version() -> None:
    assert isinstance(sam2lite.__version__, str)


def test_sam2_is_importable() -> None:
    assert sam2.__file__ is not None
