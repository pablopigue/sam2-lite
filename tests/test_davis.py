"""DAVIS mask I/O on synthetic palette PNGs (no dataset needed, so it runs in CI)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sam2lite.data.davis import load_annotation, merge_objects, save_annotation, split_objects

PALETTE = [0, 0, 0, 128, 0, 0, 0, 128, 0] + [0] * (256 * 3 - 9)  # bg, obj 1, obj 2


@pytest.fixture
def ids() -> np.ndarray:
    ids = np.zeros((4, 6), dtype=np.uint8)
    ids[0:2, 0:2] = 1
    ids[2:4, 3:6] = 2
    return ids


def test_save_then_load_round_trip(tmp_path: Path, ids: np.ndarray) -> None:
    path = tmp_path / "video" / "00000.png"
    save_annotation(path, ids, PALETTE)
    loaded, palette = load_annotation(path)
    np.testing.assert_array_equal(loaded, ids)
    assert palette[:9] == PALETTE[:9]


def test_load_rejects_non_palette_png(tmp_path: Path) -> None:
    path = tmp_path / "rgb.png"
    Image.new("RGB", (6, 4)).save(path)
    with pytest.raises(ValueError, match="palette"):
        load_annotation(path)


def test_split_objects(ids: np.ndarray) -> None:
    masks = split_objects(ids)
    assert list(masks) == [1, 2]  # background (0) is not an object
    assert all(type(k) is int for k in masks)
    assert all(m.dtype == bool and m.shape == ids.shape for m in masks.values())
    assert masks[1].sum() == 4 and masks[2].sum() == 6
    assert not (masks[1] & masks[2]).any()


def test_split_objects_empty_frame() -> None:
    assert split_objects(np.zeros((4, 6), dtype=np.uint8)) == {}


def test_merge_is_inverse_of_split(ids: np.ndarray) -> None:
    np.testing.assert_array_equal(merge_objects(split_objects(ids), *ids.shape), ids)


def test_merge_overlap_lowest_id_wins() -> None:
    full = np.ones((2, 2), dtype=bool)
    assert (merge_objects({1: full, 2: full}, 2, 2) == 1).all()
