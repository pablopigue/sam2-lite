"""DAVIS 2017 helpers: video lists, frame/annotation paths and palette PNG masks.

Layout (semi-supervised, 480p):
    DAVIS/JPEGImages/480p/<video>/00000.jpg
    DAVIS/Annotations/480p/<video>/00000.png   # palette PNG: pixel value = object id, 0 = bg
    DAVIS/ImageSets/2017/{train,val}.txt       # one video name per line

Mask I/O follows tools/vos_inference.py from the SAM 2 repo, so predictions saved here can
be scored by its evaluator.
"""

from pathlib import Path

import numpy as np
from PIL import Image

RESOLUTION = "480p"


def list_videos(davis_root: Path, split: str) -> list[str]:
    """Video names of a DAVIS 2017 split ("train" or "val"), in file order."""
    split_file = davis_root / "ImageSets" / "2017" / f"{split}.txt"
    return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]


def frame_paths(davis_root: Path, video: str) -> list[Path]:
    """Sorted JPEG frame paths of a video."""
    return sorted((davis_root / "JPEGImages" / RESOLUTION / video).glob("*.jpg"))


def annotation_paths(davis_root: Path, video: str) -> list[Path]:
    """Sorted ground-truth PNG paths of a video (one per frame)."""
    return sorted((davis_root / "Annotations" / RESOLUTION / video).glob("*.png"))


def load_annotation(path: Path) -> tuple[np.ndarray, list[int] | None]:
    """Load a palette PNG as an (H, W) uint8 array of object ids, plus its palette.

    Reading the raw indices matters: converting to RGB would give colours, not ids.
    """
    image = Image.open(path)
    if image.mode != "P":
        raise ValueError(f"Expected a palette ('P') PNG, got mode {image.mode!r}: {path}")
    return np.array(image, dtype=np.uint8), image.getpalette()


def save_annotation(path: Path, ids: np.ndarray, palette: list[int] | None) -> None:
    """Save an (H, W) uint8 array of object ids as a palette PNG (same format as the GT)."""
    if ids.dtype != np.uint8 or ids.ndim != 2:
        raise ValueError(f"Expected a 2D uint8 array, got {ids.dtype} with shape {ids.shape}")
    image = Image.fromarray(ids, mode="P")
    if palette is not None:
        image.putpalette(palette)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def split_objects(ids: np.ndarray) -> dict[int, np.ndarray]:
    """Split an (H, W) array of object ids into one boolean (H, W) mask per object id."""
    object_ids = np.unique(ids)
    # 0 is background.
    object_ids = object_ids[object_ids > 0]
    return {int(object_id): ids == object_id for object_id in object_ids}


def merge_objects(masks: dict[int, np.ndarray], height: int, width: int) -> np.ndarray:
    """Inverse of `split_objects`: pack boolean per-object masks into an (H, W) uint8 id array.

    If masks overlap, the lowest object id wins (as in SAM 2's tools/vos_inference.py).
    """
    ids = np.zeros((height, width), dtype=np.uint8)
    for object_id in sorted(masks, reverse=True):
        ids[masks[object_id].reshape(height, width)] = object_id
    return ids
