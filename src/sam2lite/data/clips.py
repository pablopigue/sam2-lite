"""Video clips for distilling SAM 2's memory attention (Plan C2).

Each sample is L consecutive frames of one DAVIS *train* video (SAM 2 preprocessing, via
FrameDataset) plus the ground-truth mask of ONE object in the first frame, prepared like
SAM2VideoPredictor.add_new_mask does (bilinear + antialias resize to image_size, >= 0.5).
Train clips start every `stride` frames with a random object; val clips start at frame 0 and
enumerate every object, so the validation set is fixed.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from sam2lite.data.davis import annotation_paths, frame_paths, load_annotation, split_objects
from sam2lite.data.frames import FrameDataset

IGNORE_ID = 255  # "ignore" label present in some DAVIS train annotations (e.g. tennis)


def prepare_mask(mask: np.ndarray, image_size: int) -> torch.Tensor:
    """Bool (H, W) mask -> float [1, 1, S, S] in {0, 1}, exactly like add_new_mask."""
    tensor = torch.from_numpy(mask)[None, None].float()
    if tensor.shape[-2:] != (image_size, image_size):
        tensor = F.interpolate(
            tensor,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        tensor = (tensor >= 0.5).float()
    return tensor


class MemoryClipDataset(Dataset):
    """(frames [L, 3, S, S], first-frame object mask [1, 1, S, S]) clips from DAVIS."""

    def __init__(
        self,
        davis_root: Path,
        videos: list[str],
        clip_len: int,
        image_size: int = 1024,
        stride: int = 4,
        train: bool = True,
        hflip_p: float = 0.5,
    ) -> None:
        self.davis_root, self.clip_len, self.image_size = davis_root, clip_len, image_size
        self.train, self.hflip_p = train, hflip_p
        self.items: list[tuple[str, int, int | None]] = []  # (video, start, object or None)
        for video in videos:
            n_frames = len(frame_paths(davis_root, video))
            if n_frames < clip_len:
                continue
            if train:
                for start in range(0, n_frames - clip_len + 1, stride):
                    self.items.append((video, start, None))  # object drawn in __getitem__
            else:
                ids, _ = load_annotation(annotation_paths(davis_root, video)[0])
                for object_id in self._objects(ids):
                    self.items.append((video, 0, object_id))

    @staticmethod
    def _objects(ids: np.ndarray) -> list[int]:
        return [o for o in split_objects(ids) if o != IGNORE_ID]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        video, start, object_id = self.items[index]
        paths = [str(p) for p in frame_paths(self.davis_root, video)[start : start + self.clip_len]]
        frames = torch.stack([FrameDataset(paths, self.image_size)[i] for i in range(len(paths))])

        ids, _ = load_annotation(annotation_paths(self.davis_root, video)[start])
        if object_id is None:  # train: random object visible in the first frame
            objects = self._objects(ids)
            if not objects:  # rare: nothing visible at this start, fall back to the next item
                return self[(index + 1) % len(self)]
            object_id = objects[int(torch.randint(len(objects), (1,)))]
        mask = prepare_mask(ids == object_id, self.image_size)

        if self.train and torch.rand(1).item() < self.hflip_p:  # same flip for the whole clip
            frames, mask = frames.flip(-1), mask.flip(-1)
        return frames, mask
