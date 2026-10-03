"""Frame dataset for distillation, with exactly SAM 2's video-frame preprocessing.

SAM 2 (sam2/utils/misc.py: _load_img_as_tensor + load_video_frames_from_jpg_images) does:
PIL RGB -> resize to image_size x image_size (aspect ratio NOT kept, PIL default filter)
-> / 255 -> (x - mean) / std with ImageNet statistics. We repeat the same steps in the same
order so augmentations can be applied to the original frame before the resize; a test checks
that, without augmentations, the output is identical to SAM 2's.
"""

import inspect
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from PIL import Image
from sam2.utils.misc import load_video_frames
from torch.utils.data import Dataset
from torchvision.transforms import v2

# Read SAM 2's normalization constants from its own function so they can never drift apart.
_SAM2_DEFAULTS = inspect.signature(load_video_frames).parameters
IMG_MEAN = torch.tensor(_SAM2_DEFAULTS["img_mean"].default, dtype=torch.float32)[:, None, None]
IMG_STD = torch.tensor(_SAM2_DEFAULTS["img_std"].default, dtype=torch.float32)[:, None, None]


class FrameDataset(Dataset):
    """Single frames preprocessed like SAM 2; optional mild augmentations on the raw frame."""

    def __init__(
        self,
        paths: list[str],
        image_size: int = 1024,
        augment: Mapping[str, Any] | None = None,
        extra_size: int | None = None,
    ) -> None:
        """With `extra_size`, each item is (frame at image_size, same frame at extra_size): the
        same crop/flip/colour, each resized from the augmented frame with SAM 2's PIL resize
        (used to distil a reduced-resolution student from a full-resolution teacher)."""
        self.paths = paths
        self.image_size = image_size
        self.extra_size = extra_size
        self.augment = augment
        self.color = (
            v2.ColorJitter(augment["brightness"], augment["contrast"], augment["saturation"])
            if augment
            else None
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.paths[index]).convert("RGB")
        if self.augment:
            image = self._augment(image)
        if self.extra_size is None:
            return self._to_tensor(image, self.image_size)
        return self._to_tensor(image, self.image_size), self._to_tensor(image, self.extra_size)

    @staticmethod
    def _to_tensor(image: Image.Image, size: int) -> torch.Tensor:
        array = np.array(image.resize((size, size))) / 255.0  # same calls as SAM 2
        tensor = torch.from_numpy(array).permute(2, 0, 1).float()
        return (tensor - IMG_MEAN) / IMG_STD

    def _augment(self, image: Image.Image) -> Image.Image:
        # Random crop that keeps the frame's aspect ratio: after the square resize it is
        # stretched exactly like the full frames the student will see at inference time.
        width, height = image.size
        scale = self.augment["crop_scale_min"] + torch.rand(1).item() * (
            1.0 - self.augment["crop_scale_min"]
        )
        crop_w, crop_h = round(width * math.sqrt(scale)), round(height * math.sqrt(scale))
        left = int(torch.randint(0, width - crop_w + 1, (1,)))
        top = int(torch.randint(0, height - crop_h + 1, (1,)))
        image = image.crop((left, top, left + crop_w, top + crop_h))
        if torch.rand(1).item() < self.augment["hflip_p"]:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return self.color(image)
