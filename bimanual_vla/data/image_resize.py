"""Shared RGB resize for collection and model input, without padding/cropping."""
from dataclasses import dataclass
import numpy as np
from PIL import Image


def resize_stretch(images, height, width):
    images = np.asarray(images)
    if images.ndim < 3 or images.dtype != np.uint8 or images.shape[-1] != 3:
        raise ValueError("stretch resize expects uint8 RGB HWC images")
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if images.shape[-3:-1] == (height, width):
        return images
    shape = images.shape
    batch = images.reshape(-1, *shape[-3:])
    result = np.stack([np.asarray(Image.fromarray(frame).resize(
        (width, height), Image.Resampling.BILINEAR)) for frame in batch])
    return result.reshape(*shape[:-3], height, width, 3)


@dataclass(frozen=True)
class StretchResizeImages:
    height: int
    width: int

    def __call__(self, data):
        data["image"] = {key: resize_stretch(value, self.height, self.width)
                         for key, value in data["image"].items()}
        return data


@dataclass(frozen=True)
class PreparedImages:
    height: int
    width: int

    def __call__(self, data):
        for key, value in data["image"].items():
            if value.shape[-3:] != (self.height, self.width, 3) or value.dtype != np.uint8:
                raise ValueError(f"{key}: expected prepared uint8 RGB {self.height}x{self.width}")
        return data
