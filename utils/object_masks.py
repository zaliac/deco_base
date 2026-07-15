"""Foreground-mask helpers for the SAM-3D-Objects conditioning path.

SAM-3D-Objects uses the alpha channel of an RGBA image as its object mask.  DECO
keeps RGB and mask tensors separate while loading a dataset, then recreates that
RGBA contract inside ``SAM3DObjectsEncoder``.  Keeping the crop/resize operations
here makes it difficult for the two tensors to drift out of pixel alignment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np


MaskSource = Union[str, os.PathLike, np.ndarray]


def binary_object_mask(mask: np.ndarray) -> np.ndarray:
    """Return a two-dimensional float mask with values exactly 0 or 1.

    ``mask`` may be a grayscale mask, a BGR/BGRA image, or an array stored in an
    NPZ.  For BGRA data we intentionally use the alpha channel, matching
    SAM-3D-Objects' ``ALPHA_CHANNEL`` preprocessing mode.
    """
    mask = np.asarray(mask)
    if mask.ndim == 0:
        raise ValueError('Object mask must have spatial dimensions')
    if mask.ndim == 3:
        if mask.shape[-1] >= 4:
            mask = mask[..., 3]
        else:
            mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f'Expected a 2D object mask, got shape {tuple(mask.shape)}')
    return (mask > 0).astype(np.float32, copy=False)


def alpha_object_mask(image: np.ndarray) -> Optional[np.ndarray]:
    """Extract an RGBA alpha mask, or ``None`` for an RGB image."""
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] >= 4:
        return binary_object_mask(image[..., 3])
    return None


def load_object_mask(source: MaskSource, dataset_root: str = '') -> np.ndarray:
    """Load a mask from an NPZ value or a path relative to ``dataset_root``."""
    if isinstance(source, np.ndarray) and source.ndim == 0:
        source = source.item()
    if isinstance(source, bytes):
        source = source.decode()

    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if not path.is_absolute():
            path = Path(dataset_root) / path
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f'Object mask not found: {path}')
        return binary_object_mask(mask)

    return binary_object_mask(np.asarray(source))


def crop_and_resize_object_mask(
    mask: np.ndarray,
    crop_bbox: Optional[Sequence[float]],
    source_size: Sequence[int],
    output_size: Union[int, Sequence[int]] = (256, 256),
) -> np.ndarray:
    """Apply DECO's image crop and resize to an object mask with nearest sampling.

    Args:
        mask: Binary or non-binary mask in original-image coordinates.
        crop_bbox: Optional ``(x0, y0, x1, y1)`` DECO person crop.
        source_size: Original RGB image size as ``(height, width)``.
        output_size: Target size as an integer or ``(height, width)``.
    """
    mask = binary_object_mask(mask)
    source_h, source_w = (int(source_size[0]), int(source_size[1]))
    if mask.shape != (source_h, source_w):
        mask = cv2.resize(mask, (source_w, source_h), interpolation=cv2.INTER_NEAREST)

    if crop_bbox is not None:
        x0, y0, x1, y1 = (int(v) for v in crop_bbox)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f'Invalid crop_bbox: {(x0, y0, x1, y1)}')
        mask = mask[y0:y1, x0:x1]
        if mask.size == 0:
            raise ValueError(
                f'Crop {tuple(crop_bbox)} is outside object-mask image size '
                f'{(source_h, source_w)}'
            )

    if isinstance(output_size, int):
        output_h = output_w = output_size
    else:
        output_h, output_w = (int(output_size[0]), int(output_size[1]))
    mask = cv2.resize(mask, (output_w, output_h), interpolation=cv2.INTER_NEAREST)
    return binary_object_mask(mask)
