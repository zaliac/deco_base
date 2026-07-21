"""Object-mask helpers for the SAM-3D-Objects conditioning path.

SAM-3D-Objects uses the alpha channel of an RGBA image as its object mask.  DECO
keeps RGB and mask tensors separate while loading a dataset, then recreates that
RGBA contract inside ``SAM3DObjectsEncoder``.  Keeping the crop/resize operations
here makes it difficult for the two tensors to drift out of pixel alignment. Task 6
uses an independent circular point prompt around each body keypoint, removes a
separately predicted human mask, and passes the remaining union through the
single-alpha encoder path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

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


def keypoints_to_sam_points(
    keypoints: np.ndarray,
    keypoint_conf: Optional[np.ndarray],
    source_size: Sequence[int],
    confidence_threshold: float = 0.3,
):
    """Convert valid 2D keypoints into positive Segment Anything point prompts.

    Coordinates remain in the original image coordinate system, as required by
    :class:`segment_anything.SamPredictor`.  Invalid, low-confidence, or off-image
    keypoints are omitted rather than being converted into negative clicks.
    """
    points = np.asarray(keypoints, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2:
        return None, None
    points = points[:, :2]

    confidence = np.ones(points.shape[0], dtype=np.float32)
    if keypoint_conf is not None:
        raw_confidence = np.asarray(keypoint_conf, dtype=np.float32)
        if raw_confidence.ndim == 1:
            confidence.fill(0.0)
            count = min(points.shape[0], raw_confidence.shape[0])
            confidence[:count] = raw_confidence[:count]

    source_h, source_w = int(source_size[0]), int(source_size[1])
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(confidence)
        & (confidence >= confidence_threshold)
        & (points[:, 0] >= 0)
        & (points[:, 0] < source_w)
        & (points[:, 1] >= 0)
        & (points[:, 1] < source_h)
    )
    if not np.any(valid):
        return None, None

    point_coords = np.ascontiguousarray(points[valid], dtype=np.float32)
    point_labels = np.ones(point_coords.shape[0], dtype=np.int32)
    return point_coords, point_labels


def keypoints_to_sam_circle_prompts(
    keypoints: np.ndarray,
    keypoint_conf: Optional[np.ndarray],
    source_size: Sequence[int],
    *,
    confidence_threshold: float = 0.3,
    radius: float = 12.0,
    num_circle_points: int = 8,
):
    """Build one local SAM point prompt for every valid 2D keypoint.

    SAM accepts discrete clicks rather than a circular prompt primitive.  Each
    requested circle is therefore represented by its center plus evenly-spaced
    positive clicks on the circle boundary.  The prompts stay separate: callers
    must invoke ``SamPredictor.predict`` once per circle instead of combining all
    body keypoints into one person prompt.
    """
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('radius must be a positive finite number')
    if num_circle_points < 3:
        raise ValueError('num_circle_points must be at least three')

    centers, _ = keypoints_to_sam_points(
        keypoints,
        keypoint_conf,
        source_size,
        confidence_threshold=confidence_threshold,
    )
    if centers is None:
        return []

    source_h, source_w = int(source_size[0]), int(source_size[1])
    angles = np.linspace(0.0, 2.0 * np.pi, num_circle_points, endpoint=False)
    boundary_offsets = np.stack(
        (radius * np.cos(angles), radius * np.sin(angles)), axis=1
    ).astype(np.float32)
    offsets = np.concatenate((np.zeros((1, 2), dtype=np.float32), boundary_offsets), axis=0)

    prompts = []
    for center in centers:
        point_coords = center[None, :] + offsets
        in_image = (
            (point_coords[:, 0] >= 0)
            & (point_coords[:, 0] < source_w)
            & (point_coords[:, 1] >= 0)
            & (point_coords[:, 1] < source_h)
        )
        # The center is known to be valid, so every prompt has at least one point.
        point_coords = np.ascontiguousarray(point_coords[in_image], dtype=np.float32)
        point_labels = np.ones(point_coords.shape[0], dtype=np.int32)
        prompts.append((point_coords, point_labels))
    return prompts


def select_prompted_sam_mask(masks: np.ndarray, scores: Optional[np.ndarray] = None):
    """Return SAM's highest-scoring mask for one prompted prediction."""
    masks = np.asarray(masks)
    if masks.ndim == 2:
        return binary_object_mask(masks)
    if masks.ndim != 3 or masks.shape[0] == 0:
        return None

    scores = np.asarray(scores).reshape(-1) if scores is not None else np.empty(0)
    if scores.size == masks.shape[0]:
        finite_scores = np.where(np.isfinite(scores), scores, -np.inf)
        mask_index = int(np.argmax(finite_scores)) if np.isfinite(finite_scores).any() else 0
    else:
        mask_index = 0
    return binary_object_mask(masks[mask_index])


def _resize_mask_to_source(mask: np.ndarray, source_size: Sequence[int]) -> np.ndarray:
    """Binarize ``mask`` and make its resolution agree with ``source_size``."""
    source_h, source_w = int(source_size[0]), int(source_size[1])
    mask = binary_object_mask(mask)
    if mask.shape != (source_h, source_w):
        mask = cv2.resize(mask, (source_w, source_h), interpolation=cv2.INTER_NEAREST)
    return binary_object_mask(mask)


def union_sam_masks(masks: Iterable[np.ndarray], source_size: Sequence[int]):
    """Union nonempty SAM masks after aligning them to the source image size."""
    union = None
    for mask in masks:
        if mask is None:
            continue
        aligned = _resize_mask_to_source(mask, source_size)
        if not np.any(aligned):
            continue
        union = aligned if union is None else np.logical_or(union > 0, aligned > 0)
    return None if union is None else binary_object_mask(union)


def remove_human_mask(
    object_mask: Optional[np.ndarray],
    human_mask: Optional[np.ndarray],
    source_size: Sequence[int],
):
    """Remove a keypoint-prompted human mask from a union of SAM object masks.

    Returning ``None`` when either input is unavailable prevents a body-containing
    circle mask from silently reaching the SAM-3D-Objects alpha channel.
    """
    if object_mask is None or human_mask is None:
        return None
    object_mask = _resize_mask_to_source(object_mask, source_size) > 0
    human_mask = _resize_mask_to_source(human_mask, source_size) > 0
    return binary_object_mask(np.logical_and(object_mask, ~human_mask))


def _iter_sam_annotations(annotations: Any) -> Iterable[tuple[Mapping[str, Any], np.ndarray]]:
    """Yield ``(metadata, segmentation)`` pairs from SAM-style proposals."""
    if annotations is None:
        return
    if isinstance(annotations, np.ndarray):
        if annotations.ndim == 2:
            yield {}, annotations
        elif annotations.ndim == 3:
            for segmentation in annotations:
                yield {}, segmentation
        return

    try:
        iterator = iter(annotations)
    except TypeError:
        return

    for annotation in iterator:
        if isinstance(annotation, Mapping):
            segmentation = annotation.get('segmentation')
            if segmentation is not None:
                yield annotation, np.asarray(segmentation)
        else:
            yield {}, np.asarray(annotation)


def _minimum_mask_area(min_area: int, image_area: int) -> int:
    """Keep the default 100-pixel floor while making tiny unit-test images usable."""
    return min(max(1, int(min_area)), max(1, int(np.ceil(image_area * 0.005))))


def _annotation_quality(annotation: Mapping[str, Any]) -> float:
    """Read SAM's optional quality metadata without trusting malformed values."""
    quality = 0.0
    for key in ('predicted_iou', 'stability_score'):
        try:
            value = float(annotation.get(key, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if np.isfinite(value):
            quality += max(0.0, value)
    return quality


def select_sam_object_mask(
    annotations: Any,
    source_size: Sequence[int],
    *,
    min_area: int = 100,
    max_area_fraction: float = 0.98,
    selection: str = 'highest_score',
):
    """Choose a generic automatic-SAM proposal for legacy/task-4 callers.

    This deliberately does *not* use human keypoints.  The active Task-6 path uses
    :func:`keypoints_to_sam_circle_prompts` instead; this function remains only for
    legacy callers that request automatic mask generation without keypoints.
    """
    if selection not in {'highest_score', 'largest', 'union'}:
        raise ValueError(f'Unsupported SAM mask selection: {selection}')
    if not 0 < max_area_fraction <= 1:
        raise ValueError('max_area_fraction must be in (0, 1]')

    source_h, source_w = int(source_size[0]), int(source_size[1])
    image_area = source_h * source_w
    min_area = _minimum_mask_area(min_area, image_area)
    max_area = max_area_fraction * image_area
    candidates = []
    for index, (annotation, segmentation) in enumerate(_iter_sam_annotations(annotations)):
        mask = _resize_mask_to_source(segmentation, (source_h, source_w))
        area = int(mask.sum())
        if min_area <= area <= max_area:
            candidates.append((annotation, mask, area, index))

    if not candidates:
        return None
    if selection == 'union':
        return binary_object_mask(np.logical_or.reduce([candidate[1] > 0 for candidate in candidates]))
    if selection == 'largest':
        selected = max(candidates, key=lambda candidate: (candidate[2], -candidate[3]))
    else:
        selected = max(
            candidates,
            key=lambda candidate: (_annotation_quality(candidate[0]), candidate[2], -candidate[3]),
        )
    return selected[1]
