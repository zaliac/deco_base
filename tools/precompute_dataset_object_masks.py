#!/usr/bin/env python3
"""Precompute task-6 keypoint-circle masks for ``BaseDataset`` image entries.

The script writes one grayscale PNG per source image and creates a *new* annotation
NPZ containing an ``object_mask_path`` array aligned with ``imgname``.  During
``sam_sam`` training, BaseDataset then reads the PNG directly and never loads SAM.

For annotations with 2D body keypoints, it creates one small circular positive-point
prompt per keypoint, obtains one SAM mask per prompt, and unions those masks. A
separate all-keypoint human mask is subtracted from that union before it reaches the
object encoder. This matches the runtime Task-6 Option-1 path.

Example:
    python tools/precompute_dataset_object_masks.py \
        --annotations datasets/Release_Datasets/damon/hot_dca_trainval_with_kpts.npz \
        --dataset-root /path/to/deco/data \
        --output-dir /path/to/deco/data/sam_masks/damon_train \
        --output-annotations datasets/Release_Datasets/damon/hot_dca_trainval_sam_masks.npz \
        --checkpoint data/weights/sam/sam_vit_b_01ec64.pth
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch


# The repository is script-style rather than an installed package.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.object_masks import (
    binary_object_mask,
    keypoints_to_sam_circle_prompts,
    keypoints_to_sam_points,
    remove_human_mask,
    select_prompted_sam_mask,
    select_sam_object_mask,
    union_sam_masks,
)


def _npz_path(value: Any) -> Path:
    """Convert an NPZ string/bytes scalar into a Path."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode()
    return Path(str(value))


def _image_path(value: Any, dataset_root: Path) -> Path:
    path = _npz_path(value)
    return path if path.is_absolute() else dataset_root / path


def _output_mask_path(image_path: Path, dataset_root: Path, output_dir: Path) -> Path:
    """Preserve the input's relative layout, avoiding collisions for external paths."""
    try:
        relative_path = image_path.relative_to(dataset_root)
    except ValueError:
        digest = hashlib.sha1(str(image_path).encode()).hexdigest()[:12]
        relative_path = Path('_external') / f'{digest}_{image_path.name}'
    return (output_dir / relative_path).with_suffix('.png')


def _stored_mask_path(mask_path: Path, dataset_root: Path) -> str:
    """Keep NPZ paths portable whenever the output directory is under dataset_root."""
    try:
        return mask_path.relative_to(dataset_root).as_posix()
    except ValueError:
        return str(mask_path)


def _load_image(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f'Could not read dataset image: {path}')
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = np.where(binary_object_mask(mask) > 0, 255, 0).astype(np.uint8)
    if not cv2.imwrite(str(path), encoded):
        raise OSError(f'Could not save object mask: {path}')


def _existing_mask_is_valid(path: Path, image_shape: Tuple[int, int]) -> bool:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return mask is not None and mask.shape[:2] == image_shape


def _sam_annotations(mask_generator, rgb_image: np.ndarray):
    with torch.no_grad():
        return (
            mask_generator.generate(rgb_image)
            if hasattr(mask_generator, 'generate') else mask_generator(rgb_image)
        )


def _keypoint_circle_object_mask(
    rgb_image: np.ndarray,
    image_shape: Tuple[int, int],
    keypoints,
    keypoint_conf,
    predictor,
    *,
    prompt_confidence: float,
    circle_radius: float,
    circle_points: int,
):
    """Return human-removed, individual Task-6 Option-1 circle-prompt masks.

    The second return value differentiates an unavailable keypoint prompt from a
    valid prompt set whose SAM outputs are all empty.  The latter becomes an empty
    alpha mask rather than falling back to one generic automatic proposal.
    """
    if predictor is None or keypoints is None:
        return None, False
    circle_prompts = keypoints_to_sam_circle_prompts(
        keypoints,
        keypoint_conf,
        image_shape,
        confidence_threshold=prompt_confidence,
        radius=circle_radius,
        num_circle_points=circle_points,
    )
    if not circle_prompts:
        return None, False
    human_point_coords, human_point_labels = keypoints_to_sam_points(
        keypoints,
        keypoint_conf,
        image_shape,
        confidence_threshold=prompt_confidence,
    )
    if human_point_coords is None:
        return None, False

    selected_masks = []
    human_mask = None
    try:
        with torch.no_grad():
            predictor.set_image(rgb_image)
            masks, scores, _ = predictor.predict(
                point_coords=human_point_coords,
                point_labels=human_point_labels,
                multimask_output=True,
            )
            human_mask = select_prompted_sam_mask(masks, scores)
            if human_mask is not None:
                for point_coords, point_labels in circle_prompts:
                    masks, scores, _ = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True,
                    )
                    selected_mask = select_prompted_sam_mask(masks, scores)
                    if selected_mask is not None:
                        selected_masks.append(selected_mask)
    finally:
        reset_image = getattr(predictor, 'reset_image', None)
        if callable(reset_image):
            reset_image()
    return remove_human_mask(
        union_sam_masks(selected_masks, image_shape),
        human_mask,
        image_shape,
    ), True


def precompute_masks(
    annotations_path: Path,
    dataset_root: Path,
    output_dir: Path,
    output_annotations_path: Path,
    mask_generator=None,
    *,
    sam_predictor=None,
    prompt_confidence: float = 0.3,
    keypoint_circle_radius: float = 12.0,
    keypoint_circle_points: int = 8,
    selection: str = 'highest_score',
    min_area: int = 100,
    max_area_fraction: float = 0.80,
    overwrite: bool = False,
) -> int:
    """Generate mask files and return the number of image-list entries written.

    When keypoints are available, ``sam_predictor`` generates the independent
    Task-6 Option-1 masks.  ``mask_generator`` is optional and only supports the
    legacy no-keypoint fallback.  Both are injectable to keep the function
    testable without a large checkpoint.
    """
    annotations_path = Path(annotations_path)
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_annotations_path = Path(output_annotations_path).resolve()

    if not annotations_path.is_file():
        raise FileNotFoundError(f'Annotation NPZ not found: {annotations_path}')
    if annotations_path.resolve() == output_annotations_path:
        raise ValueError('output_annotations_path must be a new NPZ; preserve the source annotation')
    if output_annotations_path.exists() and not overwrite:
        raise FileExistsError(
            f'Output annotation already exists: {output_annotations_path}. Use --overwrite to replace it.'
        )
    if not 0 <= prompt_confidence <= 1:
        raise ValueError('prompt_confidence must be in [0, 1]')
    if not np.isfinite(keypoint_circle_radius) or keypoint_circle_radius <= 0:
        raise ValueError('keypoint_circle_radius must be a positive finite number')
    if keypoint_circle_points < 3:
        raise ValueError('keypoint_circle_points must be at least three')

    with np.load(annotations_path, allow_pickle=True) as source:
        data: Dict[str, np.ndarray] = {key: source[key] for key in source.files}
    if 'imgname' not in data:
        raise KeyError(f'{annotations_path} has no imgname array required by BaseDataset')

    image_names = data['imgname']
    keypoints_2d = data.get('keypoint_2d')
    keypoint_conf = data.get('keypoint_conf')
    mask_paths = []
    processed_images = {}
    for index, image_name in enumerate(image_names):
        image_path = _image_path(image_name, dataset_root)
        cache_key = str(image_path.resolve())
        if cache_key in processed_images:
            mask_paths.append(processed_images[cache_key])
            continue

        bgr, rgb = _load_image(image_path)
        mask_path = _output_mask_path(image_path, dataset_root, output_dir)
        if mask_path.exists() and not overwrite:
            if not _existing_mask_is_valid(mask_path, bgr.shape[:2]):
                raise FileExistsError(
                    f'Existing mask is invalid or has the wrong size: {mask_path}. '
                    'Use --overwrite to regenerate it.'
                )
        else:
            keypoints = keypoints_2d[index] if keypoints_2d is not None else None
            confidence = keypoint_conf[index] if keypoint_conf is not None else None
            mask, used_keypoint_prompt = _keypoint_circle_object_mask(
                rgb,
                bgr.shape[:2],
                keypoints,
                confidence,
                sam_predictor,
                prompt_confidence=prompt_confidence,
                circle_radius=keypoint_circle_radius,
                circle_points=keypoint_circle_points,
            )
            if used_keypoint_prompt and mask is None:
                # Valid circle prompts with no usable prediction remain explicitly
                # empty instead of being replaced by an unrelated scene proposal.
                mask = np.zeros(bgr.shape[:2], dtype=np.float32)
            elif not used_keypoint_prompt:
                if mask_generator is None:
                    raise RuntimeError(
                        'No usable keypoint-circle prompt and no automatic-SAM fallback '
                        f'for index {index}: {image_path}'
                    )
                annotations = _sam_annotations(mask_generator, rgb)
                mask = select_sam_object_mask(
                    annotations,
                    bgr.shape[:2],
                    min_area=min_area,
                    max_area_fraction=max_area_fraction,
                    selection=selection,
                )
            if mask is None:
                raise RuntimeError(
                    f'SAM produced no usable object proposal for index {index}: {image_path}'
                )
            _write_mask(mask_path, mask)

        stored_path = _stored_mask_path(mask_path, dataset_root)
        processed_images[cache_key] = stored_path
        mask_paths.append(stored_path)
        print(f'[{index + 1}/{len(image_names)}] {image_path} -> {mask_path}')

    data['object_mask_path'] = np.asarray(mask_paths, dtype=str)
    output_annotations_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_annotations_path, **data)
    return len(mask_paths)


def _build_sam_tools(checkpoint: Path, model_type: str, device: str):
    try:
        from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
    except ImportError as exc:
        raise ImportError(
            'Offline mask generation requires segment-anything. Install the project requirements.'
        ) from exc

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    sam.eval()
    return SamAutomaticMaskGenerator(sam), SamPredictor(sam)


def _build_mask_generator(checkpoint: Path, model_type: str, device: str):
    """Backward-compatible helper for scripts importing the task-4 API."""
    return _build_sam_tools(checkpoint, model_type, device)[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate object masks for exactly the image list in a BaseDataset annotation NPZ.'
    )
    parser.add_argument('--annotations', required=True, type=Path,
                        help='Source NPZ containing BaseDataset imgname entries.')
    parser.add_argument('--dataset-root', required=True, type=Path,
                        help='Path BaseDataset prepends to relative imgname paths.')
    parser.add_argument('--output-dir', required=True, type=Path,
                        help='Directory that will contain the grayscale mask PNGs.')
    parser.add_argument('--output-annotations', required=True, type=Path,
                        help='New NPZ to write with the aligned object_mask_path field.')
    parser.add_argument('--checkpoint', type=Path,
                        default=REPOSITORY_ROOT / 'data/weights/sam/sam_vit_b_01ec64.pth',
                        help='Segment Anything checkpoint.')
    parser.add_argument('--model-type', default='vit_b', choices=('vit_h', 'vit_l', 'vit_b'),
                        help='SAM architecture matching --checkpoint.')
    parser.add_argument('--device', default=None,
                        help='Torch device; defaults to cuda when available.')
    parser.add_argument('--selection', default='highest_score',
                        choices=('highest_score', 'largest', 'union'),
                        help='Fallback selection when annotations have no usable keypoints.')
    parser.add_argument('--prompt-confidence', default=0.3, type=float,
                        help='Minimum keypoint confidence used for a SAM circle prompt.')
    parser.add_argument('--keypoint-circle-radius', default=12.0, type=float,
                        help='Circle radius in original-image pixels for every keypoint prompt.')
    parser.add_argument('--keypoint-circle-points', default=8, type=int,
                        help='Number of positive boundary samples used to represent each circle.')
    parser.add_argument('--min-area', default=100, type=int,
                        help='Ignore SAM proposals smaller than this pixel area.')
    parser.add_argument('--max-area-fraction', default=0.80, type=float,
                        help='Ignore proposals covering more than this image fraction.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace existing mask PNGs and output annotation NPZ.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'SAM checkpoint not found: {args.checkpoint}')
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    generator, predictor = _build_sam_tools(args.checkpoint, args.model_type, device)
    count = precompute_masks(
        args.annotations,
        args.dataset_root,
        args.output_dir,
        args.output_annotations,
        generator,
        sam_predictor=predictor,
        prompt_confidence=args.prompt_confidence,
        keypoint_circle_radius=args.keypoint_circle_radius,
        keypoint_circle_points=args.keypoint_circle_points,
        selection=args.selection,
        min_area=args.min_area,
        max_area_fraction=args.max_area_fraction,
        overwrite=args.overwrite,
    )
    print(f'Wrote {count} object-mask paths to {args.output_annotations}')
    print('Point common.constants.DATASET_FILES at this new NPZ before sam_sam training.')


if __name__ == '__main__':
    main()
