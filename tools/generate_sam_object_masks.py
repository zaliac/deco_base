#!/usr/bin/env python3
"""Generate SAM foreground masks as RGBA images for DECO/SAM-3D-Objects.

The output PNG keeps the source RGB pixels and writes the selected binary SAM mask
to alpha.  ``BaseDataset`` preserves this channel and provides it to
``SAM3DObjectsEncoder`` as the SAM-3D-Objects ``ALPHA_CHANNEL`` input.

Examples:
    # Select the object containing a positive click at (x=420, y=270).
    python tools/generate_sam_object_masks.py \
        --input /data/images --output-dir /data/images_sam_rgba \
        --checkpoint /weights/sam_vit_h_4b8939.pth --point 420 270

    # Automatic proposal mode; use the largest valid proposal per image.
    python tools/generate_sam_object_masks.py \
        --input /data/images --output-dir /data/images_sam_rgba \
        --checkpoint /weights/sam_vit_h_4b8939.pth --selection largest
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import torch


IMAGE_SUFFIXES = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Write selected Segment Anything masks into PNG alpha channels.'
    )
    parser.add_argument('--input', required=True, type=Path,
                        help='One image or a directory of images.')
    parser.add_argument('--output-dir', required=True, type=Path,
                        help='Root directory for generated RGBA PNGs.')
    parser.add_argument('--checkpoint', required=True, type=Path,
                        help='Local Segment Anything checkpoint (.pth).')
    parser.add_argument('--model-type', default='vit_h',
                        choices=('vit_h', 'vit_l', 'vit_b'),
                        help='SAM checkpoint architecture.')
    parser.add_argument('--device', default=None,
                        help='Torch device (defaults to cuda when available).')
    parser.add_argument('--point', type=float, nargs=2, metavar=('X', 'Y'),
                        help='Positive click selecting an object; uses SamPredictor.')
    parser.add_argument('--selection', default='largest',
                        choices=('largest', 'highest_score', 'union'),
                        help='Automatic-mask selection when --point is omitted.')
    parser.add_argument('--min-area', type=int, default=100,
                        help='Discard automatic proposals smaller than this many pixels.')
    parser.add_argument('--max-area-fraction', type=float, default=0.98,
                        help='Discard proposal masks covering more than this image fraction.')
    parser.add_argument('--manifest', type=Path, default=None,
                        help='Optional CSV mapping source image paths to RGBA outputs.')
    return parser.parse_args()


def iter_images(source: Path) -> tuple[Path, Iterable[Path]]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f'Unsupported image extension: {source}')
        return source.parent, [source]
    if not source.is_dir():
        raise FileNotFoundError(f'Input image/directory not found: {source}')
    return source, sorted(
        path for path in source.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_bgr_and_rgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f'Could not read image: {path}')
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f'Expected an RGB(A) image, got shape {image.shape} at {path}')
    bgr = image[:, :, :3]
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def select_automatic_mask(
    annotations: list[dict], image_shape: tuple[int, int], selection: str,
    min_area: int, max_area_fraction: float,
) -> np.ndarray:
    """Select one deterministic automatic-SAM proposal (or a union of proposals)."""
    image_area = image_shape[0] * image_shape[1]
    candidates = [
        annotation for annotation in annotations
        if annotation.get('area', 0) >= min_area
        and annotation.get('area', 0) <= max_area_fraction * image_area
    ]
    if not candidates:
        raise RuntimeError('SAM produced no proposal after area filtering')

    if selection == 'union':
        return np.logical_or.reduce([
            np.asarray(annotation['segmentation'], dtype=bool) for annotation in candidates
        ])
    if selection == 'highest_score':
        selected = max(
            candidates,
            key=lambda annotation: (
                annotation.get('predicted_iou', 0.0),
                annotation.get('stability_score', 0.0),
                annotation.get('area', 0),
            ),
        )
    else:  # largest
        selected = max(candidates, key=lambda annotation: annotation.get('area', 0))
    return np.asarray(selected['segmentation'], dtype=bool)


def make_mask(rgb: np.ndarray, generator, predictor, point: Optional[list[float]], args) -> np.ndarray:
    if point is not None:
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(
            point_coords=np.asarray([point], dtype=np.float32),
            point_labels=np.asarray([1], dtype=np.int32),
            multimask_output=True,
        )
        return np.asarray(masks[int(np.argmax(scores))], dtype=bool)

    return select_automatic_mask(
        generator.generate(rgb), rgb.shape[:2], args.selection,
        args.min_area, args.max_area_fraction,
    )


def save_rgba(bgr: np.ndarray, mask: np.ndarray, output_path: Path):
    if mask.shape != bgr.shape[:2]:
        raise ValueError(f'Mask shape {mask.shape} does not match image shape {bgr.shape[:2]}')
    alpha = np.where(mask, 255, 0).astype(np.uint8)
    bgra = np.dstack((bgr, alpha))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), bgra):
        raise OSError(f'Could not write RGBA image: {output_path}')


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'SAM checkpoint not found: {args.checkpoint}')
    if not 0 < args.max_area_fraction <= 1:
        raise ValueError('--max-area-fraction must be in (0, 1]')

    try:
        from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
    except ImportError as exc:
        raise ImportError(
            'This script requires Meta Segment Anything. Install it in the training '
            'environment (for example: pip install segment-anything).'
        ) from exc

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
    sam.to(device=device)
    sam.eval()
    generator = None if args.point is not None else SamAutomaticMaskGenerator(sam)
    predictor = SamPredictor(sam) if args.point is not None else None

    source_root, image_paths = iter_images(args.input)
    image_paths = list(image_paths)
    if not image_paths:
        raise RuntimeError(f'No supported images found under {args.input}')

    rows = []
    for image_path in image_paths:
        bgr, rgb = load_bgr_and_rgb(image_path)
        mask = make_mask(rgb, generator, predictor, args.point, args)
        output_path = args.output_dir / image_path.relative_to(source_root).with_suffix('.png')
        save_rgba(bgr, mask, output_path)
        rows.append((str(image_path), str(output_path)))
        print(f'[{len(rows)}/{len(image_paths)}] {image_path} -> {output_path}')

    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(('source_image', 'rgba_image'))
            writer.writerows(rows)

'''
  python tools/generate_sam_object_masks.py \
    --input <images> --output-dir <rgba_images> \
    --checkpoint data/weights/sam/sam_vit_h_4b8939.pth
'''
if __name__ == '__main__':
    main()
