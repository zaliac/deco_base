import importlib.util
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

# This project is a script-style repository rather than an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.sam3d_encoder import SAM3DObjectsEncoder
from utils.object_masks import alpha_object_mask, crop_and_resize_object_mask
from common import constants
from data.base_dataset import BaseDataset


class _DummyDino(nn.Module):
    """Small forward_features-compatible stand-in; no model checkpoint required."""

    def __init__(self, dim=4, patch_size=2):
        super().__init__()
        self.embed_dim = dim
        self.patch_embed = type('PatchEmbed', (), {'patch_size': patch_size})()
        self.scale = nn.Parameter(torch.ones(1))

    def forward_features(self, image):
        self.last_input = image
        tokens = torch.nn.functional.avg_pool2d(image[:, :1], 2)
        tokens = tokens.flatten(2).transpose(1, 2)
        return {'x_norm_patchtokens': tokens.repeat(1, 1, self.embed_dim) * self.scale}


def _encoder_for_test():
    """Build the forward-path state without loading the large SAM checkpoint."""
    encoder = object.__new__(SAM3DObjectsEncoder)
    nn.Module.__init__(encoder)
    encoder.input_is_normalized = True
    encoder.input_size = encoder.mask_input_size = 8
    encoder.normalize_images = encoder.mask_normalize_images = True
    encoder.freeze_backbone = True
    encoder.backbone = _DummyDino()
    encoder.mask_backbone = _DummyDino()
    encoder.backbone_dtype = encoder.mask_backbone_dtype = torch.float32
    encoder.patch_size = encoder.mask_patch_size = 2
    encoder.embed_dim = encoder.mask_embed_dim = 4
    encoder.image_mask_fusion = nn.Conv2d(8, 4, kernel_size=1)
    encoder.projection = nn.Conv2d(4, 6, kernel_size=1)
    encoder.output_size = (3, 3)
    encoder.register_buffer(
        'image_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    )
    encoder.register_buffer(
        'image_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    )
    return encoder


def test_rgba_mask_is_binary_and_tracks_deco_crop():
    rgba = np.zeros((8, 10, 4), dtype=np.uint8)
    rgba[2:6, 3:8, 3] = 255

    mask = alpha_object_mask(rgba)
    resized = crop_and_resize_object_mask(mask, (2, 1, 9, 7), (8, 10), (12, 14))

    assert resized.shape == (12, 14)
    assert set(np.unique(resized)).issubset({0.0, 1.0})
    assert resized.sum() > 0


def test_object_encoder_consumes_alpha_mask_and_preserves_output_contract():
    encoder = _encoder_for_test()
    image = torch.randn(2, 3, 6, 10)
    object_mask = torch.zeros(2, 1, 6, 10)
    object_mask[:, :, 2:5, 3:8] = 1

    output = encoder(image, object_prompt=object_mask)
    legacy_output = encoder(image, (2, 2))

    assert output.shape == (2, 6, 3, 3)
    assert legacy_output.shape == (2, 6, 2, 2)
    assert encoder.mask_backbone.last_input.shape == (2, 3, 8, 8)


class _FakeAutomaticMaskGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, rgb_image):
        self.calls += 1
        mask = np.zeros(rgb_image.shape[:2], dtype=bool)
        mask[4:20, 8:24] = True
        return [{
            'segmentation': mask,
            'area': int(mask.sum()),
            'predicted_iou': 0.99,
            'stability_score': 0.99,
        }]


class _FakePointPromptPredictor:
    """Small SamPredictor-compatible fake that records the supplied prompts."""

    def __init__(self):
        self.set_image_calls = 0
        self.predict_calls = 0
        self.point_coords = None
        self.point_labels = None
        self.multimask_output = None
        self.image_shape = None

    def set_image(self, rgb_image):
        self.set_image_calls += 1
        self.image_shape = rgb_image.shape[:2]

    def predict(self, point_coords, point_labels, multimask_output):
        self.predict_calls += 1
        self.point_coords = point_coords.copy()
        self.point_labels = point_labels.copy()
        self.multimask_output = multimask_output
        masks = np.zeros((3, *self.image_shape), dtype=bool)
        masks[0, 1:4, 1:4] = True
        masks[1, 4:20, 8:24] = True
        masks[2, 2:18, 4:20] = True
        return masks, np.asarray([0.2, 0.95, 0.6]), None


class _FakeContactProposalGenerator:
    """Automatic-SAM proposals: person, nearby object, and distant distractor."""

    def __init__(self):
        self.calls = 0

    def generate(self, rgb_image):
        self.calls += 1
        human = np.zeros(rgb_image.shape[:2], dtype=bool)
        human[4:20, 8:24] = True
        nearby_object = np.zeros_like(human)
        nearby_object[4:20, 24:31] = True
        distant_object = np.zeros_like(human)
        distant_object[48:56, 0:8] = True
        return [
            {'segmentation': human, 'predicted_iou': 0.99, 'stability_score': 0.99},
            {'segmentation': nearby_object, 'predicted_iou': 0.90, 'stability_score': 0.90},
            {'segmentation': distant_object, 'predicted_iou': 0.99, 'stability_score': 0.99},
        ]


def test_dataset_generates_and_caches_sam_mask_when_image_has_no_alpha():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        image = np.full((24, 32, 3), 127, dtype=np.uint8)
        assert cv2.imwrite(str(root / 'sample.jpg'), image)
        annotation_path = root / 'samples.npz'
        np.savez(
            annotation_path,
            imgname=np.array(['sample.jpg']),
            crop_bbox=np.array([[2, 1, 30, 22]]),
        )
        constants.DATASET_FILES.setdefault('train', {})['sam_cache_test'] = str(annotation_path)

        generator = _FakeAutomaticMaskGenerator()
        dataset = BaseDataset(
            'sam_cache_test', 'train', dataset_root_path=str(root),
            normalize=False, sam_mask_generator=generator,
        )
        first = dataset[0]
        second = dataset[0]

        assert generator.calls == 1
        assert dataset.object_masks[0].shape == (24, 32)
        assert dataset.object_masks[0].dtype == np.uint8
        assert first['has_object_mask'].item() == 1.0
        assert first['object_mask'].shape == (1, 256, 256)
        assert torch.equal(first['object_mask'], second['object_mask'])


def test_dataset_uses_original_2d_keypoints_as_sam_point_prompts(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        image = np.full((64, 80, 3), 127, dtype=np.uint8)
        assert cv2.imwrite(str(root / 'sample.jpg'), image)

        keypoints = np.zeros((17, 2), dtype=np.float32)
        keypoints[0] = (10, 5)      # valid
        keypoints[1] = (12, 8)      # valid
        keypoints[2] = (100, 8)     # outside image
        keypoints[3] = (8, 18)      # below confidence threshold
        confidence = np.zeros(17, dtype=np.float32)
        confidence[:4] = (0.9, 0.4, 0.9, 0.2)

        annotation_path = root / 'samples_with_keypoints.npz'
        np.savez(
            annotation_path,
            imgname=np.array(['sample.jpg']),
            crop_bbox=np.array([[2, 1, 78, 62]]),
            keypoint_2d=keypoints[None],
            keypoint_conf=confidence[None],
        )
        monkeypatch.setitem(
            constants.DATASET_FILES.setdefault('train', {}),
            'sam_point_prompt_test', str(annotation_path),
        )

        predictor = _FakePointPromptPredictor()
        generator = _FakeContactProposalGenerator()
        dataset = BaseDataset(
            'sam_point_prompt_test', 'train', dataset_root_path=str(root),
            normalize=False, sam_predictor=predictor, sam_mask_generator=generator,
        )
        first = dataset[0]
        second = dataset[0]

        # SamPredictor sees original-image coordinates, before DECO's person crop.
        np.testing.assert_array_equal(
            predictor.point_coords, np.asarray([[10, 5], [12, 8]], dtype=np.float32)
        )
        np.testing.assert_array_equal(predictor.point_labels, np.asarray([1, 1], dtype=np.int32))
        assert predictor.multimask_output is True
        assert predictor.set_image_calls == predictor.predict_calls == 1
        assert generator.calls == 1
        assert dataset.object_masks[0].shape == (64, 80)
        assert first['has_object_mask'].item() == 1.0
        assert torch.equal(first['object_prompt'], first['object_mask'])
        assert first['object_mask'].sum().item() > 0
        # The high-scoring human proposal is excluded; only the adjacent object is
        # retained in original image coordinates before DECO's crop/resize.
        assert dataset.object_masks[0][4:20, 8:24].sum() == 0
        assert dataset.object_masks[0][4:20, 24:31].sum() > 0
        assert dataset.object_masks[0][48:56, 0:8].sum() == 0
        assert torch.equal(first['object_mask'], second['object_mask'])


def test_offline_precompute_uses_the_same_non_person_contact_selection():
    script_path = Path(__file__).resolve().parents[1] / 'tools' / 'precompute_dataset_object_masks.py'
    spec = importlib.util.spec_from_file_location('precompute_dataset_object_masks', script_path)
    precompute = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(precompute)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        image = np.full((64, 80, 3), 127, dtype=np.uint8)
        assert cv2.imwrite(str(root / 'sample.jpg'), image)
        keypoints = np.zeros((1, 17, 2), dtype=np.float32)
        keypoints[0, :2] = ((10, 5), (12, 8))
        confidence = np.zeros((1, 17), dtype=np.float32)
        confidence[0, :2] = (0.9, 0.4)
        annotations_path = root / 'samples.npz'
        np.savez(
            annotations_path,
            imgname=np.array(['sample.jpg']),
            keypoint_2d=keypoints,
            keypoint_conf=confidence,
        )

        output_dir = root / 'masks'
        output_annotations = root / 'samples_with_masks.npz'
        count = precompute.precompute_masks(
            annotations_path,
            root,
            output_dir,
            output_annotations,
            _FakeContactProposalGenerator(),
            sam_predictor=_FakePointPromptPredictor(),
        )

        assert count == 1
        with np.load(output_annotations, allow_pickle=True) as result:
            mask_path = root / str(result['object_mask_path'][0])
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        assert mask is not None
        assert mask[4:20, 8:24].sum() == 0
        assert mask[4:20, 24:31].sum() > 0
        assert mask[48:56, 0:8].sum() == 0
