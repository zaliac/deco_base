import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from common import constants
import os
from pathlib import Path
from utils.object_masks import (
    alpha_object_mask,
    binary_object_mask,
    crop_and_resize_object_mask,
    keypoints_to_sam_circle_prompts,
    keypoints_to_sam_points,
    load_object_mask,
    remove_human_mask,
    select_prompted_sam_mask,
    select_sam_object_mask,
    union_sam_masks,
)

def mask_split(img, num_parts):
    if not len(img.shape) == 2:
        img = img[:, :, 0]
    mask = np.zeros((img.shape[0], img.shape[1], num_parts))
    for i in np.unique(img):
        mask[:, :, i] = np.where(img == i, 1., 0.)
    return np.transpose(mask, (2, 0, 1))

class BaseDataset(Dataset):

    def __init__(
            self,
            dataset,
            mode,
            model_type='smpl',
            dataset_root_path='',
            normalize=False,
            generate_object_masks=False,
            sam_checkpoint_path=None,
            sam_model_type='vit_b',
            sam_device=None,
            sam_mask_generator=None,
            sam_predictor=None,
            sam_prompt_confidence=0.3,
            sam_multimask_output=True,
            sam_keypoint_circle_radius=12.0,
            sam_keypoint_circle_points=8,
    ):
        self.dataset = dataset
        self.mode = mode
        self.dataset_base_path = dataset_root_path

        print(f'Loading dataset: {constants.DATASET_FILES[mode][dataset]} for mode: {mode}')

        self.data = np.load(constants.DATASET_FILES[mode][dataset], allow_pickle=True)

        self.images = self.data['imgname']

        # get 3d contact labels, if available
        try:
            self.contact_labels_3d = self.data['contact_label']
            # make a has_contact_3d numpy array which contains 1 if contact labels are no empty and 0 otherwise
            self.has_contact_3d = np.array([1 if len(x) > 0 else 0 for x in self.contact_labels_3d])
        except KeyError:
            self.has_contact_3d = np.zeros(len(self.images))

        # get 2d polygon contact labels, if available
        try:
            self.polygon_contacts_2d = self.data['polygon_2d_contact']
            self.has_polygon_contact_2d = np.ones(len(self.images))
        except KeyError:
            self.has_polygon_contact_2d = np.zeros(len(self.images))

        # Get camera parameters - only intrinsics for now
        try:
            self.cam_k = self.data['cam_k']
        except KeyError:
            self.cam_k = np.zeros((len(self.images), 3, 3))

        # Get 2D keypoint prompts (COCO-17, original-image pixels) + confidence, if available
        try:
            self.keypoints_2d = self.data['keypoint_2d']
            self.keypoint_conf = self.data['keypoint_conf']
            self.has_keypoints = np.ones(len(self.images))
        except KeyError:
            self.keypoints_2d = None
            self.keypoint_conf = None
            self.has_keypoints = np.zeros(len(self.images))

        # Scene/part segmentation masks (HOT-style). Optional: datasets like BEHAVE
        # have no such masks, so we return zeros instead of crashing. Seg losses/IoU
        # are then not meaningful, but contact metrics are unaffected.
        self.sem_masks = self.data['scene_seg'] if 'scene_seg' in self.data else None
        self.part_masks = self.data['part_seg'] if 'part_seg' in self.data else None

        # Keep a per-sample in-memory cache of *resolved* original-resolution masks.
        # Every alpha/NPZ/SAM mask is inserted here when first used, so repeat epochs
        # do not run SAM again for the same image in this Dataset instance.
        self.object_masks = [None] * len(self.images)
        self.object_mask_sources = None
        self.object_mask_key = None
        for key in ('object_mask', 'object_mask_path', 'sam_mask', 'sam_mask_path'):
            if key in self.data:
                self.object_mask_sources = self.data[key]
                self.object_mask_key = key
                break

        self.generate_object_masks = bool(
            generate_object_masks
            or sam_predictor is not None
            or sam_mask_generator is not None
        )
        self.sam_model_type = sam_model_type
        self.sam_device = sam_device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam_checkpoint_path = Path(
            sam_checkpoint_path
            or Path(__file__).resolve().parents[1]
            / 'data/weights/sam/sam_vit_b_01ec64.pth'
        )
        # Task 6 option 1 runs SAM independently for every keypoint circle, then
        # subtracts a separate all-keypoint human mask. An explicitly supplied
        # automatic generator remains a no-keypoint fallback for task-4 callers.
        self._sam_predictor = sam_predictor
        self._sam_mask_generator = sam_mask_generator
        self._has_external_sam_mask_generator = sam_mask_generator is not None
        self.sam_prompt_confidence = float(sam_prompt_confidence)
        if not 0.0 <= self.sam_prompt_confidence <= 1.0:
            raise ValueError('sam_prompt_confidence must be in [0, 1]')
        self.sam_multimask_output = bool(sam_multimask_output)
        self.sam_keypoint_circle_radius = float(sam_keypoint_circle_radius)
        self.sam_keypoint_circle_points = int(sam_keypoint_circle_points)
        if not np.isfinite(self.sam_keypoint_circle_radius) or self.sam_keypoint_circle_radius <= 0:
            raise ValueError('sam_keypoint_circle_radius must be a positive finite number')
        if self.sam_keypoint_circle_points < 3:
            raise ValueError('sam_keypoint_circle_points must be at least three')

        # Optional per-image person-crop bbox [x0,y0,x1,y1] in original-image pixels.
        # Used by datasets whose images are not person-centric (e.g. BEHAVE full
        # Kinect frames); absent for DAMON/RICH/PROX -> no cropping.
        self.crop_bbox = self.data['crop_bbox'] if 'crop_bbox' in self.data else None

        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].astype(float)
            self.betas = self.data['shape'].astype(float)
            self.transl = self.data['transl'].astype(float)
            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']
            else:
                self.has_smpl = np.ones(len(self.images))
            self.is_smplx = np.ones(len(self.images)) if model_type == 'smplx' else np.zeros(len(self.images))
        except KeyError:
            self.has_smpl = np.zeros(len(self.images))
            self.is_smplx = np.zeros(len(self.images))

        if model_type == 'smpl':
            self.n_vertices = 6890
        elif model_type == 'smplx':
            self.n_vertices = 10475
        else:
            raise NotImplementedError

        self.normalize = normalize
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

    def _get_sam_predictor(self):
        """Lazily obtain a SAM predictor for the per-keypoint circle prompts."""
        if self._sam_predictor is not None:
            return self._sam_predictor
        if not self.generate_object_masks:
            return None

        # A supplied SamAutomaticMaskGenerator already owns a SAM model. Reuse it
        # rather than loading a second model just for the circle prompts.
        generator_predictor = getattr(self._sam_mask_generator, 'predictor', None)
        shared_model = getattr(generator_predictor, 'model', None)
        if shared_model is not None:
            try:
                from segment_anything import SamPredictor
            except ImportError as exc:
                raise ImportError(
                    'Task-6 keypoint-circle masks require segment-anything. Install the '
                    'project requirements before training.'
                ) from exc
            self._sam_predictor = SamPredictor(shared_model)
            return self._sam_predictor

        # Generic task-4 generators may not expose their model. Preserve that
        # caller's fallback without implicitly allocating a second SAM model.
        if self._has_external_sam_mask_generator:
            return None
        if not self.sam_checkpoint_path.is_file():
            raise FileNotFoundError(
                'SAM checkpoint required for task-6 keypoint-circle masks was not found: '
                f'{self.sam_checkpoint_path}'
            )
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                'Task-6 keypoint-circle masks require segment-anything. Install the '
                'project requirements before training.'
            ) from exc

        print(
            f'Loading SAM keypoint-circle predictor ({self.sam_model_type}) on '
            f'{self.sam_device}'
        )
        sam = sam_model_registry[self.sam_model_type](checkpoint=str(self.sam_checkpoint_path))
        sam.to(device=self.sam_device)
        sam.eval()
        self._sam_predictor = SamPredictor(sam)
        return self._sam_predictor

    @staticmethod
    def _select_sam_object_mask(annotations, source_size):
        """Choose a generic automatic-SAM proposal for legacy task-4 callers."""
        return select_sam_object_mask(annotations, source_size)

    def _sam_circle_prompts(self, index, source_size):
        """Return one independent circular positive-point prompt per valid keypoint."""
        if self.keypoints_2d is None or not self.has_keypoints[index]:
            return []
        confidence = self.keypoint_conf[index] if self.keypoint_conf is not None else None
        return keypoints_to_sam_circle_prompts(
            self.keypoints_2d[index],
            confidence,
            source_size,
            confidence_threshold=self.sam_prompt_confidence,
            radius=self.sam_keypoint_circle_radius,
            num_circle_points=self.sam_keypoint_circle_points,
        )

    def _sam_human_prompt(self, index, source_size):
        """Return one all-keypoint prompt used only to remove the human body."""
        if self.keypoints_2d is None or not self.has_keypoints[index]:
            return None, None
        confidence = self.keypoint_conf[index] if self.keypoint_conf is not None else None
        return keypoints_to_sam_points(
            self.keypoints_2d[index],
            confidence,
            source_size,
            confidence_threshold=self.sam_prompt_confidence,
        )

    @staticmethod
    def _select_prompted_sam_mask(masks, scores):
        """Choose the highest-confidence mask returned by SamPredictor.predict."""
        return select_prompted_sam_mask(masks, scores)

    def _generate_object_mask(self, index, image_with_alpha, source_size):
        """Union per-circle SAM masks after removing the keypoint-prompted person."""
        circle_prompts = self._sam_circle_prompts(index, source_size)
        human_point_coords, human_point_labels = self._sam_human_prompt(index, source_size)
        if circle_prompts and human_point_coords is not None:
            predictor = self._get_sam_predictor()
            if predictor is not None:
                rgb_image = cv2.cvtColor(image_with_alpha[:, :, :3], cv2.COLOR_BGR2RGB)
                selected_masks = []
                human_mask = None
                try:
                    with torch.no_grad():
                        # Encode once, decode a human mask from all keypoint centers,
                        # then decode one candidate mask per independent circle.
                        predictor.set_image(rgb_image)
                        masks, scores, _ = predictor.predict(
                            point_coords=human_point_coords,
                            point_labels=human_point_labels,
                            multimask_output=self.sam_multimask_output,
                        )
                        human_mask = self._select_prompted_sam_mask(masks, scores)
                        if human_mask is not None:
                            for point_coords, point_labels in circle_prompts:
                                masks, scores, _ = predictor.predict(
                                    point_coords=point_coords,
                                    point_labels=point_labels,
                                    multimask_output=self.sam_multimask_output,
                                )
                                selected_mask = self._select_prompted_sam_mask(masks, scores)
                                if selected_mask is not None:
                                    selected_masks.append(selected_mask)
                finally:
                    reset_image = getattr(predictor, 'reset_image', None)
                    if callable(reset_image):
                        reset_image()
                return remove_human_mask(
                    union_sam_masks(selected_masks, source_size),
                    human_mask,
                    source_size,
                )

        # Retain explicit task-4 generator support for samples with no usable
        # keypoints.  Normal sam_sam training takes the circle-prompt branch above.
        if not circle_prompts and self._has_external_sam_mask_generator:
            rgb_image = cv2.cvtColor(image_with_alpha[:, :, :3], cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                annotations = (
                    self._sam_mask_generator.generate(rgb_image)
                    if hasattr(self._sam_mask_generator, 'generate')
                    else self._sam_mask_generator(rgb_image)
                )
            return self._select_sam_object_mask(annotations, source_size)
        return None

    def _object_mask_for_image(self, index, image_with_alpha, source_size, crop_bbox):
        """Resolve/cache a mask, then mirror the RGB crop/resize exactly."""
        mask = self.object_masks[index]
        has_object_mask = mask is not None and bool(np.any(mask))

        if mask is None and self.object_mask_sources is not None:
            try:
                mask = load_object_mask(
                    self.object_mask_sources[index], dataset_root=self.dataset_base_path
                )
                has_object_mask = bool(np.any(mask))
            except (FileNotFoundError, ValueError, TypeError, cv2.error) as exc:
                print(f'Object mask ({self.object_mask_key}): {exc}')

        if mask is None:
            mask = alpha_object_mask(image_with_alpha)
            has_object_mask = mask is not None and bool(np.any(mask))

        if mask is None and self.generate_object_masks:
            mask = self._generate_object_mask(index, image_with_alpha, source_size)
            has_object_mask = mask is not None and bool(np.any(mask))

        if mask is not None:
            # Store uint8 to keep a large dataset's in-memory SAM cache compact.
            mask = binary_object_mask(mask)
            self.object_masks[index] = mask.astype(np.uint8, copy=False)
        else:
            # An empty task-6 alpha prompt is safer than reusing the person/scene as
            # a false object. Explicitly disabled SAM retains full-image behavior.
            mask = (
                np.zeros(source_size, dtype=np.float32)
                if self.generate_object_masks
                else np.ones(source_size, dtype=np.float32)
            )
            if self.generate_object_masks:
                # Cache a failed/no-prompt-mask result too: retrying SAM for it on
                # every epoch wastes substantial GPU time and cannot improve it.
                self.object_masks[index] = mask.astype(np.uint8, copy=False)

        mask = crop_and_resize_object_mask(
            mask,
            crop_bbox=crop_bbox,
            source_size=source_size,
            output_size=(256, 256),
        )
        return mask, has_object_mask

    def __getitem__(self, index):
        item = {}

        # Load image
        img_path = self.images[index]
        img_path = os.path.join(self.dataset_base_path, img_path)
        try:
            # Default OpenCV loading drops PNG alpha, so use IMREAD_UNCHANGED to
            # preserve a SAM foreground mask stored as RGBA alpha.
            loaded_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if loaded_img is None:
                raise FileNotFoundError(img_path)
            if loaded_img.ndim == 2:
                loaded_img = cv2.cvtColor(loaded_img, cv2.COLOR_GRAY2BGR)
            if loaded_img.ndim != 3 or loaded_img.shape[2] < 3:
                raise ValueError(f'Expected RGB(A) image, got shape {loaded_img.shape}')

            source_size = loaded_img.shape[:2]
            crop_bbox = None
            if self.crop_bbox is not None:
                candidate_bbox = [int(v) for v in self.crop_bbox[index]]
                if candidate_bbox[2] > candidate_bbox[0] and candidate_bbox[3] > candidate_bbox[1]:
                    crop_bbox = candidate_bbox

            object_mask, has_object_mask = self._object_mask_for_image(
                index, loaded_img, source_size, crop_bbox
            )
            img = loaded_img[:, :, :3]
            if crop_bbox is not None:
                x0, y0, x1, y1 = crop_bbox
                img = img[y0:y1, x0:x1]
            img_h, img_w, _ = img.shape
            if img_h == 0 or img_w == 0:
                raise ValueError(f'Empty image crop for {img_path}')
            img = cv2.resize(img, (256, 256), cv2.INTER_CUBIC)
            img = img.transpose(2, 0, 1) / 255.0
        except (FileNotFoundError, ValueError, TypeError, cv2.error) as exc:
            raise RuntimeError(f'Could not load image/mask pair {img_path}: {exc}') from exc

        img_scale_factor = np.array([256 / img_w, 256 / img_h])

        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
            transl = self.transl[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)
            transl = np.zeros(3)

        # Load vertex_contact
        if self.has_contact_3d[index]:
            contact_label_3d = self.contact_labels_3d[index]
        else:
            contact_label_3d = np.zeros(self.n_vertices)

        sem_mask = np.zeros((133, 256, 256))
        if self.sem_masks is not None:
            sem_mask_path = os.path.join(self.dataset_base_path, self.sem_masks[index])
            try:
                sem_mask = cv2.imread(sem_mask_path)
                sem_mask = cv2.resize(sem_mask, (256, 256), cv2.INTER_CUBIC)
                sem_mask = mask_split(sem_mask, 133)
            except:
                print('Scene seg: ', sem_mask_path)
                sem_mask = np.zeros((133, 256, 256))

        part_mask = np.zeros((26, 256, 256))
        if self.part_masks is not None:
            part_mask_path = os.path.join(self.dataset_base_path, self.part_masks[index])
            try:
                part_mask = cv2.imread(part_mask_path)
                part_mask = cv2.resize(part_mask, (256, 256), cv2.INTER_CUBIC)
                part_mask = mask_split(part_mask, 26)
            except:
                print('Part seg: ', part_mask_path)
                part_mask = np.zeros((26, 256, 256))

        try:
            if self.has_polygon_contact_2d[index]:
                polygon_contact_2d_path = self.polygon_contacts_2d[index]
                polygon_contact_2d_path = os.path.join(self.dataset_base_path, polygon_contact_2d_path)
                polygon_contact_2d = cv2.imread(polygon_contact_2d_path)
                polygon_contact_2d = cv2.resize(polygon_contact_2d, (256, 256), cv2.INTER_NEAREST)
                # binarize the part mask
                polygon_contact_2d = np.where(polygon_contact_2d > 0, 1, 0)
            else:
                polygon_contact_2d = np.zeros((256, 256, 3))
        except:
            print('2D polygon contact: ', polygon_contact_2d_path)

        if polygon_contact_2d is None:
            polygon_contact_2d = np.zeros((256, 256, 3), dtype=np.float32)

        if self.normalize:
            img = torch.tensor(img, dtype=torch.float32)
            item['img'] = self.normalize_img(img)
        else:
            item['img'] = torch.tensor(img, dtype=torch.float32)

        if self.is_smplx[index]:
            # Add 6 zeros to the end of the pose vector to match with smpl
            pose = np.concatenate((pose, np.zeros(6)))

        item['img_path'] = img_path
        item['pose'] = torch.tensor(pose, dtype=torch.float32)
        item['betas'] = torch.tensor(betas, dtype=torch.float32)
        item['transl'] = torch.tensor(transl, dtype=torch.float32)
        item['cam_k'] = self.cam_k[index]
        item['img_scale_factor'] = torch.tensor(img_scale_factor, dtype=torch.float32)
        item['contact_label_3d'] = torch.tensor(contact_label_3d, dtype=torch.float32)
        item['sem_mask'] = torch.tensor(sem_mask, dtype=torch.float32)
        item['part_mask'] = torch.tensor(part_mask, dtype=torch.float32)
        item['polygon_contact_2d'] = torch.tensor(polygon_contact_2d, dtype=torch.float32)
        # Binary [1, 256, 256] nearby non-person SAM proposal, pixel-aligned with
        # img. ``object_mask`` remains available for task-4/checkpoint compatibility.
        object_prompt = torch.tensor(object_mask[None], dtype=torch.float32)
        item['object_prompt'] = object_prompt
        item['object_mask'] = object_prompt
        item['has_object_mask'] = torch.tensor(float(has_object_mask), dtype=torch.float32)

        # 2D keypoint prompts (COCO-17), in ORIGINAL-image pixels, + per-keypoint confidence.
        # Normalization into the crop and COCO->mhr70 labelling happens in the trainer via
        # utils.keypoint_prompts.build_keypoint_prompts (it needs img_scale_factor).
        # Coerce to a fixed (K, 2) / (K,) so the DataLoader can collate. Some samples have
        # 0 detected keypoints (stored as (0, 2)); missing rows stay zero with confidence 0
        # (-> marked invalid downstream by build_keypoint_prompts).
        K = 17
        kp = np.zeros((K, 2), dtype=np.float32)
        kpc = np.zeros((K,), dtype=np.float32)
        n = 0
        if self.has_keypoints[index]:
            kp_raw = np.asarray(self.keypoints_2d[index], dtype=np.float32)
            kpc_raw = np.asarray(self.keypoint_conf[index], dtype=np.float32)
            if kp_raw.ndim == 2 and kp_raw.shape[1] == 2:
                n = min(K, kp_raw.shape[0])
                kp[:n] = kp_raw[:n]
                m = min(n, kpc_raw.shape[0]) if kpc_raw.ndim == 1 else 0
                kpc[:m] = kpc_raw[:m]
        item['keypoints_2d'] = torch.tensor(kp, dtype=torch.float32)
        item['keypoint_conf'] = torch.tensor(kpc, dtype=torch.float32)
        item['has_keypoints'] = torch.tensor(1.0 if n > 0 else 0.0, dtype=torch.float32)

        item['has_smpl'] = self.has_smpl[index]
        item['is_smplx'] = self.is_smplx[index]
        item['has_contact_3d'] = self.has_contact_3d[index]
        item['has_polygon_contact_2d'] = self.has_polygon_contact_2d[index]

        return item

    def __len__(self):
        return len(self.images)


'''
data:
[
'imgname', 
'pose', 
'transl', 
'shape', 
'cam_k', 
'polygon_2d_contact', 
'contact_label', 
'scene_seg', 
'part_seg', 
'contact_label_smplx', 
'contact_label_objectwise', 
'contact_label_smplx_objectwise', 
'keypoint_2d', 
'keypoint_conf',
'object_mask' / 'object_mask_path' / 'sam_mask' / 'sam_mask_path' (optional)
]
'''
