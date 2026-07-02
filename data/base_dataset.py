import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from common import constants
import os

def mask_split(img, num_parts):
    if not len(img.shape) == 2:
        img = img[:, :, 0]
    mask = np.zeros((img.shape[0], img.shape[1], num_parts))
    for i in np.unique(img):
        mask[:, :, i] = np.where(img == i, 1., 0.)
    return np.transpose(mask, (2, 0, 1))

class BaseDataset(Dataset):

    def __init__(self, dataset, mode, model_type='smpl', dataset_root_path='', normalize=False):
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

    def __getitem__(self, index):
        item = {}

        # Load image
        img_path = self.images[index]
        img_path = os.path.join(self.dataset_base_path, img_path)
        try:
            img = cv2.imread(img_path)
            if self.crop_bbox is not None:
                x0, y0, x1, y1 = [int(v) for v in self.crop_bbox[index]]
                if x1 > x0 and y1 > y0:
                    img = img[y0:y1, x0:x1]
            img_h, img_w, _ = img.shape
            img = cv2.resize(img, (256, 256), cv2.INTER_CUBIC)
            img = img.transpose(2, 0, 1) / 255.0
        except:
            print('Img: ', img_path)

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
'keypoint_conf'
]
'''