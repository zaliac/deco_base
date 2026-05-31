"""Build SAM-3D-Body keypoint prompts from stored 2D detections.

The DAMON `*_with_kpts.npz` files store `keypoint_2d` (COCO-17 layout, in ORIGINAL-image
pixels) and `keypoint_conf`. SAM-3D-Body's PromptEncoder expects keypoints as (B, N, 3) =
normalized (x, y) in [0, 1] + a label (mhr70 joint index, or -2 = invalid). This module
maps COCO-17 -> mhr70 labels and normalizes the coords into the 256 crop.
"""

import torch

# COCO-17 keypoint order -> SAM-3D-Body mhr70 label index (matched by joint name; see
# sam_3d_body/metadata/mhr70.py). COCO indices 0-8 coincide with mhr70; wrists/hips/
# knees/ankles are remapped (e.g. COCO left_wrist=9 -> mhr70 left_wrist=62).
#   COCO: 0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear, 5 l_sho, 6 r_sho, 7 l_elb, 8 r_elb,
#         9 l_wri, 10 r_wri, 11 l_hip, 12 r_hip, 13 l_knee, 14 r_knee, 15 l_ank, 16 r_ank
COCO17_TO_MHR70 = [
    0,    # nose
    1,    # left_eye
    2,    # right_eye
    3,    # left_ear
    4,    # right_ear
    5,    # left_shoulder
    6,    # right_shoulder
    7,    # left_elbow
    8,    # right_elbow
    62,   # left_wrist
    41,   # right_wrist
    9,    # left_hip
    10,   # right_hip
    11,   # left_knee
    12,   # right_knee
    13,   # left_ankle
    14,   # right_ankle
]


def build_keypoint_prompts(
    keypoints_2d,
    keypoint_conf,
    img_scale_factor,
    has_keypoints=None,
    crop_size=256,
    conf_thr=0.3,
    coco_to_mhr=COCO17_TO_MHR70,
):
    """Format stored 2D detections as SAM-3D-Body keypoint prompts.

    Args:
        keypoints_2d: (B, K, 2) keypoints in ORIGINAL-image pixels.
        keypoint_conf: (B, K) detector confidences.
        img_scale_factor: (B, 2) = [crop/orig_w, crop/orig_h] (maps orig px -> crop px).
        has_keypoints: (B,) 1.0 if the sample has real keypoints else 0.0 (-> all invalid).
        crop_size: side length of the square crop fed to the backbone (256).
        conf_thr: keypoints below this confidence (or off-frame) are marked invalid (-2).
        coco_to_mhr: list mapping each input keypoint index to its mhr70 label.

    Returns:
        keypoints: (B, K, 3) = normalized (x, y) in [0, 1] + label (mhr70 joint index, or
                   -2 for invalid). Directly consumable by the PromptEncoder.
    """
    device = keypoints_2d.device
    B, K, _ = keypoints_2d.shape

    sx = img_scale_factor[:, 0:1]                       # (B, 1)
    sy = img_scale_factor[:, 1:2]
    xn = (keypoints_2d[..., 0] * sx) / crop_size        # (B, K), normalized to [0, 1]
    yn = (keypoints_2d[..., 1] * sy) / crop_size

    labels = torch.tensor(coco_to_mhr, device=device, dtype=torch.float32)
    labels = labels.view(1, K).expand(B, K).clone()

    # Mark low-confidence / off-frame keypoints invalid (label -2). The PromptEncoder
    # still requires every coord in [0, 1], so we clamp afterwards.
    invalid = (
        (keypoint_conf < conf_thr)
        | (xn < 0) | (xn > 1) | (yn < 0) | (yn > 1)
    )
    if has_keypoints is not None:
        invalid = invalid | (has_keypoints.view(B, 1) < 0.5)
    labels[invalid] = -2

    xn = xn.clamp(0.0, 1.0)
    yn = yn.clamp(0.0, 1.0)
    keypoints = torch.stack([xn, yn, labels], dim=-1)   # (B, K, 3)
    return keypoints
