# /home/l_z80934/projects/deco/models/sam3d_encoder.py

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAM3DBodyEncoderWithPrompts(nn.Module):
    """
    SAM-3D-Body encoder wrapper for DECO integration with 2D keypoint prompt support.
    """

    def __init__(
            self,
            checkpoint_path=None,
            mhr_path=None,
            project_to_dim=256,
            use_prompts=True,
            freeze_backbone=False,
            freeze_decoder=True,
            device='cuda',
            return_patchified=False,
    ):
        super(SAM3DBodyEncoderWithPrompts, self).__init__()

        self.project_to_dim = project_to_dim
        self.use_prompts = use_prompts
        self.return_patchified = return_patchified
        self.device = device

        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required for SAM-3D-Body")

        # Import SAM-3D-Body - use correct function name
        try:
            from sam_3d_body import load_sam_3d_body
        except ImportError:
            raise ImportError(
                "SAM-3D-Body not found. Make sure it's in PYTHONPATH or installed."
            )

        # Load SAM-3D-Body model
        print(f"Loading SAM-3D-Body from {checkpoint_path}")
        mhr_path = mhr_path or ""
        self.model, self.model_cfg = load_sam_3d_body(
            checkpoint_path=checkpoint_path,
            device=device,
            mhr_path=mhr_path
        )
        self.model.eval()

        # Get backbone info
        self.backbone = self.model.backbone
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size

        # Projection layer
        if self.embed_dim != project_to_dim:
            # self.projection = nn.Linear(self.embed_dim, project_to_dim)
            self.upsample = nn.Sequential(
                # 1. Reduce channels: 1280 -> 256
                nn.Conv2d(self.embed_dim, project_to_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(project_to_dim),
                nn.ReLU(inplace=True),
                # 2. Upsample 4x: 16x16 -> 64x64
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
            )
        else:
            self.projection = nn.Identity()

        # Freeze components
        if freeze_backbone:
            self._freeze_backbone()
        if freeze_decoder:
            self._freeze_decoder()

        # Prompt fusion layers
        if self.use_prompts:
            self._init_prompt_fusion()

    def _init_prompt_fusion(self):
        """Initialize prompt fusion layers"""
        self.prompt_fusion = nn.Sequential(
            nn.Conv2d(self.project_to_dim * 2, self.project_to_dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.project_to_dim),
            nn.ReLU(inplace=True),
        )
        self.prompt_norm = nn.LayerNorm(self.project_to_dim)

    def _freeze_backbone(self):
        """Freeze SAM-3D-Body backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("✓ SAM-3D-Body backbone frozen")

    def _freeze_decoder(self):
        """Freeze SAM-3D-Body decoder"""
        if hasattr(self.model, 'decoder'):
            for param in self.model.decoder.parameters():
                param.requires_grad = False
        print("✓ SAM-3D-Body decoder frozen")

    def _patchified_to_spatial(self, features, height, width):
        """Convert patchified (B, K, D) to spatial (B, D, H_patch, W_patch)"""
        B, K, D = features.shape
        H_patch = height // self.patch_size
        W_patch = width // self.patch_size

        spatial = features.reshape(B, H_patch, W_patch, D)
        spatial = spatial.permute(0, 3, 1, 2)
        return spatial

    def _embed_keypoints(self, keypoints_2d, img_height, img_width):
        """Convert 2D keypoints to spatial heatmap"""
        if keypoints_2d.dim() == 2:
            keypoints_2d = keypoints_2d.unsqueeze(0)

        B, N, _ = keypoints_2d.shape
        H_patch = img_height // self.patch_size
        W_patch = img_width // self.patch_size

        # Normalize to patch grid
        kp_normalized = keypoints_2d.clone().float()
        kp_normalized[..., 0] = (kp_normalized[..., 0] / img_width) * W_patch
        kp_normalized[..., 1] = (kp_normalized[..., 1] / img_height) * H_patch

        # Create keypoint heatmaps
        keypoint_maps = torch.zeros(
            B, N, H_patch, W_patch,
            device=keypoints_2d.device,
            dtype=torch.float32
        )

        sigma = 1.5
        for b in range(B):
            for k in range(N):
                x, y = kp_normalized[b, k]
                if 0 <= x < W_patch and 0 <= y < H_patch:
                    xx, yy = torch.meshgrid(
                        torch.arange(W_patch, device=keypoints_2d.device, dtype=torch.float32),
                        torch.arange(H_patch, device=keypoints_2d.device, dtype=torch.float32),
                        indexing='xy'
                    )
                    dist = ((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)
                    heatmap = torch.exp(-dist)
                    keypoint_maps[b, k] = heatmap

        # Pool across keypoints
        keypoint_map = keypoint_maps.max(dim=1)[0].unsqueeze(1)
        keypoint_features = keypoint_map.expand(B, self.project_to_dim, H_patch, W_patch)

        return keypoint_features

    def forward(self, x, keypoints_2d=None):
        """
        Args:
            x: (B, 3, 256, 256)
            keypoints_2d: (N, 2), (B, N, 2), or None

        Returns:
            (B, K, D_proj) or (B, D_proj, H_patch, W_patch)
        """
        B, C, H, W = x.shape

        # Extract backbone features
        with torch.no_grad():
            patchified = self.backbone(x)

        # No prompts case
        if not self.use_prompts or keypoints_2d is None:
            if isinstance(self.projection, nn.Identity):
                projected = patchified
            else:
                projected = self.projection(patchified)

            if self.return_patchified:
                return projected
            else:
                spatial = self._patchified_to_spatial(projected, H, W)
                return spatial

        # With prompts
        spatial = self._patchified_to_spatial(patchified, H, W)

        # Project
        if not isinstance(self.projection, nn.Identity):
            B_s, D_s, H_s, W_s = spatial.shape
            spatial = spatial.permute(0, 2, 3, 1)
            spatial = spatial.reshape(B_s * H_s * W_s, D_s)
            spatial = self.projection(spatial)
            spatial = spatial.reshape(B_s, H_s, W_s, -1).permute(0, 3, 1, 2)

        # Embed keypoints
        keypoint_features = self._embed_keypoints(keypoints_2d, H, W)

        # Fuse
        fused = torch.cat([spatial, keypoint_features], dim=1)
        fused = self.prompt_fusion(fused)

        if self.return_patchified:
            B_f, D_f, H_f, W_f = fused.shape
            patchified = fused.permute(0, 2, 3, 1)
            patchified = patchified.reshape(B_f, H_f * W_f, D_f)
            return patchified
        else:
            return fused


class SAM3DBodyEncoder(nn.Module):
    """Simple SAM-3D-Body encoder without prompt support"""

    def __init__(
            self,
            checkpoint_path=None,
            mhr_path=None,
            project_to_dim=256,
            freeze_backbone=False,
            freeze_decoder=True,
            device='cuda',
            return_patchified=False,
    ):
        super(SAM3DBodyEncoder, self).__init__()

        self.project_to_dim = project_to_dim
        self.return_patchified = return_patchified
        self.device = device
        self.freeze_backbone = freeze_backbone

        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required")

        try:
            from sam_3d_body import load_sam_3d_body
        except ImportError:
            raise ImportError("SAM-3D-Body can not be loaded !!!")

        print(f"Loading SAM-3D-Body from {checkpoint_path}")
        mhr_path = mhr_path or ""
        model, self.model_cfg = load_sam_3d_body(
            checkpoint_path=checkpoint_path,
            device=device,
            mhr_path=mhr_path,
        )

        # Keep ONLY the backbone as a tracked sub-module. The rest of SAM-3D-Body
        # (promptable decoder + MHR/camera heads, hundreds of millions of params) is
        # unused for contact prediction; dropping it keeps both the optimizer state
        # (encoder_part.parameters()) and saved checkpoints small.
        self.backbone = model.backbone
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size
        # The backbone may have been cast to fp16/bf16 at load time (cfg.TRAIN.USE_FP16).
        self.backbone_dtype = getattr(model, "backbone_dtype", torch.float32)
        del model  # free the unused decoder / heads / MHR model

        # Optional channel projection, applied on the spatial map with a 1x1 conv.
        # When project_to_dim is None or == embed_dim we keep the native channels
        # (no projection), so the output stays (B, embed_dim, H_patch, W_patch),
        # e.g. (B, 1280, 16, 16) for a 256x256 input.
        if project_to_dim is not None and project_to_dim != self.embed_dim:
            self.projection = nn.Conv2d(self.embed_dim, project_to_dim, kernel_size=1)
        else:
            self.projection = None

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✓ SAM-3D-Body backbone frozen")

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) RGB crop, ImageNet-normalized (same stats SAM-3D-Body uses).

        Returns:
            (B, C, H_patch, W_patch) spatial feature map, e.g. (B, 1280, 16, 16) for a
            256x256 input with patch_size 16. C == embed_dim unless a projection is set.
        """
        # The DINOv3/ViT backbone already returns a spatial map (B, C, Hp, Wp), so no
        # patchified -> spatial reshape is needed here.
        backbone_ctx = (
            torch.no_grad() if self.freeze_backbone else contextlib.nullcontext()
        )
        with backbone_ctx:
            feat = self.backbone(x.to(self.backbone_dtype))     # (B, 1280, 16, 16)

        if isinstance(feat, (tuple, list)):
            feat = feat[-1]
        feat = feat.float()  # cast back to fp32 for the rest of DECO

        if self.projection is not None:
            feat = self.projection(feat)

        return feat