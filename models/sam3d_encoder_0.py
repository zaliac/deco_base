import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class SAM3DBodyEncoder(nn.Module):
    """
    SAM-3D-Body encoder wrapper for DECO integration.

    SAM-3D-Body outputs:
      - Shape: (B, K, D) where K = (H/patch_size)*(W/patch_size), D = embed_dim
      - Example: (B, 256, 768) for ViT-B with 256x256 input

    DECO expects:
      - Shape: (B, C, H, W) where C is feature dimension
      - For SAM ViT: (B, 256, 64, 64)

    This adapter:
    1. Loads SAM-3D-Body model
    2. Converts patchified outputs back to spatial format
    3. Projects to DECO's expected feature dimension
    """

    def __init__(
            self,
            checkpoint_path=None,
            backbone_type='dinov2_vitb14',
            project_to_dim=256,
            freeze_backbone=False,
            device='cuda'
    ):
        super(SAM3DBodyEncoder, self).__init__()
        self.project_to_dim = project_to_dim
        self.device = device

        # Import and build SAM-3D-Body model
        import sys
        sys.path.insert(0, '/home/l_z80934/projects/sam-3d-body')

        from sam_3d_body.build_models import build_model
        from sam_3d_body.utils.config import get_config

        # Load config
        if checkpoint_path:
            config_path = Path(checkpoint_path).parent / 'model_config.yaml'
            if config_path.exists():
                cfg = get_config(str(config_path))
            else:
                raise FileNotFoundError(f"Config not found at {config_path}")
        else:
            raise ValueError("checkpoint_path is required for SAM-3D-Body")

        # Build model
        self.sam3d_model = build_model(cfg, checkpoint_path, device)
        self.sam3d_model.eval()

        # Get feature info from backbone
        backbone = self.sam3d_model.backbone
        self.embed_dim = backbone.embed_dim
        self.patch_size = backbone.patch_size

        # Projection layer to match DECO expected dimensions
        if self.embed_dim != project_to_dim:
            self.projection = nn.Linear(self.embed_dim, project_to_dim)
        else:
            self.projection = nn.Identity()

        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze SAM-3D-Body backbone parameters"""
        for param in self.sam3d_model.backbone.parameters():
            param.requires_grad = False
        print("✓ SAM-3D-Body backbone frozen")

    def _patchified_to_spatial(self, features, height, width):
        """
        Convert patchified features back to spatial format.

        Args:
            features: (B, K, D) patchified features
            height: Original image height
            width: Original image width

        Returns:
            spatial_features: (B, D_proj, H_patch, W_patch)
        """
        B, K, D = features.shape

        # Compute spatial patch dimensions
        H_patch = height // self.patch_size
        W_patch = width // self.patch_size

        # Reshape and transpose: (B, K, D) -> (B, H_patch, W_patch, D) -> (B, D, H_patch, W_patch)
        spatial = features.reshape(B, H_patch, W_patch, D)
        spatial = spatial.permute(0, 3, 1, 2)  # (B, D, H_patch, W_patch)

        return spatial

    def forward(self, x, return_patchified=False):
        """
        Args:
            x: Input image (B, 3, 256, 256)
            return_patchified: If True, return patchified format; else return spatial

        Returns:
            features: Either (B, K, D) or (B, D_proj, H_patch, W_patch)
        """
        B, C, H, W = x.shape

        # Extract features using SAM-3D-Body backbone
        with torch.no_grad():
            patchified = self.sam3d_model.backbone(x)  # (B, K, D)

        if return_patchified:
            # Project and return patchified format
            projected = self.projection(patchified)  # (B, K, D_proj)
            return projected
        else:
            # Convert to spatial format
            spatial = self._patchified_to_spatial(patchified, H, W)  # (B, D, H_patch, W_patch)

            # Project to DECO dimension if needed
            if not isinstance(self.projection, nn.Identity):
                B, D, H_p, W_p = spatial.shape
                spatial = spatial.permute(0, 2, 3, 1).reshape(B * H_p * W_p, D)
                spatial = self.projection(spatial)
                spatial = spatial.reshape(B, H_p, W_p, -1).permute(0, 3, 1, 2)

            return spatial


class SAM3DBodyEncoderWithPrompts(SAM3DBodyEncoder):
    """SAM-3D-Body with optional 2D keypoint guidance"""

    def __init__(self, *args, use_prompts=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_prompts = use_prompts

        if use_prompts:
            # Add fusion layers for keypoint integration
            self.prompt_fusion = nn.Sequential(
                nn.Conv2d(256 + 256, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )

    def forward(self, x, keypoints_2d=None):
        features = super().forward(x, return_patchified=False)  # (B, 256, 64, 64)

        if self.use_prompts and keypoints_2d is not None:
            # Process keypoints similar to SAM encoder
            # (Implementation similar to SAMEncoderWithPrompts)
            pass

        return features