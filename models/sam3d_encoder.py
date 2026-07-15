# /home/l_z80934/projects/deco/models/sam3d_encoder.py

import contextlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAM3DBodyEncoderWithPrompts(nn.Module):
    """SAM-3D-Body backbone + native PromptEncoder for DECO.

    Returns the backbone spatial feature map and, when 2D keypoint prompts are given,
    a set of sparse prompt tokens produced by SAM-3D-Body's *pretrained* PromptEncoder.
    DECO appends those tokens to the fused image tokens so the per-vertex contact head
    can attend to them (see models/deco.py).

    Only the backbone + prompt_encoder are kept as sub-modules; the MHR pose decoder /
    camera & pose heads (hundreds of millions of params) are dropped to keep the
    optimizer state and saved checkpoints small.
    """

    def __init__(
            self,
            checkpoint_path=None,
            mhr_path=None,
            project_to_dim=None,
            use_prompts=True,
            num_body_joints=70,
            freeze_backbone=True,
            # unfreeze_last_n_blocks=1,
            freeze_prompt_encoder=False,
            device='cuda',
    ):
        super(SAM3DBodyEncoderWithPrompts, self).__init__()

        self.use_prompts = use_prompts
        self.freeze_backbone = freeze_backbone
        self.device = device

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

        # Backbone (DINOv3/ViT) -> spatial feature map (B, embed_dim, H_patch, W_patch).
        self.backbone = model.backbone
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size
        # The backbone may have been cast to fp16/bf16 at load time (cfg.TRAIN.USE_FP16).
        self.backbone_dtype = getattr(model, "backbone_dtype", torch.float32)

        # Reuse the checkpoint's *pretrained* prompt encoder if present; otherwise build
        # a fresh one. Kept in fp32 (it is tiny) for stable prompt-token embedding. It
        # maps keypoints (B, N, 3) -> sparse tokens (B, N, embed_dim).
        self.prompt_encoder = None
        if self.use_prompts:
            pe = getattr(model, "prompt_encoder", None)
            if pe is not None:
                self.prompt_encoder = pe
                print("✓ Reusing SAM-3D-Body pretrained prompt encoder")
            else:
                from sam_3d_body.models.decoders import PromptEncoder
                self.prompt_encoder = PromptEncoder(
                    embed_dim=getattr(self.backbone, "embed_dims", self.embed_dim),
                    num_body_joints=num_body_joints,
                )
                print("! No pretrained prompt encoder found; built a fresh one")
            self.prompt_encoder.float()

        del model  # free the unused decoder / heads / MHR model

        # Optional channel projection of the feature map (1x1 conv). For DECO we use
        # project_to_dim == embed_dim (1280) -> no projection, so the feature map and the
        # prompt tokens share the same channel dim. When a projection IS used, prompt_proj
        # keeps the prompt tokens aligned to the same output dim.
        if project_to_dim is not None and project_to_dim != self.embed_dim:
            self.projection = nn.Conv2d(self.embed_dim, project_to_dim, kernel_size=1)
            self.prompt_proj = (
                nn.Linear(self.embed_dim, project_to_dim) if self.use_prompts else None
            )
        else:
            self.projection = None
            self.prompt_proj = None

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✓ SAM-3D-Body backbone frozen")

        # Partially unfreeze the last N transformer blocks (+ final norm) so the top of the
        # backbone adapts to contact while the rest stays frozen. The backbone runs in bf16,
        # which fine-tunes without a grad scaler; only these blocks store activations / get
        # gradients (the ViT runs on 256 tokens), so the memory overhead is small. Train them
        # with a *small* LR -- TrainStepper puts them in a dedicated optimizer (lr x 0.1).
        # self.unfreeze_last_n_blocks = int(unfreeze_last_n_blocks or 0)
        # if self.unfreeze_last_n_blocks > 0:
        #     enc = self.backbone.encoder
        #     blocks = getattr(enc, "blocks", None)
        #     if blocks is None:
        #         print("! backbone has no .blocks; cannot partially unfreeze (staying frozen)")
        #     else:
        #         n = min(self.unfreeze_last_n_blocks, len(blocks))
        #         for blk in list(blocks)[-n:]:
        #             for param in blk.parameters():
        #                 param.requires_grad = True
        #         if hasattr(enc, "norm"):          # final norm feeds the output features
        #             for param in enc.norm.parameters():
        #                 param.requires_grad = True
        #
        #         # bf16 weights can't absorb the small fine-tuning updates (they underflow the
        #         # ~7-bit mantissa and round to zero), so cast the whole backbone to fp32 for
        #         # trainable fine-tuning. ~+1.7GB -> pair with a smaller cross_grid / batch on 12GB.
        #         self.backbone.float()
        #         self.backbone_dtype = torch.float32
        #         print("✓ Backbone cast to fp32 for fine-tuning")
        #         print(f"✓ Unfroze last {n}/{len(blocks)} SAM-3D-Body backbone blocks (+ final norm)")

        if self.use_prompts and freeze_prompt_encoder:
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False
            print("✓ SAM-3D-Body prompt encoder frozen")

        # Run the backbone with autograd iff any of its params are trainable.
        self._backbone_trainable = any(p.requires_grad for p in self.backbone.parameters())

    def backbone_finetune_parameters(self):
        """Trainable backbone params (the unfrozen blocks + final norm); empty if frozen."""
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def task_parameters(self):
        """All encoder params EXCEPT the unfrozen backbone, so the task optimizers can skip
        them and a dedicated low-LR optimizer can fine-tune the backbone (see TrainStepper)."""
        ft = set(id(p) for p in self.backbone_finetune_parameters())
        return [p for p in self.parameters() if id(p) not in ft]

    def forward(self, x, keypoints=None):
        """
        Args:
            x: (B, 3, H, W) RGB crop, ImageNet-normalized (same stats SAM-3D-Body uses).
            keypoints: (B, N, 3) prompt keypoints, or None. The last channel is a label:
                a joint index in [0, num_body_joints), -1 = "not a point",
                -2 = "invalid/dummy". (x, y) must be normalized to [0, 1] in the crop.

        Returns:
            feat: (B, C, H_patch, W_patch) spatial feature map, e.g. (B, 1280, 16, 16) for
                a 256x256 input with patch_size 16. C == embed_dim unless a projection is set.
            prompt_tokens: (B, N, C) sparse prompt tokens, or None when no keypoints given.
        """
        # The DINOv3/ViT backbone already returns a spatial map (B, C, Hp, Wp).
        # backbone_ctx = (
        #     contextlib.nullcontext() if self._backbone_trainable else torch.no_grad()
        # )
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

        prompt_tokens = None
        if self.use_prompts and keypoints is not None:
            # sparse_emb: (B, N, embed_dim); sparse_mask: (B, N) (unused here)
            sparse_emb, _ = self.prompt_encoder(keypoints=keypoints.float())
            prompt_tokens = sparse_emb.float()
            if self.prompt_proj is not None:
                prompt_tokens = self.prompt_proj(prompt_tokens)

        return feat, prompt_tokens


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


class SAM3DObjectsEncoder(nn.Module):
    """Image-and-mask backbone from SAM-3D-Objects, adapted to DECO feature maps.

    SAM-3D-Objects' ``ss_encoder.ckpt`` is a voxel VAE encoder and cannot consume an
    RGB image. Its paired RGB and alpha-mask DINO conditioners live in
    ``ss_generator.yaml`` under ``module_list.{0,1}.backbone``. This wrapper loads
    both frozen condition encoders, fuses their spatial patch features, then projects
    them to DECO's channel width and pools them to the body branch's spatial grid.

    ``forward`` returns ``(B, project_to_dim, H_out, W_out)``.  For the default
    256x256 DECO crop this is ``(B, 1280, 16, 16)``, matching
    :class:`SAM3DBodyEncoderWithPrompts`.
    """

    def __init__(
            self,
            checkpoint_path=None,
            config_path=None,
            project_to_dim=None,
            output_size=(16, 16),
            image_embedder_index=0,
            mask_embedder_index=1,
            freeze_backbone=False,
            input_is_normalized=True,
            device='cuda',
    ):
        super(SAM3DObjectsEncoder, self).__init__()

        self.project_to_dim = project_to_dim
        self.freeze_backbone = freeze_backbone
        self.output_size = output_size
        self.input_is_normalized = input_is_normalized
        self.image_embedder_index = image_embedder_index
        self.mask_embedder_index = mask_embedder_index

        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required")

        checkpoint_path = Path(checkpoint_path)
        if config_path is None:
            config_path = checkpoint_path.with_name('ss_generator.yaml')
        config_path = Path(config_path)

        self.model_cfg, image_cfg = self._load_image_backbone_config(
            config_path, image_embedder_index
        )
        _, mask_cfg = self._load_image_backbone_config(config_path, mask_embedder_index)
        self.dino_model = image_cfg.get('dino_model', 'dinov2_vitl14_reg')
        self.input_size = image_cfg.get('input_size', 518)
        self.normalize_images = bool(image_cfg.get('normalize_images', True))
        self.repo_or_dir = image_cfg.get('repo_or_dir', 'facebookresearch/dinov2')
        self.source = image_cfg.get('source', 'github')
        self.mask_dino_model = mask_cfg.get('dino_model', self.dino_model)
        self.mask_input_size = mask_cfg.get('input_size', self.input_size)
        self.mask_normalize_images = bool(mask_cfg.get('normalize_images', True))
        self.mask_repo_or_dir = mask_cfg.get('repo_or_dir', self.repo_or_dir)
        self.mask_source = mask_cfg.get('source', self.source)

        print(
            f"Loading SAM-3D-Objects RGB/mask DINO "
            f"({self.dino_model}/{self.mask_dino_model}) from "
            f"{checkpoint_path}"
        )
        self.backbone = self._build_backbone(
            self.dino_model, self.repo_or_dir, self.source
        )
        self.mask_backbone = self._build_backbone(
            self.mask_dino_model, self.mask_repo_or_dir, self.mask_source
        )
        self._load_backbone_weights(
            checkpoint_path,
            {
                image_embedder_index: self.backbone,
                mask_embedder_index: self.mask_backbone,
            },
        )

        self.embed_dim = getattr(self.backbone, 'embed_dim', None)
        if self.embed_dim is None:
            raise RuntimeError('SAM-3D-Objects DINO backbone has no embed_dim')
        self.mask_embed_dim = getattr(self.mask_backbone, 'embed_dim', None)
        if self.mask_embed_dim is None:
            raise RuntimeError('SAM-3D-Objects mask DINO backbone has no embed_dim')
        patch_embed = getattr(self.backbone, 'patch_embed', None)
        self.patch_size = getattr(patch_embed, 'patch_size', 14)
        mask_patch_embed = getattr(self.mask_backbone, 'patch_embed', None)
        self.mask_patch_size = getattr(mask_patch_embed, 'patch_size', 14)

        # SAM-3D-Objects conditions on RGB and the alpha mask with separate DINO
        # encoders. DECO needs a spatial map rather than concatenated token streams,
        # so use a trainable 1x1 fusion before the existing output projection.
        self.image_mask_fusion = nn.Conv2d(
            self.embed_dim + self.mask_embed_dim, self.embed_dim, kernel_size=1
        )
        self._initialize_image_mask_fusion()

        if project_to_dim is not None and project_to_dim != self.embed_dim:
            self.projection = nn.Conv2d(self.embed_dim, project_to_dim, kernel_size=1)
        else:
            self.projection = None

        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.mask_backbone.requires_grad_(False)
            self.backbone.eval()
            self.mask_backbone.eval()
            # Run both frozen SAM-3D-Objects DINO conditioners in fp16 on CUDA,
            # while projection/fusion layers remain fp32 for stable training.
            if str(device).startswith('cuda'):
                self.backbone.half()
                self.mask_backbone.half()
            print("✓ SAM-3D-Objects backbone frozen")

        self.backbone_dtype = next(self.backbone.parameters()).dtype
        self.mask_backbone_dtype = next(self.mask_backbone.parameters()).dtype
        self.register_buffer(
            'image_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            'image_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    @staticmethod
    def _load_image_backbone_config(config_path, image_embedder_index):
        """Read one DINO conditioner entry from the local generator config."""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                'PyYAML is required to read the SAM-3D-Objects generator config'
            ) from exc

        if not config_path.is_file():
            raise FileNotFoundError(
                f'SAM-3D-Objects generator config not found: {config_path}'
            )
        with config_path.open('r') as handle:
            config = yaml.safe_load(handle)

        try:
            embedder_list = config['module']['condition_embedder']['backbone']['embedder_list']
            image_cfg = embedder_list[image_embedder_index][0]
        except (IndexError, KeyError, TypeError) as exc:
            raise RuntimeError(
                'Could not find the RGB DINO conditioner in '
                f'{config_path} at index {image_embedder_index}'
            ) from exc

        target = str(image_cfg.get('_target_', '')) if isinstance(image_cfg, dict) else ''
        if not target.endswith('.Dino'):
            raise RuntimeError(
                'The selected SAM-3D-Objects conditioner is not a DINO image '
                f'backbone: {target or image_cfg!r}'
            )
        return config, image_cfg

    def _initialize_image_mask_fusion(self):
        """Start fusion as an equal per-channel RGB/mask feature average."""
        with torch.no_grad():
            self.image_mask_fusion.weight.zero_()
            self.image_mask_fusion.bias.zero_()
            shared_channels = min(self.embed_dim, self.mask_embed_dim)
            channel_ids = torch.arange(shared_channels)
            self.image_mask_fusion.weight[channel_ids, channel_ids, 0, 0] = 0.5
            self.image_mask_fusion.weight[
                channel_ids, self.embed_dim + channel_ids, 0, 0
            ] = 0.5
            if self.embed_dim > shared_channels:
                channel_ids = torch.arange(shared_channels, self.embed_dim)
                self.image_mask_fusion.weight[channel_ids, channel_ids, 0, 0] = 1.0

    @staticmethod
    def _patch_size_value(patch_size):
        return patch_size[0] if isinstance(patch_size, (tuple, list)) else patch_size

    def _build_backbone(self, dino_model, repo_or_dir, source):
        """Instantiate the DINO module without downloading generic pretrained weights.

        SAM-3D-Objects supplies its own DINO weights in ``ss_generator.ckpt``.  Using
        ``pretrained=False`` avoids an unnecessary network download and lets this work
        from the checked-out DINOv2 hub cache used by this project.
        """
        hub_dir = Path(torch.hub.get_dir())
        local_repos = sorted(hub_dir.glob('facebookresearch_dinov2_*'))
        if local_repos:
            return torch.hub.load(
                str(local_repos[0]), dino_model, source='local', pretrained=False
            )
        return torch.hub.load(
            repo_or_dir, dino_model, source=source, pretrained=False
        )

    def _load_backbone_weights(self, checkpoint_path, backbones_by_index):
        """Load RGB and mask DINO weights from one mmap-backed checkpoint read."""
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f'SAM-3D-Objects generator checkpoint not found: {checkpoint_path}'
            )

        # mmap prevents the 6.3GB generator checkpoint from being fully materialized
        # in CPU RAM while we copy only the selected DINO weights into this module.
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location='cpu', mmap=True, weights_only=True
            )
        except TypeError:  # PyTorch versions before mmap/weights_only support
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

        state_dict = checkpoint.get('state_dict', checkpoint)
        if not isinstance(state_dict, dict):
            raise RuntimeError(
                f'Unsupported SAM-3D-Objects checkpoint format: {checkpoint_path}'
            )

        for embedder_index, backbone in backbones_by_index.items():
            prefix = (
                '_base_models.condition_embedder.module_list.'
                f'{embedder_index}.backbone.'
            )
            backbone_state = {
                name[len(prefix):]: value
                for name, value in state_dict.items()
                if name.startswith(prefix)
            }
            if not backbone_state:
                raise RuntimeError(
                    'The checkpoint has no SAM-3D-Objects DINO weights under '
                    f'{prefix!r}. Use ss_generator.ckpt, not ss_encoder.ckpt.'
                )

            incompatible = backbone.load_state_dict(backbone_state, strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    'SAM-3D-Objects DINO checkpoint does not match conditioner '
                    f'index {embedder_index}; missing={incompatible.missing_keys}, '
                    f'unexpected={incompatible.unexpected_keys}'
                )
            del backbone_state

        del state_dict
        del checkpoint

    def train(self, mode=True):
        """Keep a frozen DINO backbone deterministic when DECO enters train mode."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
            self.mask_backbone.eval()
        return self

    @staticmethod
    def _patch_tokens_to_map(features, image, patch_size, branch_name):
        if not isinstance(features, dict) or 'x_norm_patchtokens' not in features:
            raise RuntimeError(
                f'SAM-3D-Objects {branch_name} DINO must return x_norm_patchtokens '
                'from forward_features()'
            )
        tokens = features['x_norm_patchtokens']
        patch_size = SAM3DObjectsEncoder._patch_size_value(patch_size)
        grid_h = image.shape[-2] // patch_size
        grid_w = image.shape[-1] // patch_size
        if tokens.shape[1] != grid_h * grid_w:
            raise RuntimeError(
                f'SAM-3D-Objects {branch_name} patch-token count does not match its '
                f'input grid: {tokens.shape[1]} tokens for {grid_h}x{grid_w}'
            )
        return tokens.transpose(1, 2).reshape(
            tokens.shape[0], tokens.shape[2], grid_h, grid_w
        ).float()

    def _prepare_rgba_inputs(self, x, object_mask):
        """Recreate SAM-3D-Objects' RGB + ALPHA_CHANNEL input contract."""
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f'Expected RGB input (B, 3, H, W), got {tuple(x.shape)}')

        image = x.float()
        if object_mask is None:
            object_mask = image.new_ones((image.shape[0], 1, *image.shape[-2:]))
        elif object_mask.ndim == 3:
            object_mask = object_mask.unsqueeze(1)
        if object_mask.ndim != 4 or object_mask.shape[1] != 1:
            raise ValueError(
                'Expected object_mask with shape (B, 1, H, W) or (B, H, W), got '
                f'{tuple(object_mask.shape)}'
            )
        if object_mask.shape[0] != image.shape[0]:
            raise ValueError(
                f'Object-mask batch size {object_mask.shape[0]} != image batch size '
                f'{image.shape[0]}'
            )

        object_mask = F.interpolate(
            object_mask.to(device=image.device, dtype=image.dtype),
            size=image.shape[-2:], mode='nearest',
        )
        object_mask = (object_mask > 0).to(dtype=image.dtype)
        rgba = torch.cat((image, object_mask), dim=1)
        return rgba[:, :3], rgba[:, 3:4]

    def forward(self, x, object_mask=None, output_size=None):
        """Return projected SAM-3D-Objects RGB+mask patch features.

        ``x`` is DECO's normalized RGB crop and ``object_mask`` is the corresponding
        binary foreground mask. The dataset applies the same crop/resize to both;
        this method then recreates SAM-3D-Objects' RGBA/ALPHA_CHANNEL input contract.
        """
        # Preserve the old ``encoder(x, output_size)`` call convention.
        if output_size is None and isinstance(object_mask, (int, tuple, list)):
            output_size, object_mask = object_mask, None

        image, alpha_mask = self._prepare_rgba_inputs(x, object_mask)
        mean = self.image_mean.to(dtype=image.dtype, device=image.device)
        std = self.image_std.to(dtype=image.dtype, device=image.device)
        if self.input_is_normalized:
            image = image * std + mean
        image = F.interpolate(
            image, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False
        )
        if self.normalize_images:
            image = (image - mean) / std

        # The mask DINO in ss_generator.yaml receives alpha as a one-channel image;
        # replicate it to RGB and use its own pretrained conditioner weights.
        mask_image = F.interpolate(
            alpha_mask, size=(self.mask_input_size, self.mask_input_size),
            mode='nearest',
        ).repeat(1, 3, 1, 1)
        if self.mask_normalize_images:
            mask_image = (mask_image - mean) / std

        backbone_ctx = (
            torch.no_grad() if self.freeze_backbone else contextlib.nullcontext()
        )
        with backbone_ctx:
            image_features = self.backbone.forward_features(image.to(self.backbone_dtype))
            mask_features = self.mask_backbone.forward_features(
                mask_image.to(self.mask_backbone_dtype)
            )

        image_feat = self._patch_tokens_to_map(
            image_features, image, self.patch_size, 'RGB'
        )
        mask_feat = self._patch_tokens_to_map(
            mask_features, mask_image, self.mask_patch_size, 'alpha-mask'
        )
        if mask_feat.shape[-2:] != image_feat.shape[-2:]:
            mask_feat = F.interpolate(
                mask_feat, size=image_feat.shape[-2:], mode='bilinear', align_corners=False
            )
        feat = self.image_mask_fusion(torch.cat((image_feat, mask_feat), dim=1))

        if self.projection is not None:
            feat = self.projection(feat)

        target_size = self.output_size if output_size is None else output_size
        if target_size is not None:
            if isinstance(target_size, int):
                target_size = (target_size, target_size)
            feat = F.adaptive_avg_pool2d(feat, target_size)

        return feat
