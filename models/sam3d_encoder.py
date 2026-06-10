# /home/l_z80934/projects/deco/models/sam3d_encoder.py

import contextlib

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
            unfreeze_last_n_blocks=0,
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

        # Partially unfreeze the top-N DINOv3 transformer blocks (+ final norm) so the top of
        # the backbone adapts to contact (and to any backbone-attention SSL task) while the rest
        # stays frozen. The backbone is a Dinov3Backbone whose ViT lives at backbone.encoder, so
        # the blocks are backbone.encoder.blocks (NOT backbone.blocks). Train them with a *small*
        # LR -- TrainStepper puts them in a dedicated optimizer (lr x 0.1).
        self.unfreeze_last_n_blocks = int(unfreeze_last_n_blocks or 0)
        self._ft_attn_modules = []
        if self.unfreeze_last_n_blocks > 0:
            enc = getattr(self.backbone, "encoder", None)
            blocks = getattr(enc, "blocks", None)
            if blocks is None:
                print("! backbone has no .encoder.blocks; cannot partially unfreeze (staying frozen)")
            else:
                # bf16 weights can't absorb the small fine-tuning updates (they underflow the
                # ~7-bit mantissa and round to zero), so cast the WHOLE backbone to fp32 (mixing
                # dtypes across the residual stream would break the frozen->unfrozen boundary).
                # ~+1.7GB -> pair with a smaller cross_grid / batch on 12GB.
                self.backbone.float()
                self.backbone_dtype = torch.float32
                n = min(self.unfreeze_last_n_blocks, len(blocks))
                for blk in list(blocks)[-n:]:
                    for param in blk.parameters():
                        param.requires_grad = True
                    attn = getattr(blk, "attn", None)
                    if attn is not None:
                        self._ft_attn_modules.append(attn)   # exposed for the SSL attention tap
                for norm_name in ("norm", "cls_norm"):       # final norm(s) feed the output features
                    norm_mod = getattr(enc, norm_name, None)
                    if norm_mod is not None:
                        for param in norm_mod.parameters():
                            param.requires_grad = True
                print(f"✓ Backbone cast to fp32; unfroze last {n}/{len(blocks)} DINOv3 blocks (+ final norm)")

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

    def finetuned_attn_modules(self):
        """The DINOv3 SelfAttention modules of the unfrozen top-N blocks. Empty if frozen.
        The backbone-attention SSL task taps these (utils/distill.py:tap_dinov3_attention)."""
        return list(self._ft_attn_modules)

    @property
    def backbone_token_prefix(self):
        """#(cls + storage) tokens before the patch tokens in the DINOv3 sequence -- dropped
        from the attention query axis so only patch tokens (which align across views) compare."""
        enc = getattr(self.backbone, "encoder", None)
        return int(getattr(enc, "n_storage_tokens", 0)) + 1 if enc is not None else 1

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
        # The DINOv3 backbone already returns a spatial map (B, C, Hp, Wp). Run it WITH autograd
        # iff some of its params are trainable (top-N blocks unfrozen); else no_grad. The frozen
        # lower blocks build no backward graph (their params/inputs don't require grad), so the
        # activation memory cost is only the unfrozen top-N blocks.
        backbone_ctx = (
            contextlib.nullcontext() if self._backbone_trainable else torch.no_grad()
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
