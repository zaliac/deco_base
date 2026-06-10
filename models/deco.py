from models.components import Encoder, Cross_Att, Spatial_Cross_Att, Decoder, Classifier, VertexContactDecoder
import torch.nn as nn
import torch.nn.functional as F
import torch

class DECO(nn.Module):
    def __init__(self, encoder, context, device, backbone_unfreeze_n=0):
        super(DECO, self).__init__()
        self.encoder_type = encoder
        self.context = context

        # Generic shared encoders only exist for the plain CNN/transformer backbones.
        # 'sam_hrnet' builds its own encoders inside the branch below.
        if self.encoder_type in ('hrnet', 'swin'):
            self.encoder_sem = Encoder(encoder=encoder).to(device)
            self.encoder_part = Encoder(encoder=encoder).to(device)
        if self.encoder_type == 'hrnet':
            if self.context:    
                self.decoder_sem = Decoder(480, 133, encoder=encoder).to(device)
                self.decoder_part = Decoder(480, 26, encoder=encoder).to(device)
            self.sem_pool = nn.AdaptiveAvgPool2d((1))
            self.part_pool = nn.AdaptiveAvgPool2d((1))
            self.cross_att = Cross_Att(480, 480).to(device)
            self.classif = Classifier(480).to(device)
        elif self.encoder_type == 'swin':
            self.correction_conv = nn.Conv1d(768, 1024, 1).to(device)
            if self.context:    
                self.decoder_sem = Decoder(1, 133, encoder=encoder).to(device)
                self.decoder_part = Decoder(1, 26, encoder=encoder).to(device)
            self.cross_att = Cross_Att(1024, 1024).to(device)
            self.classif = Classifier(1024).to(device)
        elif self.encoder_type == 'sam_hrnet':
            from models.sam3d_encoder import SAM3DBodyEncoderWithPrompts

            # SAM-3D-Body (DINOv3 ViT-H) checkpoint + model_config.yaml + MHR assets.
            sam_ckpt_path = 'data/weights/sam-3d-body-dinov3/model.ckpt'
            sam_mhr_path = 'data/weights/sam-3d-body-dinov3/assets/mhr_model.pt'

            feature_dim = 1280  # SAM-3D-Body backbone embed_dim (kept, no projection)

            # encoder_part: SAM-3D-Body backbone -> (B, 1280, 16, 16)
            # freeze_backbone=True keeps peak memory ~3.5GB (fits the RTX 3060 Ti /
            # A100 MIG). Fine-tuning the 840M ViT-H+ (freeze_backbone=False) OOMs even
            # on a 12GB card here, partly because TrainStepper puts encoder_part in two
            # Adam optimizers; to fine-tune, use a >=24GB GPU and remove the backbone
            # from optimizer_contact in train/trainer_step.py so its Adam state isn't
            # allocated twice.
            self.encoder_part = SAM3DBodyEncoderWithPrompts(
                checkpoint_path=sam_ckpt_path,
                mhr_path=sam_mhr_path,
                project_to_dim=feature_dim,   # == embed_dim -> no projection
                use_prompts=True,
                num_body_joints=70,           # mhr70 prompt-keypoint label space
                freeze_backbone=True,
                unfreeze_last_n_blocks=backbone_unfreeze_n,  # fine-tune top-N DINOv3 blocks (0 disables); low-LR via TrainStepper
                device=device,
            ).to(device)

            # encoder_sem: HRNet -> (B, 480, 64, 64); 1x1 conv projects 480 -> 1280
            # self.encoder_sem = Encoder(encoder='hrnet').to(device)
            # self.hrnet_to_sam = nn.Conv2d(480, feature_dim, kernel_size=1).to(device)
            # encoder_sem: HRNet -> (B, 480, 64, 64); strided conv projects 480 -> 1280
            # AND downsamples 64x64 -> 16x16 (learnable 4x4 pooling) to align with the SAM grid
            self.encoder_sem = Encoder(encoder='hrnet').to(device)
            self.hrnet_to_sam = nn.Conv2d(480, feature_dim, kernel_size=4, stride=4).to(device)

            if self.context:
                # decoder_sem decodes the (B,1280,16,16) projected HRNet map -> x16 -> (B,133,256,256)
                self.decoder_sem = Decoder(feature_dim, 133, encoder='sam_vit').to(device)
                # self.decoder_sem = Decoder(480, 133, encoder='hrnet').to(device)
                # decoder_part decodes the (B,1280,16,16) SAM map   -> x16 -> (B,26,256,256)
                self.decoder_part = Decoder(feature_dim, 26, encoder='sam_vit').to(device)

            # Spatial cross-attention over the per-location tokens of each branch (not a
            # single global-pooled token), so the SAM and HRNet features interact per
            # location before aggregation, preserving spatial detail. Both branches are on
            # the same 16x16 grid (sem downsampled by hrnet_to_sam), so grid_size=16 turns on
            # per-stream input norm + a shared 2D positional embedding inside the fusion.
            self.cross_att = Spatial_Cross_Att(feature_dim, num_heads=8, grid_size=16).to(device)
            # Per-vertex contact head: 6890 learnable vertex queries cross-attend to the
            # fused image tokens (replaces the global-vector MLP). Kept as `self.classif`
            # so TrainStepper's optimizer_contact (model.classif.parameters()) still
            # trains it without changes to train/trainer_step.py.
            self.classif = VertexContactDecoder(
                context_dim=feature_dim, num_vertices=6890,
                dim=256, num_heads=8, num_layers=3,
            ).to(device)
        else:
            NotImplementedError('Encoder type not implemented')

        self.device = device

    def forward(self, img, keypoints=None):
        if self.encoder_type == 'hrnet':
            sem_enc_out = self.encoder_sem(img)
            part_enc_out = self.encoder_part(img)

            if self.context:
                sem_mask_pred = self.decoder_sem(sem_enc_out)
                part_mask_pred = self.decoder_part(part_enc_out)

            sem_enc_out = self.sem_pool(sem_enc_out)
            sem_enc_out = sem_enc_out.squeeze(2)
            sem_enc_out = sem_enc_out.squeeze(2)
            sem_enc_out = sem_enc_out.unsqueeze(1)

            part_enc_out = self.part_pool(part_enc_out)
            part_enc_out = part_enc_out.squeeze(2)
            part_enc_out = part_enc_out.squeeze(2)
            part_enc_out = part_enc_out.unsqueeze(1)

            att = self.cross_att(sem_enc_out, part_enc_out)
            cont = self.classif(att)
        elif self.encoder_type == 'sam_hrnet':
            # part branch: SAM-3D-Body backbone -> (B, 1280, 16, 16), plus optional
            # prompt tokens (B, N, 1280) from the native (pretrained) PromptEncoder.
            part_enc_out, prompt_tokens = self.encoder_part(img, keypoints)

            # semantic branch: HRNet -> (B, 480, 64, 64) -> project + downsample to (B, 1280, 16, 16)
            sem_enc_out = self.encoder_sem(img)                 # (B, 480, 64, 64)
            sem_enc_out_new = self.hrnet_to_sam(sem_enc_out)    # (B, 1280, 16, 16) learnable downsample

            if self.context:
                sem_mask_pred = self.decoder_sem(sem_enc_out_new)  # (B,1280,16,16) -> (B,133,256,256)
                # sem_mask_pred = self.decoder_sem(sem_enc_out)       # (B, 480, 64, 64) -> (B,133,256,256)
                part_mask_pred = self.decoder_part(part_enc_out)   # (B, 26, 256, 256)

            # Tokenize both branches at their native 16x16. part (SAM) is the primary signal
            # (256 tokens). sem (HRNet) was projected AND downsampled 64x64 -> 16x16 by the
            # learnable strided conv self.hrnet_to_sam, so the 4x4 -> 1 spatial reduction is a
            # learned weighted combination (not a blunt avg-pool) and the two branches align
            # 1:1. The conv hard-codes the 64 -> 16 ratio, so assert the grids match in case
            # the SAM/HRNet resolutions ever change. (Cross-attention does not require equal
            # lengths, so sem could instead be kept at higher res -- at more memory tokens.)
            Hp, Wp = part_enc_out.shape[-2:]
            assert sem_enc_out_new.shape[-2:] == (Hp, Wp), \
                f"sem grid {tuple(sem_enc_out_new.shape[-2:])} != part grid {(Hp, Wp)}; adjust hrnet_to_sam stride"
            sem_tok = sem_enc_out_new.flatten(2).transpose(1, 2)   # (B, 256, 1280)
            part_tok = part_enc_out.flatten(2).transpose(1, 2)     # (B, 256, 1280)

            # bidirectional cross-attention; returns BOTH enriched streams concatenated on
            # the token axis -> (B, 256+256, 1280), preserving every token from both modalities.
            att = self.cross_att(sem_tok, part_tok)                # (B, 512, 1280)
            # append the keypoint prompt tokens as extra memory for the contact head
            if prompt_tokens is not None:
                att = torch.cat([att, prompt_tokens], dim=1)       # (B, 512 + N, 1280)
            # vertex queries cross-attend to those tokens -> per-vertex contact
            cont = self.classif(att)                               # (B, 6890)
        else:
            sem_enc_out = self.encoder_sem(img)
            part_enc_out = self.encoder_part(img)

            sem_seg = torch.reshape(sem_enc_out, (-1, 768, 1))		
            part_seg = torch.reshape(part_enc_out, (-1, 768, 1))		

            sem_seg = self.correction_conv(sem_seg)		
            part_seg = self.correction_conv(part_seg)		

            sem_seg = torch.reshape(sem_seg, (-1, 1, 32, 32))		
            part_seg = torch.reshape(part_seg, (-1, 1, 32, 32))

            if self.context:
                sem_mask_pred = self.decoder_sem(sem_seg)
                part_mask_pred = self.decoder_part(part_seg)

            sem_enc_out = torch.reshape(sem_seg, (-1, 1, 1024))
            part_enc_out = torch.reshape(part_seg, (-1, 1, 1024))

            att = self.cross_att(sem_enc_out, part_enc_out)
            cont = self.classif(att)

        if self.context: return cont, sem_mask_pred, part_mask_pred
        return cont
