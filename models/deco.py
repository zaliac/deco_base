from models.components import Encoder, Cross_Att, Spatial_Cross_Att, Decoder, Classifier, VertexContactDecoder
import torch.nn as nn
import torch.nn.functional as F
import torch

class DECO(nn.Module):
    def __init__(self, encoder, context, device):
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
                device=device,
            ).to(device)

            # encoder_sem: HRNet -> (B, 480, 64, 64); 1x1 conv projects 480 -> 1280
            self.encoder_sem = Encoder(encoder='hrnet').to(device)
            self.hrnet_to_sam = nn.Conv2d(480, feature_dim, kernel_size=1).to(device)

            if self.context:
                # decoder_sem decodes the (B,1280,64,64) HRNet map  -> x4  -> (B,133,256,256)
                self.decoder_sem = Decoder(feature_dim, 133, encoder='hrnet').to(device)
                # decoder_part decodes the (B,1280,16,16) SAM map   -> x16 -> (B,26,256,256)
                self.decoder_part = Decoder(feature_dim, 26, encoder='sam_vit').to(device)

            # Spatial cross-attention over a common 16x16 token grid (256 tokens),
            # instead of global-pooling each branch to a single token. This lets the SAM
            # and HRNet features interact per-location before aggregation, preserving the
            # spatial detail that pooling-to-(B,1,1280) would discard.
            self.cross_grid = 64        # lowered from 64 to make room for the fp32 backbone (unfreezing)
            self.cross_att = Spatial_Cross_Att(feature_dim, num_heads=8).to(device)
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

            # semantic branch: HRNet -> (B, 480, 64, 64) -> project to (B, 1280, 64, 64)
            sem_enc_out = self.encoder_sem(img)
            sem_enc_out = self.hrnet_to_sam(sem_enc_out)

            if self.context:
                sem_mask_pred = self.decoder_sem(sem_enc_out)      # (B, 133, 256, 256)
                part_mask_pred = self.decoder_part(part_enc_out)   # (B, 26, 256, 256)

            # bring both maps to a common g x g grid and flatten to (B, g*g, 1280) tokens.
            # adaptive_avg_pool is a no-op for the part branch (already 16x16) and pools
            # the HRNet sem branch 64x64 -> 16x16.
            g = self.cross_grid
            # sem_tok = F.adaptive_avg_pool2d(sem_enc_out, (g, g)).flatten(2).transpose(1, 2)   # (B, 256, 1280)
            # part_tok = F.adaptive_avg_pool2d(part_enc_out, (g, g)).flatten(2).transpose(1, 2) # (B, 256, 1280)
            sem_tok = F.interpolate(sem_enc_out, size=(g, g), mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)   # (B, 4096, 1280)
            part_tok = F.interpolate(part_enc_out, size=(g, g), mode='bilinear', align_corners=False).flatten(2).transpose(1, 2) # (B, 4096, 1280)

            # spatial cross-attention fuses the two modalities into image tokens
            att = self.cross_att(sem_tok, part_tok)                # (B, g*g, 1280): (B,256,1280) -> (B,4096,1280)
            # append the keypoint prompt tokens as extra memory for the contact head
            if prompt_tokens is not None:
                att = torch.cat([att, prompt_tokens], dim=1)       # (B, g*g + N, 1280): (B,4096+17=4113,1280)
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