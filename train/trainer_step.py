from utils.loss import sem_loss_function, class_loss_function, pixel_anchoring_function
from utils.keypoint_prompts import build_keypoint_prompts
import torch
import os
import time
from utils.distill import (build_teacher, ema_update, two_views, DINOHead, DINOLoss,
                           FeatureGrabber, attn_consistency_loss)


class TrainStepper():
    def __init__(self, deco_model, context, learning_rate, loss_weight, pal_loss_weight, device):
        self.device = device

        self.model = deco_model
        self.context = context

        if self.context:
            self.optimizer_sem = torch.optim.Adam(params=list(self.model.encoder_sem.parameters()) + list(self.model.decoder_sem.parameters()),
                                                lr=learning_rate, weight_decay=0.0001)
            self.optimizer_part = torch.optim.Adam(
                params=list(self.model.encoder_part.parameters()) + list(self.model.decoder_part.parameters()), lr=learning_rate,
                weight_decay=0.0001)
        # hrnet_to_sam (HRNet->SAM adapter) is a top-level submodule, NOT under encoder_sem,
        # so it must be listed explicitly or it never trains. Only feeds the contact path
        # (sem_tok -> cross_att -> classif), so it belongs here. Guarded for non-sam_hrnet models.
        hrnet_to_sam_params = list(self.model.hrnet_to_sam.parameters()) if hasattr(self.model, 'hrnet_to_sam') else []
        self.optimizer_contact = torch.optim.Adam(
            params=list(self.model.encoder_sem.parameters()) + list(self.model.encoder_part.parameters()) + list(
                self.model.cross_att.parameters()) + list(self.model.classif.parameters()) + hrnet_to_sam_params,
            lr=learning_rate, weight_decay=0.0001)

        # encoder_part param split: the (optionally) unfrozen backbone blocks are fine-tuned
        # by a dedicated low-LR optimizer; everything else (prompt encoder, projections,
        # frozen backbone) goes to the task optimizers. Falls back to all params for the plain
        # CNN/transformer encoders that don't expose the split.
        # ep = self.model.encoder_part
        # ep_task = list(ep.task_parameters()) if hasattr(ep, 'task_parameters') else list(ep.parameters())
        # bb_ft = list(ep.backbone_finetune_parameters()) if hasattr(ep, 'backbone_finetune_parameters') else []
        #
        # if self.context:
        #     self.optimizer_sem = torch.optim.Adam(params=list(self.model.encoder_sem.parameters()) + list(self.model.decoder_sem.parameters()),
        #                                         lr=learning_rate, weight_decay=0.0001)
        #     self.optimizer_part = torch.optim.Adam(
        #         params=ep_task + list(self.model.decoder_part.parameters()), lr=learning_rate,
        #         weight_decay=0.0001)
        # self.optimizer_contact = torch.optim.Adam(
        #     params=list(self.model.encoder_sem.parameters()) + ep_task + list(
        #         self.model.cross_att.parameters()) + list(self.model.classif.parameters()), lr=learning_rate, weight_decay=0.0001)
        #
        # Dedicated low-LR optimizer for the unfrozen backbone blocks (None if fully frozen).
        # self.backbone_lr_scale = 0.1
        # self.optimizer_backbone = (
        #     torch.optim.Adam(bb_ft, lr=learning_rate * self.backbone_lr_scale, weight_decay=0.0001)
        #     if bb_ft else None
        # )
        # if bb_ft:
        #     print(f"✓ backbone fine-tune optimizer: {len(bb_ft)} tensors @ lr={learning_rate * self.backbone_lr_scale:g}")

        if self.context: self.sem_loss = sem_loss_function().to(device)
        self.class_loss = class_loss_function().to(device)
        self.pixel_anchoring_loss_smplx = pixel_anchoring_function(model_type='smplx').to(device)
        self.pixel_anchoring_loss_smpl = pixel_anchoring_function(model_type='smpl').to(device)
        self.lr = learning_rate
        self.loss_weight = loss_weight
        self.pal_loss_weight = pal_loss_weight
        self.distill = False

    def enable_distill(self, out_dim=4096, dino_weight=1.0, out_weight=1.0, attn_weight=0.0,
                       ema_momentum=0.996, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        """Turn on DINO-style teacher-student self-distillation during optimize() (utils/distill.py).
        Builds an EMA teacher that SHARES the frozen SAM backbone, student+teacher DINO heads and
        the DINO loss, and adds the student head to optimizer_contact so it trains."""
        from common import constants
        self.distill = True
        self.ema_momentum = ema_momentum
        self.dino_weight = dino_weight
        self.out_weight = out_weight
        self.attn_weight = attn_weight
        self.img_mean = torch.tensor(constants.IMG_NORM_MEAN, device=self.device).view(1, 3, 1, 1)
        self.img_std = torch.tensor(constants.IMG_NORM_STD, device=self.device).view(1, 3, 1, 1)
        mp = getattr(self.model.classif, 'mem_proj', None)
        feat_dim = mp.in_features if isinstance(mp, torch.nn.Linear) else 1280
        self.teacher = build_teacher(self.model, share_prefix='encoder_part')
        self.student_dino_head = DINOHead(feat_dim, out_dim).to(self.device)
        self.teacher_dino_head = build_teacher(self.student_dino_head, share_prefix=None)
        self.dino_loss = DINOLoss(out_dim, teacher_temp, student_temp, center_momentum).to(self.device)
        self.student_feat = FeatureGrabber(self.model.classif)
        self.teacher_feat = FeatureGrabber(self.teacher.classif)
        if attn_weight > 0 and hasattr(self.model, 'cross_att'):    # attention-consistency regularizer
            self.model.cross_att.capture_attn = True
            self.teacher.cross_att.capture_attn = True
        self.optimizer_contact.add_param_group({'params': list(self.student_dino_head.parameters())})
        print(f'✓ Distillation ON: DINO out_dim={out_dim}, dino_w={dino_weight}, out_w={out_weight}, '
              f'attn_w={attn_weight}, ema={ema_momentum}, feat_dim={feat_dim}')

    def optimize(self, batch):
        self.model.train()

        img_paths = batch['img_path']
        img = batch['img'].to(self.device)

        img_scale_factor = batch['img_scale_factor'].to(self.device)

        pose = batch['pose'].to(self.device)
        betas = batch['betas'].to(self.device)
        transl = batch['transl'].to(self.device)
        has_smpl = batch['has_smpl'].to(self.device)
        is_smplx = batch['is_smplx'].to(self.device)

        cam_k = batch['cam_k'].to(self.device)

        gt_contact_labels_3d = batch['contact_label_3d'].to(self.device)
        has_contact_3d = batch['has_contact_3d'].to(self.device)

        if self.context:
            sem_mask_gt = batch['sem_mask'].to(self.device)
            part_mask_gt = batch['part_mask'].to(self.device)

        polygon_contact_2d = batch['polygon_contact_2d'].to(self.device)
        has_polygon_contact_2d = batch['has_polygon_contact_2d'].to(self.device)

        # Build 2D keypoint prompts (COCO-17 -> mhr70) for the SAM-3D-Body encoder.
        keypoints = None
        if 'keypoints_2d' in batch:
            keypoints = build_keypoint_prompts(
                batch['keypoints_2d'].to(self.device),
                batch['keypoint_conf'].to(self.device),
                img_scale_factor,
                has_keypoints=batch['has_keypoints'].to(self.device),
            )

        # Distillation: build two photometric views (geometry preserved -> labels stay valid).
        # The student trains on the strong view (supervised + distilled); the EMA teacher sees
        # the weak view and provides the distillation targets.
        student_img, teacher_img = img, None
        if getattr(self, 'distill', False):
            teacher_img, student_img = two_views(img, self.img_mean, self.img_std)

        # Forward pass
        if self.context:
            cont, sem_mask_pred, part_mask_pred = self.model(student_img, keypoints)
        else:
            cont = self.model(student_img)

        if self.context:
            loss_sem = self.sem_loss(sem_mask_gt, sem_mask_pred)
            loss_part = self.sem_loss(part_mask_gt, part_mask_pred)
        valid_contact_3d = has_contact_3d
        loss_cont = self.class_loss(gt_contact_labels_3d, cont, valid_contact_3d)
        valid_polygon_contact_2d = has_polygon_contact_2d

        if self.pal_loss_weight > 0 and (is_smplx == 0).sum() > 0:
            smpl_body_params = {'pose': pose[is_smplx == 0], 'betas': betas[is_smplx == 0],
                                'transl': transl[is_smplx == 0],
                                'has_smpl': has_smpl[is_smplx == 0]}
            loss_pix_anchoring_smpl, contact_2d_pred_rgb_smpl, _ = self.pixel_anchoring_loss_smpl(cont[is_smplx == 0],
                                                                                                  smpl_body_params,
                                                                                                  cam_k[is_smplx == 0],
                                                                                                  img_scale_factor[
                                                                                                      is_smplx == 0],
                                                                                                  polygon_contact_2d[
                                                                                                      is_smplx == 0],
                                                                                                  valid_polygon_contact_2d[
                                                                                                      is_smplx == 0])
            # weigh the smpl loss based on the number of smpl sample
            loss_pix_anchoring = loss_pix_anchoring_smpl * (is_smplx == 0).sum() / len(is_smplx)
            contact_2d_pred_rgb = contact_2d_pred_rgb_smpl
        else:
            loss_pix_anchoring = 0
            contact_2d_pred_rgb = torch.zeros_like(polygon_contact_2d)

        if self.context: loss = loss_sem + loss_part + self.loss_weight * loss_cont + self.pal_loss_weight * loss_pix_anchoring
        else: loss = self.loss_weight * loss_cont + self.pal_loss_weight * loss_pix_anchoring

        # Teacher-student distillation: feature DINO (on the pooled fused features) + output
        # consistency (on the contact probs). The teacher runs on the weak view under no_grad.
        loss_dino = loss_out = loss_attn = None
        if getattr(self, 'distill', False):
            student_feat = self.student_feat.feat                       # captured by the hook in forward
            with torch.no_grad():
                t_out = self.teacher(teacher_img, keypoints)
                teacher_cont = t_out[0] if isinstance(t_out, (tuple, list)) else t_out
                teacher_logits = self.teacher_dino_head(self.teacher_feat.feat)
            loss_dino = self.dino_loss(self.student_dino_head(student_feat), teacher_logits)
            loss_out = ((cont - teacher_cont) ** 2).mean()
            loss = loss + self.dino_weight * loss_dino + self.out_weight * loss_out
            # attention-consistency regularizer on the trainable fusion (utils/distill.py)
            if self.attn_weight > 0 and getattr(self.model.cross_att, 'capture_attn', False):
                loss_attn = attn_consistency_loss(self.model.cross_att.last_attn,
                                                  self.teacher.cross_att.last_attn)
                loss = loss + self.attn_weight * loss_attn

        if self.context:
            self.optimizer_sem.zero_grad()
            self.optimizer_part.zero_grad()
        self.optimizer_contact.zero_grad()
        # if self.optimizer_backbone is not None:
        #     self.optimizer_backbone.zero_grad()

        loss.backward()

        if self.context:
            self.optimizer_sem.step()
            self.optimizer_part.step()
        self.optimizer_contact.step()
        # if self.optimizer_backbone is not None:
        #     self.optimizer_backbone.step()

        # EMA-update the teacher (skip the shared frozen backbone) and the teacher DINO head.
        if getattr(self, 'distill', False):
            ema_update(self.teacher, self.model, self.ema_momentum, skip_prefix='encoder_part')
            ema_update(self.teacher_dino_head, self.student_dino_head, self.ema_momentum, skip_prefix=None)

        if self.context:
            losses = {'sem_loss': loss_sem,
                    'part_loss': loss_part,
                    'cont_loss': loss_cont,
                    'pal_loss': loss_pix_anchoring,
                    'total_loss': loss}
        else:
            losses = {'cont_loss': loss_cont,
                    'pal_loss': loss_pix_anchoring,
                    'total_loss': loss}

        if loss_dino is not None:                          # distillation diagnostics (for logging)
            losses['dino_loss'] = loss_dino
            losses['out_loss'] = loss_out
            if loss_attn is not None:                      # mean attention-consistency KL
                losses['attn_loss'] = loss_attn

        if self.context:
            output = {
                'img': img,
                'sem_mask_gt': sem_mask_gt,
                'sem_mask_pred': sem_mask_pred,
                'part_mask_gt': part_mask_gt,
                'part_mask_pred': part_mask_pred,
                'has_contact_2d': has_polygon_contact_2d,
                'contact_2d_gt': polygon_contact_2d,
                'contact_2d_pred_rgb': contact_2d_pred_rgb,
                'has_contact_3d': has_contact_3d,
                'contact_labels_3d_gt': gt_contact_labels_3d,
                'contact_labels_3d_pred': cont}
        else:
            output = {
                'img': img,
                'has_contact_2d': has_polygon_contact_2d,
                'contact_2d_gt': polygon_contact_2d,
                'contact_2d_pred_rgb': contact_2d_pred_rgb,
                'has_contact_3d': has_contact_3d,
                'contact_labels_3d_gt': gt_contact_labels_3d,
                'contact_labels_3d_pred': cont}   

        return losses, output

    @torch.no_grad()
    def evaluate(self, batch):
        self.model.eval()

        img_paths = batch['img_path']
        img = batch['img'].to(self.device)

        img_scale_factor = batch['img_scale_factor'].to(self.device)

        pose = batch['pose'].to(self.device)
        betas = batch['betas'].to(self.device)
        transl = batch['transl'].to(self.device)
        has_smpl = batch['has_smpl'].to(self.device)
        is_smplx = batch['is_smplx'].to(self.device)

        cam_k = batch['cam_k'].to(self.device)

        gt_contact_labels_3d = batch['contact_label_3d'].to(self.device)
        has_contact_3d = batch['has_contact_3d'].to(self.device)

        if self.context:
            sem_mask_gt = batch['sem_mask'].to(self.device)
            part_mask_gt = batch['part_mask'].to(self.device)

        polygon_contact_2d = batch['polygon_contact_2d'].to(self.device)
        has_polygon_contact_2d = batch['has_polygon_contact_2d'].to(self.device)

        # Build 2D keypoint prompts (COCO-17 -> mhr70) for the SAM-3D-Body encoder.
        keypoints = None
        if 'keypoints_2d' in batch:
            keypoints = build_keypoint_prompts(
                batch['keypoints_2d'].to(self.device),
                batch['keypoint_conf'].to(self.device),
                img_scale_factor,
                has_keypoints=batch['has_keypoints'].to(self.device),
            )

        # Forward pass
        initial_time = time.time()
        if self.context: cont, sem_mask_pred, part_mask_pred = self.model(img, keypoints)
        else: cont = self.model(img, keypoints)
        time_taken = time.time() - initial_time

        if self.context:
            loss_sem = self.sem_loss(sem_mask_gt, sem_mask_pred)
            loss_part = self.sem_loss(part_mask_gt, part_mask_pred)
        valid_contact_3d = has_contact_3d
        loss_cont = self.class_loss(gt_contact_labels_3d, cont, valid_contact_3d)
        valid_polygon_contact_2d = has_polygon_contact_2d

        if self.pal_loss_weight > 0 and (is_smplx == 0).sum() > 0: # PAL loss only on 2D contacts in HOT which only has SMPL
            smpl_body_params = {'pose': pose[is_smplx == 0], 'betas': betas[is_smplx == 0], 'transl': transl[is_smplx == 0],
                                'has_smpl': has_smpl[is_smplx == 0]}
            loss_pix_anchoring_smpl, contact_2d_pred_rgb_smpl, _ = self.pixel_anchoring_loss_smpl(cont[is_smplx == 0],
                                                                                                 smpl_body_params,
                                                                                                 cam_k[is_smplx == 0],
                                                                                                 img_scale_factor[
                                                                                                     is_smplx == 0],
                                                                                                 polygon_contact_2d[
                                                                                                     is_smplx == 0],
                                                                                                 valid_polygon_contact_2d[
                                                                                                     is_smplx == 0])
            # weight the smpl loss based on the number of smpl samples
            contact_2d_pred_rgb = contact_2d_pred_rgb_smpl
            loss_pix_anchoring = loss_pix_anchoring_smpl * (is_smplx == 0).sum() / len(is_smplx)
        else:
            loss_pix_anchoring = 0
            contact_2d_pred_rgb = torch.zeros_like(polygon_contact_2d)

        if self.context: loss = loss_sem + loss_part + self.loss_weight * loss_cont + self.pal_loss_weight * loss_pix_anchoring
        else: loss = self.loss_weight * loss_cont + self.pal_loss_weight * loss_pix_anchoring

        if self.context:
            losses = {'sem_loss': loss_sem,
                    'part_loss': loss_part,
                    'cont_loss': loss_cont,
                    'pal_loss': loss_pix_anchoring,
                    'total_loss': loss}
        else:
            losses = {'cont_loss': loss_cont,
                  'pal_loss': loss_pix_anchoring,
                  'total_loss': loss}            

        if self.context:
            output = {
                'img': img,
                'sem_mask_gt': sem_mask_gt,
                'sem_mask_pred': sem_mask_pred,
                'part_mask_gt': part_mask_gt,
                'part_mask_pred': part_mask_pred,
                'has_contact_2d': has_polygon_contact_2d,
                'contact_2d_gt': polygon_contact_2d,
                'contact_2d_pred_rgb': contact_2d_pred_rgb,
                'has_contact_3d': has_contact_3d,
                'contact_labels_3d_gt': gt_contact_labels_3d,
                'contact_labels_3d_pred': cont}
        else:
            output = {
                'img': img,
                'has_contact_2d': has_polygon_contact_2d,
                'contact_2d_gt': polygon_contact_2d,
                'contact_2d_pred_rgb': contact_2d_pred_rgb,
                'has_contact_3d': has_contact_3d,
                'contact_labels_3d_gt': gt_contact_labels_3d,
                'contact_labels_3d_pred': cont}        

        return losses, output, time_taken

    def save(self, ep, f1, model_path):
        # create model directory if it does not exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if self.context:
            torch.save({
                'epoch': ep,
                'deco': self.model.state_dict(),
                'f1': f1,
                'sem_optim': self.optimizer_sem.state_dict(),
                'part_optim': self.optimizer_part.state_dict(),
                'contact_optim': self.optimizer_contact.state_dict()
            },
                model_path)
        else:
            torch.save({
                'epoch': ep,
                'deco': self.model.state_dict(),
                'f1': f1,
                'sem_optim': self.optimizer_sem.state_dict(),
                'part_optim': self.optimizer_part.state_dict(),
                'contact_optim': self.optimizer_contact.state_dict()
            },
                model_path)    

    def load(self, model_path):
        print(f'~~~ Loading existing checkpoint from {model_path} ~~~')
        checkpoint = torch.load(model_path, weights_only=False)  # trusted local ckpt (has a numpy f1 scalar); torch>=2.6 defaults weights_only=True
        # strict=False so a pre-prompt checkpoint can warm-start the prompt-enabled model:
        # encoder_part.prompt_encoder.* keys are absent in old checkpoints and keep their
        # (pretrained, loaded at construction) weights. Surface any other mismatch.
        missing, unexpected = self.model.load_state_dict(checkpoint['deco'], strict=False)
        missing = [k for k in missing if not k.startswith('encoder_part.prompt_encoder')]
        if missing or unexpected:
            print(f'  [load] unexpected={unexpected[:6]}{"..." if len(unexpected)>6 else ""} '
                  f'non-prompt missing={missing[:6]}{"..." if len(missing)>6 else ""}')

        # Optimizer states may not match if the model's parameters changed (e.g. the
        # cross_att fusion was restructured: -norm_out, +mod_embed -> contact group size
        # differs, or backbone fine-tuning splits encoder_part across optimizers). Load
        # best-effort; on a structure mismatch that optimizer just starts fresh -- the model
        # weights are already loaded above, so only the Adam moments for the changed group reset.
        def _try_load_optim(optim, key):
            try:
                optim.load_state_dict(checkpoint[key])
            except (ValueError, KeyError) as e:
                print(f"  [load] skipped {key} (optimizer structure changed): {str(e)[:80]}")
        if self.context:
            _try_load_optim(self.optimizer_sem, 'sem_optim')
            _try_load_optim(self.optimizer_part, 'part_optim')
        _try_load_optim(self.optimizer_contact, 'contact_optim')

        epoch = checkpoint['epoch']
        f1 = checkpoint['f1']
        return epoch, f1

    def update_lr(self, factor=2):
        if factor:
            new_lr = self.lr / factor

        if self.context:
            self.optimizer_sem = torch.optim.Adam(params=list(self.model.encoder_sem.parameters()) + list(self.model.decoder_sem.parameters()),
                                                lr=new_lr, weight_decay=0.0001)
            self.optimizer_part = torch.optim.Adam(
                params=list(self.model.encoder_part.parameters()) + list(self.model.decoder_part.parameters()), lr=new_lr, weight_decay=0.0001)
        self.optimizer_contact = torch.optim.Adam(
            params=list(self.model.encoder_sem.parameters()) + list(self.model.encoder_part.parameters()) + list(
                self.model.cross_att.parameters()) + list(self.model.classif.parameters()), lr=new_lr, weight_decay=0.0001)

        # ep = self.model.encoder_part
        # ep_task = list(ep.task_parameters()) if hasattr(ep, 'task_parameters') else list(ep.parameters())
        # bb_ft = list(ep.backbone_finetune_parameters()) if hasattr(ep, 'backbone_finetune_parameters') else []
        #
        # if self.context:
        #     self.optimizer_sem = torch.optim.Adam(params=list(self.model.encoder_sem.parameters()) + list(self.model.decoder_sem.parameters()),
        #                                         lr=new_lr, weight_decay=0.0001)
        #     self.optimizer_part = torch.optim.Adam(
        #         params=ep_task + list(self.model.decoder_part.parameters()), lr=new_lr, weight_decay=0.0001)
        # self.optimizer_contact = torch.optim.Adam(
        #     params=list(self.model.encoder_sem.parameters()) + ep_task + list(
        #         self.model.cross_att.parameters()) + list(self.model.classif.parameters()), lr=new_lr, weight_decay=0.0001)
        # if bb_ft:
        #     self.optimizer_backbone = torch.optim.Adam(bb_ft, lr=new_lr * self.backbone_lr_scale, weight_decay=0.0001)

        print('update learning rate: %f -> %f' % (self.lr, new_lr))
        self.lr = new_lr