from utils.loss import sem_loss_function, class_loss_function, pixel_anchoring_function
from utils.keypoint_prompts import build_keypoint_prompts
import torch
import os
import time
from utils.distill import (
    build_teacher, ema_update, two_views, DINOHead, DINOLoss, FeatureGrabber,
    TestTimeScalingAdapter, training_zoom_view, zoom_in_view,
)


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
        hrnet_to_sam = getattr(self.model, 'hrnet_to_sam', None)
        hrnet_to_sam_params = list(hrnet_to_sam.parameters()) if hrnet_to_sam is not None else []
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
        self.test_time_adapter = None

    def enable_test_time_adaptation(self, **kwargs):
        """Enable Task-7 per-image zoom-and-refine adaptation."""
        if self.test_time_adapter is not None:
            self.test_time_adapter.close()
        self.test_time_adapter = TestTimeScalingAdapter(self.model, self.device, **kwargs)

    def enable_distill(self, out_dim=4096, dino_weight=0.0, out_weight=1.0, ema_momentum=0.996,
                       teacher_temp=0.04, student_temp=0.1, center_momentum=0.9, ramp_steps=2000,
                       scale_aug_enabled=False, scale_aug_probability=0.5,
                       scale_aug_min=1.02, scale_aug_max=1.15,
                       scale_consistency_weight=0.25,
                       scale_aug_min_keypoint_retention=0.8,
                       scale_aug_min_object_retention=0.8,
                       scale_aug_focus_prompts=True):
        """DINO teacher-student self-distillation (no labels) on the contact path (utils/distill.py).

        Builds an EMA teacher that SHARES frozen SAM backbones, a student+teacher DINO head on
        the pooled fused tokens (input to classif), and the DINO loss; adds the student head to
        optimizer_contact so it trains. Per-step losses (ramped 0->1 over ramp_steps batches):
        output-consistency MSE on contact probs (on-task, out_weight) + feature-DINO CE (the named
        DINO method, dino_weight). The teacher targets the WEAK view; the student the MILD view."""
        from common import constants
        self.distill = True
        self.ema_momentum = ema_momentum
        self.dino_weight = dino_weight
        self.out_weight = out_weight
        self.distill_ramp_steps = ramp_steps
        self._distill_step = 0
        self.scale_aug_enabled = bool(scale_aug_enabled)
        self.scale_aug_probability = float(scale_aug_probability)
        self.scale_aug_min = float(scale_aug_min)
        self.scale_aug_max = float(scale_aug_max)
        self.scale_consistency_weight = float(scale_consistency_weight)
        self.scale_aug_min_keypoint_retention = float(scale_aug_min_keypoint_retention)
        self.scale_aug_min_object_retention = float(scale_aug_min_object_retention)
        self.scale_aug_focus_prompts = bool(scale_aug_focus_prompts)
        self.img_mean = torch.tensor(constants.IMG_NORM_MEAN, device=self.device).view(1, 3, 1, 1)
        self.img_std = torch.tensor(constants.IMG_NORM_STD, device=self.device).view(1, 3, 1, 1)
        mp = getattr(self.model.classif, 'mem_proj', None)
        feat_dim = mp.in_features if isinstance(mp, torch.nn.Linear) else 1280
        # The SAM-3D-Objects semantic wrapper has a frozen DINO backbone but a
        # trainable 1024->1280 adapter.  Share only its frozen backbone with the
        # teacher; the adapter remains a separate EMA copy.  This avoids duplicating
        # the large ViT on a 3060 Ti while preserving teacher/student EMA behavior.
        self.teacher_shared_prefixes = ('encoder_part',)
        sem_encoder = getattr(self.model, 'encoder_sem', None)
        if getattr(sem_encoder, 'freeze_backbone', False) and hasattr(sem_encoder, 'backbone'):
            self.teacher_shared_prefixes += ('encoder_sem.backbone',)
            if hasattr(sem_encoder, 'mask_backbone'):
                # The semantic encoder has a second frozen DINO for the object-mask
                # alpha channel; share it with the EMA teacher as well.
                self.teacher_shared_prefixes += ('encoder_sem.mask_backbone',)
        self.teacher = build_teacher(self.model, share_prefix=self.teacher_shared_prefixes)
        self.student_dino_head = DINOHead(feat_dim, out_dim).to(self.device)
        self.teacher_dino_head = build_teacher(self.student_dino_head, share_prefix=None)
        self.dino_loss = DINOLoss(out_dim, teacher_temp, student_temp, center_momentum).to(self.device)
        self.student_feat = FeatureGrabber(self.model.classif)
        self.teacher_feat = FeatureGrabber(self.teacher.classif)
        self.optimizer_contact.add_param_group({'params': list(self.student_dino_head.parameters())})
        print(f'✓ DINO distillation ON: out_dim={out_dim}, dino_w={dino_weight}, out_w={out_weight}, '
              f'ema={ema_momentum}, ramp={ramp_steps}, feat_dim={feat_dim}')
        if self.scale_aug_enabled:
            print(f'✓ Task-8 scale augmentation ON: p={self.scale_aug_probability:g}, '
                  f'scales=[{self.scale_aug_min:g}, {self.scale_aug_max:g}], '
                  f'consistency_w={self.scale_consistency_weight:g}, '
                  f'focus_prompts={self.scale_aug_focus_prompts}')

    def optimize(self, batch):
        self.model.train()

        img_paths = batch['img_path']
        img = batch['img'].to(self.device)
        # BaseDataset's human-removed union of Task-6 keypoint-circle SAM masks
        # is supplied to the SAM-3D-Objects semantic branch alongside the RGB image.
        object_prompt = batch.get('object_prompt', batch.get('object_mask'))
        if object_prompt is not None:
            object_prompt = object_prompt.to(self.device)
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

        # DINO distillation: two photometric views (geometry preserved -> labels valid). The
        # student trains on the MILD view (supervised + distilled); the EMA teacher sees the
        # WEAK view and provides the label-free targets.
        student_img, teacher_img = img, None
        student_keypoints, student_object_prompt = keypoints, object_prompt
        scale_aug = None
        if getattr(self, 'distill', False):
            teacher_img, student_img = two_views(img, self.img_mean, self.img_std)
            if self.scale_aug_enabled:
                scale_aug = training_zoom_view(
                    student_img,
                    keypoints=keypoints,
                    object_prompt=object_prompt,
                    min_scale=self.scale_aug_min,
                    max_scale=self.scale_aug_max,
                    probability=self.scale_aug_probability,
                    min_keypoint_retention=self.scale_aug_min_keypoint_retention,
                    min_object_retention=self.scale_aug_min_object_retention,
                    focus_prompts=self.scale_aug_focus_prompts,
                )
                student_img = scale_aug['image']
                student_keypoints = scale_aug['keypoints']
                student_object_prompt = scale_aug['object_prompt']

                # Context targets are image-plane labels, so transform them with
                # the same accepted zoom.  Contact labels are mesh labels and
                # deliberately remain unchanged.
                if self.context:
                    applied = scale_aug['applied'][:, None, None, None]
                    sem_zoom = zoom_in_view(
                        sem_mask_gt, scale_aug['scales'], mode='nearest', center=scale_aug['centers'],
                    )
                    part_zoom = zoom_in_view(
                        part_mask_gt, scale_aug['scales'], mode='nearest', center=scale_aug['centers'],
                    )
                    sem_mask_gt = torch.where(applied, sem_zoom, sem_mask_gt)
                    part_mask_gt = torch.where(applied, part_zoom, part_mask_gt)

        # Forward pass
        if self.context:
            cont, sem_mask_pred, part_mask_pred = self.model(
                student_img, keypoints=student_keypoints, object_prompt=student_object_prompt
            )
        else:
            cont = self.model(student_img, keypoints=student_keypoints, object_prompt=student_object_prompt)

        if self.context:
            loss_sem = self.sem_loss(sem_mask_gt, sem_mask_pred)
            loss_part = self.sem_loss(part_mask_gt, part_mask_pred)
        valid_contact_3d = has_contact_3d
        loss_cont = self.class_loss(gt_contact_labels_3d, cont, valid_contact_3d)
        valid_polygon_contact_2d = has_polygon_contact_2d

        # Pixel anchoring is defined in original crop coordinates.  Exclude
        # only accepted zoom samples; they still receive 3D contact labels.
        pal_samples = is_smplx == 0
        if scale_aug is not None:
            pal_samples = pal_samples & ~scale_aug['applied']
        if self.pal_loss_weight > 0 and pal_samples.sum() > 0:
            smpl_body_params = {'pose': pose[pal_samples], 'betas': betas[pal_samples],
                                'transl': transl[pal_samples],
                                'has_smpl': has_smpl[pal_samples]}
            loss_pix_anchoring_smpl, contact_2d_pred_rgb_smpl, _ = self.pixel_anchoring_loss_smpl(cont[pal_samples],
                                                                                                  smpl_body_params,
                                                                                                  cam_k[pal_samples],
                                                                                                  img_scale_factor[pal_samples],
                                                                                                  polygon_contact_2d[pal_samples],
                                                                                                  valid_polygon_contact_2d[pal_samples])
            # weigh the smpl loss based on the number of smpl sample
            loss_pix_anchoring = loss_pix_anchoring_smpl * pal_samples.sum() / len(is_smplx)
            contact_2d_pred_rgb = contact_2d_pred_rgb_smpl
        else:
            loss_pix_anchoring = 0
            contact_2d_pred_rgb = torch.zeros_like(polygon_contact_2d)

        if self.context: loss = loss_sem + loss_part + self.loss_weight * loss_cont + self.pal_loss_weight * loss_pix_anchoring
        else: loss = self.loss_weight * loss_cont + self.pal_loss_weight * loss_pix_anchoring

        # DINO teacher-student distillation (label-free), ramped up from 0 so the early lagging
        # teacher doesn't drag the student. Teacher runs the WEAK view under no_grad.
        loss_out = loss_dino = loss_scale = None
        if getattr(self, 'distill', False):
            ramp = min(1.0, self._distill_step / max(self.distill_ramp_steps, 1))
            self._distill_step += 1
            student_feat = self.student_feat.feat                  # pooled fused tokens (captured by hook)
            with torch.no_grad():
                t_out = self.teacher(
                    teacher_img, keypoints=keypoints, object_prompt=object_prompt
                )
                teacher_cont = t_out[0] if isinstance(t_out, (tuple, list)) else t_out
                teacher_logits = self.teacher_dino_head(self.teacher_feat.feat) if self.dino_weight > 0 else None
            per_sample_mse = (cont - teacher_cont).square().mean(dim=1)
            if scale_aug is None:
                loss_out = per_sample_mse.mean()                   # photometric output consistency
                loss = loss + ramp * self.out_weight * loss_out
            else:
                # Keep the original DINO output objective on non-zoom samples,
                # and apply a separately modest weight to accepted zooms.
                unzoomed = ~scale_aug['applied']
                if unzoomed.any():
                    loss_out = per_sample_mse[unzoomed].mean()
                    loss = loss + ramp * self.out_weight * loss_out
            if scale_aug is not None and scale_aug['applied'].any():
                applied = scale_aug['applied'].to(per_sample_mse.dtype)
                loss_scale = (per_sample_mse * applied).sum() / applied.sum()
                loss = loss + ramp * self.scale_consistency_weight * loss_scale
            if self.dino_weight > 0:                               # the named DINO method: feature self-distillation
                loss_dino = self.dino_loss(self.student_dino_head(student_feat), teacher_logits)
                loss = loss + ramp * self.dino_weight * loss_dino

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
            ema_update(
                self.teacher, self.model, self.ema_momentum,
                skip_prefix=self.teacher_shared_prefixes,
            )
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

        if getattr(self, 'distill', False):            # DINO / Task-8 diagnostics
            if loss_out is not None:
                losses['out_loss'] = loss_out
            if loss_dino is not None:
                losses['dino_loss'] = loss_dino
            if loss_scale is not None:
                losses['scale_consistency_loss'] = loss_scale
            if scale_aug is not None:
                losses['scale_aug_fraction'] = scale_aug['applied'].float().mean()
                losses['scale_aug_mean'] = scale_aug['scales'].mean()

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
        object_prompt = batch.get('object_prompt', batch.get('object_mask'))
        if object_prompt is not None:
            object_prompt = object_prompt.to(self.device)
        highres_img = batch.get('img_highres')
        if highres_img is not None:
            highres_img = highres_img.to(self.device)
        highres_object_prompt = batch.get('object_prompt_highres')
        if highres_object_prompt is not None:
            highres_object_prompt = highres_object_prompt.to(self.device)
        has_object_prompt = batch.get('has_object_mask')
        if has_object_prompt is not None:
            has_object_prompt = has_object_prompt.to(self.device)

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
        # CUDA kernels are asynchronous; synchronize at both boundaries so the
        # reported TTA latency reflects actual GPU work rather than queue time.
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        initial_time = time.perf_counter()
        if self.test_time_adapter is not None:
            # ``evaluate`` is intentionally no-grad for ordinary validation;
            # Task-7 opens a narrow autograd scope only around cross_att/classif
            # and uses centred zooms of this image as its label-free task.
            with torch.enable_grad():
                cont, tta_stats = self.test_time_adapter.adapt_and_predict(
                    img, keypoints=keypoints, object_prompt=object_prompt,
                    highres_image=highres_img,
                    highres_object_prompt=highres_object_prompt,
                    has_object_prompt=has_object_prompt,
                )
            # TTA only changes the contact prediction.  Context outputs are
            # evaluated once from the restored checkpoint and remain comparable
            # to the non-TTA segmentation protocol.
            if self.context:
                # TTA requires gradients only inside the adapter.  The
                # unchanged segmentation heads must stay inference-only.
                with torch.no_grad():
                    _, sem_mask_pred, part_mask_pred = self.model(
                        img, keypoints=keypoints, object_prompt=object_prompt
                    )
        elif self.context:
            cont, sem_mask_pred, part_mask_pred = self.model(
                img, keypoints=keypoints, object_prompt=object_prompt
            )
        else:
            cont = self.model(img, keypoints=keypoints, object_prompt=object_prompt)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        time_taken = time.perf_counter() - initial_time

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
