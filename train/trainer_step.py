from utils.loss import sem_loss_function, class_loss_function, pixel_anchoring_function
from utils.keypoint_prompts import build_keypoint_prompts
import torch
import os
import time
from utils.distill import (
    build_teacher, ema_update, two_views, DINOHead, DINOLoss, FeatureGrabber,
    TestTimeDINOAdapter,
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
        """Enable Task-7 per-image adaptation after loading the checkpoint.

        The adapter snapshots only fusion/contact-head tensors, freezes all
        encoders, and rolls changes back after every inference batch.
        """
        if self.test_time_adapter is not None:
            self.test_time_adapter.close()
        self.test_time_adapter = TestTimeDINOAdapter(self.model, self.device, **kwargs)

    def _tta_smpl_mesh_edges(self, num_vertices):
        """Return cached undirected SMPL adjacency for reliable TTA patch filling."""
        if num_vertices != self.pixel_anchoring_loss_smpl.n_vertices:
            return None
        if not hasattr(self, '_cached_tta_smpl_edges'):
            faces = self.pixel_anchoring_loss_smpl.body_faces.to(
                device=self.device, dtype=torch.long,
            )
            edges = torch.cat((
                faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)],
            ), dim=0)
            self._cached_tta_smpl_edges = torch.unique(
                torch.sort(edges, dim=1).values, dim=0,
            ).transpose(0, 1).contiguous()
        return self._cached_tta_smpl_edges

    @torch.no_grad()
    def _tta_geometry(self, pose, betas, transl, has_smpl, is_smplx, cam_k,
                      img_scale_factor, object_prompt, has_object_mask, num_vertices):
        """Build Task-7 mesh projections for SMPL samples in the current batch.

        Camera/SMPL fitting is supplied by the dataset.  We deliberately use
        only the binary SAM object proposal as supervision, never the contact
        annotation or its 2-D polygon.  SMPL-X samples are skipped because the
        contact head is indexed on SMPL's 6,890 vertices; mapping their
        visibility back through the sparse SMPL-to-SMPL-X conversion would make
        a noisy pseudo-target.  Their photometric TTA losses still apply.
        """
        if object_prompt is None:
            return None
        B = pose.shape[0]
        mask = object_prompt
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4 or mask.shape[0] != B:
            return None
        projected = torch.zeros(B, num_vertices, 2, device=self.device, dtype=pose.dtype)
        visible = torch.zeros(B, num_vertices, device=self.device, dtype=pose.dtype)
        smpl_indices = ((is_smplx == 0) & (has_smpl > 0)).nonzero(as_tuple=False).flatten()
        if smpl_indices.numel() == 0:
            return {
                'projected_vertices': projected,
                'visible_vertices': visible,
                'object_mask': mask,
                'mesh_edges': self._tta_smpl_mesh_edges(num_vertices),
            }

        params = {
            'pose': pose[smpl_indices], 'betas': betas[smpl_indices],
            'transl': transl[smpl_indices], 'has_smpl': has_smpl[smpl_indices],
        }
        vertices, _ = self.pixel_anchoring_loss_smpl.get_posed_mesh(params)
        V = min(vertices.shape[1], num_vertices)
        z = vertices[..., 2].clamp_min(1e-6)
        # Dataset preprocessing resizes every person crop to 256x256 and passes
        # the corresponding x/y scale factors.  This mirrors pixel anchoring's
        # camera scaling convention.
        sx, sy = img_scale_factor[smpl_indices, 0], img_scale_factor[smpl_indices, 1]
        fx = cam_k[smpl_indices, 0, 0] * sx
        fy = cam_k[smpl_indices, 1, 1] * sy
        cx = cam_k[smpl_indices, 0, 2] * sx
        cy = cam_k[smpl_indices, 1, 2] * sy
        uv = torch.stack((
            vertices[..., 0] / z * fx[:, None] + cx[:, None],
            vertices[..., 1] / z * fy[:, None] + cy[:, None],
        ), dim=-1).to(dtype=projected.dtype)
        H, W = mask.shape[-2:]
        in_frame = (
            (vertices[..., 2] > 1e-6)
            & (uv[..., 0] >= 0) & (uv[..., 0] <= W - 1)
            & (uv[..., 1] >= 0) & (uv[..., 1] <= H - 1)
        )
        # Lightweight vertex z-buffer: retain only vertices at the nearest
        # depth for their projected pixel.  It removes most back-side/self-
        # occluded vertices without invoking a second full mesh renderer during
        # every TTA step.  (The objective itself remains differentiable only
        # with respect to contact probabilities, not this fixed geometry.)
        pixel_x = uv[..., 0].round().long().clamp(0, W - 1)
        pixel_y = uv[..., 1].round().long().clamp(0, H - 1)
        pixel_index = pixel_y * W + pixel_x
        inf = torch.full_like(vertices[..., 2], float('inf'))
        depth_values = torch.where(in_frame, vertices[..., 2], inf)
        nearest_depth = torch.full(
            (vertices.shape[0], H * W), float('inf'),
            device=vertices.device, dtype=vertices.dtype,
        )
        nearest_depth.scatter_reduce_(
            1, pixel_index, depth_values, reduce='amin', include_self=True,
        )
        frontmost = vertices[..., 2] <= nearest_depth.gather(1, pixel_index) + 1e-4
        in_frame = in_frame & frontmost
        projected[smpl_indices, :V] = uv[:, :V]
        visible[smpl_indices, :V] = in_frame[:, :V].to(visible.dtype)
        if has_object_mask is not None:
            visible = visible * has_object_mask.to(self.device, visible.dtype).view(B, 1)
        return {
            'projected_vertices': projected,
            'visible_vertices': visible,
            'object_mask': mask,
            'mesh_edges': self._tta_smpl_mesh_edges(num_vertices),
        }

    def enable_distill(self, out_dim=4096, dino_weight=0.0, out_weight=1.0, ema_momentum=0.996,
                       teacher_temp=0.04, student_temp=0.1, center_momentum=0.9, ramp_steps=2000):
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
        if getattr(self, 'distill', False):
            teacher_img, student_img = two_views(img, self.img_mean, self.img_std)

        # Forward pass
        if self.context:
            cont, sem_mask_pred, part_mask_pred = self.model(
                student_img, keypoints=keypoints, object_prompt=object_prompt
            )
        else:
            cont = self.model(student_img, keypoints=keypoints, object_prompt=object_prompt)

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

        # DINO teacher-student distillation (label-free), ramped up from 0 so the early lagging
        # teacher doesn't drag the student. Teacher runs the WEAK view under no_grad.
        loss_out = loss_dino = None
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
            loss_out = ((cont - teacher_cont) ** 2).mean()         # on-task: output (contact-prob) consistency
            loss = loss + ramp * self.out_weight * loss_out
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

        if loss_out is not None:                       # DINO distillation diagnostics (for logging)
            losses['out_loss'] = loss_out
            if loss_dino is not None:
                losses['dino_loss'] = loss_dino

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
        if self.test_time_adapter is not None:
            # ``evaluate`` is intentionally no-grad for ordinary validation;
            # Task-7 opens a narrow autograd scope only around cross_att/classif.
            # The geometry dictionary is derived from fitted SMPL + camera and
            # the label-free Task-6 SAM proposal, never contact annotations.
            geometry = self._tta_geometry(
                pose, betas, transl, has_smpl, is_smplx, cam_k,
                img_scale_factor, object_prompt,
                batch.get('has_object_mask'),
                getattr(self.model.classif, 'num_vertices', 6890),
            )
            with torch.enable_grad():
                cont, tta_stats = self.test_time_adapter.adapt_and_predict(
                    img, keypoints=keypoints, object_prompt=object_prompt,
                    geometry=geometry,
                )
            # TTA only changes the contact prediction.  Context outputs are
            # evaluated once from the restored checkpoint and remain comparable
            # to the non-TTA segmentation protocol.
            if self.context:
                _, sem_mask_pred, part_mask_pred = self.model(
                    img, keypoints=keypoints, object_prompt=object_prompt
                )
        elif self.context:
            cont, sem_mask_pred, part_mask_pred = self.model(
                img, keypoints=keypoints, object_prompt=object_prompt
            )
        else:
            cont = self.model(img, keypoints=keypoints, object_prompt=object_prompt)
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
