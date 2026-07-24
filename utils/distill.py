"""DINO-style teacher-student self-distillation (no labels) for DECO contact prediction.

Auxiliary SSL on top of the supervised contact/seg losses. The teacher is an EMA of the student.
Each step uses two photometric views of the same image (geometry preserved, so the supervised
contact/keypoint labels stay valid): the teacher sees a WEAK view, the student a MILD view. Two
label-free distillation losses act on the contact path
(DECO.cross_att -> fused tokens -> DECO.classif -> contact):

  - output consistency (on-task, primary): MSE between the student's and teacher's per-vertex
    contact probabilities -- directly regularizes the prediction.
  - feature DINO (the named method): a projection head maps the pooled fused tokens (the input to
    DECO.classif) to `out_dim` prototype logits; the student matches the teacher's centered +
    sharpened distribution (cross-entropy) -- DINO's representation-level self-distillation.

Lessons baked in (memory: deco-dino-distillation): the student view is kept MILD (no grayscale) so
the supervised loss isn't trained on distorted images; the teacher SHARES the student's frozen SAM
backbones (neither ViT is duplicated / deep-copied); and the trainer ramps the weights 0->1.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DINO projection head + loss
# ---------------------------------------------------------------------------
class DINOHead(nn.Module):
    """3-layer MLP -> L2-normalize -> prototype layer. No weight_norm (it makes the module
    un-deepcopyable, which would break the EMA-teacher copy); the bottleneck L2-norm is the
    key DINO normalization."""
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last(x)                            # (B, out_dim) prototype logits


class DINOLoss(nn.Module):
    """DINO cross-entropy: student log-softmax vs teacher (centered + sharpened) softmax.
    Centering + teacher<student temperature prevent collapse to a trivial constant."""
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer('center', torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out):
        s = F.log_softmax(student_out / self.student_temp, dim=-1)
        t = F.softmax((teacher_out - self.center) / self.teacher_temp, dim=-1).detach()
        loss = -(t * s).sum(dim=-1).mean()
        self._update_center(teacher_out)
        return loss

    @torch.no_grad()
    def _update_center(self, teacher_out):
        self.center.mul_(self.center_momentum).add_(
            teacher_out.mean(0, keepdim=True), alpha=1 - self.center_momentum)


# ---------------------------------------------------------------------------
# EMA teacher (shares frozen backbones -> never duplicates / deep-copies a ViT)
# ---------------------------------------------------------------------------
def _prefixes(prefix_or_prefixes):
    if prefix_or_prefixes is None:
        return ()
    if isinstance(prefix_or_prefixes, str):
        return (prefix_or_prefixes,)
    return tuple(prefix for prefix in prefix_or_prefixes if prefix)


def _module_parent(module, path):
    """Return the parent module and attribute name for a dotted module path."""
    parts = path.split('.')
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def build_teacher(student, share_prefix='encoder_part'):
    """Build an EMA teacher while sharing one or more frozen submodules.

    ``share_prefix`` accepts a dotted path or an iterable of paths.  Each path is
    temporarily replaced before ``deepcopy`` and restored afterwards, so large frozen
    backbones are not duplicated.  Nested paths let a wrapper share only its frozen
    backbone while retaining a separate, EMA-updated trainable adapter in the teacher.
    """
    shared_modules = []
    for path in _prefixes(share_prefix):
        try:
            parent, name = _module_parent(student, path)
            shared = getattr(parent, name, None)
        except AttributeError:
            shared = None
        if shared is None:
            continue
        setattr(parent, name, nn.Identity())
        shared_modules.append((path, shared))

    try:
        teacher = copy.deepcopy(student)
    finally:
        for path, shared in shared_modules:
            parent, name = _module_parent(student, path)
            setattr(parent, name, shared)

    for path, shared in shared_modules:
        parent, name = _module_parent(teacher, path)
        setattr(parent, name, shared)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


@torch.no_grad()
def ema_update(teacher, student, momentum, skip_prefix='encoder_part'):
    """EMA-update everything except one or more shared frozen module paths."""
    skip_prefixes = _prefixes(skip_prefix)

    def is_shared(name):
        return any(name == prefix or name.startswith(prefix + '.') for prefix in skip_prefixes)

    sp = dict(student.named_parameters())
    for name, tp in teacher.named_parameters():
        if is_shared(name):
            continue
        s = sp.get(name)
        if s is not None:
            tp.mul_(momentum).add_(s.detach(), alpha=1 - momentum)
    sb = dict(student.named_buffers())
    for name, tb in teacher.named_buffers():
        if is_shared(name):
            continue
        b = sb.get(name)
        if b is not None and b.shape == tb.shape:
            tb.copy_(b)


# ---------------------------------------------------------------------------
# pooled-fused-feature grabber (forward pre-hook on the contact head)
# ---------------------------------------------------------------------------
class FeatureGrabber:
    """Capture the pooled fused tokens that DECO.classif consumes (its forward input), via a
    forward-pre-hook -> a (B, C) global descriptor for the feature-DINO head."""
    def __init__(self, contact_head):
        self.feat = None
        self._h = contact_head.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        self.feat = args[0].mean(dim=1)                # (B, N, C) tokens -> (B, C)

    def remove(self):
        self._h.remove()


# ---------------------------------------------------------------------------
# Test-time adaptation: image/mesh interface supervision
# ---------------------------------------------------------------------------
def object_interface_support(object_mask, radius=8):
    """Return a binary band around a binary object-mask boundary.

    The object proposal is fixed label-free evidence from Task 6.  A contact
    vertex is plausible only when its image projection lies near this boundary,
    rather than somewhere in the rest of the person crop.  Max pooling gives a
    cheap GPU implementation which avoids a scipy/OpenCV dependency at test
    time.
    """
    if object_mask is None:
        return None
    if object_mask.ndim == 3:
        object_mask = object_mask.unsqueeze(1)
    if object_mask.ndim != 4 or object_mask.shape[1] != 1:
        raise ValueError(
            'object_mask must have shape (B, 1, H, W) or (B, H, W), got '
            f'{tuple(object_mask.shape)}'
        )
    mask = (object_mask > 0).to(dtype=torch.float32)
    # A one-pixel morphological gradient.  Padding makes image-edge proposals
    # valid interfaces too.
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    edge = (dilated - eroded > 0).to(mask.dtype)
    radius = max(int(radius), 0)
    if radius:
        edge = F.max_pool2d(edge, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return edge


def _sample_interface_support(contact_prob, projected_vertices, visible_vertices,
                              object_mask, interface_radius):
    """Sample a fixed SAM interface band at projected mesh vertices."""
    if object_mask is None or projected_vertices is None or visible_vertices is None:
        return None, None
    if contact_prob.ndim != 2 or projected_vertices.shape[:2] != contact_prob.shape:
        raise ValueError('contact_prob and projected_vertices must agree on (B, V)')
    object_mask = object_mask.to(device=contact_prob.device)
    projected_vertices = projected_vertices.to(device=contact_prob.device)
    visible_vertices = visible_vertices.to(device=contact_prob.device)
    support_map = object_interface_support(object_mask, interface_radius)
    B, _, H, W = support_map.shape
    if B != contact_prob.shape[0]:
        raise ValueError('object_mask batch size must match contact_prob')
    xy = projected_vertices.to(dtype=support_map.dtype)
    grid = torch.stack((
        2.0 * xy[..., 0] / max(W - 1, 1) - 1.0,
        2.0 * xy[..., 1] / max(H - 1, 1) - 1.0,
    ), dim=-1).unsqueeze(2)
    sampled_support = F.grid_sample(
        support_map, grid, mode='bilinear', padding_mode='zeros', align_corners=True,
    ).squeeze(1).squeeze(-1).clamp(0, 1)
    # An empty/fallback proposal is not evidence that the whole image is
    # non-contact.  Disable geometry pseudo-supervision for that sample.
    has_object = (object_mask.to(dtype=contact_prob.dtype).flatten(1).sum(1) > 0)
    visible = visible_vertices.to(dtype=contact_prob.dtype) * has_object[:, None].to(contact_prob.dtype)
    return sampled_support, visible


def geometry_interface_consistency_loss(
        contact_prob, projected_vertices, visible_vertices, object_mask,
        confidence_threshold=0.70, interface_radius=8, temperature=0.05):
    """Penalize confident contacts without projected object-interface evidence.

    Args:
        contact_prob: ``(B, V)`` sigmoid contact probabilities.
        projected_vertices: ``(B, V, 2)`` vertex image coordinates in pixels.
        visible_vertices: ``(B, V)`` bool/float validity mask.  Vertices behind
            the camera or outside the crop must already be masked out.
        object_mask: Task-6 nearby non-person SAM proposal, aligned to the
            input crop, shape ``(B, 1, H, W)``.

    The confidence gate is detached deliberately: uncertain vertices create no
    pseudo-label, while a confident false positive receives a direct gradient
    downwards.  This makes the task precision-oriented and avoids the usual
    confirmation loop of unconstrained entropy minimization.
    """
    if object_mask is None or projected_vertices is None or visible_vertices is None:
        return contact_prob.new_zeros(()), {'valid_vertices': 0.0, 'support': 0.0}
    sampled_support, visible = _sample_interface_support(
        contact_prob, projected_vertices, visible_vertices, object_mask, interface_radius,
    )
    # Smooth only at the threshold but detach the gate, so lowering a confident
    # unsupported prediction is always beneficial and no low-confidence vertex
    # is promoted into a pseudo-positive.
    confidence = torch.sigmoid(
        (contact_prob.detach() - confidence_threshold) / max(float(temperature), 1e-6)
    )
    weight = visible * confidence
    denom = weight.sum().clamp_min(1.0)
    loss = (weight * contact_prob * (1.0 - sampled_support)).sum() / denom
    stats = {
        'valid_vertices': float(visible.sum().detach().cpu()),
        'support': float((sampled_support * visible).sum().detach().cpu() / visible.sum().clamp_min(1).cpu()),
    }
    return loss, stats


def _neighbor_mean(values, edges):
    """Average ``values`` over undirected mesh edges without a sparse matrix."""
    if edges is None or edges.numel() == 0:
        return values.new_zeros(values.shape)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError('mesh_edges must have shape (2, E)')
    edges = edges.to(device=values.device, dtype=torch.long)
    src, dst = edges[0], edges[1]
    B, V = values.shape
    if src.numel() and (src.max() >= V or dst.max() >= V or src.min() < 0 or dst.min() < 0):
        raise ValueError('mesh_edges contains an invalid vertex index')
    total = values.new_zeros(B, V)
    degree = values.new_zeros(B, V)
    total.scatter_add_(1, src.expand(B, -1), values[:, dst])
    total.scatter_add_(1, dst.expand(B, -1), values[:, src])
    ones = values.new_ones(B, src.numel())
    degree.scatter_add_(1, src.expand(B, -1), ones)
    degree.scatter_add_(1, dst.expand(B, -1), ones)
    return total / degree.clamp_min(1.0)


def reliable_geometry_tta_losses(
        student_prob, teacher_prob, projected_vertices, visible_vertices, object_mask,
        mesh_edges=None, confidence_threshold=0.70, positive_threshold=0.45,
        interface_radius=8, stability_temperature=0.10):
    """Bidirectional, reliability-gated Task-7 self-supervision.

    ``negative`` suppresses a stable teacher contact that lacks object-interface
    evidence (precision).  ``positive`` preserves/promotes stable, interface-
    supported contacts, including small gaps adjacent to reliable contact
    vertices (recall).  ``topology`` only smooths within that reliable interface
    band, so it cannot spread contact over unrelated body regions.
    """
    zero = student_prob.new_zeros(())
    if teacher_prob is None or object_mask is None or projected_vertices is None or visible_vertices is None:
        return {'negative': zero, 'positive': zero, 'topology': zero}, {
            'valid_vertices': 0.0, 'support': 0.0, 'reliable_positive': 0.0,
        }
    support, visible = _sample_interface_support(
        student_prob, projected_vertices, visible_vertices, object_mask, interface_radius,
    )
    teacher = teacher_prob.detach().to(device=student_prob.device, dtype=student_prob.dtype)
    # Agreement is fixed pseudo-label reliability, not a direct route for the
    # student to lower the loss by changing its own confidence.
    agreement = torch.exp(
        -(student_prob.detach() - teacher).abs() / max(float(stability_temperature), 1e-6)
    )
    high_teacher = torch.sigmoid((teacher - confidence_threshold) / 0.05)
    negative_weight = visible * agreement * high_teacher
    negative = (
        negative_weight * student_prob * (1.0 - support)
    ).sum() / negative_weight.sum().clamp_min(1.0)

    # High-confidence supported vertices seed mesh propagation.  Moderate
    # teacher predictions at the interface are retained, while a neighbour seed
    # can repair a small false-negative hole without creating an isolated blob.
    seed = visible * support * high_teacher
    neighbour_seed = _neighbor_mean(seed, mesh_edges)
    teacher_candidate = torch.sigmoid((teacher - positive_threshold) / 0.08)
    candidate = torch.maximum(teacher_candidate, neighbour_seed).detach()
    candidate = candidate * (candidate >= 0.20).to(candidate.dtype)
    positive_weight = visible * support * agreement * candidate
    # A conservative soft target: no arbitrary hard positives from the object
    # boundary alone; evidence must come from the EMA teacher or mesh neighbours.
    positive_target = torch.maximum(teacher, 0.45 + 0.25 * candidate)
    positive_bce = F.binary_cross_entropy(
        student_prob.clamp(1e-6, 1.0 - 1e-6), positive_target, reduction='none',
    )
    positive = (positive_weight * positive_bce).sum() / positive_weight.sum().clamp_min(1.0)

    topology = zero
    if mesh_edges is not None and mesh_edges.numel() > 0:
        edges = mesh_edges.to(device=student_prob.device, dtype=torch.long)
        src, dst = edges[0], edges[1]
        edge_weight = (
            positive_weight[:, src] * positive_weight[:, dst]
        )
        topology = (
            edge_weight * (student_prob[:, src] - student_prob[:, dst]).abs()
        ).sum() / edge_weight.sum().clamp_min(1.0)
    stats = {
        'valid_vertices': float(visible.sum().detach().cpu()),
        'support': float((support * visible).sum().detach().cpu() / visible.sum().clamp_min(1).cpu()),
        'reliable_positive': float(positive_weight.sum().detach().cpu()),
    }
    return {'negative': negative, 'positive': positive, 'topology': topology}, stats


class TestTimeDINOAdapter:
    """Per-instance Task-7 adaptation with guaranteed checkpoint rollback.

    Only ``cross_att`` and ``classif`` are optimized.  Frozen SAM/DINO encoders
    are shared by the student and a lightweight EMA teacher: unlike training,
    no complete second DECO model or ViT is allocated.  Each call starts from
    the loaded checkpoint and restores it before returning, which prevents
    adaptation leakage between test images.
    """
    __test__ = False  # pytest: this is a utility, despite its Test* name.
    def __init__(self, model, device, *, steps=3, learning_rate=1e-5,
                 geometry_weight=1.0, out_weight=0.05, dino_weight=0.01,
                 dino_out_dim=256, ema_momentum=0.996,
                 teacher_temp=0.04, student_temp=0.1, center_momentum=0.9,
                 interface_radius=8, confidence_threshold=0.70,
                 positive_weight=0.10, topology_weight=0.02,
                 positive_threshold=0.45, stability_temperature=0.10):
        if not hasattr(model, 'cross_att') or not hasattr(model, 'classif'):
            raise ValueError('Task-7 TTA requires model.cross_att and model.classif')
        self.model = model
        self.device = torch.device(device)
        self.steps = max(int(steps), 0)
        self.geometry_weight = float(geometry_weight)
        self.out_weight = float(out_weight)
        self.dino_weight = float(dino_weight)
        self.ema_momentum = float(ema_momentum)
        self.interface_radius = int(interface_radius)
        self.confidence_threshold = float(confidence_threshold)
        self.positive_weight = float(positive_weight)
        self.topology_weight = float(topology_weight)
        self.positive_threshold = float(positive_threshold)
        self.stability_temperature = float(stability_temperature)

        self._requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
        for p in model.parameters():
            p.requires_grad_(False)
        self.adapt_modules = (model.cross_att, model.classif)
        for module in self.adapt_modules:
            for p in module.parameters():
                p.requires_grad_(True)

        # These copies contain only the fusion/contact path, not a second SAM
        # backbone.  They are swapped into the public model transiently for the
        # no-grad teacher pass.
        self.teacher_cross_att = copy.deepcopy(model.cross_att).to(self.device).eval()
        self.teacher_classif = copy.deepcopy(model.classif).to(self.device).eval()
        for p in self.teacher_cross_att.parameters():
            p.requires_grad_(False)
        for p in self.teacher_classif.parameters():
            p.requires_grad_(False)

        mem_proj = getattr(model.classif, 'mem_proj', None)
        feat_dim = mem_proj.in_features if isinstance(mem_proj, nn.Linear) else 1280
        self.student_dino_head = DINOHead(feat_dim, dino_out_dim).to(self.device)
        self.teacher_dino_head = copy.deepcopy(self.student_dino_head).to(self.device).eval()
        for p in self.teacher_dino_head.parameters():
            p.requires_grad_(False)
        self.dino_loss = DINOLoss(dino_out_dim, teacher_temp, student_temp, center_momentum).to(self.device)
        self.student_feat = FeatureGrabber(model.classif)
        self.teacher_feat = FeatureGrabber(self.teacher_classif)

        params = [p for module in self.adapt_modules for p in module.parameters()]
        params += list(self.student_dino_head.parameters())
        self.optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0.0)

        # Do not clone model encoders: they are never trainable nor run in train
        # mode here.  Saving only adapt-module state preserves the 3060-Ti memory
        # budget while still restoring every tensor that can change.
        self._initial_student = [copy.deepcopy(module.state_dict()) for module in self.adapt_modules]
        self._initial_teacher = [copy.deepcopy(self.teacher_cross_att.state_dict()),
                                 copy.deepcopy(self.teacher_classif.state_dict())]
        self._initial_student_dino = copy.deepcopy(self.student_dino_head.state_dict())
        self._initial_teacher_dino = copy.deepcopy(self.teacher_dino_head.state_dict())
        self._initial_dino_center = self.dino_loss.center.detach().clone()

    def _reset(self):
        for module, state in zip(self.adapt_modules, self._initial_student):
            module.load_state_dict(state)
        self.teacher_cross_att.load_state_dict(self._initial_teacher[0])
        self.teacher_classif.load_state_dict(self._initial_teacher[1])
        self.student_dino_head.load_state_dict(self._initial_student_dino)
        self.teacher_dino_head.load_state_dict(self._initial_teacher_dino)
        self.dino_loss.center.copy_(self._initial_dino_center)
        self.optimizer.state.clear()

    def _teacher_forward(self, image, keypoints, object_prompt):
        """Run the EMA fusion/head while sharing the student's frozen encoders."""
        student_cross, student_classif = self.model.cross_att, self.model.classif
        self.model.cross_att, self.model.classif = self.teacher_cross_att, self.teacher_classif
        try:
            output = self.model(image, keypoints=keypoints, object_prompt=object_prompt)
        finally:
            self.model.cross_att, self.model.classif = student_cross, student_classif
        return output[0] if isinstance(output, (tuple, list)) else output

    @staticmethod
    def _contact_output(output):
        return output[0] if isinstance(output, (tuple, list)) else output

    def adapt_and_predict(self, image, *, keypoints=None, object_prompt=None, geometry=None,
                          mean=None, std=None):
        """Adapt to one batch and return detached contacts plus scalar diagnostics.

        ``geometry`` may supply ``projected_vertices`` and ``visible_vertices``
        from the known SMPL/camera parameters, plus the label-free ``object_mask``.
        It is optional so the photometric teacher-student objective remains
        usable for deployments without fitted SMPL parameters.
        """
        from common import constants

        self._reset()
        old_training = self.model.training
        self.model.eval()
        self.teacher_cross_att.eval()
        self.teacher_classif.eval()
        self.student_dino_head.train()
        self.teacher_dino_head.eval()
        if mean is None:
            mean = torch.tensor(constants.IMG_NORM_MEAN, device=image.device).view(1, 3, 1, 1)
        if std is None:
            std = torch.tensor(constants.IMG_NORM_STD, device=image.device).view(1, 3, 1, 1)

        diagnostics = {
            'loss': 0.0, 'geometry_loss': 0.0, 'geometry_negative_loss': 0.0,
            'geometry_positive_loss': 0.0, 'geometry_topology_loss': 0.0,
            'out_loss': 0.0, 'dino_loss': 0.0, 'steps': self.steps,
        }
        try:
            for _ in range(self.steps):
                teacher_img, student_img = two_views(image, mean, std)
                student_output = self.model(student_img, keypoints=keypoints, object_prompt=object_prompt)
                student_contact = self._contact_output(student_output)
                with torch.no_grad():
                    teacher_contact = self._teacher_forward(teacher_img, keypoints, object_prompt)
                    teacher_feat = self.teacher_feat.feat
                    teacher_dino = self.teacher_dino_head(teacher_feat) if self.dino_weight else None

                out_loss = (student_contact - teacher_contact).square().mean()
                total = self.out_weight * out_loss
                dino_value = student_contact.new_zeros(())
                if self.dino_weight:
                    dino_value = self.dino_loss(self.student_dino_head(self.student_feat.feat), teacher_dino)
                    total = total + self.dino_weight * dino_value
                geometry_value = student_contact.new_zeros(())
                geometry_losses = {
                    'negative': geometry_value, 'positive': geometry_value,
                    'topology': geometry_value,
                }
                if geometry is not None and self.geometry_weight:
                    geometry_losses, _ = reliable_geometry_tta_losses(
                        student_contact,
                        teacher_contact,
                        geometry.get('projected_vertices'),
                        geometry.get('visible_vertices'),
                        geometry.get('object_mask'),
                        confidence_threshold=self.confidence_threshold,
                        positive_threshold=self.positive_threshold,
                        interface_radius=self.interface_radius,
                        stability_temperature=self.stability_temperature,
                        mesh_edges=geometry.get('mesh_edges'),
                    )
                    geometry_value = (
                        geometry_losses['negative']
                        + self.positive_weight * geometry_losses['positive']
                        + self.topology_weight * geometry_losses['topology']
                    )
                    total = total + self.geometry_weight * geometry_value

                if total.requires_grad:
                    self.optimizer.zero_grad(set_to_none=True)
                    total.backward()
                    self.optimizer.step()
                    ema_update(self.teacher_cross_att, self.model.cross_att, self.ema_momentum, skip_prefix=None)
                    ema_update(self.teacher_classif, self.model.classif, self.ema_momentum, skip_prefix=None)
                    ema_update(self.teacher_dino_head, self.student_dino_head, self.ema_momentum, skip_prefix=None)
                diagnostics.update({
                    'loss': float(total.detach().cpu()),
                    'geometry_loss': float(geometry_value.detach().cpu()),
                    'geometry_negative_loss': float(geometry_losses['negative'].detach().cpu()),
                    'geometry_positive_loss': float(geometry_losses['positive'].detach().cpu()),
                    'geometry_topology_loss': float(geometry_losses['topology'].detach().cpu()),
                    'out_loss': float(out_loss.detach().cpu()),
                    'dino_loss': float(dino_value.detach().cpu()),
                })

            with torch.no_grad():
                prediction = self._contact_output(
                    self.model(image, keypoints=keypoints, object_prompt=object_prompt)
                ).detach()
            return prediction, diagnostics
        finally:
            self._reset()
            self.model.train(old_training)

    def close(self):
        """Release hooks and restore the caller's original grad flags."""
        self._reset()
        self.student_feat.remove()
        self.teacher_feat.remove()
        for name, p in self.model.named_parameters():
            p.requires_grad_(self._requires_grad[name])


# ---------------------------------------------------------------------------
# two-view photometric augmentation (GPU, denorm -> jitter -> renorm)
# ---------------------------------------------------------------------------
_LUMA = (0.299, 0.587, 0.114)


def _jitter(x, strength):
    """Batched brightness/contrast/saturation jitter on [0,1] images (per-sample factors)."""
    B, dev = x.shape[0], x.device
    rand = lambda: 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * strength
    x = x * rand()                                     # brightness
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * rand() + m                           # contrast
    luma = (x * torch.tensor(_LUMA, device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
    return (luma + (x - luma) * rand()).clamp(0, 1)    # saturation


def two_views(img, mean, std, teacher_strength=0.1, student_strength=0.25, gray_p=0.0):
    """Return (teacher_view, student_view): weak vs mild photometric aug of the SAME image.
    Geometry is preserved (so contact/keypoint labels stay valid). Defaults are SOFT (no
    grayscale) so the supervised loss -- which runs on the student view -- isn't trained on
    distorted images. `mean`/`std` are (1,3,1,1) tensors matching how `img` was normalized."""
    raw = (img * std + mean).clamp(0, 1)
    t = _jitter(raw, teacher_strength)
    s = _jitter(raw, student_strength)
    if gray_p > 0:                                     # optional random grayscale on the student
        B, dev = raw.shape[0], raw.device
        gray = (s * torch.tensor(_LUMA, device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True).repeat(1, 3, 1, 1)
        use = (torch.rand(B, 1, 1, 1, device=dev) < gray_p).float()
        s = use * gray + (1 - use) * s
    return (t - mean) / std, (s - mean) / std
