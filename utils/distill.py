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
# Test-time adaptation: scale-consistency self-supervision
# ---------------------------------------------------------------------------
def _zoom_scales(scale, batch_size, *, device, dtype):
    """Normalize a scalar or per-sample zoom tensor to shape ``(B,)``."""
    scales = torch.as_tensor(scale, device=device, dtype=dtype)
    if scales.ndim == 0:
        scales = scales.expand(batch_size)
    if scales.ndim != 1 or scales.numel() != batch_size:
        raise ValueError(f'zoom scale must be scalar or ({batch_size},), got {tuple(scales.shape)}')
    if torch.any(scales < 1.0):
        raise ValueError('zoom scale must be >= 1.0')
    return scales


def _zoom_centers(center, scales):
    """Validate crop centres and keep each scaled crop within the image."""
    batch_size = scales.numel()
    if center is None:
        centers = scales.new_full((batch_size, 2), 0.5)
    else:
        centers = torch.as_tensor(center, device=scales.device, dtype=scales.dtype)
        if centers.shape != (batch_size, 2):
            raise ValueError(f'zoom centre must have shape ({batch_size}, 2), got {tuple(centers.shape)}')
    half_window = 0.5 / scales[:, None]
    return centers.clamp(min=half_window, max=1.0 - half_window)


def prompt_focus_centers(keypoints=None, object_prompt=None, *, batch_size=None,
                         device=None, dtype=None):
    """Return automatic crop centres from body and nearby-object evidence.

    Valid keypoints contribute the centre of their bounding box; a non-empty
    object prompt contributes the centre of its foreground bounding box.  When
    both are present, their midpoint keeps the person and possible contact
    object in the same mild crop.  Missing evidence falls back to image centre.
    """
    if keypoints is None and object_prompt is None:
        if batch_size is None or device is None or dtype is None:
            raise ValueError('batch_size, device, and dtype are required without prompts')
        return torch.full((batch_size, 2), 0.5, device=device, dtype=dtype)
    source = keypoints if keypoints is not None else object_prompt
    if batch_size is None:
        batch_size = source.shape[0]
    if device is None:
        device = source.device
    if dtype is None:
        dtype = torch.float32

    body_centers = torch.zeros(batch_size, 2, device=device, dtype=dtype)
    has_body = torch.zeros(batch_size, device=device, dtype=torch.bool)
    if keypoints is not None:
        if keypoints.ndim != 3 or keypoints.shape[:1] != (batch_size,) or keypoints.shape[-1] != 3:
            raise ValueError('keypoints must have shape (B, N, 3)')
        valid = keypoints[..., 2] >= 0
        for index in range(batch_size):
            points = keypoints[index, valid[index], :2]
            if points.numel():
                body_centers[index] = (points.amin(dim=0) + points.amax(dim=0)) * 0.5
                has_body[index] = True

    object_centers = torch.zeros(batch_size, 2, device=device, dtype=dtype)
    has_object = torch.zeros(batch_size, device=device, dtype=torch.bool)
    if object_prompt is not None:
        prompt = object_prompt.unsqueeze(1) if object_prompt.ndim == 3 else object_prompt
        if prompt.ndim != 4 or prompt.shape[0] != batch_size:
            raise ValueError('object_prompt must have shape (B, H, W) or (B, C, H, W)')
        mask = prompt.amax(dim=1) > 0
        height, width = mask.shape[-2:]
        for index in range(batch_size):
            ys, xs = torch.where(mask[index])
            if xs.numel():
                object_centers[index, 0] = (xs.min() + xs.max() + 1).to(dtype) / (2 * width)
                object_centers[index, 1] = (ys.min() + ys.max() + 1).to(dtype) / (2 * height)
                has_object[index] = True

    centers = torch.full((batch_size, 2), 0.5, device=device, dtype=dtype)
    centers[has_body] = body_centers[has_body]
    centers[has_object & ~has_body] = object_centers[has_object & ~has_body]
    both = has_body & has_object
    centers[both] = (body_centers[both] + object_centers[both]) * 0.5
    return centers.clamp(0.0, 1.0)


def zoom_in_view(image, scale, *, mode='bilinear', center=None):
    """Crop then resize a normalized image without changing its shape.

    ``scale > 1`` magnifies the original crop: output coordinates sample a
    smaller region of the source image.  The default region is centred; an
    optional normalized ``center`` selects a prompt-focused crop.  Keeping the
    tensor resolution fixed lets the frozen SAM encoders consume each scale
    exactly as at test time.  ``scale == 1`` is deliberately exact, which makes
    the helper useful in tests and avoids interpolation drift in the original
    branch.
    """
    if image.ndim != 4:
        raise ValueError(f'image must be (B, C, H, W), got {tuple(image.shape)}')
    B = image.shape[0]
    scales = _zoom_scales(scale, B, device=image.device, dtype=image.dtype)
    if torch.all(scales == 1.0):
        return image
    centers = _zoom_centers(center, scales)
    theta = image.new_zeros(B, 2, 3)
    theta[:, 0, 0] = 1.0 / scales
    theta[:, 1, 1] = 1.0 / scales
    theta[:, :, 2] = 2.0 * centers - 1.0
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    kwargs = {'mode': mode, 'padding_mode': 'border', 'align_corners': False}
    return F.grid_sample(image, grid, **kwargs)


def zoom_keypoints(keypoints, scale, *, center=None):
    """Transform normalized SAM prompt points into a zoom view."""
    if keypoints is None:
        return keypoints
    if keypoints.ndim != 3 or keypoints.shape[-1] != 3:
        raise ValueError('keypoints must have shape (B, N, 3)')
    scales = _zoom_scales(scale, keypoints.shape[0], device=keypoints.device, dtype=keypoints.dtype)
    if torch.all(scales == 1.0):
        return keypoints
    centers = _zoom_centers(center, scales)
    transformed = keypoints.clone()
    transformed[..., :2] = 0.5 + (transformed[..., :2] - centers[:, None, :]) * scales[:, None, None]
    outside = ((transformed[..., :2] < 0) | (transformed[..., :2] > 1)).any(dim=-1)
    transformed[..., :2].clamp_(0, 1)
    transformed[..., 2][outside] = -2
    return transformed


def _prompt_retention(prompt, scales, centers):
    """Fraction of a binary object prompt retained by each centred crop."""
    if prompt.ndim == 3:
        prompt = prompt.unsqueeze(1)
    if prompt.ndim != 4:
        raise ValueError('object_prompt must have shape (B, H, W) or (B, C, H, W)')
    _, _, height, width = prompt.shape
    y = (torch.arange(height, device=prompt.device, dtype=scales.dtype) + 0.5) / height
    x = (torch.arange(width, device=prompt.device, dtype=scales.dtype) + 0.5) / width
    half_window = 0.5 / scales[:, None, None]
    retained_window = (
        (x.view(1, 1, width) - centers[:, 0, None, None]).abs() <= half_window
    ) & (
        (y.view(1, height, 1) - centers[:, 1, None, None]).abs() <= half_window
    )
    foreground = (prompt > 0).to(dtype=scales.dtype)
    total = foreground.sum(dim=(1, 2, 3))
    retained = (foreground * retained_window[:, None]).sum(dim=(1, 2, 3))
    # An empty proposal is no evidence either way, so it must not reject a
    # scale view by itself.
    return torch.where(total > 0, retained / total.clamp_min(1.0), torch.ones_like(total))


def training_zoom_view(image, *, keypoints=None, object_prompt=None,
                       min_scale=1.02, max_scale=1.15, probability=0.5,
                       min_keypoint_retention=0.8, min_object_retention=0.8,
                       focus_prompts=True):
    """Build a conservative, mild prompt-focused zoom for contact-head training.

    Sampling is per image.  A proposed zoom falls back to the original view if
    it would discard too much valid SAM body-prompt or object-prompt evidence.
    The returned ``applied`` mask identifies samples for which scale
    consistency is meaningful; unchanged 3D contact labels remain valid for
    both the zoomed and fallback samples.
    """
    if image.ndim != 4:
        raise ValueError(f'image must be (B, C, H, W), got {tuple(image.shape)}')
    if not (1.0 <= min_scale <= max_scale):
        raise ValueError('training zoom scales must satisfy 1.0 <= min_scale <= max_scale')
    if not (0.0 <= probability <= 1.0):
        raise ValueError('training zoom probability must lie in [0, 1]')
    if not (0.0 <= min_keypoint_retention <= 1.0 and 0.0 <= min_object_retention <= 1.0):
        raise ValueError('prompt-retention thresholds must lie in [0, 1]')

    batch_size = image.shape[0]
    scales = image.new_empty(batch_size).uniform_(float(min_scale), float(max_scale))
    focus_center = (
        prompt_focus_centers(keypoints, object_prompt, batch_size=batch_size,
                             device=image.device, dtype=image.dtype)
        if focus_prompts else image.new_full((batch_size, 2), 0.5)
    )
    centers = _zoom_centers(focus_center, scales)
    requested = torch.rand(batch_size, device=image.device) < float(probability)
    keep = requested.clone()
    keypoint_retention = image.new_ones(batch_size)
    object_retention = image.new_ones(batch_size)

    zoom_keypoints_ = zoom_keypoints(keypoints, scales, center=centers)
    if keypoints is not None:
        original_valid = keypoints[..., 2] >= 0
        zoom_valid = zoom_keypoints_[..., 2] >= 0
        original_count = original_valid.sum(dim=1)
        keypoint_retention = zoom_valid.sum(dim=1).to(image.dtype) / original_count.clamp_min(1).to(image.dtype)
        keep &= (original_count == 0) | (keypoint_retention >= float(min_keypoint_retention))

    zoom_prompt = None
    if object_prompt is not None:
        prompt_for_zoom = object_prompt.unsqueeze(1) if object_prompt.ndim == 3 else object_prompt
        zoom_prompt = zoom_in_view(prompt_for_zoom.float(), scales, mode='nearest', center=centers).to(object_prompt.dtype)
        object_retention = _prompt_retention(prompt_for_zoom, scales, centers)
        keep &= object_retention >= float(min_object_retention)

    zoom_image = zoom_in_view(image, scales, center=centers)
    image_out = torch.where(keep[:, None, None, None], zoom_image, image)
    keypoints_out = None
    if keypoints is not None:
        keypoints_out = torch.where(keep[:, None, None], zoom_keypoints_, keypoints)
    prompt_out = None
    if object_prompt is not None:
        prompt_out = torch.where(keep[:, None, None, None], zoom_prompt, prompt_for_zoom)
        if object_prompt.ndim == 3:
            prompt_out = prompt_out.squeeze(1)

    return {
        'image': image_out,
        'keypoints': keypoints_out,
        'object_prompt': prompt_out,
        'applied': keep,
        'scales': scales,
        'centers': centers,
        'keypoint_retention': keypoint_retention,
        'object_retention': object_retention,
    }


class TestTimeScalingAdapter:
    """Per-instance, scale-consistency adaptation with no labels or geometry.

    The original crop is the EMA-teacher view.  Each prompt-focused (or centred
    fallback) zoom is a student view of that same crop and is trained to
    reproduce the teacher's per-vertex contact probabilities.  Adaptation
    changes only fusion/contact weights; frozen encoders are shared and every
    mutable state is restored after the instance.  Final output averages
    original and zoomed predictions, so high-resolution evidence directly
    contributes at inference time.
    """
    __test__ = False

    def __init__(self, model, device, *, steps=2, learning_rate=1e-5,
                 zoom_scales=(1.25, 1.5), consistency_weight=1.0,
                 ensemble_original_weight=1.0, original_anchor_weight=1.0,
                 highres_zoom_enabled=True, min_keypoint_retention=0.8,
                 min_object_retention=0.8,
                 ema_momentum=0.996,
                 focus_prompts=True):
        if not hasattr(model, 'cross_att') or not hasattr(model, 'classif'):
            raise ValueError('scale TTA requires model.cross_att and model.classif')
        scales = tuple(float(scale) for scale in zoom_scales)
        if not scales or any(scale <= 1.0 for scale in scales):
            raise ValueError('ZOOM_SCALES must contain one or more values greater than 1.0')
        if (consistency_weight <= 0 or ensemble_original_weight <= 0
                or original_anchor_weight < 0):
            raise ValueError(
                'consistency and ensemble weights must be positive; anchor weight must be non-negative'
            )
        if not (0.0 <= min_keypoint_retention <= 1.0 and 0.0 <= min_object_retention <= 1.0):
            raise ValueError('zoom prompt-retention thresholds must lie in [0, 1]')
        self.model = model
        self.device = torch.device(device)
        self.steps = max(int(steps), 0)
        self.zoom_scales = scales
        self.consistency_weight = float(consistency_weight)
        self.ensemble_original_weight = float(ensemble_original_weight)
        self.original_anchor_weight = float(original_anchor_weight)
        self.highres_zoom_enabled = bool(highres_zoom_enabled)
        self.min_keypoint_retention = float(min_keypoint_retention)
        self.min_object_retention = float(min_object_retention)
        self.ema_momentum = float(ema_momentum)
        self.focus_prompts = bool(focus_prompts)
        self._requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.adapt_modules = (model.cross_att, model.classif)
        for module in self.adapt_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.teacher_cross_att = copy.deepcopy(model.cross_att).to(self.device).eval()
        self.teacher_classif = copy.deepcopy(model.classif).to(self.device).eval()
        for module in (self.teacher_cross_att, self.teacher_classif):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            [parameter for module in self.adapt_modules for parameter in module.parameters()],
            lr=learning_rate, weight_decay=0.0,
        )
        self._initial_student = [copy.deepcopy(module.state_dict()) for module in self.adapt_modules]
        self._initial_teacher = [
            copy.deepcopy(self.teacher_cross_att.state_dict()),
            copy.deepcopy(self.teacher_classif.state_dict()),
        ]

    @staticmethod
    def _contact_output(output):
        return output[0] if isinstance(output, (tuple, list)) else output

    def _reset(self):
        for module, state in zip(self.adapt_modules, self._initial_student):
            module.load_state_dict(state)
        self.teacher_cross_att.load_state_dict(self._initial_teacher[0])
        self.teacher_classif.load_state_dict(self._initial_teacher[1])
        self.optimizer.state.clear()

    def _forward(self, image, keypoints, object_prompt):
        return self._contact_output(
            self.model(image, keypoints=keypoints, object_prompt=object_prompt)
        )

    def _teacher_forward(self, image, keypoints, object_prompt):
        student_cross, student_classif = self.model.cross_att, self.model.classif
        self.model.cross_att, self.model.classif = self.teacher_cross_att, self.teacher_classif
        try:
            return self._forward(image, keypoints, object_prompt)
        finally:
            self.model.cross_att, self.model.classif = student_cross, student_classif

    def _zoom_inputs(self, image, keypoints, object_prompt, scale, *,
                     highres_image=None, highres_object_prompt=None,
                     has_object_prompt=None):
        """Create a conservative zoom, preferably from the original crop.

        ``highres_image`` is a 512px (by default) resize of the original crop.
        Cropping it before reducing back to the model's 256px input retains
        source detail; falling back to ``image`` preserves old checkpoints and
        non-TTA callers.  Invalid/cropped-away prompts are marked so their
        zoom prediction cannot affect the ensemble or adaptation loss.
        """
        use_highres = self.highres_zoom_enabled and highres_image is not None
        source_image = highres_image if use_highres else image
        source_prompt = (
            highres_object_prompt
            if use_highres and highres_object_prompt is not None
            else object_prompt
        )
        center = (
            prompt_focus_centers(keypoints, source_prompt, batch_size=image.shape[0],
                                 device=image.device, dtype=image.dtype)
            if self.focus_prompts else None
        )
        zoom_image = zoom_in_view(source_image, scale, center=center)
        if zoom_image.shape[-2:] != image.shape[-2:]:
            zoom_image = F.interpolate(
                zoom_image, size=image.shape[-2:], mode='bilinear', align_corners=False,
            )
        zoom_prompt = None
        if source_prompt is not None:
            prompt = source_prompt.unsqueeze(1) if source_prompt.ndim == 3 else source_prompt
            zoom_prompt = zoom_in_view(
                prompt.float(), scale, mode='nearest', center=center,
            )
            if zoom_prompt.shape[-2:] != image.shape[-2:]:
                zoom_prompt = F.interpolate(zoom_prompt, size=image.shape[-2:], mode='nearest')
            zoom_prompt = zoom_prompt.to(prompt.dtype)

        batch_size = image.shape[0]
        scales = _zoom_scales(scale, batch_size, device=image.device, dtype=image.dtype)
        centers = _zoom_centers(center, scales)
        valid = torch.ones(batch_size, device=image.device, dtype=torch.bool)
        keypoint_retention = image.new_ones(batch_size)
        zoom_keypoints_ = zoom_keypoints(keypoints, scale, center=center)
        if keypoints is not None:
            original_valid = keypoints[..., 2] >= 0
            zoom_valid = zoom_keypoints_[..., 2] >= 0
            original_count = original_valid.sum(dim=1)
            keypoint_retention = (
                zoom_valid.sum(dim=1).to(image.dtype)
                / original_count.clamp_min(1).to(image.dtype)
            )
            valid &= (original_count == 0) | (keypoint_retention >= self.min_keypoint_retention)

        object_retention = image.new_ones(batch_size)
        if source_prompt is not None:
            object_retention = _prompt_retention(prompt, scales, centers)
            if has_object_prompt is None:
                has_object_evidence = torch.ones(batch_size, device=image.device, dtype=torch.bool)
            else:
                has_object_evidence = has_object_prompt.to(device=image.device).reshape(batch_size) >= 0.5
            # A full-frame fallback prompt carries no object-boundary evidence;
            # do not reject a zoom merely because that sentinel is cropped.
            valid &= (~has_object_evidence) | (object_retention >= self.min_object_retention)

        return {
            'image': zoom_image,
            'keypoints': zoom_keypoints_,
            'object_prompt': zoom_prompt,
            'valid': valid,
            'keypoint_retention': keypoint_retention,
            'object_retention': object_retention,
        }

    def adapt_and_predict(self, image, *, keypoints=None, object_prompt=None,
                          highres_image=None, highres_object_prompt=None,
                          has_object_prompt=None):
        """Adapt on zoomed copies of one batch and return an ensemble prediction."""
        self._reset()
        old_training = self.model.training
        self.model.eval()
        self.teacher_cross_att.eval()
        self.teacher_classif.eval()
        diagnostics = {
            'loss': 0.0,
            'scale_consistency_loss': 0.0,
            'original_anchor_loss': 0.0,
            'zoom_valid_fraction': 1.0,
            'steps': self.steps,
        }
        try:
            # Preserve the checkpoint's original-view output.  The final
            # ensemble must be anchored to this unadapted prediction, rather
            # than to a student that may have overfit one image's zoom views.
            with torch.no_grad():
                original_prediction = self._teacher_forward(
                    image, keypoints, object_prompt,
                ).detach()
            # Geometry/prompt transforms are model-independent, so make every
            # view once and reuse it for adaptation and final fusion.
            zoom_views = [
                self._zoom_inputs(
                    image, keypoints, object_prompt, scale,
                    highres_image=highres_image,
                    highres_object_prompt=highres_object_prompt,
                    has_object_prompt=has_object_prompt,
                )
                for scale in self.zoom_scales
            ]
            diagnostics['zoom_valid_fraction'] = float(torch.stack([
                zoom['valid'].float().mean() for zoom in zoom_views
            ]).mean().detach().cpu())
            for _ in range(self.steps):
                with torch.no_grad():
                    target = self._teacher_forward(image, keypoints, object_prompt).detach()
                losses = []
                for zoom in zoom_views:
                    prediction = self._forward(
                        zoom['image'], zoom['keypoints'], zoom['object_prompt'],
                    )
                    per_sample_loss = (prediction - target).square().mean(dim=1)
                    if zoom['valid'].any():
                        losses.append(per_sample_loss[zoom['valid']].mean())
                # A heavily cropped prompt should not force an update.  The
                # saved original prediction remains the output for this image.
                if not losses:
                    break
                consistency = torch.stack(losses).mean()
                # The zoom loss alone gives no constraint on how the update
                # changes the original crop.  This anchor penalizes that
                # drift while preserving a fully label-free objective.
                original_anchor = F.mse_loss(
                    self._forward(image, keypoints, object_prompt), original_prediction,
                )
                total = (
                    self.consistency_weight * consistency
                    + self.original_anchor_weight * original_anchor
                )
                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                self.optimizer.step()
                ema_update(self.teacher_cross_att, self.model.cross_att, self.ema_momentum, skip_prefix=None)
                ema_update(self.teacher_classif, self.model.classif, self.ema_momentum, skip_prefix=None)
                diagnostics.update({
                    'loss': float(total.detach().cpu()),
                    'scale_consistency_loss': float(consistency.detach().cpu()),
                    'original_anchor_loss': float(original_anchor.detach().cpu()),
                })
            with torch.no_grad():
                # Use the saved checkpoint view, never the adapted original
                # view, as the ensemble anchor.
                predictions = [original_prediction]
                weights = [image.new_full((image.shape[0],), self.ensemble_original_weight)]
                for zoom in zoom_views:
                    predictions.append(self._forward(
                        zoom['image'], zoom['keypoints'], zoom['object_prompt'],
                    ))
                    weights.append(zoom['valid'].to(dtype=image.dtype))
                weight = torch.stack(weights, dim=0).unsqueeze(-1)
                prediction = (torch.stack(predictions, dim=0) * weight).sum(0) / weight.sum(0).clamp_min(1e-8)
            return prediction.detach(), diagnostics
        finally:
            self._reset()
            self.model.train(old_training)

    def close(self):
        """Restore the caller's original gradient flags."""
        self._reset()
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(self._requires_grad[name])


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
