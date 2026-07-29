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
def zoom_in_view(image, scale, *, mode='bilinear'):
    """Center-crop then resize a normalized image without changing its shape.

    ``scale > 1`` magnifies the original crop: output coordinates sample a
    smaller centred region of the source image.  Keeping the tensor resolution
    fixed lets the frozen SAM encoders consume each scale exactly as at test
    time.  ``scale == 1`` is deliberately exact, which makes the helper useful
    in tests and avoids interpolation drift in the original branch.
    """
    scale = float(scale)
    if scale < 1.0:
        raise ValueError('zoom scale must be >= 1.0')
    if scale == 1.0:
        return image
    if image.ndim != 4:
        raise ValueError(f'image must be (B, C, H, W), got {tuple(image.shape)}')
    B = image.shape[0]
    theta = image.new_zeros(B, 2, 3)
    theta[:, 0, 0] = 1.0 / scale
    theta[:, 1, 1] = 1.0 / scale
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    kwargs = {'mode': mode, 'padding_mode': 'border', 'align_corners': False}
    return F.grid_sample(image, grid, **kwargs)


def zoom_keypoints(keypoints, scale):
    """Transform normalized SAM prompt points into a centred zoom view."""
    if keypoints is None or float(scale) == 1.0:
        return keypoints
    if keypoints.ndim != 3 or keypoints.shape[-1] != 3:
        raise ValueError('keypoints must have shape (B, N, 3)')
    transformed = keypoints.clone()
    transformed[..., :2] = 0.5 + (transformed[..., :2] - 0.5) * float(scale)
    outside = ((transformed[..., :2] < 0) | (transformed[..., :2] > 1)).any(dim=-1)
    transformed[..., :2].clamp_(0, 1)
    transformed[..., 2][outside] = -2
    return transformed


class TestTimeScalingAdapter:
    """Per-instance, scale-consistency adaptation with no labels or geometry.

    The original crop is the EMA-teacher view.  Each centred zoom is a student
    view of that same crop and is trained to reproduce the teacher's per-vertex
    contact probabilities.  Adaptation changes only fusion/contact weights;
    frozen encoders are shared and every mutable state is restored after the
    instance.  Final output averages original and zoomed predictions, so the
    high-resolution zoom evidence directly contributes at inference time.
    """
    __test__ = False

    def __init__(self, model, device, *, steps=2, learning_rate=1e-5,
                 zoom_scales=(1.25, 1.5), consistency_weight=1.0,
                 ensemble_original_weight=1.0, ema_momentum=0.996):
        if not hasattr(model, 'cross_att') or not hasattr(model, 'classif'):
            raise ValueError('scale TTA requires model.cross_att and model.classif')
        scales = tuple(float(scale) for scale in zoom_scales)
        if not scales or any(scale <= 1.0 for scale in scales):
            raise ValueError('ZOOM_SCALES must contain one or more values greater than 1.0')
        if consistency_weight <= 0 or ensemble_original_weight < 0:
            raise ValueError('consistency_weight must be positive and original weight non-negative')
        self.model = model
        self.device = torch.device(device)
        self.steps = max(int(steps), 0)
        self.zoom_scales = scales
        self.consistency_weight = float(consistency_weight)
        self.ensemble_original_weight = float(ensemble_original_weight)
        self.ema_momentum = float(ema_momentum)
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

    def _zoom_inputs(self, image, keypoints, object_prompt, scale):
        zoom_image = zoom_in_view(image, scale)
        zoom_prompt = None
        if object_prompt is not None:
            if object_prompt.ndim == 3:
                object_prompt = object_prompt.unsqueeze(1)
            zoom_prompt = zoom_in_view(object_prompt.float(), scale, mode='nearest').to(object_prompt.dtype)
        return zoom_image, zoom_keypoints(keypoints, scale), zoom_prompt

    def adapt_and_predict(self, image, *, keypoints=None, object_prompt=None):
        """Adapt on zoomed copies of one batch and return an ensemble prediction."""
        self._reset()
        old_training = self.model.training
        self.model.eval()
        self.teacher_cross_att.eval()
        self.teacher_classif.eval()
        diagnostics = {'loss': 0.0, 'scale_consistency_loss': 0.0, 'steps': self.steps}
        try:
            for _ in range(self.steps):
                with torch.no_grad():
                    target = self._teacher_forward(image, keypoints, object_prompt).detach()
                losses = []
                for scale in self.zoom_scales:
                    zoom_image, zoom_keypoints_, zoom_prompt = self._zoom_inputs(
                        image, keypoints, object_prompt, scale,
                    )
                    prediction = self._forward(zoom_image, zoom_keypoints_, zoom_prompt)
                    losses.append(F.mse_loss(prediction, target))
                consistency = torch.stack(losses).mean()
                total = self.consistency_weight * consistency
                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                self.optimizer.step()
                ema_update(self.teacher_cross_att, self.model.cross_att, self.ema_momentum, skip_prefix=None)
                ema_update(self.teacher_classif, self.model.classif, self.ema_momentum, skip_prefix=None)
                diagnostics.update({
                    'loss': float(total.detach().cpu()),
                    'scale_consistency_loss': float(consistency.detach().cpu()),
                })
            with torch.no_grad():
                predictions = [self._forward(image, keypoints, object_prompt)]
                weights = [self.ensemble_original_weight]
                for scale in self.zoom_scales:
                    zoom_image, zoom_keypoints_, zoom_prompt = self._zoom_inputs(
                        image, keypoints, object_prompt, scale,
                    )
                    predictions.append(self._forward(zoom_image, zoom_keypoints_, zoom_prompt))
                    weights.append(1.0)
                weight = image.new_tensor(weights).view(-1, 1, 1)
                prediction = (torch.stack(predictions, dim=0) * weight).sum(0) / weight.sum()
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
