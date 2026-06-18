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
backbone (the 840M ViT is never duplicated / deep-copied); and the trainer ramps the weights 0->1.
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
# EMA teacher (shares the frozen backbone -> never duplicates / deep-copies the ViT)
# ---------------------------------------------------------------------------
def build_teacher(student, share_prefix='encoder_part'):
    """Deep-copy `student` into an EMA teacher, but SHARE (not copy) the frozen backbone module
    `student.<share_prefix>` -- temporarily swapped out before the deepcopy so the 840M ViT is
    neither duplicated in memory nor deep-copied (deepcopy can fail on parametrized submodules).
    eval() + no-grad."""
    shared = getattr(student, share_prefix, None) if share_prefix else None
    if shared is not None:
        setattr(student, share_prefix, nn.Identity())
    try:
        teacher = copy.deepcopy(student)
    finally:
        if shared is not None:
            setattr(student, share_prefix, shared)
    if shared is not None:
        setattr(teacher, share_prefix, shared)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


@torch.no_grad()
def ema_update(teacher, student, momentum, skip_prefix='encoder_part'):
    """teacher = momentum*teacher + (1-momentum)*student for params NOT under `skip_prefix`
    (the shared frozen backbone). Buffers (e.g. BN stats) are copied from the student."""
    sp = dict(student.named_parameters())
    for name, tp in teacher.named_parameters():
        if skip_prefix and name.startswith(skip_prefix):
            continue
        s = sp.get(name)
        if s is not None:
            tp.mul_(momentum).add_(s.detach(), alpha=1 - momentum)
    sb = dict(student.named_buffers())
    for name, tb in teacher.named_buffers():
        if skip_prefix and name.startswith(skip_prefix):
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
