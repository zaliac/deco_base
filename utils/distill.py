"""DINO-style teacher-student self-distillation for DECO training (auxiliary SSL).

The teacher is an EMA of the student. Each step uses two augmented views of the same image:
the teacher sees a weak view, the student a strong (photometric) view. Two distillation losses
regularize training on top of the supervised contact/seg losses:

  - feature DINO: a projection head maps the pooled fused image features to `out_dim` prototype
    logits; the student matches the teacher's sharpened + centered distribution (cross-entropy).
  - output consistency: the student's per-vertex contact prediction matches the teacher's (MSE).

Memory: the teacher SHARES the student's frozen SAM backbone (only the trainable tail is EMA'd),
so it never duplicates the 840M ViT. Photometric augmentation preserves geometry, so the
supervised contact/keypoint labels stay valid on the student view.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DINO projection head + loss
# ---------------------------------------------------------------------------
class DINOHead(nn.Module):
    """DINO projection head: 3-layer MLP -> L2-normalize -> prototype layer.

    (No weight_norm on the last layer: it makes the module un-deepcopyable, which would break
    the EMA-teacher copy. The L2-normalize of the bottleneck is the key DINO normalization.)"""
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
    """DINO cross-entropy: student log-softmax vs teacher (centered + sharpened) softmax."""
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
        batch_center = teacher_out.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)


# ---------------------------------------------------------------------------
# EMA teacher (shares the frozen backbone to avoid duplicating the ViT)
# ---------------------------------------------------------------------------
def build_teacher(student, share_prefix='encoder_part'):
    """Deep-copy `student` into an EMA teacher, but SHARE (not copy) the frozen backbone module
    `student.<share_prefix>`. The backbone is temporarily swapped out before the deepcopy, so the
    840M ViT is neither duplicated in memory nor deep-copied (deepcopy can fail on parametrized /
    weight-normed submodules). eval() + no-grad."""
    shared = getattr(student, share_prefix, None) if share_prefix else None
    if shared is not None:
        setattr(student, share_prefix, nn.Identity())        # exclude the backbone from the deepcopy
    try:
        teacher = copy.deepcopy(student)
    finally:
        if shared is not None:
            setattr(student, share_prefix, shared)           # restore the student
    if shared is not None:
        setattr(teacher, share_prefix, shared)               # teacher shares the same backbone module
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


@torch.no_grad()
def ema_update(teacher, student, momentum, skip_prefix='encoder_part'):
    """teacher = momentum*teacher + (1-momentum)*student, for params NOT under `skip_prefix`
    (those are the shared frozen backbone). Buffers (e.g. BN stats) are copied from the student."""
    s_params = dict(student.named_parameters())
    for name, tp in teacher.named_parameters():
        if skip_prefix and name.startswith(skip_prefix):
            continue
        sp = s_params.get(name)
        if sp is not None:
            tp.mul_(momentum).add_(sp.detach(), alpha=1 - momentum)
    s_buffers = dict(student.named_buffers())
    for name, tb in teacher.named_buffers():
        if skip_prefix and name.startswith(skip_prefix):
            continue
        sb = s_buffers.get(name)
        if sb is not None and sb.shape == tb.shape:
            tb.copy_(sb)


# ---------------------------------------------------------------------------
# pooled-fused-feature grabber (forward pre-hook on the contact head)
# ---------------------------------------------------------------------------
class FeatureGrabber:
    """Capture the pooled fused tokens fed into the contact head (its forward input), via a
    forward-pre-hook -- so we get a (B, C) global descriptor without modifying DECO.forward."""
    def __init__(self, contact_head):
        self.feat = None
        self._h = contact_head.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        mem = args[0]                                  # (B, N, C) tokens into the contact head
        self.feat = mem.mean(dim=1)                    # (B, C) mean-pooled descriptor

    def remove(self):
        self._h.remove()


# ---------------------------------------------------------------------------
# two-view photometric augmentation (GPU, denorm -> jitter -> renorm)
# ---------------------------------------------------------------------------
_LUMA = (0.299, 0.587, 0.114)


def _color_jitter(x, strength):
    """Batched brightness/contrast/saturation jitter on [0,1] images (per-sample factors)."""
    B, dev = x.shape[0], x.device
    def rand():                                         # (B,1,1,1) in [1-strength, 1+strength]
        return 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * strength
    x = x * rand()                                      # brightness
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * rand() + m                            # contrast
    luma = (x * torch.tensor(_LUMA, device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
    x = luma + (x - luma) * rand()                      # saturation
    return x.clamp(0, 1)


def two_views(img, mean, std, teacher_strength=0.2, student_strength=0.4, gray_p=0.2):
    """Return (teacher_view, student_view): weak vs strong photometric aug of the SAME image.
    Geometry is preserved (so contact/keypoint labels stay valid). `mean`/`std` are (1,3,1,1)
    tensors matching how `img` was normalized."""
    raw = (img * std + mean).clamp(0, 1)
    t = _color_jitter(raw, teacher_strength)
    s = _color_jitter(raw, student_strength)
    B, dev = raw.shape[0], raw.device                   # random grayscale on the student view
    gray = (s * torch.tensor(_LUMA, device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True).repeat(1, 3, 1, 1)
    use_gray = (torch.rand(B, 1, 1, 1, device=dev) < gray_p).float()
    s = use_gray * gray + (1 - use_gray) * s
    return (t - mean) / std, (s - mean) / std
