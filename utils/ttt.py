"""Test-Time Training (TTT) for DECO contact prediction, applied at inference.

Instance-level test-time adaptation: for each test image we run a few self-supervised
gradient steps on a *label-free* objective, then predict. DECO's trained weights act as
the outer-loop init and the per-image steps as the inner loop -- the "learn at test time"
idea of Sun et al. 2024 ("RNNs with Expressive Hidden States"), reusing its self-supervised
inner loop rather than its sequence layer (DECO is a single-image task, not a sequence).

Self-supervised objective: left-right flip / symmetry consistency. A horizontal image flip
swaps the body's left and right, so contact at vertex v in the original should match contact
at its mirror vertex in the flipped image:

    cont(I)[v]  ~=  cont(flip(I))[mirror(v)]

The 6890-vertex mirror map is derived once from the SMPL T-pose template and cached at
data/smpl/smpl_lr_mirror_idx.npy (see _derive_mirror_index). Midline / unpaired vertices
(mirror(v) == v) are excluded from the loss.

Adaptation is episodic by default (weights reset after each image) and adapts only small
modules *downstream of the frozen SAM backbone* (hrnet_to_sam, the fusion norms / positional
embeddings, the contact-head input projection), so the expensive ViT is never adapted and
its features could be cached (see the note in ttt_predict).
"""
import os

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# SMPL left-right vertex mirror map
# ---------------------------------------------------------------------------
def load_lr_mirror_index(smpl_dir='data/smpl', device='cpu'):
    """Return (mirror_idx [V] long, paired_mask [V] bool).

    mirror_idx[v] = the left-right counterpart of vertex v (an involution).
    paired_mask[v] = False for midline / fallback vertices (mirror(v) == v), which carry no
    L-R signal and are excluded from the consistency loss. Loads the cached asset, deriving
    and caching it from the T-pose template the first time.
    """
    path = os.path.join(smpl_dir, 'smpl_lr_mirror_idx.npy')
    if not os.path.exists(path):
        _derive_mirror_index(smpl_dir, path)
    mir = torch.as_tensor(np.load(path), dtype=torch.long, device=device)
    paired = mir != torch.arange(len(mir), device=device)
    return mir, paired


def _derive_mirror_index(smpl_dir, out_path):
    """Derive an involutive L-R vertex map from the SMPL T-pose template (greedy reciprocal
    nearest-neighbour pairing on the x-mirrored vertices, ascending residual)."""
    import trimesh
    ply = os.path.join(smpl_dir, 'smpl_neutral_tpose.ply')
    V = torch.tensor(np.asarray(trimesh.load(ply, process=False).vertices), dtype=torch.float32)
    N = V.shape[0]
    Vm = V.clone()
    Vm[:, 0] *= -1                                    # mirror across x = 0
    nn_idx = torch.cdist(Vm, V).argmin(1)
    resid = (Vm - V[nn_idx]).norm(dim=1)
    mir = torch.full((N,), -1, dtype=torch.long)
    for i in torch.argsort(resid).tolist():          # smallest residual first
        if mir[i] != -1:
            continue
        j = int(nn_idx[i])
        if mir[j] == -1 and j != i:
            mir[i], mir[j] = j, i                     # reciprocal pair
        else:
            mir[i] = i                               # midline / fallback -> self
    os.makedirs(smpl_dir, exist_ok=True)
    np.save(out_path, mir.numpy().astype(np.int64))


# ---------------------------------------------------------------------------
# forward helper + self-supervised loss
# ---------------------------------------------------------------------------
def _forward_cont(model, img, keypoints=None):
    """Call the model and return only the contact tensor (B, V)."""
    out = model(img, keypoints) if keypoints is not None else model(img)
    return out[0] if isinstance(out, (tuple, list)) else out


def flip_consistency_loss(model, img, mirror_idx, paired_mask, keypoints=None, kp_flip=None):
    """Label-free L-R flip/symmetry consistency loss (scalar MSE over paired vertices)."""
    cont = _forward_cont(model, img, keypoints)                      # (B, V) in [0, 1]
    img_f = torch.flip(img, dims=[-1])                              # horizontal flip
    kp_f = kp_flip(keypoints) if (keypoints is not None and kp_flip is not None) else None
    cont_f = _forward_cont(model, img_f, kp_f)                      # (B, V)
    cont_f_m = cont_f.index_select(1, mirror_idx)                  # remap to original vertex order
    diff = (cont - cont_f_m)[:, paired_mask]                       # only genuinely-paired vertices
    return diff.pow(2).mean()


# ---------------------------------------------------------------------------
# adapt-set selection
# ---------------------------------------------------------------------------
# Small modules downstream of the frozen SAM backbone. Adapting few params is what keeps
# single-image TTT stable (cf. norm-only adaptation in the TTT/TENT literature).
DEFAULT_ADAPT_SUBSTRINGS = (
    'hrnet_to_sam',
    'cross_att.norm_in', 'cross_att.pos_', 'cross_att.mod_embed',
    'classif.mem_proj',
)
_NORM_TYPES = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)


def collect_ttt_params(model, substrings=DEFAULT_ADAPT_SUBSTRINGS):
    """Return the adapt-set as list[(name, Parameter)].

    Uses the curated `substrings` (the sam_hrnet fusion/head modules). If none match (e.g. a
    non-sam_hrnet model), falls back to normalization-layer affine params *outside the
    encoders* -- the classic norm-only test-time adapt-set -- so the backbone stays frozen.
    """
    sel = [(n, p) for n, p in model.named_parameters() if any(s in n for s in substrings)]
    if sel:
        return sel
    for mname, m in model.named_modules():           # fallback: norm affine, not in encoders
        if isinstance(m, _NORM_TYPES) and not (mname.startswith('encoder_part')
                                               or mname.startswith('encoder_sem')):
            sel += [(f'{mname}.{pn}', p) for pn, p in m.named_parameters(recurse=False)]
    return sel


# ---------------------------------------------------------------------------
# episodic test-time adaptation + predict
# ---------------------------------------------------------------------------
@torch.enable_grad()
def ttt_predict(model, img, mirror_idx, paired_mask, params=None, keypoints=None,
                kp_flip=None, steps=1, lr=1e-2, online=False, verbose=False):
    """Adapt on `img` via flip-consistency, then predict. Returns contact (B, V), no grad.

    params  : list[(name, Parameter)] to adapt; default = collect_ttt_params(model).
    steps   : inner SGD steps (0 -> plain prediction, no adaptation).
    online  : keep adapted weights across calls (default False -> reset after each image).

    Keep the model in eval() (BN/dropout): the adapt-set is norm-affine / small layers, so
    batch-size-1 BN statistics are never recomputed. The frozen SAM backbone is excluded, so
    its (expensive) forward is identical every step -- caching it would make this ~steps x
    cheaper, at the cost of an encode/decode split in DECO.forward (not done here).
    """
    if params is None:
        params = collect_ttt_params(model)
    plist = [p for _, p in params]
    if steps == 0 or not plist:
        with torch.no_grad():
            return _forward_cont(model, img, keypoints)

    prev_rg = [p.requires_grad for p in plist]
    for p in plist:
        p.requires_grad_(True)
    snapshot = None if online else [p.detach().clone() for p in plist]

    opt = torch.optim.SGD(plist, lr=lr)
    for k in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = flip_consistency_loss(model, img, mirror_idx, paired_mask, keypoints, kp_flip)
        loss.backward()
        opt.step()
        if verbose:
            print(f'  [ttt] step {k + 1}/{steps}  flip_consistency={loss.item():.5f}')

    with torch.no_grad():
        cont = _forward_cont(model, img, keypoints)

    if snapshot is not None:                          # episodic reset
        with torch.no_grad():
            for p, s in zip(plist, snapshot):
                p.copy_(s)
    for p, rg in zip(plist, prev_rg):                 # restore requires_grad flags
        p.requires_grad_(rg)
    return cont
