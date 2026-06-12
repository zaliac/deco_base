"""DINO teacher-student TEST-TIME ADAPTATION (TTA) for DECO contact prediction.

Frozen-anchor variant. For each test batch we run K gradient steps that make the STUDENT's
prediction/features on a STRONG photometric view match a FROZEN teacher's on a WEAK view, then
predict on the clean image and reset the adapted weights (episodic). No labels are used -- the
teacher-student consistency is the only signal, which is exactly DINO's label-free regime.

  * teacher = a fixed copy of the trained model, built with utils.distill.build_teacher so it
    SHARES the frozen SAM backbone (the 840M ViT is never duplicated). It is NOT EMA-updated in
    the default (episodic) mode -- a fixed target pins the optimum so small-batch adaptation
    cannot collapse to a trivial constant (no DINO centering/sharpening needed at test time).
  * adapt-set = the small downstream modules from utils.ttt.collect_ttt_params (hrnet_to_sam,
    fusion norms / pos-emb, the contact-head input proj) -- the frozen ViT is never adapted.
  * two views = utils.distill.two_views (photometric only; geometry preserved). The losses:
    output-consistency MSE on the contact probs + cosine on the pooled fused features.

IN-DOMAIN EXPECTATION (read this before trusting any metric delta): at step 0 student == teacher,
so the loss measures only the model's own sensitivity to photometric augmentation. A backbone
trained with colour jitter is already ~robust, so in-domain the loss ~ 0, the gradient ~ 0, and
adaptation is a near-no-op -- the same mechanism that made the L-R flip-TTT a no-op (mean
flip-loss ~ 0.009). The `diag` returned by predict() (and the running mean from report()) is
exactly that pre-adaptation consistency: if it is ~0 there is no signal to adapt on and TTA
cannot move the metrics. TTA only bites under DOMAIN SHIFT, where the model IS inconsistent
across views.
"""
import torch
import torch.nn.functional as F

from utils.distill import two_views, FeatureGrabber, build_teacher, ema_update
from utils.ttt import collect_ttt_params, _forward_cont


def _forward_full(model, img, keypoints):
    """Run the model, returning its full output (contact tensor, or (cont, sem, part) tuple)."""
    return model(img, keypoints) if keypoints is not None else model(img)


class DinoTTA:
    """Holds the frozen-anchor teacher + adapt-set + optimizer once; predict() adapts per batch."""

    def __init__(self, model, mean, std, steps=2, lr=1e-3, out_w=1.0, feat_w=1.0,
                 online=False, ema_momentum=0.999, params=None):
        self.model = model
        self.steps = steps
        self.lr = lr
        self.out_w = out_w
        self.feat_w = feat_w
        self.online = online
        self.ema_momentum = ema_momentum
        self.mean = mean                                 # (1,3,1,1); use zeros/ones if input is raw [0,1]
        self.std = std
        self.params = params if params is not None else collect_ttt_params(model)
        self.plist = [p for _, p in self.params]
        # frozen-anchor teacher: shares the frozen ViT, deep-copies only the small downstream tail.
        # Built BEFORE the FeatureGrabber hooks so deepcopy never sees a hook on .classif.
        self.teacher = build_teacher(model, share_prefix='encoder_part')
        self.student_feat = FeatureGrabber(model.classif)
        self.teacher_feat = FeatureGrabber(self.teacher.classif)
        self.opt = torch.optim.SGD(self.plist, lr=lr) if self.plist else None
        self._diag_sum = 0.0                             # running pre-adaptation consistency (flip-loss style)
        self._diag_n = 0
        names = [n for n, _ in self.params]
        print(f'✓ DINO-TTA ON (frozen-anchor, {"online" if online else "episodic"}): '
              f'{len(self.plist)} adapt tensors, steps={steps}, lr={lr:g}, out_w={out_w}, feat_w={feat_w}')
        print(f'  adapt-set: {names[:6]}{"..." if len(names) > 6 else ""}')

    def _consistency(self, img, keypoints):
        """Scalar teacher(weak) vs student(strong) consistency: output MSE + feature cosine."""
        t_view, s_view = two_views(img, self.mean, self.std)
        with torch.no_grad():
            t_cont = _forward_cont(self.teacher, t_view, keypoints)
            t_feat = self.teacher_feat.feat
        s_cont = _forward_cont(self.model, s_view, keypoints)        # captures student_feat via hook
        s_feat = self.student_feat.feat
        loss = self.out_w * F.mse_loss(s_cont, t_cont)
        if self.feat_w > 0:
            loss = loss + self.feat_w * (1.0 - F.cosine_similarity(s_feat, t_feat, dim=-1)).mean()
        return loss

    @torch.enable_grad()
    def predict(self, img, keypoints=None):
        """Episodically adapt on `img`, then return (full model output, step-0 consistency diag).

        The @enable_grad re-enables autograd even though tester.py's evaluate() runs under no_grad.
        """
        if self.opt is None or self.steps == 0:
            with torch.no_grad():
                return _forward_full(self.model, img, keypoints), 0.0

        was_training = self.model.training
        self.model.eval()                                # adapt-set is norm/small layers -> keep BN in eval
        prev_rg = [p.requires_grad for p in self.plist]
        for p in self.plist:
            p.requires_grad_(True)
        snapshot = None if self.online else [p.detach().clone() for p in self.plist]

        diag = 0.0
        for k in range(self.steps):
            self.opt.zero_grad(set_to_none=True)
            loss = self._consistency(img, keypoints)
            if k == 0:
                diag = loss.detach().item()              # pre-adaptation consistency (the no-op tell)
            loss.backward()
            self.opt.step()
            if self.online:                              # CoTTA-style: let the teacher track the student
                ema_update(self.teacher, self.model, self.ema_momentum, skip_prefix='encoder_part')

        with torch.no_grad():
            out = _forward_full(self.model, img, keypoints)   # predict on the clean image

        if snapshot is not None:                         # episodic reset
            with torch.no_grad():
                for p, s in zip(self.plist, snapshot):
                    p.copy_(s)
        for p, rg in zip(self.plist, prev_rg):
            p.requires_grad_(rg)
        if was_training:
            self.model.train()

        self._diag_sum += diag
        self._diag_n += 1
        return out, diag

    def report(self):
        """Print + return the mean step-0 consistency so far. ~0 => no in-domain signal."""
        m = self._diag_sum / max(self._diag_n, 1)
        print(f'[DINO-TTA] mean step-0 consistency over {self._diag_n} batch(es): {m:.5f}  '
              f'(~0 => model already augmentation-consistent; TTA cannot move metrics here)')
        return m

    def remove(self):
        self.student_feat.remove()
        self.teacher_feat.remove()
