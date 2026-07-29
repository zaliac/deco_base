import sys
from pathlib import Path

import torch
import torch.nn as nn


# This project is a script-style repository rather than an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.distill import (
    TestTimeScalingAdapter, zoom_in_view, zoom_keypoints,
)


class _TinyContactHead(nn.Module):
    """Small token-based contact head with the same interface as DECO.classif."""
    def __init__(self):
        super().__init__()
        self.mem_proj = nn.Linear(3, 4)
        self.output = nn.Linear(4, 5)

    def forward(self, tokens):
        x = torch.tanh(self.mem_proj(tokens))
        return torch.sigmoid(self.output(x).mean(dim=1))


class _TinyDECO(nn.Module):
    """Avoids checkpoint/SMPL dependencies while exercising the TTA contract."""
    def __init__(self):
        super().__init__()
        self.encoder_part = nn.Conv2d(3, 3, kernel_size=1)
        self.encoder_sem = nn.Identity()
        self.cross_att = nn.Sequential(nn.Linear(3, 3), nn.Tanh())
        self.classif = _TinyContactHead()

    def forward(self, image, keypoints=None, object_prompt=None):
        tokens = self.encoder_part(image).mean(dim=(2, 3)).unsqueeze(1)
        return self.classif(self.cross_att(tokens))


def test_test_time_scale_adaptation_rolls_back_everything_between_instances():
    torch.manual_seed(7)
    model = _TinyDECO()
    initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    initial_requires_grad = {name: param.requires_grad for name, param in model.named_parameters()}

    adapter = TestTimeScalingAdapter(
        model,
        'cpu',
        steps=2,
        learning_rate=1e-2,
        zoom_scales=(1.25, 1.5),
    )
    for image in (torch.randn(1, 3, 8, 8), torch.randn(1, 3, 8, 8)):
        # ``TrainStepper.evaluate`` is globally no-grad and selectively reopens
        # autograd around Task-7 adaptation; exercise that exact nesting here.
        with torch.no_grad():
            with torch.enable_grad():
                prediction, stats = adapter.adapt_and_predict(image)

        assert prediction.shape == (1, 5)
        assert prediction.requires_grad is False
        assert stats['steps'] == 2
        assert torch.isfinite(torch.tensor(stats['loss']))
        # The public model is exactly back at the loaded checkpoint state before
        # the caller can move to the next image.
        for name, value in model.state_dict().items():
            assert torch.equal(value, initial_state[name]), name
        # Adam moments are reset, avoiding cross-image leakage.
        assert adapter.optimizer.state == {}

    adapter.close()
    for name, param in model.named_parameters():
        assert param.requires_grad == initial_requires_grad[name], name


def test_zoom_in_keeps_shape_and_remaps_keypoint_prompts():
    image = torch.arange(64, dtype=torch.float32).view(1, 1, 8, 8)
    zoom = zoom_in_view(image, 2.0)
    assert zoom.shape == image.shape
    # A centred zoom moves a point at x=0.25 to the left boundary; a point at
    # x=0.10 leaves the crop and becomes an invalid SAM prompt.
    prompts = torch.tensor([[[0.25, 0.50, 3.0], [0.10, 0.50, 4.0]]])
    transformed = zoom_keypoints(prompts, 2.0)
    assert torch.allclose(transformed[0, 0, :2], torch.tensor([0.0, 0.5]))
    assert transformed[0, 0, 2].item() == 3.0
    assert transformed[0, 1, 2].item() == -2.0
