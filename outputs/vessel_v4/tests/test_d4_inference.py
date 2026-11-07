import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from inference import predict_image


class Pointwise(torch.nn.Module):
    def forward(self, x):
        return torch.logit(x[:, :1].clamp(1e-5, 1-1e-5))


@pytest.mark.parametrize('mode', [False, True, 'none', 'flip4', 'd4'])
def test_tta_inverse_preserves_asymmetric_image(mode):
    image = np.random.default_rng(7).uniform(.05, .95, (47, 61, 3)).astype(np.float32)
    prediction = predict_image(Pointwise(), image, torch.device('cpu'),
                               patch_size=32, stride=16, batch_size=3, tta=mode, amp=False)
    np.testing.assert_allclose(prediction, image[..., 0], atol=3e-7)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        predict_image(Pointwise(), np.zeros((32, 32, 3), np.float32),
                      torch.device('cpu'), tta='unrecognized')
