"""CPU-only model integrity tests; no public weights are downloaded."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
import torchvision.models._api as models_api
from torchvision.models import resnet18

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pretrained_model import PretrainedUNet

torch.set_num_threads(2)


@pytest.mark.parametrize("encoder", ["resnet18", "resnet34"])
def test_no_implicit_weight_download_and_complete_parameter_groups(monkeypatch, encoder):
    def forbid_network(*args, **kwargs):
        raise AssertionError("Constructor attempted to download weights")
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbid_network)
    monkeypatch.setattr(models_api, "load_state_dict_from_url", forbid_network)
    model = PretrainedUNet(encoder_name=encoder, weights_path=None)
    encoder_parameters = list(model.encoder_parameters())
    decoder_parameters = list(model.decoder_parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    decoder_ids = {id(parameter) for parameter in decoder_parameters}
    assert encoder_ids and decoder_ids
    assert len(encoder_ids) == len(encoder_parameters)
    assert len(decoder_ids) == len(decoder_parameters)
    assert not encoder_ids & decoder_ids
    assert encoder_ids | decoder_ids == {id(parameter) for parameter in model.parameters()}
    assert all(parameter.requires_grad for parameter in model.parameters())
    description = model.describe()
    assert description["encoder_name"] == encoder
    assert description["initialization"] == "random_encoder"
    assert description["automatic_downloads"] is False
    assert description["total_parameters"] == sum(parameter.numel() for parameter in model.parameters())
    assert not any("fc." in name for name, _ in model.named_parameters())


@pytest.mark.parametrize("encoder", ["resnet18", "resnet34"])
def test_odd_and_tiny_shapes_backward_finite_and_bn_statistics_frozen(encoder):
    torch.manual_seed(17)
    model = PretrainedUNet(encoder_name=encoder, freeze_bn=True).train()
    norms = {name: module for name, module in model.encoder.named_modules()
             if isinstance(module, nn.BatchNorm2d)}
    assert norms
    before = {name: deepcopy(module.state_dict()) for name, module in norms.items()}
    assert all(not module.training for module in norms.values())
    assert all(module.weight.requires_grad and module.bias.requires_grad for module in norms.values())
    for height, width in [(35, 47), (7, 9)]:
        image = torch.rand(1, 3, height, width)
        target = torch.zeros(1, 1, height, width)
        target[..., 2:5, 3:6] = 1
        logits = model(image)
        assert logits.shape == target.shape
        assert torch.isfinite(logits).all()
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
        assert torch.isfinite(loss)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert model.encoder["conv1"].weight.grad is not None
        assert model.encoder["bn1"].weight.grad is not None
        assert model.output.weight.grad is not None
        model.zero_grad(set_to_none=True)
    for name, module in norms.items():
        for key in ("running_mean", "running_var", "num_batches_tracked"):
            torch.testing.assert_close(module.state_dict()[key], before[name][key], rtol=0, atol=0)
    model.eval()
    model.train()
    assert all(not module.training for module in norms.values())


def test_unfrozen_batchnorm_tracks_batches_and_normalization_is_internal():
    model = PretrainedUNet(freeze_bn=False).train()
    first_bn = model.encoder["bn1"]
    before = int(first_bn.num_batches_tracked)
    image = torch.rand(2, 3, 35, 41)
    seen = []
    hook = model.encoder["conv1"].register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].detach().clone()))
    model(image)
    hook.remove()
    assert first_bn.training
    assert int(first_bn.num_batches_tracked) == before + 1
    expected = (image - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    torch.testing.assert_close(seen[0], expected)


def test_strict_local_backbone_loading_includes_classifier_before_discard(tmp_path):
    # A locally generated official-shape state tests the loader without using
    # external data or asserting that these random values are pretrained.
    backbone = resnet18(weights=None)
    with torch.no_grad():
        backbone.conv1.weight.fill_(0.0123)
        backbone.bn1.running_mean.fill_(0.345)
    path = tmp_path / "local_resnet18_state.pt"
    torch.save(backbone.state_dict(), path)
    model = PretrainedUNet(weights_path=path)
    torch.testing.assert_close(model.encoder["conv1"].weight, backbone.conv1.weight, rtol=0, atol=0)
    torch.testing.assert_close(model.encoder["bn1"].running_mean, backbone.bn1.running_mean, rtol=0, atol=0)
    description = model.describe()
    assert description["pretrained_encoder_loaded"] is True
    assert description["initialization"] == "local_encoder_state_dict"
    assert len(description["weights_sha256"]) == 64
    assert description["weights_path"] == str(path.resolve())

    incomplete = dict(backbone.state_dict())
    incomplete.pop("fc.weight")
    incomplete.pop("fc.bias")
    torch.save(incomplete, path)
    with pytest.raises(RuntimeError, match="fc"):
        PretrainedUNet(weights_path=path)


def test_clear_errors_for_unknown_encoder_or_missing_local_weights(tmp_path):
    with pytest.raises(ValueError, match="encoder_name"):
        PretrainedUNet(encoder_name="resnet50")
    with pytest.raises(FileNotFoundError, match="Local encoder weights"):
        PretrainedUNet(weights_path=tmp_path / "missing.pt")
