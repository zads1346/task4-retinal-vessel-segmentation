"""CPU checks for EMA arithmetic, checkpoint identity, and exact restoration."""

from copy import deepcopy
import io
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ema import ModelEMA

torch.set_num_threads(2)


class ToyState(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, -2.0]))
        self.register_buffer("running", torch.tensor([2.0]))
        self.register_buffer("counter", torch.tensor(0, dtype=torch.long))


@pytest.mark.parametrize("decay", [0.995, 0.1, 0.0])
def test_adaptive_decay_matches_hand_calculation_and_integer_buffers_copy(decay):
    raw = ToyState()
    ema = ModelEMA(raw, decay=decay)
    initial_weight = raw.weight.detach().clone()
    initial_running = raw.running.clone()
    with torch.no_grad():
        raw.weight.copy_(torch.tensor([3.0, 4.0]))
        raw.running.fill_(8.0)
        raw.counter.fill_(7)
    ema.update(raw)
    d1 = min(decay, 2.0 / 11.0)
    expected_weight = initial_weight * d1 + torch.tensor([3.0, 4.0]) * (1.0 - d1)
    expected_running = initial_running * d1 + 8.0 * (1.0 - d1)
    torch.testing.assert_close(ema.model.weight, expected_weight)
    torch.testing.assert_close(ema.model.running, expected_running)
    assert int(ema.model.counter) == 7
    assert ema.num_updates == 1

    with torch.no_grad():
        raw.weight.copy_(torch.tensor([5.0, -4.0]))
        raw.running.fill_(-2.0)
        raw.counter.fill_(9)
    ema.update(raw)
    d2 = min(decay, 3.0 / 12.0)
    expected_weight = expected_weight * d2 + torch.tensor([5.0, -4.0]) * (1.0 - d2)
    expected_running = expected_running * d2 - 2.0 * (1.0 - d2)
    torch.testing.assert_close(ema.model.weight, expected_weight)
    torch.testing.assert_close(ema.model.running, expected_running)
    assert int(ema.model.counter) == 9
    assert ema.model.counter.dtype == torch.long
    assert ema.num_updates == 2


def test_shadow_is_independent_float32_frozen_and_eval():
    raw = ToyState().half().train()
    ema = ModelEMA(raw)
    assert not ema.model.training
    assert ema.model.weight.dtype == torch.float32
    assert ema.model.running.dtype == torch.float32
    assert ema.model.counter.dtype == torch.long
    for name, raw_value in raw.state_dict().items():
        assert raw_value.data_ptr() != ema.model.state_dict()[name].data_ptr()
    before = deepcopy(ema.model.state_dict())
    with torch.no_grad():
        raw.weight.add_(2.0)
        raw.running.add_(3.0)
        raw.counter.add_(4)
    for name, value in before.items():
        torch.testing.assert_close(ema.model.state_dict()[name], value)
    raw.weight.square().sum().backward()
    assert raw.weight.grad is not None
    ema.update(raw)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in ema.model.parameters())
    assert not ema.model.training
    assert ema.model.weight.dtype == torch.float32


def test_checkpoint_records_algorithm_and_rejects_incompatible_restore():
    raw = ToyState()
    ema = ModelEMA(raw, 0.995)
    ema.update(raw)
    state = deepcopy(ema.state_dict())
    assert {"model", "num_updates", "decay", "algorithm"} <= state.keys()
    assert state["algorithm"] == "adaptive_decay_v1"
    with pytest.raises(ValueError, match="decay"):
        ModelEMA(raw, 0.9).load_state_dict(state)
    incompatible = deepcopy(state)
    incompatible["algorithm"] = "some_other_ema_algorithm"
    with pytest.raises(ValueError, match="algorithm"):
        ModelEMA(raw, 0.995).load_state_dict(incompatible)


def _new_training_state():
    raw = nn.Linear(2, 1)
    with torch.no_grad():
        raw.weight.copy_(torch.tensor([[0.2, -0.4]]))
        raw.bias.fill_(0.1)
    optimizer = torch.optim.SGD(raw.parameters(), lr=0.05, momentum=0.9, weight_decay=0.01)
    ema = ModelEMA(raw, decay=0.995)
    return raw, optimizer, ema


def _train_steps(raw, optimizer, ema, start, stop):
    # Fixed synthetic examples let restoration be checked exactly on CPU.
    base_x = torch.tensor([[0.1, -0.7], [0.5, 0.9], [-0.3, 0.2]])
    for step in range(start, stop):
        x = base_x + step * 0.025
        target = 0.6 * x[:, :1] - 0.2 * x[:, 1:] + 0.1
        optimizer.zero_grad(set_to_none=True)
        loss = (raw(x) - target).square().mean()
        loss.backward()
        optimizer.step()
        ema.update(raw)


def test_interrupted_raw_optimizer_and_ema_restore_matches_continuous():
    continuous_raw, continuous_optimizer, continuous_ema = _new_training_state()
    _train_steps(continuous_raw, continuous_optimizer, continuous_ema, 0, 9)

    interrupted_raw, interrupted_optimizer, interrupted_ema = _new_training_state()
    _train_steps(interrupted_raw, interrupted_optimizer, interrupted_ema, 0, 4)
    buffer = io.BytesIO()
    torch.save({"model": interrupted_raw.state_dict(),
                "optimizer": interrupted_optimizer.state_dict(),
                "ema": interrupted_ema.state_dict(), "step": 4}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
    resumed_raw, resumed_optimizer, resumed_ema = _new_training_state()
    resumed_raw.load_state_dict(checkpoint["model"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    resumed_ema.load_state_dict(checkpoint["ema"])
    _train_steps(resumed_raw, resumed_optimizer, resumed_ema, checkpoint["step"], 9)

    for name, expected in continuous_raw.state_dict().items():
        torch.testing.assert_close(resumed_raw.state_dict()[name], expected, rtol=0, atol=0)
    for name, expected in continuous_ema.model.state_dict().items():
        torch.testing.assert_close(resumed_ema.model.state_dict()[name], expected, rtol=0, atol=0)
    expected_optimizer = continuous_optimizer.state_dict()
    actual_optimizer = resumed_optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    for parameter_id, parameter_state in expected_optimizer["state"].items():
        for name, expected in parameter_state.items():
            torch.testing.assert_close(actual_optimizer["state"][parameter_id][name], expected, rtol=0, atol=0)
    assert resumed_ema.num_updates == continuous_ema.num_updates == 9
    assert resumed_ema.state_dict()["algorithm"] == continuous_ema.state_dict()["algorithm"]
    assert not resumed_ema.model.training
    assert all(not parameter.requires_grad for parameter in resumed_ema.model.parameters())
