"""Mocked CPU regression tests for ensemble checkpoint and cache contracts.

No CUDA kernels, real neural networks, pretrained files, or external data are
used. These tests execute the prediction orchestration around synthetic maps.
"""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import make_folds
import predict_ensemble as predictor


@pytest.fixture
def prediction_setup(tmp_path, monkeypatch):
    cases = {str(i): {"image": np.full((2, 3, 3), i / 40, np.float32)}
             for i in range(1, 21)}
    states = {}
    selected_models = []
    folds = make_folds(cases)
    original_sha = predictor.sha
    for fold, split in enumerate(folds):
        checkpoint = tmp_path / "checkpoints" / f"fold{fold}" / "best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"mock checkpoint {fold}".encode())
        config = {"version": "v4.0.0", "encoder_name": "unet_v4",
                  "fold": fold, "patch_size": 16, "base_channels": 4,
                  "pretrained_weights": False, "external_training_data": False,
                  "initialization": "random_from_scratch",
                  **split}
        states[str(checkpoint.resolve())] = {"config": config, "model": {}}
        selected_models.append({"family": "unet_v4", "fold": fold,
                                "checkpoint": str(checkpoint),
                                "checkpoint_sha256": original_sha(checkpoint),
                                "test_prediction_weight": 0.2, **split})
    selection = {"version": "v4.0.0", "selected_variant": "none",
                 "family_weights": {"unet_v4": 1.0}, "tta": False, "tta_mode": "none",
                 "threshold": 0.5, "models": selected_models}
    selection_path = tmp_path / "selection.json"
    output = tmp_path / "prediction"
    data = tmp_path / "data"
    data.mkdir()
    architecture_digest = {"value": "architecture-version-a"}
    calls = {"predict": 0, "modes": []}

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def to(self, *args, **kwargs):
            return self

        def load_state_dict(self, *args, **kwargs):
            pass

    def fake_prediction(model, image, device, *args, **kwargs):
        calls["predict"] += 1
        calls["modes"].append(kwargs.get("tta"))
        return np.full(image.shape[:2], 0.4, np.float32)

    def fake_sha(path):
        # Model code changes are represented by a digest, without modifying a
        # source file that may be used by the currently running training job.
        if Path(path).name == "model.py":
            return architecture_digest["value"]
        return original_sha(path)

    monkeypatch.setattr(predictor.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(predictor.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(predictor.torch, "load", lambda path, **kwargs: deepcopy(states[str(Path(path).resolve())]))
    monkeypatch.setattr(predictor, "UNet", FakeModel)
    monkeypatch.setattr(predictor, "predict_image", fake_prediction)
    monkeypatch.setattr(predictor, "load_cases", lambda *args, **kwargs: cases)
    monkeypatch.setattr(predictor, "export_submission", lambda probabilities, *args, **kwargs: {"rows": len(probabilities)})
    monkeypatch.setattr(predictor, "sha", fake_sha)

    def run(payload=None):
        selection_path.write_text(json.dumps(selection if payload is None else payload), encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["predict_ensemble.py", "--data", str(data),
                                           "--selection", str(selection_path), "--output", str(output)])
        predictor.main()

    return {"run": run, "selection": selection, "calls": calls,
            "architecture_digest": architecture_digest, "states": states}


def test_model_code_change_invalidates_otherwise_matching_probability_cache(prediction_setup):
    fixture = prediction_setup
    fixture["run"]()
    assert fixture["calls"]["predict"] == 5 * 20
    fixture["run"]()
    assert fixture["calls"]["predict"] == 5 * 20
    fixture["architecture_digest"]["value"] = "architecture-version-b"
    fixture["run"]()
    assert fixture["calls"]["predict"] == 2 * 5 * 20


def test_relabeling_a_duplicate_checkpoint_as_another_fold_is_rejected(prediction_setup):
    fixture = prediction_setup
    selection = deepcopy(fixture["selection"])
    # Five fold numbers and total weights still look correct, but fold 1 now
    # supplies the fold-0 checkpoint and fold-0 validation split a second time.
    original = selection["models"][0]
    duplicate = selection["models"][1]
    for key in ("checkpoint", "checkpoint_sha256", "train_ids", "val_ids"):
        duplicate[key] = deepcopy(original[key])
    with pytest.raises(ValueError):
        fixture["run"](selection)


def test_cached_predictions_cannot_bypass_checkpoint_split_validation(prediction_setup):
    fixture = prediction_setup
    fixture["run"]()
    selection = deepcopy(fixture["selection"])
    # Keep a complete disjoint partition but associate two splits with the
    # wrong unchanged checkpoints. Cache validity must not excuse this mismatch.
    first, second = selection["models"][:2]
    for key in ("train_ids", "val_ids"):
        first[key], second[key] = deepcopy(second[key]), deepcopy(first[key])
    with pytest.raises(ValueError):
        fixture["run"](selection)


def test_flip4_to_d4_change_invalidates_cache_and_reaches_inference(prediction_setup):
    fixture = prediction_setup
    selection = deepcopy(fixture["selection"])
    selection.update(tta=True, tta_mode="flip4", selected_variant="flip4")
    fixture["run"](selection)
    assert fixture["calls"]["predict"] == 100
    assert set(fixture["calls"]["modes"]) == {"flip4"}
    selection.update(tta_mode="d4", selected_variant="d4")
    fixture["run"](selection)
    assert fixture["calls"]["predict"] == 200
    assert fixture["calls"]["modes"][100:] == ["d4"] * 100


def test_inconsistent_boolean_and_explicit_tta_mode_is_rejected(prediction_setup):
    selection = deepcopy(prediction_setup["selection"])
    selection.update(tta=False, tta_mode="d4")
    with pytest.raises(ValueError, match="TTA"):
        prediction_setup["run"](selection)


def test_cached_predictions_do_not_bypass_scratch_provenance(prediction_setup):
    fixture = prediction_setup
    fixture["run"]()
    first_checkpoint = fixture["selection"]["models"][0]["checkpoint"]
    config = fixture["states"][str(Path(first_checkpoint).resolve())]["config"]
    config["initialization"] = "old_checkpoint_warm_start"
    with pytest.raises(ValueError, match="provenance"):
        fixture["run"]()
