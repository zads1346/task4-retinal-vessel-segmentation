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
        config = {"version": "v3.0.0", "encoder_name": "resnet18",
                  "fold": fold, "patch_size": 16,
                  "pretrained_weights": True, "external_fine_tuning_data": False,
                  **split}
        states[str(checkpoint.resolve())] = {"config": config, "model": {}}
        selected_models.append({"family": "resnet18", "fold": fold,
                                "checkpoint": str(checkpoint),
                                "checkpoint_sha256": original_sha(checkpoint),
                                "test_prediction_weight": 0.2, **split})
    selection = {"version": "v3.0.0", "selected_variant": "resnet18_no_tta",
                 "family_weights": {"resnet18": 1.0}, "tta": False,
                 "threshold": 0.5, "models": selected_models}
    selection_path = tmp_path / "selection.json"
    output = tmp_path / "prediction"
    data = tmp_path / "data"
    data.mkdir()
    architecture_digest = {"value": "architecture-version-a"}
    calls = {"predict": 0}

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def to(self, *args, **kwargs):
            return self

        def load_state_dict(self, *args, **kwargs):
            pass

    def fake_prediction(model, image, device, *args, **kwargs):
        calls["predict"] += 1
        return np.full(image.shape[:2], 0.4, np.float32)

    def fake_sha(path):
        # Model code changes are represented by a digest, without modifying a
        # source file that may be used by the currently running training job.
        if Path(path).name == "pretrained_model.py":
            return architecture_digest["value"]
        return original_sha(path)

    monkeypatch.setattr(predictor.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(predictor.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(predictor.torch, "load", lambda path, **kwargs: deepcopy(states[str(Path(path).resolve())]))
    monkeypatch.setattr(predictor, "PretrainedUNet", FakeModel)
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
            "architecture_digest": architecture_digest}


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
