import json

import numpy as np
import pytest
import torch

from lstm_gru.src.calibrate import fit_temperature, nll, calibrate
from common.src.utils import save_json
from lstm_gru.src.evaluate import evaluate
from lstm_gru.src.metrics import classification_metrics
from lstm_gru.src.models import SlipRNN
from lstm_gru.src.prepare_data import causal_resample, validate_splits, window_segments
from lstm_gru.src.train import train


def test_split_leakage_rejected():
    with pytest.raises(ValueError, match="multiple splits"):
        validate_splits({"train": ["a.csv"], "validation": ["a.csv"],
                         "calibration": ["b.csv"], "test": ["c.csv"]})


def test_causal_resampling_and_gap_isolation():
    times = np.array([0, .009, .019, .029, 1, 1.009, 1.019])
    values = np.repeat(times[:, None], 15, axis=1)
    segments = causal_resample(times, values, np.arange(len(times)) % 2)
    assert len(segments) == 2
    for x, _, grid in segments:
        assert np.all(x[:, 0] <= grid + 1e-12)
    x, y, t, segment_ids = window_segments(segments, 2, 1)
    assert len(x) == len(y) == len(t) == len(segment_ids)
    assert np.all(np.diff(x[:, :, 0], axis=1) < .1)


@pytest.mark.parametrize("architecture", ["lstm", "gru"])
def test_model_uses_no_other_batch_samples(architecture):
    torch.manual_seed(0)
    model = SlipRNN(architecture, 8).eval()
    x = torch.randn(3, 10, 15)
    with torch.inference_mode():
        assert torch.allclose(model(x)[:1], model(x[:1]), atol=1e-6)


def test_temperature_preserves_decisions_and_reduces_calibration_nll():
    logits = np.array([8, 6, -8, -6, 5, -5], dtype=float)
    labels = np.array([1, 0, 0, 1, 1, 0])
    temperature = fit_temperature(logits, labels)
    assert temperature > 0
    assert nll(logits / temperature, labels) <= nll(logits, labels) + 1e-9
    before, after = classification_metrics(labels, logits), classification_metrics(labels, logits / temperature)
    assert before["f1"] == after["f1"]
    assert before["roc_auc"] == after["roc_auc"]
    assert before["pr_auc_ap"] == after["pr_auc_ap"]


def test_full_training_calibration_evaluation(tmp_path):
    rng = np.random.default_rng(9)
    processed = tmp_path / "prepared"
    processed.mkdir()
    for split in ["train", "validation", "calibration", "test"]:
        x = rng.normal(size=(40, 6, 15)).astype(np.float32)
        y = np.tile([0, 1], 20)
        x[y == 1, :, 0] += 2
        np.savez(processed / f"{split}.npz", x=x, y=y, object_id=np.full(40, split),
                 time=np.arange(40), segment_id=np.zeros(40, dtype=int))
    save_json(processed / "report.json", {"test": "synthetic integration fixture; not research evidence"})
    config = {"model": "gru", "hidden_size": 8, "num_layers": 1, "batch_size": 16,
              "epochs": 2, "patience": 2, "learning_rate": .01, "threads": 1,
              "processed_dir": str(processed), "output_dir": str(tmp_path / "runs")}
    output = train(config, 7)
    calibrate(output)
    rows = evaluate(output)
    assert len(rows) == 2
    assert rows[0]["windows"] == 40
    assert (output / "test_predictions.csv").is_file()
    # Dataset provenance is checked before evaluation, preventing accidental stale checkpoints.
    save_json(processed / "report.json", {"changed": True})
    with pytest.raises(ValueError, match="changed"):
        evaluate(output)
