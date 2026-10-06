"""Protocol checks use synthetic data, never train or tune on the public test set."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from common.src.utils import load_config, save_json
from temporal_ablation.src.data import (CHANNELS, digest, make_windows, prepare,
                                       resample_segments, select_history, validate_splits)
from temporal_ablation.src.experiment import evaluate_one, train_one, validate_config
from temporal_ablation.src.metrics import fit_temperature, metrics, nll
from temporal_ablation.src.models import SlipModel
from temporal_ablation.src.report import write_report
from temporal_ablation.src.verify import verify


@pytest.fixture
def config(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(4)
    names = ["train_a.csv", "train_b.csv", "cal.csv", "test.csv"]
    entries = []
    for i, name in enumerate(names):
        frame = pd.DataFrame(rng.normal(i, 1, (160, 15)), columns=CHANNELS)
        frame["log_time"] = pd.date_range("2026-01-01", periods=len(frame), freq="10ms")
        frame["slip"] = (np.arange(len(frame)) // 10) % 2
        frame.to_csv(raw / name, index=False)
        entries.append({"file": name, "sha256": digest(raw / name)})
    manifest = tmp_path / "source.json"
    save_json(manifest, {"revision": "synthetic-fixture", "files": entries})
    result = load_config("common/configs/temporal_ablation.yaml")
    result.update(raw_dir=str(raw), source_manifest=str(manifest), prepared_dir=str(tmp_path / "prepared"),
                  run_dir=str(tmp_path / "runs"), results_dir=str(tmp_path / "results"),
                  splits={"train": names[:2], "calibration": names[2:3], "test": names[3:]},
                  epochs=2, workers=1, hidden_size=8)
    return result


def test_splits_disjoint_and_mixed_recording_rejected():
    validate_splits({"train": ["a.csv"], "calibration": ["b.csv"], "test": ["c.csv"]})
    with pytest.raises(ValueError):
        validate_splits({"train": ["a.csv"], "calibration": ["b.csv"], "test": ["a.csv"]})
    with pytest.raises(ValueError):
        validate_splits({"train": ["a.csv"], "calibration": ["b.csv"], "test": ["no_slip.csv"]})


def test_causal_resampling_and_gap_boundaries():
    times = np.array([0., .019, .04, 1., 1.019, 1.04])
    values = np.repeat(np.arange(6)[:, None], 15, axis=1)
    segments = resample_segments(times, values, np.arange(6))
    assert len(segments) == 2
    assert segments[0][0][1, 0] == 0  # 10 ms must not read the 19 ms observation
    x, y, ids = make_windows(segments, 3, 1, "example")
    assert all(np.ptp(row[:, 0]) <= 2 for row in x)
    assert set(s.split(":")[1] for s in ids) == {"0", "1"}


def test_same_endpoints_labels_and_train_only_scaler(config, tmp_path):
    audit = prepare(config)
    with np.load(tmp_path / "prepared/train.npz") as data:
        full, short = select_history(data["x"], 50), select_history(data["x"], 1)
        np.testing.assert_array_equal(full[:, -1], short[:, 0])
        assert len(short) == len(full) == len(data["y"]) == len(set(data["endpoint_id"]))
    with np.load(tmp_path / "prepared/scaler.npz") as data:
        old_mean, old_std = data["mean"].copy(), data["std"].copy()
    for name in ["cal.csv", "test.csv"]:
        path = tmp_path / "raw" / name
        frame = pd.read_csv(path)
        frame[CHANNELS] += 1000
        frame.to_csv(path, index=False)
    source_path = tmp_path / "source.json"
    source = json.loads(source_path.read_text())
    for entry in source["files"]:
        entry["sha256"] = digest(tmp_path / "raw" / entry["file"])
    save_json(source_path, source)
    second = {**config, "prepared_dir": str(tmp_path / "prepared2")}
    prepare(second)
    with np.load(tmp_path / "prepared2/scaler.npz") as data:
        np.testing.assert_array_equal(data["mean"], old_mean)
        np.testing.assert_array_equal(data["std"], old_std)
    assert audit["scaler_fit_on"] == "unique resampled TRAIN rows only"


def test_prepared_integrity_and_protocol_guard(config, tmp_path):
    original = prepare(config)
    assert prepare(config)["preprocessing_signature"] == original["preprocessing_signature"]
    with pytest.raises(ValueError, match="another protocol"):
        prepare({**config, "stride": 10})
    (tmp_path / "prepared/test.npz").write_bytes(b"tampered fixture")
    with pytest.raises(ValueError, match="Prepared file changed"):
        prepare(config)


def test_temperature_preserves_ranking_and_fixed_threshold():
    y = np.array([0, 0, 1, 0, 1, 1])
    z = np.array([-8., -6., -4., 2., 4., 6.])
    temperature = fit_temperature(y, z)
    assert nll(y, z / temperature) <= nll(y, z) + 1e-10
    original, scaled = metrics(y, z), metrics(y, z / temperature)
    for key in ["accuracy", "f1", "ap", "roc_auc", "tn", "fp", "fn", "tp"]:
        assert original[key] == scaled[key]
    assert fit_temperature(y, z) == temperature


def test_fixed_epoch_training_does_not_need_calibration_or_test(config, tmp_path):
    prepare(config)
    (tmp_path / "prepared/calibration.npz").unlink()
    (tmp_path / "prepared/test.npz").unlink()
    train_one(config, "lstm", 1)
    run = tmp_path / "runs/lstm_t1"
    checkpoint = torch.load(run / "final.pt", weights_only=True)
    assert checkpoint["epoch"] == 2
    summary = json.loads((run / "training_summary.json").read_text())
    assert summary["evaluations_during_training"] == 0
    assert summary["selected_epoch"] == 2
    assert list(pd.read_csv(run / "history.csv").columns) == ["epoch", "train_nll", "elapsed_seconds"]
    with pytest.raises(FileExistsError):
        train_one(config, "lstm", 1)


def test_all_four_models_calibration_and_report(config, tmp_path):
    validate_config(config)
    audit = prepare(config)
    rows, objects = [], []
    for architecture in ["lstm", "gru"]:
        hashes = []
        for steps in [1, 50]:
            run = train_one(config, architecture, steps)
            metadata = json.loads((tmp_path / "runs" / f"{architecture}_t{steps}" / "config.json").read_text())
            hashes.append(metadata["initial_state_sha256"])
        assert hashes[0] == hashes[1]
    for architecture in ["lstm", "gru"]:
        for steps in [1, 50]:
            results, per_object = evaluate_one(config, architecture, steps)
            rows.extend(results)
            objects.extend(per_object)
            run = tmp_path / "runs" / f"{architecture}_t{steps}"
            calibration = json.loads((run / "temperature.json").read_text())
            with np.load(run / "predictions.npz") as data:
                assert calibration["temperature"] == fit_temperature(data["calibration_labels"], data["calibration_logits"])
            assert calibration["fit_split"] == "calibration"
            assert calibration["nll_after"] <= calibration["nll_before"] + 1e-6
            assert results[0]["f1"] == results[1]["f1"]
    save_json(tmp_path / "runs/protocol.json", config)
    write_report(config, audit, rows, objects)
    assert len(pd.read_csv(tmp_path / "results/metrics.csv")) == 8
    assert "SMOKE TEST" in (tmp_path / "results/REPORT.md").read_text("utf-8")
    assert verify(config)["models_verified"] == 4


@pytest.mark.parametrize("architecture,parameters", [("lstm", 90881), ("gru", 72321)])
def test_parameter_count_and_shapes(architecture, parameters):
    model = SlipModel(architecture)
    assert sum(p.numel() for p in model.parameters()) == parameters
    assert model(torch.zeros(2, 50, 15)).shape == (2,)
    assert model(torch.zeros(2, 1, 15)).shape == (2,)


def test_configuration_rejects_other_factors(config):
    invalid = copy.deepcopy(config)
    invalid["threshold"] = .3
    with pytest.raises(ValueError):
        validate_config(invalid)
