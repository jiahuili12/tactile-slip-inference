"""Protocol tests use synthetic fixtures, never claimed as research results."""
import numpy as np
import json
import pytest
import torch
import pollen_reproduction.src.reproduce_pollen as reproduction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pollen_reproduction.src.pollen_data import prepare_arrays
from pollen_reproduction.src.pollen_model import PollenLSTM
from pollen_reproduction.src.pollen_report import score_predictions, write_report
from pollen_reproduction.src.reproduce_pollen import train_reproduction


def test_pollen_preprocessing_reproduces_row_split_and_all_data_scaler():
    features = np.arange(600, dtype=np.float64).reshape(40, 15)
    labels = np.tile([0, 1], 20)
    x, y, train, selection, scaler = prepare_arrays(features, labels)
    expected_x = StandardScaler().fit_transform(features).astype(np.float32)
    tx, sx, ty, sy = train_test_split(expected_x, labels, test_size=.2, random_state=42)
    assert x.shape == (40, 1, 15)
    np.testing.assert_array_equal(x[train, 0], tx)
    np.testing.assert_array_equal(x[selection, 0], sx)
    np.testing.assert_array_equal(y[train, 0], ty)
    np.testing.assert_array_equal(y[selection, 0], sy)
    np.testing.assert_allclose(scaler.mean_, features.mean(axis=0))
    assert not set(train) & set(selection)


def test_pollen_model_matches_explicit_upstream_forward():
    torch.set_num_threads(1)
    torch.manual_seed(2)
    model = PollenLSTM().eval()
    x = torch.randn(5, 1, 15)
    assert sum(p.numel() for p in model.parameters()) == 90881
    with torch.inference_mode():
        recurrent, _ = model.lstm(x, (torch.zeros(1, 5, 128), torch.zeros(1, 5, 128)))
        expected = model.out(model.fc(recurrent[:, -1, :]))
        torch.testing.assert_close(model(x), expected)
        torch.testing.assert_close(model(x)[:1], model(x[:1]))


def test_pollen_uses_strict_threshold_at_zero_logit():
    metrics = score_predictions(np.array([0, 1]), np.array([0., 1.]))
    assert metrics["accuracy"] == 1.0
    assert metrics["tn"] == metrics["tp"] == 1


def test_pollen_two_phases_and_saved_prediction_consistency(tmp_path):
    rng = np.random.default_rng(4)
    x, y, train, selection, _ = prepare_arrays(rng.normal(size=(100, 15)), np.tile([0, 1], 50))
    config, summary, logits = train_reproduction(
        tmp_path, x, y, train, selection, seed=7, epochs_per_phase=1)
    assert not config["full_epoch_budget"]
    assert [record["global_epoch"] for record in summary["phase_bests"]] == [1, 2]
    saved = np.load(tmp_path / "selection_predictions.npz")
    np.testing.assert_array_equal(saved["row_id"], selection)
    np.testing.assert_array_equal(saved["logits"], logits)
    assert (tmp_path / "phase1_best.pt").is_file()
    assert (tmp_path / "phase2_best.pt").is_file()
    audit = {"revision": "synthetic test fixture, NOT research data", "rows": 100,
             "train_rows": len(train), "selection_rows": len(selection)}
    results = tmp_path / "results"
    write_report(results, tmp_path, config, summary, audit,
                 y[selection].reshape(-1), logits, np.full(len(selection), "synthetic_fixture"))
    assert "SMOKE TEST ONLY" in (results / "REPORT.md").read_text("utf-8")
    assert (results / "training.png").is_file()
    metrics = json.loads((results / "metrics.json").read_text("utf-8"))
    assert metrics["rows"] == len(selection)
    assert metrics["independent_test"] is False


def test_phase2_selects_its_own_best_even_when_phase1_is_better(tmp_path, monkeypatch):
    rng = np.random.default_rng(4)
    x, y, train, selection, _ = prepare_arrays(rng.normal(size=(100, 15)), np.tile([0, 1], 50))
    scores = iter([.9, .8, .5, .4])
    monkeypatch.setattr(reproduction, "upstream_accuracy", lambda *args: next(scores))
    _, summary, _ = train_reproduction(tmp_path, x, y, train, selection, epochs_per_phase=2)
    assert [row["global_epoch"] for row in summary["phase_bests"]] == [1, 3]
    saved = torch.load(tmp_path / "phase2_best.pt", weights_only=True)
    assert saved["selection"]["selection_accuracy"] == .5


def test_refuses_to_overwrite_existing_run_before_loading_data(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["reproduce_pollen", "--run-dir", str(tmp_path),
                                     "--results-dir", str(tmp_path / "new_results")])
    with pytest.raises(SystemExit) as error:
        reproduction.main()
    assert error.value.code == 2
    assert not (tmp_path / "new_results").exists()
