"""Read-only audit of completed runs; does not train, select or update models."""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from common.src.utils import load_config, project_path
from .data import digest, validate_splits
from .experiment import predict
from .metrics import fit_temperature, metrics
from .models import SlipModel


def verify(config):
    prepared, runs = project_path(config["prepared_dir"]), project_path(config["run_dir"])
    results = project_path(config["results_dir"])
    validate_splits(config["splits"])
    torch.set_num_threads(config["threads_per_worker"])
    audit = json.loads((prepared / "report.json").read_text("utf-8"))
    for filename, checksum in audit["prepared_sha256"].items():
        assert digest(prepared / filename) == checksum, filename
    scores = pd.read_csv(results / "metrics.csv")
    run_audit = json.loads((results / "run_audit.json").read_text("utf-8"))
    objects_by_split = {}
    for split in ["train", "calibration", "test"]:
        with np.load(prepared / f"{split}.npz") as arrays:
            objects_by_split[split] = set(arrays["object_id"])
            expected = {name.removesuffix(".csv") for name in config["splits"][split]}
            assert objects_by_split[split] == expected
            assert len(set(arrays["endpoint_id"])) == len(arrays["y"])
    assert not objects_by_split["train"] & objects_by_split["test"]
    assert not objects_by_split["calibration"] & objects_by_split["test"]
    with np.load(prepared / "test.npz") as arrays:
        test_x, test_y, endpoints = arrays["x"], arrays["y"], arrays["endpoint_id"]
    with np.load(prepared / "calibration.npz") as arrays:
        cal_x, cal_y = arrays["x"], arrays["y"]
    for architecture in config["architectures"]:
        hashes = []
        for steps in config["sequence_lengths"]:
            trial = f"{architecture}_t{steps}"
            run = runs / trial
            saved = torch.load(run / "final.pt", map_location="cpu", weights_only=True)
            assert saved["epoch"] == config["epochs"]
            assert digest(run / "final.pt") == run_audit[trial]["final_checkpoint_sha256"]
            assert digest(run / "predictions.npz") == run_audit[trial]["predictions_sha256"]
            assert saved["config"]["train_data_sha256"] == digest(prepared / "train.npz")
            assert saved["config"]["data_report_sha256"] == digest(prepared / "report.json")
            hashes.append(saved["config"]["initial_state_sha256"])
            history = pd.read_csv(run / "history.csv")
            assert list(history.epoch) == list(range(1, config["epochs"] + 1))
            assert np.isfinite(history.train_nll).all()
            assert list(history.columns) == ["epoch", "train_nll", "elapsed_seconds"]
            model = SlipModel(architecture, config["hidden_size"])
            model.load_state_dict(saved["state_dict"])
            with np.load(run / "predictions.npz") as values:
                np.testing.assert_array_equal(values["test_endpoint_id"], endpoints)
                np.testing.assert_array_equal(values["test_labels"], test_y)
                np.testing.assert_array_equal(values["calibration_labels"], cal_y)
                np.testing.assert_allclose(predict(model, test_x, steps), values["test_logits"], atol=1e-6)
                np.testing.assert_allclose(predict(model, cal_x, steps), values["calibration_logits"], atol=1e-6)
                temperature = fit_temperature(cal_y, values["calibration_logits"], config["temperature_bounds"])
                for method, t in [("raw", 1.), ("temperature", temperature)]:
                    row = scores[(scores.architecture == architecture) & (scores.steps == steps) & (scores.method == method)]
                    assert len(row) == 1
                    row = row.iloc[0]
                    np.testing.assert_allclose(row.temperature, t)
                    recomputed = metrics(test_y, values["test_logits"].astype(float) / t)
                    for key, expected in recomputed.items():
                        np.testing.assert_allclose(row[key], expected, rtol=1e-8, atol=1e-10)
        assert hashes[0] == hashes[1], "History-length variants must have matching initial weights"
    return {"models_verified": 4, "epochs_each": config["epochs"],
            "shared_test_endpoints": len(test_y), "metrics_rows_verified": len(scores)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="common/configs/temporal_ablation.yaml")
    parser.add_argument("--run-dir")
    parser.add_argument("--results-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ["run_dir", "results_dir"]:
        value = getattr(args, key)
        if value:
            config[key] = value
    print(json.dumps(verify(config), indent=2))


if __name__ == "__main__":
    main()
