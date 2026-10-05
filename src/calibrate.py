"""Fit positive scalar temperature using only the calibration split."""
import argparse
import hashlib

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from .common import project_path, save_json
from .dataset import SlipDataset
from .models import load_model
from .train import predict


def nll(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    return float(np.mean(np.logaddexp(0, logits) - np.asarray(labels) * logits))


def fit_temperature(logits, labels):
    if np.unique(labels).size != 2:
        raise ValueError("Calibration split must contain both classes")
    result = minimize_scalar(lambda log_t: nll(logits / np.exp(log_t), labels),
                             bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    candidates = [1.0, 0.05, 20.0, float(np.exp(result.x))]
    return min(candidates, key=lambda t: nll(logits / t, labels))


def calibrate(run):
    run = project_path(run)
    model, saved = load_model(run / "best.pt")
    torch.set_num_threads(saved["config"].get("threads", 2))
    processed = project_path(saved["config"]["processed_dir"])
    if hashlib.sha256((processed / "report.json").read_bytes()).hexdigest() != saved["data_fingerprint"]:
        raise ValueError("Prepared dataset has changed since training")
    dataset = SlipDataset(processed / "calibration.npz")
    logits, labels = predict(model, dataset), dataset.y.numpy()
    temperature = fit_temperature(logits, labels)
    summary = {"temperature": temperature, "fit_split": "calibration",
               "calibration_nll_before": nll(logits, labels),
               "calibration_nll_after": nll(logits / temperature, labels),
               "bounds": [0.05, 20.0], "windows": len(labels)}
    save_json(run / "temperature.json", summary)
    print(f"{run.name}: T={temperature:.3f}, calibration NLL {summary['calibration_nll_before']:.4f} -> {summary['calibration_nll_after']:.4f}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    calibrate(parser.parse_args().run)
