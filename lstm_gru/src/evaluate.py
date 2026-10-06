"""Evaluate the fixed test set; select any decision threshold on validation only."""
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from common.src.utils import project_path, save_json
from .dataset import SlipDataset
from .metrics import classification_metrics, validation_f1_threshold
from .models import load_model
from .train import predict


@torch.inference_mode()
def latency_ms(model, sample, repeats=200):
    for _ in range(20):
        model(sample)
    durations = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        model(sample)
        durations.append((time.perf_counter_ns() - started) / 1e6)
    return {"latency_median_ms": float(np.median(durations)),
            "latency_p95_ms": float(np.percentile(durations, 95)),
            "latency_repeats": repeats, "latency_batch_size": 1}


def evaluate(run):
    run = project_path(run)
    model, saved = load_model(run / "best.pt")
    config = saved["config"]
    torch.set_num_threads(config.get("threads", 2))
    processed = project_path(config["processed_dir"])
    if hashlib.sha256((processed / "report.json").read_bytes()).hexdigest() != saved["data_fingerprint"]:
        raise ValueError("Prepared dataset has changed since training")
    temperature = json.loads((run / "temperature.json").read_text(encoding="utf-8"))["temperature"]
    test = SlipDataset(processed / "test.npz")
    validation = SlipDataset(processed / "validation.npz")
    logits, val_logits = predict(model, test).astype(float), predict(model, validation).astype(float)
    labels, val_labels = test.y.numpy(), validation.y.numpy()
    latency = latency_ms(model, test.x[:1])
    timing = json.loads((run / "training_summary.json").read_text(encoding="utf-8"))
    rows, per_object = [], []
    predictions = pd.DataFrame({"object_id": test.object_ids, "label": labels.astype(int),
                                "logit": logits, "probability_raw": expit(logits),
                                "probability_temperature": expit(logits / temperature)})
    with np.load(processed / "test.npz", allow_pickle=False) as arrays:
        predictions["time_seconds"] = arrays["time"]
        predictions["segment_id"] = arrays["segment_id"]
    predictions.to_csv(run / "test_predictions.csv", index=False)
    for method, scale in (("raw", 1.0), ("temperature", temperature)):
        test_scaled, val_scaled = logits / scale, val_logits / scale
        threshold = validation_f1_threshold(val_labels, val_scaled)
        metrics = classification_metrics(labels, test_scaled)
        tuned = classification_metrics(labels, test_scaled, threshold)
        row = {"run": run.name, "model": config["model"], "seed": config["seed"], "method": method,
               "temperature": scale, **metrics,
               "validation_selected_threshold": threshold,
               "f1_at_val_threshold": tuned["f1"], "recall_at_val_threshold": tuned["recall"],
               "precision_at_val_threshold": tuned["precision"],
               "parameters": timing["parameters"], "best_epoch": saved["epoch"],
               "training_seconds": timing["training_seconds"], **latency}
        rows.append(row)
        for object_id in np.unique(test.object_ids):
            ids = test.object_ids == object_id
            per_object.append({"model": config["model"], "seed": config["seed"], "method": method,
                               "object_id": object_id,
                               **classification_metrics(labels[ids], test_scaled[ids])})
        print(f"{run.name} {method}: F1={metrics['f1']:.3f}, AP={metrics['pr_auc_ap']:.3f}, ECE={metrics['ece']:.3f}, NLL={metrics['nll']:.3f}", flush=True)
    save_json(run / "evaluation.json", rows)
    pd.DataFrame(per_object).to_csv(run / "per_object.csv", index=False)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    evaluate(parser.parse_args().run)
