"""Train all four fixed-epoch models BEFORE calibrating or scoring any of them."""
import argparse
import csv
import hashlib
import json
import multiprocessing
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from torch.utils.data import DataLoader, TensorDataset

from common.src.utils import load_config, project_path, save_json
from .data import digest, prepare, select_history
from .metrics import fit_temperature, metrics, nll
from .models import SlipModel


def train_one(config, architecture, length):
    """Only train.npz is read; the final epoch is saved unconditionally."""
    torch.set_num_threads(config["threads_per_worker"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    prepared = project_path(config["prepared_dir"])
    run = project_path(config["run_dir"]) / f"{architecture}_t{length}"
    run.mkdir(parents=True, exist_ok=False)
    with np.load(prepared / "train.npz", allow_pickle=False) as arrays:
        x = torch.from_numpy(select_history(arrays["x"], length))
        y = torch.from_numpy(arrays["y"])
    loader = DataLoader(TensorDataset(x, y), batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(config["seed"]))
    model = SlipModel(architecture, config["hidden_size"])
    initial_hash = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.BCEWithLogitsLoss()
    saved_config = {**config, "architecture": architecture, "sequence_length": length,
                    "checkpoint_rule": "fixed final epoch; NO evaluation-based selection",
                    "training_rows": len(y), "parameters": sum(p.numel() for p in model.parameters()),
                    "initial_state_sha256": initial_hash,
                    "train_data_sha256": digest(prepared / "train.npz"),
                    "data_report_sha256": digest(prepared / "report.json"),
                    "torch_version": str(torch.__version__), "python_version": platform.python_version()}
    save_json(run / "config.json", saved_config)
    start = time.perf_counter()
    with (run / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_nll", "elapsed_seconds"])
        writer.writeheader()
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            total = 0.
            for features, labels in loader:
                optimizer.zero_grad()
                loss = criterion(model(features), labels)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(labels)
            row = {"epoch": epoch, "train_nll": total / len(y), "elapsed_seconds": time.perf_counter() - start}
            writer.writerow(row)
            stream.flush()
            if epoch == 1 or epoch % 25 == 0 or epoch == config["epochs"]:
                print(f"{architecture} T={length}: epoch {epoch}/{config['epochs']}, "
                      f"training NLL={row['train_nll']:.4f} (no calibration/test access)", flush=True)
    torch.save({"config": saved_config, "state_dict": model.state_dict(), "epoch": config["epochs"]},
               run / "final.pt")
    save_json(run / "training_summary.json", {"epochs_completed": config["epochs"],
              "training_seconds": time.perf_counter() - start,
              "last_training_nll": row["train_nll"], "selected_epoch": config["epochs"],
              "evaluations_during_training": 0})
    return str(run)


@torch.inference_mode()
def predict(model, windows, length):
    model.eval()
    features = torch.from_numpy(select_history(windows, length))
    return np.concatenate([model(batch).numpy() for batch in DataLoader(features, batch_size=128)])


def evaluate_one(config, architecture, length):
    torch.set_num_threads(config["threads_per_worker"])
    run = project_path(config["run_dir"]) / f"{architecture}_t{length}"
    prepared = project_path(config["prepared_dir"])
    saved = torch.load(run / "final.pt", map_location="cpu", weights_only=True)
    if digest(prepared / "report.json") != saved["config"]["data_report_sha256"]:
        raise ValueError("Prepared data report changed since training")
    audit = json.loads((prepared / "report.json").read_text("utf-8"))
    for name, checksum in audit["prepared_sha256"].items():
        if digest(prepared / name) != checksum:
            raise ValueError(f"Prepared data changed since training: {name}")
    if saved["epoch"] != config["epochs"]:
        raise ValueError("Not the predeclared final-epoch model")
    model = SlipModel(architecture, config["hidden_size"])
    model.load_state_dict(saved["state_dict"])
    with np.load(prepared / "calibration.npz", allow_pickle=False) as arrays:
        cal_y, cal_logits = arrays["y"], predict(model, arrays["x"], length)
    temperature = fit_temperature(cal_y, cal_logits, config["temperature_bounds"])
    save_json(run / "temperature.json", {"fit_split": "calibration", "temperature": temperature,
              "nll_before": nll(cal_y, cal_logits), "nll_after": nll(cal_y, cal_logits / temperature)})
    with np.load(prepared / "test.npz", allow_pickle=False) as arrays:
        labels, logits = arrays["y"], predict(model, arrays["x"], length)
        objects, endpoints = arrays["object_id"], arrays["endpoint_id"]
    np.savez(run / "predictions.npz", calibration_labels=cal_y, calibration_logits=cal_logits,
             test_labels=labels, test_logits=logits, test_object_id=objects, test_endpoint_id=endpoints)
    pd.DataFrame({"endpoint_id": endpoints, "object_id": objects, "label": labels.astype(int),
                  "logit": logits, "probability_raw": expit(logits.astype(float)),
                  "probability_temperature": expit(logits.astype(float) / temperature)}).to_csv(
                      run / "test_predictions.csv", index=False)
    rows, per_object = [], []
    for method, t in [("raw", 1.), ("temperature", temperature)]:
        metadata = {"architecture": architecture, "steps": length, "method": method,
                    "temperature": t, "seed": config["seed"], "epoch": saved["epoch"],
                    "parameters": saved["config"]["parameters"]}
        rows.append({**metadata, **metrics(labels, logits.astype(float) / t)})
        for obj in sorted(set(objects)):
            selected = objects == obj
            per_object.append({**metadata, "object": obj, **metrics(labels[selected], logits[selected].astype(float) / t)})
    save_json(run / "metrics.json", rows)
    return rows, per_object


def validate_config(config):
    if sorted(config["architectures"]) != ["gru", "lstm"] or config["sequence_lengths"] != [1, 50]:
        raise ValueError("This experiment compares exactly LSTM/GRU x 1/50 steps")
    if config["epochs"] < 1 or config["batch_size"] < 1 or config["workers"] not in (1, 2):
        raise ValueError("Invalid epochs, batch size or worker count")
    if config["threshold"] != .5:
        raise ValueError("The predeclared classification threshold is 0.5")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="common/configs/temporal_ablation.yaml")
    parser.add_argument("--run-dir")
    parser.add_argument("--results-dir")
    parser.add_argument("--epochs", type=int, help="Short smoke tests only; final study uses 200")
    parser.add_argument("--workers", type=int, choices=[1, 2])
    args = parser.parse_args()
    config = load_config(args.config)
    for key, value in {"run_dir": args.run_dir, "results_dir": args.results_dir,
                       "epochs": args.epochs, "workers": args.workers}.items():
        if value is not None:
            config[key] = value
    validate_config(config)
    run, results = project_path(config["run_dir"]), project_path(config["results_dir"])
    if run.exists() or results.exists():
        parser.error("Run/results directory already exists: choose BOTH new paths; no overwrite.")
    audit = prepare(config)
    run.mkdir(parents=True)
    save_json(run / "protocol.json", {**config, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                                     "preprocessing_signature": audit["preprocessing_signature"]})
    # Long-window jobs first; two CPU workers, each with one PyTorch thread.
    trials = [(a, t) for t in reversed(config["sequence_lengths"]) for a in config["architectures"]]
    print(f"Prepared common endpoints: {audit['counts']}", flush=True)
    if config["workers"] == 1:
        for architecture, length in trials:
            train_one(config, architecture, length)
    else:
        with ProcessPoolExecutor(max_workers=config["workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(train_one, config, architecture, length) for architecture, length in trials]
            for future in as_completed(futures):
                print(f"Completed training: {future.result()}", flush=True)
    print("All training finished. Fitting temperatures on calibration objects, then scoring test objects.", flush=True)
    rows, per_object = [], []
    for architecture, length in trials:
        trial_rows, trial_objects = evaluate_one(config, architecture, length)
        rows.extend(trial_rows)
        per_object.extend(trial_objects)
    from .report import write_report
    write_report(config, audit, rows, per_object)
    print(f"Complete: {results / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
