"""Reproduce Pollen's public training protocol, including its evaluation limits.

Run: python -m pollen_reproduction
This does NOT run the object-held-out baseline or calibrate a model.
"""
import argparse
import csv
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common.src.utils import project_path, save_json
from .pollen_data import load_data, save_data_audit
from .pollen_model import PollenLSTM

UPSTREAM_COMMIT = "24a3728c35985def720ff292a8a45ef7e8662a64"
REPORTED_ACCURACY = 0.9826


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    predictions = []
    for features, _ in loader:
        predictions.append(model(features.to(device)).cpu())
    return torch.cat(predictions).reshape(-1).numpy()


def upstream_accuracy(logits, labels):
    # Keep the upstream float32 sigmoid and strict > 0.5 comparison.
    predicted = torch.sigmoid(torch.as_tensor(logits)) > 0.5
    return float((predicted == torch.as_tensor(labels).reshape(-1)).float().mean())


def environment():
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["torch", "numpy", "pandas", "scikit-learn", "datasets"]},
        "cuda_available": torch.cuda.is_available(),
    }


def train_reproduction(run, x, y, train_ids, test_ids, seed=42, epochs_per_phase=100,
                       threads=1, device="cpu"):
    """Two consecutive phases; reset best score, NOT model/Adam, in phase 2.

    Output is deliberately called 'selection', because upstream uses its test
    set every epoch to choose weights. No independent test is claimed.
    """
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x[train_ids]),
                                          torch.from_numpy(y[train_ids])),
                              batch_size=32, shuffle=True)
    selection_loader = DataLoader(TensorDataset(torch.from_numpy(x[test_ids]),
                                              torch.from_numpy(y[test_ids])),
                                  batch_size=32, shuffle=False)
    model = PollenLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    config = {
        "upstream_commit": UPSTREAM_COMMIT, "reported_accuracy": REPORTED_ACCURACY,
        "seed": seed, "split_seed": 42, "device": device, "threads": threads,
        "batch_size": 32, "hidden_size": 128, "num_layers": 1, "sequence_length": 1,
        "learning_rate": 0.001, "phases": 2, "epochs_per_phase": epochs_per_phase,
        "total_epochs": 2 * epochs_per_phase,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "full_epoch_budget": epochs_per_phase == 100,
        "environment": environment(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(run / "config.json", config)
    best_records, start = [], time.perf_counter()
    with (run / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["phase", "epoch_in_phase", "global_epoch", "train_nll", "selection_accuracy",
                  "epoch_seconds", "elapsed_seconds", "checkpoint_updated"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for phase in (1, 2):
            best_accuracy, best_record = 0.0, None
            for epoch in range(1, epochs_per_phase + 1):
                epoch_start = time.perf_counter()
                model.train()
                loss_sum = 0.0
                for features, labels in train_loader:
                    features, labels = features.to(device), labels.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(features), labels)
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item() * len(labels)
                logits = predict(model, selection_loader, device)
                score = upstream_accuracy(logits, y[test_ids])
                updated = score > best_accuracy
                row = {
                    "phase": phase, "epoch_in_phase": epoch,
                    "global_epoch": (phase - 1) * epochs_per_phase + epoch,
                    "train_nll": loss_sum / len(train_ids), "selection_accuracy": score,
                    "epoch_seconds": time.perf_counter() - epoch_start,
                    "elapsed_seconds": time.perf_counter() - start,
                    "checkpoint_updated": updated,
                }
                if updated:
                    best_accuracy, best_record = score, dict(row)
                    # Upstream overwrites the same file in both phases; retain
                    # phase-1 best as an extra audit, never reload it for phase 2.
                    torch.save({"state_dict": model.state_dict(), "config": config,
                                "selection": row}, run / f"phase{phase}_best.pt")
                writer.writerow(row)
                stream.flush()
                print(f"epoch {row['global_epoch']:3}/{2 * epochs_per_phase} | "
                      f"phase {phase} | train NLL {row['train_nll']:.4f} | "
                      f"selection accuracy {score:.4%} | "
                      f"{row['epoch_seconds']:.1f}s", flush=True)
            if best_record is None:
                raise RuntimeError("No checkpoint selected")
            best_records.append(best_record)
    # The artifact remaining after running upstream is the best of phase 2.
    best = torch.load(run / "phase2_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["state_dict"])
    logits = predict(model, selection_loader, device)
    np.savez(run / "selection_predictions.npz", row_id=test_ids,
             labels=y[test_ids].reshape(-1), logits=logits)
    summary = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "training_seconds": time.perf_counter() - start,
               "phase_bests": best_records,
               "final_artifact": "phase2_best.pt", "independent_test": False}
    save_json(run / "training_summary.json", summary)
    return config, summary, logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs-per-phase", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--run-dir", default="pollen_reproduction/runs/seed42")
    parser.add_argument("--results-dir", default="pollen_reproduction/results")
    args = parser.parse_args()
    if args.epochs_per_phase < 1 or args.threads < 1:
        parser.error("epochs-per-phase and threads must be positive")
    run, results = project_path(args.run_dir), project_path(args.results_dir)
    if run.exists() or results.exists():
        parser.error("Output directory already exists; choose NEW run-dir AND results-dir.")
    x, y, train_ids, test_ids, objects, scaler, provenance = load_data()
    run.mkdir(parents=True)
    save_data_audit(run, train_ids, test_ids, scaler, provenance)
    print(f"Rows: {len(y):,}; train {len(train_ids):,}; selection {len(test_ids):,}; "
          "one time step; NOT an independent test.", flush=True)
    config, summary, logits = train_reproduction(
        run, x, y, train_ids, test_ids, args.seed, args.epochs_per_phase,
        args.threads, args.device)
    from .pollen_report import write_report
    write_report(results, run, config, summary, provenance,
                 y[test_ids].reshape(-1), logits, objects[test_ids])
    print(f"Finished. Read {results / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
