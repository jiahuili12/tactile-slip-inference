"""Aggregate actual run results and save publication-independent diagnostic plots."""
import argparse
import json
import os

import numpy as np
import pandas as pd

from .common import ROOT, project_path
from .metrics import reliability_bins, risk_coverage

# Cache stays inside the project; works under restricted Windows environments.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def aggregate(runs, output):
    runs = [project_path(run) for run in runs]
    output = project_path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = [row for run in runs for row in json.loads((run / "evaluation.json").read_text(encoding="utf-8"))]
    frame = pd.DataFrame(rows).sort_values(["model", "seed", "method"])
    frame.to_csv(output / "metrics.csv", index=False)
    measures = ["accuracy", "precision", "recall", "f1", "pr_auc_ap", "roc_auc", "ece", "nll", "brier",
                "f1_at_val_threshold", "parameters", "latency_median_ms", "latency_p95_ms", "training_seconds"]
    summary = frame.groupby(["model", "method"])[measures].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output / "summary.csv", index=False)
    per_object = pd.concat([pd.read_csv(run / "per_object.csv") for run in runs], ignore_index=True)
    per_object.to_csv(output / "per_object.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, metric, label in zip(axes, ["f1", "ece", "nll"], ["F1 (threshold 0.5)", "ECE (15 bins)", "NLL"]):
        data = frame.groupby(["model", "method"])[metric].agg(["mean", "std"])
        ax.bar(np.arange(len(data)), data["mean"], yerr=data["std"].fillna(0), capsize=3)
        ax.set_xticks(np.arange(len(data)), [f"{m}\n{c}" for m, c in data.index])
        ax.set_ylabel(label)
    fig.tight_layout()
    fig.savefig(figures / "comparison.png", dpi=160)
    plt.close(fig)
    # One seed per model for detailed curves; no best-test-seed selection.
    representative = {}
    for run in runs:
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if config["model"] not in representative or config["seed"] < representative[config["model"]][0]:
            representative[config["model"]] = (config["seed"], run)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([0.5, 1], [0.5, 1], "k--", linewidth=1)
    for model, (seed, run) in sorted(representative.items()):
        prediction = pd.read_csv(run / "test_predictions.csv")
        for method, column in [("raw", "probability_raw"), ("temperature", "probability_temperature")]:
            labels, p = prediction["label"].to_numpy(), prediction[column].to_numpy()
            bins = reliability_bins(labels, p)
            legend = f"{model} {method} (seed {seed})"
            axes[0].plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", label=legend)
            coverage, risk = risk_coverage(labels, p)
            axes[1].plot(coverage, risk, label=legend)
    axes[0].set(xlabel="Mean top-label confidence", ylabel="Accuracy", title="Reliability diagram")
    axes[1].set(xlabel="Accepted fraction", ylabel="Accepted-set error", title="Risk–coverage (offline)")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "calibration_and_risk.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, (seed, run) in sorted(representative.items()):
        history = pd.read_csv(run / "history.csv")
        ax.plot(history["epoch"], history["train_nll"], label=f"{model} train")
        ax.plot(history["epoch"], history["validation_nll"], "--", label=f"{model} validation")
    ax.set(xlabel="Epoch", ylabel="NLL", title="Training curves: lowest seed per model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "training_curves.png", dpi=160)
    plt.close(fig)
    print(summary.to_string(), flush=True)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    aggregate(args.runs, args.output)
