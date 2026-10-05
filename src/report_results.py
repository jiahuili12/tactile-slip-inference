"""Write a readable report from measured results, without invented targets."""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .common import ROOT, project_path, save_json
from .metrics import classification_metrics


def report(output="results", processed="data/processed"):
    output, processed = project_path(output), project_path(processed)
    summary = pd.read_csv(output / "summary.csv")
    metrics = pd.read_csv(output / "metrics.csv")
    data_report = json.loads((processed / "report.json").read_text(encoding="utf-8"))
    with np.load(processed / "train.npz", allow_pickle=False) as arrays:
        prior = float(arrays["y"].mean())
    with np.load(processed / "test.npz", allow_pickle=False) as arrays:
        labels = arrays["y"]
    constant = np.full(len(labels), np.log(prior / (1 - prior)))
    baseline = classification_metrics(labels, constant)
    save_json(output / "constant_prior_baseline.json", baseline)
    lines = ["# Offline AnySkin baseline report", "",
             f"Generated: {datetime.now(timezone.utc).isoformat()} (UTC).", "",
             "All values below are computed from the public data and actual local runs. No physical robot was used.", "",
             "## Data and experiment", "",
             "Source: [Pollen Robotics AnySkin dataset](https://huggingface.co/datasets/pollen-robotics/anyskin_slip_detection), pinned revision `d9f5e315006482635a2effa34488715871d55c13`.", "",
             "Four object-disjoint sets; causal 100 Hz resampling, 50-sample input, stride 5, current-slip labels. Training-only standardisation. Matched hidden size, CPU, up to 20 epochs, validation-NLL early stopping.", "",
             "| Split | Objects | Windows | Slip fraction |", "|---|---:|---:|---:|"]
    for name, values in data_report["splits"].items():
        lines.append(f"| {name} | {len(values['files'])} | {values['windows']} | {values['positive_fraction']:.3f} |")
    lines += ["", f"Seeds: {', '.join(map(str, sorted(metrics['seed'].unique())))}. Mean ± sample standard deviation across seeds; these are not confidence intervals.", "",
              "## Held-out test results", "",
              "| Model | Calibration | F1 at 0.5 | F1 at validation-selected threshold | AP | ECE | NLL |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for _, row in summary.iterrows():
        vals = [f"{row[m + '_mean']:.3f} ± {row[m + '_std']:.3f}" for m in
                ["f1", "f1_at_val_threshold", "pr_auc_ap", "ece", "nll"]]
        lines.append(f"| {row['model']} | {row['method']} | " + " | ".join(vals) + " |")
    before = metrics[metrics.method == "raw"].set_index(["model", "seed"])
    after = metrics[metrics.method == "temperature"].set_index(["model", "seed"])
    worse = int((after["nll"] > before["nll"]).sum())
    lines += ["", "## Interpretation", "",
              f"- A constant training-prior model achieves accuracy {baseline['accuracy']:.3f}, F1 {baseline['f1']:.3f}, AP {baseline['pr_auc_ap']:.3f}, NLL {baseline['nll']:.3f}. Accuracy alone is not useful evidence of successful slip detection here.",
              "- LSTM's default-threshold predictions are dominated by the no-slip class. Both models need substantially better discrimination before robot deployment. A validation-selected threshold recovers some recall, with a precision trade-off; it is not selected on the test data.",
              "- Mean ECE is reduced by temperature scaling in this run. This changes confidence, not the learned features or the classification decision at 0.5.",
              f"- Test NLL worsened after temperature scaling in {worse} of {len(after)} individual runs despite calibration-set NLL improving. Calibration under held-out object shift is not guaranteed.",
              "- This experiment does not establish an overall LSTM/GRU winner. GRU has fewer parameters but the models trade off recall, ranking and calibration, and only three test objects are available.", "",
              "## Runtime", "",
              "Batch-1 CPU model-only latency, 20 warmups, 200 timings, two PyTorch threads. Includes a complete window forward pass; excludes sensing, preprocessing, calibration and control.", "",
              "| Model | Parameters | Median latency (ms), mean across seeds |", "|---|---:|---:|"]
    for _, row in summary[summary.method == "raw"].iterrows():
        lines.append(f"| {row['model']} | {int(row['parameters_mean'])} | {row['latency_median_ms_mean']:.3f} |")
    lines += ["", "## Reproduce and inspect", "",
              "Run `python -m src.download_data`, `python -m src.prepare_data`, `python -m src.run_experiment` in a new environment/project, or use `--resume` for completed local runs.",
              "See `metrics.csv`, `summary.csv`, `per_object.csv`, `data_report.json`, and `figures/`. Per-run logits, histories, fitted temperatures and checkpoints are in `runs/final_v1/` by default (ignored by Git).", "",
              "## Limits and next experiments", "",
              "Overlapping windows are correlated, trial/reset identifiers are absent, and the fixed split contains only three test objects. These results are not drop-rate, force, safety or real-robot measurements.",
              "Potential next studies include causal signal differences or baseline correction, additional object-held-out folds, richer recurrent models, and evaluation on locally collected SO-100/AnySkin data. These are hypotheses to test, not explanations already established by this experiment. Further development after seeing these test results should use a fresh holdout for final claims.", "",
              "See `THIRD_PARTY.md` for source and method references. No upstream code or weights were vendored. Temperature scaling is implemented; DAC is not.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {output / 'REPORT.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results")
    parser.add_argument("--processed", default="data/processed")
    args = parser.parse_args()
    report(args.output, args.processed)
