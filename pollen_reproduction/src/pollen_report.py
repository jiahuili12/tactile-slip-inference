"""Write an auditable report without presenting selection scores as test scores."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from common.src.utils import ROOT, save_json


def score_predictions(labels, logits):
    labels = np.asarray(labels, dtype=int)
    probabilities = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float32)).numpy()
    decisions = probabilities > 0.5
    tn, fp, fn, tp = confusion_matrix(labels, decisions, labels=[0, 1]).ravel()
    both_classes = len(np.unique(labels)) == 2
    return {
        "rows": len(labels), "accuracy": float(accuracy_score(labels, decisions)),
        "precision": float(precision_score(labels, decisions, zero_division=0)),
        "recall": float(recall_score(labels, decisions, zero_division=0)),
        "f1": float(f1_score(labels, decisions, zero_division=0)),
        "average_precision": float(average_precision_score(labels, probabilities)) if both_classes else None,
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both_classes else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "slip_fraction": float(labels.mean()), "decision_rule": "sigmoid(logit) > 0.5",
    }


def write_report(results, run, config, summary, audit, labels, logits, objects):
    results.mkdir(parents=True, exist_ok=False)
    metrics = score_predictions(labels, logits)
    delta = 100 * (metrics["accuracy"] - config["reported_accuracy"])
    final_epoch = summary["phase_bests"][1]["global_epoch"]
    full = config["full_epoch_budget"]
    metrics.update({"protocol": "pollen_random_row_selection", "seed": config["seed"],
                    "selected_global_epoch": final_epoch, "full_epoch_budget": full,
                    "reported_accuracy": config["reported_accuracy"],
                    "difference_percentage_points": delta, "independent_test": False})
    save_json(results / "metrics.json", metrics)
    save_json(results / "data_audit.json", audit)
    save_json(results / "run_config.json", config)
    save_json(results / "training_summary.json", summary)
    pd.read_csv(run / "history.csv").to_csv(results / "history.csv", index=False)
    per_object = [{"recording": name, **score_predictions(labels[objects == name], logits[objects == name])}
                  for name in sorted(set(objects))]
    pd.DataFrame(per_object).to_csv(results / "per_recording.csv", index=False)

    history = pd.read_csv(run / "history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.global_epoch, history.selection_accuracy * 100, label="Selection accuracy")
    axes[0].axhline(config["reported_accuracy"] * 100, color="tab:red", linestyle="--", label="Published 98.26%")
    axes[0].axvline(config["epochs_per_phase"] + .5, color="gray", linestyle=":", label="Best-score reset")
    axes[0].set(xlabel="Global epoch", ylabel="Accuracy (%)", title="NOT an independent test")
    axes[0].legend(fontsize=8)
    axes[1].plot(history.global_epoch, history.train_nll)
    axes[1].set(xlabel="Global epoch", ylabel="Mean training BCE", title="Training loss")
    fig.tight_layout()
    fig.savefig(results / "training.png", dpi=150)
    plt.close(fig)

    status = "Full 2 x 100 epoch protocol run" if full else "SMOKE TEST ONLY: shortened epoch budget"
    report = f"""# Pollen training-protocol reproduction

{status}. One seed ({config['seed']}), {config['device']}, trained from scratch.
Generated from saved predictions; no robot, calibration or DAC involved.

中文摘要：本次模型选择集准确率为 **{100 * metrics['accuracy']:.4f}%**，
与发布者的 98.26% 相差 **{delta:+.4f} 个百分点**。滑移 F1 为 **{metrics['f1']:.4f}**。
该集合参与了模型选择，**不是独立测试成绩，也不代表未见物体或真实机器人性能**。

## Result

| Metric | This local run |
|---|---:|
| Upstream model-card accuracy | {100 * config['reported_accuracy']:.2f}% |
| Local selection-set accuracy | {100 * metrics['accuracy']:.4f}% |
| Difference (local minus published) | {delta:+.4f} percentage points |
| Slip precision | {metrics['precision']:.4f} |
| Slip recall | {metrics['recall']:.4f} |
| Slip F1 | {metrics['f1']:.4f} |
| Average precision | {metrics['average_precision']:.4f} |
| Majority no-slip accuracy | {100 * (1 - metrics['slip_fraction']):.4f}% |
| Selected global epoch (phase-2 best) | {final_epoch} / {config['total_epochs']} |
| Parameters | {config['parameter_count']:,} |
| Training + per-epoch evaluation | {summary['training_seconds'] / 60:.2f} minutes |

Confusion matrix, rows = true [no slip, slip], columns = predicted [no slip, slip]:

| | Predicted no slip | Predicted slip |
|---|---:|---:|
| True no slip | {metrics['tn']} | {metrics['fp']} |
| True slip | {metrics['fn']} | {metrics['tp']} |

These are checkpoint-selection scores, NOT an independent final test. Closeness
to 98.26% is a numerical comparison only, not proof of exact published-run replication.
The publisher does not identify the initialization seed, full runtime environment,
dataset revision or selected epoch that produced the model-card number.

## Training history

![All epochs: selection accuracy and training loss](training.png)

## What was reproduced

- Pollen source commit `{config['upstream_commit']}`.
- Pinned dataset `{audit['revision']}`: all 17 CSVs, including `no_slip.csv`.
- Actual Hugging Face dataset loader; every row/file order verified against local CSV hashes.
- {audit['rows']:,} rows: {audit['train_rows']:,} training, {audit['selection_rows']:,} selection.
- All-data StandardScaler before random 80/20 row split, seed 42; no stratification.
- Input `[batch, 1, 15]`; no resampling or history window.
- LSTM hidden 128, then Linear 128->128 and Linear 128->1; no intervening activation.
- Batch 32, Adam 0.001, unweighted BCEWithLogitsLoss; no early stopping/clipping.
- Two consecutive {config['epochs_per_phase']}-epoch phases, preserving model and optimizer.
  Best score resets between phases, just as in the duplicated upstream loops.
  The upstream final artifact corresponds to the best checkpoint of phase 2.
- Strict `sigmoid(logit) > 0.5` prediction threshold.

## Explicit adaptations, not silent changes

- CPU instead of upstream hard-coded CUDA; PyTorch initialization seed {config['seed']} is
  recorded because upstream does not fix one. Package versions are in `run_config.json`.
- Upstream computes augmented arrays but never uses them in training. That unused work
  is omitted; all training uses the original rows, as upstream actually does.
- Equivalent zero initial states; additional mean-loss logging and audit artifacts.
  No upstream pretrained weights are used.

## Evaluation limits

1. Random rows from the same recordings can occur in both subsets; nearby observations
   are correlated. This is not held-out-object, held-out-trial or future-time evaluation.
2. Standardisation uses selection-set features: preprocessing leakage is retained solely
   for protocol reproduction, not endorsed as good practice.
3. The set called 'test' upstream selects checkpoints every epoch. It is therefore a
   selection set; there is no untouched final test in this protocol.
4. A one-time-step LSTM does not demonstrate temporal slip-pattern learning.
5. One initialization on one split gives no uncertainty interval for model performance.
6. The earlier object-held-out baseline differs in data, split, sequence length,
   architecture and training. Its ~79% accuracy is NOT a controlled comparison.
7. These offline results do not establish SO-100/SO-101 transfer, drop-rate reduction,
   physical safety, real-time control, confidence calibration or DAC performance.

## Artifacts and sources

- `history.csv` and `training.png`: all epochs, not only the selected one.
- `per_recording.csv`: selection scores broken down by source recording (not new objects).
- `data_audit.json`: verified file order, hashes, row counts, split hashes and protocol.
- `run_config.json`, `training_summary.json`: environment and both phase-best records.
- Local weights, row indices and prediction arrays: `{run.relative_to(ROOT) if run.is_relative_to(ROOT) else run}` (Git-ignored).
- [Pinned Pollen code](https://github.com/pollen-robotics/anyskin-slip-detection/tree/{config['upstream_commit']})
- [Published model card](https://huggingface.co/pollen-robotics/anyskin-slip-detection)
- [Dataset](https://huggingface.co/datasets/pollen-robotics/anyskin_slip_detection)
- [Protocol explanation](../README.md)
"""
    (results / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
