"""Binary classification and equal-width top-label calibration metrics."""
import numpy as np
from scipy.special import expit
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             precision_recall_curve)

from .calibrate import nll


def reliability_bins(labels, probabilities, bins=15):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    prediction = probabilities >= 0.5
    confidence = np.maximum(probabilities, 1 - probabilities)
    correct = prediction == labels
    assignments = np.minimum((confidence * bins).astype(int), bins - 1)
    rows = []
    for i in range(bins):
        selected = assignments == i
        if selected.any():
            rows.append({"bin": i, "count": int(selected.sum()),
                         "confidence": float(confidence[selected].mean()),
                         "accuracy": float(correct[selected].mean())})
    return rows


def classification_metrics(labels, logits, threshold=0.5, bins=15):
    labels = np.asarray(labels, dtype=int)
    logits = np.asarray(logits, dtype=float)
    probabilities = expit(logits)
    prediction = probabilities >= threshold
    rows = reliability_bins(labels, probabilities, bins)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    both_classes = np.unique(labels).size == 2
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "pr_auc_ap": float(average_precision_score(labels, probabilities)) if both_classes else None,
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both_classes else None,
        "nll": nll(logits, labels),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "ece": sum(r["count"] / len(labels) * abs(r["accuracy"] - r["confidence"]) for r in rows),
        "threshold": float(threshold), "windows": len(labels),
        "positive_fraction": float(labels.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def validation_f1_threshold(labels, logits):
    precision, recall, thresholds = precision_recall_curve(labels, expit(np.asarray(logits, dtype=float)))
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)])


def risk_coverage(labels, probabilities):
    confidence = np.maximum(probabilities, 1 - probabilities)
    order = np.argsort(-confidence, kind="stable")
    errors = ((probabilities >= 0.5) != np.asarray(labels))[order]
    accepted = np.arange(1, len(errors) + 1)
    return accepted / len(errors), np.cumsum(errors) / accepted
