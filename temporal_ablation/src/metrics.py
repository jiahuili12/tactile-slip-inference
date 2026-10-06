"""Fixed-threshold metrics and calibration fitted without final-test labels."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def nll(labels, logits):
    z, y = np.asarray(logits, dtype=float), np.asarray(labels, dtype=float)
    return float(np.mean(np.logaddexp(0, z) - y * z))


def fit_temperature(labels, logits, bounds=(.05, 20.)):
    if np.unique(labels).size != 2:
        raise ValueError("Calibration requires both classes")
    logits = np.asarray(logits, dtype=float)
    result = minimize_scalar(lambda v: nll(labels, logits / np.exp(v)),
                             bounds=tuple(np.log(bounds)), method="bounded")
    candidates = [1., float(bounds[0]), float(bounds[1]), float(np.exp(result.x))]
    return min(candidates, key=lambda t: nll(labels, logits / t))


def metrics(labels, logits):
    labels = np.asarray(labels, dtype=int)
    logits = np.asarray(logits, dtype=float)
    probabilities = expit(logits)
    decisions = logits > 0
    confidence = np.maximum(probabilities, 1 - probabilities)
    correct = decisions == labels
    bins = np.minimum((confidence * 15).astype(int), 14)
    ece = sum(float((bins == i).mean()) * abs(float(correct[bins == i].mean()) -
              float(confidence[bins == i].mean())) for i in range(15) if (bins == i).any())
    tn, fp, fn, tp = confusion_matrix(labels, decisions, labels=[0, 1]).ravel()
    both = np.unique(labels).size == 2
    return {"accuracy": float(correct.mean()), "precision": float(precision_score(labels, decisions, zero_division=0)),
            "recall": float(recall_score(labels, decisions, zero_division=0)),
            "f1": float(f1_score(labels, decisions, zero_division=0)),
            "ap": float(average_precision_score(labels, logits)) if both else None,
            "roc_auc": float(roc_auc_score(labels, logits)) if both else None,
            "ece": ece, "nll": nll(labels, logits),
            "brier": float(np.mean((probabilities - labels) ** 2)), "rows": len(labels),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
