from typing import Dict, Any
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Compute comprehensive diagnostic metrics for binary classification.

    Args:
        y_true: Ground truth binary array of shape (N,).
        y_prob: Predicted probabilities array of shape (N,).
        threshold: Decision threshold for discrete class assignment.

    Returns:
        Dictionary containing quantitative evaluation metrics.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob).ravel().astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = 0.5

    brier = float(brier_score_loss(y_true, y_prob))
    y_prob_clipped = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
    loss = float(log_loss(y_true, y_prob_clipped))

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "log_loss": loss,
        "confusion_matrix": cm,
    }
