"""
Utility functions for training and evaluation.
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict:
    """Compute all evaluation metrics.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Binary predictions (thresholded).
        y_prob: Predicted probabilities.

    Returns:
        Dictionary of metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_sweep(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[List[Dict], float]:
    """Evaluate metrics across thresholds 0.05 to 0.95.

    Returns:
        Tuple of (per-threshold results, best_threshold by F1).
    """
    results = []
    best_f1 = -1.0
    best_threshold = 0.5

    for thresh in np.arange(0.05, 0.96, 0.05):
        thresh = round(thresh, 2)
        y_pred = (y_prob >= thresh).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        results.append({
            "threshold": thresh,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "specificity": spec,
            "f1": f1,
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    return results, best_threshold


def save_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, auc: float, output_path: Path) -> None:
    """Save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           xlim=[-0.01, 1.01], ylim=[-0.01, 1.01])
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    """Save confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Normal", "Pneumonia"],
           yticklabels=["Normal", "Pneumonia"],
           xlabel="Predicted", ylabel="True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_training_curves(history: Dict[str, List[float]], output_path: Path) -> None:
    """Save training loss/AUC curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # AUC
    axes[1].plot(epochs, history["val_auc"], label="Val AUC", color="green")
    axes[1].set(xlabel="Epoch", ylabel="AUC", title="Validation AUC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_metrics(metrics: Dict, title: str = "Evaluation Results") -> None:
    """Print formatted metrics to console."""
    print("\n" + "=" * 52)
    print(title)
    print("=" * 52)
    print(f"  Accuracy:            {metrics['accuracy']:.4f}")
    print(f"  Precision:           {metrics['precision']:.4f}")
    print(f"  Recall (Sensitivity):{metrics['recall']:.4f}")
    print(f"  Specificity:         {metrics['specificity']:.4f}")
    print(f"  F1 Score:            {metrics['f1_score']:.4f}")
    print(f"  ROC AUC:             {metrics['roc_auc']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion Matrix:")
    print(f"                       Pred Normal  Pred Pneumonia")
    print(f"  Actual Normal        {cm[0,0]:>7d}       {cm[0,1]:>7d}")
    print(f"  Actual Pneumonia     {cm[1,0]:>7d}       {cm[1,1]:>7d}")
    print("=" * 52)


def print_threshold_sweep(results: List[Dict], best_threshold: float) -> None:
    """Print threshold sweep table."""
    print("\n" + "=" * 75)
    print("Threshold Sweep")
    print("=" * 75)
    header = f"{'Thresh':>7} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'Spec':>7} {'F1':>7}"
    print(header)
    print("-" * 75)

    for r in results:
        line = (f"{r['threshold']:>7.2f} {r['accuracy']:>7.4f} {r['precision']:>7.4f} "
                f"{r['recall']:>7.4f} {r['specificity']:>7.4f} {r['f1']:>7.4f}")
        if r['threshold'] == best_threshold:
            line += "  <-- best F1"
        print(line)

    print("-" * 75)
    print(f"  Best Threshold: {best_threshold:.2f}")
    print("=" * 75)
