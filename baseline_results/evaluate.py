"""
TorchXRayVision Baseline Evaluation (No Training / No Fine-Tuning)
==================================================================
Evaluates a pretrained DenseNet121 on the Kaggle Chest X-Ray (Pneumonia)
dataset and produces a complete evaluation report.
"""

import time
import logging
import warnings
from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
import torchxrayvision as xrv
import numpy as np
import skimage.io
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve as sk_roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BATCH_SIZE: int = 32
THRESHOLD: float = 0.5
IMAGE_SIZE: int = 224
DATA_DIR: Path = Path("chest_xray")
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChestXRayDataset(Dataset):
    """PyTorch Dataset for the Kaggle Chest X-Ray (Pneumonia) test folder.

    Expects the following structure under ``root``::

        root/
            NORMAL/
            PNEUMONIA/

    Attributes:
        image_paths: List of Path objects for every valid image.
        labels: Corresponding integer labels (0 = NORMAL, 1 = PNEUMONIA).
        resizer: XRayResizer callable.
    """

    def __init__(self, root: Path) -> None:
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self.resizer = xrv.datasets.XRayResizer(IMAGE_SIZE)

        normal_dir = root / "NORMAL"
        pneumonia_dir = root / "PNEUMONIA"

        if not normal_dir.is_dir() and not pneumonia_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {root} to contain NORMAL/ and/or PNEUMONIA/ subdirectories."
            )

        for p in normal_dir.glob("*"):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
                self.image_paths.append(p)
                self.labels.append(0)

        for p in pneumonia_dir.glob("*"):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
                self.image_paths.append(p)
                self.labels.append(1)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, int]:
        path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            img = preprocess_image(path)
        except Exception as exc:
            warnings.warn(f"Skipping corrupted image {path.name}: {exc}")
            dummy = torch.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
            return path.name, dummy, label

        return path.name, img, label


def load_model() -> torch.nn.Module:
    """Load the pretrained TorchXRayVision DenseNet121 and set to eval mode.

    The model outputs are already probabilities (sigmoid + op_norm are applied
    internally via the ``op_threshs`` path).  No additional sigmoid on the
    user side is needed.

    Returns:
        The model in evaluation mode (no gradients, no training).
    """
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model.to(DEVICE)


def preprocess_image(path: Path) -> torch.Tensor:
    """Load and preprocess a single chest X-ray image.

    Pipeline (matches the official TorchXRayVision inference example):
        1. Read with ``skimage.io.imread``.
        2. Normalise via ``xrv.datasets.normalize(img, 255)`` → [-1024, 1024].
        3. Convert RGB → grayscale by averaging the last axis if needed.
        4. Add channel dimension.
        5. Resize to 224×224 via ``XRayResizer``.
        6. Convert to PyTorch float tensor.

    Args:
        path: Image file path.

    Returns:
        Tensor of shape ``(1, 224, 224)`` ready for the model.
    """
    img = skimage.io.imread(str(path))
    img = xrv.datasets.normalize(img, 255)

    if img.ndim == 3:
        img = img.mean(axis=-1)

    img = img[None, ...]
    img = xrv.datasets.XRayResizer(IMAGE_SIZE)(img)
    img = torch.from_numpy(img).float()
    return img


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    pneumonia_index: int,
    diagnostics: bool = True,
) -> Tuple[List[str], List[int], np.ndarray]:
    """Run inference over the entire dataloader.

    The model outputs are already calibrated probabilities (sigmoid + op_norm
    applied internally), so no further activation is applied.

    Args:
        model: Pretrained TorchXRayVision model in eval mode.
        dataloader: DataLoader supplying (filename, image, label) batches.
        pneumonia_index: Column index for "Pneumonia" in model outputs.
        diagnostics: If True, print first-batch diagnostic info.

    Returns:
        Tuple of (filenames, ground_truth_labels, pneumonia_probabilities).
    """
    all_filenames: List[str] = []
    all_labels: List[int] = []
    all_probs: List[float] = []

    first_batch = True

    for names, images, labels in tqdm(dataloader, desc="Evaluating", unit="batch"):
        images = images.to(DEVICE)
        outputs = model(images)

        # Model outputs are already probabilities (sigmoid + op_norm applied
        # internally).  DO NOT apply an additional sigmoid.
        pneumonia_out = outputs[:, pneumonia_index].cpu().numpy()

        if diagnostics and first_batch:
            print("\n--- First-batch diagnostics ---")
            print(f"  model.pathologies:            {model.pathologies}")
            print(f"  pneumonia_index:              {pneumonia_index}")
            print(f"  output shape:                 {outputs.shape}")
            print(f"  output min (all pathologies): {outputs.min().item():.6f}")
            print(f"  output max (all pathologies): {outputs.max().item():.6f}")
            raw_first_10 = outputs[0, :10].cpu().numpy()
            print(f"  first 10 raw pneumonia outputs (batch element 0): {raw_first_10}")
            print(f"  first 10 final probabilities: {pneumonia_out[:10]}")
            print("---\n")
            first_batch = False

        all_filenames.extend(names)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(pneumonia_out.tolist())

    return all_filenames, all_labels, np.array(all_probs)


def compute_metrics(
    y_true: List[int], y_pred: List[int], y_prob: np.ndarray
) -> dict:
    """Compute all requested evaluation metrics.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Binary predictions (thresholded probabilities).
        y_prob: Raw pneumonia probabilities.

    Returns:
        Dictionary containing accuracy, precision, recall, f1,
        roc_auc, confusion matrix entries, and specificity.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "specificity": specificity,
    }


def threshold_sweep(
    y_true: List[int], y_prob: np.ndarray
) -> Tuple[List[dict], float]:
    """Evaluate metrics across a range of classification thresholds.

    Thresholds from 0.10 to 0.90 in steps of 0.05 are tested.  The
    threshold that maximises the F1 score is identified.

    Args:
        y_true: Ground-truth labels.
        y_prob: Pneumonia probabilities.

    Returns:
        Tuple of (list of per-threshold result dicts, best_threshold).
    """
    results = []
    best_f1 = -1.0
    best_threshold = 0.5

    for thresh in np.arange(0.10, 0.95, 0.05):
        thresh = round(thresh, 2)
        y_pred = (y_prob >= thresh).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
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


def save_reports(
    y_true: List[int],
    y_pred: List[int],
    y_prob: np.ndarray,
    filenames: List[str],
    metrics: dict,
    threshold_results: List[dict],
    best_threshold: float,
    output_dir: Path,
) -> None:
    """Save plots, CSVs, and print top false positives/negatives.

    Args:
        y_true: Ground-truth labels.
        y_pred: Binary predictions at default (0.5) threshold.
        y_prob: Pneumonia probabilities.
        filenames: Image filenames.
        metrics: Dictionary from :func:`compute_metrics`.
        threshold_results: Per-threshold results from :func:`threshold_sweep`.
        best_threshold: Threshold that maximises F1.
        output_dir: Output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── confusion matrix plot ──────────────────────────────────────────────
    cm = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Normal", "Pneumonia"],
        yticklabels=["Normal", "Pneumonia"],
        xlabel="Predicted",
        ylabel="True",
    )
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ── ROC curve (uses raw probabilities, NOT thresholded predictions) ────
    fpr, tpr, _ = sk_roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        xlim=[-0.01, 1.01],
        ylim=[-0.01, 1.01],
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    # ── predictions CSV ────────────────────────────────────────────────────
    csv_path = output_dir / "predictions.csv"
    with open(csv_path, "w") as f:
        f.write("filename,ground_truth,predicted_probability,predicted_label\n")
        for fn, gt, prob, pred in zip(filenames, y_true, y_prob, y_pred):
            f.write(f"{fn},{gt},{prob:.6f},{pred}\n")

    # ── threshold sweep CSV ────────────────────────────────────────────────
    sweep_path = output_dir / "threshold_results.csv"
    with open(sweep_path, "w") as f:
        f.write("threshold,accuracy,precision,recall,specificity,f1\n")
        for r in threshold_results:
            f.write(
                f"{r['threshold']:.2f},{r['accuracy']:.4f},{r['precision']:.4f},"
                f"{r['recall']:.4f},{r['specificity']:.4f},{r['f1']:.4f}\n"
            )

    # ── top-20 false positives & false negatives ───────────────────────────
    fps = [
        (fn, prob)
        for fn, gt, prob, pred in zip(filenames, y_true, y_prob, y_pred)
        if gt == 0 and pred == 1
    ]
    fns = [
        (fn, prob)
        for fn, gt, prob, pred in zip(filenames, y_true, y_prob, y_pred)
        if gt == 1 and pred == 0
    ]

    fps.sort(key=lambda x: x[1], reverse=True)
    fns.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 20 False Positives (Normal predicted as Pneumonia):")
    for rank, (fn, prob) in enumerate(fps[:20], 1):
        print(f"  {rank:2d}. {fn}  (prob = {prob:.4f})")

    print("\nTop 20 False Negatives (Pneumonia predicted as Normal):")
    for rank, (fn, prob) in enumerate(fns[:20], 1):
        print(f"  {rank:2d}. {fn}  (prob = {prob:.4f})")


def print_report(metrics: dict, total_images: int, normal_count: int, pneumonia_count: int) -> None:
    """Print a formatted evaluation report to the console.

    Args:
        metrics: Dictionary from :func:`compute_metrics`.
        total_images: Number of images evaluated.
        normal_count: Number of normal (label 0) images.
        pneumonia_count: Number of pneumonia (label 1) images.
    """
    cm = metrics["confusion_matrix"]
    print("=" * 52)
    print("TorchXRayVision Baseline Evaluation")
    print("=" * 52)
    print(f"\nTotal Test Images:  {total_images}")
    print(f"Normal Images:      {normal_count}")
    print(f"Pneumonia Images:   {pneumonia_count}")
    print(f"\nAccuracy:           {metrics['accuracy']:.4f}")
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall (Sensitivity): {metrics['recall']:.4f}")
    print(f"Specificity:        {metrics['specificity']:.4f}")
    print(f"F1 Score:           {metrics['f1_score']:.4f}")
    print(f"ROC AUC:            {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted Normal  Predicted Pneumonia")
    print(f"Actual Normal       {cm[0,0]:>5d}               {cm[0,1]:>5d}")
    print(f"Actual Pneumonia    {cm[1,0]:>5d}               {cm[1,1]:>5d}")
    print(f"\nTN: {metrics['tn']}")
    print(f"FP: {metrics['fp']}")
    print(f"FN: {metrics['fn']}")
    print(f"TP: {metrics['tp']}")
    print("=" * 52)


def print_threshold_sweep(results: List[dict], best_threshold: float) -> None:
    """Print the threshold-sweep table and optimal threshold summary.

    Args:
        results: Per-threshold result dicts from :func:`threshold_sweep`.
        best_threshold: Threshold that maximised F1.
    """
    print("\n" + "=" * 75)
    print("Threshold Sweep Evaluation")
    print("=" * 75)
    header = f"{'Thresh':>7} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'Spec':>7} {'F1':>7}"
    print(header)
    print("-" * 75)

    best_row = None
    for r in results:
        line = (
            f"{r['threshold']:>7.2f} {r['accuracy']:>7.4f} {r['precision']:>7.4f} "
            f"{r['recall']:>7.4f} {r['specificity']:>7.4f} {r['f1']:>7.4f}"
        )
        if r['threshold'] == best_threshold:
            line += "  <-- best F1"
            best_row = r
        print(line)

    print("-" * 75)
    if best_row:
        print(f"\nBest Threshold:      {best_threshold:.2f}")
        print(f"Best Accuracy:       {best_row['accuracy']:.4f}")
        print(f"Best Precision:      {best_row['precision']:.4f}")
        print(f"Best Recall:         {best_row['recall']:.4f}")
        print(f"Best Specificity:    {best_row['specificity']:.4f}")
        print(f"Best F1:             {best_row['f1']:.4f}")
    print("=" * 75)


def print_probability_diagnostics(probs: np.ndarray) -> None:
    """Print distribution statistics of predicted pneumonia probabilities.

    Args:
        probs: Array of pneumonia probabilities.
    """
    print("\nProbability Distribution Diagnostics:")
    print(f"  Minimum probability:    {probs.min():.6f}")
    print(f"  Maximum probability:    {probs.max():.6f}")
    print(f"  Mean probability:       {probs.mean():.6f}")
    print(f"  Median probability:     {np.median(probs):.6f}")
    percentiles = [0, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    p_vals = np.percentile(probs, percentiles)
    print(f"  Percentiles:")
    for pct, val in zip(percentiles, p_vals):
        print(f"    {pct:3d}%:  {val:.6f}")


def main() -> None:
    """Entry point: load model, evaluate test set, compute metrics, save outputs."""
    start_time = time.perf_counter()

    print(f"Device: {DEVICE}")
    print("Loading model ...")
    model = load_model()
    print("Model loaded.\n")

    # Determine pneumonia index dynamically from model.pathologies
    pneumonia_index = model.pathologies.index("Pneumonia")
    print(f"Pathologies ({len(model.pathologies)}): {model.pathologies}")
    print(f"Pneumonia index: {pneumonia_index}\n")

    dataset = ChestXRayDataset(DATA_DIR / "test")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE.type == "cuda" else False,
    )

    filenames, labels, probs = evaluate(model, dataloader, pneumonia_index)

    # Probability distribution diagnostics
    print_probability_diagnostics(probs)

    # Default threshold (0.5) evaluation
    y_pred_binary = (probs >= THRESHOLD).astype(int)
    metrics = compute_metrics(labels, y_pred_binary, probs)

    normal_count = labels.count(0)
    pneumonia_count = labels.count(1)
    total = len(labels)

    print_report(metrics, total, normal_count, pneumonia_count)

    # Threshold sweep
    threshold_results, best_threshold = threshold_sweep(labels, probs)
    print_threshold_sweep(threshold_results, best_threshold)

    # Save reports
    save_reports(
        labels, y_pred_binary, probs, filenames, metrics,
        threshold_results, best_threshold, DATA_DIR,
    )

    elapsed = time.perf_counter() - start_time
    print(f"\nTotal evaluation time: {elapsed:.2f} seconds")
    print("Outputs saved to:", DATA_DIR.resolve())


if __name__ == "__main__":
    main()
