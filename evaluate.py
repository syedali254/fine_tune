"""
Run the pretrained TorchXRayVision model on the test set and return the pneumonia probabilities.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchxrayvision as xrv
from tqdm import tqdm

from pneumonia_config import Config
from dataset import ChestXRayDataset
from pneumonia_model import PneumoniaClassifier, load_model_for_eval
from utils import (
    compute_metrics,
    threshold_sweep,
    print_metrics,
    print_threshold_sweep,
    save_roc_curve,
    save_confusion_matrix,
)


@torch.no_grad()
def evaluate_baseline(
    dataloader: DataLoader, device: torch.device
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    #Evaluate the pretrained TorchXRayVision model (baseline).

  
    
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    model.to(device)

    pneumonia_index = model.pathologies.index("Pneumonia")
    print(f"  Pathologies: {model.pathologies}")
    print(f"  Pneumonia index: {pneumonia_index}")

    all_labels = []
    all_probs = []

    for images, labels in tqdm(dataloader, desc="Evaluating (baseline)", unit="batch"):
        images = images.to(device)
        outputs = model(images)
        # Outputs are already probabilities (sigmoid + op_norm applied internally)
        probs = outputs[:, pneumonia_index].cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())

    return np.array(all_labels), np.array(all_probs)


@torch.no_grad()
def evaluate_finetuned(
    dataloader: DataLoader, checkpoint_path: str, config: Config
) -> Tuple[np.ndarray, np.ndarray]:
    #Evaluate a fine-tuned model by loading the checkpoint 

    
    model = load_model_for_eval(checkpoint_path, config)
    print(f"  Loaded checkpoint: {checkpoint_path}")

    all_labels = []
    all_probs = []

    for images, labels in tqdm(dataloader, desc="Evaluating (fine-tuned)", unit="batch"):
        images = images.to(config.device)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())

    return np.array(all_labels), np.array(all_probs)


def save_predictions_csv(
    dataset: ChestXRayDataset,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Save per-image predictions to CSV."""
    with open(output_path, "w") as f:
        f.write("filename,ground_truth,predicted_probability,predicted_label\n")
        for i in range(len(y_true)):
            fn = dataset.get_filename(i)
            f.write(f"{fn},{int(y_true[i])},{y_prob[i]:.6f},{int(y_pred[i])}\n")


def save_threshold_csv(results, output_path: Path) -> None:
    """Save threshold sweep results to CSV."""
    with open(output_path, "w") as f:
        f.write("threshold,accuracy,precision,recall,specificity,f1\n")
        for r in results:
            f.write(
                f"{r['threshold']:.2f},{r['accuracy']:.4f},{r['precision']:.4f},"
                f"{r['recall']:.4f},{r['specificity']:.4f},{r['f1']:.4f}\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained baseline or fine-tuned model"
    )
    parser.add_argument(
        "--mode", type=str, required=True, choices=["baseline", "finetuned"],
        help="Evaluation mode: 'baseline' for pretrained model, 'finetuned' for checkpoint",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pth",
        help="Path to fine-tuned checkpoint (used only in 'finetuned' mode)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Classification threshold for primary metrics",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/<mode>)",
    )
    args = parser.parse_args()

    config = Config()

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("results") / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Evaluation Mode: {args.mode.upper()}")
    print(f"Device: {config.device}")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {output_dir.resolve()}")
    print("=" * 60)

    # ── Dataset & DataLoader ───────────────────────────────────────────────
    test_dataset = ChestXRayDataset(config.data_dir / "test", config.image_size)
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=(config.device.type == "cuda"),
    )

    print(f"\nTest set: {len(test_dataset)} images "
          f"(Normal: {test_dataset.num_normal}, Pneumonia: {test_dataset.num_pneumonia})")

    # ── Run Evaluation ─────────────────────────────────────────────────────
    start_time = time.perf_counter()

    if args.mode == "baseline":
        y_true, y_prob = evaluate_baseline(test_loader, config.device)
    else:
        checkpoint_path = args.checkpoint
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "Train a model first with: python train.py"
            )
        y_true, y_prob = evaluate_finetuned(test_loader, checkpoint_path, config)

    elapsed = time.perf_counter() - start_time

    # ── Compute Metrics at Primary Threshold ───────────────────────────────
    y_pred = (y_prob >= args.threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_prob)

    title = ("Baseline (Pretrained)" if args.mode == "baseline"
             else f"Fine-Tuned ({Path(args.checkpoint).name})")
    print_metrics(metrics, title=title)

    # ── Threshold Sweep ────────────────────────────────────────────────────
    sweep_results, best_threshold = threshold_sweep(y_true, y_prob)
    print_threshold_sweep(sweep_results, best_threshold)

    # Recompute metrics at best threshold
    y_pred_best = (y_prob >= best_threshold).astype(int)
    metrics_best = compute_metrics(y_true, y_pred_best, y_prob)
    print(f"\n  Metrics at Best Threshold ({best_threshold:.2f}):")
    print(f"    F1:        {metrics_best['f1_score']:.4f}")
    print(f"    Precision: {metrics_best['precision']:.4f}")
    print(f"    Recall:    {metrics_best['recall']:.4f}")

    # ── Save Outputs ───────────────────────────────────────────────────────
    save_roc_curve(y_true, y_prob, metrics["roc_auc"], output_dir / "roc_curve.png")
    save_confusion_matrix(metrics["confusion_matrix"], output_dir / "confusion_matrix.png")
    save_predictions_csv(test_dataset, y_true, y_prob, y_pred, output_dir / "predictions.csv")
    save_threshold_csv(sweep_results, output_dir / "threshold_results.csv")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\nEvaluation completed in {elapsed:.2f}s")
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
