"""
Training pipeline for fine-tuning TorchXRayVision DenseNet121
on pediatric chest X-rays (Normal vs Pneumonia).
"""

import argparse
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from pneumonia_config import Config
from dataset import ChestXRayDataset
from pneumonia_model import PneumoniaClassifier
from utils import set_seed, save_training_curves


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    all_labels = []
    all_probs = []

    for images, labels in tqdm(dataloader, desc="  Train", leave=False):
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, auc


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run validation and return avg loss + AUC."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []

    for images, labels in tqdm(dataloader, desc="  Val", leave=False):
        images = images.to(device)
        labels = labels.float().to(device)

        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, auc


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_auc: float,
    path: str,
) -> None:
    """Save model checkpoint to disk."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_auc": val_auc,
    }, path)


def train(config: Config) -> Dict[str, List[float]]:
    """Full training loop — loads data, trains, saves checkpoints."""

    # Reproducibility
    
    set_seed(config.seed)
    print(f"Device: {config.device}")
    print(f"Backbone frozen: {config.freeze_backbone}")

    # Dataset — merge train/val then re-split 80/20
    # The original Kaggle split is unreliable, so we merge and re-split

    dataset_train_folder = ChestXRayDataset(config.data_dir / "train", config.image_size)
    dataset_val_folder = ChestXRayDataset(config.data_dir / "val", config.image_size)

    all_image_paths = dataset_train_folder.image_paths + dataset_val_folder.image_paths
    all_labels = dataset_train_folder.labels + dataset_val_folder.labels

    combined_dataset = ChestXRayDataset.__new__(ChestXRayDataset)
    combined_dataset.image_paths = all_image_paths
    combined_dataset.labels = all_labels
    combined_dataset.image_size = config.image_size
    combined_dataset.resizer = dataset_train_folder.resizer

    # Stratified 80/20 split
    indices = list(range(len(combined_dataset)))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=all_labels,
        random_state=42,
    )

    train_dataset = Subset(combined_dataset, train_indices)
    val_dataset = Subset(combined_dataset, val_indices)

    # Class counts in each split
    train_labels = [all_labels[i] for i in train_indices]
    val_labels = [all_labels[i] for i in val_indices]
    train_normal = train_labels.count(0)
    train_pneumonia = train_labels.count(1)
    val_normal = val_labels.count(0)
    val_pneumonia = val_labels.count(1)

    print(f"\nCombined train+val: {len(combined_dataset)} images")
    print(f"Train split: {len(train_dataset)} images "
          f"(Normal: {train_normal}, Pneumonia: {train_pneumonia})")
    print(f"Val split:   {len(val_dataset)} images "
          f"(Normal: {val_normal}, Pneumonia: {val_pneumonia})")

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=(config.device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=(config.device.type == "cuda"),
    )

    # Model
   
    model = PneumoniaClassifier(config).to(config.device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Loss, Optimizer, Scheduler
    
    # Give more weight to the minority class (pneumonia)
    pos_weight = torch.tensor([train_normal / train_pneumonia], device=config.device)
    print(f"  pos_weight (normal/pneumonia): {pos_weight.item():.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=config.scheduler_patience,
        factor=config.scheduler_factor, min_lr=config.scheduler_min_lr,
    )

    # Training loop
    
    history = {
        "train_loss": [], "train_auc": [],
        "val_loss": [], "val_auc": [], "lr": [],
    }

    best_val_auc = 0.0
    epochs_without_improvement = 0

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    start_time = time.perf_counter()

    for epoch in range(1, config.num_epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{config.num_epochs} (lr={current_lr:.2e})")

        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device
        )

        val_loss, val_auc = validate(model, val_loader, criterion, config.device)

        scheduler.step(val_auc)

        history["train_loss"].append(train_loss)
        history["train_auc"].append(train_auc)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["lr"].append(current_lr)

        print(f"  Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   AUC: {val_auc:.4f}")

        # Save checkpoint for every epoch (overwrites last)
        save_checkpoint(
            model, optimizer, epoch, val_auc,
            str(config.checkpoint_dir / "last_model.pth"),
        )

        # Save best model separately
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_without_improvement = 0
            save_checkpoint(
                model, optimizer, epoch, val_auc,
                str(config.checkpoint_dir / "best_model.pth"),
            )
            print(f"  *** New best model saved (AUC: {val_auc:.4f}) ***")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 60)
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Best Validation AUC: {best_val_auc:.4f}")
    print(f"Checkpoints saved to: {config.checkpoint_dir.resolve()}")
    print("=" * 60)

    save_training_curves(history, config.output_dir / "training_curves.png")

    return history


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TorchXRayVision DenseNet121")
    parser.add_argument("--unfreeze", action="store_true",
                        help="Unfreeze backbone for full fine-tuning")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    args = parser.parse_args()

    config = Config()

    if args.unfreeze:
        config.freeze_backbone = False
        # Lower LR for full fine-tuning to avoid damaging pretrained weights
        if args.lr is None:
            config.learning_rate = 1e-4
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    train(config)


if __name__ == "__main__":
    main()
