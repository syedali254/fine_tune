"""
Configuration for TorchXRayVision Fine-Tuning Pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


@dataclass
class Config:
    """Central configuration for training and evaluation."""

    # ── Paths ──────────────────────────────────────────────────────────────
    data_dir: Path = Path("chest_xray")
    checkpoint_dir: Path = Path("checkpoints")
    output_dir: Path = Path("results")

    # ── Model ──────────────────────────────────────────────────────────────
    pretrained_weights: str = "densenet121-res224-all"
    image_size: int = 224
    freeze_backbone: bool = True  # Set False for full fine-tuning

    # ── Training ───────────────────────────────────────────────────────────
    batch_size: int = 32
    num_epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0

    # ── Scheduler ──────────────────────────────────────────────────────────
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6

    # ── Early Stopping ─────────────────────────────────────────────────────
    early_stopping_patience: int = 7

    # ── Evaluation ─────────────────────────────────────────────────────────
    threshold: float = 0.5

    # ── Reproducibility ────────────────────────────────────────────────────
    seed: int = 42

    # ── Device ─────────────────────────────────────────────────────────────
    device: torch.device = field(default_factory=lambda: torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ))

    # ── Checkpoint to evaluate (None = pretrained baseline) ────────────────
    eval_checkpoint: Optional[Path] = None

    def __post_init__(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
