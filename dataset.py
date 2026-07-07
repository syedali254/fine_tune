"""
Dataset class for Kaggle Chest X-Ray Images (Pneumonia).

Preprocessing matches the official TorchXRayVision inference pipeline exactly.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import skimage.io
import torch
from torch.utils.data import Dataset
import torchxrayvision as xrv


class ChestXRayDataset(Dataset):
    """PyTorch Dataset for the Kaggle Chest X-Ray (Pneumonia) dataset.

    Expects structure::

        root/
            NORMAL/
            PNEUMONIA/

    Preprocessing (matches official TorchXRayVision pipeline):
        1. Read with skimage.io.imread.
        2. Normalize via xrv.datasets.normalize(img, 255) → [-1024, 1024].
        3. If RGB/RGBA, take first channel (official behavior).
        4. Add channel dimension → (1, H, W).
        5. Resize to 224×224 via XRayResizer.
        6. Convert to float32 tensor.
    """

    def __init__(self, root: Path, image_size: int = 224) -> None:
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self.image_size = image_size
        self.resizer = xrv.datasets.XRayResizer(image_size)

        normal_dir = root / "NORMAL"
        pneumonia_dir = root / "PNEUMONIA"

        if not normal_dir.is_dir() and not pneumonia_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {root} to contain NORMAL/ and/or PNEUMONIA/ subdirectories."
            )

        valid_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

        if normal_dir.is_dir():
            for p in sorted(normal_dir.glob("*")):
                if p.suffix.lower() in valid_ext:
                    self.image_paths.append(p)
                    self.labels.append(0)

        if pneumonia_dir.is_dir():
            for p in sorted(pneumonia_dir.glob("*")):
                if p.suffix.lower() in valid_ext:
                    self.image_paths.append(p)
                    self.labels.append(1)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.image_paths[idx]
        label = self.labels[idx]
        img = self._preprocess(path)
        return img, label

    def _preprocess(self, path: Path) -> torch.Tensor:
        """Load and preprocess matching official TorchXRayVision pipeline."""
        img = skimage.io.imread(str(path))
        img = xrv.datasets.normalize(img, 255)

        # Official pipeline: take first channel for multi-channel images
        if img.ndim == 3:
            img = img[:, :, 0]

        # Add channel dimension → (1, H, W)
        img = img[None, :, :]

        # Resize to target size
        img = self.resizer(img)

        return torch.from_numpy(img).float()

    def get_filename(self, idx: int) -> str:
        return self.image_paths[idx].name

    @property
    def num_normal(self) -> int:
        return self.labels.count(0)

    @property
    def num_pneumonia(self) -> int:
        return self.labels.count(1)
