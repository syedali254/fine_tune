"""
Model definition for fine-tuning TorchXRayVision DenseNet121.

Architecture:
    - Pretrained DenseNet121 feature extractor (from TorchXRayVision)
    - New binary classification head (single logit output)
"""

import torch
import torch.nn as nn
import torchxrayvision as xrv

from pneumonia_config import Config


class PneumoniaClassifier(nn.Module):
    """Binary pneumonia classifier built on TorchXRayVision DenseNet121.

    The pretrained multi-label head is removed and replaced with a
    single-output binary classification head.

    """

    def __init__(self, config: Config) -> None:
        super().__init__()

        # Load pretrained model
        base_model = xrv.models.DenseNet(weights=config.pretrained_weights)

        # Extract the feature backbone (everything except the classifier)
        self.features = base_model.features

        # The DenseNet121 final feature map has 1024 channels
        # (after the last dense block + final batch norm)
        self.num_features = 1024

        # New binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        #set  backbone freezing
        self.set_backbone_frozen(config.freeze_backbone)

    def set_backbone_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze the DenseNet backbone."""
        for param in self.features.parameters():
            param.requires_grad = not frozen
        self._backbone_frozen = frozen

    @property
    def backbone_frozen(self) -> bool:
        return self._backbone_frozen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing a single logit per sample.
        """
        features = self.features(x)
        out = torch.nn.functional.relu(features, inplace=True)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def load_model_for_eval(checkpoint_path: str, config: Config) -> nn.Module:
    """Load a fine-tuned model from checkpoint for evaluation.

    """
    model = PneumoniaClassifier(config)
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(config.device)
    return model
