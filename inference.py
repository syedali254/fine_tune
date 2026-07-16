"""
Inference + Grad-CAM for the fine-tuned pneumonia classifier.

Reuses existing model and preprocessing code so predictions stay
consistent with training/evaluation.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import skimage.io
import torch
import torch.nn.functional as F
import torchxrayvision as xrv

from pneumonia_config import Config
from pneumonia_model import load_model_for_eval


CHECKPOINT_PATH: Path = Path("checkpoints/best_model.pth")


def load_model(
    checkpoint_path: str = str(CHECKPOINT_PATH),
    config: Config = None,
) -> torch.nn.Module:
    """Load the fine-tuned model in eval mode."""
    if config is None:
        config = Config()
    model = load_model_for_eval(checkpoint_path, config)
    model.eval()
    return model


def preprocess_image(image_path: Path) -> torch.Tensor:
    """Preprocess a chest X-ray exactly like training/evaluation."""
    img = skimage.io.imread(str(image_path))
    img = xrv.datasets.normalize(img, 255)

    if img.ndim == 3:
        img = img[:, :, 0]

    img = img[None, :, :]
    img = xrv.datasets.XRayResizer(224)(img)
    return torch.from_numpy(img).float()


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device = None,
) -> Tuple[str, float, float]:
    """Run inference: returns (class, pneumonia_prob, confidence%)."""
    if device is None:
        device = next(model.parameters()).device

    batch = image_tensor.unsqueeze(0).to(device)
    logits = model(batch).squeeze(1)
    prob = torch.sigmoid(logits).item()

    predicted_class = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    confidence = max(prob, 1.0 - prob) * 100.0

    return predicted_class, prob, confidence


def predict_from_path(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device = None,
) -> Tuple[str, float, float]:
    """Preprocess + predict in one call."""
    tensor = preprocess_image(image_path)
    return predict(model, tensor, device)


# Grad-CAM


class GradCAM:
    """
    Grad-CAM heatmap for our pneumonia classifier.

    Uses the last convolutional block's feature maps to show
    which parts of the X-ray the model is looking at.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        # The last dense block in DenseNet — its output is (B, 1024, 7, 7)
        self.target_module = model.features.denseblock4
        self.activations = None

       
        def save_activations(module, input, output):
            self.activations = output

        self._fwd_handle = self.target_module.register_forward_hook(save_activations)

    def generate(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap.

        Returns:
            (224, 224) numpy array in [0, 1].
        """
        self.activations = None

        # Forward pass (with gradients this time)
        batch = image_tensor.unsqueeze(0)
        batch.requires_grad_(True)
        output = self.model(batch)

        if self.activations is None:
            raise RuntimeError("Forward hook did not fire.")

        
        gradients = torch.autograd.grad(
            outputs=output[0, 0],  # the single pneumonia logit
            inputs=self.activations,
            retain_graph=True,
        )[0]

        # Global-average-pool gradients → importance weight per channel
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Now safe to detach activations
        activations = self.activations.detach()

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to 224 x 224
        cam = F.interpolate(
            cam, size=(224, 224), mode="bilinear", align_corners=False
        )

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def remove_hooks(self):
        """Clean up."""
        self._fwd_handle.remove()


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap (jet colormap) onto the original X-ray.

    Args:
        original_image: (H, W) or (H, W, 3) uint8 array.
        heatmap: (224, 224) float array in [0, 1].
        alpha: blending strength of the heatmap.

    Returns:
        (H, W, 3) uint8 overlay image.
    """
    import matplotlib.cm as cm

    # Get original image as 2-D grayscale
    if original_image.ndim == 3:
        original_image = original_image[:, :, 0]

    h, w = original_image.shape

    # Resize heatmap to original image size
    from skimage.transform import resize
    heatmap_resized = resize(heatmap, (h, w), mode="constant", preserve_range=True)

    # Apply jet colormap → (H, W, 3) in [0, 1]
    colored = cm.jet(heatmap_resized)[:, :, :3]

    # Normalize original to [0, 1]
    original_norm = original_image.astype(np.float32) / 255.0
    original_rgb = np.stack([original_norm] * 3, axis=-1)

    # Blend
    overlay = (1 - alpha) * original_rgb + alpha * colored
    overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    return overlay
