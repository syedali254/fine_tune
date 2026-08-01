import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from image_branch.inference import model, device, XRV_TO_OURS

OURS_TO_XRV = {v: k for k, v in XRV_TO_OURS.items()}

target_layer = model.features[10][-1][5]
cam = GradCAM(model=model, target_layers=[target_layer])


def grad_cam_heatmap(image_path: str, disease: str) -> np.ndarray:
    img = Image.open(image_path).convert("L").resize((224, 224))
    rgb = np.array(img.convert("RGB"), dtype=np.float32) / 255.0

    arr = np.array(img, dtype=np.float32)
    arr = xrv.datasets.normalize(arr, maxval=255)
    tensor = torch.from_numpy(arr[None, ...]).unsqueeze(0).to(device)

    target_index = list(model.pathologies).index(OURS_TO_XRV[disease])

    grayscale = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(target_index)])

    return show_cam_on_image(rgb, grayscale[0, :], use_rgb=True)
