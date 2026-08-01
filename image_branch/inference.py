import torch
import torchxrayvision as xrv
from PIL import Image
import numpy as np

model = xrv.models.get_model("densenet121-res224-all")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

DISEASE_NAMES = [
    "Atelectasis",
    "Emphysema",
    "Hiatal Hernia",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
    "Pulmonary Congestion",
    "Pulmonary Fibrosis",
]

XRV_TO_OURS = {
    "Atelectasis": "Atelectasis",
    "Emphysema": "Emphysema",
    "Hernia": "Hiatal Hernia",
    "Effusion": "Pleural Effusion",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
    "Edema": "Pulmonary Congestion",
    "Fibrosis": "Pulmonary Fibrosis",
}


def predict_image(image_path: str) -> dict:
    img = Image.open(image_path).convert("L")
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32)
    img = xrv.datasets.normalize(img, maxval=255)
    img = img[None, ...]

    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    xrv_labels = model.pathologies
    result = {d: 0.0 for d in DISEASE_NAMES}

    for i, label in enumerate(xrv_labels):
        if label in XRV_TO_OURS:
            result[XRV_TO_OURS[label]] = float(probs[i])

    return result
