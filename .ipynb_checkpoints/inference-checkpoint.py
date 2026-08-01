import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import torch.nn as nn


# ── Model definition (must match training exactly) ──
class BioBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes=8, dropout_rate=0.3):
        super(BioBERTClassifier, self).__init__()
        self.bert       = BertModel.from_pretrained(model_name)
        self.dropout    = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_vector = outputs.last_hidden_state[:, 0, :]
        cls_vector = self.dropout(cls_vector)
        logits     = self.classifier(cls_vector)
        return logits


# ── Constants ──
MODEL_NAME   = "dmis-lab/biobert-base-cased-v1.1"
WEIGHTS_PATH = "best_model.pt"
MAX_LENGTH   = 128
TEMPERATURE  = 3
NUM_CLASSES  = 8

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

# ── Load model and tokenizer once at module level ──
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

model     = BioBERTClassifier(MODEL_NAME, num_classes=NUM_CLASSES)
checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model     = model.to(device)
model.eval()


def predict(text: str) -> dict:
    """
    Takes a symptom sentence and returns calibrated
    disease probabilities ready for the fusion network.

    Args:
        text : symptom description string

    Returns:
        dict with keys = disease names, values = probabilities
        probabilities sum to 1.0
    """
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits        = model(input_ids, attention_mask)
        scaled_logits = logits / TEMPERATURE
        probs         = torch.softmax(scaled_logits, dim=1)
        probs         = probs.squeeze(0).cpu().numpy()

    return dict(zip(DISEASE_NAMES, probs))


if __name__ == "__main__":

    # Confirm checkpoint info
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    print(f"Loaded from epoch:  {checkpoint['epoch']}")
    print(f"Val Macro F1:       {checkpoint['val_macro_f1']:.4f}")
    print(f"Val Loss:           {checkpoint['val_loss']:.4f}")
    print()

    # Run inference
    text   = "The patient presents with shortness of breath, fever, and cough."
    result = predict(text)
    for disease, prob in result.items():
        print(f"{disease:<25} {prob:.4f}")