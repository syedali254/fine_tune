"""
Inference for the fine-tuned BioBERT binary pneumonia classifier (Experiment 1).

Loads the trained checkpoint in text_branch/best_model.pt and maps
free-text symptoms -> P(non-pneumonia) / P(pneumonia) / prediction.

The checkpoint contains the full fine-tuned BioBERT backbone plus a
2-class linear head. Training used CLS-token pooling (verified against
the checkpoint: the pooler is untouched while the encoder is fine-tuned).
"""

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
CHECKPOINT_PATH = Path("text_branch/best_model.pt")
MAX_LENGTH = 128
NUM_CLASSES = 2


class BioBERTClassifier(nn.Module):
    """BioBERT backbone + single linear classification head (2 classes)."""

    def __init__(self, model_name: str = MODEL_NAME, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token (matches training)
        return self.classifier(pooled)


def load_text_model(
    checkpoint_path: str = str(CHECKPOINT_PATH),
    device: torch.device = None,
) -> nn.Module:
    """Load the trained BioBERT classifier in eval mode."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BioBERTClassifier()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    return model


def load_text_tokenizer(model_name: str = MODEL_NAME) -> AutoTokenizer:
    """Load the tokenizer matching the trained BioBERT checkpoint."""
    return AutoTokenizer.from_pretrained(model_name)


@torch.no_grad()
def predict_text(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device = None,
    max_length: int = MAX_LENGTH,
) -> Tuple[float, float, str]:
    """Tokenize, run inference and return (p_normal, p_pneumonia, prediction)."""
    if device is None:
        device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    logits = model(input_ids, attention_mask)
    probs = torch.softmax(logits, dim=1)[0]
    p_normal = probs[0].item()
    p_pneumonia = probs[1].item()
    predicted_class = "Pneumonia" if p_pneumonia >= 0.5 else "Non-pneumonia"

    return p_normal, p_pneumonia, predicted_class