
import os
import json
import time
import zipfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ── Configuration ──

PROJECT_ROOT = "/content/drive/MyDrive/BioBERT_Project"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
ZIP_PATH = os.path.join(PROJECT_ROOT, "split_dataset.zip")

TRAIN_PATH = os.path.join(DATA_DIR, "split_dataset/train.csv")
VAL_PATH = os.path.join(DATA_DIR, "split_dataset/validation.csv")
TEST_PATH = os.path.join(DATA_DIR, "split_dataset/test.csv")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_model.pt")
HISTORY_PATH = os.path.join(LOG_DIR, "training_history.json")

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

DISEASE_NAMES = [
    "Atelectasis",          # 0
    "Emphysema",            # 1
    "Hiatal Hernia",        # 2
    "Pleural Effusion",     # 3
    "Pneumonia",            # 4
    "Pneumothorax",         # 5
    "Pulmonary Congestion", # 6
    "Pulmonary Fibrosis",   # 7
]
NUM_CLASSES = len(DISEASE_NAMES)

TEXT_COL = "text"
LABEL_COL = "label"

# Training hyperparameters
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
DROPOUT_RATE = 0.3

# ── Device ──

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── Dataset ──

class ChestDiseaseDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.texts = dataframe[TEXT_COL].astype(str).tolist()
        self.labels = dataframe[LABEL_COL].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):  # helps return the single item from dataset its input_id,attention_mask and label
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ── Model ──

class BioBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes=8, dropout_rate=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask): # defines the forward pass of the model, taking input_ids and attention_mask as inputs and return raw logits
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_vector = outputs.last_hidden_state[:, 0, :] #0th index of last_hidden_state corresponds to the [CLS] (special classification) token representation
        cls_vector = self.dropout(cls_vector)
        logits = self.classifier(cls_vector)
        return logits


# ── Training and evaluation functions ──

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device): # helper func to run the single epoch 
    
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask) # 1st let the model make the prediction in the training
        loss = criterion(logits, labels) # then find the loss 

        loss.backward()  #then move backward to compute the direction of loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step() #then update the weights 
        scheduler.step() 

        total_loss += loss.item()  # keep track of the total loss for the epoch
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx+1:>3}/{len(loader)} "
                  f"| Batch Loss: {loss.item():.4f} "
                  f"| LR: {scheduler.get_last_lr()[0]:.2e}")

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, accuracy, macro_f1, all_preds, all_labels


def main():
    # ── Mount Google Drive ──
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        print("Google Drive mounted.")
    except ImportError:
        print("Not in Colab; skipping Drive mount.")

    # ── Create required directories ──
    for d in [CHECKPOINT_DIR, LOG_DIR, OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Extract dataset ──
    print(f"Extracting {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)
    print("Extraction complete.")

    # ── Load CSVs ──
    print("Loading datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # ── Tokenizer ──
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # ── Datasets and DataLoaders ──
    train_dataset = ChestDiseaseDataset(train_df, tokenizer, max_length=MAX_LENGTH)
    val_dataset = ChestDiseaseDataset(val_df, tokenizer, max_length=MAX_LENGTH)
    test_dataset = ChestDiseaseDataset(test_df, tokenizer, max_length=MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # ── Model ──
    print(f"Loading BioBERT backbone: {MODEL_NAME}")
    model = BioBERTClassifier(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
    ).to(device)

    # ── Class weights ──
    train_labels = train_df[LABEL_COL].values
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=train_labels,
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # ── Loss ──
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # ── Optimizer with parameter groups ──
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    # ── Scheduler ──
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training history ──
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "best_epoch": None,
        "best_val_macro_f1": None,
    }

    best_val_macro_f1 = -1.0

    print(f"\nStarting training: {NUM_EPOCHS} epochs, {len(train_loader)} batches/epoch")

    # ── Training loop ──
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )

        val_loss, val_acc, val_macro_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        epoch_time = time.time() - epoch_start

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_macro_f1:.4f}")
        print(f"  Time: {epoch_time:.1f}s")

        history["train_loss"].append(round(train_loss, 6))
        history["train_accuracy"].append(round(train_acc, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_accuracy"].append(round(val_acc, 6))
        history["val_macro_f1"].append(round(val_macro_f1, 6))

        # Save best checkpoint
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            history["best_epoch"] = epoch
            history["best_val_macro_f1"] = round(best_val_macro_f1, 6)

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_macro_f1,
                "val_loss": val_loss,
                "disease_names": DISEASE_NAMES,
                "num_classes": NUM_CLASSES,
                "model_name": MODEL_NAME,
            }, BEST_MODEL_PATH)

            print(f"  -> Best model saved (F1: {val_macro_f1:.4f})")
        else:
            print(f"  -> No improvement (best: {best_val_macro_f1:.4f}, epoch {history['best_epoch']})")

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_macro_f1": val_macro_f1,
            "val_loss": val_loss,
            "disease_names": DISEASE_NAMES,
            "num_classes": NUM_CLASSES,
            "model_name": MODEL_NAME,
        }, LAST_MODEL_PATH)

    # ── Save training history ──
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best epoch: {history['best_epoch']}, Best Val Macro F1: {history['best_val_macro_f1']:.4f}")

    # ── Final evaluation on test set ──
    print("\nLoading best checkpoint for final evaluation...")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_loss, test_acc, test_macro_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nTest Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Macro F1: {test_macro_f1:.4f}")

    print("\nPer-class classification report:")
    print(classification_report(
        test_labels, test_preds,
        target_names=DISEASE_NAMES, digits=4, zero_division=0,
    ))

    cm = confusion_matrix(test_labels, test_preds)
    print("Confusion matrix (rows=actual, columns=predicted):")
    print("  " + " ".join([f"{name[:6]:>7}" for name in DISEASE_NAMES]))
    for i, row in enumerate(cm):
        print(f"{DISEASE_NAMES[i][:10]:>10} " + " ".join([f"{v:>7}" for v in row]))

    print("\nPer-class accuracy:")
    for i in range(NUM_CLASSES):
        correct_i = cm[i][i]
        total_i = cm[i].sum()
        acc_i = correct_i / total_i * 100 if total_i > 0 else 0
        print(f"  {DISEASE_NAMES[i]:<25} {correct_i:>4}/{total_i:<4} {acc_i:>6.2f}%")

    print(f"\nResults saved to: {BEST_MODEL_PATH}, {LAST_MODEL_PATH}, {HISTORY_PATH}")


if __name__ == "__main__":
    main()
