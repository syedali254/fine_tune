
# BioBERT Fine-Tuning — Chest Disease Text Classifier

Fine-tuned BioBERT for 8-class chest disease classification from symptom descriptions.

## Results

**Best checkpoint: Epoch 4 of 5**

| Metric | Full Test Set (565) | Clean Subset (428) | Overlap Subset (137) |
|---|---|---|---|
| Accuracy | 98.05% | 99.30% | 94.16% |
| Macro F1 | 0.9640 | 0.9813 | 0.8338 |

**Per-class F1 (Full Test Set)**

| Disease | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Atelectasis | 1.00 | 1.00 | 1.00 | 15 |
| Emphysema | 0.83 | 1.00 | 0.91 | 5 |
| Hiatal Hernia | 1.00 | 1.00 | 1.00 | 136 |
| Pleural Effusion | 0.96 | 1.00 | 0.98 | 91 |
| Pneumonia | 0.99 | 0.97 | 0.98 | 182 |
| Pneumothorax | 1.00 | 0.98 | 0.99 | 44 |
| Pulmonary Congestion | 0.97 | 0.96 | 0.97 | 75 |
| Pulmonary Fibrosis | 0.84 | 0.94 | 0.89 | 17 |
| **Macro Avg** | **0.95** | **0.98** | **0.96** | **565** |

**Training history**

| Epoch | Train Loss | Val Loss | Val Accuracy | Val Macro F1 |
|---|---|---|---|---|
| 1 | 1.0247 | 0.1674 | 96.81% | 0.9248 |
| 2 | 0.0883 | 0.1605 | 97.35% | 0.9323 |
| 3 | 0.0692 | 0.1510 | 97.17% | 0.9315 |
| 4 | 0.0474 | 0.1637 | 97.88% | **0.9579** ← saved |
| 5 | 0.0375 | 0.1742 | 97.52% | 0.9546 |

**Note on overlap:** 23% of test samples share exact symptom text with training samples due to the multi-label nature of the source dataset — identical symptom combinations can map to multiple diseases. Clean subset performance (99.30%) being higher than overlap subset (94.16%) confirms the model learned generalised patterns rather than memorising.

**Temperature scaling:** T=3 applied at inference to calibrate overconfident softmax outputs for downstream fusion compatibility.

---

## Model Weights

Too large for GitHub. Download `best_model.pt` from:
https://drive.google.com/file/d/16ij_cjh-aG_pI0NTw2jA32nH-t-_4OGv/view?usp=drive_link

Place it in the root directory before running inference.

---

## Structure

```
dataset_stages/               # raw labeled data → encoded → converted to symptom sentences
split_dataset/                # final train.csv, validation.csv, test.csv
pre_processing_dataset.ipynb  # data cleaning and sentence generation
data_preparation.ipynb        # label encoding and stratified splitting
biobert_finetuning.ipynb      # full training notebook (Colab)
train_clean.py                # minimal training script (essential steps only)
inference.py                  # load weights and predict from symptom text
inference_biobert.ipynb       # notebook version of inference
```

---

## Diseases

Atelectasis, Emphysema, Hiatal Hernia, Pleural Effusion, Pneumonia, Pneumothorax, Pulmonary Congestion, Pulmonary Fibrosis

---

## Quick Inference

```python
from inference import predict

result = predict("The patient presents with shortness of breath, fever, and cough.")
print(result)
```

Output is a dictionary of 8 calibrated probabilities summing to 1 (temperature scaling T=3 applied).

---

## Part of

Explainable Multimodal Chest Disease AI System — text branch only.
```
