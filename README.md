
# BioBERT Fine-Tuning — Chest Disease Text Classifier

Fine-tuned BioBERT for 8-class chest disease classification from symptom descriptions. Achieved Macro F1 ≈ 0.96 on the test set.

## Model Weights

Too large for GitHub. Download `best_model.pt` from:
https://drive.google.com/file/d/16ij_cjh-aG_pI0NTw2jA32nH-t-_4OGv/view?usp=drive_link

Place it in the root directory before running inference.

## Structure

```
dataset_stages/        # raw labeled data → encoded → converted to symptom sentences
split_dataset/         # final train.csv, validation.csv, test.csv
pre_processing_dataset.ipynb   # data cleaning and sentence generation
data_preparation.ipynb         # label encoding and stratified splitting
biobert_finetuning.ipynb       # full training notebook (Colab)
train_clean.py                 # minimal training script (essential steps only)
inference.py                   # load weights and predict from symptom text
inference_biobert.ipynb        # notebook version of inference
```

## Diseases

Atelectasis, Emphysema, Hiatal Hernia, Pleural Effusion, Pneumonia, Pneumothorax, Pulmonary Congestion, Pulmonary Fibrosis

## Quick Inference

```python
from inference import predict

result = predict("The patient presents with shortness of breath, fever, and cough.")
print(result)
```

Output is a dictionary of 8 calibrated probabilities summing to 1 (temperature scaling T=3 applied).

## Part of

Explainable Multimodal Chest Disease AI System — text branch only.
```
