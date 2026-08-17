# KNOWLEDGE.md

**Project:** Domain Shift in Chest X-Ray Analysis — MSc research project.
**Root:** `C:\Ai_lab\torch`
**Generated / last verified:** 2026-08-16, from review of project files only.

**Single rule:** Someone opening this file must be able to reconstruct exactly what was actually built, trained, tested, and observed — without guessing and without inventing anything. Every value below was read from a project file, checkpoint, saved CSV/JSON, notebook, or git commit, or recomputed from a saved artifact (marked `DERIVED`). Anything planned or unverifiable is explicitly labelled `PLANNED / NOT IMPLEMENTED` or `NOT VERIFIED FROM PROJECT FILES`.

---

## 1. Project Overview

**Project aim (stated in code/UI):** a multimodal demo for pneumonia screening combining a chest X-ray classifier, a symptom-text classifier, and a clinical-evidence extractor. Deliberately framed as a research prototype, **not** for clinical use.

**Research problem (stated in code/UI):** "Domain Shift in Chest X-Ray Analysis" (the app sidebar caption). No further elaboration exists in project files. **IMPORTANT:** despite the title, **no domain-shift experiment exists anywhere in the repository** — the image model is trained and evaluated on a single dataset only.

**Overall system architecture:** three independent analysis branches run in parallel on one button click, each producing its own evidence output. **No fusion is applied** (`app.py` docstring + UI). Concretely: `No multimodal probability fusion is implemented.`

- **Module 1 — Image branch:** fine-tuned DenseNet121 (TorchXRayVision) + Grad-CAM.
- **Module 2 — Text branch:** Experiment 1 = fine-tuned BioBERT symptom classifier; Experiment 2 = deterministic clinical-finding extraction (evidence engine).
- **Module 3:** **NOT IMPLEMENTED** — only a Future-Work mention of "structured clinical variable intake and validated risk scoring (CURB-65/PSI)" in `text_branch/experiment2/README.md`.

**Environment (verified at audit time, local machine):** Windows, Python `3.13.3`, PyTorch `2.11.0+cpu` (no CUDA), `transformers 4.51.3` (notebook pinned `4.40.0` — see §7/§14), `streamlit 1.56.0`, `torchxrayvision 1.4.0`, scikit-learn `1.6.1`, numpy, matplotlib, skimage. BioBERT base model/tokenizer (`dmis-lab/biobert-base-cased-v1.1`) load from the HuggingFace cache (`C:\Users\Hp\.cache\huggingface`); **not stored in the repo.**

**Version control (git):** 3 commits: `c8a7cb3` (fine-tuning pipeline) → `fa6679b` (initial commit) → `49ebea0` ("newer version … streamlit ui + GradCam"). Working tree: `app.py` modified; untracked: `BioBERT_Text_Dataset_Encoded.csv`, `KNOWLEDGE.md`, `evidence_inference.py`, `report/`, `text_branch/`, `text_inference.py`. `chest_xray/` is gitignored.

---

## 2. Research/Technical Evolution (Stages 1–4)

| Stage | What was done | Where it lives | Status |
|---|---|---|---|
| **Stage 1 — Image baseline** | Pretrained DenseNet121 (`densenet121-res224-all`) evaluated on `chest_xray/test` with **no fine-tuning**. | `baseline_results/` (`evaluate.py`, `predictions.csv`, `threshold_results.csv`, `roc_curve.png`, `confusion_matrix.png`) | Complete |
| **Stage 2 — Image fine-tuning** | Merged `train`+`val` folders (5232 images) re-split stratified 80/20 (seed 42); fine-tuned new head on frozen backbone (config default); checkpoint `best_model.pth` at epoch 14 with `val_auc 0.9962`. Evaluated on `chest_xray/test`. | `train.py`, `results/finetuned/`, `checkpoints/` | Complete |
| **Stage 3 — Text Experiment 1** | BioBERT (CLS-pooled) binary symptom classifier trained on Colab (T4 GPU, `transformers==4.40.0`) from the encoded symptom CSV; test acc 0.9947 / AUC 1.0. | `text_branch/biobert_pneumonia.ipynb`, `text_branch/best_model.pt`, `experiment_summary.json`, `test_results.json`, `text_inference.py` | Complete |
| **Stage 4 — Text Experiment 2 + Demo** | Deterministic knowledge-base evidence engine (interpretable symptom-level reasoning) + Streamlit app integrating all three branches with **no fusion**. | `text_branch/experiment2/`, `evidence_inference.py`, `app.py` | Exp2 = POC; demo complete |
| **Planned (not implemented)** | Image last-DenseNet-block + classifier fine-tuning; Module 3 (CURB-65/PSI risk scoring); knowledge-grounded LLM extraction. | — | `PLANNED / NOT IMPLEMENTED` (no code or results exist) |

---

## 6. Module 1 — Image Branch

### Dataset

- **Name/source (only citation in code):** "Kaggle Chest X-Ray Images (Pneumonia)" — `chest_xray/` (gitignored).
- **Exact image counts (verified by directory listing; JPEG only):**
  - `train/NORMAL` **1341**, `train/PNEUMONIA` **3875**
  - `val/NORMAL` **8**, `val/PNEUMONIA` **8**
  - `test/NORMAL` **234**, `test/PNEUMONIA` **390**
  - Total **5856** JPEGs (+ 8 `.DS_Store` files).
- **Classes:** `NORMAL` (0), `PNEUMONIA` (1).
- **Split actually used for training:** `train.py` merges `train/` + `val/` (**5232 images**: 1349 NORMAL + 3883 PNEUMONIA), then re-splits **stratified 80/20, `random_state=42`**. The original `val/` folder is **not** used directly (code comment: "original Kaggle split is unreliable"). Exact per-split class counts were only printed to console, **not persisted** → `NOT VERIFIED FROM PROJECT FILES`.
- **Test set:** `chest_xray/test`, **624 images (234 NORMAL / 390 PNEUMONIA)** — used by `evaluate.py`.
- **Paediatric claim** appears in `app.py` title/docstring and `train.py` docstring. **Resolved by researcher:** the Kaggle dataset documentation describes the collection as genuinely paediatric chest X-rays. Not verifiable from repo files alone; formal citation still to be added.

### Preprocessing (identical train / eval / inference — verified by code comparison)

1. `skimage.io.imread`
2. `xrv.datasets.normalize(img, 255)` → range `[-1024, 1024]`
3. If `ndim == 3`, take first channel `img[:, :, 0]`
4. Add channel dim → `(1, H, W)`
5. `xrv.datasets.XRayResizer(224)`
6. `torch.from_numpy(...).float()`

Implemented identically in `dataset.py`, `inference.py`, root `evaluate.py`. **CONFLICT:** `baseline_results/evaluate.py` uses `img.mean(axis=-1)` (mean-axis gray) instead of first channel (see §14).

### Model

- `xrv.models.DenseNet(weights="densenet121-res224-all")`; backbone = `model.features`.
- New head: `Linear(1024→256) → ReLU(inplace) → Dropout(0.3) → Linear(256→1)` — single logit output.
- `freeze_backbone` default `True` (config); `--unfreeze` sets it False and lowers LR to `1e-4`.
- Parameters (DERIVED from `checkpoints/best_model.pth` state dict):
  - Total state-dict elements: **7,294,010** (includes batch-norm buffers).
  - Classifier head: **262,657** (verified: `1024·256+256 = 262,400` + `256+1 = 257` = 262,657).
  - Backbone (`features`): **7,031,353**.
  - The dissertation draft's "total 7,210,241" is the parameter-only count (excluding BN buffers; difference 83,769 ≈ buffer elements) — consistent with the verified state dict, but **only 7,294,010 / 262,657 are directly verified from the checkpoint**.
- **What was frozen / trained in the final run: `NOT VERIFIED FROM PROJECT FILES`** (the checkpoint records no such flag; config default says frozen backbone).

### Baseline (pretrained DenseNet121, no fine-tuning)

Artifacts in `baseline_results/`. Test set 624 (234/390). Metrics read from `baseline_results/threshold_results.csv`.

| Threshold | Acc | Prec | Recall | Spec | F1 |
|---|---|---|---|---|---|
| 0.50 | 0.4311 | 0.9487 | 0.0949 | 0.9915 | 0.1725 |
| 0.10 (best F1) | 0.5385 | 0.8542 | 0.3154 | 0.9103 | 0.4607 |

- ROC-AUC **0.7210** (DERIVED from `baseline_results/predictions.csv`; not stored elsewhere).
- CM @0.5 (DERIVED): TN 232, FP 2, FN 353, TP 37.
- **Which script produced these artifacts: `NOT VERIFIED FROM PROJECT FILES`** — the saved sweep range (0.10–0.90) matches `baseline_results/evaluate.py`, but that script writes to `chest_xray/` per its code while artifacts live in `baseline_results/`; root `evaluate.py` would sweep 0.05–0.95 (see §14).

### Fine-tuned Model

Artifacts in `results/finetuned/`. Test set 624 (234/390).

- **Exact training configuration (`pneumonia_config.py` + `train.py`):** batch_size 32, num_epochs 15, lr 1e-3, weight_decay 1e-4, Adam (on `requires_grad` params), `BCEWithLogitsLoss(pos_weight = train_normal/train_pneumonia ≈ 0.35; exact value not persisted)`, `ReduceLROnPlateau(mode=max, patience 5, factor 0.5, min_lr 1e-6)`, early-stopping patience 7, seed 42.
- **Checkpoints (`checkpoints/`):** `best_model.pth` and `last_model.pth` both record `epoch=14`, `val_auc=0.9962009628676296`, size 31,554,294 bytes each — but their MD5 hashes differ (`07cd3e7552064cd0c692a677b4f92d46` vs `c6c88d7a613f3ded48855884dbfa54ee`), so they are **NOT byte-identical** (contradicts the draft's "byte-identical" claim; see §14).
  - **Training stopped at epoch 14 of configured 15** — reason (`--epochs`? interrupt? crash?) `NOT VERIFIED FROM PROJECT FILES`.
  - Note: `val_auc 0.9962` is measured on the **20% holdout of merged train+val** — **not** the held-out test folder.
- **Exact evaluation metrics (from `results/finetuned/threshold_results.csv`):**

| Threshold | Acc | Prec | Recall | Spec | F1 |
|---|---|---|---|---|---|
| 0.50 (deployed) | 0.8317 | 0.7902 | 0.9949 | 0.5598 | 0.8808 |
| 0.95 (best F1) | 0.9199 | 0.9187 | 0.9564 | 0.8590 | 0.9372 |

- ROC-AUC **0.9619** (DERIVED from `results/finetuned/predictions.csv`; not stored in any CSV/JSON).
- CM @0.5 (DERIVED): TN 131, FP 103, FN 2, TP 388.
- **Threshold used at deploy:** 0.5 (`inference.py`/`app.py`) — a conventional sigmoid default, **not** the best-F1 threshold (0.95).
- **Training curves:** `results/` contains NO `training_curves.png` (verified listing). Any draft reference to training curves is unverifiable.

### Grad-CAM

- Implemented in `inference.py` (class `GradCAM`).
- **Target layer:** `model.features.denseblock4` — last dense block, output `(B, 1024, 7, 7)` (last conv stage before global pooling).
- **Heatmap generation:** forward hook captures activations → `autograd.grad` of the pneumonia logit `output[0,0]` w.r.t. activations → GAP of gradients (per-channel weights) → weighted sum of activations → `ReLU` → bilinear upsampling to 224×224 → min-max normalized to [0,1].
- Overlay: jet colormap, alpha 0.45, resized to original image (`overlay_heatmap`). In `app.py`, Grad-CAM failure degrades gracefully (warning, no overlay). **Qualitative only — no quantitative CAM evaluation exists.**

### Pending Experiment — Last DenseNet block + classifier fine-tuning

- **Status: `PLANNED / NOT IMPLEMENTED`.** No code, script, checkpoint, or result for this experiment exists in the repository.
- Planned comparison (from project plan): fine-tune **only `denseblock4` + the new classifier head**, earlier blocks frozen — vs the frozen-backbone model and the full-unfreeze mode. A plan, not something already built or measured.

---

## 7. Module 2 — Text Experiment 1

### Dataset

- **File:** `BioBERT_Text_Dataset_Encoded.csv` (project root, 481 KB, untracked). Columns: `text, disease, label`.
- **Original size:** **3766 rows** + header.
- **Original disease/category counts (8 classes):** pneumonia **1212**, hiatal hernia **906**, pleural effusion **607**, pulmonary congestion **497**, pneumothorax **296**, pulmonary fibrosis **118**, atelectasis **99**, emphysema **31**.
- **`label` column is the disease index 0–7, NOT a binary target.** The binary target (pneumonia = 1) is derived from `disease == 'pneumonia'` during processing.
- **Exact filtering performed (notebook):** removed **27 rows** where identical texts carried conflicting disease labels.
- **Final size:** **3739 rows = 1200 pneumonia / 2539 non-pneumonia** (≈2.11:1 imbalance).
- **Split (group-aware, zero leakage — verified from `experiment_summary.json`):** train **3349** (1082 pos / 2267 neg), val **200** (60 / 140), test **190** (58 / 132). Reported leakage: **0**.
- **Provenance (resolved by researcher):** derived from the public Kaggle dataset **`Final_Augmented_dataset_Diseases_and_Symptoms`** — a tabular dataset (symptom columns, disease target). Processing: (1) map to ~8 chest-disease categories, (2) keep symptom columns co-occurring with those 8 diseases, (3) convert tabular rows to sentences (e.g. "The patient presents with fever, cough, and chills."), (4) binary target pneumonia=1 / other 7=0. The notebook loaded the processed file from a Google Drive path. **Exact Kaggle URL/version not recorded; the conversion code is NOT in the repo** → provenance is researcher-documented, not repo-verifiable.
- **"Synthetic" terminology:** the repo's own result files (`experiment_summary.json`, `test_results.json`) attribute the near-perfect performance to the **"synthetic structured nature of symptom sentences generated from disease-specific binary symptom vectors"**. This phrasing is repo-documented and may be used; the dissertation draft additionally describes the corpus as "template-like".

### Experiment 1 — BioBERT Classifier

Source of truth: `text_branch/biobert_pneumonia.ipynb` (Colab, T4 GPU) + local `text_branch/best_model.pt`, `experiment_summary.json`, `test_results.json`, `text_inference.py`.

- **Model:** BioBERT base (`dmis-lab/biobert-base-cased-v1.1`) → Dropout(0.3) → `Linear(768→2)`.
- **Pooling: CLS token** — `last_hidden_state[:, 0]` (NOT `pooler_output`). Reproduces checkpoint predictions (verified live, see below).
- **Tokenizer:** AutoTokenizer, same BioBERT checkpoint; **not stored in project** (HF cache).
- **Fine-tuning:** full fine-tune of encoder + head; AdamW + linear warmup; class weights non_pneumonia **0.7386** / pneumonia **1.5476**.
- **Hyperparameters:** max_length 128, batch_size 16, lr 2e-5, epochs 5, warmup_ratio 0.1, weight_decay 0.01, dropout 0.3, grad_clip 1.0, seed 42, early-stopping patience 3 on val ROC-AUC. **CONFLICT:** notebook pinned `transformers==4.40.0`; local env uses `4.51.3` (checkpoint loads fine under 4.51.3 — verified).
- **Parameters (DERIVED):** **108,311,810** (BERT base + head).
- **Evaluation method:** test set 190 (58 pneumonia / 132 non-pneumonia), softmax over the 2 logits.
- **Exact metrics (from `test_results.json` + `experiment_summary.json`):**

| Metric | Value |
|---|---|
| Accuracy | 0.9947 |
| Precision | 0.9831 |
| Recall | 1.0000 |
| Specificity | 0.9924 |
| F1 | 0.9915 |
| ROC-AUC | 1.0000 |
| Confusion matrix | [[131, 1], [0, 58]] |

- Probability stats: pneumonia mean **0.9998** (min 0.9996, max 0.9998); non-pneumonia mean **0.0078** (min 0.0001, max 0.9988 — the single false positive); **0** uncertain samples.
- **Saved checkpoint (`text_branch/best_model.pt`, ~1.26 GB; keys `epoch, model_state_dict, optimizer_state_dict, val_metrics, config`):** epoch **1**; val_metrics `loss 0.030351, accuracy 0.995, precision 1.0, recall 0.9833, f1 0.9916, roc_auc 1.0, specificity 1.0`; config `seed 42, model_name dmis-lab/biobert-base-cased-v1.1, max_length 128, batch_size 16, learning_rate 2e-5, num_epochs 5, base_dir /content/drive/MyDrive/BioBERT_Project/biobert_pneumonia` (verified directly from the checkpoint; note: the old draft/KNOWLEDGE stated `base_dir` as `.../pneumonia_biobert` — **incorrect**, corrected here).
- **Notebook run history (artifact evidence):** the training cell printed epochs 1–3 and then was interrupted by KeyboardInterrupt; epoch 1 was best (val ROC-AUC 1.0), epochs 2–3 showed "No improvement (2/3)". `early_stopped: false` in `experiment_summary.json` is therefore **consistent with the notebook artifact** (early stopping needs 3 consecutive non-improvements; only 2 were logged before interrupt). The dissertation draft's claim of "early stopping triggered at epoch 2 / run stopped after 2 epochs" is **NOT supported by the saved notebook** (see §14).
- **Inference verified live (local, `text_inference.py`, checkpoint `text_branch/best_model.pt`):**
  - "I have fever, cough, shortness of breath, and chills." → **Pneumonia, P=0.9992** (matches notebook Cell 9).
  - "I have fever, cough and chills." → **Pneumonia, P=0.9987**.
  - "The patient presents with nausea, back pain, heartburn, and regurgitation." → **Non-pneumonia**.
  - "The is having sharp chest pain, drug abuse, and shoulder pain." → **Non-pneumonia**.
  - **CORRECTION:** the bare phrase "fever, cough" → **Non-pneumonia, P=0.4723** (NOT 0.9991). The old "0.9991" figure corresponds to the longer sentence above; do not repeat the bare-phrase claim.

### Experiment 1 — Limitation (analysis, not a stored metric)

- Binary prediction is highly dataset-dependent — near-perfect separation (test acc 0.9947, AUC 1.0, all pneumonia probs ≥ 0.9996) on a small, single-source, filtered **synthetic-structured** corpus indicates the model exploits dataset patterns rather than generalising robustly.
- Similar symptom combinations can produce near-1 probabilities when they match training-distribution patterns.
- Reduced/filtered corpus limits generalization: 8 disease classes, 27 conflicts removed, 3739 rows total; no external corpus, no real-world clinical text.
- No interpretable symptom-level reasoning: output is a single binary probability with no per-symptom attribution.

---

## 8. Module 2 — Text Experiment 2

- **Why introduced:** to provide interpretable, source-grounded, symptom-level reasoning that Experiment 1's black-box binary classifier cannot. Located in `text_branch/experiment2/`.
- **Architecture:** deterministic, rule-based primary path (no ML by default):
  - **Clause splitting:** text normalized and segmented into clauses before matching.
  - **Negation handling:** positional 3-word negation window (e.g., "no fever", "no difficulty breathing").
  - **Knowledge base (`knowledge_base.py`):** findings grouped in categories, each with phrase lists, critical phrases, severity/negatable/overlap flags, and contextual mappings (e.g., chills → fever).
  - **Finding categories:** supportive (cough, fever, chest_pain, sputum); concerning/severity (dyspnea, tachypnea, altered_mental_status); upper-respiratory (runny_nose, nasal_congestion, sneezing, sore_throat); influenza-like (body_aches, headache, fatigue); non-specific (weakness, loss_of_appetite).
  - **Deterministic matching:** word-boundary regex, longest variation first.
  - **BioBERT similarity fallback:** still present (`evidence_engine.py`), threshold 0.85, semantic-anchor guard, **cannot override a deterministic match**, **OFF by default** (`evidence_inference.py` calls `analyze_symptoms(text, use_biobert=False)` and blocks TensorFlow via `sys.modules`). Fallback never exercised → `IMPLEMENTED — EVALUATION NOT VERIFIED`.
  - **Sources cited (KB/README):** CDC, NHLBI, WHO, IDSA/ATS, CURB-65, Metlay 2019, Eccles 2005, Monto 2000.
- **Exact output format (keys):** `findings`, `summary` (supportive/concerning/upper_respiratory/influenza_like/nonspecific counts), `supportive_findings`, `concerning_findings`, `upper_respiratory_findings`, `influenza_like_findings`, `nonspecific_findings`, `unmapped_findings`, `interpretation`, `disclaimer`.
- **Example inputs/outputs (from `tests.py`, 16 cases; suite re-run and confirmed 16/16 passing at audit time):**
  - "I have fever, cough and chills." → cough present, fever present (chills→fever); supportive_count 2.
  - "I have cough but no fever." → cough present, fever **absent** (negated); supportive_count 1.
  - "The patient is having difficulty to breathe." → dyspnea present; concerning_count 1.
  - "The weather is very sunny today and I need to buy groceries." → 0 findings everywhere, 0 false positives.
- **Current limitations:** deterministic coverage bounded by KB phrases; fixed 3-word negation window; unmapped symptoms captured but unclassified; BioBERT fallback path unexercised; not validated on any clinical corpus; import-side-effect handled only in the `evidence_inference.py` wrapper.
- **Current status: POC — pending supervisor confirmation.** Implemented and passing its own 16-case suite (deterministic path), but not validated on clinical text.

### Future Experiment — knowledge-grounded LLM approach

- **Status: `PLANNED / NOT IMPLEMENTED`.** No code or results exist. Planned direction: replace/augment the deterministic extractor with a knowledge-grounded LLM for symptom→finding extraction and clinical reasoning. Do not treat as done.

---

## 9. Streamlit Demonstration

- **File:** `app.py` (current state, 349 lines; modified after the last commit `49ebea0`).
- **UI:** title + description; sidebar with model list and project caption; file uploader (jpg/jpeg/png); free-text symptoms area; "Run Multimodal Assessment" button.
- **Image input → image inference:** upload → `_temp_inference.png` → `preprocess_image` → `predict` (sigmoid logit; class `PNEUMONIA` if prob ≥ 0.5) → displays P(Pneumonia), P(Non-pneumonia), confidence.
- **Text input → text inference:** BioBERT `predict_text` (CLS-pooled) → P(normal), P(pneumonia), argmax class.
- **Grad-CAM:** heatmap overlaid on original X-ray, displayed with caption; failure degrades to a warning.
- **Evidence (Exp 2):** structured section listing supportive / concerning / contextual findings, unmapped findings (expander), counts, interpretation, disclaimer.
- **How outputs are displayed:** three sections + a 3-column "Multimodal View" when both image and text are present; evidence metric shown as total `(supportive, concerning)`, e.g. `3 (2 supportive, 1 concerning)`.
- **Fusion:** **modalities are independent — no fusion** (explicit in docstring and UI).
- **Parallel execution:** three branches run via `ThreadPoolExecutor(max_workers=3)`.
- **Model/checkpoint paths:** image → `checkpoints/best_model.pth` (Config defaults); text → `text_branch/best_model.pt` (via `text_inference.CHECKPOINT_PATH`); evidence → `evidence_inference.extract_evidence` (deterministic, BioBERT off).
- Models cached with `@st.cache_resource`. Disclaimer: research prototype, not for clinical use.
- **Verified:** Streamlit `AppTest` flows (image-only, text-only, image+text+evidence, neither-input) run with 0 exceptions.

---

## 10. Evaluation Methodology

| Branch | Task | Test set | Threshold rule | Metrics & sources |
|---|---|---|---|---|
| Image baseline | Binary pneumonia (DenseNet121, no tuning) | `chest_xray/test`, 624 (234/390) | Sweep stored 0.10–0.90; reported @0.5 and best-F1 (0.10) | `baseline_results/threshold_results.csv` (direct); ROC-AUC 0.7210 & CM (232,2,353,37) DERIVED from `predictions.csv` |
| Image fine-tuned | Binary pneumonia (DenseNet121 + head) | `chest_xray/test`, 624 (234/390) | Sweep 0.05–0.95; reported @0.5 (deployed) and best-F1 (0.95) | `results/finetuned/threshold_results.csv` (direct); ROC-AUC 0.9619 & CM (131,103,2,388) DERIVED from `predictions.csv` |
| Image validation | Val AUC during training | 20% holdout of merged train+val (5,232) | — | Checkpoint `val_auc 0.9962009628676296` (epoch 14) — **not** the test folder |
| Text Exp 1 | Binary pneumonia (BioBERT) | test 190 (58/132) | Softmax argmax (decision at 0.5 on P(pneumonia)) | `test_results.json` + `experiment_summary.json` (direct) |
| Text Exp 2 | Deterministic finding extraction | Own 16-case suite | Rule-based | `tests.py` — 16/16 pass (re-verified) |

Metrics are the standard binary definitions: Accuracy = (TP+TN)/N, Precision = TP/(TP+FP), Recall = TP/(TP+FN), Specificity = TN/(TN+FP), F1 = harmonic mean of Precision/Recall, ROC-AUC from prediction scores. ROC-AUC values for the image models are **not persisted** in the repo and were recomputed from `predictions.csv` (marked `DERIVED`).

---

## 11. Results Master Table

Only verified numbers. `DERIVED` = recomputed from a saved artifact (not stored as such in the repo).

| Experiment | Model | Dataset | Result | Status |
|---|---|---|---|---|
| Image baseline | DenseNet121 pretrained (`densenet121-res224-all`), no tuning | `chest_xray/test`, 624 (234/390) | @0.5: acc 0.4311, prec 0.9487, rec 0.0949, spec 0.9915, f1 0.1725; @0.10 (best F1): acc 0.5385, prec 0.8542, rec 0.3154, spec 0.9103, f1 0.4607; ROC-AUC 0.7210 (DERIVED); CM (232,2,353,37) (DERIVED) | Complete |
| Image fine-tuning | DenseNet121 + head (1024→256→1), backbone frozen (config default) | merged train+val (5232) re-split 80/20 seed 42; test 624 (234/390) | checkpoint epoch 14, val_auc 0.9962; @0.5 (deployed): acc 0.8317, prec 0.7902, rec 0.9949, spec 0.5598, f1 0.8808; @0.95 (best F1): acc 0.9199, prec 0.9187, rec 0.9564, spec 0.8590, f1 0.9372; ROC-AUC 0.9619 (DERIVED); CM (131,103,2,388) (DERIVED) | Complete |
| Image last-block tuning (denseblock4 + head) | — | — | — | `PLANNED / NOT IMPLEMENTED` (no code/results) |
| Text Exp 1 | BioBERT base, CLS-pooled + head (768→2) | filtered CSV 3739 (1200/2539); test 190 (58/132) | acc 0.9947, prec 0.9831, rec 1.0, spec 0.9924, f1 0.9915, ROC-AUC 1.0, CM [[131,1],[0,58]]; all pneumonia probs ≥ 0.9996 | Complete (see limitations §13) |
| Text Exp 2 | Deterministic KB extractor (+ BioBERT fallback, off) | own 16-case test suite | 16/16 pass (deterministic path, re-verified); fallback not exercised | POC (pending supervisor confirmation) |
| Module 3 (CURB-65/PSI) | — | — | — | `PLANNED / NOT IMPLEMENTED` |

---

## 12. Reproducibility

**Environment (verified at audit time):** Python 3.13.3, PyTorch 2.11.0+cpu, transformers 4.51.3, streamlit 1.56.0, torchxrayvision 1.4.0, scikit-learn 1.6.1. No `requirements.txt`, no root README, no lockfile → exact dependency reproduction from the repo alone is not possible; only the notebook pins `transformers==4.40.0`.

- **Image branch:** scripts/config at root with seed 42. But the exact final training run is **NOT fully reproducible from the repo alone**: `chest_xray/` is gitignored (absent in git), per-split class counts were not persisted, `pos_weight` exact value was not persisted, and the epoch-14 stop reason + freeze mode are unverified. **Reproducible from the repo:** the evaluation numbers (they are deterministic outputs of `evaluate.py` on the saved checkpoints; ROC-AUC/CM recomputable from `predictions.csv`).
- **Text Exp 1:** full pipeline in `biobert_pneumonia.ipynb` (Colab, T4, transformers 4.40.0). The encoded CSV is untracked but present locally; the notebook loads it from a Google Drive path (`/content/drive/MyDrive/BioBERT_Project/biobert_pneumonia`). Checkpoint + metrics are in `text_branch/`. Local inference via `text_inference.py` reproduces the notebook's inference outputs (verified live). Original Kaggle conversion code is **not** in the repo.
- **Text Exp 2:** deterministic, fully reproducible from `text_branch/experiment2/`; `tests.py` passes 16/16.
- **Streamlit:** `app.py` + checkpoints + HF cache reproduce the demo; requires the two checkpoints and the HuggingFace model cache.
- **External dependencies (outside repo):** HuggingFace cache (`models--dmis-lab--biobert-base-cased-v1.1`); `C:\Ai_lab\demo_2\` (a minimal demo copy outside this repo).
- **Artifact inventory (verified listing, excluding `chest_xray/`, `.git/`, `__pycache__/`):** root scripts `app.py, dataset.py, evaluate.py, evidence_inference.py, inference.py, pneumonia_config.py, pneumonia_model.py, text_inference.py, train.py, utils.py`; `baseline_results/{evaluate.py, predictions.csv, threshold_results.csv, roc_curve.png, confusion_matrix.png}`; `checkpoints/{best_model.pth, last_model.pth}`; `results/finetuned/{predictions.csv, threshold_results.csv, roc_curve.png, confusion_matrix.png}` (NO `training_curves.png` anywhere); `text_branch/{best_model.pt, biobert_pneumonia.ipynb, experiment_summary.json, test_results.json, experiment2/{README.md, knowledge_base.py, evidence_engine.py, demo.py, tests.py}}`; `BioBERT_Text_Dataset_Encoded.csv`; `report/DISSERTATION_DRAFT.md`.

---

## 13. Known Limitations

**Confirmed from observed artifacts:**
- **Image — single dataset, no domain-shift experiment.** The project title is "Domain Shift in Chest X-Ray Analysis", but no domain-shift or external/out-of-distribution test exists in the repo. val AUC 0.9962 (20% holdout) / test AUC 0.9619 (same single dataset).
- **Image — original Kaggle split unused.** `val/` (16 images) is merged and re-split; per-split class counts not persisted.
- **Image — checkpoint provenance gaps.** Freeze mode of the final run unverified; epoch-14 stop reason unknown; best/last checkpoints are NOT byte-identical (contradicts draft).
- **Image — deployed threshold.** 0.5 is the conventional sigmoid default, not the best-F1 operating point (0.95); specificity at 0.5 is only 0.5598.
- **Text Exp 1 — synthetic-structured corpus + memorization risk.** Near-perfect separation (AUC 1.0, all pneumonia probs ≥ 0.9996) on a small filtered synthetic corpus; no real-world clinical text; no symptom-level reasoning. Repo result files themselves attribute performance to the "synthetic structured nature" of the data.
- **Text Exp 1 — provenance not repo-verifiable.** Original Kaggle dataset URL/version and the tabular→sentence conversion code are not in the repo (researcher-documented).
- **Text Exp 2 — bounded coverage.** Deterministic KB phrase coverage only; fixed 3-word negation window; unmapped symptoms unclassified; BioBERT fallback implemented but never exercised; not validated on clinical text. Status: POC pending supervisor confirmation.
- **No multimodal fusion.** The three branches are independent evidence outputs; no combined decision is produced.
- **"Not for clinical use"** — stated in code/UI/docstrings.

**Planned / not implemented (do not present as done):**
- Image last-DenseNet-block + classifier fine-tuning.
- Module 3 — structured clinical variable intake + validated risk scoring (CURB-65/PSI).
- Knowledge-grounded LLM approach for extraction/reasoning.
- Formal citations for the two Kaggle datasets; training-curves artifact.

---

## 14. Unresolved / Conflicting Evidence

| # | Item | Evidence | Status |
|---|---|---|---|
| 1 | CSV `label` column is disease index 0–7, NOT binary | Column counts verified (8 classes); binary derived from `disease` | **Resolved** — documented, no conflict |
| 2 | `best_model.pth` vs `last_model.pth` byte-identical (draft claim) | Both epoch 14, val_auc 0.9962, size 31,554,294 B, but MD5 differs | **CONFLICT** — not identical |
| 3 | BioBERT early stopping "triggered at epoch 2" (draft claim) | Notebook artifact: 3 epochs ran, then KeyboardInterrupt; `early_stopped:false` consistent (patience 3, only 2 non-improvements logged) | **CONFLICT** — draft claim unsupported by artifact |
| 4 | Which script produced `baseline_results/` artifacts | Saved sweep 0.10–0.90 matches `baseline_results/evaluate.py`; that script writes to `chest_xray/` per code, but artifacts live in `baseline_results/`; its preprocessing uses mean-axis gray vs first-channel everywhere else | **UNRESOLVED** (provenance) |
| 5 | Freeze/unfreeze mode of the final image training run | Checkpoint records no flag; config default `freeze_backbone=True` | **UNVERIFIED** |
| 6 | Why image training stopped at epoch 14 of 15 | Not recorded | **UNVERIFIED** |
| 7 | `training_curves.png` referenced in draft | Absent from `results/` (verified listing) | **UNVERIFIED / MISSING** |
| 8 | Total image parameters 7,210,241 (draft) vs state-dict sum 7,294,010 | Head 262,657 verified; 7,294,010 includes BN buffers; 7,210,241 = parameter-only estimate | **Resolved** — reconciled; only 7,294,010 / 262,657 directly verified |
| 9 | `transformers` 4.40.0 (notebook) vs 4.51.3 (local) | Checkpoint verified loading and reproducing predictions under 4.51.3 | **Resolved** (repro output matches); version drift possible |
| 10 | "fever, cough → P(pneumonia) 0.9991" (old note) | Verified live: bare "fever, cough" → **Non-pneumonia 0.4723**; the 0.9992 figure corresponds to "…fever, cough, shortness of breath, and chills." | **CONFLICT** — old claim corrected |
| 11 | Paediatric image dataset; Kaggle text-dataset provenance | Stated by researcher; Kaggle docs/URLs not in repo; conversion code not in repo | **UNVERIFIED FROM REPO** (researcher-documented) |
| 12 | Image `pos_weight` exact value | Computed as ratio, only approx 0.35 documented; not persisted | **UNVERIFIED** (approx only) |
| 13 | Original Kaggle URLs / licences / ethics | Not recorded anywhere in repo | **UNVERIFIED** |

---

## 15. Dissertation Safety Rules

**Never claim (absent repository evidence):**
- Never claim the image or text model generalises to unseen clinical populations, hospitals, or domains — no domain-shift / external test exists.
- Never claim "early stopping triggered at epoch 2" for BioBERT — the notebook shows 3 epochs + KeyboardInterrupt; report it as "the run was interrupted during epoch 3; best checkpoint is epoch 1".
- Never claim `best_model.pth` and `last_model.pth` are byte-identical — they differ (MD5).
- Never claim the image training used a specific freeze/unfreeze mode — unverified (config default was frozen backbone).
- Never claim "fever, cough → P(pneumonia) 0.9991" — the bare phrase gives Non-pneumonia 0.4723; use the verified longer sentence if an example is needed.
- Never present Exp 2 as a validated clinical tool — it is a POC passing its own 16-case suite; the BioBERT fallback was never exercised.
- Never claim a multimodal fusion/combined decision — the app explicitly has none.
- Never present the planned experiments (last-block tuning, Module 3, LLM extraction) as performed.
- Never claim training curves were produced — no `training_curves.png` exists.
- Never claim total image params are 7,210,241 as verified — the verified values are 7,294,010 (state dict incl. buffers) and 262,657 (head).
- Never claim the symptom corpus was real clinical text — repo files describe it as synthetic-structured.

**Safe terminology (supported by repository evidence):**
- "research prototype, not for clinical use" (code/UI).
- "No multimodal probability fusion is implemented." (app behavior).
- Text corpus described with the repo's own phrasing: "synthetic structured nature of symptom sentences generated from disease-specific binary symptom vectors"; alternatively "template-like" (draft) — both traceable.
- For BioBERT run history: "best checkpoint at epoch 1; the Colab session was interrupted during epoch 3; `early_stopped: false` is consistent with the notebook output."
- For image checkpoints: "best and final checkpoints both record epoch 14 / val_auc 0.9962 but are not byte-identical."
- Thresholds: "deployed threshold 0.5 (conventional); best-F1 operating point 0.95 for the fine-tuned image model."
- Grad-CAM: "qualitative visualisation targeting `denseblock4`; no quantitative evaluation performed."

---

*Audit notes: This file was rewritten on 2026-08-16 after a full pass over the repository (code, configs, checkpoints, saved CSVs/JSONs, notebook dump, git history, and live re-runs of Exp-1 inference and the Exp-2 test suite). All numbers were re-verified; conflicts vs earlier drafts are itemised in §14. The dissertation draft (`report/DISSERTATION_DRAFT.md`) predates several of these corrections and must be brought in line with this file before submission.*