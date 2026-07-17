# Pneumonia Detection using Fine-Tuned TorchXRayVision DenseNet121

A deep learning project for detecting **Pneumonia** from chest X-ray images using a fine-tuned **TorchXRayVision DenseNet121** model.

The project also includes:

- Chest X-ray preprocessing
- Model training
- Model evaluation
- Single-image inference
- Grad-CAM visual explanations
- Streamlit web application

---

# Features

- Binary classification (Normal / Pneumonia)
- Fine-tuned TorchXRayVision DenseNet121
- Grad-CAM visualization
- Training and evaluation pipeline
- Interactive Streamlit interface
- Model checkpoint saving/loading

---

# Project Structure

```
project/
│
├── app.py
├── inference.py
├── train.py
├── evaluate.py
├── pneumonia_model.py
├── dataset.py
├── pneumonia_config.py
├── utils.py
│
├── checkpoints/
├── outputs/
├── results/
└── data/
```

---

# File Description

### pneumonia_config.py

Stores all project settings in one place.

Includes:

- Dataset paths
- Image size
- Batch size
- Learning rate
- Number of epochs
- Device (CPU/GPU)
- Pretrained model name
- Output directories

Changing values here updates the entire project.

---

### dataset.py

Loads the Chest X-ray dataset.

Responsibilities:

- Reads Normal and Pneumonia images
- Applies TorchXRayVision preprocessing
- Resizes images
- Converts images into PyTorch tensors
- Returns image-label pairs to the DataLoader

---

### pneumonia_model.py

Defines the deep learning model.

Steps:

1. Loads pretrained TorchXRayVision DenseNet121.
2. Removes the original multi-label classifier.
3. Keeps the pretrained feature extractor.
4. Adds a new binary classifier head.
5. Supports freezing or unfreezing the backbone.
6. Loads saved checkpoints during evaluation or inference.

---

### train.py

Handles complete model training.

Responsibilities:

- Loads dataset
- Creates train/validation split
- Builds DataLoaders
- Creates the model
- Defines loss function
- Creates optimizer
- Trains for multiple epochs
- Validates after every epoch
- Saves best and latest checkpoints
- Uses learning-rate scheduling
- Performs early stopping
- Saves training curves

---

### evaluate.py

Evaluates model performance on the test dataset.

Can evaluate:

- Pretrained baseline model
- Fine-tuned model

Calculates:

- Accuracy
- Precision
- Recall
- Specificity
- F1 Score
- ROC-AUC

Also generates:

- ROC Curve
- Confusion Matrix
- Prediction CSV
- Threshold analysis

---

### inference.py

Used for predicting a single chest X-ray.

Responsibilities:

- Loads trained model
- Preprocesses uploaded image
- Performs prediction
- Converts output into probability
- Returns Normal or Pneumonia prediction

This file is mainly used by the Streamlit application.

---

### app.py

Provides a simple Streamlit web interface.

Workflow:

1. Upload chest X-ray
2. Image preprocessing
3. Model prediction
4. Display prediction
5. Display confidence score
6. Display Grad-CAM heatmap

---

### utils.py

Contains helper functions used throughout the project.

Examples:

- Random seed initialization
- Metric calculation
- Threshold sweep
- ROC plotting
- Confusion matrix plotting
- Training curve plotting

Keeping these functions here avoids repeating code.

---

# Dataset

Dataset:

**Chest X-ray Images (Pneumonia)**

Classes:

- Normal
- Pneumonia

The training and validation folders are merged and split again using stratified sampling to create balanced train/validation sets.

---

# Model Architecture

- TorchXRayVision DenseNet121 backbone
- Pretrained weights:
  `densenet121-res224-all`
- Custom binary classification head
- Binary Cross Entropy Loss
- Adam Optimizer

---

# Training

Train the classifier head:

```bash
python train.py
```

Train the full network:

```bash
python train.py --unfreeze
```

---

# Evaluation

Evaluate pretrained model:

```bash
python evaluate.py --mode baseline
```

Evaluate fine-tuned model:

```bash
python evaluate.py --mode finetuned
```

---

# Run the Application

```bash
streamlit run app.py
```

Upload a chest X-ray image to:

- Predict Normal/Pneumonia
- View confidence score
- View Grad-CAM explanation

---

# Technologies Used

- Python
- PyTorch
- TorchXRayVision
- Streamlit
- NumPy
- OpenCV
- Matplotlib
- scikit-learn

---

# Future Improvements

- Fine-tune the complete DenseNet backbone
- Extend to multiple thoracic diseases
- Add multimodal clinical data
- Improve explainability techniques

---

# Author

Muhammad Ali
