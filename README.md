# 🧠 PROSPECT: Prediction and Risk Observation System for Chronic Pancreatitis using CNN

An AI-powered medical imaging project using deep learning to classify pneumonia from chest X-rays — serving as a surrogate for chronic pancreatitis detection using endoscopic ultrasound images. Built with **ResNet50**, **SMOTE**, and **advanced augmentation techniques**.

---

## 📌 Project Summary

- **Goal**: Automatically classify medical images as normal or pneumonia (simulating chronic pancreatitis detection)  
- **Dataset**: PneumoniaMNIST (subset of MedMNIST v2)  
- **Model**: Transfer learning using pre-trained **ResNet50**  
- **Techniques**:
  - Enhanced image augmentation  
  - SMOTE for class imbalance  
  - Custom metrics including F1 Score  
  - Grad-CAM for explainability  

---

## 📂 Folder Structure

```
PROSPECT/
│
├── data/                # Contains .npy files (train, val, test images/labels)
├── models/              # Saved model checkpoints
├── output/              # Confusion matrix, ROC curve, logs
├── utils/               # Preprocessing scripts
│   └── preprocessing.py
├── config.py            # Hyperparameters and directory setup
├── model.py             # ResNet50-based model architecture
├── train.py             # Training script with augmentation + SMOTE
├── predict.py           # Evaluation on test data
├── gradcam.py           # Grad-CAM visualization for model interpretability
└── requirements.txt     # Required Python packages
```

---

## ⚙️ Hyperparameter Choices

| Hyperparameter      | Value        | Justification                                                                 |
|---------------------|--------------|--------------------------------------------------------------------------------|
| `learning_rate`     | `1e-4`        | Chosen for stable and smooth convergence with the Adam optimizer              |
| `batch_size`        | `32`          | Balances memory usage and generalization; common for ResNet-based models      |
| `epochs`            | `10`          | Sufficient due to EarlyStopping based on validation AUC                        |
| `early_stop_patience` | `5`        | Prevents overfitting by stopping if validation AUC plateaus for 5 epochs       |
| `img_size`          | `(128, 128)`  | Required input shape for ResNet50 while retaining sufficient image detail      |
| `dropout_rate`      | `0.5`         | Regularization to reduce overfitting in dense layers                          |
| `optimizer`         | `Adam`        | Adaptive learning rate and widely used for deep learning tasks                 |
| `loss_function`     | `Binary Crossentropy` | Suitable for binary classification tasks                              |

These hyperparameters were selected based on best practices and empirical testing to ensure stable training, good generalization, and optimal convergence speed.

---

## 📊 Dataset

- **Source**: [MedMNIST v2 – PneumoniaMNIST](https://medmnist.com/)
- **Size**:
  - Train: 3,883 images
  - Validation: 524 images
  - Test: 624 images
- **Format**: 28×28 grayscale `.npy` images with binary labels (0 = Normal, 1 = Pneumonia)

---

## 🏗️ Model Architecture

- **Base**: ResNet50 (ImageNet pretrained, frozen)
- **Head**:
  - GlobalAveragePooling2D  
  - BatchNormalization  
  - Dense(128, ReLU) → Dropout(0.5)  
  - BatchNormalization → Dropout(0.5)  
  - Dense(1, Sigmoid)  

---

## 🧪 Training & Optimization

- **Augmentation**:
  - Horizontal flip, brightness, contrast, zoom, rotation  
- **Imbalance Handling**:
  - Option 1: `SMOTE` on training set  
  - Option 2: `class_weight` in training  
- **Loss**: Binary Cross-Entropy  
- **Optimizer**: Adam (`lr = 1e-4`)  
- **Metrics**: Accuracy, Precision, Recall, AUC, F1 Score  
- **Callbacks**:
  - EarlyStopping (patience = 5 on val AUC)  
  - ReduceLROnPlateau  
  - ModelCheckpoint → `best_model_smote.keras`

---

## 🧾 Evaluation Results (Sample)

> *(Replace with your actual test set results)*

- ✅ **Accuracy**: 87.0%  
- ✅ **AUC**: 0.975  
- ✅ **F1 Score (Pneumonia)**: 0.90  
- ✅ **Precision / Recall**: Balanced  
- ✅ **Confusion Matrix & ROC Curve** saved in `/output` folder

---


## 🚀 How to Run

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train.py
```

### 3. Evaluate on Test Set
```bash
python predict.py
```

---

## 📌 Key Highlights

- 🔄 **Flexible pipeline**: Supports training with or without SMOTE  
- ⚖️ **Handles imbalance**: SMOTE + class weighting  
- 📈 **Performance-driven**: EarlyStopping, ReduceLROnPlateau, AUC-based saving  
- 🔎 **Explainable AI**: Grad-CAM for model interpretability  
- 🧼 **Clean, modular code**: Suitable for research & real-world extensions  

---

## 📚 References

- [MedMNIST Dataset](https://medmnist.com/)  
- [ResNet50 Paper](https://arxiv.org/abs/1512.03385)  
- [SMOTE - Chawla et al. (2002)](https://doi.org/10.1613/jair.953)

---

## 🙌 Acknowledgements

- Built using: TensorFlow, Scikit-learn, imbalanced-learn, OpenCV  
- Developed by: *[Kesavan C]*  