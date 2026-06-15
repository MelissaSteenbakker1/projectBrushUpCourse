# projectBrushUpCourse
# Medical AI Project – AI Programming Brush-up

## Project Description

This project investigates the use of machine learning and deep learning for medical image classification using the ChestMNIST dataset from MedMNIST.

The goal is to compare a traditional machine learning approach (Logistic Regression) with a Convolutional Neural Network (CNN) implemented in PyTorch. The project follows a complete machine learning workflow, including data loading, preprocessing, training, validation, evaluation, and model comparison.

Dataset: ChestMNIST
Task: Multi-label classification of chest X-ray images
Number of labels: 14
Image size: 28 × 28 pixels (grayscale)

---

## Repository Structure

```text
projectBrushUpCourse/
│
├── configs/
│   └── cnn.yaml              # Training configuration (hyperparameters, paths)
│
├── data/
│   ├── raw/                  # Downloaded dataset (excluded from git)
│   └── splits/               # Fixed train/val/test split ID files
│       ├── train_ids.csv
│       ├── val_ids.csv
│       └── test_ids.csv
│
├── outputs/                  # Saved models and results (excluded from git)
│
├── src/
│   ├── baseline.py
│   ├── evaluate.py
│   ├── importdata.py
│   ├── model.py
│   ├── train.py
│   └── viewimg.py
│
├── .gitignore
├── README.md
└── requirements.txt
```
---

## Dataset

The project uses the ChestMNIST dataset from MedMNIST. The splits are fixed and standardized by the library.

Official dataset split:

| Split      | Samples |
| ---------- | ------: |
| Training   |  78,468 |
| Validation |  11,219 |
| Test       |  22,433 |
| Total      | 112,120 |

Split indices are exported to data/splits/ on first run and committed to version control for documentation purposes.

ChestMNIST is a dataset designed for multi-label classification, meaning that a single chest X-ray image can be associated with multiple medical conditions at the same time.

---

## Data Preprocessing

The following preprocessing steps are applied:

1. Images are converted to PyTorch tensors.
2. Pixel values are normalized to the range [0,1].
3. Images are loaded in batches using PyTorch DataLoaders.
4. For the baseline model, images are flattened from 28×28 pixels into a vector of 784 features.

---

## File Descriptions

### importdata.py

Responsible for loading the ChestMNIST dataset and creating PyTorch DataLoaders.

Functions:

* `load_chestmnist_dataset(split)`
* `save_split_ids()`
* `get_chestmnist_loaders(batch_size)`

The dataset is automatically downloaded if it is not already available.

---

### baseline.py

Implements the classical machine learning baseline using:
* Logistic Regression
* MultiOutputClassifier
* ROC-AUC evaluation

The images are flattened from:
28 × 28 → 784 features

Pixel values are normalized to the range:
[0, 255] → [0, 1]

For faster experimentation, the baseline currently trains on the first 10,000 training samples.

---

### model.py

Defines the CNN architecture.

```text
Architecture:
Input (1×28×28)
↓
Conv2D (16 filters)
↓
ReLU
↓
MaxPool
↓
Conv2D (32 filters)
↓
ReLU
↓
MaxPool
↓
Flatten
↓
Linear (1568 → 128)
↓
ReLU
↓
Linear (128 → 14)
```

The final layer contains 14 outputs, one for each ChestMNIST label.

---

### train.py

Trains the CNN using settings from `configs/cnn.yaml`.

Configuration:
* Optimizer: Adam
* Loss Function: BCEWithLogitsLoss
* Output: Best model saved to `outputs/`

---

### evaluate.py

Loads the saved model and evaluates it on the test set.

Evaluation metric:
* Macro ROC-AUC

Predictions are converted from logits to probabilities using:
torch.sigmoid()

---

### viewimg.py

Loads and visualizes a ChestMNIST image.

Generated images are stored in:
src/images/

---

### pytorch.py

Utility script used to verify:
* Installed PyTorch version
* GPU availability
* Active device (CPU or CUDA)

---

## Running the Project

### Install dependencies
```bash
pip install -r requirements.txt
```

### Inspect the dataset
```bash
python src/importdata.py
```

### Run the Logistic Regression baseline
```bash
python src/baseline.py
```

### Train the CNN
```bash
python src/train.py --config configs/cnn.yaml
```

### Evaluate the CNN
```bash
python src/evaluate.py
```
---

## Evaluation

The project compares:
1. Logistic Regression baseline
2. Convolutional Neural Network

Performance is measured using:
* Validation ROC-AUC
* Test ROC-AUC
* Training time

This comparison allows us to determine whether a CNN provides improved predictive performance over a simpler machine learning approach.

---

## Authors
Kai
Melissa
Medical AI Project
Course project using MedMNIST, scikit-learn, and PyTorch.

