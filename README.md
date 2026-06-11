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

projectBrushUpCourse/
│
├── src/
│   ├── images/
│   │   └── generated images and visualisations
│   │
│   ├── baseline.py
│   ├── evaluate.py
│   ├── importdata.py
│   ├── model.py
│   ├── pytorch.py
│   ├── train.py
│   ├── viewing.py
│   └── best_model.pth
│
├── README.md
└── requirements.txt

---

## Dataset

The project uses the ChestMNIST dataset from MedMNIST.

Official dataset split:

| Split      | Samples |
| ---------- | ------: |
| Training   |  78,468 |
| Validation |  11,219 |
| Test       |  22,433 |
| Total      | 112,120 |

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

The final layer contains 14 outputs, one for each ChestMNIST label.

---

### train.py

Trains the CNN model.

Configuration:
* Optimizer: Adam
* Loss Function: BCEWithLogitsLoss
* Epochs: 20
* Batch Size: 64
* Learning Rate: 0.001

The best-performing model on the validation set is automatically saved as:
best_model.pth

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
pip install -r requirements.txt

### Verify PyTorch installation
python src/pytorch.py

### Inspect the dataset
python src/importdata.py

### Visualize an image
python src/viewimg.py

### Run the Logistic Regression baseline
python src/baseline.py

### Train the CNN
python src/train.py

### Evaluate the CNN
python src/evaluate.py

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

