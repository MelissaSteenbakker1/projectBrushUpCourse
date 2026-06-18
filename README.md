<a id="readme-top"></a>
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Project Brush-Up Course</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#evaluation--interpretation">Evaluation & Interpretation</a></li>
    <li><a href="#file-descriptions">File Descriptions</a></li>
    <li><a href="#authors">Authors</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

### Project Brush-Up Course

This project investigates the use of machine learning and deep learning for medical image classification using the ChestMNIST dataset from MedMNIST.

The goal is to compare a traditional machine learning approach (Logistic Regression) with a Convolutional Neural Network (CNN) implemented in PyTorch. The project follows a complete machine learning workflow, including data loading, preprocessing, training, validation, evaluation, and model comparison.

* Dataset: ChestMNIST
* Task: Multi-label classification of chest X-ray images
* Number of labels: 14
* Image size: 28 × 28 pixels (grayscale)

### Built With

This project was built using the following frameworks and libraries:
* **Python** – main programming language used for the project.
* **PyTorch** – used to build, train and evaluate the CNN model.
* **MedMNIST** – provides the ChestMNIST medical image dataset.
* **scikit-learn** – used for the Logistic Regression baseline and ROC AUC evaluation.
* **NumPy** – used for numerical operations and array handling.
* **Pandas** – used to save dataset split IDs as CSV files.
* **PyYAML** – used to read the CNN configuration file.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

Follow these steps to run the project locally.

### Prerequisites

Make sure you have Python installed. This project was developed using Python 3.9 or higher.

Check your Python version:

```
python --version
```

You also need pip, which is used to install Python packages:

```
pip --version
```

### Installation

Open the project folder in your terminal.

1. Make sure you are inside the project folder:
```
cd projectBrushUpCourse
```
2. Upgrade pip.
```
python -m pip install --upgrade pip
```
3. Install the required packages.
```
python -m pip install -r requirements.txt
```
The project is now ready to run.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

This section explains how to run the main parts of the project.

### 1. Load the Dataset

Run the following command to download and prepare the ChestMNIST dataset:
```
python src/importdata.py
```
This script loads the ChestMNIST dataset and creates DataLoaders for the training, validation and test sets. It also saves the split IDs in the data/splits/ folder.

### 2. Inspect the Data

```
python src/data_inspect.py
```
Prints split sizes, label distributions, and checks for class imbalance. Saves an image montage to `outputs/images/`. Run this before training to check the data.

### 3. Check the Configuration

Before training the CNN model, check the configuration file:
```
configs/config.yaml
```
This file contains the training settings both the CNN and the baseline model, such as the batch size, learning rate, number of epochs and baseline hyperparameters.

Our configuration:
```yaml
train_subset: 10000
seed: 42
batch_size: 64
learning_rate: 0.001
epochs: 25

baseline:
  max_iter: 1000
  class_weight:
  solver: lbfgs
  C: 0.1
```
### 4. Run the Baseline Model

To run the Logistic Regression baseline, use:
```
python src/baseline.py
```
The baseline model is a traditional machine learning model. It flattens each 28 × 28 image into a vector of 784 pixel values and trains a Logistic Regression classifier for each of the 14 labels using the settings from `configs/config.yaml`. Finally, it saves the val and test ROC AUC scores to `outputs/baseline/baseline_results.json`.

### 5. Train the CNN Model

To train the Convolutional Neural Network, use:
```
python src/train.py
```
The training script uses the settings from ``` configs/config.yaml ```

Each run is saved to `outputs/run_<timestamp>/` containing the best model weights (`model.pt`), settings used (`config.yaml`) and training metrics (`metrics.json`).
### 6. Evaluate the CNN Model

After training, evaluate the saved model on the test set:
```
python src/evaluate.py
```
Loads the most recent trained model from `outputs/` and prints a side-by-side comparison of the baseline and CNN macro ROC AUC on the test set.

## Evaluation & Interpretation
For a detailed description of the experiments, results, and discussion of limitations, see the [project report](src/REPORT.md).

## File Descriptions
* `src/baseline.py` runs the Logistic Regression baseline model.
* `src/evaluate.py` evaluates the trained CNN model on the validation and test set, and compares them to the baseline model.
* `src/importdata.py` loads the ChestMNIST dataset and creates the datasets.
* `src/data_inspect.py` inspects the data: shapes, label distribution and image montage
* `src/model.py` defines the CNN.
* `src/train.py` trains the CNN model and saves the best performing model.
* `configs/config.yaml` stores the training configuration and baseline hyperparameters for the CNN and the baseline model.
* `data/splits/` contains CSV files with the sample IDs for the training, validation and test sets.
* `outputs/` is generated automatically at runtime. Contains the baseline results and CNN run folders (model weights, config, metrics).

* `requirements.txt` lists the Python packages needed to run the project.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Authors
* Kai Rook - kai.rook@student.uva.nl
* Melissa - m.l.steenbakker@student.uva.nl

