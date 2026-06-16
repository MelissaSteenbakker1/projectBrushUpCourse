<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
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
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

### Project Brush Up Course

This project investigates the use of machine learning and deep learning for medical image classification using the ChestMNIST dataset from MedMNIST.

The goal is to compare a traditional machine learning approach (Logistic Regression) with a Convolutional Neural Network (CNN) implemented in PyTorch. The project follows a complete machine learning workflow, including data loading, preprocessing, training, validation, evaluation, and model comparison.

* Dataset: ChestMNIST
* Task: Multi-label classification of chest X-ray images
* Number of labels: 14
* Image size: 28 × 28 pixels (grayscale)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

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

### 2. Run the Baseline Model

To run the Logistic Regression baseline, use:
```
python src/baseline.py
```
The baseline model is a traditional machine learning model. It flattens each 28 × 28 image into a vector of 784 pixel values and trains a Logistic Regression classifier for each of the 14 labels.

This gives a simple model to compare the CNN against.

### 3. Check the Configuration

Before training the CNN model, check the configuration file:
```
configs/cnn.yaml
```
This file contains the training settings for the CNN model, such as the batch size, learning rate and number of epochs.

Our configuration:
seed: 42 
batch_size: 64 
learning_rate: 0.001 
epochs: 25 
train_subset: 10000

### 4. Train the CNN Model

To train the Convolutional Neural Network, use:
```
python src/train.py
```
The training script uses the settings from ``` configs/cnn.yaml ```

The model with the lowest validation loss is saved as the best model.

### 5. Evaluate the CNN Model

After training, evaluate the saved model on the test set:
```
python src/evaluate.py
```
The evaluation script loads the trained CNN model and calculates the macro ROC AUC score on the test set.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## File Descriptions
* ```baseline.py``` runs the Logistic Regression baseline model.
* ```evaluate.py``` evaluates the trained CNN model on the test set.
* ```importdata.py``` loads the ChestMNIST dataset and creates the datasets.
* ```model.py``` defines the CNN.
* ```pytorch.py``` checks the PyTorch installation and whether a GPU is available.
* ```train.py``` trains the CNN model and saves the best performing model.
* ```configs/cnn.yaml``` stores the training configuration, such as the batch size, learning rate and number of epochs.
* `data/splits/` contains CSV files with the sample IDs for the training, validation and test sets.
* ```requirements.txt``` lists the Python packages needed to run the project.
* ```best_model.pth``` stores the saved model weights.

## Authors
* Kai
* Melissa


[JQuery-url]: https://jquery.com 

