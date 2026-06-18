# Report

This report summarises the experimental setup, results and interpretation for the MedMNIST project. It describes how the data is preprocessed, how the baseline and CNN models are configured and trained, and how their performance is evaluated and compared. It also discusses limitations of the current approach and possible future directions for improvement.


## Evaluation
### Dataset
| Split | Samples (n) | Batches (n) |
|---|---|---|
| Train | 78,468 (subset: 10,000) | 1227 |
| Validation | 11,219 | 176 |
| Test | 22,433 | 351 |
* Both models are trained on a subset of 10,000 samples (out of 78,468) to keep the training time manageable during experimentation. This  limitation is discussed further in the limitations section.
* Image batch shape: `(64, 1, 28, 28)` grayscale, 28x28 pixels.
* Label batch shape: `(64, 14)` 14 multi-label chest conditions.

#### Label distribution across splits
Using `data_inspect.py`:

| Split | >0 positive labels | All-zero labels |
|---|---|---|
| Train | 45.9% | 54.1% |
| Validation | 45.8% | 54.2% |
| Test | 46.8% | 53.2% |

This shows that the distribution across the splits is similar. However, within the positive samples, some conditions are rarer than others, which may introduce some class imbalance.

### Experiment Settings

#### Baseline (Logistic Regression)
- Train subset: 10,000 samples
- Solver: lbfgs
- Max iterations: 100
- Class weight: None
- Regularisation: 0.1
- One classifier per label (14 total)

#### CNN
- Train subset: 10,000 samples
- Epochs: 25
- Batch size: 64
- Learning rate: 0.001
- Optimizer: Adaptive Moment Estimation (Adam)
- Loss: BCEWithLogitsLoss

The architecture of the CNN consists of two convolutional layers. Following this is the classifier. The first convolutional layer uses 16 filters with a kernel size of 3×3. This is to train itself on the basic patterns. In the second layer consist of 32 layers which allows the network to train itself on more complex patterns. After each convolutional layer there is a ReLu function this reduces the vanishing gradient problem and makes it possible to train faster and more stable. Then the output is converted into a one-dimensional vector of 1568 values (32 × 7 × 7). The flattened vector is passed to a connected layer with 128 neurons. This layer acts as a classifier by combining the features extracted by the convolutional layers. The final output layer then produces one prediction (logit) for each of the 14 chest condition labels.

##### Optimizer
The CNN was trained using the Adam (Adaptive Moment Estimation) optimizer. This optimizer updated the weights of the neural network after each batch based on the computed gradients. Therefor, Adam adapts the learning rate inficidually for each model parameter. This results in faster and more stable training and thereby using little hyperparameter.

##### Loss function
The model uses BCEWithLogitsLoss because ChestMNIST is a multi-label classification dataset. Which means that an image can belong to multiple chest conditions. BCEWithLogitsLoss treats each of the 14 output labels as an independent binary classification problem, making it appropriate for multi-label classification.

### Results
| Model | Validation AUC | Test AUC |
|---|---|---|
| Logistic Regression (baseline) | 0.6643 | 0.6517 |
| CNN | 0.7422 | 0.7387 |

The best CNN model was taken at epoch 11/25 (Train Loss: 0.1613, Val Loss: 0.1630).

## Interpretation
The Logistic Regression baseline achieved at it's highest a test ROC AUC of 0.6517, which confirms that the linear model can extract some discriminative signal from raw pixel values. 

The CNN achieved at highest a test ROC AUC of 0.7387, an improvement of ~0.09 compared to the baseline model. This suggests that the convolutional layers can learn the spatial features from the images that a linear model cannot capture. After multiple experiments, the model usually stopped improving between epoch 11 and 15, which suggests the model converged early and that training for the full 25 epochs was not necessary.

## Limitations
Both models were trained on a subset of 10,000 samples out of the full 78,468 available, mainly to keep the training time manageable during experimentation. Most likely, this limits the CNN's ability to learn better representations, particularly for rare conditions that appear less frequent. Training on the full dataset would be the most straightforward way to improve performance.

### Baseline model

The baseline was trained on flattened pixel values, which is a poor representation for image data. There was no image-specific preprocessing such as normalisation or augmentation applied beyond what MedMNIST does by default.

 * During tuning, `class_weight="balanced"` slightly reduced the macro ROC AUC compared to no class weighting. This is probably because balanced weighting up-weights rare classes, which can distort the decision boundary for the majority of labels that are already represented pretty well.

* Lowering the regularisation (`C=0.1` instead of the default `C=1.0`) improved the model performance, which suggest that the default was overfitting to the training subset. With 784 input features but only 10,000 samples, the model will start fitting noise in the pixel values instead of a real signal. Lowering C to 0.1 forces the model to be more conservative and generalise better.

* Lowering `max_iter` to 100 caused convergence warnings, increasing it beyond 1000 had no effect on the AUC. This indicates that the solver finds a stable solution within 1000 iterations.

### CNN model

* No data augmentation was applied. The model sees each image only in its original orientation, which reduces its ability to generalise to natural variation in X-ray images (for example slight rotations or flips). Augmentation increases the diversity of the training distribution without requiring additional data.

* The model usually converged between epoch 11-16 out of 25. This suggests that with the current model architecture and dataset size, the model reaches a performance plateau relatively quickly. More training longer does not improve AUC and may risk overfitting. This behaviour would likely change if the full dataset was used, as more data would sustain meaningful updates for more epochs.


## Future Improvements
Training on the full dataset is the most direct improvement available and would likely benefit the CNN significantly. Beyond that, adding data augmentation could also improve generalisation. For the baseline, replacing Logistic Regression with a gradient boosting model such as XGBoost could provide a stronger and more informative point of comparison.
