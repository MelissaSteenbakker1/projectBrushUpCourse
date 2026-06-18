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

### Results
| Model | Validation AUC | Test AUC |
|---|---|---|
| Logistic Regression (baseline) | 0.6643 | 0.6517 |
| CNN | 0.7422 | 0.7387 |

The best CNN model was taken at epoch 11/25 (Train Loss: 0.1613, Val Loss: 0.1630).

## Interpretation

## Limitations
- Changing class_weight for CNN from balanced to none improved AUC
- Lowering max_iter gave convergence warnings, increasing had no effect
- Adding regularisation 0.1 (Default 1.0) to config imrpoved AUC
- 10k subset

## Future Improvements
