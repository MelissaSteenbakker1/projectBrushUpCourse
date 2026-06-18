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


### Results baseline model
* Validation AUC: 0.6527
* Test AUC: 0.6389

### Results CNN model
Trained run: run_20260618_104656  
Epoch 14/25 — Train Loss: 0.1597 | Val Loss: 0.1631

* Baseline Test ROC AUC (macro): 0.6389
* CNN      Test ROC AUC (macro):  0.7357

## Interpretation

## Limitations

## Future Improvements
