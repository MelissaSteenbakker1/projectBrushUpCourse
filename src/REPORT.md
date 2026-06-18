# Report

## Evaluation
### Datasets
* Train batches: 1227
* Validation batches: 176
* Test batches: 351
* Image batch shape: torch.Size([64, 1, 28, 28])
* Label batch shape: torch.Size([64, 14])

From data_inspect.py:
#### Trainset
* 45.9% have at least one positive label
* 54.1% have all-zero labels

#### Validationset
* 45.8% positive
* 54.2% all-zero

#### Testset
* 46.8% positive
* 53.2% all-zero

This shows that the distribution across the splits is similar.

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
