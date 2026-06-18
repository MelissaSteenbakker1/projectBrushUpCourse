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


### Results CNN model

## Interpretation

## Limitations

## Future Improvements
