import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score
from importdata import load_chestmnist_dataset
import yaml
import json

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

def dataset_to_flat_arrays(dataset):
    """
    Convert ChestMNIST images to flat sklearn input.
    28x28 pixels becomes 784 features.
    """

    X = dataset.imgs.reshape(len(dataset.imgs), -1) / 255.0
    y = dataset.labels

    return X, y


def predict_probabilities(model, X):
    """
    Get the probability that each label is present.
    """

    probabilities = []

    for estimator in model.estimators_:
        proba = estimator.predict_proba(X)
        probabilities.append(proba[:, 1])

    return np.array(probabilities).T


def main():
    """
    Baseline: multi-label classification using logistic regression 
    (one binary classifier per label via MultiOutputClassifier).
    """

    train_dataset = load_chestmnist_dataset("train")
    val_dataset = load_chestmnist_dataset("val")
    test_dataset = load_chestmnist_dataset("test")

    X_train, y_train = dataset_to_flat_arrays(train_dataset)
    X_val, y_val = dataset_to_flat_arrays(val_dataset)
    X_test, y_test = dataset_to_flat_arrays(test_dataset)

    X_train = X_train[:config["train_subset"]]
    y_train = y_train[:config["train_subset"]]

    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    model = MultiOutputClassifier(
        LogisticRegression(
            max_iter=config["baseline"]["max_iter"],
            class_weight=config["baseline"]["class_weight"],
            solver=config["baseline"]["solver"],
            C=config["baseline"]["C"]
        )
    )

    print("Training Logistic Regression baseline...")

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    print(f"Training took {end - start:.1f} seconds")

    val_probs = predict_probabilities(model, X_val)
    test_probs = predict_probabilities(model, X_test)

    val_auc = roc_auc_score(y_val, val_probs, average="macro")
    test_auc = roc_auc_score(y_test, test_probs, average="macro")

    print("Baseline: Logistic Regression (multi-label classifier)")
    print(f"Validation AUC: {val_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    # Save the baseline model's results for later comparison
    results = {
        "config": {
        "max_iter": config["baseline"]["max_iter"],
        "solver": config["baseline"]["solver"],
        "C": config["baseline"]["C"],
        "class_weight": config["baseline"]["class_weight"],
        "train_subset": config["train_subset"]
    },
    "val_auc": round(val_auc, 4),
    "test_auc": round(test_auc, 4)
}
    with open("outputs/baseline/baseline_results.json", "w") as f:
        json.dump(results, f)
    print("Saved baseline results to outputs/baseline/baseline_results.json")

    


if __name__ == "__main__":

    main()
