import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from model import ChestMNISTModel
from importdata import get_chestmnist_loaders
import glob
import json


def evaluate(batch_size=64):
    """Evaluate the saved best model on the test set using ROC AUC"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Only need the test loader
    _, _, test_loader = get_chestmnist_loaders(batch_size=batch_size)

    model = ChestMNISTModel(num_classes=14).to(device)

    # Load the latest model from train.py
    run_dirs = sorted(glob.glob("outputs/run_*/model.pt"))
    latest = run_dirs[-1]
    print(f"Loading model from: {latest}")
    model.load_state_dict(torch.load(latest, map_location=device))
    model.eval()

    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.float().to(device)

            logits = model(images)

            probs = torch.sigmoid(logits)

            all_outputs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    test_auc = roc_auc_score(all_labels, all_outputs, average="macro")
    
    with open("outputs/baseline/baseline_results.json") as f:
        baseline = json.load(f)
    print(f"Baseline Test ROC AUC (macro): {baseline['test_auc']:.4f}")
    print(f"CNN      Test ROC AUC (macro):  {test_auc:.4f}")


if __name__ == "__main__":
    evaluate()
