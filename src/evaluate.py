import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from model import ChestMNISTModel
from importdata import get_chestmnist_loaders


def evaluate(batch_size=64):
    """Evaluate the saved best model on the test set using ROC AUC"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Only need the test loader
    _, _, test_loader = get_chestmnist_loaders(batch_size=batch_size)

    model = ChestMNISTModel(num_classes=14).to(device)

    # Load the best model from train.py
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    all_outputs = []  # model probabilities
    all_labels = []   # ground-truth labels

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.float().to(device)

            logits = model(images)

            # Convert logits → probabilities in [0, 1]
            probs = torch.sigmoid(logits)

            all_outputs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Shape: (num_samples, 14)
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    test_auc = roc_auc_score(all_labels, all_outputs, average="macro")
    print(f"Test ROC AUC (macro): {test_auc:.4f}")


if __name__ == "__main__":
    evaluate()