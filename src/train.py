# Authors: 144112337 and 13801554
# Date: 18-06-2026
# Course: 9A2

import torch
import torch.nn as nn
from model import ChestMNISTModel
from importdata import get_chestmnist_loaders
import yaml
import os
import json
from datetime import datetime

with open("./configs/config.yaml") as f:
    config = yaml.safe_load(f)


def train():

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a unique output folder for this run based on the current time
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/run_{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")

    # Save the config used for this run so it can be reproduced
    with open(f"{output_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)

    train_loader, val_loader, _ = get_chestmnist_loaders(batch_size=config["batch_size"])

    model = ChestMNISTModel(num_classes=14).to(device)

    # BCEWithLogitsLoss is used because this is a multi-label problem, each of the 14 conditions is an independent binary classification
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    best_val_loss = float("inf")

    for epoch in range(config["epochs"]):

        # Training
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()       # clear gradients from previous step
            outputs = model(images)     # forward pass
            loss = criterion(outputs, labels)
            loss.backward()             # compute gradients (backpropagation)
            optimizer.step()            # update weights

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{config['epochs']} — Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Checkpointing
        # Save the model whenever validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{output_dir}/model.pt")
            print(f"  -> Saved best model (val loss: {best_val_loss:.4f})")
            
    # Save final metrics after all epochs are done
    metrics = {"best_val_loss": best_val_loss}
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {output_dir}/metrics.json")

if __name__ == "__main__":
    train()
