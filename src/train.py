import torch
import torch.nn as nn
from model import ChestMNISTModel
from importdata import get_chestmnist_loaders

def train(num_epochs=20, batch_size=64, learning_rate=0.001):

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_chestmnist_loaders(batch_size=batch_size)

    model = ChestMNISTModel(num_classes=14).to(device)

    # BCEWithLogitsLoss is used because this is a multi-label problem, each of the 14 conditions
    # is an independent binary classification
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):

        # Training
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()       # clear gradients from previous step
            outputs = model(images)     # forward pass
            loss = criterion(outputs, labels)
            loss.backward()             # compute gradients
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

        print(f"Epoch {epoch+1}/{num_epochs} — Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Checkpointing
        # Save the model whenever validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  -> Saved best model (val loss: {best_val_loss:.4f})")


if __name__ == "__main__":
    train()