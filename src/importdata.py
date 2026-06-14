import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import pandas as pd

# Data will be downloaded to and loaded from this folder. This folder is excluded from version control using .gitignore.
DATA_ROOT = "data/raw"
SPLITS_ROOT = "data/splits"

def load_chestmnist_dataset(split):
    """
    Load one ChestMNIST dataset split.
    split = 'train', 'val' or 'test'
    """

    data_flag = "chestmnist"
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])
    transform = transforms.ToTensor()

    dataset = DataClass(
        split=split,
        transform=transform,
        download=True,
        root=DATA_ROOT
    )

    return dataset

def save_split_ids():
    """
    Save the indices of each split to CSV files in data/splits/.
    This documents which samples belong to each split for reproducibility.
    Only runs if the CSV files do not already exist.
    """

    os.makedirs(SPLITS_ROOT, exist_ok=True)

    for split in ["train", "val", "test"]:
        out_path = f"{SPLITS_ROOT}/{split}_ids.csv"

        if os.path.exists(out_path):
            print(f"Split file already exists, skipping: {out_path}")
            continue

        dataset = load_chestmnist_dataset(split)
        indices = list(range(len(dataset)))

        df = pd.DataFrame({"id": indices})
        df.to_csv(out_path, index=False)
        print(f"Saved {len(indices)} IDs to {out_path}")

def get_chestmnist_loaders(batch_size=64):
    """
    Make train, validation and test DataLoaders.
    Also saves split IDs to data/splits/ if not already present.
    """

    save_split_ids()

    train_dataset = load_chestmnist_dataset("train")
    val_dataset = load_chestmnist_dataset("val")
    test_dataset = load_chestmnist_dataset("test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":

    train_loader, val_loader, test_loader = get_chestmnist_loaders()

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    images, labels = next(iter(train_loader))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
