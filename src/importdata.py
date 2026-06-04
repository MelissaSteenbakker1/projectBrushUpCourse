import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader
from torchvision import transforms

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
        download=True
    )

    return dataset

def get_chestmnist_loaders(batch_size=64):

    """

    Make train, validation and test DataLoaders.

    """

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
