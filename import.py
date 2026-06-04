import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader
from torchvision import transforms

data_flag = 'chestmnist'

info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

transform = transforms.ToTensor()

train_dataset = DataClass(
    split='train',
    transform=transform,
    download=True
)

val_dataset = DataClass(
    split='val',
    transform=transform,
    download=True
)

test_dataset = DataClass(
    split='test',
    transform=transform,
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
