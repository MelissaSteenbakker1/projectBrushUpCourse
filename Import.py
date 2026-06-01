import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader

data_flag = 'pathmnist'

info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

# Dataset laden
train_dataset = DataClass(split='train', download=True)
val_dataset = DataClass(split='val', download=True)
test_dataset = DataClass(split='test', download=True)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
