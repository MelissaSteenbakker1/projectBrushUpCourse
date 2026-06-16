import torch.nn as nn

class ChestMNISTModel(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()

        # Two convolutional layers to extract spatial features from the image
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (1, 28, 28) -> (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (16, 28, 28) -> (16, 14, 14)

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # (16, 14, 14) -> (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (32, 14, 14) -> (32, 7, 7)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # (32, 7, 7) -> (1568,)
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            # Output has 14 units, one per chest condition
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
