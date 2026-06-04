import torch.nn as nn

class ChestMNISTModel(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()

        # Two convolutional layers to extract spatial features from the image
        self.features = nn.Sequential(
            # Input is 1 grayscale channel. Output is 16 feature maps
            # padding=1 keeps the spatial size the same after convolution
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (1, 28, 28) -> (16, 28, 28)
            nn.ReLU(),
            # MaxPool halves the spatial dimensions which reduces computations
            nn.MaxPool2d(2),                              # (16, 28, 28) -> (16, 14, 14)

            # Second conv layer learns higher-level features from the first
            # More filters (32) to capture more complex patterns
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # (16, 14, 14) -> (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (32, 14, 14) -> (32, 7, 7)
        )
        self.classifier = nn.Sequential(
            # Flatten turns the 3D feature map (32, 7, 7) into a 1D vector
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
