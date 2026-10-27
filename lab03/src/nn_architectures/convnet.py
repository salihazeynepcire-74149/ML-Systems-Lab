import torch
import torch.nn as nn

class SalihaCNN(nn.Module):
    """
    2-Stage Deep Convolutional Network
    Stage 1: Conv2d(1, 24, k=3, p=1) -> ReLU -> MaxPool2d(2, 2)  [14x14]
    Stage 2: Conv2d(24, 48, k=3, p=1) -> ReLU -> MaxPool2d(2, 2) [7x7]
    Classification: Flatten(48*7*7) -> Linear(2352, 128) -> ReLU -> Linear(128, 10)
    """
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv_features = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(48 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.conv_features(x))
