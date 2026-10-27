import torch
import torch.nn as nn

class SalihaMLP(nn.Module):
    """
    Feed-Forward Perceptron with ELU Non-linearities
    Structure: 784 -> Linear(200) -> ELU -> Linear(100) -> ELU -> Linear(10)
    """
    def __init__(self, in_features=784, h1=200, h2=100, num_classes=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, h1),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(h1, h2),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Linear(h2, num_classes)
        )

    def forward(self, x):
        return self.layers(x)
