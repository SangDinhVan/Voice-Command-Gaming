import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNNKWS(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B,1,40,T)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Global average pooling over (H,W)
        x = x.mean(dim=(-2, -1))  # (B,64)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)        # logits
