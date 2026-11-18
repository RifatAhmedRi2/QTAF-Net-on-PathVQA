import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class ImageEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, feature_dim)

    def forward(self, images):
        features = self.cnn(images).flatten(1)
        return self.fc(features)

class ImageOnlyCNN(nn.Module):
    def __init__(self, num_open_answers=1000):
        super().__init__()
        self.image_encoder = ImageEncoder(feature_dim=256)
        self.fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.yes_no_head = nn.Linear(256, 2)
        self.open_head = nn.Linear(256, num_open_answers)

    def forward(self, images):
        features = self.image_encoder(images)
        shared = self.fusion(features)
        return self.yes_no_head(shared), self.open_head(shared)
