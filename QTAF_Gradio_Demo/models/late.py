import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class ImageEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(256, feature_dim)

    def forward(self, images):
        features = self.cnn(images)
        features = features.flatten(1)
        return self.fc(features)

class QuestionEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, questions):
        embedded = self.embedding(questions)

        lengths = (questions != 0).sum(dim=1).clamp(min=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        # Concatenate forward and backward final states
        question_features = torch.cat(
            [hidden[-2], hidden[-1]], dim=1
        )

        return question_features

class LateFusionVQA(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_open_answers=1000
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(
            feature_dim=256
        )

        self.question_encoder = QuestionEncoder(
            vocab_size=vocab_size
        )

        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.yes_no_head = nn.Linear(256, 2)

        self.open_head = nn.Linear(
            256, num_open_answers
        )

    def forward(self, images, questions):
        image_features = self.image_encoder(images)
        question_features = self.question_encoder(questions)

        combined = torch.cat(
            [image_features, question_features],
            dim=1
        )

        fused = self.fusion(combined)

        yes_no_logits = self.yes_no_head(fused)
        open_logits = self.open_head(fused)

        return yes_no_logits, open_logits
