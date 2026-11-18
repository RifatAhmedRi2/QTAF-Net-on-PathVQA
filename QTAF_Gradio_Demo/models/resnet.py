import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import resnet18

class ResNet18ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=None)  # CRITICAL: random initialization; no pretrained weights
        self.backbone.fc = nn.Identity()         # [B, 512]

    def forward(self, images):
        return self.backbone(images)

class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, questions):
        embeddings = self.embedding(questions)
        lengths = (questions != 0).sum(dim=1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return torch.cat([hidden[-2], hidden[-1]], dim=1)

class ResNet18BiLSTMBaseline(nn.Module):
    def __init__(self, vocab_size, num_open_answers=1000):
        super().__init__()
        self.image_encoder = ResNet18ImageEncoder()
        self.question_encoder = QuestionEncoder(vocab_size)
        self.fusion = nn.Sequential(
            nn.Linear(512 + 256, 256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.yes_no_head = nn.Linear(256, 2)
        self.open_head = nn.Linear(256, num_open_answers)

    def forward(self, images, questions):
        image_features = self.image_encoder(images)
        question_features = self.question_encoder(questions)
        fused = self.fusion(torch.cat([image_features, question_features], dim=1))
        return self.yes_no_head(fused), self.open_head(fused)
