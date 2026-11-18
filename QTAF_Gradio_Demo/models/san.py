import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class SpatialImageEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.project = nn.Conv2d(256, feature_dim, kernel_size=1)

    def forward(self, images):
        features = self.project(self.cnn(images))   # [B, 256, 28, 28]
        return features.flatten(2).transpose(1, 2)

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

class AttentionHop(nn.Module):
    def __init__(self, feature_dim=256, attention_dim=256):
        super().__init__()
        self.visual = nn.Linear(feature_dim, attention_dim)
        self.query = nn.Linear(feature_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1)
    def forward(self, spatial_features, query):
        hidden = torch.tanh(self.visual(spatial_features) + self.query(query).unsqueeze(1))
        weights = torch.softmax(self.score(hidden).squeeze(-1), dim=1)
        attended = torch.sum(spatial_features * weights.unsqueeze(-1), dim=1)
        return query + attended, weights

class StackedAttentionNetwork(nn.Module):
    def __init__(self, vocab_size, num_open_answers=1000, num_hops=2):
        super().__init__()
        self.image_encoder = SpatialImageEncoder()
        self.question_encoder = QuestionEncoder(vocab_size)
        self.hops = nn.ModuleList([AttentionHop() for _ in range(num_hops)])
        self.classifier = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3))
        self.yes_no_head = nn.Linear(256, 2)
        self.open_head = nn.Linear(256, num_open_answers)

    def forward(self, images, questions, return_attention=False):
        visual = self.image_encoder(images)
        query = self.question_encoder(questions)
        maps = []
        for hop in self.hops:
            query, weights = hop(visual, query)
            if return_attention: maps.append(weights)
        features = self.classifier(query)
        outputs = (self.yes_no_head(features), self.open_head(features))
        return (*outputs, maps) if return_attention else outputs
