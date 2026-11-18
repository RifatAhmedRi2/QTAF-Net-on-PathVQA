import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_dim,
            batch_first=True, bidirectional=True
        )

    def forward(self, questions):
        embedded = self.embedding(questions)
        lengths = (questions != 0).sum(dim=1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return torch.cat([hidden[-2], hidden[-1]], dim=1)

class QuestionOnlyBiLSTM(nn.Module):
    def __init__(self, vocab_size, num_open_answers=1000):
        super().__init__()
        self.question_encoder = QuestionEncoder(vocab_size)
        self.fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.yes_no_head = nn.Linear(256, 2)
        self.open_head = nn.Linear(256, num_open_answers)

    def forward(self, questions):
        features = self.question_encoder(questions)
        shared = self.fusion(features)
        return self.yes_no_head(shared), self.open_head(shared)
