import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class SpatialImageEncoder(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU())
        self.project = nn.Conv2d(256,dim,1)
    def forward(self,x):
        return self.project(self.cnn(x)).flatten(2).transpose(1,2)

class QuestionEncoder(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,128,padding_idx=0)
        self.lstm=nn.LSTM(128,128,batch_first=True,bidirectional=True)
    def forward(self,ids):
        lengths=(ids!=0).sum(1).clamp(min=1)
        packed=nn.utils.rnn.pack_padded_sequence(self.embedding(ids),lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,(h,_)=self.lstm(packed)
        return torch.cat([h[-2],h[-1]],dim=1)

class AblatedQTAFNet(nn.Module):
    def __init__(self,vocab_size,num_open_answers=1000,variant='no_attention'):
        super().__init__()
        assert variant in ('no_attention','no_type_fusion','single_head')
        self.variant=variant
        self.image_encoder=SpatialImageEncoder()
        self.question_encoder=QuestionEncoder(vocab_size)
        if variant!='no_attention':
            self.att_visual=nn.Linear(256,256)
            self.att_question=nn.Linear(256,256)
            self.att_score=nn.Linear(256,1)
        self.type_head=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,2))
        if variant=='no_type_fusion':
            self.shared_fusion=nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.3))
        else:
            self.closed_fusion=nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.3))
            self.open_fusion=nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.3))
        if variant=='single_head':
            self.answer_head=nn.Linear(256,2+num_open_answers)
        else:
            self.yes_no_head=nn.Linear(256,2)
            self.open_head=nn.Linear(256,num_open_answers)

    def forward(self,images,questions):
        spatial=self.image_encoder(images)
        question=self.question_encoder(questions)
        if self.variant=='no_attention':
            visual=spatial.mean(1)
        else:
            energy=torch.tanh(self.att_visual(spatial)+self.att_question(question).unsqueeze(1))
            attention=torch.softmax(self.att_score(energy).squeeze(-1),dim=1)
            visual=(spatial*attention.unsqueeze(-1)).sum(1)
        type_logits=self.type_head(question)
        joint=torch.cat([visual,question],dim=1)
        if self.variant=='no_type_fusion':
            fused=self.shared_fusion(joint)
        else:
            type_probs=torch.softmax(type_logits,dim=1)
            closed_feature=self.closed_fusion(joint)
            open_feature=self.open_fusion(joint)
            fused=type_probs[:,0:1]*closed_feature+type_probs[:,1:2]*open_feature
        if self.variant=='single_head':
            return self.answer_head(fused),type_logits
        return self.yes_no_head(fused),self.open_head(fused),type_logits
