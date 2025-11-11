import torch.nn as nn
from fen_model import fen_model


class Local_switcher_model(nn.Module):
    def __init__(self,fen_model_hidden1,fen_model_hidden2,hidden1,hidden2,action_num,dropout=0.1):
        super().__init__()
        self.fen_model=fen_model(fen_model_hidden1,fen_model_hidden2)
        self.fc1=nn.Linear(fen_model_hidden2,hidden1)
        self.relu=nn.ReLU()
        self.bn1=nn.BatchNorm1d(hidden1)
        self.fc2=nn.Linear(hidden1,hidden2)
        self.do=nn.Dropout(dropout)
        self.bn2=nn.BatchNorm1d(hidden2)
        self.outlayer=nn.Linear(hidden2,action_num)
    
    def forward(self,image):
        feature_tensor=self.fen_model(image)
        out=self.fc1(feature_tensor)
        out=self.relu(out)
        out=self.bn1(out)
        out=self.do(out)
        out=self.fc2(out)
        out=self.relu(out)
        out=self.bn2(out)
        out=self.outlayer(out)
        
        return out