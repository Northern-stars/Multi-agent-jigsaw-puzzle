import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3,efficientnet_b0

class piece_compare_model_b3(nn.Module):
    def __init__(self,hidden_1,hidden_2,out=9,dropout=0.1):
        super().__init__()
        self.ef=efficientnet_b3()
        self.ef.classifier=nn.Linear(1536,hidden_1)
        self.compare_layer=nn.Sequential(
            nn.Linear(2*hidden_1,hidden_1),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU()
        )

        self.out_layer=nn.Sequential(
            nn.Linear(hidden_1,hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2,out)
        )
    def forward(self,center_piece,outsider_piece):
        center_feature=self.ef(center_piece)
        outsider_feature=self.ef(outsider_piece)

        feature=torch.cat([center_feature,outsider_feature],dim=-1)
        feature=self.compare_layer(feature)
        out=self.out_layer(feature)
        return out


class piece_compare_model_b0(nn.Module):
    def __init__(self,hidden_1,hidden_2,out=9,dropout=0.1):
        super().__init__()
        self.ef=efficientnet_b0()
        self.ef.classifier=nn.Linear(1280,hidden_1)
        self.compare_layer=nn.Sequential(
            nn.Linear(2*hidden_1,hidden_1),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU()
        )

        self.out_layer=nn.Sequential(
            nn.Linear(hidden_1,hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2,out)
        )
    def forward(self,center_piece,outsider_piece):
        center_feature=self.ef(center_piece)
        outsider_feature=self.ef(outsider_piece)

        feature=torch.cat([center_feature,outsider_feature],dim=-1)
        feature=self.compare_layer(feature)
        out=self.out_layer(feature)
        return out