from fen_model import central_fen_model
import torch.nn as nn
import torch


class Buffer_switcher_model(nn.Module):
    def __init__(self,hidden_size1,hidden_size2,hidden_size3,action_num,dropout=0.1):
        super(Buffer_switcher_model,self).__init__()
        self.fen_model=central_fen_model(hidden_size1,hidden_size2,dropout=dropout)
        self.contract_layer=nn.Sequential(
            nn.Linear(2*hidden_size2,hidden_size3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size3,hidden_size3),
            nn.ReLU()
        )
        self.out_layer=nn.Linear(hidden_size3,action_num)
    def forward(self,image1,image2):
        image1_tensor=self.fen_model(image1)
        image2_tensor=self.fen_model(image2)
        out=self.contract_layer(torch.cat([image1_tensor,image2_tensor],dim=-1))
        out=self.out_layer(out)

        return out