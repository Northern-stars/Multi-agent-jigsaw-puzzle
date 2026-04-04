import torch.nn as nn
import torch
from model_code.fen_model import fen_model
from torchvision.models import efficientnet_b0
from pretrain.pretrain_1 import pretrain_model
class General_switcher_model(nn.Module):
    def __init__(self,hidden_size1,hidden_size2,outsider_hidden_size,out_size,dropout=0.1,model_name="ef"):
        super(General_switcher_model,self).__init__()
        self.fen_model=fen_model(hidden_size1=hidden_size1,hidden_size2=hidden_size1,model_name=model_name)
        self.outsider_fen_model=efficientnet_b0(weights="DEFAULT")
        self.outsider_fen_model.classifier=nn.Linear(1280,outsider_hidden_size)
        self.outsider_contrast_fc=nn.Linear(2*outsider_hidden_size,outsider_hidden_size)
        self.outsider_fc=nn.Linear(outsider_hidden_size*9,outsider_hidden_size)
        # state_dict=torch.load("pairwise_pretrain.pth")
        # state_dict_replace = {
        # k: v 
        # for k, v in state_dict.items() 
        # if k.startswith("ef.")
        # }
        # load_result_hori=self.fen_model.load_state_dict(state_dict_replace,strict=False)
        # print("Critic missing keys hori",load_result_hori.missing_keys)
        # print("Critic unexpected keys hori",load_result_hori.unexpected_keys)
        self.fc1=nn.Linear(hidden_size1+outsider_hidden_size,hidden_size2)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=dropout)
        self.bn=nn.BatchNorm1d(hidden_size2)
        self.fc=nn.Linear(hidden_size2,out_size)
    
    def forward(self,image,outsider):
        B=image.size(0)
        
        image_input=self.fen_model(image)
        outsider_input=self.outsider_fen_model(outsider)

        outsider_image_tensor=image.unfold(2,96,96).unfold(3,96,96)
        outsider_image_tensor=outsider_image_tensor.permute(0,2,3,1,4,5).contiguous()
        outsider_image_tensor=outsider_image_tensor.view(B*9,-1,96,96)

        outsider_image_tensor=self.outsider_fen_model(outsider_image_tensor)
        outsider_image_tensor=outsider_image_tensor.view(B,9,-1)
        outsider_input=outsider_input.unsqueeze(1).expand(B,9,-1)

        outsider_tensor=self.outsider_contrast_fc(torch.cat([outsider_input,outsider_image_tensor],dim=-1)) 

        outsider_tensor=outsider_tensor.view(B,-1)
        outsider_tensor=self.outsider_fc(outsider_tensor)

        feature_tensor=torch.cat([image_input,outsider_tensor],dim=-1)
        out=self.fc1(feature_tensor)
        out=self.dropout(out)
        out=self.bn(out)
        out=self.relu(out)
        out=self.fc(out)
        return out