import torch
import torch.nn as nn
import numpy as np
import copy
import itertools
from torchvision.models import efficientnet_b0,efficientnet_b3
from pretrain import pretrain_model
import cv2
import os
import time

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

class fen_model(nn.Module):
    def __init__(self, hidden_size1, hidden_size2):
        super(fen_model, self).__init__()
        self.ef = efficientnet_b3(weights="DEFAULT")
        self.ef.classifier = nn.Linear(1536, 1024)

        self.contrast_fc_hori = nn.Linear(2048, 1024)
        self.contrast_fc_vert = nn.Linear(2048, 1024)

        self.fc1 = nn.Linear(1024*12, hidden_size1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_size1)
        self.do = nn.Dropout1d(p=0.1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)

        # 定义 index 对
        self.hori_set = [(i, i+1) for i in range(9) if i % 3 != 2]
        self.vert_set = [(i, i+3) for i in range(6)]

    def forward(self, image, mask=None):
        B, C, H, W = image.shape
        
        patches = image.unfold(2, 96, 96).unfold(3, 96, 96)
        # patches: (B, C, 3, 3, 96, 96)
        patches = patches.permute(0,2,3,1,4,5).contiguous()
        patches = patches.view(B*9, C, 96, 96)  

        feats = self.ef(patches)  
        feats = feats.view(B, 9, 1024) 

        hori_pairs = torch.stack([torch.cat([feats[:, i, :], feats[:, j, :]], dim=-1) 
                                  for (i,j) in self.hori_set], dim=1)
        hori_feats = self.contrast_fc_hori(hori_pairs) 

        vert_pairs = torch.stack([torch.cat([feats[:, i, :], feats[:, j, :]], dim=-1) 
                                  for (i,j) in self.vert_set], dim=1)  
        vert_feats = self.contrast_fc_vert(vert_pairs) 

        if mask is not None:
            hori_mask=torch.ones(B,len(self.hori_set)).to(DEVICE)
            vert_mask=torch.ones(B,len(self.vert_set)).to(DEVICE)
            for i in range(len(mask)):
                for j in range(len(mask[i])):
                    batch_id,piece_id=i,mask[i][j]
                    for k in range(len(self.hori_set)):
                        if piece_id in self.hori_set[k]:
                            hori_mask[batch_id][k]=0
                        if piece_id in self.vert_set[k]:
                            vert_mask[batch_id][k]=0

            

            hori_mask=hori_mask.unsqueeze(-1)
            vert_mask=vert_mask.unsqueeze(-1)

            hori_feats=hori_feats*hori_mask
            vert_feats=vert_feats*vert_mask


        feature_tensor = torch.cat([hori_feats, vert_feats], dim=1)
        feature_tensor = feature_tensor.view(B, -1)  

        x = self.do(feature_tensor)
        x = self.fc1(x)
        x = self.do(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class Buffer_switcher_model(nn.Module):
    def __init__(self,hidden_size1,hidden_size2,outsider_hidden_size,action_num):
        super(Buffer_switcher_model,self).__init__()
        self.fen_model=fen_model(hidden_size1,hidden_size1)
        # state_dict=torch.load("pairwise_pretrain.pth")
        # state_dict_replace = {
        # k: v 
        # for k, v in state_dict.items() 
        # if k.startswith("ef.")
        # }
        # load_result_hori=self.image_fen_model.load_state_dict(state_dict_replace,strict=False)
        # print("Actor missing keys hori",load_result_hori.missing_keys)
        # print("Actor unexpected keys hori",load_result_hori.unexpected_keys)
        self.outsider_fen_model=efficientnet_b0(weights="DEFAULT")
        self.outsider_fen_model.classifier=nn.Linear(1280,outsider_hidden_size)
        self.outsider_contrast_fc=nn.Linear(2*outsider_hidden_size,outsider_hidden_size)
        self.outsider_fc=nn.Linear(9*outsider_hidden_size,outsider_hidden_size)
        self.fc1=nn.Linear(hidden_size1+outsider_hidden_size,hidden_size2)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=0.1)
        self.bn=nn.BatchNorm1d(hidden_size2)
        self.fc2=nn.Linear(hidden_size2,action_num)
        
    
    def forward(self,image,outsider_piece,mask=None):
        # image_fragments=[
        #     image[:,:,0:96,0:96],
        #     image[:,:,0:96,96:192],
        #     image[:,:,0:96,192:288],
        #     image[:,:,96:192,0:96],
        #     image[:,:,96:192,96:192],
        #     image[:,:,96:192,192:288],
        #     image[:,:,192:288,0:96],
        #     image[:,:,192:288,96:192],
        #     image[:,:,192:288,192:288]
        # ]
        B=image.size(0)

        image_input=self.fen_model(image,mask)
        outsider_input=self.outsider_fen_model(outsider_piece)

        outsider_image_tensor=image.unfold(2,96,96).unfold(3,96,96)
        outsider_image_tensor=outsider_image_tensor.permute(0,2,3,1,4,5).contiguous()
        outsider_image_tensor=outsider_image_tensor.view(B*9,-1,96,96)

        # outsider_image_tensor=[self.outsider_fen_model(image_fragments[i]) for i in range(len(image_fragments)) ]
        outsider_image_tensor=self.outsider_fen_model(outsider_image_tensor)
        outsider_image_tensor=outsider_image_tensor.view(B,9,-1)
        outsider_input=outsider_input.unsqueeze(1).expand(B,9,-1)

        outsider_tensor=self.outsider_contrast_fc(torch.cat([outsider_input,outsider_image_tensor],dim=-1)) 
        # outsider_tensor=self.outsider_contrast_fc(torch.cat([outsider_input,image_input],dim=-1))
        if mask is not None:
            mask_matrix=torch.ones(B,9).to(DEVICE)
            for i in range(len(mask)):
                for j in range(len(mask[i])):
                    if mask[i][j]==9:
                        mask_matrix[i]=0
                    else:
                        mask_matrix[i][mask[i][j]]=0

            mask_matrix=mask_matrix.unsqueeze(-1)

            outsider_tensor=outsider_tensor*mask_matrix

        outsider_tensor=outsider_tensor.view(B,-1)
        outsider_tensor=self.outsider_fc(outsider_tensor)

        feature_tensor=torch.cat([image_input,outsider_tensor],dim=-1)
        out=self.fc1(feature_tensor)
        out=self.dropout(out)
        out=self.bn(out)
        out=self.relu(out)
        out=self.fc2(out)
        # out=nn.functional.sigmoid(out)
        return out

class Decider_model(nn.Module):
    def __init__(self, fen_model_hidden1,fen_model_hidden2,outsider_hidden,hidden_1,hidden_2,action_num,dropout=0.1):
        super().__init__()
        self.fen_model=fen_model(fen_model_hidden1,fen_model_hidden2)
        # state_dict=torch.load("pairwise_pretrain.pth")
        # state_dict_replace = {
        # k: v 
        # for k, v in state_dict.items() 
        # if k.startswith("ef.")
        # }
        # load_result_hori=self.image_fen_model.load_state_dict(state_dict_replace,strict=False)
        # print("Actor missing keys hori",load_result_hori.missing_keys)
        # print("Actor unexpected keys hori",load_result_hori.unexpected_keys)
        self.outsider_fen=efficientnet_b0(weights="DEFAULT")
        self.outsider_fen.classifier=nn.Linear(1280,outsider_hidden)
        self.outsider_contrast_fc=nn.Linear(2*outsider_hidden,outsider_hidden)
        self.outsider_fc=nn.Linear(9*outsider_hidden,outsider_hidden)
        self.fc1=nn.Linear(fen_model_hidden2+outsider_hidden,hidden_1)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=dropout)
        self.bn=nn.BatchNorm1d(hidden_1)
        self.fc2=nn.Linear(hidden_1,hidden_2)
        self.outlayer=nn.Linear(hidden_2,action_num)
    
    def forward(self,image,outsider_piece,mask=None):
        # image_fragments=[
        #     image[:,:,0:96,0:96],
        #     image[:,:,0:96,96:192],
        #     image[:,:,0:96,192:288],
        #     image[:,:,96:192,0:96],
        #     image[:,:,96:192,96:192],
        #     image[:,:,96:192,192:288],
        #     image[:,:,192:288,0:96],
        #     image[:,:,192:288,96:192],
        #     image[:,:,192:288,192:288]
        # ]
        B=image.size(0)

        image_input=self.fen_model(image,mask)
        outsider_input=self.outsider_fen(outsider_piece)

        outsider_image_tensor=image.unfold(2,96,96).unfold(3,96,96)
        outsider_image_tensor=outsider_image_tensor.permute(0,2,3,1,4,5).contiguous()
        outsider_image_tensor=outsider_image_tensor.view(B*9,-1,96,96)

        # outsider_image_tensor=[self.outsider_fen_model(image_fragments[i]) for i in range(len(image_fragments)) ]
        outsider_image_tensor=self.outsider_fen(outsider_image_tensor)
        outsider_image_tensor=outsider_image_tensor.view(B,9,-1)
        outsider_input=outsider_input.unsqueeze(1).expand(B,9,-1)

        outsider_tensor=self.outsider_contrast_fc(torch.cat([outsider_input,outsider_image_tensor],dim=-1)) 
        # outsider_tensor=self.outsider_contrast_fc(torch.cat([outsider_input,image_input],dim=-1))
        if mask is not None:
            mask_matrix=torch.ones(B,9).to(DEVICE)
            for i in range(len(mask)):
                for j in range(len(mask[i])):
                    if mask[i][j]==9:
                        mask_matrix[i]=0
                    else:
                        mask_matrix[i][mask[i][j]]=0
            mask_matrix=mask_matrix.unsqueeze(-1)

            outsider_tensor=outsider_tensor*mask_matrix
        outsider_tensor=outsider_image_tensor.view(B,-1)
        outsider_tensor=self.outsider_fc(outsider_tensor)
        feature_tensor=torch.cat([image_input,outsider_tensor],dim=-1)
        out=self.fc1(feature_tensor)
        out=self.dropout(out)
        out=self.bn(out)
        out=self.relu(out)
        out=self.fc2(out)
        out=self.dropout(out)
        out=self.relu(out)
        out=self.outlayer(out)
        # out=nn.functional.sigmoid(out)
        return out

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
    
    def forward(self,image,mask=None):
        feature_tensor=self.fen_model(image,mask)
        out=self.fc1(feature_tensor)
        out=self.relu(out)
        out=self.bn1(out)
        out=self.do(out)
        out=self.fc2(out)
        out=self.relu(out)
        out=self.bn2(out)
        out=self.outlayer(out)
        
        return out