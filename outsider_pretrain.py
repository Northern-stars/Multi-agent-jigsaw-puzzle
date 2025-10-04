import time
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import copy
import torch
import itertools
from torch import nn
from torchvision.models import efficientnet_b0
from torch.utils.data import Dataset, DataLoader
from Vit import VisionTransformer
from tqdm import tqdm



train_x_path = 'dataset/train_img_48gap_33-001.npy'
train_y_path = 'dataset/train_label_48gap_33.npy'
# test_x_path = 'dataset/train_img_48gap_33-001.npy'
# test_y_path = 'dataset/train_label_48gap_33.npy'
test_x_path = 'dataset/test_img_48gap_33.npy'
test_y_path = 'dataset/test_label_48gap_33.npy'
# test_x_path = 'dataset/valid_img_48gap_33.npy'
# test_y_path = 'dataset/valid_label_48gap_33.npy'

train_x=np.load(train_x_path)
train_y=np.load(train_y_path)

test_x=np.load(test_x_path)
test_y=np.load(test_y_path)


BATCH_SIZE=25
MODEL_NAME="model/outsider_model.pth"


class imageData(Dataset):
    def __init__(self,data,label):
        self.data=data
        self.label=label
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        outsider_index=index
        image=torch.tensor(self.data[index]).permute([2,0,1]).to(torch.float)
        while outsider_index==index:
            outsider_index=random.randint(0,len(self.data)-1)
        # print(index,outsider_index)
        outsider_image=torch.tensor(self.data[outsider_index]).permute([2,0,1]).to(torch.float)
        # print(type(outsider_image))
        image_fragments=[
            image[:,0:96,0:96],
            image[:,0:96,96:192],
            image[:,0:96,192:288],
            image[:,96:192,0:96],
            image[:,96:192,96:192],
            image[:,96:192,192:288],
            image[:,192:288,0:96],
            image[:,192:288,96:192],
            image[:,192:288,192:288]
        ]

        outsider_image_fragments=[
            outsider_image[:,0:96,0:96],
            outsider_image[:,0:96,96:192],
            outsider_image[:,0:96,192:288],
            outsider_image[:,96:192,0:96],
            outsider_image[:,96:192,96:192],
            outsider_image[:,96:192,192:288],
            outsider_image[:,192:288,0:96],
            outsider_image[:,192:288,96:192],
            outsider_image[:,192:288,192:288]
        ]

        label=random.randint(0,8)
        random.shuffle(image_fragments)
        random.shuffle(outsider_image_fragments)
        # print(type(outsider_image_fragments))
        # image_fragments[label]=outsider_image_fragments[label]
        random_number=random.randint(0,4)
        if random_number==0:
            return image,4
        image=torch.zeros(3,288,288)
        swap_index=[]
        for i in range(random_number):
            random_index=random.randint(0,8)
            if random_index not in swap_index:
                swap_index.append(random_index)
                image_fragments[random_index]=outsider_image_fragments[random_index]
        
        for i in range(9):
            image[:,(0+i//3)*96:(1+i//3)*96,(0+i%3)*96:(1+i%3)*96]=image_fragments[i]
        
        # return image,label
        return image,0.5*(8-len(swap_index))


train_data=imageData(train_x,train_y)
train_dataloader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)

class central_fen_model(nn.Module):
    def __init__(self, hidden_size1,hidden_size2,dropout=0.1):
        super().__init__()
        self.ef=efficientnet_b0()
        self.ef.classifier=nn.Linear(1280,hidden_size1)
        self.contract_fc=nn.Linear(2*hidden_size1,hidden_size1)
        self.fc1=nn.Linear(8*hidden_size1,hidden_size1)
        self.relu=nn.ReLU()
        self.bn=nn.BatchNorm1d(hidden_size1)

        self.out_layer=nn.Sequential(
            nn.Linear(hidden_size1,hidden_size2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size2,hidden_size2)
        )
    
    def forward(self, image):
        # 输入: [batch, 3, 288, 288]
        B, C, H, W = image.shape
        # 切分为 9 个碎片 [B, 3, 3, 3, 96, 96]
        patches = image.unfold(2, 96, 96).unfold(3, 96, 96)  # [B, C, 3, 3, 96, 96]
        patches = patches.permute(0,2,3,1,4,5).contiguous()  # [B, 3, 3, C, 96, 96]
        patches = patches.view(B*9, C, 96, 96)  # [B*9, 3, 96, 96]
        # 送入 EfficientNet
        patch_features = self.ef(patches)  # [B*9, hidden_size1]
        patch_features = patch_features.view(B, 9, -1)  # [B, 9, hidden_size1]
        # 中心碎片
        central_tensor = patch_features[:, 4, :]  # [B, hidden_size1]
        # 其他碎片
        other_tensor = torch.cat([patch_features[:, :4, :], patch_features[:, 5:, :]], dim=1)  # [B, 8, hidden_size1]
        # 中心碎片扩展为 [B, 8, hidden_size1]
        central_expanded = central_tensor.unsqueeze(1).expand(-1, 8, -1)
        # 拼接中心与其他碎片 [B, 8, 2*hidden_size1]
        concat_tensor = torch.cat([central_expanded, other_tensor], dim=-1)
        # contract_fc 并行处理
        contracted = self.contract_fc(concat_tensor)  # [B, 8, hidden_size1]
        # 展平成 [B, 8*hidden_size1]
        feature_tensor = contracted.reshape(B, -1)
        feature_tensor = self.fc1(feature_tensor)
        feature_tensor = self.bn(feature_tensor)
        feature_tensor = self.relu(feature_tensor)
        out = self.out_layer(feature_tensor)
        return out

class Buffer_switcher_model(nn.Module):
    def __init__(self,hidden_size1,hidden_size2,hidden_size3,action_num,dropout=0.1):
        super(Buffer_switcher_model,self).__init__()
        self.fen_model=central_fen_model(hidden_size1,hidden_size2,dropout=dropout)
        self.contract_layer=nn.Sequential(
            nn.Linear(hidden_size2,hidden_size3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size3,hidden_size3),
            nn.ReLU()
        )
        self.out_layer=nn.Linear(hidden_size3,action_num)
    def forward(self,image1):
        image1_tensor=self.fen_model(image1)
        out=self.contract_layer(image1_tensor)
        out=self.out_layer(out)

        return out



class fen_model(nn.Module):
    def __init__(self,hidden_size1,hidden_size2):
        super(fen_model,self).__init__()
        self.ef=efficientnet_b0(weights="DEFAULT")
        self.ef.classifier=nn.Linear(1280,64)
        self.fc1=nn.Linear(128*12,hidden_size1)
        self.relu=nn.ReLU()
        self.bn=nn.BatchNorm1d(hidden_size1)
        self.do=nn.Dropout1d(p=0.1)
        self.fc2=nn.Linear(hidden_size1,hidden_size2)
        self.hori_set=[(i,i+1) for i in [j for j in range(9) if j%3!=3-1 ]]
        self.vert_set=[(i,i+3) for i in range(3*2)]
    
    def forward(self,image):
        image_fragments=[
            image[:,:,0:96,0:96],
            image[:,:,0:96,96:192],
            image[:,:,0:96,192:288],
            image[:,:,96:192,0:96],
            image[:,:,96:192,96:192],
            image[:,:,96:192,192:288],
            image[:,:,192:288,0:96],
            image[:,:,192:288,96:192],
            image[:,:,192:288,192:288]
        ]
        
        hori_tensor=torch.cat([torch.cat([self.ef(image_fragments[self.hori_set[i][0]]),self.ef(image_fragments[self.hori_set[i][1]])],dim=-1) for i in range(len(self.hori_set))],dim=-1)
        vert_tensor=torch.cat([torch.cat([self.ef(image_fragments[self.vert_set[i][0]]),self.ef(image_fragments[self.vert_set[i][1]])],dim=-1) for i in range(len(self.vert_set))],dim=-1)
        feature_tensor=torch.cat([hori_tensor,vert_tensor],dim=-1)
        x=self.do(feature_tensor)
        x=self.fc1(x)
        x=self.do(x)
        x=self.bn(x)
        x=self.relu(x)
        x=self.fc2(x)
        return x

device="cuda" if torch.cuda.is_available() else "cpu"
# device="cpu"

class outsider_model(nn.Module):
    def __init__(self,hidden_size1,hidden_size2):
        super(outsider_model,self).__init__()
        self.pre_model=fen_model(hidden_size1,hidden_size2)
        # self.pre_model=VisionTransformer(
        # picture_size=[5,3,96,96],
        # patch_size=12,
        # encoder_hidden=hidden_size1,
        # out_size=hidden_size2,
        # n_head=12,
        # encoder_layer_num=12,
        # unet_hidden=hidden_size1,
        # output_channel=3
        # )
        
        self.fc1=nn.Linear(hidden_size2,hidden_size2)
        self.bn1=nn.BatchNorm1d(hidden_size2)
        self.relu1=nn.ReLU()
        self.dp1=nn.Dropout1d(p=0.1)
        self.fc2=nn.Linear(hidden_size2,hidden_size2)
        self.bn2=nn.BatchNorm1d(hidden_size2)
        self.relu2=nn.ReLU()
        # self.outlayer=nn.Linear(hidden_size2,9)
        self.outlayer=nn.Linear(hidden_size2,5)
    def forward(self,x):
        x=self.pre_model(x)
        if len(x.size())<2:
            x=x.unsqueeze(0)
        x=self.fc1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.dp1(x)
        x=self.fc2(x)
        x=self.bn2(x)
        x=self.relu2(x)
        out=self.outlayer(x)
        return out



# model=outsider_model(1024,1024).to(device)
model=Buffer_switcher_model(
        hidden_size1=2048,
        hidden_size2=1024,
        hidden_size3=1024,
        action_num=1
    ).to(device)



loss_fn=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)

def train(epoch_num=5000,load=True):
    if load:
        model.load_state_dict(torch.load(MODEL_NAME))
    print("start training")
    model.train()
    for epoch in range(epoch_num):
        loss_sum=0
        right=0
        for image,label in tqdm(train_dataloader):
           image,label=image.to(device),label.unsqueeze(-1).to(torch.float).to(device)
           outsider_probs=model(image)
           loss=loss_fn(outsider_probs,label)
        #    y_pred=torch.argmax(outsider_probs).item()
           loss_sum+=loss.item()
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
        
        print(f"epoch: {epoch}, loss_sum: {loss_sum}")
        torch.save(model.state_dict(),MODEL_NAME)
        torch.save(model.fen_model.state_dict(),"model/central_fen.pth")


class test_imageData(Dataset):
    def __init__(self,data,label):
        self.data=data
        self.label=label
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        outsider_index=index
        image=torch.tensor(self.data[index]).permute([2,0,1]).to(torch.float)
        while outsider_index==index:
            outsider_index=random.randint(0,len(self.data)-1)
        # print(index,outsider_index)
        outsider_image=torch.tensor(self.data[outsider_index]).permute([2,0,1]).to(torch.float)
        # print(type(outsider_image))
        image_fragments=[
            image[:,0:96,0:96],
            image[:,0:96,96:192],
            image[:,0:96,192:288],
            image[:,96:192,0:96],
            image[:,96:192,96:192],
            image[:,96:192,192:288],
            image[:,192:288,0:96],
            image[:,192:288,96:192],
            image[:,192:288,192:288]
        ]

        outsider_image_fragments=[
            outsider_image[:,0:96,0:96],
            outsider_image[:,0:96,96:192],
            outsider_image[:,0:96,192:288],
            outsider_image[:,96:192,0:96],
            outsider_image[:,96:192,96:192],
            outsider_image[:,96:192,192:288],
            outsider_image[:,192:288,0:96],
            outsider_image[:,192:288,96:192],
            outsider_image[:,192:288,192:288]
        ]
        replace_num=4
        label=[random.randint(0,8) for i in range(replace_num)]
        random.shuffle(image_fragments)
        random.shuffle(outsider_image_fragments)
        # print(type(outsider_image_fragments))
        for i in range(replace_num):
            image_fragments[label[i]]=outsider_image_fragments[label[i]]
        image=torch.zeros(3,288,288)
        for i in range(9):
            image[:,(0+i//3)*96:(1+i//3)*96,(0+i%3)*96:(1+i%3)*96]=image_fragments[i]
        
        return image,label

test_data=imageData(test_x,test_y)
test_dataloader=DataLoader(test_data,batch_size=1,shuffle=True)
def test():
    model.load_state_dict(torch.load(MODEL_NAME))
    model.eval()
    right=0
    with torch.no_grad():
        for image,label in tqdm(test_dataloader):
            image=image.to(device)
            y_pred=model(image)
            loss_sum+=loss_fn(y_pred,label).item()
        print(f"Loss_sum: {loss_sum}")




if __name__=="__main__":
    # HORI_MODEL_NAME="model/hori_ef0.pth"
    # VERT_MODEL_NAME="model/vert_ef0.pth"
    train(50,load=False)
    # test()
