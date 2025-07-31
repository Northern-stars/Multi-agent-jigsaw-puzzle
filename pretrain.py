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
import Vit




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

HORI_MODEL_NAME="hori_ef0.pth"
VERT_MODEL_NAME="vert_ef0.pth"
BATCH_SIZE=150

class imageData(Dataset):
    def __init__(self,data,label):
        self.data=data
        self.label=label
        self.choice=list(itertools.combinations(list(range(9)), 2))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        
        image=torch.tensor(self.data[index]).permute([2,0,1]).to(torch.float)
        
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
        hori_label=0
        vert_label=0
        
        imageIndex=random.choice(self.choice)
        

        if imageIndex[0]%3==imageIndex[1]%3 and abs((imageIndex[0]//3)-(imageIndex[1]//3))==1:
            vert_label=1

        elif abs((imageIndex[0]%3)-(imageIndex[1]%3))==1 and imageIndex[0]//3==imageIndex[1]//3:
            hori_label=1
        imageLabel=self.label[index]
        imageLabel=list(imageLabel)
        # print(imageLabel)
        for i in range(8):
            for j in range(8):
                # print(type(imageLabel[i]))
                # print(imageLabel)
                if imageLabel[i][j]==True:
                    if j>=4:
                        imageLabel[i]=j+1
                        break
                    else:
                        imageLabel[i]=j
                        break
        imageLabel.insert(4,4)
        # print(imageLabel)
        final_image_hori_a=image_fragments[imageLabel[imageIndex[0]]]
        final_image_hori_b=image_fragments[imageLabel[imageIndex[1]]]
        final_image_vert_a=image_fragments[imageLabel[imageIndex[0]]]
        final_image_vert_b=image_fragments[imageLabel[imageIndex[1]]]
        return final_image_hori_a,final_image_hori_b,final_image_vert_a,final_image_vert_b,hori_label,vert_label


train_data=imageData(train_x,train_y)
train_dataloader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
test_data=imageData(test_x,test_y)
test_dataloader=DataLoader(test_data,batch_size=1,shuffle=True)

# class fen_model(nn.Module):
#     def __init__(self,hidden_size1,hidden_size2):
#         super(fen_model,self).__init__()
#         self.ef=efficientnet_b0(weights="DEFAULT")
#         self.ef.classifier=nn.Identity()
#         self.fc1=nn.Linear(1280,hidden_size1)
#         self.relu=nn.ReLU()
#         self.bn=nn.BatchNorm1d(hidden_size1)
#         self.do=nn.Dropout1d(p=0.3)
#         self.fc2=nn.Linear(hidden_size1,hidden_size2)
    
#     def forward(self,x):
#         x=self.ef(x)
#         x=self.do(x)
#         x=self.fc1(x)
#         x=self.do(x)
#         x=self.bn(x)
#         x=self.relu(x)
#         x=self.fc2(x)
#         return x

device="cuda" if torch.cuda.is_available() else "cpu"
# device="cpu"

class pretrain_model(nn.Module):
    def __init__(self,hidden_size1,hidden_size2):
        super(pretrain_model,self).__init__()
        # self.ef=efficientnet_b0(weights="DEFAULT")
        # self.ef.classifier=nn.Linear(1280,hidden_size1)
        self.ef=Vit.VisionTransformer(
        picture_size=[5,3,96,96],
        patch_size=12,
        encoder_hidden=hidden_size1,
        out_size=hidden_size1,
        n_head=12,
        encoder_layer_num=12,
        unet_hidden=hidden_size1,
        output_channel=3
        )
        self.fc1=nn.Linear(2*hidden_size1,hidden_size2)
        self.bn1=nn.BatchNorm1d(hidden_size2)
        self.relu1=nn.ReLU()
        self.dp1=nn.Dropout1d(p=0.3)
        self.fc2=nn.Linear(hidden_size2,hidden_size2)
        self.bn2=nn.BatchNorm1d(hidden_size2)
        self.relu2=nn.ReLU()
        self.outlayer=nn.Linear(hidden_size2,2)
    def forward(self,x1,x2):
        feature_tensor=torch.cat([self.ef(x1),self.ef(x2)],dim=-1)
        x=self.fc1(feature_tensor)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.dp1(x)
        x=self.fc2(x)
        x=self.bn2(x)
        x=self.relu2(x)
        out=self.outlayer(x)
        return out



model=pretrain_model(64,256).to(device)



loss_fn=nn.CrossEntropyLoss().to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,eps=1e-8)


def single_train(model,optimizer,data,label):
    label=label.to(device)
    output=model(data[0].to(device),data[1].to(device))
    loss=loss_fn(output,label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def train(epoch_num=5000,load=True):
    if load:
        model.load_state_dict(torch.load(MODEL_NAME))
    print("start training")
    model.train()
    for epoch in range(epoch_num):
        hori_loss_sum=0
        vert_loss_sum=0
        for batch_num,(final_image_hori_a,final_image_hori_b,final_image_vert_a,final_image_vert_b,hori_label,vert_label) in enumerate(train_dataloader):
            hori_loss_sum+=single_train(model=model,optimizer=optimizer,data=[final_image_hori_a,final_image_hori_b],label=hori_label)
            vert_loss_sum+=single_train(model=model,optimizer=optimizer,data=[final_image_vert_a,final_image_vert_b],label=vert_label)

            
        print(f"epochnum: {epoch}, hori_loss_sum: {hori_loss_sum*BATCH_SIZE}, vert_loss_sum: {vert_loss_sum*BATCH_SIZE}")
        torch.save(model.state_dict(),MODEL_NAME)


def test():
    with torch.no_grad():
        model.load_state_dict(torch.load(MODEL_NAME))
        model.eval()
        hori_right=0
        vert_right=0
        print("start testing")
        for batch_num,(final_image_hori_a,final_image_hori_b,final_image_vert_a,final_image_vert_b,hori_label,vert_label) in enumerate(test_dataloader):
            hori_output=torch.argmax(model(final_image_hori_a.to(device),final_image_hori_b.to(device))).item()
            vert_output=torch.argmax(model(final_image_vert_a.to(device),final_image_vert_b.to(device))).item()
            hori_right+=(hori_output==hori_label[0].item())
            vert_right+=(vert_output==vert_label[0].item())
            if (batch_num+1)%100==0:
                print(f"batch num: {batch_num+1}, hori right level: {hori_right/(batch_num+1)}, vert right level: {vert_right/(batch_num+1)}")
        
    print(f"hori right level: {hori_right/len(test_dataloader)}, vert right level: {vert_right/len(test_dataloader)}")

if __name__=="__main__":
    # HORI_MODEL_NAME="hori_ef0.pth"
    # VERT_MODEL_NAME="vert_ef0.pth"
    MODEL_NAME="pairwise_pretrain_Vit.pth"
    # train(50,load=False)
    test()
