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
from gated_linear_sample import gated_linear
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

HORI_MODEL_NAME="model/hori_ef0_gated.pth"
VERT_MODEL_NAME="model/vert_ef0_gated.pth"
BATCH_SIZE=128

class imageData(Dataset):
    def __init__(self,data,label):
        self.data=data
        self.label=label
        self.choice=list(itertools.combinations(list(range(9)), 2))
        self.hori_set=[(i,i+1) for i in [j for j in range(9) if j%3!=3-1 ]]
        self.vert_set=[(i,i+3) for i in range(3*2)]
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
        
        right=random.random()
        if right>=0.5:
            vert_label=1
            hori_label=1

            hori_idx=random.choice(self.hori_set)
            vert_idx=random.choice(self.vert_set)

        else:
            vert_label=0
            hori_label=0

            hori_idx=(random.randint(0,8),random.randint(0,8))
            while (hori_idx in self.hori_set) or (hori_idx[0]==hori_idx[1]):
                hori_idx=(random.randint(0,8),random.randint(0,8))
            vert_idx=(random.randint(0,8),random.randint(0,8))
            while (vert_idx in self.vert_set) or (vert_idx[0]==vert_idx[1]):
                vert_idx=(random.randint(0,8),random.randint(0,8))

        imageLabel=self.label[index]
        imageLabel=list(imageLabel)
        # print(imageLabel)
        for a in range(8):
            for b in range(8):
                # print(type(imageLabel[i]))
                # print(imageLabel)
                if imageLabel[a][b]==True:
                    if b>=4:
                        imageLabel[a]=b+1
                        break
                    else:
                        imageLabel[a]=b
                        break
        imageLabel.insert(4,4)
        # print(imageLabel)



        final_image_hori_a=image_fragments[imageLabel.index(hori_idx[0])]
        final_image_hori_b=image_fragments[imageLabel.index(hori_idx[1])]
        final_image_vert_a=image_fragments[imageLabel.index(vert_idx[0])]
        final_image_vert_b=image_fragments[imageLabel.index(vert_idx[1])]
        final_image_hori=torch.cat([final_image_hori_a,final_image_hori_b],dim=-1)
        final_image_vert=torch.cat([final_image_vert_a,final_image_vert_b],dim=-2)
        return final_image_hori,final_image_vert,hori_label,vert_label




train_data=imageData(train_x,train_y)
train_dataloader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
test_data=imageData(test_x,test_y)
test_dataloader=DataLoader(test_data,batch_size=1,shuffle=True)

class fen_model(nn.Module):
    def __init__(self,hidden_size1,hidden_size2):
        super(fen_model,self).__init__()
        self.ef=efficientnet_b0(weights="DEFAULT")
        self.ef.classifier=nn.Identity()
        self.fc1=nn.Linear(1280,hidden_size1)
        self.relu=nn.ReLU()
        self.bn=nn.BatchNorm1d(hidden_size1)
        self.do=nn.Dropout1d(p=0.3)
        self.fc2=nn.Linear(hidden_size1,hidden_size2)
    
    def forward(self,x,tau):
        x=self.ef(x)
        x=self.do(x)
        # x=self.fc1(x,tau)
        x=self.fc1(x)
        x=self.do(x)
        x=self.bn(x)
        x=self.relu(x)
        # x=self.fc2(x,tau)
        x=self.fc2(x)
        return x

device="cuda" if torch.cuda.is_available() else "cpu"
# device="cpu"

class pretrain_model(nn.Module):
    def __init__(self,hidden_size1,hidden_size2):
        super(pretrain_model,self).__init__()
        self.pre_model=fen_model(hidden_size1,hidden_size2)
        self.fc1=nn.Linear(hidden_size2,hidden_size2)
        self.bn1=nn.BatchNorm1d(hidden_size2)
        self.relu1=nn.ReLU()
        self.dp1=nn.Dropout1d(p=0.3)
        self.fc2=nn.Linear(hidden_size2,hidden_size2)
        self.bn2=nn.BatchNorm1d(hidden_size2)
        self.relu2=nn.ReLU()
        self.outlayer=nn.Linear(hidden_size2,2)
    def forward(self,x,tau=1):
        x=self.pre_model(x,tau)
        # x=self.fc1(x,tau)
        x=self.fc1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.dp1(x)
        # x=self.fc2(x,tau)
        x=self.fc2(x)
        x=self.bn2(x)
        x=self.relu2(x)
        # out=self.outlayer(x,tau)
        out=self.outlayer(x)
        return out



hori_model=pretrain_model(512,512).to(device)
vert_model=pretrain_model(512,512).to(device)


loss_fn=nn.CrossEntropyLoss().to(device)
hori_optimizer=torch.optim.Adam(hori_model.parameters(),lr=1e-3,eps=1e-8)
vert_optimizer=torch.optim.Adam(vert_model.parameters(),lr=1e-3,eps=1e-8)

def train(epoch_num=5000,load=True):
    if load:
        hori_model.load_state_dict(torch.load(HORI_MODEL_NAME))
        vert_model.load_state_dict(torch.load(VERT_MODEL_NAME))
    print("start training")
    hori_model.train()
    vert_model.train()
    tau=5
    for epoch in range(epoch_num):
        hori_loss_sum=0
        vert_loss_sum=0
        for final_hori_image,final_vert_image,hori_label,vert_label in tqdm(train_dataloader):
            final_hori_image=final_hori_image.to(device)
            final_vert_image=final_vert_image.to(device)

            hori_label=hori_label.to(device)
            hori_output=hori_model(final_hori_image,tau)
            hori_loss=loss_fn(hori_output,hori_label)
            
            vert_output=vert_model(final_vert_image,tau)
            vert_label=vert_label.to(device)
            vert_loss=loss_fn(vert_output,vert_label)

            hori_loss_sum+=(hori_loss.item())
            vert_loss_sum+=+vert_loss.item()

            hori_optimizer.zero_grad()
            hori_loss.backward()
            hori_optimizer.step()

            vert_optimizer.zero_grad()
            vert_loss.backward()
            vert_optimizer.step()
            tau=tau*0.995
        print(f"epochnum: {epoch}, hori_loss_sum: {hori_loss_sum*BATCH_SIZE}, vert_loss_sum: {vert_loss_sum*BATCH_SIZE}")
        torch.save(hori_model.state_dict(),HORI_MODEL_NAME)
        torch.save(vert_model.state_dict(),VERT_MODEL_NAME)

def test():
    with torch.no_grad():
        hori_model.load_state_dict(torch.load(HORI_MODEL_NAME))
        vert_model.load_state_dict(torch.load(VERT_MODEL_NAME))
        hori_model.eval()
        vert_model.eval()
        hori_right=0
        vert_right=0
        print("start testing")
        step=0
        for final_hori_image,final_vert_image,hori_label,vert_label in tqdm(test_dataloader):
            final_hori_image=final_hori_image.to(device)
            final_vert_image=final_vert_image.to(device)
            # hori_label=hori_label.to(device)
            # vert_label=vert_label.to(device)
            hori_output=torch.argmax(hori_model(final_hori_image)).item()
            vert_output=torch.argmax(vert_model(final_vert_image)).item()
            hori_right+=(hori_output==hori_label[0].item())
            vert_right+=(vert_output==vert_label[0].item())
            if (step+1)%100==0:
                print(f"Step: {step}, hori right level: {hori_right/(step+1)}, vert right level: {vert_right/(step+1)}")
        
    print(f"hori right level: {hori_right/len(test_dataloader)}, vert right level: {vert_right/len(test_dataloader)}")

if __name__=="__main__":
    HORI_MODEL_NAME="model/gated_hori_ef0.pth"
    VERT_MODEL_NAME="model/gated_vert_ef0.pth"
    train(50,load=False)
    test()
