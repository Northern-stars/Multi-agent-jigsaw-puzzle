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


BATCH_SIZE=64
MODEL_NAME="outsider_model.pth"


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
        image_fragments[label]=outsider_image_fragments[label]
        image=torch.zeros(3,288,288)
        for i in range(9):
            image[:,(0+i//3)*96:(1+i//3)*96,(0+i%3)*96:(1+i%3)*96]=image_fragments[i]
        
        return image,label


train_data=imageData(train_x,train_y)
train_dataloader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)


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
    
    def forward(self,x):
        x=self.ef(x)
        x=self.do(x)
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
        self.fc1=nn.Linear(hidden_size2,hidden_size2)
        self.bn1=nn.BatchNorm1d(hidden_size2)
        self.relu1=nn.ReLU()
        self.dp1=nn.Dropout1d(p=0.3)
        self.fc2=nn.Linear(hidden_size2,hidden_size2)
        self.bn2=nn.BatchNorm1d(hidden_size2)
        self.relu2=nn.ReLU()
        self.outlayer=nn.Linear(hidden_size2,9)
    def forward(self,x):
        x=self.pre_model(x)
        x=self.fc1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.dp1(x)
        x=self.fc2(x)
        x=self.bn2(x)
        x=self.relu2(x)
        out=self.outlayer(x)
        return out



model=outsider_model(512,512).to(device)



loss_fn=nn.CrossEntropyLoss().to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)

def train(epoch_num=5000,load=True):
    if load:
        model.load_state_dict(torch.load(MODEL_NAME))
    print("start training")
    model.train()
    for epoch in range(epoch_num):
        loss_sum=0
        right=0
        for batch_num,(image,label) in enumerate(train_dataloader):
           image,label=image.to(device),label.to(device)
           outsider_probs=model(image)
           loss=loss_fn(outsider_probs,label)
        #    y_pred=torch.argmax(outsider_probs).item()
           loss_sum+=loss.item()
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
        
        print(f"epoch: {epoch}, loss_sum: {loss_sum}")
    torch.save(model.state_dict(),MODEL_NAME)


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

test_data=test_imageData(test_x,test_y)
test_dataloader=DataLoader(test_data,batch_size=1,shuffle=True)
def test():
    model.load_state_dict(torch.load(MODEL_NAME))
    model.eval()
    right=0
    with torch.no_grad():
        for batch_num,(image,label) in enumerate(test_dataloader):
            image=image.to(device)
            y_pred=torch.argmax(model(image)).item()
            right+=y_pred in label
        print(f"Accuracy: {right/len(test_dataloader)}")




if __name__=="__main__":
    HORI_MODEL_NAME="hori_ef0.pth"
    VERT_MODEL_NAME="vert_ef0.pth"
    # train(50)
    test()
