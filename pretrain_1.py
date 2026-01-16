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

HORI_MODEL_NAME="model/hori_ef0.pth"
VERT_MODEL_NAME="model/vert_ef0.pth"
BATCH_SIZE=150

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
        return final_image_hori_a,final_image_hori_b,final_image_vert_a,final_image_vert_b,torch.tensor(hori_label,dtype=torch.float).unsqueeze(0),torch.tensor(vert_label,dtype=torch.float).unsqueeze(0)


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
        self.ef=efficientnet_b0(weights="DEFAULT")
        self.ef.classifier=nn.Linear(1280,hidden_size1)
        # self.ef=Vit.VisionTransformer(
        # picture_size=[5,3,96,96],
        # patch_size=12,
        # encoder_hidden=hidden_size1,
        # out_size=hidden_size1,
        # n_head=12,
        # encoder_layer_num=12,
        # unet_hidden=hidden_size1,
        # output_channel=3
        # )
        self.contrast_fc=nn.Linear(2*hidden_size1,hidden_size2)
        self.bn1=nn.BatchNorm1d(hidden_size2)
        self.relu1=nn.ReLU()
        self.dp1=nn.Dropout1d(p=0.3)
        self.fc2=nn.Linear(hidden_size2,hidden_size2)
        self.bn2=nn.BatchNorm1d(hidden_size2)
        self.relu2=nn.ReLU()
        self.outlayer=nn.Sequential(
            nn.Linear(hidden_size2,1),
            nn.Sigmoid()
        )
    def forward(self,x1,x2):
        feature1=self.ef(x1)
        feature2=self.ef(x2)

        feature_tensor=torch.cat([feature1,feature2],dim=-1)
        pairwise_feature=self.contrast_fc(feature_tensor)
        pairwise_feature=self.bn1(pairwise_feature)
        pairwise_feature=self.relu1(pairwise_feature)
        pairwise_feature=self.dp1(pairwise_feature)
        pairwise_feature=self.fc2(pairwise_feature)
        pairwise_feature=self.bn2(pairwise_feature)
        pairwise_feature=self.relu2(pairwise_feature)
        out=self.outlayer(pairwise_feature)
        return out





hori_model=pretrain_model(512,512).to(device)
vert_model=pretrain_model(512,512).to(device)
# hori_model=pretrain_model(256,256).to(device)
# vert_model=pretrain_model(256,256).to(device)



loss_fn=nn.BCELoss()
# hori_optimizer=torch.optim.Adam(hori_model.parameters(),lr=1e-4,eps=1e-8)
# vert_optimizer=torch.optim.Adam(vert_model.parameters(),lr=1e-4,eps=1e-8)
hori_optimizer=torch.optim.Adam(hori_model.parameters(),lr=1e-4,eps=1e-8)
vert_optimizer=torch.optim.Adam(vert_model.parameters(),lr=1e-4,eps=1e-8)


def single_train(model,optimizer,data,label):
    label=label.to(device)
    output=model(data[0].to(device),data[1].to(device))
    loss=loss_fn(output,label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def single_train_hori(model,optimizer,data,label):
    label=label.to(device)
    output,_=model(data[0].to(device),data[1].to(device))
    loss=loss_fn(output,label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def single_train_vert(model,optimizer,data,label):
    label=label.to(device)
    _,output=model(data[0].to(device),data[1].to(device))
    loss=loss_fn(output,label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def train(epoch_num=5000,load=True):
    if load:
        # model.load_state_dict(torch.load(MODEL_NAME))
        hori_model.load_state_dict(torch.load(HORI_MODEL_NAME))
        vert_model.load_state_dict(torch.load(VERT_MODEL_NAME))
    print("start training")
    # model.train()
    hori_model.train()
    vert_model.train()
    for epoch in range(epoch_num):
        hori_loss_sum=0
        vert_loss_sum=0
        for final_image_hori_a,final_image_hori_b,final_image_vert_a,final_image_vert_b,hori_label,vert_label in tqdm(train_dataloader):
            # hori_loss_sum+=single_train_hori(model=model,optimizer=optimizer,data=[final_image_hori_a,final_image_hori_b],label=hori_label)
            # vert_loss_sum+=single_train_vert(model=model,optimizer=optimizer,data=[final_image_vert_a,final_image_vert_b],label=vert_label)
            hori_loss_sum+=single_train(model=hori_model,optimizer=hori_optimizer,data=[final_image_hori_a,final_image_hori_b],label=hori_label)
            vert_loss_sum+=single_train(model=vert_model,optimizer=vert_optimizer,data=[final_image_vert_a,final_image_vert_b],label=vert_label)

            
            
        print(f"epochnum: {epoch}, hori_loss_sum: {hori_loss_sum*BATCH_SIZE}, vert_loss_sum: {vert_loss_sum*BATCH_SIZE}")
        # torch.save(model.state_dict(),MODEL_NAME)
        torch.save(hori_model.state_dict(),HORI_MODEL_NAME)
        torch.save(vert_model.state_dict(),VERT_MODEL_NAME)



def test(hori_model,vert_model):
    """二分类测试函数"""
    with torch.no_grad():
        # model.load_state_dict(torch.load(MODEL_NAME))
        # model.eval()
        # hori_model.load_state_dict(torch.load(HORI_MODEL_NAME))
        # vert_model.load_state_dict(torch.load(VERT_MODEL_NAME))
        hori_model.eval()
        vert_model.eval()
        
        hori_right = 0
        vert_right = 0
        total_samples = 0
        
        print("start testing")
        
        # 使用tqdm显示进度
        for batch_num, (final_image_hori_a, final_image_hori_b, final_image_vert_a, final_image_vert_b, hori_label, vert_label) in enumerate(tqdm(test_dataloader)):
            # 前向传播
            # hori_output, vert_output = model(
            #     final_image_hori_a.to(device),
            #     final_image_hori_b.to(device)
            # )

            hori_output=hori_model(final_image_hori_a.to(device),final_image_hori_b.to(device))
            vert_output=vert_model(final_image_vert_a.to(device),final_image_vert_b.to(device))
            
            # 二分类：使用0.5作为阈值
            # hori_output和vert_output已经是sigmoid输出，在[0, 1]之间
            hori_pred = (hori_output > 0.5).float().cpu()
            vert_pred = (vert_output > 0.5).float().cpu()
            
            # 确保标签形状匹配
            hori_label = hori_label.float()
            vert_label = vert_label.float()
            
            # 计算正确率
            batch_size = final_image_hori_a.size(0)
            total_samples += batch_size
            
            hori_right += (hori_pred == hori_label).sum().item()
            vert_right += (vert_pred == vert_label).sum().item()
            
            # 每100个batch打印一次进度
            if (batch_num + 1) % 100 == 0:
                hori_acc = hori_right / total_samples
                vert_acc = vert_right / total_samples
                print(f"Batch {batch_num+1}/{len(test_dataloader)} - "
                      f"Hori Acc: {hori_acc:.4f}, Vert Acc: {vert_acc:.4f}")
        
        # 最终结果
        hori_accuracy = hori_right / total_samples
        vert_accuracy = vert_right / total_samples
        
        print(f"\n{'='*50}")
        print("Final Test Results:")
        print(f"{'='*50}")
        print(f"Total samples tested: {total_samples}")
        print(f"Horizontal accuracy: {hori_accuracy:.4f} ({hori_right}/{total_samples})")
        print(f"Vertical accuracy:   {vert_accuracy:.4f} ({vert_right}/{total_samples})")
        print(f"{'='*50}")
        
        return hori_accuracy, vert_accuracy
    
    
if __name__=="__main__":
    HORI_MODEL_NAME="model/hori_ef0.pth"
    VERT_MODEL_NAME="model/vert_ef0.pth"
    # MODEL_NAME="model/pairwise_pretrain.pth"
    train(100,load=True)
    hori_model.load_state_dict(torch.load(HORI_MODEL_NAME))
    vert_model.load_state_dict(torch.load(VERT_MODEL_NAME))
    test(hori_model,vert_model)
