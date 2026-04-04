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
from torchvision.models import efficientnet_b0,efficientnet_b3
from torch.utils.data import Dataset, DataLoader
import model_code.Vit as Vit
from tqdm import tqdm
from pretrain.pretrain_1 import pretrain_model



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

FEN_NAME="modulator"
MODEL_NAME="model/pairwise_pretrain.pth"
EF_MODEL_NAME="model/pairwise_pretrain_{}.pth".format(FEN_NAME)
BATCH_SIZE=150
LOAD=False
LR=1e-4
TEST_PER_EPOCH=5

class imageData(Dataset):
    def __init__(self,data,label):
        self.data=data
        self.label=label
        self.choice=list(itertools.combinations(list(range(9)), 2))
        self.hori_set=[(i,i+1) for i in [j for j in range(9) if j%3!=3-1 ]]
        self.vert_set=[(i,i+3) for i in range(3*2)]
        self.sector_num=np.max([1,self.data.shape[0]//3])
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
        hori_label=0
        vert_label=0
        
        right=random.random()
        if right>=0.5:
            vert_label=1
            hori_label=1

            hori_idx=random.choice(self.hori_set)
            vert_idx=random.choice(self.vert_set)

            final_image_hori_a=image_fragments[imageLabel.index(hori_idx[0])]
            final_image_hori_b=image_fragments[imageLabel.index(hori_idx[1])]
            final_image_vert_a=image_fragments[imageLabel.index(vert_idx[0])]
            final_image_vert_b=image_fragments[imageLabel.index(vert_idx[1])]

        else:
            vert_label=0
            hori_label=0
            outsider_flag=random.random()
            if outsider_flag>0.5:
                guest_idx=random.randint(0,self.data.shape[0]-1)
                while index//self.sector_num==guest_idx//self.sector_num:
                    guest_idx=random.randint(0,self.data.shape[0]-1)
                guest_img=torch.tensor(self.data[guest_idx]).permute([2,0,1]).to(torch.float)
                guest_fragments=[guest_img[:,(i//3) * 96 : (i//3+ 1) * 96, (i%3) * 96 : (i%3 + 1) * 96] for i in range(9)]
                final_image_hori_a=random.choice(image_fragments)
                final_image_hori_b=random.choice(guest_fragments)
                final_image_vert_a=random.choice(image_fragments)
                final_image_vert_b=random.choice(guest_fragments)
            else:
                hori_idx=(random.randint(0,8),random.randint(0,8))
                while (hori_idx in self.hori_set) or (hori_idx[0]==hori_idx[1]):
                    hori_idx=(random.randint(0,8),random.randint(0,8))
                vert_idx=(random.randint(0,8),random.randint(0,8))
                while (vert_idx in self.vert_set) or (vert_idx[0]==vert_idx[1]):
                    vert_idx=(random.randint(0,8),random.randint(0,8))

                final_image_hori_a=image_fragments[imageLabel.index(hori_idx[0])]
                final_image_hori_b=image_fragments[imageLabel.index(hori_idx[1])]
                final_image_vert_a=image_fragments[imageLabel.index(vert_idx[0])]
                final_image_vert_b=image_fragments[imageLabel.index(vert_idx[1])]
        
        
        return final_image_hori_a,final_image_hori_b,final_image_vert_a,final_image_vert_b,torch.tensor(hori_label,dtype=torch.float).unsqueeze(0),torch.tensor(vert_label,dtype=torch.float).unsqueeze(0)


train_data=imageData(train_x,train_y)
train_dataloader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
test_data=imageData(test_x,test_y)
test_dataloader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True)

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






model=pretrain_model(512,512,FEN_NAME).to(device)
# hori_model=pretrain_model(256,256).to(device)
# vert_model=pretrain_model(256,256).to(device)



loss_fn=nn.BCELoss()
# hori_optimizer=torch.optim.Adam(hori_model.parameters(),lr=1e-4,eps=1e-8)
# vert_optimizer=torch.optim.Adam(vert_model.parameters(),lr=1e-4,eps=1e-8)
optimizer=torch.optim.Adam(model.parameters(),lr=LR,eps=1e-8)


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
        model.load_state_dict(torch.load(MODEL_NAME))
        # hori_model.load_state_dict(torch.load(HORI_MODEL_NAME))
        # vert_model.load_state_dict(torch.load(VERT_MODEL_NAME))
    print("start training")
    best_acc=0
    model.train()
    # hori_model.train()
    # vert_model.train()
    for epoch in range(epoch_num):
        hori_loss_sum=0
        vert_loss_sum=0
        for final_image_hori_a,final_image_hori_b,final_image_vert_a,final_image_vert_b,hori_label,vert_label in tqdm(train_dataloader):
            # hori_loss_sum+=single_train_hori(model=model,optimizer=optimizer,data=[final_image_hori_a,final_image_hori_b],label=hori_label)
            # vert_loss_sum+=single_train_vert(model=model,optimizer=optimizer,data=[final_image_vert_a,final_image_vert_b],label=vert_label)
            hori_loss_sum+=single_train(model,optimizer,(final_image_hori_a,final_image_hori_b),hori_label)
            vert_loss_sum+=single_train(model,optimizer,(final_image_vert_a,final_image_vert_b),vert_label)

        print(f"epochnum: {epoch}, hori_loss_sum: {hori_loss_sum*BATCH_SIZE}, vert_loss_sum: {vert_loss_sum*BATCH_SIZE}")
        if epoch%TEST_PER_EPOCH==0:
            test_hori_acc,test_vert_acc=test(model)
            if test_hori_acc+test_vert_acc>best_acc:
                torch.save(model.state_dict(),MODEL_NAME)
                torch.save(model.ef.state_dict(),EF_MODEL_NAME)
        # torch.save(hori_model.state_dict(),HORI_MODEL_NAME)
        # torch.save(vert_model.state_dict(),VERT_MODEL_NAME)



# def test():
#     with torch.no_grad():
#         model.load_state_dict(torch.load(MODEL_NAME))
#         model.eval()
#         # hori_model.load_state_dict(torch.load(HORI_MODEL_NAME))
#         # vert_model.load_state_dict(torch.load(VERT_MODEL_NAME))
#         # hori_model.eval()
#         # vert_model.eval()
#         hori_right=0
#         vert_right=0
#         print("start testing")
#         for batch_num,(final_image_hori_a,final_image_hori_b,final_image_vert_a,final_image_vert_b,hori_label,vert_label) in enumerate(test_dataloader):
#             # hori_output=torch.argmax(hori_model(final_image_hori_a.to(device),final_image_hori_b.to(device))).item()
#             # vert_output=torch.argmax(vert_model(final_image_vert_a.to(device),final_image_vert_b.to(device))).item()
#             hori_output,_=model(final_image_hori_a.to(device),final_image_hori_b.to(device))
#             _,vert_output=model(final_image_vert_a.to(device),final_image_vert_b.to(device))
#             # hori_output=torch.argmax(hori_output).item()
#             # vert_output=torch.argmax(vert_output).item()
#             hori_output=torch
#             hori_right+=(hori_output==hori_label.item())
#             vert_right+=(vert_output==vert_label.item())
#             if (batch_num+1)%100==0:
#                 print(f"batch num: {batch_num+1}, hori right level: {hori_right/(batch_num+1)}, vert right level: {vert_right/(batch_num+1)}")
        
#     print(f"hori right level: {hori_right/len(test_dataloader)}, vert right level: {vert_right/len(test_dataloader)}")

def test(model):
    """二分类测试函数 - 批处理并行版本"""
    with torch.no_grad():
        # 加载模型
        
        print("start testing in parallel mode...")
        print(f"Total batches: {len(test_dataloader)}")
        
        # 预分配列表存储所有结果
        all_hori_preds = []
        all_vert_preds = []
        all_hori_labels = []
        all_vert_labels = []
        
        # 收集所有batch的数据
        for batch_data in tqdm(test_dataloader, desc="Processing batches"):
            final_image_hori_a, final_image_hori_b, final_image_vert_a, final_image_vert_b, hori_label, vert_label = batch_data
            
            # 将所有数据移到GPU（一次移动，避免重复移动）
            hori_a = final_image_hori_a.to(device)
            hori_b = final_image_hori_b.to(device)
            vert_a = final_image_vert_a.to(device)
            vert_b = final_image_vert_b.to(device)
            
            # 前向传播 - 批处理
            hori_output = model(hori_a, hori_b)
            vert_output=model(vert_a,vert_b)
            # 二分类预测
            hori_pred = (hori_output > 0.5).float().cpu()
            vert_pred = (vert_output > 0.5).float().cpu()
            
            # 收集结果
            all_hori_preds.append(hori_pred)
            all_vert_preds.append(vert_pred)
            all_hori_labels.append(hori_label.float())
            all_vert_labels.append(vert_label.float())
        
        # 拼接所有batch的结果
        all_hori_preds = torch.cat(all_hori_preds, dim=0)
        all_vert_preds = torch.cat(all_vert_preds, dim=0)
        all_hori_labels = torch.cat(all_hori_labels, dim=0)
        all_vert_labels = torch.cat(all_vert_labels, dim=0)
        
        # 一次性计算所有指标
        total_samples = len(all_hori_labels)
        hori_correct = (all_hori_preds == all_hori_labels).sum().item()
        vert_correct = (all_vert_preds == all_vert_labels).sum().item()
        
        # 计算准确率
        hori_accuracy = hori_correct / total_samples
        vert_accuracy = vert_correct / total_samples
        
        # 打印结果
        print(f"\n{'='*50}")
        print("Final Test Results (Parallel Mode):")
        print(f"{'='*50}")
        print(f"Total samples tested: {total_samples}")
        print(f"Horizontal accuracy: {hori_accuracy:.4f} ({hori_correct}/{total_samples})")
        print(f"Vertical accuracy:   {vert_accuracy:.4f} ({vert_correct}/{total_samples})")

        return hori_accuracy,vert_accuracy


if __name__=="__main__":
    # HORI_MODEL_NAME="hori_ef0.pth"
    # VERT_MODEL_NAME="vert_ef0.pth"
    
    train(100,load=LOAD)
    model.load_state_dict(torch.load(MODEL_NAME))
    test(model)
