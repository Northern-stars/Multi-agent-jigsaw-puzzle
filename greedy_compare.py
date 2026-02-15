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
# from pretrain import pretrain_model
from pretrain_1 import pretrain_model
from piece_compare import DeepuzzleModel_pieceStyle as compare_model
import cv2

from greedy import load_cate_matrix, greedy_test, score_check

MODEL1="piece_style"
MODEL2="ef0_modulator"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

hori_model1=pretrain_model(512,512,MODEL1).to(DEVICE)
hori_model2=pretrain_model(512,512,MODEL2).to(DEVICE)

vert_model1=pretrain_model(512,512,MODEL1).to(DEVICE)
vert_model2=pretrain_model(512,512,MODEL2).to(DEVICE)

test_x_path = 'dataset/test_img_48gap_33.npy'
test_y_path = 'dataset/test_label_48gap_33.npy'
test_x=np.load(test_x_path)
test_y=np.load(test_y_path)

artifact_img=test_x[1334:,:,:]
artifact_label=test_y[1334:,:,:]
painting_img=test_x[0:667,:,:]
painting_label=test_y[0:667,:,:]
engraving_img=test_x[667:1334,:,:]
engraving_label=test_y[667:1334,:,:]



def make_score_matrix(img1_set:np.ndarray
                      ,img_lbl1:np.ndarray
                      ,img2_set:np.ndarray
                      ,img_lbl2:np.ndarray
                      ,hori_model:nn.Module,
                      vert_model:nn.Module
                      ,file_name:str):
    
    img1_set_=copy.deepcopy(img1_set)
    img2_set_=copy.deepcopy(img2_set)
    img_lbl1_=copy.deepcopy(img_lbl1)
    img_lbl2_=copy.deepcopy(img_lbl2)

    hori_path=os.path.join("dataset","hori_score_"+file_name+".npy")
    vert_path=os.path.join("dataset","vert_score_"+file_name+".npy")
    if os.path.exists(hori_path):
        print(f"loading {file_name} from existing file")
        hori_score_matrix=np.load(hori_path)
        vert_score_matrix=np.load(vert_path)
        return hori_score_matrix,vert_score_matrix
    print(f"Start calculating: {file_name}")
    hori_model.eval()
    vert_model.eval()
    sample_num=min([img1_set_.shape[0],img2_set_.shape[0]])

    hori_score_matrix=np.ones([sample_num,18,18],dtype=float)
    vert_score_matrix=np.ones([sample_num,18,18],dtype=float)

    for index in range(sample_num):
        print(f"\rCalculating: {file_name} {(index/sample_num) * 100 : .2f}%",end="")
        img1=torch.tensor(img1_set_[index]).to(torch.float).permute([2,0,1])
        random_idx=sample_num-index-1
        while random_idx==index:
            random_idx-=1
        img2=torch.tensor(img2_set_[random_idx]).to(torch.float).permute([2,0,1])


        image_label1=list(img_lbl1_[index])
        for a in range(8):
            for b in range(8):
                if image_label1[a][b]==True:
                    if b>=4:
                        image_label1[a]=b+1
                        break
                    else:
                        image_label1[a]=b
                        break
                        
        image_label2=list(img_lbl2_[random_idx])
        for a in range(8):
            for b in range(8):
                if image_label2[a][b]==True:
                    if b>=4:
                        image_label2[a]=b+1
                        break
                    else:
                        image_label2[a]=b
                        break
        
        image_label1.insert(4,4)
        image_label2.insert(4,4)
        for i in range(len(image_label2)):
            image_label2[i]+=9

        image_label=image_label1+image_label2
        image_fragments=[
                img1[:,0:96,0:96],
                img1[:,0:96,96:192],
                img1[:,0:96,192:288],
                img1[:,96:192,0:96],
                img1[:,96:192,96:192],
                img1[:,96:192,192:288],
                img1[:,192:288,0:96],
                img1[:,192:288,96:192],
                img1[:,192:288,192:288],
                img2[:,0:96,0:96],
                img2[:,0:96,96:192],
                img2[:,0:96,192:288],
                img2[:,96:192,0:96],
                img2[:,96:192,96:192],
                img2[:,96:192,192:288],
                img2[:,192:288,0:96],
                img2[:,192:288,96:192],
                img2[:,192:288,192:288]
            ]

        # Collect all pairs
        pairs = []
        for i0 in range(18):
            for i1 in range(18):
                if i0 == i1:
                    continue
                pairs.append((i0, i1, image_fragments[i0], image_fragments[i1]))
        
        # Batch processing
        batch_size = 128
        for start in range(0, len(pairs), batch_size):
            if start+batch_size>=len(pairs):
                batch=pairs[start:]
            else:
                batch = pairs[start:start + batch_size]
            frag0_batch = torch.stack([p[2] for p in batch]).to(DEVICE)
            frag1_batch = torch.stack([p[3] for p in batch]).to(DEVICE)
            
            with torch.no_grad():
                hori_score0_batch = hori_model(frag0_batch, frag1_batch)
                hori_score1_batch = hori_model(frag1_batch, frag0_batch)
                
                vert_score0_batch = vert_model(frag0_batch, frag1_batch)
                vert_score1_batch = vert_model(frag1_batch, frag0_batch)
            
            for idx, (i0, i1, _, _) in enumerate(batch):
                hori_score_matrix[index][image_label[i0]][image_label[i1]] = float(hori_score0_batch[idx].detach())
                hori_score_matrix[index][image_label[i1]][image_label[i0]] = float(hori_score1_batch[idx].detach())
                
                vert_score_matrix[index][image_label[i0]][image_label[i1]] = float(vert_score0_batch[idx].detach())
                vert_score_matrix[index][image_label[i1]][image_label[i0]] = float(vert_score1_batch[idx].detach())
    np.save("dataset/hori_score_"+file_name,hori_score_matrix)
    np.save("dataset/vert_score_"+file_name,vert_score_matrix)
    return hori_score_matrix,vert_score_matrix

def get_image(img1:np.ndarray,img2:np.ndarray,image_label1:np.ndarray,image_label2:np.ndarray,perm:list):
    '''img: np.array[288,288,3]
        label: np.array[8,8]
        perm: list [9]
       '''
    image_fragments=[
                img1[0:96,0:96,:],
                img1[0:96,96:192,:],
                img1[0:96,192:288,:],
                img1[96:192,0:96,:],
                img1[96:192,96:192,:],
                img1[96:192,192:288,:],
                img1[192:288,0:96,:],
                img1[192:288,96:192,:],
                img1[192:288,192:288,:],
                img2[0:96,0:96,:],
                img2[0:96,96:192,:],
                img2[0:96,192:288,:],
                img2[96:192,0:96,:],
                img2[96:192,96:192,:],
                img2[96:192,192:288,:],
                img2[192:288,0:96,:],
                img2[192:288,96:192,:],
                img2[192:288,192:288,:]
            ]
    image_label1=list(image_label1)
    image_label2=list(image_label2)
    for a in range(8):
        for b in range(8):
            if image_label1[a][b]==True:
                if b>=4:
                    image_label1[a]=b+1
                    break
                else:
                    image_label1[a]=b
                    break
    for a in range(8):
        for b in range(8):
            if image_label2[a][b]==True:
                if b>=4:
                    image_label2[a]=b+1
                    break
                else:
                    image_label2[a]=b
                    break
    
    image_label1.insert(4,4)
    image_label2.insert(4,4)
    for i in range(len(image_label2)):
        image_label2[i]+=9

    image_label=image_label1+image_label2

    reassembled_img=np.zeros([288,288,3],dtype=np.uint8)
    for i in range(len(perm)):
        reassembled_img[(0+i//3)*96:(1+i//3)*96,(0+i%3)*96:(1+i%3)*96,:]=image_fragments[image_label.index(perm[i])]
    return reassembled_img

def display_img(img1,img2,right1,right2):
    cv2.imshow(f"img1: {right1}",img1)
    cv2.imshow(f"img2 {right2}",img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def result_compare(img_set1,label_set1,img_set2,label_set2,file_name):
    sample_num=min([img_set1.shape[0],img_set2.shape[0]])
    hori_score1,vert_score1=make_score_matrix(img_set1,label_set1,img_set2,label_set2,hori_model1,vert_model1,file_name+"_1")
    hori_score2,vert_score2=make_score_matrix(img_set1,label_set1,img_set2,label_set2,hori_model2,vert_model2,file_name+"_2")

    acc1,hori_acc1,vert_acc1,cate_acc1,right_list_1,perm1=greedy_test(hori_score1,vert_score1)
    acc2,hori_acc2,vert_acc2,cate_acc2,right_list_2,perm2=greedy_test(hori_score2,vert_score2)
    pictures=0
    for i in range(len(right_list_1)):
        if pictures>=0:
            break
        
        # if right_list_1[i]!=right_list_2[i]:
        if right_list_1[i]==0 or right_list_2[i]==0:
            pictures+=1
            current_img=img_set1[i]
            current_label=label_set1[i]
            outsider_idx=sample_num-i-1 if sample_num-i-1!=i else sample_num-i-2
            outsider_img=img_set2[outsider_idx]
            outsider_label=label_set2[outsider_idx]

            img1=get_image(current_img,outsider_img,current_label,outsider_label,perm1[i])
            img2=get_image(current_img,outsider_img,current_label,outsider_label,perm2[i])

            display_img(img1,img2,right_list_1[i],right_list_2[i])
        
    return acc1,acc2




if __name__=="__main__":
    hori_model1.load_state_dict(torch.load("model/hori_"+MODEL1+".pth"))
    hori_model2.load_state_dict(torch.load("model/hori_"+MODEL2+".pth"))

    vert_model1.load_state_dict(torch.load("model/vert_"+MODEL1+".pth"))
    vert_model2.load_state_dict(torch.load("model/vert_"+MODEL2+".pth"))

    combinations = [
        ("artifact", "artifact"),
        ("artifact", "painting"),
        ("artifact", "engraving"),
        ("painting", "artifact"),
        ("painting", "painting"),
        ("painting", "engraving"),
        ("engraving", "artifact"),
        ("engraving", "painting"),
        ("engraving", "engraving"),
    ]
    acc_record1=[]
    acc_record2=[]
    for img1_name, img2_name in combinations:
        if img1_name == "artifact":
            img1_set = artifact_img
            img1_label = artifact_label
        elif img1_name == "painting":
            img1_set = painting_img
            img1_label = painting_label
        elif img1_name == "engraving":
            img1_set = engraving_img
            img1_label = engraving_label
        
        if img2_name == "artifact":
            img2_set = artifact_img
            img2_label = artifact_label
        elif img2_name == "painting":
            img2_set = painting_img
            img2_label = painting_label
        elif img2_name == "engraving":
            img2_set = engraving_img
            img2_label = engraving_label

        file_name=f"{img1_name}_{img2_name}"
        acc1,acc2=result_compare(img1_set,img1_label,img2_set,img2_label,file_name)
        acc_record1.append(acc1)
        acc_record2.append(acc2)
    print(f"Final result compare: acc 1 :{np.mean(acc_record1)}, acc 2 :{np.mean(acc_record2)}")