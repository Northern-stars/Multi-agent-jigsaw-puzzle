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


DEVICE="cuda" if torch.cuda.is_available() else "cpu"
# MODEL_NAME="ef0_modulator"
MODEL_NAME="piece_style"

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


hori_model=pretrain_model(512,512,MODEL_NAME).to(DEVICE)
vert_model=pretrain_model(512,512,MODEL_NAME).to(DEVICE)

hori_model.load_state_dict(torch.load("model/hori_"+MODEL_NAME+".pth"))
vert_model.load_state_dict(torch.load("model/vert_"+MODEL_NAME+".pth"))

def make_score_matrix(img1_set:np.ndarray
                      ,img_lbl1:np.ndarray
                      ,img2_set:np.ndarray
                      ,img_lbl2:np.ndarray
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
        random_idx=random.randint(0,sample_num-1)
        while random_idx==index:
            random_idx=random.randint(0,sample_num-1)
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

from greedy import load_cate_matrix, greedy_test, score_check




if __name__ == "__main__":
    # cate_score = np.zeros([1000,18,9])
    
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
    acc_record=[]
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
        
        file_name = f"{img1_name}_{img2_name}_modulator_no_chunk"
        hori_score, vert_score = make_score_matrix(img1_set, img1_label, img2_set, img2_label, file_name)
        
        print(f"Testing {file_name}")
        acc,hori_acc,vert_acc,cate_acc,_,_=greedy_test(hori_score, vert_score)
        acc_record.append((acc,hori_acc,vert_acc,cate_acc))
        # score_check(hori_score, vert_score, cate_score)
        final_acc=0
    for i in range(len(combinations)):
        print(f"Acc of {combinations[i][0]} and {combinations[i][1]}: {acc_record[i][0]:.4f}, hori: {acc_record[i][1]:.4f}, vert: {acc_record[i][2]:.4f}, cate: {acc_record[i][3]:.4f}")
        final_acc+=acc_record[i][0]
    print(f"Final avg accuracy:{final_acc/len(combinations) :.4f}")
