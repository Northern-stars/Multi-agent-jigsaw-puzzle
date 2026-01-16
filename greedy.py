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
from piece_compare import piece_compare_model_b3 as compare_model

test_x_path = 'dataset/test_img_48gap_33.npy'
test_y_path = 'dataset/test_label_48gap_33.npy'

test_x=np.load(test_x_path)
test_y=np.load(test_y_path)

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
HORI_SCORE_NAME="dataset/hori_score_single_model_no_random.npy"
VERT_SCORE_NAME="dataset/vert_score_single_model_no_random.npy"
CATE_SCORE_NAME="dataset/cate_score.npy"

# model=pretrain_model(256,256).to(DEVICE)
hori_model=pretrain_model(512,512).to(DEVICE)
vert_model=pretrain_model(512,512).to(DEVICE)
cate_model=compare_model(512,512,8).to(DEVICE)

# model.load_state_dict(torch.load("model/pairwise_pretrain.pth"))

hori_model.load_state_dict(torch.load("model/hori_ef0.pth"))
vert_model.load_state_dict(torch.load("model/vert_ef0.pth"))
cate_model.load_state_dict(torch.load("model/deepuzzle8.pth"))


def load_score_matrix():
    if os.path.exists(HORI_SCORE_NAME):
        print("loading from existing file")
        hori_score_matrix=np.load(HORI_SCORE_NAME)
        vert_score_matrix=np.load(VERT_SCORE_NAME)
    else:
        print("start calculating score matrix")
        hori_model.eval()
        vert_model.eval()
        hori_score_matrix=np.ones([1000,18,18],dtype=float)
        vert_score_matrix=np.ones([1000,18,18],dtype=float)
        for index in range(test_x.shape[0]//2):
            if index>=666:
                i=index+667
            elif index>=333:
                i=index+334
            else:
                i=index
            print(f"Calculating image: {index}")
            random_index=(i+666)%test_x.shape[0]
            image0, image1=torch.tensor(test_x[i]).permute([2,0,1]).to(torch.float), torch.tensor(test_x[random_index]).permute([2,0,1]).to(torch.float)
            image_label0=list(test_y[i])
            for a in range(8):
                for b in range(8):
                # print(type(imageLabel[i]))
                # print(imageLabel)
                    if image_label0[a][b]==True:
                        if b>=4:
                            image_label0[a]=b+1
                            break
                        else:
                            image_label0[a]=b
                            break
            image_label0.insert(4,4)
            image_label1=list(test_y[random_index])
            for a in range(8):
                for b in range(8):
                # print(type(imageLabel[i]))
                # print(imageLabel)
                    if image_label1[a][b]==True:
                        if b>=4:
                            image_label1[a]=b+1
                            break
                        else:
                            image_label1[a]=b
                            break
            image_label1.insert(4,4)
            for j in range(len(image_label1)):
                image_label1[j]+=9
            image_label=image_label0+image_label1
            image_fragments=[
                image0[:,0:96,0:96],
                image0[:,0:96,96:192],
                image0[:,0:96,192:288],
                image0[:,96:192,0:96],
                image0[:,96:192,96:192],
                image0[:,96:192,192:288],
                image0[:,192:288,0:96],
                image0[:,192:288,96:192],
                image0[:,192:288,192:288],
                image1[:,0:96,0:96],
                image1[:,0:96,96:192],
                image1[:,0:96,192:288],
                image1[:,96:192,0:96],
                image1[:,96:192,96:192],
                image1[:,96:192,192:288],
                image1[:,192:288,0:96],
                image1[:,192:288,96:192],
                image1[:,192:288,192:288]
            ]

            for i0 in range(0,18):
                for i1 in range(0,18):
                    if i0==i1:
                        continue
                    image_frag0=image_fragments[i0].unsqueeze(0).to(DEVICE)
                    image_frag1=image_fragments[i1].unsqueeze(0).to(DEVICE)
                    hori_score0=hori_model(image_frag0,image_frag1)
                    hori_score1=hori_model(image_frag1,image_frag0)

                    vert_score0=vert_model(image_frag0,image_frag1)
                    vert_score1=vert_model(image_frag1,image_frag0)

                    hori_score_matrix[index][image_label[i0]][image_label[i1]]=float(hori_score0.detach())
                    hori_score_matrix[index][image_label[i1]][image_label[i0]]=float(hori_score1.detach())

                    vert_score_matrix[index][image_label[i0]][image_label[i1]]=float(vert_score0.detach())
                    vert_score_matrix[index][image_label[i1]][image_label[i0]]=float(vert_score1.detach())

    np.save(HORI_SCORE_NAME,hori_score_matrix)
    np.save(VERT_SCORE_NAME,vert_score_matrix)
    return hori_score_matrix,vert_score_matrix

def load_cate_matrix():
    if os.path.exists(CATE_SCORE_NAME):
        print("loading cate matrix from existing file")
        cate_matrix=np.load(CATE_SCORE_NAME)
        return cate_matrix
    else:
        print("start calculating cate matrix")
        cate_model.eval()
        cate_matrix=torch.zeros([1000,18,8])
        for index in range(test_x.shape[0]//2):
            if index>=666:
                i=index+667
            elif index>=333:
                i=index+334
            else:
                i=index
            print(f"Calculating image: {index}")
            random_index=(i+666)%test_x.shape[0]
            image0, image1=torch.tensor(test_x[i]).permute([2,0,1]).to(torch.float), torch.tensor(test_x[random_index]).permute([2,0,1]).to(torch.float)
            image_label0=list(test_y[i])
            for a in range(8):
                for b in range(8):
                    if image_label0[a][b]==True:
                        if b>=4:
                            image_label0[a]=b+1
                            break
                        else:
                            image_label0[a]=b
                            break
            image_label0.insert(4,4)
            image_label1=list(test_y[random_index])
            for a in range(8):
                for b in range(8):
                # print(type(imageLabel[i]))
                # print(imageLabel)
                    if image_label1[a][b]==True:
                        if b>=4:
                            image_label1[a]=b+1
                            break
                        else:
                            image_label1[a]=b
                            break
            image_label1.insert(4,4)
            for j in range(len(image_label1)):
                image_label1[j]+=9
            image_label=image_label0+image_label1
            image_fragments=[
                image0[:,0:96,0:96],
                image0[:,0:96,96:192],
                image0[:,0:96,192:288],
                image0[:,96:192,0:96],
                image0[:,96:192,96:192],
                image0[:,96:192,192:288],
                image0[:,192:288,0:96],
                image0[:,192:288,96:192],
                image0[:,192:288,192:288],
                image1[:,0:96,0:96],
                image1[:,0:96,96:192],
                image1[:,0:96,192:288],
                image1[:,96:192,0:96],
                image1[:,96:192,96:192],
                image1[:,96:192,192:288],
                image1[:,192:288,0:96],
                image1[:,192:288,96:192],
                image1[:,192:288,192:288]
            ]
            central_piece=image_fragments[4].unsqueeze(0).to(DEVICE)

            for i0 in range(0,18):
                if i0==4 or i0==13:
                    continue
                with torch.no_grad():
                    score_tensor=cate_model(central_piece,image_fragments[i0].unsqueeze(0).to(DEVICE))
                cate_matrix[index,image_label[i0]]=copy.deepcopy(score_tensor.cpu())
        
        cate_matrix=cate_matrix.numpy()
        np.save(CATE_SCORE_NAME,cate_matrix)

        return cate_matrix

def get_cate_certainty(cate_matrix,img_id,position,piece_id):
    if position==4 | piece_id==4 | piece_id==13:
        return 1
    if position>4:
        return cate_matrix[img_id,piece_id,position-1]
    else:
        return cate_matrix[img_id,piece_id,position]


def get_score(img_id,permutation,hori_score, vert_score,cate_score):
    permutation_=copy.deepcopy(permutation)
    # for j in range(len(permutation)):
    #     if (permutation_[j]>=4):permutation_[j]+=1
    #     elif permutation_[j]>=13: permutation_[j]+=2
    # permutation_.insert(4,4)
    # print(f"Permutation: {permutation_}")
    hori_set=[(i,i+1) for i in [j for j in range(9) if j%3!=3-1 ]]
    vert_set=[(i,i+3) for i in range(3*2)]
    support_list=[1,3,5,7]
    final_score=0
    for i in range(len(hori_set)):
        i0,i1=permutation_[hori_set[i][0]],permutation_[hori_set[i][1]]
        if i0!=-1 & i1!=-1:
            score=hori_score[img_id][i0][i1]

            if 4 in hori_set[i]:#certainty check
                final_score+=score
            else:
                if hori_set[i][0] in support_list:
                    final_score+=get_cate_certainty(cate_score,img_id,hori_set[i][0],i0)*score
                elif hori_set[i][1] in support_list:
                    final_score+=get_cate_certainty(cate_score,img_id,hori_set[i][1],i1)*score
            
            # final_score+=score

    for i in range(len(vert_set)):
        i0,i1=permutation_[vert_set[i][0]],permutation_[vert_set[i][1]]
        if i0!=-1 & i1!=-1:
            score=vert_score[img_id][i0][i1]

            if 4 in vert_set[i]:#certainty check
                final_score+=score
            else:
                if vert_set[i][0] in support_list:
                    final_score+=get_cate_certainty(cate_score,img_id,vert_set[i][0],i0)*score
                elif vert_set[i][1] in support_list:
                    final_score+=get_cate_certainty(cate_score,img_id,vert_set[i][1],i1)*score


            # final_score+=score
    
    
    return final_score


def score_check(hori_score_matrix, vert_score_matrix,cate_score):
    hori_set=[(i,i+1) for i in [j for j in range(9) if j%3!=3-1 ]]
    vert_set=[(i,i+3) for i in range(3*2)]

    hori_acc=[]
    vert_acc=[]
    cate_acc=[]

    for img_idx in range(hori_score_matrix.shape[0]):

        for i in range(9):
            for j in range(9):
                hori_score=float(hori_score_matrix[img_idx,i,j])
                vert_score=float(vert_score_matrix[img_idx,i,j])

                if (hori_score>0.5)==((i,j) in hori_set):
                    hori_acc.append(1)
                else:
                    hori_acc.append(0)
                
                if (vert_score>0.5)==((i,j) in vert_set):
                    vert_acc.append(1)
                else:
                    vert_acc.append(0)
            if i==4:
                continue
            else:
                # print(cate_score[img_idx,i].shape)
                if i<4:
                    if (int(np.argmax(cate_score[img_idx,i]))==i):
                        cate_acc.append(1)
                    else:
                        cate_acc.append(0)
                else:
                    if (int(np.argmax(cate_score[img_idx,i]))==i-1):
                        cate_acc.append(1)
                    else:
                        cate_acc.append(0)
    print(f"Score matrix check: hori acc: {np.mean(hori_acc)}, vert acc: {np.mean(vert_acc)}, cate acc: {np.mean(cate_acc)}")


def generate_permutations():
    if os.path.exists("dataset/possible_permutations.npy"):
        print("Loading permutation from existing file")
        permutations=list(np.load("dataset/possible_permutations.npy"))
    else:
        print("Generating permutations")
        permutations=list(itertools.permutations(range(16),8))
        np.save("dataset/possible_permutations.npy",np.array(permutations))
    return permutations
        

class Node:
    def __init__(self,position,parent=None):
        self.left=None
        self.right=None
        self.up=None
        self.down=None
        self.absolute_position=position
        self.piece=None
        self.parent=parent

        if position//3==0:
            self.up=-1
        if position//3==2:
            self.down=-1
        if position%3==0:
            self.left=-1
        if position%3==2:
            self.right=-1
        
        if parent!=None:
            position=self.absolute_position-self.parent.absolute_position
            if position==-1: #left
                self.right=-2
            elif position==1: #right
                self.left=-2
            elif position==-3: #up
                self.down=-2
            elif position==3: #down
                self.up=-2

    
    def generate_candidate(self):
        candidate=[]
        if self.left==None:
            candidate.append(Node(self.calculate_candidate("left"),parent=self))
        if self.right==None:
            candidate.append(Node(self.calculate_candidate("right"),parent=self))
        if self.up==None:
            candidate.append(Node(self.calculate_candidate("up"),parent=self))
        if self.down==None:
            candidate.append(Node(self.calculate_candidate("down"),parent=self))
        return candidate
    

    def calculate_candidate(self,position):
        if position=="left":
            return self.absolute_position-1
        elif position=="right":
            return self.absolute_position+1
        elif position=="up":
            return self.absolute_position-3
        elif position=="down":
            return self.absolute_position+3
        

    def set_parent(self):
        if self.left==-2:
            self.parent.right=-2
        elif self.right==-2:
            self.parent.left=-2
        elif self.up==-2:
            self.parent.down=-2
        elif self.down==-2:
            self.parent.up=-2


def get_candidate(node_list):
    candidate_list=[]
    for node in node_list:
        node_candidate=node.generate_candidate()
        for new_candidate in node_candidate:
            # flag=True
            # for candidate in candidate_list:
            #     if new_candidate.absolute_position==candidate.absolute_position:
            #         flag=False
            #         break
            # if flag:
                candidate_list.append(new_candidate)
    return candidate_list

def get_best_candidate(img_idx,cur_node:Node,hori_score,vert_score,cate_score,node_list):
    node_list_=copy.deepcopy(node_list)
    used_list=get_used_list(node_list)
    parent_piece=cur_node.parent.piece
    position=cur_node.absolute_position-cur_node.parent.absolute_position
    if position==-1: #left
        # score_matrix=hori_score[img_idx,:,parent_piece]
        score_matrix=hori_score[img_idx,0:9,parent_piece]
        best_piece_list=np.argsort(score_matrix)[::-1]
        best_piece=int(best_piece_list[0])
        i=0
        while best_piece in used_list:
            i+=1
            best_piece=int(best_piece_list[i])
        cur_node.piece=best_piece
        node_list_.append(cur_node)
        score=get_score(img_idx,get_permutation(node_list_),hori_score,vert_score,cate_score)-get_score(img_idx,get_permutation(node_list),hori_score,vert_score,cate_score)
        # score=hori_score[img_idx,best_piece,parent_piece]
    elif position==1: #right
        # score_matrix=hori_score[img_idx,parent_piece,:]
        score_matrix=hori_score[img_idx,parent_piece,0:9]
        best_piece_list=np.argsort(score_matrix)[::-1]
        best_piece=int(best_piece_list[0])
        i=0
        while best_piece in used_list:
            i+=1
            best_piece=int(best_piece_list[i])
        cur_node.piece=best_piece
        node_list_.append(cur_node)
        score=get_score(img_idx,get_permutation(node_list_),hori_score,vert_score,cate_score)-get_score(img_idx,get_permutation(node_list),hori_score,vert_score,cate_score)
        # score=hori_score[img_idx,parent_piece,best_piece]
    elif position==-3: #up
        # score_matrix=vert_score[img_idx,:,parent_piece]
        score_matrix=vert_score[img_idx,0:9,parent_piece]
        best_piece_list=np.argsort(score_matrix)[::-1]
        best_piece=int(best_piece_list[0])
        i=0
        while best_piece in used_list:
            i+=1
            best_piece=int(best_piece_list[i])
        cur_node.piece=best_piece
        node_list_.append(cur_node)
        score=get_score(img_idx,get_permutation(node_list_),hori_score,vert_score,cate_score)-get_score(img_idx,get_permutation(node_list),hori_score,vert_score,cate_score)
        # score=vert_score[img_idx,best_piece,parent_piece]
    elif position==3: #down
        # score_matrix=vert_score[img_idx,parent_piece,:]
        score_matrix=vert_score[img_idx,parent_piece,0:9]
        best_piece_list=np.argsort(score_matrix)[::-1]        
        best_piece=int(best_piece_list[0])
        i=0
        while best_piece in used_list:
            i+=1
            best_piece=int(best_piece_list[i])

        cur_node.piece=best_piece
        node_list_.append(cur_node)
        score=get_score(img_idx,get_permutation(node_list_),hori_score,vert_score,cate_score)-get_score(img_idx,get_permutation(node_list),hori_score,vert_score,cate_score)
        # score=vert_score[img_idx,parent_piece,best_piece]
    
    return cur_node,float(score)

def check_valid(node_list):
    absolute_list=get_absolute_list(node_list)
    for cur_node in node_list:
        

        left=cur_node.calculate_candidate("left")
        right=cur_node.calculate_candidate("right")
        up=cur_node.calculate_candidate("up")
        down=cur_node.calculate_candidate("down")

        if (left in absolute_list) & (cur_node.left==None):
                cur_node.left=-1
        if (right in absolute_list) & (cur_node.right==None):
                cur_node.right=-1
        if (up in absolute_list) & (cur_node.up==None): 
                cur_node.up=-1
        if (down in absolute_list) & (cur_node.down==None):
                cur_node.down=-1
            

def get_used_list(node_list):
    used_list=[]
    for node in node_list:
        used_list.append(node.piece)
    return used_list

def get_absolute_list(node_list):
    absolute_list=[]
    for node in node_list:
        absolute_list.append(node.absolute_position)
    return absolute_list

def get_permutation(node_list):
    permutation=[-1 for _ in range(9)]
    for node in node_list:
        permutation[node.absolute_position]=node.piece
    
    return permutation

def generating_spanning_tree(img_idx,hori_score,vert_score,cate_score):
    node_list=[Node(4,None)]
    node_list[0].piece=4
    step=0
    while True:
        step+=1
        check_valid(node_list)
        current_candidate_list=get_candidate(node_list)
        candidate_position=get_absolute_list(current_candidate_list)
        # print(f"Step: {step},candidate_position: {candidate_position},node: {get_absolute_list(node_list)}")
        # for node in node_list:
        #     print(f"Absolute position: {node.absolute_position}, left: {node.left}, right: {node.right}, up: {node.up}, down: {node.down}")
        
        if len(node_list)==9:
            break
        # if len(current_candidate_list)==0:
        #     break
        best_node=None
        best_score=0

        for candidate in current_candidate_list:
            candidate,score=get_best_candidate(img_idx,candidate,hori_score,vert_score,cate_score,node_list)
            if score>best_score or best_node is None:
                best_score=score
                best_node=candidate

        best_node.set_parent()
        node_list.append(best_node)

        
    
    permutation=get_permutation(node_list)
    
    return permutation

def get_local_accuracy(permutation):
    hori_set=[(i,i+1) for i in [j for j in range(9) if j%3!=3-1 ]]
    vert_set=[(i,i+3) for i in range(3*2)]
    hori_right=0
    vert_right=0
    cate_right=0
    for i in range(len(hori_set)):
        i0,i1=permutation[hori_set[i][0]],permutation[hori_set[i][1]]
        if (i0,i1) in hori_set:
            hori_right+=1
    for i in range(len(vert_set)):
        i0,i1=permutation[vert_set[i][0]],permutation[vert_set[i][1]]
        if (i0,i1) in vert_set:
            vert_right+=1
    for i in range(len(permutation)):
        if permutation[i]==i:
            cate_right+=1
    return hori_right/len(hori_set), vert_right/len(vert_set), cate_right/len(permutation)



def greedy_test(hori_score, vert_score,cate_score):
    print("Start testing")
    right=[]
    hori_acc=[]
    vert_acc=[]
    cate_acc=[]
    error_count=0
    greedy_right_rate=[]
    # possible_permutations=generate_permutations()
    
    for i in range(hori_score.shape[0]):
        print(f"Calculating id: {i}")
        # best_score=0
        # best_perm=[]
        # for permutation in possible_permutations:
        #     permutation=permutation.tolist()
        #     score=get_score(i,permutation,hori_score,vert_score,cate_score)
        #     if score>best_score:
        #         best_perm=permutation
        #         best_score=score
        # best_perm.insert(4,4)


        # for j in range(len(best_perm)):
        #     if (best_perm[j]>=4):best_perm[j]+=1
        #     elif best_perm[j]>=13: best_perm[j]+=2
        best_perm=generating_spanning_tree(i,hori_score,vert_score,cate_score)
        hori_right,vert_right,cate_right=get_local_accuracy(best_perm)
        hori_acc.append(hori_right)
        vert_acc.append(vert_right)
        cate_acc.append(cate_right)
        if best_perm==list(range(0,9)):
            right.append(1)
        elif -1 in best_perm:
            error_count+=1
        else:
            right.append(0)
        greedy_score=get_score(i,best_perm,hori_score,vert_score,cate_score)
        true_score=get_score(i,list(range(0,9)),hori_score,vert_score,cate_score)
        print(f"Accuracy: {np.mean(right)}, final permutation: {best_perm}")
        print(f"Local acc: hori: {np.mean(hori_acc)}, vert: {np.mean(vert_acc)}, cate: {np.mean(cate_acc)}")
        print(f"Greedy score: {greedy_score}, answer score: {true_score}")
        if greedy_score>=true_score:
            greedy_right_rate.append(1)
        else:
            greedy_right_rate.append(0)

    
    print(f"accuracy: {np.mean(right)}, error count: {error_count},  greedy better rate: {np.mean(greedy_right_rate)}")
    return

if __name__=="__main__":
    hori_score,vert_score=load_score_matrix()
    cate_score=load_cate_matrix()
    # print(f"Hori yes: {float(hori_score[1,1,2])}, Hori no: {float(hori_score[1,2,1])}")

    greedy_test(hori_score,vert_score,cate_score)
    score_check(hori_score,vert_score,cate_score)
