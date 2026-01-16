import random
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
import cv2
import itertools


class Env:
    def __init__(self,
                 train_x,
                 train_y,
                 gamma,
                 image_num,
                 buffer_size,
                 epsilon,
                 epsilon_gamma,
                 piece_num=9,
                 epochs=10,
                 tau=0.01,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 reward_dict={"PAIRWISE":.2,"CATE":.8,"CONSISTENCY":0,"DONE_REWARD":1000,"CONSISTENCY_REWARD":0,"PANELTY":-0.5}
                 ):
        self.image=train_x
        self.sample_number=train_x.shape[0]
        self.label=train_y
        self.permutation2piece={}
        self.cur_permutation={}
        self.image_num=image_num
        self.buffer_size=buffer_size
        self.piece_num=piece_num
        self.epsilon=epsilon
        self.epsilon_gamma=epsilon_gamma
        self.epochs=epochs
        self.tau=tau
        self.gamma=gamma
        self.done_list=[False for _ in range(self.image_num)]
        self.consistency_list=[False for _ in range(self.image_num)]
        self.permutation_list=[]
        self.device=device
        self.reward_dict=reward_dict

    
    def load_image(self,image_num,id=[]):
        if id:
            image_index=id
        else:
            image_index=random.choices(range(0,self.sample_number),k=image_num)
        for i in range(len(image_index)):
            image_raw=self.image[image_index[i]]
            permutation_raw=self.label[image_index[i]]
            image_raw=torch.tensor(image_raw).permute(2,0,1).to(torch.float).to(self.device)
            image_fragments=[
                image_raw[:,0:96,0:96],
                image_raw[:,0:96,96:192],
                image_raw[:,0:96,192:288],
                image_raw[:,96:192,0:96],
                image_raw[:,96:192,96:192],
                image_raw[:,96:192,192:288],
                image_raw[:,192:288,0:96],
                image_raw[:,192:288,96:192],
                image_raw[:,192:288,192:288]
            ]
            permutation_raw=list(np.argmax(permutation_raw,axis=1))
            for j in range(len(permutation_raw)):
                if permutation_raw[j]>=4:
                    permutation_raw[j]+=1
            permutation_raw.insert(4,4)
            for j in range(9):
                self.permutation2piece[permutation_raw[j]+9*i]=image_fragments[j]
            self.permutation2piece[-1]=torch.zeros(3,96,96).to(self.device)
        
    def get_image(self,permutation_list,image_index):
        image=torch.zeros(3,288,288).to(self.device)
        final_permutation=copy.deepcopy(permutation_list[0])
        final_permutation.insert(9//2,image_index*9+9//2)
        for i in range(9):
            image[:,(0+i//3)*96:(1+i//3)*96,(0+i%3)*96:(1+i%3)*96]=self.permutation2piece[final_permutation[i]]
    
        return image.unsqueeze(0),self.permutation2piece[-1]
    
    def request_for_image(self,image_id,permutation,image_index):
        self.load_image(image_num=self.image_num,id=image_id)
        image,outsider=self.get_image(permutation_list=permutation,image_index=image_index)
        self.load_image(image_num=self.image_num,id=self.image_id)
        return image,outsider

    def get_local_score(self,permutation,image_index):
        permutation_=permutation[0]
        permutation_copy=copy.deepcopy(permutation_)

        permutation_copy.insert(self.piece_num//2,image_index*self.piece_num+self.piece_num//2)

        if permutation_copy==list(range(image_index*self.piece_num,(image_index+1)*self.piece_num)):
            return self.reward_dict["DONE_REWARD"]

        local_reward=0
        edge_length=int(len(permutation_copy)**0.5)
        piece_num=self.piece_num
        hori_set=[(i,i+1) for i in [j for j in range(piece_num) if j%edge_length!=edge_length-1 ]]
        vert_set=[(i,i+edge_length) for i in range(piece_num-edge_length)]
        

        for j in range(len(hori_set)):#Pair reward

                hori_pair_set=(permutation_copy[hori_set[j][0]],permutation_copy[hori_set[j][1]])
                vert_pair_set=(permutation_copy[vert_set[j][0]],permutation_copy[vert_set[j][1]])
                if (-1 not in hori_pair_set) and (hori_pair_set[0]%piece_num,hori_pair_set[1]%piece_num) in hori_set and (hori_pair_set[0]//piece_num==hori_pair_set[1]//piece_num)and(hori_pair_set[0]//piece_num==image_index):
                    local_reward+=1*self.reward_dict["PAIRWISE"]
                if (-1 not in vert_pair_set) and (vert_pair_set[0]%piece_num,vert_pair_set[1]%piece_num) in vert_set and (vert_pair_set[0]//piece_num==vert_pair_set[1]//piece_num)and (vert_pair_set[0]//piece_num==image_index):
                    local_reward+=1*self.reward_dict["PAIRWISE"]
        for j in range(piece_num):
                if permutation_copy[j]!=-1:
                    local_reward+=(permutation_copy[j]%piece_num==j and permutation_copy[j]//piece_num==image_index)*self.reward_dict["CATE"]#Category reward
        return local_reward
    
    def get_consistency_reward(self,permutation,image_index):
        permutation_=permutation[0]
        score=0
        for piece in permutation_:
            if piece//self.piece_num==image_index:
                score+=1
        if score==self.piece_num-1:
            return self.reward_dict["CONSISTENCY_REWARD"]
        return score*self.reward_dict["CONSISTENCY"]




    def get_reward(self,permutation_list):
        """Return: local reward list, consistency reward list, done list, consistency list"""
        # permutation_list=permutation_list[::][:len(permutation_list[0])-1]#Comment if the buffer is not after the permutation
        permutation_copy=copy.deepcopy(permutation_list)
        for i in range(len(permutation_list)):
            permutation_copy[i].insert(self.piece_num//2,i*self.piece_num+self.piece_num//2)
        done_list=[0 for i in range(len(permutation_copy))]
        consistency_list=[0 for i in range(len(permutation_copy))]
        local_reward_list=[0 for i in range(len(permutation_copy))]
        consistency_reward_list=[0 for i in range(len(permutation_copy))]
        edge_length=int(len(permutation_copy[0])**0.5)
        piece_num=len(permutation_copy[0])
        hori_set=[(i,i+1) for i in [j for j in range(piece_num) if j%edge_length!=edge_length-1 ]]
        vert_set=[(i,i+edge_length) for i in range(piece_num-edge_length)]
        
        for i in range(len(permutation_list)):
            for j in range(len(hori_set)):#Pair reward

                hori_pair_set=(permutation_copy[i][hori_set[j][0]],permutation_copy[i][hori_set[j][1]])
                vert_pair_set=(permutation_copy[i][vert_set[j][0]],permutation_copy[i][vert_set[j][1]])
                if (-1 not in hori_pair_set) and (hori_pair_set[0]%piece_num,hori_pair_set[1]%piece_num) in hori_set and (hori_pair_set[0]//piece_num==hori_pair_set[1]//piece_num)and(hori_pair_set[0]//piece_num==i):
                    local_reward_list[i]+=1*self.reward_dict["PAIRWISE"]
                if (-1 not in vert_pair_set) and (vert_pair_set[0]%piece_num,vert_pair_set[1]%piece_num) in vert_set and (vert_pair_set[0]//piece_num==vert_pair_set[1]//piece_num)and (vert_pair_set[0]//piece_num==i):
                    local_reward_list[i]+=1*self.reward_dict["PAIRWISE"]

            piece_range=[0 for j in range (len(permutation_list))]
            # print(piece_range)
        
            for j in range(piece_num):
                if permutation_copy[i][j]!=-1:
                    piece_range[permutation_copy[i][j]//piece_num]+=1
                    local_reward_list[i]+=(permutation_copy[i][j]%piece_num==j and permutation_copy[i][j]//piece_num==i)*self.reward_dict["CATE"]#Category reward
            
            
            max_piece=piece_range[i]#Consistancy reward

            
            # if -1 in permutation_copy[i]:
            #     consistency_reward_list[j]+=0.5*CONSISTENCY_REWARD
            consistency_reward_list[i]=max_piece*self.reward_dict["CONSISTENCY"]
            if max_piece==piece_num:
                consistency_reward_list[i]=self.reward_dict["CONSISTENCY_REWARD"]
                consistency_list[i]=True

            local_reward_list[i]+=self.reward_dict["PANELTY"]
            consistency_reward_list[i]+=self.reward_dict["PANELTY"]
            start_index=min(permutation_copy[i])//piece_num*piece_num#Done reward
            if permutation_copy[i]==list(range(start_index,start_index+piece_num)):
                done_list[i]=True
                local_reward_list[i]=self.reward_dict["DONE_REWARD"]
        return local_reward_list[0],consistency_reward_list,done_list[0],consistency_list
        #Change after determined

    def show_image(self,image_permutation_list):

        image,_=self.get_image(permutation_list=image_permutation_list,image_index=0)
        image=image.squeeze().to("cpu")
        image=image.permute([1,2,0]).numpy().astype(np.uint8)
        cv2.imshow(f"Final image",image)
        cv2.waitKey(1)

    
    def permute(self,cur_permutation,action_index):
        # print("Env permuting")
        # print(f"Action: {action_index}")
        
        new_permutation=copy.deepcopy(cur_permutation)
        if action_index==92:
            return new_permutation
        if action_index<28:
            action=list(itertools.combinations(list(range(len(cur_permutation[0]))), 2))[action_index]
            value0=cur_permutation[0][action[0]]
            value1=cur_permutation[0][action[1]]
            new_permutation[0][action[0]]=value1
            new_permutation[0][action[1]]=value0
            return new_permutation
        else:
            action_index=action_index-28
            local_piece_index=action_index//8
            global_piece_index=action_index%8
            value0=cur_permutation[0][local_piece_index]
            value1=cur_permutation[1][global_piece_index]
            new_permutation[0][local_piece_index]=value1
            new_permutation[1][global_piece_index]=value0
            return new_permutation


    def summon_permutation_list(self,swap_num,id=[]):
        # print("Summon initial permutation")
        if id:
            image_index=id
        else:
            image_index=random.choices(range(0,self.sample_number),k=self.image_num)
        self.image_id=image_index
        self.load_image(image_num=self.image_num,id=self.image_id)
        print(f"Episode image:{self.image_id}")
        initial_permutation=list(range(self.piece_num*self.image_num))
        
        for i in range(self.image_num):
            initial_permutation.pop(9*i+9//2-i)
        self.permutation_list=[initial_permutation[j*(self.piece_num-1):(j+1)*(self.piece_num-1)]
                                for j in range(self.image_num)]
        for i in range(swap_num):
            action_index=random.randint(0,91)
            self.permutation_list=self.permute(self.permutation_list,action_index)
        print(f"Initial permutation {self.permutation_list}")
        
        
    def get_accuracy(self,permutation_list):
        """return: done_acc, consistency_acc, category_acc, hori_acc, vert_acc"""
        permutation_copy=[copy.deepcopy(permutation_list)[0]]
        for i in range(len(permutation_copy)):
            permutation_copy[i].insert(self.piece_num//2,i*self.piece_num+self.piece_num//2)
        done_list=[0 for i in range(len(permutation_copy))]
        consistency_list=[0 for i in range(len(permutation_copy))]
        category_list=[0 for i in range(len(permutation_copy))]
        hori_list=[0 for i in range(len(permutation_copy))]
        vert_list=[0 for i in range(len(permutation_copy))]
        edge_length=int(len(permutation_copy[0])**0.5)
        piece_num=len(permutation_copy[0])
        hori_set=[(i,i+1) for i in [j for j in range(piece_num) if j%edge_length!=edge_length-1 ]]
        vert_set=[(i,i+edge_length) for i in range(piece_num-edge_length)]

        for i in range(len(permutation_copy)):
            for j in range(len(hori_set)):#Pair reward

                hori_pair_set=(permutation_copy[i][hori_set[j][0]],permutation_copy[i][hori_set[j][1]])
                vert_pair_set=(permutation_copy[i][vert_set[j][0]],permutation_copy[i][vert_set[j][1]])
                if (-1 not in hori_pair_set) and (hori_pair_set[0]%piece_num,hori_pair_set[1]%piece_num) in hori_set and (hori_pair_set[0]//piece_num==hori_pair_set[1]//piece_num)and(hori_pair_set[0]//piece_num==i):
                    hori_list[i]+=1
                if (-1 not in vert_pair_set) and (vert_pair_set[0]%piece_num,vert_pair_set[1]%piece_num) in vert_set and (vert_pair_set[0]//piece_num==vert_pair_set[1]//piece_num)and (vert_pair_set[0]//piece_num==i):
                    vert_list[i]+=1
            piece_range=[0 for j in range (len(permutation_list))]
            for j in range(piece_num):
                if permutation_copy[i][j]!=-1:
                    piece_range[permutation_copy[i][j]//piece_num]+=1
                    category_list[i]+=(permutation_copy[i][j]%piece_num==j and permutation_copy[i][j]//piece_num==i)#Category reward
            
            max_piece=piece_range[i]

            consistency_list[i]=max_piece

            start_index=min(permutation_copy[i])//piece_num*piece_num
            if permutation_copy[i]==list(range(start_index,start_index+piece_num)):
                done_list[i]=1
        done_accuracy=np.mean(done_list)
        consistency_accuracy=np.mean([(consistency_list[i]-1)/(piece_num-1) for i in range(len(permutation_copy))])
        category_accuracy=np.mean([category_list[i]/(piece_num-1) for i in range(len(permutation_copy))])
        hori_accuracy=np.mean([hori_list[i]/len(hori_set) for i in range(len(permutation_copy))])
        vert_accuracy=np.mean([vert_list[i]/len(vert_set) for i in range(len(permutation_copy))])
        return done_accuracy,consistency_accuracy,category_accuracy,hori_accuracy,vert_accuracy
    
