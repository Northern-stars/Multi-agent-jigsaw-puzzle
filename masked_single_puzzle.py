import torch
import torch.nn as nn
import numpy as np
import random
import copy
import itertools
# from outsider_pretrain import fen_model
from torchvision.models import efficientnet_b0,efficientnet_b3
# from torch.utils.data import Dataset,DataLoader
import cv2
import Vit
import os
import time
from matplotlib import pyplot as plt

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DONE_REWARD=1000
GAMMA=0.998
CLIP_GRAD_NORM=0.1
TRAIN_PER_STEP=25
ACTOR_LR=1e-4
ACTOR_LR_MIN=1e-6
CRITIC_LR=1e-3
CRITIC_LR_MIN=1e-5
CRITIC_UPDATE_TIME=3
ENCODER_LR=1e-4
ACTOR_SCHEDULAR_STEP=500
CRITIC_SCHEDULAR_STEP=500
ENCODER_SCHEDULAR_STEP=100
BASIC_BIAS=1e-8
SHOW_IMAGE=True


PAIR_WISE_REWARD=.2
CATE_REWARD=.8
CONSISTENCY_REWARD=0
CONSISTENCY_REWARD_WEIGHT=0.5
PANELTY=-0.2
ENTROPY_WEIGHT=0.05
ENTROPY_GAMMA=0.998
ENTROPY_MIN=0.005

EPOCH_NUM=1000
LOAD_MODEL=True
SWAP_NUM=[1,3,5,5]
MAX_STEP=[400,300,300,200]
MODEL_NAME="(3).pth"
ACTOR_PATH=os.path.join("model/Actor"+MODEL_NAME)
CRITIC_PATH=os.path.join("model/Critic"+MODEL_NAME)

BATCH_SIZE=50
EPSILON=0.5
EPSILON_GAMMA=0.995
EPSILON_MIN=0.1
AGENT_EPOCHS=5

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
print(f"Data shape: x {train_x.shape}, y {train_y.shape}")

class Fen_model(nn.Module):
    def __init__(self, hidden_size1, hidden_size2):
        super(Fen_model, self).__init__()
        self.ef = efficientnet_b3(weights="DEFAULT")
        self.ef.classifier = nn.Linear(1536, 1024)

        self.contrast_fc_hori = nn.Linear(2048, 1024)
        self.contrast_fc_vert = nn.Linear(2048, 1024)

        self.fc1 = nn.Linear(1024*12, hidden_size1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_size1)
        self.do = nn.Dropout1d(p=0.1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)

        # 定义 index 对
        self.hori_set = [(i, i+1) for i in range(9) if i % 3 != 2]
        self.vert_set = [(i, i+3) for i in range(6)]

    def forward(self, image, mask=None):
        B, C, H, W = image.shape
        
        patches = image.unfold(2, 96, 96).unfold(3, 96, 96)
        # patches: (B, C, 3, 3, 96, 96)
        patches = patches.permute(0,2,3,1,4,5).contiguous()
        patches = patches.view(B*9, C, 96, 96)  

        feats = self.ef(patches)  
        feats = feats.view(B, 9, 1024) 

        hori_pairs = torch.stack([torch.cat([feats[:, i, :], feats[:, j, :]], dim=-1) 
                                  for (i,j) in self.hori_set], dim=1)
        hori_feats = self.contrast_fc_hori(hori_pairs) 

        vert_pairs = torch.stack([torch.cat([feats[:, i, :], feats[:, j, :]], dim=-1) 
                                  for (i,j) in self.vert_set], dim=1)  
        vert_feats = self.contrast_fc_vert(vert_pairs) 

        if mask is not None:
            hori_mask=torch.ones(B,len(self.hori_set)).to(DEVICE)
            vert_mask=torch.ones(B,len(self.vert_set)).to(DEVICE)
            for i in range(len(mask)):
                for j in range(len(mask[i])):
                    batch_id,piece_id=i,mask[i][j]
                    for k in range(len(self.hori_set)):
                        if piece_id in self.hori_set[k]:
                            hori_mask[batch_id][k]=0
                        if piece_id in self.vert_set[k]:
                            vert_mask[batch_id][k]=0

            

            hori_mask=hori_mask.unsqueeze(-1)
            vert_mask=vert_mask.unsqueeze(-1)

            hori_feats=hori_feats*hori_mask
            vert_feats=vert_feats*vert_mask


        feature_tensor = torch.cat([hori_feats, vert_feats], dim=1)
        feature_tensor = feature_tensor.view(B, -1)  

        x = self.do(feature_tensor)
        x = self.fc1(x)
        x = self.do(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class Actor_model(nn.Module):
    def __init__(self,fen_hidden1,fen_hidden2,hidden1,hidden2,action_num,dropout=0.1):
        super(Actor_model,self).__init__()
        self.fen_model=Fen_model(fen_hidden1,fen_hidden2)
        self.fc1=nn.Linear(fen_hidden2,hidden1)
        self.processing_block1=nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(dropout)
        )
        self.fc2=nn.Linear(hidden1,hidden2)
        self.processing_block2=nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(dropout)
        )
        self.out_fc=nn.Linear(hidden2,action_num)
    
    def forward(self,image,mask=None):
        feature_tensor=self.fen_model(image,mask)
        feature_tensor=self.fc1(feature_tensor)
        feature_tensor=self.processing_block1(feature_tensor)
        feature_tensor=self.fc2(feature_tensor)
        feature_tensor=self.processing_block2(feature_tensor)
        out=self.out_fc(feature_tensor)
        # out=nn.functional.softmax(out,dim=-1)
        return out
    
class Critic_model(nn.Module):
    def __init__(self,fen_hidden1,fen_hidden2,hidden1,hidden2,dropout=0.1):
        super().__init__()
        self.fen_model=Fen_model(fen_hidden1,fen_hidden2)
        self.fc1=nn.Linear(fen_hidden2,hidden1)
        self.processing_block1=nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(dropout)
        )
        self.fc2=nn.Linear(hidden1,hidden2)
        self.processing_block2=nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(dropout)
        )
        self.out_fc=nn.Linear(hidden2,1)
    
    def forward(self,image,mask=None):
        feature_tensor=self.fen_model(image,mask)
        feature_tensor=self.fc1(feature_tensor)
        feature_tensor=self.processing_block1(feature_tensor)
        feature_tensor=self.fc2(feature_tensor)
        feature_tensor=self.processing_block2(feature_tensor)
        out=self.out_fc(feature_tensor)
        return out



def check_gradients(model, verbose=True, min_abs_grad=1e-10):
    summary = {}
    total_params = 0
    no_grad_params = 0

    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is None:
            summary[name] = "No grad (None)"
            no_grad_params += 1
        elif torch.all(torch.abs(param.grad) < min_abs_grad):
            summary[name] = "Near-zero grad"
            no_grad_params += 1
        else:
            summary[name] = "OK"

    if verbose:
        print(f"\n[Gradient Check] {no_grad_params}/{total_params} params have no/near-zero gradients.")
        for name, status in summary.items():
            print(f"{name:50s} : {status}")

    return summary



class env:
    def __init__(self,
                 train_x,
                 train_y,
                 memory_size,
                 batch_size,
                 gamma,
                 device,
                 model,#input: image. Output: Q value
                 image_num,
                 buffer_size,
                 entropy_weight,
                 epsilon,
                 epsilon_gamma,
                 action_num=28,
                 piece_num=9,
                 epochs=10,
                 tau=0.01,
                 lmb=0.95
                 ):
        self.image=train_x
        self.sample_number=train_x.shape[0]
        self.label=train_y
        self.permutation2piece={}
        self.cur_permutation={}
        self.mkv_memory=[]
        self.mkv_memory_size=memory_size
        self.memory_counter=0
        self.trace_start_point=0
        self.model=model
        self.main_model=copy.deepcopy(model)
        self.optimizer=torch.optim.AdamW(self.main_model.parameters(),lr=ACTOR_LR)
        self.schedular=torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=ACTOR_SCHEDULAR_STEP, gamma=0.1)
        self.loss_fn=nn.MSELoss()
        self.device=device
        self.batch_size=batch_size
        self.gamma=gamma
        self.image_num=image_num
        self.buffer_size=buffer_size
        self.action_list=[[0 for _ in range(action_num)] for __ in range(image_num)]
        self.piece_num=piece_num
        self.entropy_weight=entropy_weight
        self.epsilon=epsilon
        self.epsilon_gamma=epsilon_gamma
        self.epochs=epochs
        self.tau=tau
        self.lmb=lmb
        self.action_num=action_num
    
    def load_image(self,image_num,id=[]):
        if id:
            image_index=id
        else:
            image_index=random.choices(range(0,self.sample_number),k=image_num)

        for i in range(len(image_index)):
            image_raw=self.image[image_index[i]]
            permutation_raw=self.label[image_index[i]]
            image_raw=torch.tensor(image_raw).permute(2,0,1).to(torch.float)
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
            permutation_raw=list(permutation_raw)
            for m in range(8):
                for n in range(8):
                    # print(type(imageLabel[i]))
                    # print(imageLabel)
                    if permutation_raw[m][n]==True:
                        if n>=4:
                            permutation_raw[m]=n+1
                            break
                        else:
                            permutation_raw[m]=n
                            break
            permutation_raw.insert(4,4)
            for j in range(9):
                self.permutation2piece[permutation_raw[j]+9*i]=image_fragments[j]
            self.permutation2piece[-1]=torch.zeros(3,96,96)
        
    def get_image(self,permutation,image_index):
        image=torch.zeros(3,288,288)
        final_permutation=copy.deepcopy(permutation)
        final_permutation.insert(9//2,image_index*9+9//2)
        for i in range(9):
            image[:,(0+i//3)*96:(1+i//3)*96,(0+i%3)*96:(1+i%3)*96]=self.permutation2piece[final_permutation[i]]
    
        # outsider_piece=self.permutation2piece[permutation[-1]]
        # return image.unsqueeze(0).to(self.device),outsider_piece.unsqueeze(0).to(self.device)
        return image.unsqueeze(0).to(self.device)

    def get_reward(self,permutation_list):

        # permutation_list=permutation_list[::][:len(permutation_list[0])-1]#Comment if the buffer is not after the permutation
        permutation_copy=copy.deepcopy(permutation_list)
        for i in range(len(permutation_list)):
            permutation_copy[i].insert(self.piece_num//2,i*self.piece_num+self.piece_num//2)
        done_list=[0 for i in range(len(permutation_copy))]
        local_reward_list=[0 for i in range(len(permutation_copy))]
        edge_length=int(len(permutation_copy[0])**0.5)
        piece_num=len(permutation_copy[0])
        hori_set=[(i,i+1) for i in [j for j in range(piece_num) if j%edge_length!=edge_length-1 ]]
        vert_set=[(i,i+edge_length) for i in range(piece_num-edge_length)]
        right_piece=[0 for i in range(len(permutation_copy))]
        
        for i in range(len(permutation_list)):
            for j in range(len(hori_set)):#Pair reward

                hori_pair_set=(permutation_copy[i][hori_set[j][0]],permutation_copy[i][hori_set[j][1]])
                vert_pair_set=(permutation_copy[i][vert_set[j][0]],permutation_copy[i][vert_set[j][1]])
                if (-1 not in hori_pair_set) and (hori_pair_set[0]%piece_num,hori_pair_set[1]%piece_num) in hori_set and (hori_pair_set[0]//piece_num==hori_pair_set[1]//piece_num)and(hori_pair_set[0]//piece_num==i):
                    local_reward_list[i]+=1*PAIR_WISE_REWARD
                if (-1 not in vert_pair_set) and (vert_pair_set[0]%piece_num,vert_pair_set[1]%piece_num) in vert_set and (vert_pair_set[0]//piece_num==vert_pair_set[1]//piece_num)and (vert_pair_set[0]//piece_num==i):
                    local_reward_list[i]+=1*PAIR_WISE_REWARD

            piece_range=[0 for j in range (len(permutation_list))]
            # print(piece_range)
            
        
            for j in range(piece_num):
                if permutation_copy[i][j]!=-1:
                    piece_range[permutation_copy[i][j]//piece_num]+=1
                    
                    right_piece[i]+=(permutation_copy[i][j]%piece_num==j and permutation_copy[i][j]//piece_num==i)#Category reward
            


            if right_piece[i]==8:
                local_reward_list[i]=DONE_REWARD
                done_list[i]=1
            else:

                local_reward_list[i]+=right_piece[i]*CATE_REWARD
                local_reward_list[i]+=PANELTY
            
        return [local_reward_list[i] for i in range (self.image_num)],done_list
    

    def clean_memory(self):
        self.mkv_memory=[]
        self.memory_counter=0

    def show_image(self,image_permutation_list):
        for i in range(self.image_num):

            image=self.get_image(permutation=image_permutation_list[i],image_index=i)
            image=image.squeeze().to("cpu")
            image=image.permute([1,2,0]).numpy().astype(np.uint8)
            cv2.imshow(f"Final image {i}",image)
        cv2.waitKey(1)
        # time.sleep(10)
        # cv2.destroyAllWindows()
    
    def epsilon_greedy(self,action):
        prob=random.random()
        if prob>self.epsilon:
            return action
        else:
            epsilon_action=(action+random.randint(1,len(self.action_list[0])))%len(self.action_list[0])
            return epsilon_action


    def permute(self,cur_permutation,action_index):
        
        new_permutation=copy.deepcopy(cur_permutation)
        if action_index==len(self.action_list[0]):
            return new_permutation
        # print(action_index,len(self.action_list[0]))
        action=list(itertools.combinations(list(range(len(cur_permutation))), 2))[action_index]
        
        value0=cur_permutation[action[0]]
        value1=cur_permutation[action[1]]
        new_permutation[action[0]]=value1
        new_permutation[action[1]]=value0
        return new_permutation
    

    def summon_permutation_list(self,swap_num,id=[]):
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
        for i in range(swap_num):
            action_index=random.randint(0,len(self.action_list[0]))
            initial_permutation=self.permute(initial_permutation,action_index)
        masked_piece=random.randint(0,7)
        initial_permutation[masked_piece]=-1
        print(f"Initial permutation {initial_permutation}")
        return initial_permutation

    def recording_memory(self,image_id,image_index,state,action,reward,next_state,log_prob,done):
        memory={"Image_id":image_id,"Image_index":image_index,"State":state,"Action": action,"Log_prob":log_prob,"Reward":reward,"Next_state":next_state,"Done": done}
        if len(self.mkv_memory)<self.mkv_memory_size:
            self.mkv_memory.append(memory)
        else:
            self.mkv_memory[self.memory_counter]=memory
        self.memory_counter=(self.memory_counter+1)%self.mkv_memory_size

        
    def choose_action(self,perm_with_buf,image_index):
        perm_=copy.deepcopy(perm_with_buf)
        value_list=[]
        image_list=[]

        mask_list=[]

        for i in range(len(self.action_list[0])):
            perm_=self.permute(perm_with_buf,i)
            mask=self.get_mask(perm_)
            image=self.get_image(perm_,image_index=image_index)
            image_list.append(copy.deepcopy(image.cpu()))
            mask_list.append(mask)
        
        i=0
        with torch.no_grad():
            while i < len(self.action_list):
                if len(self.action_list)-i<BATCH_SIZE:
                    image=torch.cat(image_list[i:],dim=0).to(DEVICE)
                    mask=mask_list[i:]
                else:
                    image=torch.cat(image_list[i:i+BATCH_SIZE],dim=0).to(DEVICE)
                    mask=mask_list[i:i+BATCH_SIZE]

                value=self.model(image,mask)
                value_list.append(value.squeeze(-1).to("cpu"))
                i+=BATCH_SIZE
            
            value_list=torch.cat(value_list)
            best_action=torch.argmax(value_list).item()
        return int(best_action)





    def recording_memory(self,image_id,image_index,state,action,reward,next_state,done):
        memory={"Image_id":image_id,"Image_index":image_index,"State":state,"Action": action,"Reward":reward,"Next_state":next_state,"Done": done}
        if len(self.mkv_memory)<self.mkv_memory_size:
            self.mkv_memory.append(memory)
        else:
            self.mkv_memory[self.memory_counter]=memory
        self.memory_counter=(self.memory_counter+1)%self.mkv_memory_size
    
    def get_mask(self,permutation):
        mask=[]
        for i in range(len(permutation)):
            if permutation[i]==-1:
                if i>=self.piece_num//2:
                    index=i+1
                else:
                    index=i
                mask.append(index)
        return mask

        
    def update(self,show=False):

        self.model.train()
        self.main_model.train()
        #select batch_size sample
        order=list(range(len(self.mkv_memory)))
        random.shuffle(order)
        loss_sum=0
        for i in range(self.epochs):
            if i*self.batch_size>=len(order):
                break
            if len(order)-i*self.batch_size<self.batch_size:
                sample_dicts=[self.mkv_memory[j] for j in order[i*self.batch_size:]]
            else:
                sample_dicts=[self.mkv_memory[j] for j in order[i*self.batch_size:(i+1)*self.batch_size]]
            states=[]

            actions=[]
            next_states=[]
            reward=[]
            cur_mask=[]
            next_mask=[]
            for a in range(len(sample_dicts)):
                self.load_image(image_num=self.image_num,id=sample_dicts[a]["Image_id"])

                    
                current_image=self.get_image(sample_dicts[a]["State"],image_index=sample_dicts[a]["Image_index"])
                states.append(current_image)
                cur_mask.append(self.get_mask(sample_dicts[a]["State"]))

                actions.append(sample_dicts[a]["Action"])
                next_image=self.get_image(sample_dicts[a]["Next_state"],image_index=sample_dicts[a]["Image_index"])
                next_states.append(next_image)

                next_mask.append(self.get_mask(sample_dicts[a]["Next_state"]))
                
                reward.append(sample_dicts[a]["Reward"])
            


            state_tensor=torch.cat(states,dim=0)
            batch_size=state_tensor.size(0)
            if batch_size==1:
                self.model.eval()
                self.main_model.eval()
            else:
                self.model.train()
                self.main_model.train()
            next_state_tensor=torch.cat(next_states,dim=0)

            q_next=self.model(next_state_tensor,next_mask).detach()
            
            q_eval=self.main_model(next_state_tensor,next_mask)

            reward=torch.tensor(reward,dtype=torch.float32).to(self.device).unsqueeze(-1)

            q_target=reward+self.gamma*q_next
            q_target=q_target.to(torch.float)

            loss=self.loss_fn(q_target,q_eval)
            self.optimizer.zero_grad()
            loss.float().backward()
            self.optimizer.step()
            loss_sum+=loss.item()
        self.schedular.step()
        for target_param, main_param in zip(self.model.parameters(),self.main_model.parameters()):
            target_param.data.copy_(self.tau*main_param.data+(1-self.tau)*target_param.data)
        if show:
            print(f"Average_loss: {loss_sum/self.epochs}")


    def step(self, epoch=500, load=True):

        
        if load:
            self.model.load_state_dict(torch.load(ACTOR_PATH))
            self.main_model=copy.deepcopy(self.model)
        reward_record=[]
        for i in range(epoch):
            # self.clean_memory()
            if i > 300:
                max_step, swap_num = MAX_STEP[3], SWAP_NUM[3]
            elif i > 200:
                max_step, swap_num = MAX_STEP[2], SWAP_NUM[2]
            elif i > 100:
                max_step, swap_num = MAX_STEP[1], SWAP_NUM[1]
            else:
                max_step, swap_num = MAX_STEP[0], SWAP_NUM[0]

            
            initial_perm   = self.summon_permutation_list(swap_num=swap_num)
            permutation_list = [initial_perm[j*self.piece_num:(j+1)*self.piece_num]
                                for j in range(self.image_num)]
            buffer               = [-1] * self.buffer_size
            done_list            = [False] * self.image_num
            reward_sum_list      = [[] for _ in range(self.image_num)]
            termination_list     = [0 for _ in range(self.image_num)]

            pending_transitions={j: None for j in range(self.image_num)}

            self.action_list=[[0 for _ in range(self.action_num)] for __ in range(self.image_num)]
            self.trace_start_point=self.memory_counter
            last_action=[-1 for _ in range(self.image_num)]


            step = 0; done = False ; train_flag=False
            while not done and step < max_step:
                state_list=[]
                do_list = []
                model_action=[]
                last_reward_list,_=self.get_reward(permutation_list=permutation_list)
                with torch.no_grad():
                    self.model.eval()
                    
                    for j in range(self.image_num):
                        if done_list[j] :           
                            continue
                        
                        
                        # perm_with_buf = permutation_list[j] + buffer
                        perm_with_buf=permutation_list[j]
                        state_list.append(copy.deepcopy(perm_with_buf))
                        image = self.get_image(perm_with_buf,image_index=j)

                        action=self.epsilon_greedy(self.choose_action(perm_with_buf,image_index=j))

                        if pending_transitions[j] is not None:
                            prev_state,prev_action,prev_reward=pending_transitions[j]
                            self.recording_memory(image_id=self.image_id,
                                                  image_index=j,
                                                  state=prev_state,
                                                  action= prev_action,
                                                  reward= prev_reward,
                                                  next_state= perm_with_buf,
                                                  done=1 if done_list[j] else 0)
                        
                        do_list.append(j)



                        # if last_action[j]==int(action.item()):
                        #     termination_list[j]+=1
                        # else:
                        #     termination_list[j]=0
                        #     last_action[j]=int(action.item())


                        
                        # action=self.epsilon_greedy(action=action.item())
                        # action=action.item()
                        model_action.append(action)
                        self.action_list[j][action]+=1
                        

                        # if action==36:
                        #     termination_list[j]=True
                        #     continue

                        pending_transitions[j]=(perm_with_buf,action,0)

                        new_perm = self.permute(perm_with_buf, action)
                        permutation_list[j], buffer = new_perm[:self.piece_num-1], new_perm[self.piece_num-1:]

                        



                reward_list, done_list = self.get_reward(permutation_list)
                if SHOW_IMAGE:
                    self.show_image(permutation_list)
                for j in do_list:
                    reward_sum_list[j].append(reward_list[j])

                    prev_state,prev_action,_=pending_transitions[j]
                    pending_transitions[j]=(prev_state,prev_action,reward_list[j])
                
                done = all(done_list)
                

                if step!=0 and step%TRAIN_PER_STEP==0 and len(self.mkv_memory)>0:
                    train_flag=False
                    self.update()
                    self.load_image(image_num=self.image_num,id=self.image_id)
                    # self.clean_memory()
                    self.trace_start_point=self.memory_counter
                elif train_flag and len(self.mkv_memory)>0:
                    train_flag=False
                    self.epsilon/=(EPSILON_GAMMA)
                    self.update()
                    self.load_image(image_num=self.image_num,id=self.image_id)
                    # self.clean_memory()
                    self.trace_start_point=self.memory_counter
                step += 1

            
            for j in range(self.image_num):
                if pending_transitions[j] is not None and done_list[j]:
                    prev_state,prev_action,prev_reward=pending_transitions[j]
                    self.recording_memory(image_id=self.image_id,
                                                  image_index=j,
                                                  state=prev_state,
                                                  action= prev_action,
                                                  reward= prev_reward,
                                                  next_state= perm_with_buf,
                                                  done=1 if done_list[j] else 0)

            print(f"Epoch: {i}, step: {step}, reward: {[sum(reward_sum_list[j])/len(reward_sum_list[j]) for j in range(len(reward_sum_list)) if len(reward_sum_list[j])!=0 ]}")
            print(f"Action_list: {self.action_list}")
            print(f"Permutation list: {permutation_list}")
            reward_record.append([sum(reward_sum_list[j])/len(reward_sum_list[j]) for j in range(len(reward_sum_list)) if len(reward_sum_list[j])!=0 ])
            for j in range(self.image_num):
                if termination_list[j]>=20:
                    self.epsilon/=(EPSILON_GAMMA)
            if len(self.mkv_memory)>0:
                self.update(show=True)
            if self.epsilon>EPSILON_MIN:
                self.epsilon*=EPSILON_GAMMA
            torch.save(self.model.state_dict(),ACTOR_PATH)
        

        self.plot_reward_curve(reward_record)
            
                
    def plot_reward_curve(self,reward_record):
        avg_reward=[]
        for i in range(len(reward_record)):
            avg_reward.append(sum(reward_record[i])/len(reward_record[i]))
        plt.plot(range(len(avg_reward)),avg_reward)
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.title("Reward Curve")
        plt.show()
            






if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    local_switcher_model=Critic_model(fen_hidden1=2048,fen_hidden2=1024,hidden1=2048,hidden2=1024)
    local_switcher_model.load_state_dict(torch.load("model/sd2rl_pretrain.pth"),strict=False)
    
    critic_model=Critic_model(fen_hidden1=2048,fen_hidden2=1024,hidden1=1024,hidden2=512).to(DEVICE)
    critic_model.fen_model.load_state_dict(local_switcher_model.fen_model.state_dict())
    # actor_model=Actor_model(fen_hidden1=2048,fen_hidden2=1024,hidden1=1024,hidden2=512,action_num=1).to(DEVICE)
    # critic=critic_model(picture_size=[1,3,288,288],
    #                     outsider_size=[1,3,96,96],
    #                     patch_size=16,
    #                     encoder_hidden=1024,
    #                     n_head=12,
    #                     unet_hidden=1024,
    #                     encoder_layer_num=12,
    #                     critic_hidden=1024,
    #                     output_channel=3,
    #                     transformer_out_size=2048).to(DEVICE)
    # actor=actor_model(picture_size=[1,3,288,288],
    #                   outsider_size=[1,3,96,96],
    #                   patch_size=12,
    #                   encoder_layer_num=12,
    #                   n_head=16,
    #                   transformer_output_size=2048,
    #                   unet_hidden=1024,
    #                   encoder_hidden=1024,
    #                   output_channel=3,
    #                   actor_hidden=1024,
    #                   action_num=46).to(DEVICE)

    # feature_encoder=fen_model(512,512).to(device=DEVICE)
    environment=env(train_x=train_x,
                    train_y=train_y,
                    memory_size=2000,
                    batch_size=BATCH_SIZE,
                    gamma=GAMMA,
                    device=DEVICE,
                    model=critic_model,
                    # encoder=feature_encoder,
                    image_num=1,
                    buffer_size=0,
                    epsilon=EPSILON,
                    epsilon_gamma=EPSILON_GAMMA,
                   entropy_weight=ENTROPY_WEIGHT)
    environment.step(epoch=EPOCH_NUM,load=LOAD_MODEL)
    # environment.load_image(1,[100])
    # environment.recording_memory(100,0,[1,0,2,3,5,6,7,8],1,200,[0,1,2,3,5,6,7,8],0.5,1)
    # environment.update(show=True)

    # environment.load_image(1,id=[1000])
    # environment.show_image([[0,1,2,3,4,5,6,7,8]])
    # reward_list,done_list=environment.get_reward([[0,1,2,3,4,5,6,7,8],[9,10,11,12,13,14,15,16,17]])
    # print(reward_list,done_list)
    
