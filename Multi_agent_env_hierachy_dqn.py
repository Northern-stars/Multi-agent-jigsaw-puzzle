import torch
import torch.nn as nn
import numpy as np
import random
import copy
import itertools
# from outsider_pretrain import fen_model
from torchvision.models import efficientnet_b0,efficientnet_b3
from torch.utils.data import Dataset,DataLoader
from pretrain import pretrain_model
import cv2
import Vit
import os
import time

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DONE_REWARD=1000
GAMMA=0.995
CLIP_GRAD_NORM=0.1
TRAIN_PER_STEP=25
ACTOR_LR=1e-4
ACTOR_LR_MIN=1e-6
CRITIC_LR=1e-3
CRITIC_LR_MIN=1e-5
ENCODER_LR=1e-4
ACTOR_SCHEDULAR_STEP=200

CRITIC_SCHEDULAR_STEP=100
ENCODER_SCHEDULAR_STEP=100
BASIC_BIAS=1e-8
SHOW_IMAGE=True


PAIR_WISE_REWARD=.2
CATE_REWARD=.8
CONSISTENCY_REWARD=.2
PANELTY=-0.5
ENTROPY_WEIGHT=0.01
ENTROPY_GAMMA=0.998
ENTROPY_MIN=0.005

EPOCH_NUM=2000
LOAD_MODEL=False
SWAP_NUM=[2,3,4,8]
MAX_STEP=[200,200,200,200]
MODEL_NAME="(1).pth"
MODEL_PATH=os.path.join("DQN"+MODEL_NAME)


BATCH_SIZE=20
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

def has_any_gradient(model):
    """检查模型是否有任何梯度"""
    return any(param.grad is not None for param in model.parameters() if param.requires_grad)

def selective_load_state_dict(source_model, target_model, layer_mapping):
    """
    选择性加载state_dict
    
    参数:
    - source_model: 源模型
    - target_model: 目标模型  
    - layer_mapping: 字典，{源层名: 目标层名}
    """
    source_state_dict = source_model.state_dict()
    target_state_dict = target_model.state_dict()
    
    # 创建新的state_dict，只更新指定的层
    new_state_dict = target_state_dict.copy()
    
    for src_layer, tgt_layer in layer_mapping.items():
        src_weight_key = f"{src_layer}.weight"
        src_bias_key = f"{src_layer}.bias"
        
        tgt_weight_key = f"{tgt_layer}.weight"
        tgt_bias_key = f"{tgt_layer}.bias"
        
        # 复制权重
        if src_weight_key in source_state_dict and tgt_weight_key in new_state_dict:
            if source_state_dict[src_weight_key].shape == new_state_dict[tgt_weight_key].shape:
                new_state_dict[tgt_weight_key] = source_state_dict[src_weight_key].clone()
            else:
                print(f"权重形状不匹配: {src_weight_key} -> {tgt_weight_key}")
        
        # 复制偏置
        if src_bias_key in source_state_dict and tgt_bias_key in new_state_dict:
            if source_state_dict[src_bias_key].shape == new_state_dict[tgt_bias_key].shape:
                new_state_dict[tgt_bias_key] = source_state_dict[src_bias_key].clone()
            else:
                print(f"偏置形状不匹配: {src_bias_key} -> {tgt_bias_key}")
    
    # 加载更新后的state_dict
    target_model.load_state_dict(new_state_dict, strict=False)
    return target_model


class fen_model(nn.Module):
    def __init__(self, hidden_size1, hidden_size2):
        super(fen_model, self).__init__()
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

    def forward(self, image):
        B, C, H, W = image.shape   # 假设输入 (B, 3, 288, 288)

        # ---- 1. 切片并 reshape 成 batch ----
        patches = image.unfold(2, 96, 96).unfold(3, 96, 96)
        # patches: (B, C, 3, 3, 96, 96)
        patches = patches.permute(0,2,3,1,4,5).contiguous()
        patches = patches.view(B*9, C, 96, 96)   # (B*9, 3, 96, 96)

        # ---- 2. 一次性送进 ef ----
        feats = self.ef(patches)   # (B*9, 1024)
        feats = feats.view(B, 9, 1024)  # (B, 9, 1024)

        # ---- 3. 构建横向对 ----
        hori_pairs = torch.stack([torch.cat([feats[:, i, :], feats[:, j, :]], dim=-1) 
                                  for (i,j) in self.hori_set], dim=1)  # (B, #pairs, 1024)
        hori_feats = self.contrast_fc_hori(hori_pairs)  # (B, #pairs, 512)

        # ---- 4. 构建纵向对 ----
        vert_pairs = torch.stack([torch.cat([feats[:, i, :], feats[:, j, :]], dim=-1) 
                                  for (i,j) in self.vert_set], dim=1)  # (B, #pairs, 1024)
        vert_feats = self.contrast_fc_vert(vert_pairs)  # (B, #pairs, 512)

        # ---- 5. 拼接所有特征 ----
        feature_tensor = torch.cat([hori_feats, vert_feats], dim=1)  # (B, 12, 512)
        feature_tensor = feature_tensor.view(B, -1)   # (B, 12*512)

        # ---- 6. 全连接部分 ----
        x = self.do(feature_tensor)
        x = self.fc1(x)
        x = self.do(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class Buffer_switcher_model(nn.Module):
    def __init__(self,hidden_size1,hidden_size2,outsider_hidden_size,action_num):
        super(Buffer_switcher_model,self).__init__()
        self.fen_model=fen_model(hidden_size1=hidden_size1,hidden_size2=hidden_size1)
        self.outsider_fen_model=efficientnet_b0(weights="DEFAULT")
        self.outsider_fen_model.classifier=nn.Linear(1280,outsider_hidden_size)
        self.outsider_contrast_fc=nn.Linear(2*outsider_hidden_size,outsider_hidden_size)
        self.outsider_fc=nn.Linear(outsider_hidden_size*9,outsider_hidden_size)
        # state_dict=torch.load("pairwise_pretrain.pth")
        # state_dict_replace = {
        # k: v 
        # for k, v in state_dict.items() 
        # if k.startswith("ef.")
        # }
        # load_result_hori=self.fen_model.load_state_dict(state_dict_replace,strict=False)
        # print("Critic missing keys hori",load_result_hori.missing_keys)
        # print("Critic unexpected keys hori",load_result_hori.unexpected_keys)
        self.fc1=nn.Linear(hidden_size1+outsider_hidden_size,hidden_size2)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=0.1)
        self.fc2=nn.Linear(hidden_size2,action_num)
        self.bn=nn.BatchNorm1d(hidden_size2)
    
    def forward(self,image,outsider):
        image_fragments=[
            image[:,:,0:96,0:96],
            image[:,:,0:96,96:192],
            image[:,:,0:96,192:288],
            image[:,:,96:192,0:96],
            image[:,:,96:192,96:192],
            image[:,:,96:192,192:288],
            image[:,:,192:288,0:96],
            image[:,:,192:288,96:192],
            image[:,:,192:288,192:288]
        ]
        image_input=self.fen_model(image)
        outsider_input=self.outsider_fen_model(outsider)
        outsider_image_tensor=[self.outsider_fen_model(image_fragments[i]) for i in range(len(image_fragments)) ]
        outsider_tensor=torch.cat([self.outsider_contrast_fc(torch.cat([outsider_input,outsider_image_tensor[i]],dim=-1)) for i in range(len(image_fragments))],dim=-1)
        outsider_tensor=self.outsider_fc(outsider_tensor)
        feature_tensor=torch.cat([image_input,outsider_tensor],dim=-1)
        out=self.fc1(feature_tensor)
        out=self.relu(out)
        out=self.bn(out)
        out=self.dropout(out)
        out=self.fc2(out)
        return out

class Decider_model(nn.Module):
    def __init__(self, fen_model_hidden1,fen_model_hidden2,outsider_hidden,hidden_1,hidden_2,action_num,dropout=0.1):
        super().__init__()
        self.fen_model=fen_model(fen_model_hidden1,fen_model_hidden2)
        self.outsider_fen=efficientnet_b0()
        self.outsider_fen.classifier=nn.Linear(1280,outsider_hidden)
        self.fc1=nn.Linear(2*fen_model_hidden2+outsider_hidden,hidden_1)
        self.relu=nn.ReLU()
        self.bn1=nn.BatchNorm1d(hidden_1)
        self.do=nn.Dropout(dropout)
        self.fc2=nn.Linear(hidden_1,hidden_2)
        self.bn2=nn.BatchNorm1d(hidden_2)
        self.out_layer=nn.Linear(hidden_2,action_num)
    
    def forward(self,cur_image,other_image,outsider):
        cur_image_tensor=self.fen_model(cur_image)
        other_image_tensor=self.fen_model(other_image)
        outsider_tensor=self.outsider_fen(outsider)
        feature_tensor=torch.cat([cur_image_tensor,other_image_tensor,outsider_tensor],dim=-1)
        out=self.fc1(feature_tensor)
        out=self.relu(out)
        out=self.bn1(out)
        out=self.do(out)
        out=self.fc2(out)
        out=self.relu(out)
        out=self.bn2(out)
        out=self.out_layer(out)
        return out

class Local_switcher_model(nn.Module):
    def __init__(self,fen_model_hidden1,fen_model_hidden2,hidden1,hidden2,action_num,dropout=0.1):
        super().__init__()
        self.fen_model=fen_model(fen_model_hidden1,fen_model_hidden2)
        self.fc1=nn.Linear(fen_model_hidden2,hidden1)
        self.relu=nn.ReLU()
        self.bn1=nn.BatchNorm1d(hidden1)
        self.fc2=nn.Linear(hidden1,hidden2)
        self.do=nn.Dropout(dropout)
        self.bn2=nn.BatchNorm1d(hidden2)
        self.outlayer=nn.Linear(hidden2,action_num)
    
    def forward(self,image):
        feature_tensor=self.fen_model(image)
        out=self.fc1(feature_tensor)
        out=self.relu(out)
        out=self.bn1(out)
        out=self.do(out)
        out=self.fc2(out)
        out=self.relu(out)
        out=self.bn2(out)
        out=self.outlayer(out)
        
        return out



class env:
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
                 tau=0.01
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

    
    def load_image(self,image_num,id=[]):
        if id:
            image_index=id
        else:
            image_index=random.choices(range(0,self.sample_number),k=image_num)
        for i in range(len(image_index)):
            image_raw=self.image[image_index[i]]
            permutation_raw=self.label[image_index[i]]
            image_raw=torch.tensor(image_raw).permute(2,0,1).to(torch.float).to(DEVICE)
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
            self.permutation2piece[-1]=torch.zeros(3,96,96).to(DEVICE)
        
    def get_image(self,permutation,image_index):
        image=torch.zeros(3,288,288).to(DEVICE)
        final_permutation=copy.deepcopy(permutation)
        final_permutation.insert(9//2,image_index*9+9//2)
        for i in range(9):
            image[:,(0+i//3)*96:(1+i//3)*96,(0+i%3)*96:(1+i%3)*96]=self.permutation2piece[final_permutation[i]]
    
        outsider_piece=self.permutation2piece[permutation[-1]]
        
        return image.unsqueeze(0),outsider_piece.unsqueeze(0)
    
    def request_for_image(self,image_id,permutation,image_index):
        self.load_image(image_num=self.image_num,id=image_id)
        image,outsider=self.get_image(permutation=permutation,image_index=image_index)
        self.load_image(image_num=self.image_num,id=self.image_id)
        return image,outsider

    def get_reward(self,permutation_list):

        # permutation_list=permutation_list[::][:len(permutation_list[0])-1]#Comment if the buffer is not after the permutation
        permutation_copy=copy.deepcopy(permutation_list)
        for i in range(len(permutation_list)):
            permutation_copy[i].insert(self.piece_num//2,i*self.piece_num+self.piece_num//2)
        done_list=[0 for i in range(len(permutation_copy))]
        reward_list=[0 for i in range(len(permutation_copy))]
        edge_length=int(len(permutation_copy[0])**0.5)
        piece_num=len(permutation_copy[0])
        hori_set=[(i,i+1) for i in [j for j in range(piece_num) if j%edge_length!=edge_length-1 ]]
        vert_set=[(i,i+edge_length) for i in range(piece_num-edge_length)]
        
        for i in range(len(permutation_list)):
            for j in range(len(hori_set)):#Pair reward

                hori_pair_set=(permutation_copy[i][hori_set[j][0]],permutation_copy[i][hori_set[j][1]])
                vert_pair_set=(permutation_copy[i][vert_set[j][0]],permutation_copy[i][vert_set[j][1]])
                if (-1 not in hori_pair_set) and (hori_pair_set[0]%piece_num,hori_pair_set[1]%piece_num) in hori_set and (hori_pair_set[0]//piece_num==hori_pair_set[1]//piece_num):
                    reward_list[i]+=1*PAIR_WISE_REWARD
                if (-1 not in vert_pair_set) and (vert_pair_set[0]%piece_num,vert_pair_set[1]%piece_num) in vert_set and (vert_pair_set[0]//piece_num==vert_pair_set[1]//piece_num):
                    reward_list[i]+=1*PAIR_WISE_REWARD

            piece_range=[0 for j in range (len(permutation_list))]
            # print(piece_range)
        
            for j in range(piece_num):
                if permutation_copy[i][j]!=-1:
                    piece_range[permutation_copy[i][j]//piece_num]+=1
                    reward_list[i]+=(permutation_copy[i][j]%piece_num==j and permutation_copy[i][j]//piece_num==i)*CATE_REWARD#Category reward
            
            
            max_piece=piece_range[i]#Consistancy reward

            weight=0.2
            if -1 in permutation_copy[i]:
                weight+=0.5*CONSISTENCY_REWARD
            if max_piece==piece_num-3:
                weight+=1*CONSISTENCY_REWARD
            elif max_piece==piece_num-2:
                weight+=2*CONSISTENCY_REWARD
            elif max_piece==piece_num-1:
                weight+=3*CONSISTENCY_REWARD
            elif max_piece==piece_num:
                weight+=5*CONSISTENCY_REWARD

            reward_list[i]*=weight
            reward_list[i]+=PANELTY
            start_index=min(permutation_copy[i])//piece_num*piece_num#Done reward
            if permutation_copy[i]==list(range(start_index,start_index+piece_num)):
                done_list[i]=True
                reward_list[i]=DONE_REWARD
        return reward_list,done_list
        #Change after determined


    def show_image(self,image_permutation_list):
        for i in range(self.image_num):

            image,_=self.get_image(permutation=image_permutation_list[i],image_index=i)
            image=image.squeeze().to("cpu")
            image=image.permute([1,2,0]).numpy().astype(np.uint8)
            cv2.imshow(f"Final image {i}",image)
        cv2.waitKey(1)

    
    def permute(self,cur_permutation,action_index):
        new_permutation=copy.deepcopy(cur_permutation)
        if action_index==(self.piece_num+1)*self.piece_num//2:
            return new_permutation
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
            action_index=random.randint(0,len(initial_permutation)*(len(initial_permutation)-1)//2-1)
            initial_permutation=self.permute(initial_permutation,action_index)
        print(f"Initial permutation {initial_permutation}")
        return initial_permutation


class Decider:
    def __init__(self,
                 memory_size,
                 model,
                 env,
                 action_num,
                 batch_size,
                 train_epoch,
                 tau=1e-3
                 ):
        self.memory=[]
        self.memory_size=memory_size
        self.memory_counter=0
        self.trace_start_point=0
        self.model=model
        self.main_model=copy.deepcopy(self.model)
        self.optimizer=torch.optim.Adam(self.main_model.parameters(),lr=CRITIC_LR,eps=1e-8)

        self.schedular=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=self.optimizer,
            T_0=10,
            T_mult=1,
            eta_min=1e-6,
            last_epoch=-1
        )

        self.tau=tau
        self.action_num=action_num
        self.env=env
        self.batch_size=batch_size

        
        self.epochs=train_epoch
    
    def recording_memory(self,
                         image_id,
                         image_index,
                         other_image_index,
                         state
                         ,other_state
                         ,action
                         ,reward
                         ,next_state
                         ,next_other_state
                         ,done):
        # print("Recording decider")
        memory={"Image_id":image_id,"Image_index":image_index,"Other_image_index":other_image_index,"State":state,"Other_state":other_state,"Action": action,"Reward":reward,"Next_state":next_state,"Next_other_state":next_other_state,"Done": done}
        if len(self.memory)<self.memory_size:
            self.memory.append(memory)
        else:
            self.memory[self.memory_counter]=memory
        self.memory_counter=(self.memory_counter+1)%self.memory_size

    def clean_memory(self):
        self.memory=[]
        self.memory_counter=0

    def epsilon_greedy(self,action):
        prob=random.random()
        if prob>self.env.epsilon:
            return action
        else:
            epsilon_action=(action+random.randint(1,self.action_num))%self.action_num
            return epsilon_action
        
    def act(self,current_image,other_image,outsider_piece):
        with torch.no_grad():
            self.model.eval()
            action=torch.argmax(self.model(current_image,other_image,outsider_piece),dim=-1)
            action=self.epsilon_greedy(action=action.item())

        return action

    def update(self,show=False):
        order=list(range(len(self.memory)))
        random.shuffle(order)
        self.model.train()
        self.main_model.train()
        loss_sum=[]

        for i in range(self.epochs):
            if i*self.batch_size>=len(order):
                break
            if len(order)-i*self.batch_size<self.batch_size:
                sample_dicts=[self.memory[j] for j in order[i*self.batch_size:]]
            else:
                sample_dicts=[self.memory[j] for j in order[i*self.batch_size:(i+1)*self.batch_size]]
            
            states=[]
            outsider_pieces=[]
            other_images=[]
            actions=[]
            next_states=[]
            next_outsiders=[]
            next_other_images=[]
            reward=[]
            done=[]

            
            for a in range(len(sample_dicts)):


                current_image,current_outsider=self.env.request_for_image(image_id=sample_dicts[a]["Image_id"],permutation=sample_dicts[a]["State"],image_index=sample_dicts[a]["Image_index"])
                other_image,_=self.env.request_for_image(image_id=sample_dicts[a]["Image_id"],permutation=sample_dicts[a]["Other_state"],image_index=sample_dicts[a]["Other_image_index"])
                states.append(current_image)
                outsider_pieces.append(current_outsider)
                other_images.append(other_image)


                actions.append(sample_dicts[a]["Action"])
                next_image,next_outsider_piece=self.env.request_for_image(image_id=sample_dicts[a]["Image_id"],permutation=sample_dicts[a]["Next_state"],image_index=sample_dicts[a]["Image_index"])
                next_other_image,_=self.env.request_for_image(image_id=sample_dicts[a]["Image_id"],permutation=sample_dicts[a]["Next_other_state"],image_index=sample_dicts[a]["Other_image_index"])
                next_states.append(next_image)
                next_outsiders.append(next_outsider_piece)
                next_other_images.append(next_other_image)
                
                reward.append(sample_dicts[a]["Reward"])
                done.append(sample_dicts[a]["Done"])


                    
            
            state_tensor=torch.cat(states,dim=0)
            if state_tensor.size(0)==1:
                self.model.eval()
                self.main_model.eval()
            else:
                self.model.train()
                self.main_model.train()

            outsider_tensor=torch.cat(outsider_pieces,dim=0)
            other_image_tensor=torch.cat(other_images,dim=0)

            next_state_tensor=torch.cat(next_states,dim=0)
            next_outsiders_tensor=torch.cat(next_outsiders,dim=0)
            next_other_image_tensor=torch.cat(next_other_images,dim=0)


            
            action_tensor=torch.tensor(actions).to(DEVICE).unsqueeze(-1)


            reward=torch.tensor(reward,dtype=torch.float32).to(DEVICE).unsqueeze(-1)
            done=torch.tensor(done,dtype=torch.float32).to(DEVICE).unsqueeze(-1)

            q_main=self.main_model(state_tensor,other_image_tensor,outsider_tensor).gather(1,action_tensor)
            
            q_next=self.model(next_state_tensor,next_other_image_tensor,next_outsiders_tensor).max(1)[0].unsqueeze(-1).detach()
            q_target=reward+self.env.gamma*q_next*(1-done)

            loss=nn.MSELoss()(q_main,q_target)
            loss_sum.append(loss.item())

            self.optimizer.zero_grad()
            
            loss.backward()

            # print(f"Decider_actor: {has_any_gradient(self.actor_model)}")

            torch.nn.utils.clip_grad_norm_(self.main_model.parameters(), CLIP_GRAD_NORM)
            self.optimizer.step()
            self.schedular.step() 
            

            for target_param, main_param in zip(self.model.parameters(),self.main_model.parameters()):
                target_param.data.copy_(self.tau*main_param.data+(1-self.tau)*target_param.data)

        if show and len(loss_sum)>0 : print(f"Decider loss: {np.mean(loss_sum)}.")
            

        

class Local_switcher:
    def __init__(self,
                 model,
                 memory_size,
                 gamma,
                 batch_size,
                 env,
                 action_num,
                 tau=1e-3):
        
        self.model=model
        self.main_model=copy.deepcopy(self.model)
        self.optimizer=torch.optim.Adam(self.main_model.parameters(),lr=ACTOR_LR,eps=1e-8)
        self.schedular=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=ACTOR_SCHEDULAR_STEP)
        self.memory_size=memory_size
        self.memory=[]
        self.memory_counter=0
        self.gamma=gamma
        self.batch_size=batch_size
        self.env=env
        self.action_num=action_num
        self.tau=tau
    
    def epsilon_greedy(self,action):
        prob=random.random()
        if prob>self.env.epsilon:
            return action
        else:
            epsilon_action=(action+random.randint(1,self.action_num))%self.action_num
            return epsilon_action
    
    def recording_memory(self,
                         image_id,
                         image_index,
                         state,
                         action
                         ,reward
                         ,next_state
                         ,done):
        # print("Recording local switcher")
        memory={"Image_id":image_id,"Image_index":image_index,"State":state,"Action": action,"Reward":reward,"Next_state":next_state,"Done": done}
        if len(self.memory)<self.memory_size:
            self.memory.append(memory)
        else:
            self.memory[self.memory_counter]=memory
        self.memory_counter=(self.memory_counter+1)%self.memory_size

    def clean_memory(self):
        self.memory=[]
        self.memory_counter=0
    
    def permute(self,cur_permutation,action_index):
        new_permutation=copy.deepcopy(cur_permutation)
        action=list(itertools.combinations(list(range(len(cur_permutation))), 2))[action_index]
        value0=cur_permutation[action[0]]
        value1=cur_permutation[action[1]]
        new_permutation[action[0]]=value1
        new_permutation[action[1]]=value0
        return new_permutation

    def choose_action(self,permutation,image_index):
        perm_=copy.deepcopy(permutation)
        value_list=[]
        image_list=[]
        outsider_list=[]

        for i in range(self.action_num):
            perm_=self.permute(permutation,i)
            image,_=self.env.get_image(perm_,image_index=image_index)
            image_list.append(copy.deepcopy(image.cpu()))
            # outsider_list.append(copy.deepcopy(outsider.cpu()))
        
        i=0
        with torch.no_grad():
            while i < self.action_num:
                if self.action_num-i<self.batch_size:
                    image=torch.cat(image_list[i:],dim=0).to(DEVICE)
                    # outsider=torch.cat(outsider_list[i:],dim=0).to(DEVICE)
                else:
                    image=torch.cat(image_list[i:i+self.batch_size],dim=0).to(DEVICE)
                    # outsider=torch.cat(outsider_list[i:i+BATCH_SIZE],dim=0).to(DEVICE)

                value=self.model(image)
                value_list.append(value.squeeze(-1).to("cpu"))
                i+=self.batch_size
            
            value_list=torch.cat(value_list)
            best_action=torch.argmax(value_list).item()
        return int(best_action)

    def update(self,show=False):
        # print("Updating local switcher")
        self.model.train()
        self.main_model.train()
        #select batch_size sample
        order=list(range(len(self.memory)))
        random.shuffle(order)
        loss_sum=[]
        for i in range(self.env.epochs):
            if i*self.batch_size>=len(order):
                break
            if len(order)-i*self.batch_size<self.batch_size:
                sample_dicts=[self.memory[j] for j in order[i*self.batch_size:]]
            else:
                sample_dicts=[self.memory[j] for j in order[i*self.batch_size:(i+1)*self.batch_size]]
            states=[]
            outsider_pieces=[]

            actions=[]
            next_states=[]
            next_outsiders=[]
            reward=[]
            for a in range(len(sample_dicts)):

                current_image,current_outsider=self.env.request_for_image(image_id=sample_dicts[a]["Image_id"],permutation=sample_dicts[a]["State"],image_index=sample_dicts[a]["Image_index"])
                states.append(current_image)
                outsider_pieces.append(current_outsider)

                actions.append(sample_dicts[a]["Action"])
                next_image,next_outsider_piece=self.env.request_for_image(image_id=sample_dicts[a]["Image_id"],permutation=sample_dicts[a]["Next_state"],image_index=sample_dicts[a]["Image_index"])
                next_states.append(next_image)
                next_outsiders.append(next_outsider_piece)
                
                reward.append(sample_dicts[a]["Reward"])
            

            if len(states)==1:
                self.model.eval()
                self.main_model.eval()
            else:
                self.model.train()
                self.main_model.train()
            state_tensor=torch.cat(states,dim=0)
            outsider_tensor=torch.cat(outsider_pieces,dim=0)
            next_state_tensor=torch.cat(next_states,dim=0)
            next_outsiders_tensor=torch.cat(next_outsiders,dim=0)

            q_next=self.model(next_state_tensor).detach()
            
            q_eval=self.main_model(next_state_tensor)

            reward=torch.tensor(reward,dtype=torch.float32).to(DEVICE).unsqueeze(-1)

            q_target=reward+self.gamma*q_next
            q_target=q_target.to(torch.float)

            loss=nn.MSELoss()(q_target,q_eval)
            self.optimizer.zero_grad()
            loss.float().backward()
            self.optimizer.step()
            
            loss_sum.append(loss.item())

        # print(f"Local_switcher: {has_any_gradient(self.main_model)}")

        for target_param, main_param in zip(self.model.parameters(),self.main_model.parameters()):
            target_param.data.copy_(self.tau*main_param.data+(1-self.tau)*target_param.data)
        self.schedular.step()
        if show and len(loss_sum)>0:
            print(f"Local switcher loss: {np.mean(loss_sum)}")

    def act(self,permutation,image_index):
        self.model.eval()
        action=self.choose_action(permutation=permutation,image_index=image_index)
        action=self.epsilon_greedy(action)
        permutation_=self.permute(cur_permutation=permutation,action_index=action)
        return permutation_,action

    
        

class Buffer_switcher:
    def __init__(self,
                 memory_size,
                 model,
                 env,
                 action_num,
                 batch_size,
                 train_epoch,
                 tau=1e-3):
        
        self.model=model
        self.main_model=copy.deepcopy(model)
        self.optimizer=torch.optim.Adam(self.main_model.parameters(),lr=ACTOR_LR,eps=1e-8)
        self.schedular=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=self.optimizer,
            T_0=10,
            T_mult=1,
            eta_min=1e-6,
            last_epoch=-1
        )


        self.memory_size=memory_size
        self.memory=[]
        self.memory_counter=0
        self.trace_start_point=0
        self.batch_size=batch_size
        self.env=env
        self.action_num=action_num
        self.epochs=train_epoch
        self.tau=tau

    def epsilon_greedy(self,action):
        prob=random.random()
        if prob>self.env.epsilon:
            return action
        else:
            epsilon_action=(action+random.randint(1,self.action_num))%self.action_num
            return epsilon_action
    
    def recording_memory(self
                         ,image_id
                         ,image_index
                         ,state
                         ,action
                         ,reward
                         ,next_state
                         ,done):


        memory={"Image_id":image_id,"Image_index":image_index,"State":state,"Action": action,"Reward":reward,"Next_state":next_state,"Done": done}
        # print("Recording buffer_switcher")
        if len(self.memory)<self.memory_size:
            self.memory.append(memory)
        else:
            self.memory[self.memory_counter]=memory
        self.memory_counter=(self.memory_counter+1)%self.memory_size


    def clean_memory(self):
        self.memory=[]
        self.memory_counter=0

    def act(self,image,outsider,permutation):
        self.model.eval()
        with torch.no_grad():
            action=torch.argmax(self.model(image,outsider),dim=-1)
            action=self.epsilon_greedy(action=action.item())
            swap_index,outsider_index=permutation[action],permutation[-1]
            permutation[action],permutation[-1]=outsider_index,swap_index
        return permutation,action

    def update(self,show=False):
        # print("Updating buffer switcher")
        order=list(range(len(self.memory)))
        random.shuffle(order)
        self.main_model.train()
        self.model.train()
        actor_loss_sum=[]
        for i in range(self.epochs):
            if i*self.batch_size>=len(order):
                break
            if len(order)-i*self.batch_size<self.batch_size:
                sample_dicts=[self.memory[j] for j in order[i*self.batch_size:]]
            else:
                sample_dicts=[self.memory[j] for j in order[i*self.batch_size:(i+1)*self.batch_size]]
            
            states=[]
            outsider_pieces=[]
            actions=[]
            next_states=[]
            next_outsiders=[]
            reward=[]
            done=[]

            
            for a in range(len(sample_dicts)):

                current_image,current_outsider=self.env.request_for_image(image_id=sample_dicts[a]["Image_id"],permutation=sample_dicts[a]["State"],image_index=sample_dicts[a]["Image_index"])
                states.append(current_image)
                outsider_pieces.append(current_outsider)

                actions.append(sample_dicts[a]["Action"])
                next_image,next_outsider_piece=self.env.request_for_image(image_id=sample_dicts[a]["Image_id"],permutation=sample_dicts[a]["Next_state"],image_index=sample_dicts[a]["Image_index"])
                next_states.append(next_image)
                next_outsiders.append(next_outsider_piece)
                
                reward.append(sample_dicts[a]["Reward"])
                done.append(sample_dicts[a]["Done"])


                    
            
            state_tensor=torch.cat(states,dim=0)
            if state_tensor.size(0)==1:

                self.model.eval()
                self.main_model.eval()
            else:

                self.model.train()
                self.main_model.train()

            outsider_tensor=torch.cat(outsider_pieces,dim=0)

            next_state_tensor=torch.cat(next_states,dim=0)
            next_outsiders_tensor=torch.cat(next_outsiders,dim=0)
            
            action_tensor=torch.tensor(actions).to(DEVICE).unsqueeze(-1)

            reward=torch.tensor(reward,dtype=torch.float32).to(DEVICE).unsqueeze(-1)
            done=torch.tensor(done,dtype=torch.float32).to(DEVICE).unsqueeze(-1) 

            q_main=self.main_model(state_tensor,outsider_tensor).gather(1,action_tensor)
            
            q_next=self.model(next_state_tensor,next_outsiders_tensor).max(1)[0].unsqueeze(-1).detach()
            q_target=reward+self.env.gamma*q_next*(1-done)
            
            loss=nn.MSELoss()(q_main,q_target)
            actor_loss_sum.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()


            



            torch.nn.utils.clip_grad_norm_(self.main_model.parameters(), CLIP_GRAD_NORM)
            self.optimizer.step()
            self.schedular.step()



            for target_param, main_param in zip(self.model.parameters(),self.main_model.parameters()):
                target_param.data.copy_(self.tau*main_param.data+(1-self.tau)*target_param.data)


            
        # print(f"Buffer_switcher: {has_any_gradient(self.main_model)}")
            
        if show and len(actor_loss_sum)>0: print(f"Buffer switcher loss: {np.mean(actor_loss_sum)}")
        
        
        

        
def update(decider,local_switcher,buffer_switcher,show=False):
    # print("Updating")
    
    decider.update(show)
    local_switcher.update(show)
    buffer_switcher.update(show)

def clean_memory(decider,local_switcher,buffer_switcher):
    decider.clean_memory()
    local_switcher.clean_memory()
    buffer_switcher.clean_memory()
    
def save(decider,local_switcher,buffer_switcher):
    torch.save(decider.model.state_dict(),os.path.join("Decider"+MODEL_NAME))

    torch.save(buffer_switcher.model.state_dict(),os.path.join("Buffer_switcher"+MODEL_NAME))
    torch.save(local_switcher.model.state_dict(),os.path.join("Local_switcher"+MODEL_NAME))

def load(decider,local_switcher,buffer_switcher):
    decider.model.load_state_dict(torch.load(os.path.join("Decider"+MODEL_NAME)))
    decider.main_model=copy.deepcopy(decider.model)
    buffer_switcher.model.load_state_dict(torch.load(os.path.join("Buffer_switcher"+MODEL_NAME)))
    buffer_switcher.main_model=copy.deepcopy(buffer_switcher.model)
    local_switcher.model.load_state_dict(torch.load(os.path.join("Local_switcher"+MODEL_NAME)))
    local_switcher.main_model=copy.deepcopy(local_switcher.model)


def run_maze(env,decider,buffer_switcher,local_switcher,load_flag=True,epoch_num=500):
    if load_flag:
        load(decider=decider,local_switcher=local_switcher,buffer_switcher=buffer_switcher)

    for i in range(epoch_num):
        if i > 300:
                max_step, swap_num = MAX_STEP[3], SWAP_NUM[3]
        elif i > 200:
                max_step, swap_num = MAX_STEP[2], SWAP_NUM[2]
        elif i > 100:
                max_step, swap_num = MAX_STEP[1], SWAP_NUM[1]
        else:
                max_step, swap_num = MAX_STEP[0], SWAP_NUM[0]
        
        initial_perm=env.summon_permutation_list(swap_num)
        permutation_list = [initial_perm[j*(env.piece_num-1):(j+1)*(env.piece_num-1)]
                                for j in range(env.image_num)]
        buffer=[-1]*env.buffer_size
        done_list=[False] * env.image_num
        reward_sum_list= [[] for _ in range(env.image_num)]
        termination_list=[0 for _ in range(env.image_num)]



        last_actions=[-1 for _ in range(env.image_num)]

        step=0; done = False ; 
        clean_memory(decider,local_switcher,buffer_switcher)
        pending_transitions_decider = {j: None for j in range(env.image_num)}
        pending_transitions_local_switcher= {j: None for j in range(env.image_num)}
        pending_transitions_buffer_switcher={j: None for j in range(env.image_num)}
        while not done and step < max_step:
            state_list=[]
            do_list=[]
            model_action=[0 for j in range(env.image_num)] 
            last_reward_list,_=env.get_reward(permutation_list)
            
            
            
            for j in range(env.image_num):
                if done_list[j]:
                    continue
                elif termination_list[j]>=40:
                    termination_list[j]=0
                perm_with_buf=permutation_list[j]+buffer
                do_list.append(j)
                if pending_transitions_decider[j] is not None:
                    prev_state, prev_other_state,prev_action,prev_reward = pending_transitions_decider[j]
                    decider.recording_memory(image_id=env.image_id
                                                        ,image_index=j,
                                                        other_image_index=(j+1)%env.image_num
                                                        ,state=prev_state
                                                        ,other_state=prev_other_state
                                                        ,action= prev_action
                                                        ,reward= prev_reward
                                                        ,next_state= perm_with_buf
                                                        ,next_other_state=permutation_list[(j+1)%env.image_num]
                                                        , done=done_list[j])
                    pending_transitions_decider[j]=None

                if pending_transitions_local_switcher[j] is not None:
                    # print(pending_transitions_local_switcher[j])
                    prev_state, prev_action, prev_next_state,prev_reward ,prev_done= pending_transitions_local_switcher[j]
                    local_switcher.recording_memory(image_id=env.image_id
                                                    ,image_index=j
                                                    ,state=prev_state
                                                    ,action= prev_action
                                                    ,reward= prev_reward
                                                    ,next_state= prev_next_state
                                                    , done=prev_done)
                    pending_transitions_local_switcher[j]=None

                if pending_transitions_buffer_switcher[j] is not None:
                    state, action, reward,done = pending_transitions_buffer_switcher[j]
                    buffer_switcher.recording_memory(image_id=env.image_id
                                                        ,image_index=j
                                                        ,state=state
                                                        ,action= action
                                                        ,reward= reward
                                                        ,next_state= perm_with_buf
                                                        , done=done)
                    pending_transitions_buffer_switcher[j]=None

                
                
                state_list.append(copy.deepcopy(perm_with_buf))
                image,outsider=env.get_image(perm_with_buf,image_index=j)
                other_image,_=env.get_image(permutation_list[(j+1)%env.image_num],image_index=(j+1)%env.image_num)
                decider_action=decider.act(current_image=image,outsider_piece=outsider,other_image=other_image)
                pending_transitions_decider[j]=(permutation_list[j],permutation_list[(j+1)%env.image_num],decider_action)
                
                model_action[j]=decider_action

                if decider_action:

                    perm_with_buf_,action=buffer_switcher.act(image,outsider,perm_with_buf)
                    permutation_list[j]=copy.deepcopy(perm_with_buf[:len(perm_with_buf_)-env.buffer_size])
                    buffer=copy.deepcopy(perm_with_buf_[len(perm_with_buf_)-env.buffer_size:])
                    pending_transitions_buffer_switcher[j]=(perm_with_buf,action)
                    
                
                
                else:
                    permutation_,action=local_switcher.act(permutation=permutation_list[j],image_index=j)
                    pending_transitions_local_switcher[j]=(permutation_list[j],action,permutation_)
                    permutation_list[j]=copy.deepcopy(permutation_)
                    
                
            reward_list, done_list=env.get_reward(permutation_list)

            if SHOW_IMAGE:
                env.show_image(permutation_list)
            
            for j in do_list:
                reward_sum_list[j].append(reward_list[j])
                
                prev_state, prev_other_state,prev_action, = pending_transitions_decider[j]
                pending_transitions_decider[j]=(prev_state, prev_other_state,prev_action,reward_list[j])

                if model_action[j]:
                    state,action=pending_transitions_buffer_switcher[j]
                    pending_transitions_buffer_switcher[j]=(state,action,reward_list[j],done_list[j])
                else:
                    state,action,next_state=pending_transitions_local_switcher[j]
                    pending_transitions_local_switcher[j]=(state,action,next_state,reward_list[j],done_list[j])
            
            done=all(done_list)
            step+=1
            # print(f"Decider: {pending_transitions_decider}\n Local_switcher: {pending_transitions_local_switcher}\n Buffer_switcher: {pending_transitions_buffer_switcher}")

            if step%TRAIN_PER_STEP==0:
                update(
                    decider=decider,
                    local_switcher=local_switcher,
                    buffer_switcher=buffer_switcher,
                    show=False
                )  
                env.load_image(image_num=env.image_num, id=env.image_id)
                    
        print(f"Epoch: {i}, step: {step}, reward: {[sum(reward_sum_list[j])/len(reward_sum_list[j]) for j in range(len(reward_sum_list)) if len(reward_sum_list[j])!=0 ]}")
        # print(f"Action_list: {self.action_list}")
        print(f"Permutation list: {permutation_list}")
        if env.epsilon>EPSILON_MIN:
            env.epsilon*=EPSILON_GAMMA
        update(
            decider=decider,
            local_switcher=local_switcher,
            buffer_switcher=buffer_switcher,
            show=True
        )  
        save(
            decider=decider,
            local_switcher=local_switcher,
            buffer_switcher=buffer_switcher
        )
            
                



if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    environment=env(train_x=train_x,
                    train_y=train_y,
                    gamma=GAMMA,
                    image_num=2,
                    buffer_size=1,
                    epsilon=EPSILON,
                    epsilon_gamma=EPSILON_GAMMA)
    # critic=critic_model(hidden_size1=1024,hidden_size2=1024,outsider_hidden_size=256).to(device=DEVICE)
    decider_actor=Decider_model(
        fen_model_hidden1=1024,
        fen_model_hidden2=512,
        outsider_hidden=512,
        hidden_1=1024,
        hidden_2=512,
        action_num=2
    ).to(DEVICE)
    
    decider=Decider(memory_size=2000,
                    model=decider_actor,
                    env=environment,
                    action_num=2,
                    batch_size=BATCH_SIZE,
                    train_epoch=AGENT_EPOCHS
                    )
    
    buffer_switcher_model=Buffer_switcher_model(
        hidden_size1=2048,
        hidden_size2=1024,
        outsider_hidden_size=1024,
        action_num=8
    ).to(DEVICE)
    buffer_switcher_model.load_state_dict(torch.load("outsider_switcher_pretrain.pth"))
    buffer_switcher=Buffer_switcher(
        memory_size=2000,
        model=buffer_switcher_model,
        action_num=8,
        batch_size=BATCH_SIZE,
        train_epoch=AGENT_EPOCHS,
        env=environment
    )

    local_switcher_model=Local_switcher_model(fen_model_hidden1=2048,
                                              fen_model_hidden2=1024,
                                              hidden1=2048,
                                              hidden2=1024,
                                              action_num=1).to(DEVICE)
    local_switcher_model.load_state_dict(torch.load("sd2rl_pretrain.pth"))
    local_switcher=Local_switcher(
        memory_size=2000,
        gamma=environment.gamma,
        batch_size=BATCH_SIZE,
        action_num=28,
        env=environment,
        model=local_switcher_model
    )
    # pretrain_model_dict=pretrain_model(256,256)
    # pretrain_model_dict.load_state_dict(torch.load("pairwise_pretrain.pth"))
    # selective_load_state_dict(pretrain_model_dict,critic,{"ef":"fen_model.ef"})
    # selective_load_state_dict(pretrain_model_dict,critic,{"contrast_fc_hori":"fen_model.contrast_fc_hori"})
    # selective_load_state_dict(pretrain_model_dict,critic,{"contrast_fc_vert":"fen_model.contrast_fc_vert"})
    print(f"Device: {DEVICE}")
    run_maze(env=environment,
        decider=decider,
             local_switcher=local_switcher
             ,buffer_switcher=buffer_switcher
             ,epoch_num=EPOCH_NUM
             ,load_flag=LOAD_MODEL)
    
