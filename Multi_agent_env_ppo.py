import torch
import torch.nn as nn
import numpy as np
import random
import copy
import itertools
from outsider_pretrain import fen_model
from torchvision.models import efficientnet_b0
from torch.utils.data import Dataset,DataLoader
import cv2
import Vit
import os
import time

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DONE_REWARD=20
GAMMA=0.998
CLIP_GRAD_NORM=0.1
TRAIN_PER_STEP=8
ACTOR_LR=1e-5
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
SWAP_NUM=[4,8,8,8]
MAX_STEP=[400,400,400,400]
MODEL_NAME="(7).pth"
ACTOR_PATH=os.path.join("Actor"+MODEL_NAME)
CRITIC_PATH=os.path.join("Critic"+MODEL_NAME)

BATCH_SIZE=18
EPSILON=0.3
EPSILON_GAMMA=0.995
EPSILON_MIN=0.1
AGENT_EPOCHS=10

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

class fen_model(nn.Module):
    def __init__(self,hidden_size1,hidden_size2):
        super(fen_model,self).__init__()
        self.ef=efficientnet_b0(weights="DEFAULT")
        self.ef.classifier=nn.Linear(1280,128)
        
        self.fc1=nn.Linear(128*24,hidden_size1)
        self.relu=nn.ReLU()
        self.bn=nn.BatchNorm1d(hidden_size1)
        self.do=nn.Dropout1d(p=0.1)
        self.fc2=nn.Linear(hidden_size1,hidden_size2)
        self.hori_set=[(i,i+1) for i in [j for j in range(9) if j%3!=3-1 ]]
        self.vert_set=[(i,i+3) for i in range(3*2)]
    
    def forward(self,image):
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
        
        hori_tensor=torch.cat([torch.cat([self.ef(image_fragments[self.hori_set[i][0]]),self.ef(image_fragments[self.hori_set[i][1]])],dim=-1) for i in range(len(self.hori_set))],dim=-1)
        vert_tensor=torch.cat([torch.cat([self.ef(image_fragments[self.vert_set[i][0]]),self.ef(image_fragments[self.vert_set[i][1]])],dim=-1) for i in range(len(self.vert_set))],dim=-1)
        feature_tensor=torch.cat([hori_tensor,vert_tensor],dim=-1)
        x=self.do(feature_tensor)
        x=self.fc1(x)
        x=self.do(x)
        # x=self.bn(x)
        x=self.relu(x)
        x=self.fc2(x)
        return x




class actor_model(nn.Module):
    def __init__(self,hidden_size1,hidden_size2,outsider_hidden_size,action_num):
        super(actor_model,self).__init__()
        self.image_fen_model=fen_model(hidden_size1,hidden_size1)
        # state_dict=torch.load("pairwise_pretrain.pth")
        # state_dict_replace = {
        # k: v 
        # for k, v in state_dict.items() 
        # if k.startswith("ef.")
        # }
        # load_result_hori=self.image_fen_model.load_state_dict(state_dict_replace,strict=False)
        # print("Actor missing keys hori",load_result_hori.missing_keys)
        # print("Actor unexpected keys hori",load_result_hori.unexpected_keys)
        self.outsider_fen_model=efficientnet_b0(weights="DEFAULT")
        self.outsider_fen_model.classifier=nn.Linear(1280,outsider_hidden_size)
        self.fc1=nn.Linear(hidden_size1+outsider_hidden_size,hidden_size2)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=0.1)
        self.fc2=nn.Linear(hidden_size2,action_num)
    
    def forward(self,image,outsider_piece):
        image_input=self.image_fen_model(image)
        outsider_input=self.outsider_fen_model(outsider_piece)
        feature_tensor=torch.cat([image_input,outsider_input],dim=1)
        out=self.fc1(feature_tensor)
        out=self.relu(out)
        out=self.dropout(out)
        out=self.fc2(out)
        out=nn.functional.softmax(out,dim=1)
        return out

class critic_model(nn.Module):
    def __init__(self,hidden_size1,hidden_size2,outsider_hidden_size):
        super(critic_model,self).__init__()
        self.fen_model=fen_model(hidden_size1=hidden_size1,hidden_size2=hidden_size1)
        self.outsider_fen_model=efficientnet_b0(weights="DEFAULT")
        self.outsider_fen_model.classifier=nn.Linear(1280,outsider_hidden_size)
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
        self.fc=nn.Linear(hidden_size2,1)
    
    def forward(self,image,outsider):
        image_input=self.fen_model(image)
        outsider_input=self.outsider_fen_model(outsider)
        feature_tensor=torch.cat([image_input,outsider_input],dim=1)
        out=self.fc1(feature_tensor)
        out=self.relu(out)
        out=self.dropout(out)
        out=self.fc(out)
        return out

# class actor_model(nn.Module):
#     def __init__(self, 
#                  picture_size,
#                  outsider_size,
#                  patch_size,
#                  encoder_layer_num,
#                  n_head,
#                  transformer_output_size,
#                  unet_hidden,
#                  encoder_hidden,
#                  output_channel,
#                  actor_hidden,
#                  action_num,
#                  dropout=0.1):
#         super().__init__()
#         num_patches=(outsider_size[2]*outsider_size[3]//(patch_size**2)+picture_size[2]*picture_size[3]//(patch_size**2))
#         self.local_fen=Vit.UNet(input_channel=picture_size[1],output_channel=output_channel,hidden=unet_hidden)
#         self.image_embedding=Vit.PictureEmbedding(picture_size=picture_size,patch_size=patch_size)
#         self.outsider_embedding=Vit.PictureEmbedding(picture_size=outsider_size,patch_size=patch_size)
#         self.positional_embedding=Vit.PositionalEmbedding(d_model=output_channel*(patch_size**2),num_patches=num_patches)
#         self.encoder_layers=nn.ModuleList(
#             [
#                 Vit.EncoderLayer((patch_size**2)*picture_size[1],encoder_hidden,n_head)
#                 for _ in range(encoder_layer_num)
#             ]
#         )
#         self.transformer_fc=nn.Linear(output_channel*(patch_size**2),1)
#         self.fc1=nn.Linear(num_patches,transformer_output_size)
#         self.relu=nn.ReLU()
#         self.dropout=nn.Dropout(dropout)
#         self.fc2=nn.Linear(transformer_output_size,actor_hidden)
#         self.output=nn.Linear(actor_hidden,action_num)


#     def forward(self,image,outsider_piece):
#         image=self.local_fen(image)
#         outsider_piece=self.local_fen(outsider_piece)
#         image_tensor=self.image_embedding(image)
#         outsider_tensor=self.outsider_embedding(outsider_piece)
#         transformer_feature_tensor=torch.cat([image_tensor,outsider_tensor],dim=1)
#         transformer_feature_tensor=self.positional_embedding(transformer_feature_tensor)
#         for layer in self.encoder_layers:
#             transformer_feature_tensor=layer(transformer_feature_tensor)
#         feature_tensor=self.transformer_fc(transformer_feature_tensor)
#         feature_tensor=feature_tensor.squeeze()
#         x=self.fc1(feature_tensor)
#         x=self.relu(x)
#         x=self.dropout(x)
#         x=self.fc2(x)
#         x=self.relu(x)
#         x=self.dropout(x)
#         x=self.output(x)
#         output=nn.functional.softmax(x,dim=-1)
#         return output


# class critic_model(nn.Module):
#     def __init__(self,
#                  picture_size,
#                  outsider_size,
#                  patch_size,
#                  encoder_layer_num,
#                  n_head,
#                  transformer_out_size,
#                  output_channel,
#                  unet_hidden,
#                  encoder_hidden,
#                  critic_hidden,
#                  dropout=0.1):
#         super().__init__()
#         # self.local_fen=Vit.VisionTransformer(
#         #     picture_size=picture_size,
#         #     patch_size=patch_size,
#         #     encoder_layer_num=encoder_layer_num,
#         #     n_head=n_head,
#         #     out_size=transformer_out_size,
#         #     output_channel=output_channel,
#         #     unet_hidden=unet_hidden,
#         #     encoder_hidden=encoder_hidden,
#         #     dropout=dropout
#         # )
#         num_patches=(outsider_size[2]*outsider_size[3]//(patch_size**2)+picture_size[2]*picture_size[3]//(patch_size**2))
#         self.local_fen=Vit.UNet(input_channel=picture_size[1],output_channel=output_channel,hidden=unet_hidden)
#         self.image_embedding=Vit.PictureEmbedding(picture_size=picture_size,patch_size=patch_size)
#         self.outsider_embedding=Vit.PictureEmbedding(picture_size=outsider_size,patch_size=patch_size)
#         self.positional_embedding=Vit.PositionalEmbedding(d_model=output_channel*(patch_size**2),num_patches=num_patches)
#         self.encoder_layers=nn.ModuleList(
#             [
#                 Vit.EncoderLayer((patch_size**2)*picture_size[1],encoder_hidden,n_head)
#                 for _ in range(encoder_layer_num)
#             ]
#         )
#         self.transformer_fc=nn.Linear(output_channel*(patch_size**2),1)
#         self.fc=nn.Linear(num_patches,transformer_out_size)
#         # self.local_fen.local_fen=nn.Identity()
#         self.fc1=nn.Linear(transformer_out_size,critic_hidden)
#         self.relu=nn.ReLU()
#         self.dropout=nn.Dropout(dropout)
#         self.fc2=nn.Linear(critic_hidden,1)
#     def forward(self,image,outsider_piece):
#         # transformer_output=self.local_fen(image)
#         # image=self.local_fen(image)
#         # outsider_piece=self.local_fen(outsider_piece)
#         image_tensor=self.image_embedding(image)
#         outsider_tensor=self.outsider_embedding(outsider_piece)
#         transformer_feature_tensor=torch.cat([image_tensor,outsider_tensor],dim=1)
#         transformer_feature_tensor=self.positional_embedding(transformer_feature_tensor)
#         for layer in self.encoder_layers:
#             transformer_feature_tensor=layer(transformer_feature_tensor)
#         feature_tensor=self.transformer_fc(transformer_feature_tensor)
#         feature_tensor=feature_tensor.squeeze()
#         transformer_output=self.fc(feature_tensor)
#         x=self.fc1(transformer_output)
#         x=self.relu(x)
#         x=self.dropout(x)
#         out=self.fc2(x)
#         return out




class env:
    def __init__(self,
                 train_x,
                 train_y,
                 memory_size,
                 batch_size,
                 gamma,
                 device,
                 actor_model,#input: image,outsider_piece. Output: action index
                 critic_model,#input: image,outsider_piece. Output: Q value
                #  encoder,
                 image_num,
                 buffer_size,
                 entropy_weight,
                 epsilon,
                 epsilon_gamma,
                 piece_num=9,
                 epochs=10
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
        self.actor_model=actor_model
        self.critic_model=critic_model
        self.actor_optimizer=torch.optim.Adam(self.actor_model.parameters(),lr=ACTOR_LR)
        self.actor_schedular=torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=ACTOR_SCHEDULAR_STEP, gamma=0.1)
        self.critic_optimizer=torch.optim.Adam(self.critic_model.parameters(),lr=CRITIC_LR)
        self.critic_schedular=torch.optim.lr_scheduler.StepLR(optimizer=self.critic_optimizer,gamma=0.1,step_size=CRITIC_SCHEDULAR_STEP)
        self.device=device
        self.batch_size=batch_size
        self.gamma=gamma
        self.image_num=image_num
        self.buffer_size=buffer_size
        self.action_list=[[0 for _ in range((piece_num+buffer_size)*piece_num//2+1)] for __ in range(image_num)]
        self.piece_num=piece_num
        self.entropy_weight=entropy_weight
        self.epsilon=epsilon
        self.epsilon_gamma=epsilon_gamma
        self.epochs=epochs

    
    def load_image(self,image_num,id=[]):
        if id:
            image_index=id
        else:
            image_index=random.choices(range(0,self.sample_number),k=image_num)
        self.image_id=image_index
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
        
    def get_image(self,permutation,image_id):
        image=torch.zeros(3,288,288)
        final_permutation=copy.deepcopy(permutation)
        final_permutation.insert(9//2,image_id*9+9//2)
        for i in range(9):
            image[:,(0+i//3)*96:(1+i//3)*96,(0+i%3)*96:(1+i%3)*96]=self.permutation2piece[final_permutation[i]]
    
        outsider_piece=self.permutation2piece[permutation[-1]]
        return image.unsqueeze(0).to(self.device),outsider_piece.unsqueeze(0).to(self.device)

    def get_reward(self,permutation_list):

        # permutation_list=permutation_list[::][:len(permutation_list[0])-1]#Comment if the buffer is not after the permutation
        permutation_copy=copy.deepcopy(permutation_list)
        for i in range(len(permutation_list)):
            permutation_copy[i].insert(9//2,i*9+9//2)
        done_list=[0 for i in range(len(permutation_copy))]
        reward_list=[0 for i in range(len(permutation_copy))]
        edge_length=int(len(permutation_copy[0])**0.5)
        piece_num=len(permutation_copy[0])
        hori_set=[(i,i+1) for i in [j for j in range(piece_num) if j%edge_length!=edge_length-1 ]]
        vert_set=[(i,i+edge_length) for i in range(edge_length*2)]
        
        for i in range(len(permutation_list)):
            for j in range(len(hori_set)):#Pair reward
                hori_pair_set=(permutation_copy[i][hori_set[j][0]],permutation_copy[i][hori_set[j][1]])
                vert_pair_set=(permutation_copy[i][vert_set[j][0]],permutation_copy[i][vert_set[j][1]])
                if -1 not in hori_pair_set and (hori_pair_set[0]%piece_num,hori_pair_set[1]%piece_num) in hori_set:
                    reward_list[i]+=1*PAIR_WISE_REWARD
                if -1 not in vert_pair_set and (vert_pair_set[0]%piece_num,vert_pair_set[1]%piece_num) in vert_set:
                    reward_list[i]+=1*PAIR_WISE_REWARD

            piece_range=[0 for j in range (len(permutation_list))]
            # print(piece_range)
        
            for j in range(piece_num):
                if permutation_copy[i][j]!=-1:
                    piece_range[permutation_copy[i][j]//piece_num]+=1
                reward_list[i]+=(permutation_copy[i][j]%piece_num==j)*CATE_REWARD#Category reward
            
            
            max_piece=max(piece_range)#Consistancy reward

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
    

    def clean_memory(self):
        self.mkv_memory=[]
        self.memory_counter=0

    def show_image(self,image_permutation_list):
        for i in range(self.image_num):

            image,_=self.get_image(permutation=image_permutation_list[i],image_id=i)
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
            self.load_image(image_num=self.image_num,id=id)
        else:
            self.load_image(image_num=self.image_num)
        print(f"Episode image:{self.image_id}")
        initial_permutation=list(range(self.piece_num*self.image_num))
        for i in range(self.image_num):
            initial_permutation.pop(9*i+9//2-i)
        for i in range(swap_num):
            action_index=random.randint(0,len(initial_permutation)*(len(initial_permutation)-1)//2-1)
            initial_permutation=self.permute(initial_permutation,action_index)
        print(f"Initial permutation {initial_permutation}")
        return initial_permutation

    def recording_memory(self,image_id,state_list,log_probs,action_list,reward_list,next_state_list,do_list,done_list):
        memory={"Image_id":image_id,"State_list":state_list,"Log_probs":log_probs,"Action_list": action_list,"Reward_list":reward_list,"Next_state_list":next_state_list,"Do_list":do_list,"Done_list": done_list}
        if len(self.mkv_memory)<self.mkv_memory_size:
            self.mkv_memory.append(memory)
        else:
            self.mkv_memory[self.memory_counter]=memory
        self.memory_counter=(self.memory_counter+1)%self.mkv_memory_size

        
    def update(self):
        eps=0.2
        #Calculating return
        i=(self.memory_counter-1)%self.mkv_memory_size
        
        R=[0 for _ in range(self.image_num)]
        R_start=[True for _ in range(self.image_num)]
        R_track=[[] for _ in range(self.image_num)]
        self.critic_model.eval()
        while i!=(self.trace_start_point-1)%self.mkv_memory_size:
            do_list=self.mkv_memory[i]["Do_list"]
            done_list=self.mkv_memory[i]["Done_list"]
            reward_list=self.mkv_memory[i]["Reward_list"]
            for j in range(len(do_list)):
                if R_start[do_list[j]]:
                    R_start[do_list[j]]=False
                    current_image,current_outsider=self.get_image(self.mkv_memory[i]["State_list"][j],image_id=do_list[j])
                    value=self.critic_model(current_image,current_outsider)
                    R[do_list[j]]=value.item()
                else:
                    R[do_list[j]]=self.gamma*R[do_list[j]]*(1-done_list[j])+reward_list[j]
                R_track[do_list[j]].append(R[do_list[j]])
            self.mkv_memory[i]["Return_list"]=R.copy()
            i=(i-1)%self.mkv_memory_size
        i=(self.memory_counter-1)%self.mkv_memory_size
        R_track=[np.mean(R_track[j]) if R_track[j]!=[] else 1 for j in range(len(R_track)) ]
        while i!=(self.trace_start_point-1)%self.mkv_memory_size:
            self.mkv_memory[i]["Return_list"]=[self.mkv_memory[i]["Return_list"][j]/R_track[j] for j in range(self.image_num)]
            i=(i-1)%self.mkv_memory_size

        
        order=list(range(len(self.mkv_memory)))
        random.shuffle(order)
        self.actor_model.train()
        self.critic_model.train()
        critic_loss_sum=0
        for i in range(self.epochs):
            if i*self.batch_size>=len(order):
                break
            if len(order)-i*self.batch_size<self.batch_size:
                sample_dicts=[self.mkv_memory[j] for j in order[i*self.batch_size:]]
            else:
                sample_dicts=[self.mkv_memory[j] for j in order[i*self.batch_size:(i+1)*self.batch_size]]
            
            states=[]
            outsider_pieces=[]
            ret=[]
            actions=[]
            next_states=[]
            reward=[]
            done=[]
            old_log_probs=[]
            
            for a in range(len(sample_dicts)):
                self.load_image(image_num=self.image_num,id=self.mkv_memory[a]["Image_id"])
                do_list=self.mkv_memory[a]["Do_list"]
                for b in range(len(do_list)):
                    
                    current_image,current_outsider=self.get_image(self.mkv_memory[a]["State_list"][b],image_id=b)
                    states.append(current_image)
                    outsider_pieces.append(current_outsider)
                    old_log_probs.append(self.mkv_memory[a]["Log_probs"][b])
                    actions.append(self.mkv_memory[a]["Action_list"][b])
                    next_image,_=self.get_image(self.mkv_memory[a]["Next_state_list"][b],image_id=b)
                    next_states.append(next_image)
                    ret.append(self.mkv_memory[a]["Return_list"][do_list[b]])
                    reward.append(self.mkv_memory[a]["Reward_list"][b])
                    done.append(self.mkv_memory[a]["Done_list"][b])

                    
            
            state_tensor=torch.cat(states,dim=0)
            outsider_tensor=torch.cat(outsider_pieces,dim=0)

            probs=self.actor_model(state_tensor,outsider_tensor)
            
            action_tensor=torch.tensor(actions).to(self.device).unsqueeze(-1)
            selected_probs=probs.gather(1,action_tensor).clamp(min=1e-8)
            log_probs=torch.log(selected_probs)
            old_log_probs=torch.cat(old_log_probs,dim=0)
            entropy = torch.distributions.Categorical(probs).entropy()
            
            # next_state_tensor=torch.cat(next_states)
            # pred_next_ret=self.critic_model(next_state_tensor)
            pred_ret=self.critic_model(state_tensor,outsider_tensor)
            ret=torch.tensor(ret,dtype=torch.float32).to(self.device).unsqueeze(-1)
            reward=torch.tensor(reward,dtype=torch.float32).to(self.device).unsqueeze(-1)
            done=torch.tensor(done,dtype=torch.float32).to(self.device).unsqueeze(-1)

            advantage=ret-pred_ret.detach()


            ratio=torch.exp(log_probs-old_log_probs)
            actor_loss=-torch.min(ratio*advantage,torch.clamp(ratio,1-eps,1+eps)*advantage).mean()-entropy.mean()*self.entropy_weight
            critic_loss=nn.functional.mse_loss(pred_ret,ret)
            # critic_loss=nn.functional.mse_loss(pred_ret,(reward+self.gamma*pred_next_ret*(1-done)))
            
            critic_loss_sum+=critic_loss.item()

            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()

            critic_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), CLIP_GRAD_NORM)
            self.actor_optimizer.step()

            self.critic_optimizer.step()
        print(f"Critic loss: {critic_loss_sum*self.batch_size/len(self.mkv_memory)}")
        if self.entropy_weight>=ENTROPY_MIN:
            self.entropy_weight*=ENTROPY_GAMMA
        torch.save(self.critic_model.state_dict(), CRITIC_PATH)
        if self.critic_optimizer.state_dict()["param_groups"][0]["lr"]>CRITIC_LR_MIN:
            self.critic_schedular.step()
        torch.save(self.actor_model.state_dict(),ACTOR_PATH)
        if self.actor_optimizer.state_dict()["param_groups"][0]["lr"]>ACTOR_LR_MIN:
            self.actor_schedular.step()



    def step(self, epoch=500, load=True):

        
        if load:
            self.critic_model.load_state_dict(torch.load(CRITIC_PATH))
            self.actor_model.load_state_dict(torch.load(ACTOR_PATH))

        
        for i in range(epoch):
            self.clean_memory()
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
            termination_list     = [False for _ in range(self.image_num)]

            self.action_list=[[0 for _ in range((self.piece_num+self.buffer_size)*self.piece_num//2+1)] for __ in range(self.image_num)]
            self.trace_start_point=self.memory_counter


            step = 0; done = False
            while not done and step < max_step:
                state_list=[]
                do_list = []
                model_action=[]
                log_probs=[]
                last_reward_list,_=self.get_reward(permutation_list=permutation_list)
                with torch.no_grad():
                    self.actor_model.eval()
                    
                    for j in range(self.image_num):
                        if done_list[j] or termination_list[j]:           
                            continue
                        
                        perm_with_buf = permutation_list[j] + buffer
                        state_list.append(copy.deepcopy(perm_with_buf))
                        image, outsider = self.get_image(perm_with_buf,image_id=j)

                        probs = self.actor_model(image, outsider)
                        dist  = torch.distributions.Categorical(probs)
                        

                        action = dist.sample()
                        
                        # action=self.epsilon_greedy(action=action.item())
                        log_probs.append(dist.log_prob(action).detach())
                        action=action.item()
                        model_action.append(action)
                        self.action_list[j][action]+=1
                        do_list.append(j)

                        if action==37:
                            termination_list[j]=True
                            continue

                        new_perm = self.permute(perm_with_buf, action)
                        permutation_list[j], buffer = new_perm[:self.piece_num-1], new_perm[self.piece_num-1:]

                        



                reward_list, done_list = self.get_reward(permutation_list)
                if SHOW_IMAGE:
                    self.show_image(permutation_list)
                for j in do_list:
                    reward_sum_list[j].append(reward_list[j])
                if state_list:
                    self.recording_memory(image_id=self.image_id,state_list=state_list,log_probs=log_probs,action_list=model_action,reward_list=[reward_list[j]-last_reward_list[j] for j in do_list],next_state_list=[permutation_list[j] for j in do_list ],do_list=do_list,done_list=[done_list[j] for j in do_list])
                done = all(done_list)
                step += 1


            print(f"Epoch: {i}, step: {step}, reward: {[sum(reward_sum_list[j])/len(reward_sum_list[j]) for j in range(len(reward_sum_list)) if len(reward_sum_list[j])!=0 ]}")
            print(f"Action_list: {self.action_list}")
            print(f"Permutation list: {permutation_list}")
            action_num=[sum([1 if action!=0 else 0 for action in self.action_list[j]]) for j in range(self.image_num)]
            for j in range(self.image_num):
                if action_num[j]<=5:
                    self.epsilon/=(EPSILON_GAMMA**10)
            if self.epsilon>EPSILON_MIN:
                self.epsilon*=EPSILON_GAMMA
            if self.mkv_memory!=[]:
                print("Start training")
                self.update()

            






if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    
    critic=critic_model(hidden_size1=1024,hidden_size2=1024,outsider_hidden_size=256).to(device=DEVICE)
    actor=actor_model(hidden_size1=1024,hidden_size2=1024,outsider_hidden_size=256,action_num=37).to(DEVICE)

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
                    actor_model=actor,
                    critic_model=critic,
                    # encoder=feature_encoder,
                    image_num=1,
                    buffer_size=1,
                    epsilon=EPSILON,
                    epsilon_gamma=EPSILON_GAMMA,
                   entropy_weight=ENTROPY_WEIGHT)
    environment.step(epoch=EPOCH_NUM,load=LOAD_MODEL)
    # environment.load_image(1,id=[1000])
    # environment.show_image([[0,1,2,3,4,5,6,7,8]])
    # reward_list,done_list=environment.get_reward([[0,1,2,3,4,5,6,7,8],[9,10,11,12,13,14,15,16,17]])
    # print(reward_list,done_list)
    
