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
TRAIN_PER_STEP=20
ACTOR_LR=1e-5
ACTOR_LR_MIN=1e-6
CRITIC_LR=1e-3
CRITIC_LR_MIN=1e-5
ENCODER_LR=1e-4
SCHEDULAR_STEP=200
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
LOAD_MODEL=True
SWAP_NUM=[2,3,4,8]
MAX_STEP=[400,400,400,400]
MODEL_NAME="(7).pth"
MODEL_PATH=os.path.join("DQN"+MODEL_NAME)


BATCH_SIZE=25
EPSILON=0.9
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

class fen_model(nn.Module):
    def __init__(self,hidden_size1,hidden_size2):
        super(fen_model,self).__init__()
        self.ef=efficientnet_b0(weights="DEFAULT")
        self.ef.classifier=nn.Linear(1280,256)
        self.contrast_fc_hori=nn.Linear(512,256)
        self.contrast_fc_vert=nn.Linear(512,256)
        self.fc1=nn.Linear(256*12,hidden_size1)
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
        
        image_tensor=[self.ef(image_fragments[i]) for i in range(len(image_fragments))]

        hori_tensor=torch.cat([self.contrast_fc_hori(torch.cat([image_tensor[self.hori_set[i][0]],image_tensor[self.hori_set[i][1]]],dim=-1)) for i in range(len(self.hori_set))],dim=-1)
        vert_tensor=torch.cat([self.contrast_fc_vert(torch.cat([image_tensor[self.vert_set[i][0]],image_tensor[self.vert_set[i][1]]],dim=-1)) for i in range(len(self.vert_set))],dim=-1)
        feature_tensor=torch.cat([hori_tensor,vert_tensor],dim=-1)
        x=self.do(feature_tensor)
        x=self.fc1(x)
        x=self.do(x)
        # x=self.bn(x)
        x=self.relu(x)
        x=self.fc2(x)
        return x


class critic_model(nn.Module):
    def __init__(self,hidden_size1,hidden_size2,outsider_hidden_size):
        super(critic_model,self).__init__()
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
        self.fc=nn.Linear(hidden_size2,1)
    
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
        out=self.dropout(out)
        out=self.fc(out)
        return out




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
                 model,#input: image,outsider_piece. Output: q_value
                 image_num,
                 buffer_size,
                 entropy_weight,
                 epsilon,
                 epsilon_gamma,
                 loss_fn=nn.MSELoss(),
                 piece_num=9,
                 epochs=10,
                 tau=0.01
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
        self.main_model=copy.deepcopy(self.model)
        self.optimizer=torch.optim.Adam(self.main_model.parameters(),lr=CRITIC_LR)
        self.schedular=torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,gamma=0.1,step_size=SCHEDULAR_STEP)
        self.device=device
        self.batch_size=batch_size
        self.gamma=gamma
        self.image_num=image_num
        self.buffer_size=buffer_size
        self.action_list=[[0 for _ in range((piece_num-1+buffer_size)*(piece_num-1)//2)] for __ in range(image_num)]
        self.piece_num=piece_num
        self.entropy_weight=entropy_weight
        self.epsilon=epsilon
        self.epsilon_gamma=epsilon_gamma
        self.epochs=epochs
        self.tau=tau
        self.loss_fn=loss_fn

    
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
    
        outsider_piece=self.permutation2piece[permutation[-1]]
        return image.unsqueeze(0).to(self.device),outsider_piece.unsqueeze(0).to(self.device)

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
    

    def clean_memory(self):
        self.mkv_memory=[]
        self.memory_counter=0

    def show_image(self,image_permutation_list):
        for i in range(self.image_num):

            image,_=self.get_image(permutation=image_permutation_list[i],image_index=i)
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

    def choose_action(self,perm_with_buf,image_index):
        perm_=copy.deepcopy(perm_with_buf)
        value_list=[]
        image_list=[]
        outsider_list=[]

        for i in range(len(self.action_list[0])):
            perm_=self.permute(perm_with_buf,i)
            image,outsider=self.get_image(perm_,image_index=image_index)
            image_list.append(copy.deepcopy(image.cpu()))
            outsider_list.append(copy.deepcopy(outsider.cpu()))
        
        i=0
        with torch.no_grad():
            while i < len(self.action_list):
                if len(self.action_list)-i<BATCH_SIZE:
                    image=torch.cat(image_list[i:],dim=0).to(DEVICE)
                    outsider=torch.cat(outsider_list[i:],dim=0).to(DEVICE)
                else:
                    image=torch.cat(image_list[i:i+BATCH_SIZE],dim=0).to(DEVICE)
                    outsider=torch.cat(outsider_list[i:i+BATCH_SIZE],dim=0).to(DEVICE)

                value=self.model(image,outsider)
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
    
    

        
    def update(self,show=False):

        
        for target_param, main_param in zip(self.model.parameters(),self.main_model.parameters()):
            target_param.data.copy_(self.tau*main_param.data+(1-self.tau)*target_param.data)
        
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
            outsider_pieces=[]

            actions=[]
            next_states=[]
            next_outsiders=[]
            reward=[]
            for a in range(len(sample_dicts)):
                self.load_image(image_num=self.image_num,id=sample_dicts[a]["Image_id"])

                    
                current_image,current_outsider=self.get_image(sample_dicts[a]["State"],image_index=sample_dicts[a]["Image_index"])
                states.append(current_image)
                outsider_pieces.append(current_outsider)

                actions.append(sample_dicts[a]["Action"])
                next_image,next_outsider_piece=self.get_image(sample_dicts[a]["Next_state"],image_index=sample_dicts[a]["Image_index"])
                next_states.append(next_image)
                next_outsiders.append(next_outsider_piece)
                
                reward.append(sample_dicts[a]["Reward"])
            


            state_tensor=torch.cat(states,dim=0)
            outsider_tensor=torch.cat(outsider_pieces,dim=0)
            next_state_tensor=torch.cat(next_states,dim=0)
            next_outsiders_tensor=torch.cat(next_outsiders,dim=0)

            q_next=self.model(next_state_tensor,next_outsiders_tensor).detach()
            q_eval=self.main_model(next_state_tensor,next_outsiders_tensor)

            reward=torch.tensor(reward,dtype=torch.float32).to(self.device).unsqueeze(-1)

            q_target=reward+self.gamma*q_next
            q_target=q_target.to(torch.float)

            loss=self.loss_fn(q_target,q_eval)
            self.optimizer.zero_grad()
            loss.float().backward()
            self.optimizer.step()
            loss_sum+=loss.item()
        self.schedular.step()
        if show:
            print(f"Average_loss: {loss_sum/self.epochs}")
        






    def step(self, epoch=500, load=True):
        if load:
            self.model.load_state_dict(torch.load(MODEL_PATH))
            self.main_model=copy.deepcopy(self.model)
            self.optimizer=torch.optim.Adam(self.main_model.parameters(),lr=CRITIC_LR)

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

            initial_perm = self.summon_permutation_list(swap_num=swap_num)
            permutation_list = [initial_perm[j*(self.piece_num-1):(j+1)*(self.piece_num-1)]
                                for j in range(self.image_num)]
            buffer = [-1] * self.buffer_size
            done_list = [False] * self.image_num
            termination_list = [False for _ in range(self.image_num)]
            reward_sum_list = [[] for _ in range(self.image_num)]
            
            pending_transitions = {j: None for j in range(self.image_num)}

            step = 0
            done = False
            while not done and step < max_step:
                do_list = []  
                last_reward_list, _ = self.get_reward(permutation_list=permutation_list)

                for j in range(self.image_num):
                    if done_list[j] or termination_list[j]:
                        continue

                    perm_with_buf = permutation_list[j] + buffer
                    action = self.choose_action(perm_with_buf,image_index=j)
                    action=self.epsilon_greedy(action=action)  

                    
                    if pending_transitions[j] is not None:
                        prev_state, prev_action, prev_reward = pending_transitions[j]
                        self.recording_memory(image_id=self.image_id,image_index=j,state=prev_state,action= prev_action,reward= prev_reward,next_state= perm_with_buf, done=done_list[j])



                    pending_transitions[j] = (perm_with_buf, action, 0) 

                    do_list.append(j)

                    # if action == 36:
                    #     termination_list[j] = True
                    #     continue

                    new_perm = self.permute(perm_with_buf, action)
                    permutation_list[j], buffer = new_perm[:self.piece_num-1], new_perm[self.piece_num-1:]

                reward_list, done_list = self.get_reward(permutation_list)
                if SHOW_IMAGE:
                    self.show_image(permutation_list)


                for j in do_list:
                    prev_state, prev_action, _ = pending_transitions[j]
                    pending_transitions[j] = (prev_state, prev_action, reward_list[j])
                    reward_sum_list[j].append(reward_list[j])

                done = all(done_list)
                step += 1

                if step%TRAIN_PER_STEP==0:
                    self.update()
                    self.load_image(image_num=self.image_num,id=self.image_id)


            for j in range(self.image_num):
                if pending_transitions[j] is not None and done_list[j]:
                    prev_state, prev_action, prev_reward = pending_transitions[j]
                    self.recording_memory(image_id=self.image_id,image_index=j,state=prev_state,action= prev_action,reward= prev_reward,next_state= perm_with_buf, done=done_list[j])  

            print(f"Epoch: {i}, step: {step}, reward: {[sum(rs)/len(rs) for rs in reward_sum_list if rs]}")
            self.update(show=True) 
            torch.save(self.model.state_dict(),MODEL_PATH)
            if self.epsilon>EPSILON_MIN:
                self.epsilon*=EPSILON_GAMMA


            






if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    
    critic=critic_model(hidden_size1=1024,hidden_size2=1024,outsider_hidden_size=256).to(device=DEVICE)
    
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

                    model=critic,
                    # encoder=feature_encoder,
                    image_num=2,
                    buffer_size=1,
                    epsilon=EPSILON,
                    epsilon_gamma=EPSILON_GAMMA,
                   entropy_weight=ENTROPY_WEIGHT)
    environment.step(epoch=EPOCH_NUM,load=LOAD_MODEL)
    # environment.load_image(1,id=[1000])
    # environment.show_image([[0,1,2,3,4,5,6,7,8]])
    # reward_list,done_list=environment.get_reward([[0,1,2,3,4,5,6,7,8],[9,10,11,12,13,14,15,16,17]])
    # print(reward_list,done_list)
    
