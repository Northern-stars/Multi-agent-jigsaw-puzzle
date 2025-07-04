import torch
import torch.nn as nn
import numpy as np
import random
import copy
import itertools
from outsider_pretrain import fen_model
from torchvision.models import efficientnet_b0


DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DONE_REWARD=1000
CLIP_GRAD_NORM=0.5
TRAIN_PER_STEP=8
ACTOR_LR=1e-5
CRITIC_LR=1e-3
ENCODER_LR=1e-4
ACTOR_SCHEDULAR_STEP=100
CRITIC_SCHEDULAR_STEP=100
ENCODER_SCHEDULAR_STEP=100
BASIC_BIAS=1e-8
PAIR_WISE_REWARD=.2
CATE_REWARD=.8
CONSISTENCY_REWARD=0.5
PANELTY=-1
ENTROPY_WEIGHT=0.1
ENTROPY_GAMMA=0.998
ENTROPY_MIN=0.005
EPOCH_NUM=500
LOAD_MODEL=True
SWAP_NUM=[4,1,2,4]
MAX_STEP=[200,100,100,100]

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
        self.fc1=nn.Linear(128*12,hidden_size1)
        self.relu=nn.ReLU()
        self.bn=nn.BatchNorm1d(hidden_size1)
        self.do=nn.Dropout1d(p=0.1)
        self.fc2=nn.Linear(hidden_size1,hidden_size2)
    
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
        hori_set=[(i,i+1) for i in [j for j in range(9) if j%3!=3-1 ]]
        vert_set=[(i,i+3) for i in range(3*2)]
        hori_tensor=torch.cat([self.ef(image_fragments[hori_set[i][0]])-self.ef(image_fragments[hori_set[i][0]]) for i in range(len(hori_set))],dim=-1)
        vert_tensor=torch.cat([self.ef(image_fragments[vert_set[i][0]])-self.ef(image_fragments[vert_set[i][0]]) for i in range(len(vert_set))],dim=-1)
        feature_tensor=torch.cat([hori_tensor,vert_tensor],dim=-1)
        x=self.do(feature_tensor)
        x=self.fc1(x)
        x=self.do(x)
        x=self.bn(x)
        x=self.relu(x)
        x=self.fc2(x)
        return x


class actor_model(nn.Module):
    def __init__(self,action_num):
        super(actor_model,self).__init__()
        self.image_fen_model=fen_model(256,256)
        self.outsider_fen_model=efficientnet_b0(weights="DEFAULT")
        self.outsider_fen_model.classifier=nn.Linear(1280,128)
        self.fc1=nn.Linear(384,128)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=0.1)
        self.fc2=nn.Linear(128,action_num)
    
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
    def __init__(self,hidden_size1,hidden_size2):
        super(critic_model,self).__init__()
        self.fen_model=fen_model(hidden_size1=hidden_size1,hidden_size2=hidden_size1)
        self.fc1=nn.Linear(hidden_size1,hidden_size2)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=0.1)
        self.fc=nn.Linear(hidden_size2,1)
    
    def forward(self,image):
        feature_tensor=self.fen_model(image)
        out=self.fc1(feature_tensor)
        out=self.relu(out)
        out=self.dropout(out)
        out=self.fc(out)
        return out


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
                 piece_num=9
                 ):
        self.image=train_x
        self.sample_number=train_x.shape[0]
        self.label=train_y
        self.permutation2piece={}
        self.cur_permutation={}
        self.mkv_memory=[]
        self.mkv_memory_size=memory_size
        self.memory_counter=0
        self.actor_model_list=[copy.deepcopy(actor_model) for i in range(image_num)]
        self.critic_model=critic_model
        self.actor_optimizer_list=[torch.optim.Adam(self.actor_model_list[i].parameters(),lr=ACTOR_LR) for i in range(image_num)]
        self.actor_schedular_list=[torch.optim.lr_scheduler.StepLR(self.actor_optimizer_list[j], step_size=ACTOR_SCHEDULAR_STEP, gamma=0.1) for j in range(image_num)]
        self.critic_optimizer=torch.optim.Adam(self.critic_model.parameters(),lr=CRITIC_LR)
        self.critic_schedular=torch.optim.lr_scheduler.StepLR(optimizer=self.critic_optimizer,gamma=0.1,step_size=CRITIC_SCHEDULAR_STEP)
        self.device=device
        self.batch_size=batch_size
        self.gamma=gamma
        self.image_num=image_num
        self.buffer_size=buffer_size
        self.action_list=[[0 for j in range(45)] for i in range(image_num)]
        self.piece_num=piece_num
        # self.encoder=encoder
        # self.encoder_optimizer=torch.optim.Adam(self.encoder.parameters(),lr=ENCODER_LR)
        # self.encoder_schedular=torch.optim.lr_scheduler.StepLR(optimizer=self.encoder_optimizer,gamma=0.1,step_size=ENCODER_SCHEDULAR_STEP)

        # print("Init")
    
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
        
    def get_image(self,permutation):
        image=torch.zeros(3,288,288)
        for i in range(9):
            image[:,(0+i//3)*96:(1+i//3)*96,(0+i%3)*96:(1+i%3)*96]=self.permutation2piece[permutation[i]]
    
        outsider_piece=self.permutation2piece[permutation[-1]]
        return image.unsqueeze(0).to(self.device),outsider_piece.unsqueeze(0).to(self.device)

    def get_reward(self,permutation_list):

        permutation_list=permutation_list[::][:len(permutation_list[0])-1]#Comment if the buffer is not after the permutation
        
        done_list=[0 for i in range(len(permutation_list))]
        reward_list=[PANELTY for i in range(len(permutation_list))]
        edge_length=int(len(permutation_list[0])**0.5)
        piece_num=len(permutation_list[0])
        hori_set=[(i,i+1) for i in [j for j in range(piece_num) if j%edge_length!=edge_length-1 ]]
        vert_set=[(i,i+edge_length) for i in range(edge_length*2)]
        for i in range(len(permutation_list)):
            
            for j in range(len(hori_set)):#Pair reward
                hori_pair_set=(permutation_list[i][hori_set[j][0]],permutation_list[i][hori_set[j][1]])
                vert_pair_set=(permutation_list[i][vert_set[j][0]],permutation_list[i][vert_set[j][1]])
                if -1 not in hori_pair_set and (hori_pair_set[0]%piece_num,hori_pair_set[1]%piece_num) in hori_set:
                    reward_list[i]+=1*PAIR_WISE_REWARD
                if -1 not in vert_pair_set and (vert_pair_set[0]%piece_num,vert_pair_set[1]%piece_num) in vert_set:
                    reward_list[i]+=1*PAIR_WISE_REWARD

            piece_range=[0 for j in range (len(permutation_list))]
        
            for j in range(piece_num):
                if permutation_list[i][j]!=-1:
                    piece_range[permutation_list[i][j]//piece_num]+=1
                reward_list[i]+=(permutation_list[i][j]%piece_num==j)*CATE_REWARD#Category reward
            
            max_piece=max(piece_range)#Consistancy reward
            if max_piece==piece_num-3:
                reward_list[i]+=1*CONSISTENCY_REWARD
            elif max_piece==piece_num-2:
                reward_list[i]+=2*CONSISTENCY_REWARD
            elif max_piece==piece_num-1:
                reward_list[i]+=3*CONSISTENCY_REWARD
            elif max_piece==piece_num:
                reward_list[i]+=5*CONSISTENCY_REWARD
            
            start_index=min(permutation_list[i])//piece_num*piece_num#Done reward
            if permutation_list[i]==range(start_index,start_index+piece_num):
                done_list[i]=True
                reward_list[i]+=DONE_REWARD
        return reward_list,done_list
        #Change after determined
    

    def critic_update(self,state_value,returns_list):
        self.critic_model.train()
        state_value=torch.cat(state_value,dim=0)
        returns=torch.cat(returns_list,dim=0)
        critic_loss=nn.functional.mse_loss(state_value.squeeze(),returns)
        self.critic_optimizer.zero_grad()
        # self.encoder_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), CLIP_GRAD_NORM)
        # self.encoder_optimizer.step()
        return critic_loss.item()
        

    def actor_update(self,actor_index,log_prob_list,state_value,returns,entropy):
        self.actor_model_list[actor_index].train()
        # self.encoder.train()
        adavantage=returns-state_value.detach()
        log_prob=torch.cat(log_prob_list)
        actor_loss=-(log_prob*adavantage).mean()-ENTROPY_WEIGHT*entropy
        self.actor_optimizer_list[actor_index].zero_grad()
        # self.encoder_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_model_list[actor_index].parameters(), CLIP_GRAD_NORM)
        # torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), CLIP_GRAD_NORM)
        self.actor_optimizer_list[actor_index].step()
        # self.encoder_optimizer.step()


    def permute(self,cur_permutation,action_index):
        new_permutation=copy.deepcopy(cur_permutation)
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
        initial_permutation=list(range(self.piece_num*self.image_num))
        for i in range(swap_num):
            action_index=random.randint(0,len(initial_permutation)*(len(initial_permutation)-1)//2-1)
            initial_permutation=self.permute(initial_permutation,action_index)
        return initial_permutation


    def step(self,epoch=500,load=True):#Change after determined
        global ENTROPY_WEIGHT
        global ENTROPY_GAMMA
        global ENTROPY_MIN
        if load:
            self.critic_model.load_state_dict(torch.load("Critic"+MODEL_NAME))
            for j in range(self.image_num):
                self.actor_model_list[j].load_state_dict(torch.load("Actor_"+str(j)+MODEL_NAME))
        for i in range(epoch):
            if i>300:
                max_step=MAX_STEP[3]
                swap_num=SWAP_NUM[0]
            elif i>200:
                max_step=MAX_STEP[2]
                swap_num=SWAP_NUM[2]
            elif i>100:
                max_step=MAX_STEP[1]
                swap_num=SWAP_NUM[1]
            else:
                max_step=MAX_STEP[0]
                swap_num=SWAP_NUM[0]
            initial_permutation=self.summon_permutation_list(swap_num=swap_num)
            reward_sum_list=[[] for j in range(self.image_num)]
            done_list=[False for j in range(self.image_num)]
            done=False
            step=0
            permutation_list=[]
            for j in range(self.image_num):
                permutation_list.append(initial_permutation[j*self.piece_num:(j+1)*self.piece_num])

            buffer=[-1 for i in range(self.buffer_size)]
            log_prob_list=[[] for j in range(self.image_num)]
            entropy=[0 for i in range(self.image_num)]
            train_reward=[[] for j in range(self.image_num)]
            critic_output_list=[[] for j in range(self.image_num)]
            self.action_list=[[0 for j in range(45)] for i in range(self.image_num)]
            critic_loss_sum=0
            while not done and step<max_step:
                image_list=[]
                do_list=[]
                # actor_feature_tensor=[torch.zeros([3,288,288]) for i in range(self.image_num)]
                for j in range(self.image_num):
                    # self.encoder.eval()
                    if not done_list[j]:
                        self.actor_model_list[j].eval()
                        permutation=permutation_list[j]+buffer
                        image,outsider=self.get_image(permutation)
                        # actor_feature_tensor[j]=self.encoder(image)
                        action_probs=self.actor_model_list[j](image,outsider)
                        action_dist=torch.distributions.Categorical(action_probs)
                        action=action_dist.sample()
                        new_permutation=self.permute(permutation,action.item())
                        self.action_list[j][action.item()]+=1
                        log_prob_list[j].append(action_dist.log_prob(action))
                        entropy[j]+=action_dist.entropy().mean()
                        new_image,_=self.get_image(new_permutation)
                        image_list.append(new_image)
                        do_list.append(j)
                        permutation_list[j],buffer=new_permutation[0:9],new_permutation[9::]
                
                    

                image_list=torch.cat(image_list)
                # critic_feature_tensor=self.encoder(image_list)
                critic_output=self.critic_model(image_list)
                for j in range(critic_output.size(0)):
                    if critic_output_list[do_list[j]]!=[]:
                        critic_output_list[do_list[j]]=torch.cat([critic_output_list[do_list[j]],critic_output[j].unsqueeze(0)],dim=1)
                    else:
                        critic_output_list[do_list[j]]=critic_output[j].unsqueeze(0)
                    
                reward_list,done_list=self.get_reward(permutation_list)
                done=True
                for j in do_list:
                    reward_sum_list[j].append(reward_list[j])
                    # if train_reward[j]:
                    #     train_reward[j].append(reward_list[j]-reward_sum_list[j][-2])
                    # else:
                    #     train_reward[j].append(reward_list[j])
                    train_reward[j].append(reward_list[j])
                for j in do_list:
                    if not done_list[j]:
                        done=False
                        break
                
                if (step+1)%TRAIN_PER_STEP==0:
                    return_list=[[] for j in range(self.image_num)]
                    for j in range(self.image_num):
                        if train_reward[j]:
                            R=0
                            for r in train_reward[j][::-1]:
                                R=r+self.gamma*R
                                return_list[j].insert(0,R)
                    
                            return_list[j]=torch.tensor(return_list[j]).float().to(self.device).unsqueeze(0)
                            return_list[j]=(return_list[j]-return_list[j].mean())/(return_list[j].std()+BASIC_BIAS)
                    # print(return_list)
                    critic_loss_sum+=self.critic_update(state_value=critic_output_list,returns_list=return_list)
                    for j in range(self.image_num):
                        if log_prob_list[j]:
                            self.actor_update(actor_index=j,log_prob_list=log_prob_list[j],state_value=critic_output_list[j],returns=return_list[j],entropy=entropy[j])
                    log_prob_list=[[] for j in range(self.image_num)]
                    entropy=[0 for i in range(self.image_num)]
                    train_reward=[[] for j in range(self.image_num)]
                    critic_output_list=[[] for j in range(self.image_num)]
                    if ENTROPY_WEIGHT>=ENTROPY_MIN:
                        ENTROPY_WEIGHT=ENTROPY_WEIGHT*ENTROPY_GAMMA
                step=step+1
            print(f"Epoch: {i}. Done: {done}, step: {step},reward: {[sum(reward_sum_list[j])/step for j in range(self.image_num)]}, critic_loss: {critic_loss_sum}")
            
            print(f"Action_list: {self.action_list}")
            torch.save(self.critic_model.state_dict(),"Critic"+MODEL_NAME)
            self.critic_schedular.step()
            # self.encoder_schedular.step()
            for j in range(self.image_num):
                torch.save(self.actor_model_list[j].state_dict(),"Actor_"+str(j)+MODEL_NAME)
                self.actor_schedular_list[j].step()






if __name__ == "__main__":
    MODEL_NAME="(1).pth"
    critic=critic_model(hidden_size1=256,hidden_size2=128).to(device=DEVICE)
    actor=actor_model(45).to(DEVICE)
    # feature_encoder=fen_model(512,512).to(device=DEVICE)
    environment=env(train_x=train_x,
                    train_y=train_y,
                    memory_size=1e4,
                    batch_size=5,
                    gamma=0.99,
                    device=DEVICE,
                    actor_model=actor,
                    critic_model=critic,
                    # encoder=feature_encoder,
                    image_num=2,
                    buffer_size=1)
    environment.step(epoch=EPOCH_NUM,load=LOAD_MODEL)
    