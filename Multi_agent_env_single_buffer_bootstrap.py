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
CLIP_GRAD_NORM=0.1
TRAIN_PER_STEP=8
ACTOR_LR=1e-5
CRITIC_LR=1e-3
ENCODER_LR=1e-4
ACTOR_SCHEDULAR_STEP=200
CRITIC_SCHEDULAR_STEP=100
ENCODER_SCHEDULAR_STEP=100
BASIC_BIAS=1e-8
PAIR_WISE_REWARD=.2
CATE_REWARD=.8
CONSISTENCY_REWARD=1
PANELTY=-1
ENTROPY_WEIGHT=0.01
ENTROPY_GAMMA=0.998
ENTROPY_MIN=0.005
EPOCH_NUM=500
LOAD_MODEL=True
SWAP_NUM=[1,2,2,4]
MAX_STEP=[120,240,120,240]
MODEL_NAME="(1).pth"

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
        self.action_list=[[0 for _ in range((piece_num+buffer_size)*piece_num//2+1)] for __ in range(image_num)]
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
            if permutation_list[i]==list(range(start_index,start_index+piece_num)):
                done_list[i]=True
                reward_list[i]=DONE_REWARD
        return reward_list,done_list
        #Change after determined
    

    def critic_update(self,state_value,returns_list):
        self.critic_model.train()
        state_value=torch.cat(state_value,dim=1)
        returns=torch.cat(returns_list,dim=1)
        critic_loss=nn.functional.mse_loss(state_value.squeeze(),returns.squeeze())
        self.critic_optimizer.zero_grad()
        # self.encoder_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), CLIP_GRAD_NORM)
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
        initial_permutation=list(range(self.piece_num*self.image_num))
        for i in range(swap_num):
            action_index=random.randint(0,len(initial_permutation)*(len(initial_permutation)-1)//2-1)
            initial_permutation=self.permute(initial_permutation,action_index)
        return initial_permutation


    def step(self, epoch=500, load=True):
        global ENTROPY_WEIGHT, ENTROPY_GAMMA, ENTROPY_MIN

        
        if load:
            self.critic_model.load_state_dict(torch.load("Critic" + MODEL_NAME))
            for j in range(self.image_num):
                self.actor_model_list[j].load_state_dict(torch.load(f"Actor_{j}" + MODEL_NAME))

        
        for i in range(epoch):
            
            if i > 300:
                max_step, swap_num = MAX_STEP[3], SWAP_NUM[0]
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

            log_prob_list        = [[] for _ in range(self.image_num)]
            entropy_list         = [0. for _ in range(self.image_num)]
            train_reward         = [[] for _ in range(self.image_num)]
            critic_output_list   = [[] for _ in range(self.image_num)]
            self.action_list=[[0 for _ in range((self.piece_num+self.buffer_size)*self.piece_num//2+1)] for __ in range(self.image_num)]
            critic_loss_sum=0
            last_reward          = [0. for _ in range(self.image_num)]  # 记录上一步即时奖励

            step = 0; done = False
            while not done and step < max_step:

                image_batch, do_list = [], []
                for j in range(self.image_num):
                    if done_list[j]:           
                        continue
                    self.actor_model_list[j].eval()

                    perm_with_buf = permutation_list[j] + buffer
                    image, outsider = self.get_image(perm_with_buf)

                    probs = self.actor_model_list[j](image, outsider)
                    dist  = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    self.action_list[j][action.item()]+=1

                    log_prob_list[j].append(dist.log_prob(action))
                    entropy_list[j] += dist.entropy().mean()

                    new_perm = self.permute(perm_with_buf, action.item())
                    permutation_list[j], buffer = new_perm[:self.piece_num], new_perm[self.piece_num:]

                    next_img, _ = self.get_image(new_perm)
                    image_batch.append(next_img)
                    do_list.append(j)
                if len(image_batch)==1:
                    self.critic_model.eval()
                critic_out = self.critic_model(torch.cat(image_batch))
        
                for idx, j in enumerate(do_list):
                    if critic_output_list[j]!=[]:
                        critic_output_list[j] = torch.cat(
                            [critic_output_list[j], critic_out[idx].unsqueeze(0)], dim=1)
                    else:
                        critic_output_list[j] = critic_out[idx].unsqueeze(0)


                reward_list, done_list = self.get_reward(permutation_list)
                done = all(done_list)

                for j in do_list:
                    r = reward_list[j]
                    train_reward[j].append(r)
                    reward_sum_list[j].append(r)
                    last_reward[j] = r

                if (step + 1) % TRAIN_PER_STEP == 0 or done:
                    returns_list = [[] for _ in range(self.image_num)]

                    for j in range(self.image_num):
                        if not train_reward[j]:
                            continue

                        R0 = 0. if done_list[j] else last_reward[j]  
                        R = R0
                        for r in reversed(train_reward[j]):
                            R = r + self.gamma * R
                            returns_list[j].insert(0, R)

                        ret = torch.tensor(returns_list[j], dtype=torch.float32,
                                        device=self.device).unsqueeze(0)
                        returns_list[j] = (ret - ret.mean()) / (ret.std() + BASIC_BIAS)

                    filtered_critic_output_list=[x for x in critic_output_list if x!=[]]
                    filtered_return_list=[x for x in returns_list if x!=[]]
                    critic_loss = self.critic_update(filtered_critic_output_list, filtered_return_list)
                    critic_loss_sum+=critic_loss*TRAIN_PER_STEP


                    for j in range(self.image_num):
                        if log_prob_list[j]:
                            self.actor_update(
                                actor_index=j,
                                log_prob_list=log_prob_list[j],
                                state_value=critic_output_list[j],
                                returns=returns_list[j],
                                entropy=entropy_list[j]
                            )


                    log_prob_list      = [[] for _ in range(self.image_num)]
                    entropy_list       = [0. for _ in range(self.image_num)]
                    train_reward       = [[] for _ in range(self.image_num)]
                    critic_output_list = [[] for _ in range(self.image_num)]
                    if ENTROPY_WEIGHT >= ENTROPY_MIN:
                        ENTROPY_WEIGHT *= ENTROPY_GAMMA

                step += 1


            print(f"Epoch {i}: step={step}, done: {done}, return={[sum(r)/step for r in reward_sum_list]}, critic_loss_sum= {critic_loss_sum}")
            print(f"Action list: {self.action_list}")
            print(f"Final permutation: {permutation_list}")
            torch.save(self.critic_model.state_dict(), "Critic" + MODEL_NAME)
            self.critic_schedular.step()
            for j in range(self.image_num):
                torch.save(self.actor_model_list[j].state_dict(), f"Actor_{j}" + MODEL_NAME)
                self.actor_schedular_list[j].step()






if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    
    critic=critic_model(hidden_size1=256,hidden_size2=128).to(device=DEVICE)
    actor=actor_model(46).to(DEVICE)
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
    # reward_list,done_list=environment.get_reward([[0,1,2,3,4,5,6,7,8],[9,10,11,12,13,14,15,16,17]])
    # print(reward_list,done_list)
    