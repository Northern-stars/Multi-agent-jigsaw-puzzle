import torch
import torch.nn as nn
import numpy as np
import random
import copy
import itertools
from outsider_pretrain import fen_model


DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DONE_REWARD=1000

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


class actor_model(nn.Module):
    def __init__(self,action_num):
        super(actor_model,self).__init__()
        self.image_fen_model=fen_model(512,512)
        self.outsider_fen_model=fen_model(512,128)
        self.fc1=nn.Linear(640,160)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=0.3)
        self.fc2=nn.Linear(160,action_num)
    
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
    def __init__(self):
        super(critic_model,self).__init__()
        self.a_fen_model=fen_model(512,512)
        self.b_fen_model=fen_model(512,512)
        self.fc1=nn.Linear(1024,256)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=0.3)
        self.fc_a=nn.Linear(256,1)
        self.fc_b=nn.Linear(256,1)
    
    def forward(self,image_a,image_b):
        image_a_tensor=self.a_fen_model(image_a)
        image_b_tensor=self.b_fen_model(image_b)
        feature_tensor=torch.cat([image_a_tensor,image_b_tensor],dim=1)
        out=self.fc1(feature_tensor)
        out=self.relu(out)
        out=self.dropout(out)
        out_a=self.fc_a(out)
        out_b=self.fc_b(out)
        return out_a,out_b

class outsider_model(nn.Module):
    def __init__(self,hidden_size1,hidden_size2):
        super(outsider_model,self).__init__()
        self.pre_model=fen_model(hidden_size1,hidden_size2)
        self.fc1=nn.Linear(hidden_size2,hidden_size2)
        self.bn1=nn.BatchNorm1d(hidden_size2)
        self.relu1=nn.ReLU()
        self.dp1=nn.Dropout1d(p=0.3)
        self.fc2=nn.Linear(hidden_size2,hidden_size2)
        self.bn2=nn.BatchNorm1d(hidden_size2)
        self.relu2=nn.ReLU()
        self.outlayer=nn.Linear(hidden_size2,9)
    def forward(self,x):
        x=self.pre_model(x)
        x=self.fc1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.dp1(x)
        x=self.fc2(x)
        x=self.bn2(x)
        x=self.relu2(x)
        out=self.outlayer(x)
        out=nn.functional.softmax(out,dim=1)
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
                 outsider_model#input:image,outsider_piece. Output: Outsider index
                 ):
        self.image=train_x
        self.sample_number=train_x.shape[0]
        self.label=train_y
        self.permutation2piece={}
        self.cur_permutation={}
        self.mkv_memory=[]
        self.mkv_memory_size=memory_size
        self.memory_counter=0
        self.actor_model=actor_model
        self.critic_model=critic_model
        self.outsider_model=outsider_model
        self.actor_optimizer=torch.optim.Adam(actor_model.parameters(),lr=1e-4)
        self.critic_optimizer=torch.optim.Adam(critic_model.parameters(),lr=1e-4)
        self.outsider_optimizer=torch.optim.Adam(outsider_model.parameters(),lr=1e-4)
        self.device=device
        self.batch_size=batch_size
        self.gamma=gamma
        # print("Init")
    
    def load_image(self,image_num):
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
        
    def get_image(self,permutation):
        image=torch.zeros(3,288,288)
        for i in range(9):
            image[:,(0+i//3)*96:(1+i//3)*96,(0+i%3)*96:(1+i%3)*96]=self.permutation2piece[permutation[i]]
        if permutation[-1]==-1:
            outsider_piece=torch.zeros(3,96,96)
        else:
            outsider_piece=self.permutation2piece[permutation[-1]]
        return image.unsqueeze(0).to(self.device),outsider_piece.unsqueeze(0).to(self.device)

    def get_reward(self,permutation_a,permutation_b):

        permutation_a=permutation_a[:len(permutation_a)-1]
        permutation_b=permutation_b[:len(permutation_b)-1]
        
        min_index_a=min(permutation_a)
        max_index_a=max(permutation_a)

        min_index_b=min(permutation_b)
        max_index_b=max(permutation_b)

        done_a=0
        done_b=0
        reward_a=-1
        reward_b=-1

        hori_set=[(0,1),(1,2),(3,4),(4,5),(6,7),(7,8),(9,10),(10,11),(12,13),(13,14),(15,16),(16,17)]
        vert_set=[(0,3),(3,6),(1,4),(4,7),(2,5),(5,8),(9,12),(12,15),(10,13),(13,16),(11,14),(14,17)]

        

        for i in range(len(hori_set)//2):
            if (permutation_a[hori_set[i][0]],permutation_a[hori_set[i][1]]) in hori_set:
                reward_a+=1
            if (permutation_a[vert_set[i][0]],permutation_a[vert_set[i][1]]) in vert_set:
                reward_a+=1
        
        for i in range(len(hori_set)//2):
            if (permutation_b[hori_set[i][0]],permutation_b[hori_set[i][1]]) in hori_set:
                reward_b+=1
            if (permutation_b[vert_set[i][0]],permutation_b[vert_set[i][1]]) in vert_set:
                reward_b+=1
        
        if max_index_a-min_index_a<9:
            reward_a+=3
        if max_index_b-min_index_b<9:
            reward_b+=3
        
        for i in range(9):
            reward_a+=(permutation_a[i]%9==i)
            reward_b+=(permutation_b[i]%9==i)

        if permutation_a==range(min_index_a,max_index_a):
            reward_a+=DONE_REWARD
            done_a=True
        
        if permutation_b==range(min_index_b,max_index_b):
            reward_b+=DONE_REWARD
            done_b=True

        

        return reward_a,reward_b,done_a,done_b
        #Change after determined
    
    def empty_memory(self):
        self.mkv_memory=[]
        self.memory_counter=0


    def update(self):
        self.actor_model.train()
        self.critic_model.train()
        self.outsider_model.train()
        sample_dicts=random.choices(self.mkv_memory,k=self.batch_size)
        states=[]
        outsider_pieces=[]
        rewards=[]
        actions=[]
        next_states=[]
        next_outsider_pieces=[]
        dones=[]
        outsiders=[]
        for i in range(len(sample_dicts)):
            cur_image,cur_outsider=self.get_image(sample_dicts[i]["states"])
            states.append(cur_image)
            outsider_pieces.append(cur_outsider)
            next_image,next_outsider=self.get_image(sample_dicts[i]["next_states"])
            next_states.append(next_image)
            next_outsider_pieces.append(next_outsider)
            rewards.append(sample_dicts[i]['rewards'])
            actions.append(sample_dicts[i]['actions'][0])
            outsiders.append(sample_dicts[i]['actions'][1])
            dones.append(sample_dicts[i]['dones'])
            
        
        states=torch.cat(states,dim=0).to(device=self.device)
        outsider_pieces=torch.cat(outsider_pieces,dim=0).to(self.device)
        next_states=torch.cat(next_states,dim=0).to(self.device)
        next_outsider_pieces=torch.cat(next_outsider_pieces,dim=0).to(self.device)
        rewards=torch.tensor(rewards,dtype=torch.float).view(-1,1).to(self.device)
        actions=torch.tensor(actions).view(-1,1).to(self.device)
        outsiders=torch.tensor(outsiders).view(-1,1).to(self.device)
        dones=torch.tensor(dones,dtype=torch.float).view(-1,1).to(self.device)

        td_target=rewards+self.gamma*self.critic_model(next_states,next_outsider_pieces)*(1-dones)
        td_delta=td_target-self.critic_model(states,outsider_pieces)
        predicted_action=self.actor_model(states,outsider_pieces)
        predicted_outsider=self.outsider_model(states)
        action_log_probs=torch.log(predicted_action.gather(1,actions))
        outsider_log_probs=torch.log(predicted_outsider.gather(1,outsiders))
        actor_loss=torch.mean(-action_log_probs*td_delta.detach())
        outsider_loss=torch.mean(-outsider_log_probs*td_delta.detach())
        critic_loss=torch.mean(torch.nn.functional.mse_loss(self.critic_model(states,outsider_pieces),td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.outsider_model.zero_grad()
        actor_loss.backward()
        outsider_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.outsider_model.step()
        self.critic_optimizer.step()



        


    def recording_buffer(self,state,action,reward,next_state,dones):
        #state:[permutation]
        #next_state:[next_permutation]
        #action:[swap_action,outsider_index]
        #reward:reward
        #dones: if it is done(0/1)
        memory_dic={"states":state,"actions":action,"rewards":reward,"next_states":next_state,"dones":dones}
        if len(self.mkv_memory)<=self.mkv_memory_size:
            self.mkv_memory.append(memory_dic)
        else:
            self.mkv_memory[self.memory_counter]=memory_dic
        
        self.memory_counter=(self.memory_counter+1)%self.mkv_memory_size

    def double_recording_buffer(self,state_a,state_b,action_a,action_b,reward_a,reward_b,next_state_a,next_state_b,dones_a,dones_b):
        memory_dic={"states":[state_a,state_b],"actions":[action_a,action_b],"rewards":[reward_a,reward_b],"next_states":[next_state_a,next_state_b],"dones":[dones_a,dones_b]}
        if len(self.mkv_memory)<=self.mkv_memory_size:
            self.mkv_memory.append(memory_dic)
        else:
            self.mkv_memory[self.memory_counter]=memory_dic
        
        self.memory_counter=(self.memory_counter+1)%self.mkv_memory_size
    
    def double_update(self):
        self.actor_model.train()
        self.critic_model.train()
        self.outsider_model.train()
        sample_dicts=random.choices(self.mkv_memory,k=self.batch_size)
        states_a=[]
        states_b=[]
        outsider_pieces_a=[]
        outsider_pieces_b=[]
        rewards_a=[]
        rewards_b=[]
        actions_a=[]
        actions_b=[]
        next_states_a=[]
        next_states_b=[]
        next_outsider_pieces_a=[]
        next_outsider_pieces_b=[]
        dones_a=[]
        dones_b=[]
        outsiders_a=[]
        outsiders_b=[]


        for i in range(len(sample_dicts)):
            
            cur_image_a,cur_outsider_a=self.get_image(sample_dicts[i]["states"][0])
            states_a.append(cur_image_a)
            outsider_pieces_a.append(cur_outsider_a)

            cur_image_b,cur_outsider_b=self.get_image(sample_dicts[i]["states"][1])
            states_b.append(cur_image_b)
            outsider_pieces_b.append(cur_outsider_b)


            next_image_a,next_outsider_a=self.get_image(sample_dicts[i]["next_states"][0])
            next_states_a.append(next_image_a)
            next_outsider_pieces_a.append(next_outsider_a)

            next_image_b,next_outsider_b=self.get_image(sample_dicts[i]["next_states"][1])
            next_states_b.append(next_image_b)
            next_outsider_pieces_b.append(next_outsider_b)


            rewards_a.append(sample_dicts[i]['rewards'][0])
            rewards_b.append(sample_dicts[i]["rewards"][1])

            
            actions_a.append(sample_dicts[i]['actions'][0][0])
            outsiders_a.append(sample_dicts[i]['actions'][0][1])
            

            actions_b.append(sample_dicts[i]['actions'][1][0])
            outsiders_b.append(sample_dicts[i]['actions'][1][1])
            
            dones_a.append(sample_dicts[i]['dones'][0])
            dones_b.append(sample_dicts[i]['dones'][1])
            
        
        states_a=torch.cat(states_a,dim=0).to(device=self.device)
        states_b=torch.cat(states_b,dim=0).to(device=self.device)


        outsider_pieces_a=torch.cat(outsider_pieces_a,dim=0).to(self.device)
        outsider_pieces_b=torch.cat(outsider_pieces_b,dim=0).to(self.device)
        

        next_states_a=torch.cat(next_states_a,dim=0).to(self.device)
        next_states_b=torch.cat(next_states_b,dim=0).to(self.device)
        
        
        next_outsider_pieces_a=torch.cat(next_outsider_pieces_a,dim=0).to(self.device)
        next_outsider_pieces_b=torch.cat(next_outsider_pieces_b,dim=0).to(self.device)
        
        
        rewards_a=torch.tensor(rewards_a,dtype=torch.float).view(-1,1).to(self.device)
        rewards_b=torch.tensor(rewards_b,dtype=torch.float).view(-1,1).to(self.device)
        

        actions_a=torch.tensor(actions_a).view(-1,1).to(self.device)
        actions_b=torch.tensor(actions_b).view(-1,1).to(self.device)
        
        
        outsiders_a=torch.tensor(outsiders_a).view(-1,1).to(self.device)
        outsiders_b=torch.tensor(outsiders_b).view(-1,1).to(self.device)
        
        dones_a=torch.tensor(dones_a,dtype=torch.float).view(-1,1).to(self.device)
        dones_b=torch.tensor(dones_b,dtype=torch.float).view(-1,1).to(self.device)


        critic_value_a,critic_value_b=self.critic_model(states_a,states_b)
        next_critic_value_a,next_critic_value_b=self.critic_model(next_states_a,next_states_b)

        td_target_a=rewards_a+self.gamma*next_critic_value_a*(1-dones_a)
        td_delta_a=td_target_a-critic_value_a

        td_target_b=rewards_b+self.gamma*next_critic_value_b*(1-dones_b)
        td_delta_b=td_target_b-critic_value_b

        predicted_action_a=self.actor_model(states_a,outsider_pieces_a)
        predicted_outsider_a=self.outsider_model(states_a)
        action_log_probs_a=torch.log(predicted_action_a.gather(1,actions_a))
        outsider_log_probs_a=torch.log(predicted_outsider_a.gather(1,outsiders_a))

        predicted_action_b=self.actor_model(states_b,outsider_pieces_b)
        predicted_outsider_b=self.outsider_model(states_b)
        action_log_probs_b=torch.log(predicted_action_b.gather(1,actions_b))
        outsider_log_probs_b=torch.log(predicted_outsider_b.gather(1,outsiders_b))

        actor_loss_a=torch.mean(-action_log_probs_a*td_delta_a.detach())
        actor_loss_b=torch.mean(-action_log_probs_b*td_delta_b.detach())
        
        outsider_loss_a=torch.mean(-outsider_log_probs_a*td_delta_a.detach())
        outsider_loss_b=torch.mean(-outsider_log_probs_b*td_delta_b.detach())
        
        critic_loss_a=torch.mean(torch.nn.functional.mse_loss(critic_value_a,td_target_a.detach()))
        critic_loss_b=torch.mean(torch.nn.functional.mse_loss(critic_value_b,td_target_b.detach()))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.outsider_optimizer.zero_grad()
        
        actor_loss_a.backward(retain_graph=True)
        actor_loss_b.backward()

        outsider_loss_a.backward(retain_graph=True)
        outsider_loss_b.backward()
        
        critic_loss_a.backward(retain_graph=True)
        critic_loss_b.backward()

        torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(self.outsider_model.parameters(), max_norm=5.0)

        self.actor_optimizer.step()
        self.outsider_optimizer.step()
        self.critic_optimizer.step()


    def permute(self,cur_permutation,action_index):
        new_permutation=copy.deepcopy(cur_permutation)
        action=list(itertools.combinations(list(range(len(cur_permutation))), 2))[action_index]
        value0=cur_permutation[action[0]]
        value1=cur_permutation[action[1]]
        new_permutation[action[0]]=value1
        new_permutation[action[1]]=value0
        return new_permutation
    
    def cross_permute(self,permutation_a,last_permutation_a,permutation_b,last_permutation_b):
        last_outsider_a=last_permutation_a[-1]
        last_outsider_b=last_permutation_b[-1]
        # print(f"permutation_a: {permutation_a}, last_outsider_a: {last_outsider_a}. permutation_b: {permutation_b}, last_outsider_b: {last_outsider_b}")
        new_permutation_a=copy.deepcopy(permutation_a)
        new_permutation_b=copy.deepcopy(permutation_b)
        a_swap=(permutation_a[-1],last_outsider_a)
        b_swap=(last_outsider_b,permutation_b[-1])
        a_swap_flag=(a_swap[0]!=a_swap[1]) and (-1 not in a_swap)
        b_swap_flag=(b_swap[0]!=b_swap[1]) and (-1 not in b_swap)
        # print(f"a_swap_flag: {a_swap_flag}, b_swap_flag: {b_swap_flag}")
        if -1 in new_permutation_a[0:9]:
            invalid_index=new_permutation_a.index(-1)
            new_permutation_a[invalid_index]=new_permutation_a[-1]
            new_permutation_a[-1]=-1
            
        if -1 in new_permutation_b[0:9]:
            invalid_index=new_permutation_b.index(-1)
            new_permutation_b[invalid_index]=new_permutation_b[-1]
            new_permutation_b[-1]=-1

        if (not a_swap_flag) and (not b_swap_flag):
                return new_permutation_a,new_permutation_b
        if a_swap==b_swap and a_swap_flag and b_swap_flag:
                last_outsider_a_index=permutation_b.index(last_outsider_a)
                last_outsider_b_index=permutation_a.index(last_outsider_b)
                new_permutation_a[last_outsider_b_index]=permutation_b[-1]
                new_permutation_b[last_outsider_a_index]=permutation_a[-1]
                new_permutation_a[-1]=-1
                new_permutation_b[-1]=-1
                return new_permutation_a,new_permutation_b
        if a_swap[0]==b_swap[0] and a_swap_flag and b_swap_flag:
            a_swap_index=new_permutation_b.index(last_outsider_a)
            b_swap_index=last_permutation_b.index(new_permutation_b[-1])
            new_permutation_b[b_swap_index]=new_permutation_a[-1]
            new_permutation_b[a_swap_index]=new_permutation_b[-1]
            new_permutation_a[-1]=-1
            new_permutation_b[-1]=-1
            return new_permutation_a,new_permutation_b

        if a_swap_flag :
                swap_index=new_permutation_b.index(last_outsider_a)
                new_permutation_b[swap_index]=new_permutation_a[-1]
                new_permutation_a[-1]=-1
        if b_swap_flag:
                swap_index=new_permutation_a.index(last_outsider_b)
                new_permutation_a[swap_index]=new_permutation_b[-1]
                new_permutation_b[-1]=-1
        return new_permutation_a,new_permutation_b


    def step(self,epoch=500,load=True):#Change after determined
        if load:
            self.critic_model.load_state_dict(torch.load("Critic"+MODEL_NAME))
            self.actor_model.load_state_dict(torch.load("Actor"+MODEL_NAME))
            self.outsider_model.load_state_dict(torch.load("Outsider"+MODEL_NAME))
            # self.outsider_model.load_state_dict(torch.load("outsider_model.pth"))
        for i in range(epoch):
            if i>300:
                max_step=100
                self.actor_optimizer=torch.optim.Adam(actor_model.parameters(),lr=1e-5)
                self.critic_optimizer=torch.optim.Adam(critic_model.parameters(),lr=1e-5)
                self.outsider_optimizer=torch.optim.Adam(outsider_model.parameters(),lr=1e-4)
            elif i>200:
                max_step=200
                self.actor_optimizer=torch.optim.Adam(actor_model.parameters(),lr=1e-4)
                self.critic_optimizer=torch.optim.Adam(critic_model.parameters(),lr=1e-4)
                self.outsider_optimizer=torch.optim.Adam(outsider_model.parameters(),lr=1e-4)
            elif i>100:
                max_step=300
                self.actor_optimizer=torch.optim.Adam(actor_model.parameters(),lr=1e-3)
                self.critic_optimizer=torch.optim.Adam(critic_model.parameters(),lr=1e-3)
                self.outsider_optimizer=torch.optim.Adam(outsider_model.parameters(),lr=1e-4)
            else:
                self.actor_optimizer=torch.optim.Adam(actor_model.parameters(),lr=1e-3)
                self.critic_optimizer=torch.optim.Adam(critic_model.parameters(),lr=1e-3)
                self.outsider_optimizer=torch.optim.Adam(outsider_model.parameters(),lr=1e-3)
                max_step=400
            self.empty_memory()
            self.load_image(2)
            reward_a_sum=0
            reward_b_sum=0
            done=0
            done_a=0
            done_b=0
            step=0
            initial_permutation=list(range(0,18))
            random.shuffle(initial_permutation)
            permutation_a=initial_permutation[0:9]
            permutation_a.append(-1)

            permutation_b=initial_permutation[9:18]
            permutation_b.append(-1)

            while done!=1 and step<max_step:
                self.outsider_model.eval()
                self.actor_model.eval()
                # last_outsider_a=permutation_a[-1]
                image_a,outsider_a=self.get_image(permutation_a)
                action_probs_a=self.actor_model(image_a,outsider_a)
                action_dist_a=torch.distributions.Categorical(action_probs_a)
                action_a=action_dist_a.sample().item()
                new_permutation_a=self.permute(permutation_a,action_a)
                

                # last_outsider_b=permutation_b[-1]
                image_b,outsider_b=self.get_image(permutation_b)
                action_probs_b=self.actor_model(image_b,outsider_b)
                action_dist_b=torch.distributions.Categorical(action_probs_b)
                action_b=action_dist_b.sample().item()
                new_permutation_b=self.permute(permutation_b,action_b)

                new_permutation_a,new_permutation_b=self.cross_permute(new_permutation_a,permutation_a,new_permutation_b,permutation_b)

                image_a,outsider_a=self.get_image(new_permutation_a)
                outsider_probs_a=self.outsider_model(image_a)
                outsider_dist_a=torch.distributions.Categorical(outsider_probs_a)
                outsider_index_a=outsider_dist_a.sample()

                image_b,outsider_b=self.get_image(new_permutation_b)
                outsider_probs_b=self.outsider_model(image_b)
                outsider_dist_b=torch.distributions.Categorical(outsider_probs_b)
                outsider_index_b=outsider_dist_b.sample()

                if outsider_index_a==9:
                    new_permutation_b[-1]=-1
                else:
                    new_permutation_b[-1]=new_permutation_a[outsider_index_a]

                if outsider_index_b==9:
                    new_permutation_a[-1]=-1
                else:
                    new_permutation_a[-1]=new_permutation_b[outsider_index_b]
                
                reward_a,reward_b,done_a,done_b=self.get_reward(new_permutation_a,new_permutation_b)
                reward_a_sum+=reward_a
                reward_b_sum+=reward_b
                self.recording_buffer(state=permutation_a,action=(action_a,outsider_index_a),reward=reward_a,next_state=new_permutation_a,dones=done_a)
                self.recording_buffer(state=permutation_b,action=(action_b,outsider_index_b),reward=reward_b,next_state=new_permutation_b,dones=done_b)
                self.update()
                if done_a==1 and done_b==1:
                    done=1
                permutation_a=copy.deepcopy(new_permutation_a)
                permutation_b=copy.deepcopy(new_permutation_b)
            print(f"Epoch: {i}. Success: {done==1}, step: {step},reward: {(reward_a_sum/step,reward_b_sum/step)}")
            torch.save(self.critic_model.state_dict(),"Critic"+MODEL_NAME)
            torch.save(self.outsider_model.state_dict(),"Outsider"+MODEL_NAME)
            torch.save(self.actor_model.state_dict(),"Actor"+MODEL_NAME)


    def double_step(self,epoch=500,load=True):#Change after determined
        if load:
            self.critic_model.load_state_dict(torch.load("Critic"+MODEL_NAME))
            self.actor_model.load_state_dict(torch.load("Actor"+MODEL_NAME))
            self.outsider_model.load_state_dict(torch.load("Outsider"+MODEL_NAME))
            # self.outsider_model.load_state_dict(torch.load("outsider_model.pth"))
        for i in range(epoch):
            if i>300:
                max_step=100
                self.actor_optimizer=torch.optim.Adam(self.actor_model.parameters(),lr=1e-5)
                self.critic_optimizer=torch.optim.Adam(self.critic_model.parameters(),lr=1e-4)
                self.outsider_optimizer=torch.optim.Adam(self.outsider_model.parameters(),lr=1e-4)
            elif i>200:
                max_step=200
                self.actor_optimizer=torch.optim.Adam(self.actor_model.parameters(),lr=1e-5)
                self.critic_optimizer=torch.optim.Adam(self.critic_model.parameters(),lr=1e-4)
                self.outsider_optimizer=torch.optim.Adam(self.outsider_model.parameters(),lr=1e-4)
            elif i>100:
                max_step=300
                self.actor_optimizer=torch.optim.Adam(self.actor_model.parameters(),lr=1e-4)
                self.critic_optimizer=torch.optim.Adam(self.critic_model.parameters(),lr=1e-3)
                self.outsider_optimizer=torch.optim.Adam(self.outsider_model.parameters(),lr=1e-4)
            else:
                self.actor_optimizer=torch.optim.Adam(self.actor_model.parameters(),lr=1e-4)
                self.critic_optimizer=torch.optim.Adam(self.critic_model.parameters(),lr=1e-3)
                self.outsider_optimizer=torch.optim.Adam(self.outsider_model.parameters(),lr=1e-3)
                max_step=400
            self.empty_memory()
            self.load_image(2)
            done=0
            done_a=0
            done_b=0
            step=0
            reward_a_sum=0
            reward_b_sum=0
            initial_permutation=list(range(0,18))
            random.shuffle(initial_permutation)
            permutation_a=initial_permutation[0:9]
            permutation_a.append(-1)

            permutation_b=initial_permutation[9:18]
            permutation_b.append(-1)

            while done!=1 and step<max_step:
                self.outsider_model.eval()
                self.actor_model.eval()
                if not done_a:
                    # last_outsider_a=permutation_a[-1]
                    image_a,outsider_a=self.get_image(permutation_a)
                    action_probs_a=self.actor_model(image_a,outsider_a)
                    action_dist_a=torch.distributions.Categorical(action_probs_a)
                    action_a=action_dist_a.sample().item()
                    new_permutation_a=self.permute(permutation_a,action_a)
                
                if not done_b:
                    # last_outsider_b=permutation_b[-1]
                    image_b,outsider_b=self.get_image(permutation_b)
                    action_probs_b=self.actor_model(image_b,outsider_b)
                    action_dist_b=torch.distributions.Categorical(action_probs_b)
                    action_b=action_dist_b.sample().item()
                    new_permutation_b=self.permute(permutation_b,action_b)

                new_permutation_a,new_permutation_b=self.cross_permute(new_permutation_a,permutation_a,new_permutation_b,permutation_b)
                if not done_a:
                    image_a,outsider_a=self.get_image(new_permutation_a)
                    outsider_probs_a=self.outsider_model(image_a)
                    outsider_dist_a=torch.distributions.Categorical(outsider_probs_a)
                    outsider_index_a=outsider_dist_a.sample().item()
                    if outsider_index_a==9:
                        new_permutation_b[-1]=-1
                    else:
                        new_permutation_b[-1]=new_permutation_a[outsider_index_a]
                
                if not done_b:
                    image_b,outsider_b=self.get_image(new_permutation_b)
                    outsider_probs_b=self.outsider_model(image_b)
                    outsider_dist_b=torch.distributions.Categorical(outsider_probs_b)
                    outsider_index_b=outsider_dist_b.sample().item()
                    if outsider_index_b==9:
                        new_permutation_a[-1]=-1
                    else:
                        new_permutation_a[-1]=new_permutation_b[outsider_index_b
                

                ]
                
                reward_a,reward_b,done_a,done_b=self.get_reward(new_permutation_a,new_permutation_b)
                reward_a_sum+=reward_a
                reward_b_sum+=reward_b
                self.double_recording_buffer(state_a=permutation_a,
                                             state_b=permutation_b,
                                             action_a=(action_a,outsider_index_a),
                                             action_b=(action_b,outsider_index_b),
                                             reward_a=reward_a,
                                             reward_b=reward_b,
                                             next_state_a=new_permutation_a,
                                             next_state_b=new_permutation_b,
                                             dones_a=done_a,
                                             dones_b=done_b)
                self.double_update()
                if done_a==1 and done_b==1:
                    done=1
                if step%50==0:
                    print(f"Step: {step}, reward_a: {reward_a}, reward_b:{reward_b}")
                step=step+1
                permutation_a=copy.deepcopy(new_permutation_a)
                permutation_b=copy.deepcopy(new_permutation_b)
            print(f"Epoch: {i}. Success: {done==1}, step: {step},reward: {(reward_a_sum/step,reward_b_sum/step)}")
            torch.save(self.critic_model.state_dict(),"Critic"+MODEL_NAME)
            torch.save(self.outsider_model.state_dict(),"Outsider"+MODEL_NAME)
            torch.save(self.actor_model.state_dict(),"Actor"+MODEL_NAME)



if __name__ == "__main__":
    MODEL_NAME="1.pth"
    critic=critic_model().to(device=DEVICE)
    actor=actor_model(45).to(DEVICE)
    outsider=outsider_model(512,512).to(DEVICE)
    environment=env(train_x=train_x,
                    train_y=train_y,
                    memory_size=1e4,
                    batch_size=5,
                    gamma=0.99,
                    device=DEVICE,
                    actor_model=actor,
                    critic_model=critic,
                    outsider_model=outsider)
    environment.double_step(epoch=50,load=True)
    