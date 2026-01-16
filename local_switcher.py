from env import Env
import torch
import random
import copy
import itertools
import torch.nn as nn
import numpy as np


ACTOR_LR=1e-4
ACTOR_LR_MIN=1e-6
CLIP_GRAD_NORM=0.1
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
ACTOR_SCHEDULAR_STEP=200

class Local_switcher:
    def __init__(self,
                 model,
                 memory_size,
                 gamma,
                 batch_size,
                 env:Env,
                 action_num,
                 tau=1e-3,
                 recommand=True,
                 recommand_num=10):
        
        self.model=model
        self.main_model=copy.deepcopy(self.model)
        self.optimizer=torch.optim.Adam(self.main_model.parameters(),lr=ACTOR_LR,eps=1e-8)
        self.schedular=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=ACTOR_SCHEDULAR_STEP)
        self.tau=tau
        
        self.memory_size=memory_size
        self.memory=[]
        self.memory_counter=0
        self.gamma=gamma
        self.batch_size=batch_size
        
        self.env=env
        
        self.action_num=action_num


        self.recommand=recommand
        self.recommand_num=recommand_num
    
    def epsilon_greedy(self,action):
        prob=random.random()
        if prob>self.env.epsilon:
            return action
        else:
            epsilon_action=random.randint(0,self.action_num-1)
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
        # print("Local switcher permuting")
        # if action_index==self.action_num-1:
        #     return cur_permutation
        # new_permutation=copy.deepcopy(cur_permutation)
        # action=list(itertools.combinations(list(range(len(cur_permutation))), 2))[action_index]
        # value0=cur_permutation[action[0]]
        # value1=cur_permutation[action[1]]
        # new_permutation[action[0]]=value1
        # new_permutation[action[1]]=value0
        return self.env.permute(cur_permutation,action_index)

    def choose_action(self,permutation,image_index):
        perm_=copy.deepcopy(permutation)
        action_list=[]
        value_list=[]
        image_list=[]
        outsider_list=[]

        if self.recommand:
            action_list=self.recommanded_action(permutation,image_index)
        else:
            action_list=list(range(self.action_num))

        for i in action_list:
            perm_=self.permute(permutation,i)
            image,_=self.env.get_image(perm_,image_index=image_index)
            image_list.append(copy.deepcopy(image.cpu()))
            # outsider_list.append(copy.deepcopy(outsider.cpu()))
        
        i=0
        with torch.no_grad():
            while i < len(image_list):
                if len(image_list)-i<self.batch_size:
                    image=torch.cat(image_list[i:],dim=0).to(DEVICE)
                    # outsider=torch.cat(outsider_list[i:],dim=0).to(DEVICE)
                else:
                    image=torch.cat(image_list[i:i+self.batch_size],dim=0).to(DEVICE)
                    # outsider=torch.cat(outsider_list[i:i+BATCH_SIZE],dim=0).to(DEVICE)

                value=self.model(image)
                value_list.append(value.squeeze(-1).to("cpu"))
                i+=self.batch_size
            
            value_list=torch.cat(value_list)
            best_action=int(torch.argmax(value_list).item())
        return action_list[best_action]

    def recommanded_action(self,permutation,image_index):
        permutation_copy=copy.deepcopy(permutation)
        score_list=np.zeros(self.action_num)
        for i in range(self.action_num):
            permutation_copy=self.permute(permutation,i)
            score_list[i]=self.env.get_local_score(permutation_copy,image_index)
        best_action=(np.argsort(score_list)[::-1]).tolist()
        return best_action[:self.recommand_num]


        


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
