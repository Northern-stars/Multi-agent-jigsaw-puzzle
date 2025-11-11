import torch
from buffer_switcher_model import Buffer_switcher_model
from env import Env
import random
import copy
import torch.nn as nn
import numpy as np

ACTOR_LR=1e-4
ACTOR_LR_MIN=1e-6
CLIP_GRAD_NORM=0.1
DEVICE="cuda" if torch.cuda.is_available() else "cpu"




class Buffer_switcher:
    def __init__(self,
                 memory_size,
                 model:Buffer_switcher_model,
                 env:Env,
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
        self.action_list=[(i,j) for j in range(self.env.piece_num-1) for i in range(self.env.piece_num-1)]

    def epsilon_greedy(self,action):
        prob=random.random()
        if prob>self.env.epsilon:
            return action
        else:
            epsilon_action=random.randint(0,self.action_num-1)
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

    # def act(self,image,outsider,permutation):
    #     self.model.eval()
    #     with torch.no_grad():
    #         action=torch.argmax(self.model(image,outsider),dim=-1)
    #         action=self.epsilon_greedy(action=action.item())
    #         swap_index,outsider_index=permutation[action],permutation[-1]
    #         permutation[action],permutation[-1]=outsider_index,swap_index
    #     return permutation,action

    def permute(self,cur_permutation_list,action_index):
        # print("Buffer switcher permuting")
        if action_index==self.action_num-1:
            return cur_permutation_list
        new_permutation=copy.deepcopy(cur_permutation_list)
        action=self.action_list[action_index]
        value0=cur_permutation_list[0][action[0]]
        value1=cur_permutation_list[1][action[1]]
        new_permutation[0][action[0]]=value1
        new_permutation[1][action[1]]=value0
        return new_permutation
    
    def choose_action(self,permutation_list):
        perm_=copy.deepcopy(permutation_list)
        value_list=[]
        image1_list=[]
        image2_list=[]
        outsider_list=[]

        for i in range(self.action_num):#get permuted image
            perm_=self.permute(permutation_list,i)
            image1,_=self.env.get_image(perm_[0],image_index=0)
            image2,_=self.env.get_image(perm_[1],image_index=1)
            image1_list.append(copy.deepcopy(image1.cpu()))
            image2_list.append(copy.deepcopy(image2.cpu()))
            # outsider_list.append(copy.deepcopy(outsider.cpu()))
        
        i=0
        with torch.no_grad():#evaluate every action
            while i < self.action_num:
                if self.action_num-i<self.batch_size:
                    image1=torch.cat(image1_list[i:],dim=0).to(DEVICE)
                    image2=torch.cat(image2_list[i:],dim=0).to(DEVICE)
                    # outsider=torch.cat(outsider_list[i:],dim=0).to(DEVICE)
                else:
                    image1=torch.cat(image1_list[i:i+self.batch_size],dim=0).to(DEVICE)
                    image2=torch.cat(image2_list[i:i+self.batch_size],dim=0).to(DEVICE)
                    # outsider=torch.cat(outsider_list[i:i+BATCH_SIZE],dim=0).to(DEVICE)

                value=self.model(image1,image2)
                value_list.append(value.squeeze(-1).to("cpu"))
                i+=self.batch_size
            
            value_list=torch.cat(value_list)
            best_action=torch.argmax(value_list).item()
        return int(best_action)

    def act(self,permutation_list):
        self.model.eval()
        action=self.choose_action(permutation_list)
        action=self.epsilon_greedy(action)
        permutation_=self.permute(cur_permutation_list=permutation_list,action_index=action)
        return permutation_,action
    
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
            
            image1_list=[]
            image2_list=[]
            actions=[]
            next_image1_list=[]
            next_image2_list=[]
            reward=[]
            done=[]

            
            for a in range(len(sample_dicts)):

                cur_permutation_list=sample_dicts[a]["State"]
                image1,_=self.env.request_for_image(image_id=sample_dicts[a]["Image_id"],
                                                         permutation=cur_permutation_list[0],
                                                         image_index=0)
                image2,_=self.env.request_for_image(image_id=sample_dicts[a]["Image_id"],
                                                         permutation=cur_permutation_list[1],
                                                         image_index=1)
                image1_list.append(image1)
                image2_list.append(image2)

                next_permutation_list=sample_dicts[a]["Next_state"]
                image1,_=self.env.request_for_image(image_id=sample_dicts[a]["Image_id"],
                                                         permutation=next_permutation_list[0],
                                                         image_index=0)
                image2,_=self.env.request_for_image(image_id=sample_dicts[a]["Image_id"],
                                                         permutation=next_permutation_list[1],
                                                         image_index=1)
                next_image1_list.append(image1)
                next_image2_list.append(image2)
                

                actions.append(sample_dicts[a]["Action"])
                
                reward.append(sample_dicts[a]["Reward"])
                done.append(sample_dicts[a]["Done"])


                    
            
            image1_tensor=torch.cat(image1_list,dim=0)
            image2_tensor=torch.cat(image2_list,dim=0)
            if image1_tensor.size(0)==1:

                self.model.eval()
                self.main_model.eval()
            else:

                self.model.train()
                self.main_model.train()

            

            next_image1_tensor=torch.cat(next_image1_list,dim=0)
            next_image2_tensor=torch.cat(next_image2_list,dim=0)
            
            action_tensor=torch.tensor(actions).to(DEVICE).unsqueeze(-1)

            reward=torch.tensor(reward,dtype=torch.float32).to(DEVICE).unsqueeze(-1)
            done=torch.tensor(done,dtype=torch.float32).to(DEVICE).unsqueeze(-1) 

            q_main=self.main_model(next_image1_tensor,next_image2_tensor)
            
            q_next=self.model(next_image1_tensor,next_image2_tensor).detach()
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

