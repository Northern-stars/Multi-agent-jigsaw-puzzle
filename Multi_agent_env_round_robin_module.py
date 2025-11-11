import torch
import copy
from local_switcher import Local_switcher
from local_switcher_model import Local_switcher_model
import os
from env import Env
from buffer_switcher_sd2rl_64action import Buffer_switcher
from buffer_switcher_model import Buffer_switcher_model
import random
import numpy as np


DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DONE_REWARD=1000
CONSISTENCY_REWARD=200
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
CONSISTENCY_REWARD_WEIGHT=.5

PANELTY=-0.5
ENTROPY_WEIGHT=0.01
ENTROPY_GAMMA=0.998
ENTROPY_MIN=0.005

EPOCH_NUM=2000
LOAD_MODEL=True
SWAP_NUM=[4,4,4,8]
MAX_STEP=[20,20,20,20]
PHASE_SWITCH_NUM=10
MODEL_NAME="(2)_RoundRobin.pth"
MODEL_PATH=os.path.join("model/DQN"+MODEL_NAME)


BATCH_SIZE=20
EPSILON=0.2
EPSILON_GAMMA=0.998
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


def update(local_switcher,buffer_switcher,show=False):
    # print("Updating")
    
    local_switcher.update(show)
    buffer_switcher.update(show)

def clean_memory(local_switcher,buffer_switcher):
    local_switcher.clean_memory()
    buffer_switcher.clean_memory()
    
def save(local_switcher,buffer_switcher):
    torch.save(buffer_switcher.model.state_dict(),os.path.join("model/Buffer_switcher"+MODEL_NAME))
    torch.save(local_switcher.model.state_dict(),os.path.join("model/Local_switcher"+MODEL_NAME))

def load(local_switcher,buffer_switcher):
    buffer_switcher.model.load_state_dict(torch.load(os.path.join("model/Buffer_switcher"+MODEL_NAME)))
    buffer_switcher.main_model=copy.deepcopy(buffer_switcher.model)
    local_switcher.model.load_state_dict(torch.load(os.path.join("model/Local_switcher"+MODEL_NAME)))
    local_switcher.main_model=copy.deepcopy(local_switcher.model)

def local_switch_phase(env:Env,local_switcher:Local_switcher,max_step,test_flag=False):
    terminal_list=[False for _ in range(env.image_num)]
    pending_transition={j:None for j in range(env.image_num)}
    step=0
    while not (all(terminal_list)or (all(env.done_list) and not test_flag) or step>max_step) :
        step+=1
        do_list=[]
        for i in range(env.image_num):
            if env.done_list[i] and not test_flag:
                terminal_list[i]=True
            if terminal_list[i]:
                continue

            do_list.append(i)
            permutation=env.permutation_list[i]
            permutation_,action=local_switcher.act(permutation,image_index=i)

            if action==local_switcher.action_num-1:
                if test_flag:
                    terminal_list[i]=True
                else:
                    action=random.randint(0,local_switcher.action_num-2)
                    permutation_=local_switcher.permute(permutation,action)
            
            pending_transition[i]=(permutation,action,permutation_)
            env.permutation_list[i]=permutation_
        if not test_flag:
            local_reward_list,_,env.done_list,_=env.get_reward(env.permutation_list)
            for i in do_list:
                if pending_transition[i] is not None:
                    cur_state,action,next_state=pending_transition[i]
                    local_switcher.recording_memory(
                        image_id=env.image_id,
                        image_index=i,
                        state=cur_state,
                        action=action,
                        reward=local_reward_list[i],
                        next_state=next_state,
                        done=(env.done_list[i] or terminal_list[i])
                    )
        if SHOW_IMAGE:
            env.show_image(env.permutation_list)
        
        if step%TRAIN_PER_STEP==0:
            local_switcher.update()

            


def buffer_switch_phase(env:Env,buffer_switcher:Buffer_switcher,max_step,test_flag=False):
    step=0
    terminal_flag=False
    while not (terminal_flag or (all(env.consistency_list) and not test_flag) or step>max_step):
        step+=1
        perm_,action=buffer_switcher.act(env.permutation_list)
        if action==buffer_switcher.action_num-1:
            if test_flag:
                terminal_flag=True
            else:
                action=random.randint(0,buffer_switcher.action_num-2)
                perm_=buffer_switcher.permute(env.permutation_list,action)
        if not test_flag:
            _,consistency_reward_list,_,env.consistency_list=env.get_reward(perm_)
            buffer_switcher.recording_memory(
                image_id=env.image_id,
                image_index=None,
                state=env.permutation_list,
                action=action,
                reward=sum(consistency_reward_list),
                next_state=perm_,
                done=(all(env.consistency_list) or terminal_flag)
            )
        env.permutation_list=perm_

        if SHOW_IMAGE:
            env.show_image(env.permutation_list)

        if step%TRAIN_PER_STEP==0:
            buffer_switcher.update()

def run_maze(env:Env,buffer_switcher:Buffer_switcher,local_switcher:Local_switcher,load_flag=True,epoch_num=500):
    if load_flag:
        load(local_switcher=local_switcher,buffer_switcher=buffer_switcher)

    for i in range(epoch_num):
        if i > 300:
                max_step, swap_num = MAX_STEP[3], SWAP_NUM[3]
        elif i > 200:
                max_step, swap_num = MAX_STEP[2], SWAP_NUM[2]
        elif i > 100:
                max_step, swap_num = MAX_STEP[1], SWAP_NUM[1]
        else:
                max_step, swap_num = MAX_STEP[0], SWAP_NUM[0]
        
        env.summon_permutation_list(swap_num)
        # clean_memory(local_switcher,buffer_switcher)
        reward_sum_list= [0 for _ in range(env.image_num)]

        step=0
        _,_,env.done_list,env.consistency_list=env.get_reward(env.permutation_list)
        while step<PHASE_SWITCH_NUM and not (all(env.done_list)):
            step+=1
            buffer_switch_phase(env,buffer_switcher,max_step)
            local_switch_phase(env,local_switcher,max_step)
            local_reward,consistency_reward,env.done_list,env.consistency_list=env.get_reward(env.permutation_list)
            reward_sum_list=[reward_sum_list[j]+local_reward[j]+consistency_reward[j] for j in range(env.image_num)]
            update(local_switcher,buffer_switcher)
        print(f"Epoch: {i}, step: {step}, done: {all(env.done_list)}, reward: {[reward_sum_list[j]/step if step!=0 else 0 for j in range(len(reward_sum_list)) ]}")
        print(f"Permutation list: {env.permutation_list}")
        if env.epsilon>EPSILON_MIN:
            env.epsilon*=EPSILON_GAMMA
        update(
            local_switcher=local_switcher,
            buffer_switcher=buffer_switcher,
            show=True
        )  
        save(
            local_switcher=local_switcher,
            buffer_switcher=buffer_switcher
        )




                

def test(env:Env,buffer_switcher:Buffer_switcher,local_switcher:Local_switcher,test_num=300):
    load(local_switcher=local_switcher,buffer_switcher=buffer_switcher)
    accuracy=[]
    consistency_accuracy=[]
    hori_accuracy=[]
    vert_accuracy=[]
    category_accuracy=[]
    for i in range(test_num):
        env.summon_permutation_list(swap_num=8)
        for _ in range(PHASE_SWITCH_NUM):
            buffer_switch_phase(env,buffer_switcher,max_step=200,test_flag=True)
            local_switch_phase(env,local_switcher,max_step=200,test_flag=True)
            local_reward,consistency_reward,done_list,consistency_list=env.get_reward(env.permutation_list)
            if all(done_list):
                break
        done,consistency,category,hori,vert=env.get_accuracy(env.permutation_list)
        accuracy.append(done)
        consistency_accuracy.append(consistency)
        hori_accuracy.append(hori)
        vert_accuracy.append(vert)
        category_accuracy.append(category)
        print(f"Test num:{i}, id:{env.image_id}, done_accuracy: {np.mean(accuracy)}, consistency_accuracy: {np.mean(consistency_accuracy)}, category_accuracy: {np.mean(category_accuracy)}, hori_accuracy: {np.mean(hori_accuracy)}, vert_accuracy: {np.mean(vert_accuracy)}")






if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    environment=Env(train_x=train_x,
                    train_y=train_y,
                    gamma=GAMMA,
                    image_num=2,
                    buffer_size=1,
                    epsilon=EPSILON,
                    epsilon_gamma=EPSILON_GAMMA)
    # critic=critic_model(hidden_size1=1024,hidden_size2=1024,outsider_hidden_size=256).to(device=DEVICE)

    
    buffer_switcher_model=Buffer_switcher_model(
        hidden_size1=2048,
        hidden_size2=1024,
        hidden_size3=1024,
        action_num=1
    ).to(DEVICE)
    # buffer_switcher_model.load_state_dict(torch.load("model/outsider_switcher_pretrain.pth"))
    buffer_switcher_model.fen_model.load_state_dict(torch.load("model/central_fen.pth"))
    buffer_switcher=Buffer_switcher(
        memory_size=1000,
        model=buffer_switcher_model,
        action_num=65,
        batch_size=BATCH_SIZE,
        train_epoch=AGENT_EPOCHS,
        env=environment
    )

    local_switcher_model=Local_switcher_model(fen_model_hidden1=2048,
                                              fen_model_hidden2=1024,
                                              hidden1=2048,
                                              hidden2=1024,
                                              action_num=1).to(DEVICE)
    local_switcher_model.load_state_dict(torch.load("model/sd2rl_pretrain.pth"))
    local_switcher=Local_switcher(
        memory_size=1000,
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
            local_switcher=local_switcher
            ,buffer_switcher=buffer_switcher
            ,epoch_num=EPOCH_NUM
            ,load_flag=LOAD_MODEL)