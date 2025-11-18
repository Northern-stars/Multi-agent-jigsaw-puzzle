from single_puzzle_env import Env
from local_switcher import Local_switcher
from local_switcher_model import Local_switcher_model
from utils import plot_reward_curve,save_log,read_log
import torch
import numpy as np
import os
import random

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
GAMMA=0.998
BATCH_SIZE=30
MODEL_NAME="LocalSwitcher.pth"
MODEL_PATH=os.path.join("model",MODEL_NAME)
SWAP_NUM=[5,5,5,5]
MAX_STEP=[400,300,200,200]
SHOW_IMAGE=False
LOAD_MODEL=False
TRAIN_PER_STEP=25
EPSILON=0.5
EPSILON_MIN=0.1
EPSILON_GAMMA=0.998
FILE_NAME="_sd2rl_new_env"

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
print(f"Data loaded. Shape: x {train_x.shape}, y {train_y.shape}")



def run_maze(env:Env,local_switcher:Local_switcher,epoch_num=500,load=True):
    if load:
        local_switcher.model.load_state_dict(torch.load(MODEL_PATH))
    reward_record=[]
    done_record=[]
    pending_transition={j:None for j in range(env.image_num)}
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
        reward_sum_list=[[] for _ in range(env.image_num)]
        step=0
        reward_list,_,env.done_list,_=env.get_reward(env.permutation_list)
        done_list=[False for _ in range(env.image_num)]
        while step<max_step and not all(done_list):
            step+=1
            do_list=[]
            for j in range(env.image_num):
                do_list.append(j)
                permutation=env.permutation_list[j]
                permutation_,action=local_switcher.act(permutation,image_index=j)
                if action==local_switcher.action_num-1:
                    action=random.randint(0,local_switcher.action_num-2)
                    permutation_=local_switcher.permute(permutation,action)
            pending_transition[j]=(permutation,action,permutation_)
            env.permutation_list[j]=permutation_
            reward_list,_,done_list,_=env.get_reward(env.permutation_list)
            for j in do_list:
                reward_sum_list[j].append(reward_list[j])
                if pending_transition[j] is not None:
                    cur_state,action,next_state=pending_transition[j]
                    local_switcher.recording_memory(
                        image_id=env.image_id,
                        image_index=j,
                        state=cur_state,
                        action=action,
                        reward=reward_list[j],
                        next_state=next_state,
                        done=done_list[j]
                    )
            
            if SHOW_IMAGE:
                env.show_image(env.permutation_list)
            
            if step%TRAIN_PER_STEP==0:
                local_switcher.update()
        print(f"Epoch: {i}, step: {step}, done: {all(done_list)}, reward: {[sum(reward_sum_list[j])/step if step!=0 else 0 for j in range(len(reward_sum_list)) ]}")
        print(f"Permutation list: {env.permutation_list}")
        reward_record.append([sum(reward_sum_list[j])/len(reward_sum_list[j]) for j in range(len(reward_sum_list)) if len(reward_sum_list[j])!=0 ])
        done_record.append(all(done_list))
        if env.epsilon>EPSILON_MIN:
            env.epsilon*=EPSILON_GAMMA
        local_switcher.update(True)
        torch.save(local_switcher.model.state_dict(),MODEL_PATH)
        save_log("reward"+FILE_NAME,reward_record)
        save_log("done"+FILE_NAME,done_record)
    plot_reward_curve(reward_record,done_record,FILE_NAME)
            
                  
if __name__=="__main__":
    env=Env(train_x,train_y,image_num=1,buffer_size=0,epsilon=EPSILON)
    model=Local_switcher_model(2048,1024,2048,1024,1).to(DEVICE)
    model.load_state_dict(torch.load("model/sd2rl_pretrain.pth"))
    switcher=Local_switcher(model,2000,GAMMA,BATCH_SIZE,env,29)

    run_maze(
        env,
        switcher,
        1000,
        LOAD_MODEL
    )