from env import Env
from general_switcher import General_switcher
from general_switcher_model import General_switcher_model
from utils import save_log,plot_reward_curve
import torch
import numpy as np
import random
import os
import torch.nn as nn
import copy

from pretrain_1 import pretrain_model


MODEL_NAME="modulator"
MODEL_PATH=os.path.join("model","General_switcher_"+MODEL_NAME+".pth")
SWAP_NUM=[5,5,5,5]
MAX_STEP=[200,200,200,200]
SHOW_IMAGE=True
LOAD_MODEL=False
TRAIN_PER_STEP=10
EPSILON=0.3
EPSILON_MIN=0.1
EPSILON_GAMMA=0.998
FILE_NAME="_double_agent_" + MODEL_NAME
GAMMA=0.995
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE=20
LR=1e-4

train_x_path = 'dataset/train_img_48gap_33-001.npy'
train_y_path = 'dataset/train_label_48gap_33.npy'


# test_x_path = 'dataset/train_img_48gap_33-001.npy'
# test_y_path = 'dataset/train_label_48gap_33.npy'
test_x_path = 'dataset/test_img_48gap_33.npy'
test_y_path = 'dataset/test_label_48gap_33.npy'
# test_x_path = 'dataset/valid_img_48gap_33.npy'
# test_y_path = 'dataset/valid_label_48gap_33.npy'

# train_x=np.load(train_x_path)
# train_y=np.load(train_y_path)

train_x=np.load(train_x_path)
train_y=np.load(train_y_path)
test_x=np.load(test_x_path)
test_y=np.load(test_y_path)

def model_fen_load(model:nn.Module,model_name):
    hori_pretrain=pretrain_model(512,512,model_name)
    hori_pretrain.load_state_dict(torch.load(f"model/hori_{model_name}.pth"))
    vert_pretrain=pretrain_model(512,512,model_name)
    vert_pretrain.load_state_dict(torch.load(f"model/vert_{model_name}.pth"))
    model.fen_model.hori_ef.load_state_dict(hori_pretrain.ef.state_dict())
    model.fen_model.vert_ef.load_state_dict(vert_pretrain.ef.state_dict())

def run_maze(env:Env,switcher:General_switcher,epoch=500, load=True):
        if load:
            switcher.model.load_state_dict(torch.load(MODEL_PATH))
            switcher.main_model=copy.deepcopy(switcher.model)
            switcher.optimizer=torch.optim.Adam(switcher.main_model.parameters(),lr=LR,eps=1e-8)
        reward_record=[]
        done_record=[]
        for i in range(epoch):
            switcher.clean_memory()
            if i > 300:
                max_step, swap_num = MAX_STEP[3], SWAP_NUM[3]
            elif i > 200:
                max_step, swap_num = MAX_STEP[2], SWAP_NUM[2]
                switcher.recommand=False
            elif i > 100:
                max_step, swap_num = MAX_STEP[1], SWAP_NUM[1]
                switcher.recommand_num=20
            else:
                max_step, swap_num = MAX_STEP[0], SWAP_NUM[0]
                switcher.recommand=True
                switcher.recommand_num=10
            env.summon_permutation_list(swap_num=swap_num)
            buffer = [-1] * env.buffer_size
            done_list = [False] * env.image_num
            termination_list = [False for _ in range(env.image_num)]
            reward_sum_list = [[] for _ in range(env.image_num)]
            
            pending_transitions = {j: None for j in range(env.image_num)}

            step = 0
            done = False
            while not done and step < max_step:
                do_list = []  
                # last_reward_list, _ = env.get_reward(permutation_list=env.permutation_list)

                for j in range(env.image_num):
                    if done_list[j] or termination_list[j]:
                        continue

                    perm_with_buf = env.permutation_list[j] + buffer
                    action = switcher.choose_action(perm_with_buf,image_index=j)
                    action=switcher.epsilon_greedy(action=action)
                    if action==env.terminate_action:
                        action=random.randint(0,env.terminate_action-1)  

                    
                    if pending_transitions[j] is not None:
                        prev_state, prev_action, prev_reward = pending_transitions[j]
                        switcher.recording_memory(image_id=env.image_id,image_index=j,state=prev_state,action= prev_action,reward= prev_reward,next_state= perm_with_buf, done=done_list[j])



                    pending_transitions[j] = (perm_with_buf, action, 0) 

                    do_list.append(j)

                    # if action == 36:
                    #     termination_list[j] = True
                    #     continue

                    new_perm = env.permute(perm_with_buf, action)
                    env.permutation_list[j], buffer = new_perm[:env.piece_num-1], new_perm[env.piece_num-1:]

                reward_list,consistency_reward_list, done_list,consistency_list = env.get_reward(permutation_list=env.permutation_list)
                if SHOW_IMAGE:
                    env.show_image(env.permutation_list)


                for j in do_list:
                    prev_state, prev_action, _ = pending_transitions[j]
                    pending_transitions[j] = (prev_state, prev_action, reward_list[j]+consistency_reward_list[j])
                    reward_sum_list[j].append(reward_list[j])

                done = all(done_list)
                step += 1

                if step%TRAIN_PER_STEP==0:
                    switcher.update()


            for j in range(env.image_num):
                if pending_transitions[j] is not None and done_list[j]:
                    prev_state, prev_action, prev_reward = pending_transitions[j]
                    switcher.recording_memory(image_id=env.image_id,image_index=j,state=prev_state,action= prev_action,reward= prev_reward,next_state= perm_with_buf, done=done_list[j])  

            if len(reward_sum_list)!=0:
                print(f"Epoch: {epoch}, step: {step}, done: {done}, reward: {[sum(rs)/len(rs) for rs in reward_sum_list if rs] if step!=0 else 0 }")
                print(f"Permutation list: {env.permutation_list}")
                reward_record.append([sum(rs)/len(rs) for rs in reward_sum_list if rs])
                done_record.append(done)
            switcher.update(show=True) 
            torch.save(switcher.model.state_dict(),MODEL_PATH)
            if env.epsilon>EPSILON_MIN:
                env.epsilon*=env.epsilon_gamma
            save_log("result/reward"+FILE_NAME,reward_record)
            save_log("result/done"+FILE_NAME,done_record)
            plot_reward_curve(reward_record,done_record,FILE_NAME)


if __name__=="__main__":
    env=Env(
        train_x=train_x,
        train_y=train_y,
        gamma=GAMMA,
        image_num=2,
        buffer_size=1,
        epsilon=EPSILON,
        epsilon_gamma=EPSILON_GAMMA,
        terminate_action=36
    )
    model=General_switcher_model(512,1024,256,1,model_name=MODEL_NAME).to(DEVICE)
    model_fen_load(model,MODEL_NAME)
    switcher=General_switcher(model,2000,GAMMA,BATCH_SIZE,env,37,recommand=True,recommand_num=10
                            )
    
    run_maze(env,switcher,epoch=500,load=LOAD_MODEL)