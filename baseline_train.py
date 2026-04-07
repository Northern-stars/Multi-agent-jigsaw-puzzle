from env.global_env92 import Env
from model_code.local_switcher_model import Local_switcher_model
from agent.local_switcher import Local_switcher
from utils.utils import save_log,read_log,plot_reward_curve
import numpy as np
import torch
import torch.nn as nn
import os
import random

from pretrain.pretrain_1 import pretrain_model


MODEL_NAME="modulator"
MODEL_PATH=os.path.join("model","LocalSwitcher_92_"+MODEL_NAME+".pth")
# MODEL_PATH=os.path.join("model","LocalSwitcher_92_"+"ef_sole"+".pth")
SWAP_NUM=[5,5,8,8]
MAX_STEP=[200,200,200,200]
SHOW_IMAGE=True
LOAD_MODEL=True
TRAIN_PER_STEP=10
EPSILON=0.3
EPSILON_MIN=0.1
EPSILON_GAMMA=0.998
FILE_NAME="_baseline92_train_{}".format(MODEL_NAME)
GAMMA=0.998
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE=30

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

print(f"Data shape: x {train_x.shape}, y {train_y.shape}")

def run_maze(env:Env,local_switcher:Local_switcher,epoch_num=500,load=True):
    if load:
        local_switcher.model.load_state_dict(torch.load(MODEL_PATH))
    reward_record=[]
    done_record=[]
    pending_trainsition=None
    for epoch in range(epoch_num):
        if epoch > 300:
            max_step, swap_num = MAX_STEP[3], SWAP_NUM[3]
        elif epoch > 200:
            max_step, swap_num = MAX_STEP[2], SWAP_NUM[2]
            # local_switcher.recommand=False
        elif epoch > 100:
            max_step, swap_num = MAX_STEP[1], SWAP_NUM[1]
            # local_switcher.recommand_num=40
        else:
            max_step, swap_num = MAX_STEP[0], SWAP_NUM[0]
            # local_switcher.recommand=False
            # local_switcher.recommand_num=20


        index=random.randint(0,8999)
        random_index=random.randint(0,8999)
        while index//3000==random_index//3000:
            random_index=random.randint(0,8999)
        env.summon_permutation_list(swap_num,id=[index,random_index])
        reward_sum=[]
        step=0
        reward,_,done,_=env.get_reward(env.permutation_list)
        while step<max_step and not done:
            step+=1
            
            permutation=env.permutation_list
            permutation_,action=local_switcher.act(permutation,0)
            if action==local_switcher.action_num-1:
                action=random.randint(0,local_switcher.action_num-2)
                permutation_=local_switcher.permute(permutation,action)
            pending_trainsition=(permutation,action,permutation_)
            env.permutation_list=permutation_
            reward,_,done,_=env.get_reward(env.permutation_list)
            reward_sum.append(reward)
            if pending_trainsition is not None:
                cur_state,action,next_state=pending_trainsition
                local_switcher.recording_memory(
                    image_id=env.image_id,
                    image_index=0,
                    state=cur_state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
            
            if SHOW_IMAGE:
                env.show_image(env.permutation_list)
            
            if step%TRAIN_PER_STEP==0:
                local_switcher.update()
        
        if len(reward_sum)!=0:
            print(f"Epoch: {epoch}, step: {step}, done: {done}, reward: {sum(reward_sum)/len(reward_sum) if step!=0 else 0 }")
            print(f"Permutation list: {env.permutation_list}")
            reward_record.append([sum(reward_sum)/len(reward_sum)])
            done_record.append(done)
        else:
            print(f"Epoch: {epoch}, invalid initial state")
            
        if env.epsilon>EPSILON_MIN:
            env.epsilon*=env.epsilon_gamma
        
        local_switcher.update(True)
        torch.save(local_switcher.model.state_dict(),MODEL_PATH)
        save_log("reward"+FILE_NAME,reward_record)
        save_log("done"+FILE_NAME,done_record)

        plot_reward_curve(reward_record,done_record,FILE_NAME)

def test(test_env:Env,test_local_switcher:Local_switcher,swap_num=5,max_step=50):
    acc=[]
    cate_acc=[]
    hori_acc=[]
    vert_acc=[]
    consistency_acc=[]
    
    sector_number=test_env.sample_number//3
    test_local_switcher.model.load_state_dict(torch.load(MODEL_PATH))
    print(f"Start testing, shape{test_env.image.shape}")

    for i in range(test_env.sample_number//2):
        if i>sector_number:
            index=i+sector_number
        elif i>sector_number//2:
            index=i+sector_number//2
        else:
            index=i

        test_env.summon_permutation_list(swap_num,id=[index,(index+sector_number+1)%test_env.sample_number])
        step=0
        action=-1
        while step<max_step and action!=test_local_switcher.action_num-1:
            step+=1
            
            permutation=test_env.permutation_list
            # print(f"Permutation: {permutation}")
            permutation_,action=test_local_switcher.act(permutation,0)
            if action==test_local_switcher.action_num-1:
                break
            test_env.permutation_list=permutation_

            if SHOW_IMAGE:
                test_env.show_image(test_env.permutation_list)
        done,consistency,cate,hori,vert=test_env.get_accuracy(test_env.permutation_list)
        acc.append(done)
        consistency_acc.append(consistency)
        cate_acc.append(cate)
        hori_acc.append(hori)
        vert_acc.append(vert)
    
        save_log("cate"+FILE_NAME,cate_acc)
        save_log("hori"+FILE_NAME,hori_acc)
        save_log("vert"+FILE_NAME,vert_acc)
        save_log("consistency"+FILE_NAME,consistency_acc)
        save_log("test_done"+FILE_NAME,acc)
        print(f"Final permutation: {test_env.permutation_list[0]}")
        print(f"Test {i}: done:{done}, cate:{cate}, hori:{hori}, vert:{vert}, consistency:{consistency}")
    print(f"Final acc: {np.mean(acc)}, hori: {np.mean(hori_acc)}, vert: {np.mean(vert_acc)}, cate: {np.mean(cate_acc)}")


def model_fen_load(model:nn.Module,model_name):
    hori_pretrain=pretrain_model(512,512,model_name)
    hori_pretrain.load_state_dict(torch.load(f"model/hori_{model_name}.pth"))
    vert_pretrain=pretrain_model(512,512,model_name)
    vert_pretrain.load_state_dict(torch.load(f"model/vert_{model_name}.pth"))
    model.fen_model.hori_ef.load_state_dict(hori_pretrain.ef.state_dict())
    model.fen_model.vert_ef.load_state_dict(vert_pretrain.ef.state_dict())
        

if __name__=="__main__":


    hori_pretrain=pretrain_model(512,512,MODEL_NAME).to(DEVICE)
    hori_pretrain.load_state_dict(torch.load(f"model/hori_{MODEL_NAME}.pth"))
    vert_pretrain=pretrain_model(512,512,MODEL_NAME).to(DEVICE)
    vert_pretrain.load_state_dict(torch.load(f"model/vert_{MODEL_NAME}.pth"))
    env=Env(
        train_x,
        train_y,
        GAMMA,
        image_num=2,
        epsilon=EPSILON,
        epsilon_gamma=EPSILON_GAMMA,
        buffer_size=0,
        greedy_initial=True,
        hori_model=hori_pretrain,
        vert_model=vert_pretrain
    )
    model=Local_switcher_model(512,512,1024,512,1,model_name=MODEL_NAME,dropout=0.3).to(DEVICE)
    # model.load_state_dict(torch.load("model/sd2rl_pretrain.pth"))
    # model.fen_model.ef.load_state_dict(torch.load("model/pairwise_pretrain_ef.pth"))


    model.fen_model.hori_ef.load_state_dict(hori_pretrain.ef.state_dict())
    model.fen_model.vert_ef.load_state_dict(vert_pretrain.ef.state_dict())

    # model=Local_switcher_model(512,512,1024,512,1,model_name="ef_sole",dropout=0.1).to(DEVICE)
    
    switcher=Local_switcher(model,2000,GAMMA,BATCH_SIZE,env,93,recommand=True,recommand_num=20)

    run_maze(
        env,
        switcher,
        500,
        LOAD_MODEL
    )
    test_env=Env(
        test_x,
        test_y,
        GAMMA,
        image_num=2,
        epsilon=0,
        epsilon_gamma=0,
        buffer_size=0
    )

    test_switcher=Local_switcher(model,0,GAMMA,BATCH_SIZE,test_env,93,recommand=False)
    test(test_env,test_switcher)

    # test_done=read_log("test_done"+FILE_NAME)
    # print(f"Test acc: {np.mean(test_done)}")


    # reward_record=read_log("reward"+FILE_NAME)
    # done_record=read_log("done"+FILE_NAME)
    # reward_record=[[a] for a in reward_record]
    # plot_reward_curve(reward_record,done_record,FILE_NAME)
        

