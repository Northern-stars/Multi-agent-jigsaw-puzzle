import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import os
import numpy as np
import sys
sys.path.append("..")
sys.path.append(".")
from pretrain.pretrain_1 import pretrain_model
import torch


def plot_reward_curve(reward_record=None,done_record=None,file_name=""):
        avg_reward=[]
        acc=[]
        reward_flag=reward_record is not None
        done_flag=done_record is not None

        if reward_flag:
            episode_reward=[sum(record)/len(record) for record in reward_record if len(record)!=0]
            for i in range(len(episode_reward)):
                start_index=max(0,i-99)
                avg_reward.append(float(np.mean(episode_reward[start_index:i+1])))

        if done_flag:
            for i in range(len(done_record)):
                start_index=max(0,i-99)
                acc.append(float(np.mean(done_record[start_index:i+1])))

        if reward_flag:
            plt.clf()
            plt.plot(range(len(avg_reward)),avg_reward)
            plt.xlabel("Episode")
            plt.ylabel("Average Reward_"+file_name)
            plt.title("Reward Curve (100-Episode Moving Average)")
            plt.savefig(os.path.join("result","reward"+file_name+".png"))
        
        if done_flag:
            plt.clf()
            plt.plot(range(len(acc)),acc)
            plt.xlabel("Episode")
            plt.ylabel("Average accuracy")
            plt.title("Acc Curve_"+file_name)
            plt.savefig(os.path.join("result","acc"+file_name+".png"))
    
def save_log(file_name,log):
    with open(os.path.join("json",file_name+".json"),"w") as file:
        json.dump(log,file)

def read_log(file_name):
    with open(os.path.join("json",file_name+".json"),"r") as file:
        loaded_data=json.load(file)
    
    return loaded_data

def plot_confusion_matrix(confusion_matrix,category_list,filename):
    fig = plt.figure()
    
    ax = fig.add_subplot(111)
    ax.set_title(filename)
    cax = ax.matshow(confusion_matrix.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xlabel("True")
    ax.set_xticklabels([""]+category_list, rotation=90)
    ax.set_ylabel("Pred")
    ax.set_yticklabels([""]+category_list)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.savefig(os.path.join("result","confusion_matrix_"+filename+".png"))

def model_fen_load(model:torch.nn.Module,model_name):
    hori_pretrain=pretrain_model(512,512,model_name)
    hori_pretrain.load_state_dict(torch.load(f"model/hori_{model_name}.pth"))
    vert_pretrain=pretrain_model(512,512,model_name)
    vert_pretrain.load_state_dict(torch.load(f"model/vert_{model_name}.pth"))
    model.fen_model.hori_ef.load_state_dict(hori_pretrain.ef.state_dict())
    model.fen_model.vert_ef.load_state_dict(vert_pretrain.ef.state_dict())
