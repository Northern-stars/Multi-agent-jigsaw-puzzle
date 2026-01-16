import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import os

def plot_reward_curve(reward_record,done_record,file_name):
        avg_reward=[]
        acc=[]
        for i in range(len(reward_record)):
            avg_reward.append(sum(reward_record[i])/len(reward_record[i]))
            acc.append(done_record[i] if i==0 else (acc[i-1]*i+done_record[i])/(i+1))
        plt.ylim((0,1000))
        plt.plot(range(len(avg_reward)),avg_reward)
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.title("Reward Curve")
        plt.savefig(os.path.join("result","reward"+file_name+".png"))
        plt.cla()
        plt.ylim((0,1))
        plt.plot(range(len(acc)),acc)
        plt.xlabel("Episode")
        plt.ylabel("Average accuracy")
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
    plt.savefig(os.path.join("result","confusion_matrix"+filename+".png"))