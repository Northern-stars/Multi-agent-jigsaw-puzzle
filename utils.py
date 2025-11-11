import matplotlib.pyplot as plt
import json
import os

def plot_reward_curve(reward_record,done_record):
        avg_reward=[]
        acc=[]
        for i in range(len(reward_record)):
            avg_reward.append(sum(reward_record[i])/len(reward_record[i]))
            acc.append(done_record[i] if i==0 else (acc[i-1]*i+done_record[i])/(i+1))
        plt.plot(range(len(avg_reward)),avg_reward)
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.title("Reward Curve")
        plt.savefig(os.path.join("result","reward.png"))
        plt.plot(range(len(acc)),acc)
        plt.xlabel("Episode")
        plt.ylabel("Average accuracy")
        plt.savefig(os.path.join("result","acc.png"))
    
def save_log(file_name,log):
    with open(file_name+".json","w") as file:
        json.dump(log,file)

def read_log(file_name):
    with open(file_name+".json","r") as file:
        loaded_data=json.load(file)
    
    return loaded_data