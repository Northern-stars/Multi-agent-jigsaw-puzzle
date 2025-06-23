import time
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import copy
import torch
import itertools
from colorama import init, Fore, Back, Style

from torch import nn
from pytorch_hori_pretrain import fen_model
from torch.utils.tensorboard import SummaryWriter

# color print
init(autoreset=True)
device="cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_ACTION_NUMBER=10

total_reward_list=[]

n_actions=1

LEARNING_RATE = 1e-7
REWARD_CATE_RATE = .8
REWARD_PAIRWISE_RATE = .2
EPISODE_LOSS_RATE = 1.
ACHIEVE_REWARD = 1000.


BATCH_SIZE=5
n_features=16

EPSILON_MAX=0.2
EPSILON_MIN=0.1
GAMMA=0.995
BATCH_SIZE=5

SAMPLE_ACTION_NUMBER=28

TRAIN_NUMBER=200
TEST_NUMBER=300

IS_TEST_SHUFFLE=True

train_x_path = 'dataset/train_img_48gap_33-001.npy'
train_y_path = 'dataset/train_label_48gap_33.npy'

valid_x_path = 'dataset/valid_img_48gap_33.npy'
valid_y_path = 'dataset/valid_label_48gap_33.npy'

test_x_path = 'dataset/test_img_48gap_33.npy'
test_y_path = 'dataset/test_label_48gap_33.npy'

# test_x_path = 'dataset/valid_img_48gap_33.npy'
# test_y_path = 'dataset/valid_label_48gap_33.npy'

# test_x_path = 'dataset/train_img_48gap_33-001.npy'
# test_y_path = 'dataset/train_label_48gap_33.npy'

train_x=np.load(train_x_path)
train_y=np.load(train_y_path)
test_x=np.load(test_x_path)
test_y=np.load(test_y_path)
test_cate_list=np.zeros(9000)
cate_list=np.zeros(9000)
test_hori_list=np.zeros(9000)
hori_list=np.zeros(9000)
test_vert_list=np.zeros(9000)
vert_list=np.zeros(9000)


TARGET_MODEL_NAME="sd2rl512_3x6.pth"
MAIN_MODEL_NAME='sd2rl_main512_3x6.pth'




def show_plot():
    x = list(range(len(total_reward_list)))
    plt.plot(x, total_reward_list, 'o-', color='r')
    plt.show()


def onehot_2_index(onehot_ndarray):
    index_list = []
    for i in onehot_ndarray:
        index = np.argmax(i)
        index_list.append(index)
    return index_list



def get_cur_image(data_img_i1, data_img_i2, reassemble_list):
    #return the general image in current order(permutation)
    # fragment_list1 = []
    # fragment_list2 = []
    #
    # for i in range(9):
    #     fragment_list1.append(data_img_i1[96 * (i // 3): 96 * (i // 3) + 96, 96 * (i % 3): 96 * (i % 3) + 96, :])
    #
    # for i in range(9):
    #     fragment_list2.append(data_img_i2[96 * (i // 3): 96 * (i // 3) + 96, 96 * (i % 3): 96 * (i % 3) + 96, :])
    #
    # combined_fragments = fragment_list1 + fragment_list2

    image = np.concatenate((data_img_i1, data_img_i2), axis=1)

    combined_fragments = [
        image[0:96, 0:96, :],
        image[0:96, 96:192, :],
        image[0:96, 192:288, :],

        image[96:192, 0:96, :],
        image[96:192, 96:192, :],
        image[96:192, 192:288, :],

        image[192:288, 0:96, :],
        image[192:288, 96:192, :],
        image[192:288, 192:288, :],

        image[0:96, 288:384, :],
        image[0:96, 384:480, :],
        image[0:96, 480:576, :],

        image[96:192, 288:384, :],
        image[96:192, 384:480, :],
        image[96:192, 480:576, :],

        image[192:288, 288:384, :],
        image[192:288, 384:480, :],
        image[192:288, 480:576, :]
    ]

    index_list = [0] * 16

    for i in range(16):
        if reassemble_list[i] >= 7 and reassemble_list[i] < 9:
            index_list[i] = reassemble_list[i] + 1
        elif reassemble_list[i] >= 9:
            index_list[i] = reassemble_list[i] + 2
        else:
            index_list[i] = reassemble_list[i]

    index_list.insert(4, 7)
    index_list.insert(13, 10)

    # img_0 = np.concatenate((combined_fragments[index_list[0]],
    #                         combined_fragments[index_list[1]],
    #                         combined_fragments[index_list[2]],
    #                         combined_fragments[index_list[9]],
    #                         combined_fragments[index_list[10]],
    #                         combined_fragments[index_list[11]]), axis=1)
    #
    # img_1 = np.concatenate((combined_fragments[index_list[3]],
    #                         combined_fragments[index_list[4]],
    #                         combined_fragments[index_list[5]],
    #                         combined_fragments[index_list[12]],
    #                         combined_fragments[index_list[13]],
    #                         combined_fragments[index_list[14]]), axis=1)
    #
    # img_2 = np.concatenate((combined_fragments[index_list[6]],
    #                         combined_fragments[index_list[7]],
    #                         combined_fragments[index_list[8]],
    #                         combined_fragments[index_list[15]],
    #                         combined_fragments[index_list[16]],
    #                         combined_fragments[index_list[17]]), axis=1)
    #
    # whole_img = np.concatenate((img_0, img_1, img_2), axis=0)

    whole_img = np.zeros([288, 576, 3], dtype=np.uint8)
    for i in range(18):
        label_i = index_list[i]
        whole_img[(0 + label_i // 6) * 96:(1 + label_i // 6) * 96, (0 + label_i % 6) * 96:(1 + label_i % 6) * 96, :] = \
        combined_fragments[i]

    return whole_img


def get_reassemble_result(cur_loc, target_loc):
    #return the right piece num in total level, horizon, vertex?
    cate, hori, vert = 0, 0, 0
    for i in range(len(cur_loc)):
        if cur_loc[i] == target_loc[i]:
            cate += 1

    LEFT_CENTER = 16
    RIGHT_CENTER = 17

    loc_hori_pairwise = [(cur_loc[0], cur_loc[1]), (cur_loc[1], cur_loc[2]),(cur_loc[8], cur_loc[9]), (cur_loc[9], cur_loc[10]),
                         (cur_loc[3], LEFT_CENTER), (LEFT_CENTER, cur_loc[4]),(cur_loc[11],RIGHT_CENTER),(RIGHT_CENTER,cur_loc[12]),
                         (cur_loc[5], cur_loc[6]), (cur_loc[6], cur_loc[7]),(cur_loc[13], cur_loc[14]), (cur_loc[14], cur_loc[15]),
                         ]

    loc_vert_pairwise = [(cur_loc[0], cur_loc[3]), (cur_loc[1], LEFT_CENTER), (cur_loc[2], cur_loc[4]),    #LEFT
                         (cur_loc[8], cur_loc[11]),(cur_loc[9],RIGHT_CENTER),(cur_loc[10],cur_loc[12]),       #RIGHT
                         (cur_loc[3],cur_loc[5]),(RIGHT_CENTER, cur_loc[6]),(cur_loc[4], cur_loc[7]),   #LEFT
                         (cur_loc[11], cur_loc[13]),(RIGHT_CENTER, cur_loc[14]),(cur_loc[12], cur_loc[15]),  #RIGHT
                         ]


    y_hori_pairwise = [(target_loc[0], target_loc[1]), (target_loc[1], target_loc[2]),(target_loc[8], target_loc[9]), (target_loc[9], target_loc[10]),
                         (target_loc[3], LEFT_CENTER), (LEFT_CENTER, target_loc[4]),(target_loc[11],RIGHT_CENTER),(RIGHT_CENTER,target_loc[12]),
                         (target_loc[5], target_loc[6]), (target_loc[6], target_loc[7]),(target_loc[13], target_loc[14]), (target_loc[14], target_loc[15]),
                         ]

    y_vert_pairwise = [(target_loc[0], target_loc[3]), (target_loc[1], LEFT_CENTER), (target_loc[2], target_loc[4]),    #LEFT
                         (target_loc[8], target_loc[11]),(target_loc[9],RIGHT_CENTER),(target_loc[10],target_loc[12]),       #RIGHT
                         (target_loc[3],target_loc[5]),(RIGHT_CENTER, target_loc[6]),(target_loc[4], target_loc[7]),   #LEFT
                         (target_loc[11], target_loc[13]),(RIGHT_CENTER, target_loc[14]),(target_loc[12], target_loc[15]),  #RIGHT
                         ]

    hori = len(set(loc_hori_pairwise) & set(y_hori_pairwise))
    vert = len(set(loc_vert_pairwise) & set(y_vert_pairwise))
    return cate, hori, vert



class Puzzle_Env(object):
    def __init__(self):
        self.index_list = list(range(16))
        self.initial_fragment_list = []
        self.fragment_list = []
        self.y_true = []
        self.prior_score_list = []
        self.initial_hori_score_list = []
        self.initial_vert_score_list = []
        self.initial_cate_score_list = []
        self.initial_img_list = []

        self.action_space = list(range(n_actions))
        self.n_actions = len(self.action_space)

        # self.reset(fragment_list, y_true)
    

    def reset(self, hori_score, vert_score, cate_score, episode_img, y_true=[]):
            # self.initial_fragment_list = fragment_list.copy()
            # self.fragment_list = fragment_list.copy()
            self.y_true = y_true    # store the ground truth information
            self.initial_hori_score_list = copy.deepcopy(hori_score)
            self.initial_vert_score_list = copy.deepcopy(vert_score)
            self.initial_cate_score_list = copy.deepcopy(cate_score)
            self.initial_img = copy.deepcopy(episode_img)

            self.index_list = list(range(16))

            return self.index_list


    def step(self, a, training=True): #a是action
        # y_predict = [8] * 8
        # for j in range(len(y_predict)):
        #     y_predict[self.index_list[j]] = j
        # cur_image = get_cur_image(self.initial_img, self.index_list)
        # TE_res = TE_model.predict(np.expand_dims(cur_image, axis=0))[0]
        # print("step res: ", TE_res, y_predict == self.y_true)
        # if TE_res > 0.8:
        #     if y_predict == self.y_true:
        #         reward = ACHIEVE_REWARD
        #         return self.index_list, reward, True, 8, 6, 6
        #     else:
        #         cate, hori, vert = get_reassemble_result(self.index_list, self.y_true)
        #         reward = cate * REWARD_CATE_RATE + (hori + vert) * REWARD_PAIRWISE_RATE
        #         return self.index_list, reward, True, cate, hori, vert
        cur_index_list = copy.deepcopy(self.index_list)
        if a < 120:
            all_possible_actions = list(itertools.combinations(range(16), 2))
            pos1, pos2 = all_possible_actions[a]
            cur_index_list[pos1], cur_index_list[pos2] = cur_index_list[pos2], cur_index_list[pos1]

            self.index_list = copy.deepcopy(cur_index_list)
        

        loc = [8] * 16
        for i in range(len(cur_index_list)):
            loc[cur_index_list[i]] = i

        if training:
            if loc == self.y_true:
                reward = ACHIEVE_REWARD
                return cur_index_list, reward, True, 16, 12, 12
            else:
                cate, hori, vert = get_reassemble_result(loc, self.y_true)  #Compare the result and get reward
                # reward = cate / 8 * REWARD_CATE_RATE + (hori + vert) / 12 * REWARD_PAIRWISE_RATE - EPISODE_LOSS_RATE
                reward = cate * REWARD_CATE_RATE + (hori + vert) * REWARD_PAIRWISE_RATE - EPISODE_LOSS_RATE
                #Reward calculation, cate,hori&vert, Episode_loss is used to rush the process

                return cur_index_list, reward, False, cate, hori, vert

        else:
            cate, hori, vert = get_reassemble_result(loc, self.y_true)
            #Evaluation
            return cur_index_list, 0, False, cate, hori, vert
            # cur_index_list作为observation传出

    def index_list_2_fragment_list(self):
        fragment_list = self.initial_fragment_list.copy()
        for i in range(len(self.index_list)):
            fragment_list[i] = self.initial_fragment_list[self.index_list[i]]
        return fragment_list
    



#Model structure
class dqn_model(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        
        self.FEN_hori=fen_model(512,512)
        self.FEN_vert=fen_model(512,512)

        state_dict=torch.load("hori_ef0.pth")
        hori_state_dict = {
        k.replace("pre_model.", ""): v 
        for k, v in state_dict.items() 
        if k.startswith("pre_model.")
        }
        # hori_state_dict=state_dict
        load_result_hori=self.FEN_hori.load_state_dict(hori_state_dict,strict=False)
        print("missing keys hori",load_result_hori.missing_keys)
        print("unexpected keys hori",load_result_hori.unexpected_keys)
        
        state_dict=torch.load("vert_ef0.pth")
        vert_state_dict = {
        k.replace("pre_model.", ""): v 
        for k, v in state_dict.items() 
        if k.startswith("pre_model.")
        }
        load_result_vert=self.FEN_vert.load_state_dict(vert_state_dict,strict=False)
        print("missing keys vert",load_result_vert.missing_keys)
        print("unexpected keys vert",load_result_vert.unexpected_keys)


        self.fc_0=nn.Linear(512,128)
        self.fc_1=nn.Linear(128*24,hidden_size)   #128*12 -> 128*24
        self.dropout=nn.Dropout(0.1)
        self.bn1=nn.BatchNorm1d(hidden_size)
        self.relu=nn.ReLU()
        self.fc_2=nn.Linear(hidden_size,1)
    
    def forward(self,image):
        #image [batch,3,288,576]
        fragment_0=image[:,:,0:96,0:96]
        fragment_1=image[:,:,0:96,96:192]
        fragment_2=image[:,:,0:96,192:288]

        fragment_3=image[:,:,96:192,0:96]
        fragment_4=image[:,:,96:192,96:192]
        fragment_5=image[:,:,96:192,192:288]

        fragment_6=image[:,:,192:288,0:96]
        fragment_7=image[:,:,192:288,96:192]
        fragment_8=image[:,:,192:288,192:288]


        fragment_9=image[:,:,0:96,288:384]
        fragment_10=image[:,:,0:96,384:480]
        fragment_11=image[:,:,0:96,480:576]

        fragment_12=image[:,:,96:192,288:384]
        fragment_13=image[:,:,96:192,384:480]
        fragment_14=image[:,:,96:192,480:576]

        fragment_15=image[:,:,192:288,288:384]
        fragment_16=image[:,:,192:288,384:480]
        fragment_17=image[:,:,192:288,480:576]



        hori_feature_0_1 = self.FEN_hori(torch.cat([fragment_0, fragment_1], axis=2))
        hori_feature_0_1 = self.fc_0(hori_feature_0_1)
        hori_feature_1_2 = self.FEN_hori(torch.cat([fragment_1, fragment_2], axis=2))
        hori_feature_1_2 = self.fc_0(hori_feature_1_2)
        hori_feature_3_4 = self.FEN_hori(torch.cat([fragment_3, fragment_4], axis=2))
        hori_feature_3_4 = self.fc_0(hori_feature_3_4)
        hori_feature_4_5 = self.FEN_hori(torch.cat([fragment_4, fragment_5], axis=2))
        hori_feature_4_5 = self.fc_0(hori_feature_4_5)
        hori_feature_6_7 = self.FEN_hori(torch.cat([fragment_6, fragment_7], axis=2))
        hori_feature_6_7 = self.fc_0(hori_feature_6_7)
        hori_feature_7_8 = self.FEN_hori(torch.cat([fragment_7, fragment_8], axis=2))
        hori_feature_7_8 = self.fc_0(hori_feature_7_8)
        hori_feature_9_10 = self.FEN_hori(torch.cat([fragment_9, fragment_10], axis=2))
        hori_feature_9_10 = self.fc_0(hori_feature_9_10)
        hori_feature_10_11 = self.FEN_hori(torch.cat([fragment_10, fragment_11], axis=2))
        hori_feature_10_11 = self.fc_0(hori_feature_10_11)
        hori_feature_12_13 = self.FEN_hori(torch.cat([fragment_12, fragment_13], axis=2))
        hori_feature_12_13 = self.fc_0(hori_feature_12_13)
        hori_feature_13_14 = self.FEN_hori(torch.cat([fragment_13, fragment_14], axis=2))
        hori_feature_13_14 = self.fc_0(hori_feature_13_14)
        hori_feature_15_16 = self.FEN_hori(torch.cat([fragment_15, fragment_16], axis=2))
        hori_feature_15_16 = self.fc_0(hori_feature_15_16)
        hori_feature_16_17 = self.FEN_hori(torch.cat([fragment_16, fragment_17], axis=2))
        hori_feature_16_17 = self.fc_0(hori_feature_16_17)


        hori_feature = torch.cat([hori_feature_0_1, hori_feature_1_2,
                                  hori_feature_3_4, hori_feature_4_5,
                                  hori_feature_6_7, hori_feature_7_8,
                                  hori_feature_9_10,hori_feature_10_11,
                                  hori_feature_12_13,hori_feature_13_14,
                                  hori_feature_15_16,hori_feature_16_17], dim=1)


        vert_feature_0_3 = self.FEN_vert(torch.cat([fragment_0, fragment_3],axis=3))
        vert_feature_0_3=self.fc_0(vert_feature_0_3)
        vert_feature_1_4 = self.FEN_vert(torch.cat([fragment_1, fragment_4],axis=3))
        vert_feature_1_4=self.fc_0(vert_feature_1_4)
        vert_feature_2_5 = self.FEN_vert(torch.cat([fragment_2, fragment_5],axis=3))
        vert_feature_2_5=self.fc_0(vert_feature_2_5)
        vert_feature_3_6 = self.FEN_vert(torch.cat([fragment_3, fragment_6],axis=3))
        vert_feature_3_6=self.fc_0(vert_feature_3_6)
        vert_feature_4_7 = self.FEN_vert(torch.cat([fragment_4, fragment_7],axis=3))
        vert_feature_4_7=self.fc_0(vert_feature_4_7)
        vert_feature_5_8 = self.FEN_vert(torch.cat([fragment_5, fragment_8],axis=3))
        vert_feature_5_8=self.fc_0(vert_feature_5_8)
        vert_feature_9_12 = self.FEN_vert(torch.cat([fragment_9, fragment_12],axis=3))
        vert_feature_9_12=self.fc_0(vert_feature_9_12)
        vert_feature_10_13 = self.FEN_vert(torch.cat([fragment_10, fragment_13],axis=3))
        vert_feature_10_13=self.fc_0(vert_feature_10_13)
        vert_feature_11_14 = self.FEN_vert(torch.cat([fragment_11, fragment_14],axis=3))
        vert_feature_11_14=self.fc_0(vert_feature_11_14)
        vert_feature_12_15 = self.FEN_vert(torch.cat([fragment_12, fragment_15],axis=3))
        vert_feature_12_15=self.fc_0(vert_feature_12_15)
        vert_feature_13_16 = self.FEN_vert(torch.cat([fragment_13, fragment_16],axis=3))
        vert_feature_13_16=self.fc_0(vert_feature_13_16)
        vert_feature_14_17 = self.FEN_vert(torch.cat([fragment_14, fragment_17],axis=3))
        vert_feature_14_17=self.fc_0(vert_feature_14_17)

        vert_feature = torch.cat([vert_feature_0_3, vert_feature_1_4, vert_feature_2_5,
                                        vert_feature_3_6, vert_feature_4_7, vert_feature_5_8,
                                        vert_feature_9_12, vert_feature_10_13, vert_feature_11_14,
                                        vert_feature_12_15,vert_feature_13_16,vert_feature_14_17],dim=1)

        concated_feature = torch.cat([hori_feature, vert_feature], dim=1)

        out=self.fc_1(concated_feature)
        out=self.dropout(out)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.fc_2(out)
        return out

target_DQN=dqn_model(256).to(device)
main_DQN=dqn_model(256).to(device)

state_dict=target_DQN.state_dict()
main_DQN.load_state_dict(state_dict)

loss_fn=torch.nn.MSELoss()
optimizer=torch.optim.Adam(main_DQN.parameters(),lr=1e-7)




class Agent:
    # define the agent
    def __init__(self,
                 reward_decay=0.9,
                 replace_target_iter=2,
                 memory_size=1000,
                 batch_size=BATCH_SIZE,
                 epsilon=0,
                 epsilon_min=0,
                 epsilon_decay=0.995,
                 tau=0.01
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, index1, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 3))

        self.cost = []

        # self.mainQN = NetWork()
        # self.targetQN = NetWork()

        # self.mainQN.build(input_shape=(None, self.n_features))
        # self.mainQN.summary()
        #
        # self.targetQN.build(input_shape=(None, self.n_features))
        # self.targetQN.summary()


    def store_transition(self, s, a, r, episode, s_):
        #Buffer?
        #s is current state, a represents action, r is reward and s_ is the next state
        if not hasattr(self, 'memory_counter'):  # hasattr:Check if the return object has named attributes
            self.memory_counter = 0

        transition = np.hstack((s, [a, r, episode], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1


    def empty_memory(self):
        self.memory = np.zeros((self.memory_size, n_features * 2 + 3))
        self.memory_counter = 0

    
    # input obs, return estimate Q from network
    # def evaluate(self, image1, image2, obs):
    #     # t = time.time()
    #     eval_img_list = torch.tensor(get_cur_image(image1,image2,obs[0]),dtype=torch.float32).unsqueeze(0)
    #     for i in range(1,len(obs)):
    #         eval_img_list=torch.cat([eval_img_list,torch.tensor(get_cur_image(image1,image2,obs[i]),dtype=torch.float32).unsqueeze(0)])
    #     eval_img_list=eval_img_list.permute([0,3,1,2]).to(device)
    #
    #     result = target_DQN(eval_img_list)
    #     # print(time.time()-t)
    #     return result

    def evaluate(self, image1, image2, obs):
        target_DQN.eval()
        with torch.no_grad():
            eval_img_list = torch.zeros((len(obs), 3, 288, 576), dtype=torch.float32, device=device)
            for i in range(len(obs)):
                eval_img_list[i] = torch.tensor(get_cur_image(image1, image2, obs[i]), dtype=torch.float32).permute(2,0,1)
            result = target_DQN(eval_img_list)
        return result

    def get_obs(self, p_index_list, a):
        #get the swap pair and return swaped index_list?

        index_list = copy.deepcopy(p_index_list)
        action_tuple = list(itertools.combinations(list(range(16)), 2))[a]
        value_0 = index_list[action_tuple[0]]
        value_1 = index_list[action_tuple[1]]
        index_list[action_tuple[0]] = value_1
        index_list[action_tuple[1]] = value_0
        return index_list

    def choose_action(self, observation, episode_hori, episode_vert, episode_cate, episode_img1, episode_img2):
        #Evaluate every action in best_action list and with epsilon-greedy choose action
        '''
        :param observation: the sequence of the fragments
        :param episode_hori:
        :param episode_vert:
        :param episode_cate:
        :param episode_img:
        :param index_list:
        :return: the action index
        '''
        #best_action_list = get_best_actions_list(episode_hori, episode_vert, episode_cate, observation)
        best_action_list=range(120)
        if np.random.uniform() > self.epsilon:  #epsilon greedy
            # forward feed the observation and get q value for every actions
            # obs_image = get_cur_image(episode_img, observation)
            # cur_value = main_DQN.call(np.expand_dims(obs_image, axis=0))[0]

            obs_list = []
            # greedy choose
            for action in best_action_list:
                # observation_, reward, done = env.step(action)
                # q_value = main_DQN.call(np.array(observation_)[np.newaxis, :])
                # obs = self.get_obs(episode_hori.tolist(), episode_vert.tolist(), episode_cate, index_list, action)
                obs = self.get_obs(observation, action)
                # q_value = self.evaluate(obs)

                # q_value = q_value.numpy()
                obs_list.append(obs)

            actions_value = self.evaluate(episode_img1, episode_img2, obs_list)
            if torch.argmax(actions_value) <= SAMPLE_ACTION_NUMBER:
                action = best_action_list[torch.argmax(actions_value)]
        else:
            action_index = np.random.randint(SAMPLE_ACTION_NUMBER)  
            action = best_action_list[action_index]
        return action
    
    def train(self):

        #soft update

        # mainQN_weights=main_DQN.state_dict()
        # targetQN_weights=target_DQN.state_dict()
        # op_holder={}

        # for key,var in mainQN_weights.items():

        #     op_holder[key]=(var*self.tau)+(1-self.tau)*targetQN_weights[key]
        # target_DQN.load_state_dict(op_holder)
        for target_param, main_param in zip(target_DQN.parameters(),main_DQN.parameters()):
            target_param.data.copy_(self.tau*main_param.data+(1-self.tau)*target_param.data)
        

        #select batch_size sample
        
        if self.memory_counter>self.memory_size:
            sample_index=np.random.choice(a=np.arange(self.memory_size),size=self.batch_size)
        else:
            sample_index=np.random.choice(a=np.arange(self.memory_counter),size=self.batch_size)
        
        batch_memory=self.memory[sample_index,:]
        # print(len(batch_memory))
        # print(len(batch_memory[0]))
        #batch_memory:size 0 is the batch size
        #For size 1, 0:self.n_features is state, n_feature+1 is reward, n_feature is action,n_feature+2 is episode index the rest is next state


        #loss
        #Rearrange this part yourself!
        #estimate current state q
        target_image_list = torch.zeros([1,288,576,3])  #modified
        for i in range(len(batch_memory)):
            target_image_index=batch_memory[i,-self.n_features:]
            episode_index=int(batch_memory[i,self.n_features+2])

            img1 = train_x[episode_index]
            img2 = train_x[episode_index + 1]

            target_image_list=torch.cat([
                target_image_list,
                torch.tensor(get_cur_image(img1, img2, [int(x) for x in target_image_index])).unsqueeze(0)],axis=0)

        target_image_list=target_image_list[1:,:,:,:]
        target_image_list=target_image_list.permute([0,3,1,2]).to(device)
        # print(target_image_list.size())
        target_image_list=target_image_list.to(device)
        q_next=target_DQN(target_image_list)
        q_eval=main_DQN(target_image_list)

        #estimate next state q from reward

        # q_target=q_eval.detach().numpy().copy()
        # batch_index=np.arange(self.batch_size,dtype=np.int32)
        # eval_act_index = batch_memory[:, self.n_features].astype(int)
        # reward=batch_memory[:,self.n_features+1]
        # print(q_next.size())
        # q_target[batch_index,0]=reward[batch_index][0]+self.gamma*q_next[batch_index][0]


        reward=torch.tensor(batch_memory[:,self.n_features+1]).unsqueeze(1).to(device)
        # print(reward.size())
        q_target=reward+self.gamma*q_next
        q_target=q_target.to(torch.float)
        # print(q_target.type(),q_eval.type())

        #back_propogation
        loss=loss_fn(q_target,q_eval)

        optimizer.zero_grad()
        loss.float().backward()
        optimizer.step()
        self.cost.append(loss.item())

        self.learn_step_counter+=1

        if self.learn_step_counter%50==0:
            print(Fore.BLUE + f"step:{self.learn_step_counter},  loss:{loss.item()}")

    
def run_maze(load=False,start_phase_lr=1e-4,middle_phase_lr=1e-4,final_phase_lr=1e-5):
    #This is the main train function
    step = 0
    # writer = SummaryWriter('runs/puzzle_experiment')
    # np.random.seed(1)

    train_id_list = np.arange(len(train_y)-1)
    # print("y shape:",train_y.shape)
    # print(len(hori_list))

    np.random.shuffle(train_id_list)
    global optimizer
    if load:
        target_state_dict=torch.load(TARGET_MODEL_NAME)
        target_DQN.load_state_dict(target_state_dict)
        main_state_dict=torch.load(MAIN_MODEL_NAME)
        main_DQN.load_state_dict(main_state_dict)

    for i in range(TRAIN_NUMBER):
        # RL.empty_memory()
        if i > 300:
            global SAMPLE_ACTION_NUMBER
            SAMPLE_ACTION_NUMBER = 28
            STOP_STEP = 300
            optimizer=torch.optim.Adam(main_DQN.parameters(),lr=final_phase_lr,eps=1e-8)
        elif i > 100:
            SAMPLE_ACTION_NUMBER = 28
            STOP_STEP =300
            optimizer=torch.optim.Adam(main_DQN.parameters(),lr=middle_phase_lr,eps=1e-8)
            print(i)
        else:
            SAMPLE_ACTION_NUMBER = 28
            STOP_STEP = 400
            optimizer=torch.optim.Adam(main_DQN.parameters(),lr=start_phase_lr,eps=1e-8)
            print(i)


        pair_idx = train_id_list[i % len(train_id_list)]

        img1 = train_x[pair_idx]
        img2 = train_x[pair_idx + 1]
        combined_img = np.concatenate((img1, img2),axis=1)

        y_true_img1 = onehot_2_index(train_y[pair_idx])
        y_true_img2 = onehot_2_index(train_y[pair_idx + 1])
        combined_y_true = y_true_img1 + [x + 8 for x in y_true_img2]  #combined label

        label_mapping = {
            0: 0, 1: 1, 2: 2,
            8: 3, 9: 4, 10: 5,
            3: 6, 4: 7,
            11: 8, 12: 9,
            5: 10, 6: 11, 7: 12,
            13: 13, 14: 14, 15: 15,
        }

        new_label = [0 for i in range(16)]
        for j in range(len(combined_y_true)):
            new_label[j] = label_mapping[combined_y_true[j]]

        init_index_list = env.reset(hori_list[pair_idx], vert_list[pair_idx], cate_list[pair_idx],
                                    combined_img, y_true=new_label)

        observation = copy.deepcopy(init_index_list)
        total_reward = 0.
        p_step = 0

        # image = get_cur_image(img1, img2, new_label)
        # cv2.imshow("wwww", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        while True:
            # RL choose action based on observation
            action = RL.choose_action(observation, hori_list[pair_idx], vert_list[pair_idx],
                                      cate_list[pair_idx], img1, img2)
            # RL take action and get next observation and reward
            observation_, reward, done, cate, _, _ = env.step(action)
            total_reward += reward
            RL.store_transition(observation, action, reward, pair_idx, observation_)

            # if (step > 200) and (step % 5 == 0):
            # if (step <= 10) or (step % 5 == 0) or done:
            if (step % 5 == 4):
                    RL.train()

            # swap observation
            observation = copy.deepcopy(observation_)

            # break while loop when end of this episode
            if done:
                RL.train()
                print(Fore.LIGHTGREEN_EX + f"{str(i)} \tsuccess\tstep length:  {str(p_step)}")
                # target_DQN.set_weights(main_DQN.get_weights())
                break
            elif p_step >= STOP_STEP - 1:
                print(Fore.LIGHTGREEN_EX + f"{str(i)} \tunsuccess\tstep length:  {str(p_step)}")
                break
            step += 1
            p_step += 1

            if p_step % 100 == 0:
                print("step:",p_step)
                print("reward: ",reward)
                print("total reward: ",total_reward/p_step)

                # # 获取当前拼接状态
                # current_img = get_cur_image(img1, img2, observation)  # observation 是当前状态
                # # 转换为 Tensor 并调整维度 (H,W,C) -> (C,H,W)
                # img_tensor = torch.tensor(current_img).permute(2, 0, 1).float() / 255.0
                # # 记录到 TensorBoard
                # writer.add_image('Puzzle/Current_State', img_tensor, p_step)
                #

                # target_img = get_cur_image(img1, img2, env.y_true)# y_true 是目标状态
                #
                # target_tensor = torch.tensor(target_img).permute(2, 0, 1).float() / 255.0
                # writer.add_image('Puzzle/Target_State', target_tensor, p_step)



        total_reward_list.append(total_reward / (p_step + 1))
        print(Fore.LIGHTRED_EX + f"Episode Id: {str(pair_idx)} \tTotal Reward:  {total_reward / (p_step + 1)}")
        torch.save(target_DQN.state_dict(),TARGET_MODEL_NAME)
        torch.save(main_DQN.state_dict(),MAIN_MODEL_NAME)

        # update epsilon
        RL.epsilon = RL.epsilon * RL.epsilon_decay if RL.epsilon > RL.epsilon_min else RL.epsilon_min
        # print(RL.epsilon)

        # show_plot()

    # end of game
    torch.save(target_DQN.state_dict(),TARGET_MODEL_NAME)
    torch.save(main_DQN.state_dict(),MAIN_MODEL_NAME)
    print('game over')



def evaluate_test_set():
    global TARGET_MODEL_NAME
    target_DQN.load_state_dict(torch.load(TARGET_MODEL_NAME))
    target_DQN.eval()
    success = 0
    whole_cate = 0
    whole_hori = 0
    whole_vert = 0
    sequence_list = []
    test_id_list = np.arange(len(test_y))
    if IS_TEST_SHUFFLE:
        # np.random.seed(2)
        np.random.shuffle(test_id_list)
    with torch.no_grad():
        for i in range(TEST_NUMBER):
            pair_idx = i % (len(test_y))

            img1 = train_x[pair_idx]
            img2 = train_x[pair_idx + 1]
            combined_img = np.concatenate((img1, img2), axis=1)

            y_true_img1 = onehot_2_index(test_y[pair_idx])
            y_true_img2 = onehot_2_index(test_y[pair_idx + 1])
            combined_y_true = y_true_img1 + [x + 8 for x in y_true_img2]

            label_mapping = {
                0: 0, 1: 1, 2: 2,
                8: 3, 9: 4, 10: 5,
                3: 6, 4: 7,
                11: 8, 12: 9,
                5: 10, 6: 11, 7: 12,
                13: 13, 14: 14, 15: 15,
            }

            new_label = [0 for i in range(16)]
            for i in range(len(combined_y_true)):
                new_label[i] = label_mapping[combined_y_true[i]]

            # episode = test_id_list[i]
            # y_true = onehot_2_index(test_y[episode])
            index_list = env.reset(test_hori_list[pair_idx], test_vert_list[pair_idx],
                                test_cate_list[pair_idx], combined_img, y_true=new_label)

            observation = copy.deepcopy(index_list)

            # observation, _, _, cate, hori, vert = env.step(28)
            obs_img = get_cur_image(test_x[pair_idx], observation)
            obs_img=torch.tensor(obs_img).permute([2,0,1])
            obs_img=obs_img.unsqueeze(0).to(torch.float).to(device)
            best_reward = target_DQN(obs_img)
            # best_reward = 0.
            step = 0
            # step, cate, hori, vert = 0, 0, 0, 0
            done = False
            while True:
                if step >= TEST_STEP:
                    observation_, _, done_, cate_, hori_, vert_ = env.step(120)
                    if done_:
                        success += 1
                        print(str(i), "\tsuccess\tstep length: ", str(step), "\tsuccess rate\t",
                            str(success / (i + 1)))
                    else:
                        print(str(i), "\tunsuccess\tstep length: ", str(step))
                    whole_cate += cate_
                    whole_hori += hori_
                    whole_vert += vert_
                    break
                else:
                    # RL choose action based on observation
                    action = RL.choose_action(observation, test_hori_list[pair_idx], test_vert_list[pair_idx],
                                            test_cate_list[pair_idx], img1, img2)

                    # RL take action and get next observation and reward
                    _, _, done, cate, hori, vert = env.step(120)
                    observation_, _, done_, cate_, hori_, vert_ = env.step(action)
                    # old_obs_img =get_cur_image(test_x[episode],observation)
                    obs_img_ = get_cur_image(test_x[pair_idx], observation_)
                    # old_obs_img=torch.tensor(old_obs_img).permute([2,0,1]).unsqueeze(0).to(torch.float).to(device)
                    obs_img_=torch.tensor(obs_img_).permute([2,0,1]).unsqueeze(0).to(torch.float).to(device)
                    rl_reward = target_DQN(obs_img_)
                    # rl_reward = sum(observation_)

                    if rl_reward >= best_reward:
                        best_reward = rl_reward
                        observation = copy.deepcopy(observation_)
                        done = done_
                    # if not done_:
                    #     best_reward = rl_reward
                    #     observation = copy.deepcopy(observation_)
                    #     done = done_


                    # break while loop when end of this episode
                    else:
                        if done:# change to done if switch back
                            success += 1
                            print(str(i), "\tsuccess\tstep length: ", str(step), "\tsuccess rate\t",
                                str(success / (i + 1)))
                        else:
                            loc=[8]*8
                            for j in range(len(env.index_list)):
                                loc[env.index_list[j]]=j
                            # print(f"loc:{loc}, true{env.y_true}")
                            print(str(i), "\tunsuccess\tstep length: ", str(step))
                        # _, _, _, cate, hori, vert = env.step(28)
                        whole_cate += cate
                        whole_hori += hori
                        whole_vert += vert
                        break
                step += 1

    print('Success Number', success)
    print('Success Cate', whole_cate/(TEST_NUMBER*8))
    print('Success Hori', whole_hori)
    print('Success Vert', whole_vert)
    # np.save('resnet2f_05_sequence_1.npy', sequence_list)



if __name__ == "__main__":
    # maze game
    env = Puzzle_Env()
    SAMPLE_ACTION_NUMBER = 120
    TARGET_MODEL_NAME='sd2rl512_pretrain.pth'
    
    MAIN_MODEL_NAME='sd2rl_main512_pretrain.pth'
    EPSILON_MAX=0.9
    EPSILON_MIN=0.1
    

    RL = Agent(reward_decay=0.995,
               replace_target_iter=2,
               memory_size=2000,
               epsilon=EPSILON_MAX,
               epsilon_min=EPSILON_MIN,
               epsilon_decay=GAMMA
               )  
    run_maze(load=True,start_phase_lr=1e-4,middle_phase_lr=1e-4,final_phase_lr=1e-5)

    # target_DQN.save_weights('RL_weight/RL_SD2RL_3_512feature_model_weight_1')

    TEST_STEP = 20  # the step used in evaluation
    SAMPLE_ACTION_NUMBER = 120
    # target_DQN = FC_Model()
    # target_DQN.load_weights('RL_weight/RL_SD2RL_3_512feature_model_weight_1')
    # print(target_DQN.weights)

    RL = Agent(reward_decay=0.995,
               memory_size=200000,
               )
    start_time = time.time()
    evaluate_test_set()
    end_time = time.time()
    print(f"time:{end_time - start_time}")