import itertools
import time
import random
import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os
import heapq
import math
import copy

tf.get_logger().setLevel('ERROR')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.config.experimental.set_virtual_device_configuration(
    tf.config.experimental.list_physical_devices('GPU')[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)]
)

LEARNING_RATE = 1e-7
REWARD_CATE_RATE = .8
REWARD_PAIRWISE_RATE = .2
EPISODE_LOSS_RATE = 1.
ACHIEVE_REWARD = 1000.  # the reward after achieving the whole puzzle
# STOP_STEP = 20            # the step used in evaluation
global STOP_STEP
STOP_STEP = 50  # the step used in training
# EPSILON_MAX = 0
TEST_STEP = 20

EPSILON_MAX = .9
EPSILON_MIN = .1
GAMMA = 0.995
# EPSILON = 0.95
BATCH_SIZE = 5

n_features = 8
n_actions = 1
SAMPLE_ACTION_NUMBER = 10

TRAIN_NUMBER = 1000
TEST_NUMBER = 2000
IS_TEST_SHUFFLE = False

total_reward_list = []

train_x_path = 'MET_Dataset/select_image/train_img_no_gap.npy'
train_y_path = 'MET_Dataset/select_image/train_label_no_gap.npy'

valid_x_path = 'MET_Dataset/select_image/valid_img_no_gap.npy'
valid_y_path = 'MET_Dataset/select_image/valid_label_no_gap.npy'

test_x_path = 'MET_Dataset/select_image/test_img_no_gap.npy'
test_y_path = 'MET_Dataset/select_image/test_label_no_gap.npy'

test_y = np.load(test_y_path)
# np.random.seed(2)
# np.random.shuffle(test_y)
# test_y = test_y[:667]
train_y = np.load(valid_y_path)
# np.random.seed(1)
# np.random.shuffle(train_y)
# train_y = train_y[:500]

test_x = np.load(test_x_path)
# np.random.seed(2)
# np.random.shuffle(test_x)
# test_x = test_x[:667]
train_x = np.load(valid_x_path)
# np.random.seed(1)
# np.random.shuffle(train_x)
# train_y = train_y[:500]

# test_cate_list = np.load('test_cate_score.npy')
test_cate_list = np.load('test_cate_score_eff_9_ft.npy')
# np.random.seed(2)
# np.random.shuffle(test_cate_list)
# test_cate_list = test_cate_list[:667]
# cate_list = np.load('valid_cate_score.npy')
cate_list = np.load('valid_cate_score_eff_9_ft.npy')
# np.random.seed(1)
# np.random.shuffle(cate_list)
# cate_list = cate_list[:500]

# test_hori_list = np.load('test_hori_score.npy')
test_hori_list = np.load('test_hori_score_eff_9_ft.npy')
# np.random.seed(2)
# np.random.shuffle(test_hori_list)
# test_hori_list = test_hori_list[:667]
# hori_list = np.load('valid_hori_score.npy')
hori_list = np.load('valid_hori_score_eff_9_ft.npy')
# np.random.seed(1)
# np.random.shuffle(hori_list)
# hori_list = hori_list[:500]

# test_vert_list = np.load('test_vert_score.npy')
test_vert_list = np.load('test_vert_score_eff_9_ft.npy')
# np.random.seed(2)
# np.random.shuffle(test_vert_list)
# test_vert_list = test_vert_list[:667]
# vert_list = np.load('valid_vert_score.npy')
vert_list = np.load('valid_vert_score_eff_9_ft.npy')
# np.random.seed(1)
# np.random.shuffle(vert_list)
# vert_list = vert_list[:500]


FEN_hori = tf.keras.models.load_model('hori_3_EfficientNetB0_9class_ft.h5')
x = FEN_hori.layers[-2].output
# x = tf.keras.layers.Dense(512, activation='relu', name='hori_fc_1')(x)
# x = tf.keras.layers.Dense(128, activation='relu', name='hori_fc_2')(x)
# x = tf.keras.layers.Dense(64, activation='relu', name='hori_fc_3')(x)
FEN_hori = tf.keras.Model(inputs=FEN_hori.input, outputs=x)

FEN_vert = tf.keras.models.load_model('vert_3_EfficientNetB0_9class_ft.h5')
x = FEN_vert.layers[-2].output
# x = tf.keras.layers.Dense(512, activation='relu', name='vert_fc_1')(x)
# x = tf.keras.layers.Dense(128, activation='relu', name='vert_fc_2')(x)
# x = tf.keras.layers.Dense(64, activation='relu', name='vert_fc_3')(x)
FEN_vert = tf.keras.Model(inputs=FEN_vert.input, outputs=x)


def get_reward(pt_permutation, data_hori_i, data_vert_i, data_cate_i):
    cur_permutation = copy.deepcopy(pt_permutation)
    for cur_permutation_index in range(len(cur_permutation)):
        if cur_permutation[cur_permutation_index] >= 4:
            cur_permutation[cur_permutation_index] += 1
    cur_permutation.insert(4, 4)

    data_cate_i = np.insert(data_cate_i, 4, [1] * 8, 0)
    data_cate_i = np.insert(data_cate_i, 4, [1] * 9, 1).tolist()

    reward = 0.   # initial reward
    for c_index in range(len(cur_permutation)):
        if c_index % 3 != 0:
            reward += data_hori_i[cur_permutation[c_index - 1]][cur_permutation[c_index]] * \
                      data_cate_i[cur_permutation[c_index - 1]][c_index - 1]
        if c_index % 3 != 2:
            reward += data_hori_i[cur_permutation[c_index]][cur_permutation[c_index + 1]] * \
                      data_cate_i[cur_permutation[c_index + 1]][c_index + 1]
        if c_index // 3 != 0:
            reward += data_vert_i[cur_permutation[c_index - 3]][cur_permutation[c_index]] * \
                      data_cate_i[cur_permutation[c_index - 3]][c_index - 3]
        if c_index // 3 != 2:
            reward += data_vert_i[cur_permutation[c_index]][cur_permutation[c_index + 3]] * \
                      data_cate_i[cur_permutation[c_index + 3]][c_index + 3]
    return reward


def get_init_evi_greedy(data_hori_i, data_vert_i, data_cate_i, reassembble_list):
    cur_permutation = copy.deepcopy(reassembble_list)

    while True:
        cur_reward = get_reward(cur_permutation, data_hori_i, data_vert_i, data_cate_i)

        best_reward = cur_reward
        best_permutation = copy.deepcopy(cur_permutation)
        # best_cate = cur_data_cate_i.copy()
        for j in range(7):
            for k in range(j + 1, 8):
                p_permutation = copy.deepcopy(cur_permutation)
                p_permutation[j] = cur_permutation[k]
                p_permutation[k] = cur_permutation[j]
                p_reward = get_reward(p_permutation, data_hori_i, data_vert_i, data_cate_i)
                if p_reward > best_reward:
                    best_reward = p_reward
                    best_permutation = copy.deepcopy(p_permutation)
                    # best_cate = p_data_cate_i.copy()
        if best_reward <= cur_reward:
            break
        else:
            cur_permutation = copy.deepcopy(best_permutation)
    return cur_permutation


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


def get_best_actions_list(data_hori_i, data_vert_i, data_cate_i, reassemble_list):
    action_tuple_list = list(itertools.combinations(list(range(8)), 2))
    score_list = []
    for action_tuple in action_tuple_list:
        index_list = copy.deepcopy(reassemble_list)
        index_list[action_tuple[0]] = reassemble_list[action_tuple[1]]
        index_list[action_tuple[1]] = reassemble_list[action_tuple[0]]
        score_list.append(get_reward(index_list, data_hori_i, data_vert_i, data_cate_i))

    action_list = []
    for i in range(SAMPLE_ACTION_NUMBER):
        max_index = score_list.index(max(score_list))
        action_list.append(max_index)
        score_list[max_index] = -math.inf
    return action_list


# def get_prior_feature(data_hori_i, data_vert_i, data_cate_i, reassemble_list):
def get_cur_image(data_img_i, reassemble_list):
    fragment_list = []
    for i in range(9):
        fragment_list.append(data_img_i[96 * (i // 3): 96 * (i // 3) + 96, 96 * (i % 3): 96 * (i % 3) + 96, :])
    index_list = [0] * 9
    # print(reassemble_list)
    for i in range(len(reassemble_list)):
        if i >= 4:
            index = i + 1
        else:
            index = i
        if reassemble_list[i] >= 4:
            index_list[index] = reassemble_list[i] + 1
        else:
            index_list[index] = reassemble_list[i]
    index_list[4] = 4

    img_0 = np.concatenate((fragment_list[index_list[0]],
                            fragment_list[index_list[1]],
                            fragment_list[index_list[2]]), axis=1)
    img_1 = np.concatenate((fragment_list[index_list[3]],
                            fragment_list[index_list[4]],
                            fragment_list[index_list[5]]), axis=1)
    img_2 = np.concatenate((fragment_list[index_list[6]],
                            fragment_list[index_list[7]],
                            fragment_list[index_list[8]]), axis=1)
    whole_img = np.concatenate((img_0, img_1, img_2), axis=0)

    return whole_img


def get_reassemble_result(cur_loc, target_loc):
    cate, hori, vert = 0, 0, 0
    for i in range(len(cur_loc)):
        if cur_loc[i] == target_loc[i]:
            cate += 1
    loc_hori_pairwise = [(cur_loc[0], cur_loc[1]), (cur_loc[1], cur_loc[2]),
                         (cur_loc[3], 9), (9, cur_loc[4]),
                         (cur_loc[5], cur_loc[6]), (cur_loc[6], cur_loc[7])]
    loc_vert_pairwise = [(cur_loc[0], cur_loc[3]), (cur_loc[1], 9), (cur_loc[2], cur_loc[4]),
                         (cur_loc[3], cur_loc[5]), (9, cur_loc[6]), (cur_loc[4], cur_loc[7])]
    y_hori_pairwise = [(target_loc[0], target_loc[1]), (target_loc[1], target_loc[2]),
                       (target_loc[3], 9), (9, target_loc[4]),
                       (target_loc[5], target_loc[6]), (target_loc[6], target_loc[7])]
    y_vert_pairwise = [(target_loc[0], target_loc[3]), (target_loc[1], 9), (target_loc[2], target_loc[4]),
                       (target_loc[3], target_loc[5]), (9, target_loc[6]), (target_loc[4], target_loc[7])]

    hori = len(set(loc_hori_pairwise) & set(y_hori_pairwise))
    vert = len(set(loc_vert_pairwise) & set(y_vert_pairwise))
    return cate, hori, vert


class Puzzle_Env(object):
    def __init__(self):
        self.index_list = list(range(8))
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

    # def reset(self, fragment_list, y_true=[]):
    # def reset(self, episode_hori, episode_vert, episode_cate, y_true=[]):
    def reset(self, hori_score, vert_score, cate_score, episode_img, y_true=[]):
        # self.initial_fragment_list = fragment_list.copy()
        # self.fragment_list = fragment_list.copy()
        self.y_true = y_true    # store the ground truth information
        self.initial_hori_score_list = copy.deepcopy(hori_score)
        self.initial_vert_score_list = copy.deepcopy(vert_score)
        self.initial_cate_score_list = copy.deepcopy(cate_score)
        self.initial_img = copy.deepcopy(episode_img)

        self.index_list = list(range(8))
        # random.seed(1)
        # random.shuffle(self.index_list)
        # if np.random.uniform() < 0.4:
        # self.index_list = get_init_evi_greedy(self.initial_hori_score_list, self.initial_vert_score_list,
        #                                       self.initial_cate_score_list, self.index_list)
        return self.index_list

    def step(self, a, training=True):
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
        if a < 28:
            action_tuple = list(itertools.combinations(list(range(8)), 2))[a]
            value_0 = self.index_list[action_tuple[0]]
            value_1 = self.index_list[action_tuple[1]]
            cur_index_list[action_tuple[0]] = value_1
            cur_index_list[action_tuple[1]] = value_0
            self.index_list = copy.deepcopy(cur_index_list)

        loc = [8] * 8
        for i in range(len(cur_index_list)):
            loc[cur_index_list[i]] = i
        if training:
            if loc == self.y_true:
                reward = ACHIEVE_REWARD
                return cur_index_list, reward, True, 8, 6, 6
            else:
                cate, hori, vert = get_reassemble_result(loc, self.y_true)
                # reward = cate / 8 * REWARD_CATE_RATE + (hori + vert) / 12 * REWARD_PAIRWISE_RATE - EPISODE_LOSS_RATE
                reward = cate * REWARD_CATE_RATE + (hori + vert) * REWARD_PAIRWISE_RATE - EPISODE_LOSS_RATE
                return cur_index_list, reward, False, cate, hori, vert
        else:
            cate, hori, vert = get_reassemble_result(loc, self.y_true)
            return cur_index_list, 0, False, cate, hori, vert

    def index_list_2_fragment_list(self):
        fragment_list = self.initial_fragment_list.copy()
        for i in range(len(self.index_list)):
            fragment_list[i] = self.initial_fragment_list[self.index_list[i]]
        return fragment_list


# class FC_Model(tf.keras.Model):
#     def __init__(self, action_num=1):
#         super(FC_Model, self).__init__()
#         # tf.keras.backend.clear_session()
#         self.inputs = tf.keras.Input(shape=(288, 288, 3))
#         self.fc_1 = tf.keras.layers.Dense(16,
#                                           activation='relu',
#                                           # kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.8,
#                                           #                                                        maxval=1.2),
#                                           # kernel_constraint=tf.keras.constraints.min_max_norm(min_value=0.8,
#                                           #                                                     max_value=1.2),
#                                           name='fc_1')
#         # self.bn_1 = tf.keras.layers.BatchNormalization(name='bn_1')
#         # self.relu_1 = tf.keras.layers.ReLU(name='relu_1')

#         self.fc_2 = tf.keras.layers.Dense(8,
#                                           # activation='relu',
#                                           # kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.99,
#                                           #                                                        maxval=1.01),
#                                           # kernel_constraint=tf.keras.constraints.min_max_norm(min_value=0.8,
#                                           #                                                     max_value=1.2),
#                                           name='fc_2')
#         # self.bn_2 = tf.keras.layers.BatchNormalization(name='bn_2')
#         # self.relu_2 = tf.keras.layers.ReLU(name='relu_2')

#         self.fc_out = tf.keras.layers.Dense(action_num,
#                                             # activation='softmax',
#                                             # kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.99,
#                                             #                                                        maxval=1.01),
#                                             # kernel_constraint=tf.keras.constraints.min_max_norm(min_value=0.8,
#                                             #                                                     max_value=1.2),
#                                             name='fc_out')
#         self.bn_out = tf.keras.layers.BatchNormalization(name='bn_out')
#         self.relu_out = tf.keras.layers.ReLU(name='relu_out')
#         # self.wc1 = tf.Variable(tf.random.normal([0.9, 1.1], dtype='float32'))

#     def call(self, inputs, training=None):
#         # crop the input image into 9 fragments
#         fragment_0 = tf.keras.layers.Cropping2D(cropping=((0, 192), (0, 192)))(inputs)
#         fragment_1 = tf.keras.layers.Cropping2D(cropping=((0, 192), (96, 96)))(inputs)
#         fragment_2 = tf.keras.layers.Cropping2D(cropping=((0, 192), (192, 0)))(inputs)

#         fragment_3 = tf.keras.layers.Cropping2D(cropping=((96, 96), (0, 192)))(inputs)
#         fragment_4 = tf.keras.layers.Cropping2D(cropping=((96, 96), (96, 96)))(inputs)
#         fragment_5 = tf.keras.layers.Cropping2D(cropping=((96, 96), (192, 0)))(inputs)

#         fragment_6 = tf.keras.layers.Cropping2D(cropping=((192, 0), (0, 192)))(inputs)
#         fragment_7 = tf.keras.layers.Cropping2D(cropping=((192, 0), (96, 96)))(inputs)
#         fragment_8 = tf.keras.layers.Cropping2D(cropping=((192, 0), (192, 0)))(inputs)

#         # to extract the horizontal

#         hori_feature_0_1 = FEN_hori([fragment_0, fragment_1])
#         hori_feature_1_2 = FEN_hori([fragment_1, fragment_2])
#         hori_feature_3_4 = FEN_hori([fragment_3, fragment_4])
#         hori_feature_4_5 = FEN_hori([fragment_4, fragment_5])
#         hori_feature_6_7 = FEN_hori([fragment_6, fragment_7])
#         hori_feature_7_8 = FEN_hori([fragment_7, fragment_8])
#         hori_feature = tf.keras.layers.Concatenate()([hori_feature_0_1, hori_feature_1_2,
#                                                       hori_feature_3_4, hori_feature_4_5,
#                                                       hori_feature_6_7, hori_feature_7_8])

#         vert_feature_0_3 = FEN_vert([fragment_0, fragment_3])
#         vert_feature_1_4 = FEN_vert([fragment_1, fragment_4])
#         vert_feature_2_5 = FEN_vert([fragment_2, fragment_5])
#         vert_feature_3_6 = FEN_vert([fragment_3, fragment_6])
#         vert_feature_4_7 = FEN_vert([fragment_4, fragment_7])
#         vert_feature_5_8 = FEN_vert([fragment_5, fragment_8])
#         vert_feature = tf.keras.layers.Concatenate()([vert_feature_0_3, vert_feature_1_4,
#                                                       vert_feature_2_5, vert_feature_3_6,
#                                                       vert_feature_4_7, vert_feature_5_8])

#         global_feature = GP_model(self.inputs)
#         global_feature = tf.keras.layers.Flatten()(global_feature)

#         x = tf.keras.layers.Concatenate()([hori_feature, vert_feature, global_feature])

#         # for i in range(25):
#         #     out = tf.keras.layers.Lambda(lambda x: x[i])(inputs)
#         # x = self.fc_1(global_feature)
#         x = self.fc_1(x)
#         # x = self.bn_1(x)
#         # x = self.relu_1(x)
#         x = self.fc_2(x)
#         # x = self.bn_2(x)
#         # x = self.relu_2(x)

#         x = self.fc_out(x)
#         # x = self.bn_out(x)
#         # x = self.relu_out(x)
#         # x = tf.matmul(x, self.wc1, name='out')
#         return x


def dqn_model():
    inputs = tf.keras.Input(shape=[288, 288, 3], name='img_input')

    fragment_0 = tf.keras.layers.Cropping2D(cropping=((0, 192), (0, 192)))(inputs)
    fragment_1 = tf.keras.layers.Cropping2D(cropping=((0, 192), (96, 96)))(inputs)
    fragment_2 = tf.keras.layers.Cropping2D(cropping=((0, 192), (192, 0)))(inputs)

    fragment_3 = tf.keras.layers.Cropping2D(cropping=((96, 96), (0, 192)))(inputs)
    fragment_4 = tf.keras.layers.Cropping2D(cropping=((96, 96), (96, 96)))(inputs)
    fragment_5 = tf.keras.layers.Cropping2D(cropping=((96, 96), (192, 0)))(inputs)

    fragment_6 = tf.keras.layers.Cropping2D(cropping=((192, 0), (0, 192)))(inputs)
    fragment_7 = tf.keras.layers.Cropping2D(cropping=((192, 0), (96, 96)))(inputs)
    fragment_8 = tf.keras.layers.Cropping2D(cropping=((192, 0), (192, 0)))(inputs)
    #getting pieces



    # to extract the horizontal
    hori_feature_0_1 = FEN_hori([fragment_0, fragment_1])
    hori_feature_1_2 = FEN_hori([fragment_1, fragment_2])
    hori_feature_3_4 = FEN_hori([fragment_3, fragment_4])
    hori_feature_4_5 = FEN_hori([fragment_4, fragment_5])
    hori_feature_6_7 = FEN_hori([fragment_6, fragment_7])
    hori_feature_7_8 = FEN_hori([fragment_7, fragment_8])
    hori_feature = tf.keras.layers.Concatenate()([hori_feature_0_1, hori_feature_1_2,
                                                  hori_feature_3_4, hori_feature_4_5,
                                                  hori_feature_6_7, hori_feature_7_8])

    vert_feature_0_3 = FEN_vert([fragment_0, fragment_3])
    vert_feature_1_4 = FEN_vert([fragment_1, fragment_4])
    vert_feature_2_5 = FEN_vert([fragment_2, fragment_5])
    vert_feature_3_6 = FEN_vert([fragment_3, fragment_6])
    vert_feature_4_7 = FEN_vert([fragment_4, fragment_7])
    vert_feature_5_8 = FEN_vert([fragment_5, fragment_8])
    vert_feature = tf.keras.layers.Concatenate()([vert_feature_0_3, vert_feature_1_4,
                                                  vert_feature_2_5, vert_feature_3_6,
                                                  vert_feature_4_7, vert_feature_5_8])



    concate_layer = tf.keras.layers.Concatenate()([hori_feature, vert_feature])
    dropout_layer = tf.keras.layers.Dropout(.1)(concate_layer)
    fc_1 = tf.keras.layers.Dense(128,
                                 # kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.8,
                                 #                                                        maxval=1.2),
                                 # kernel_constraint=tf.keras.constraints.min_max_norm(min_value=0.8,
                                 #                                                     max_value=1.2),
                                 name='fc_1')(dropout_layer)
    bn_1 = tf.keras.layers.BatchNormalization()(fc_1)
    relu_1 = tf.keras.layers.ReLU()(bn_1)

    fc_2 = tf.keras.layers.Dense(64,
                                 activation='relu',
                                 # kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.8,
                                 #                                                        maxval=1.2),
                                 # kernel_constraint=tf.keras.constraints.min_max_norm(min_value=0.8,
                                 #                                                     max_value=1.2),
                                 name='fc_2')(relu_1)

    out = tf.keras.layers.Dense(1,
                                # kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.8,
                                #                                                        maxval=1.2),
                                # kernel_constraint=tf.keras.constraints.min_max_norm(min_value=0.8,
                                #                                                     max_value=1.2),
                                name='fc_out')(fc_2)
    dqn_model = tf.keras.models.Model(inputs, out)
    return dqn_model


main_DQN = dqn_model()
main_DQN.summary()
target_DQN = dqn_model()
target_DQN.summary()
target_DQN.set_weights(main_DQN.get_weights())


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
                 tau=0.0005
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

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self.cost = []

        # self.mainQN = NetWork()
        # self.targetQN = NetWork()

        # self.mainQN.build(input_shape=(None, self.n_features))
        # self.mainQN.summary()
        #
        # self.targetQN.build(input_shape=(None, self.n_features))
        # self.targetQN.summary()

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):  # hasattr:返回对象是否具有给定名称的属性。
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))  # hatack:水平（按列）顺序堆叠数组。
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def empty_memory(self):
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.memory_counter = 0

    # input obs, return estimate Q from network
    def evaluate(self, image, obs):
        # t = time.time()
        eval_img_list = []
        for ob in obs:
            eval_img_list.append(get_cur_image(image, ob))
        result = target_DQN.predict(tf.convert_to_tensor(eval_img_list))
        # print(time.time()-t)
        return result

    # input action, return updated observation
    # def get_obs(self, cur_hori_list, cur_vert_list, cur_cate_list, p_index_list, a):
    def get_obs(self, p_index_list, a):
        index_list = copy.deepcopy(p_index_list)
        action_tuple = list(itertools.combinations(list(range(8)), 2))[a]
        value_0 = index_list[action_tuple[0]]
        value_1 = index_list[action_tuple[1]]
        index_list[action_tuple[0]] = value_1
        index_list[action_tuple[1]] = value_0
        return index_list
        # cur_features = get_cur_image(cur_img_list, index_list)
        # return cur_features

    def choose_action(self, observation, episode_hori, episode_vert, episode_cate, episode_img):
        '''
        :param observation: the sequence of the fragments
        :param episode_hori:
        :param episode_vert:
        :param episode_cate:
        :param episode_img:
        :param index_list:
        :return: the action index
        '''
        best_action_list = get_best_actions_list(episode_hori, episode_vert, episode_cate, observation)
        if np.random.uniform() > self.epsilon:  # np.random.uniform():均匀分布中抽取样本
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
            actions_value = self.evaluate(episode_img, obs_list)
            if np.argmax(actions_value) <= SAMPLE_ACTION_NUMBER:
                action = best_action_list[np.argmax(actions_value)]
        else:
            action_index = np.random.randint(SAMPLE_ACTION_NUMBER)  # np.random.randint：在区间返回随机的整数
            action = best_action_list[action_index]
        return action

    def train(self, episode_image):
        # replace target paramsx
        # if self.learn_step_counter % self.replace_target_iter == 0:
        #     weights = main_DQN.get_weights()
        #     target_DQN.set_weights(main_DQN.get_weights())
        #     print("\ntargetQN_params_replaced\n")
        mainQN_weights = main_DQN.get_weights()
        targetQN_weights = target_DQN.get_weights()
        op_holder = []
        for idx, var in enumerate(mainQN_weights):
            op_holder.append((var * self.tau) + ((1 - self.tau) * targetQN_weights[idx]))
        target_DQN.set_weights(op_holder)

        # select batchsz sample
        if self.memory_counter > self.memory_size:
            # td_error_list = list(self.memory[:, -1]) / sum(self.memory[:, -1])
            # sample_index = np.random.choice(a=np.arange(self.memory_size), size=self.batch_size, p=td_error_list)
            sample_index = np.random.choice(a=np.arange(self.memory_size), size=self.batch_size)
        else:
            # td_error_list = list(self.memory[:self.memory_counter, -1]) / sum(self.memory[:self.memory_counter, -1])
            # sample_index = np.random.choice(a=np.arange(self.memory_counter), size=self.batch_size, p=td_error_list)
            sample_index = np.random.choice(a=np.arange(self.memory_counter), size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        # q_next = targetQN.call(batch_memory[:, -self.n_features:], training=False)
        # q_eval = mainQN.call(batch_memory[:, :self.n_features], training=True)

        # q_eval = q_eval.numpy()
        # print("q_eval:", q_eval)
        # print("q_eval.shape:", q_eval.shape)
        # print("q_eval.dtype:", q_eval.dtype)
        # q_target = q_eval.numpy().copy()
        # print("q_target:", q_target)

        # batch_index = np.arange(self.batch_size, dtype=np.int32)
        # eval_act_index = batch_memory[:, self.n_features].astype(int)
        # reward = batch_memory[:, self.n_features + 1]
        # print("reward:", reward)  # [32]
        # print("q_target:", q_target)  # [32, 4]
        # print("np.max(q_target):", np.max(q_target, axis=1))  # [32]
        # print("q_eval:", q_eval)  # [32, 4]

        # q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        # print(q_target[batch_index, eval_act_index])
        # q_target = tf.convert_to_tensor(q_target)
        # q_eval = tf.convert_to_tensor(q_eval)

        # loss
        with tf.GradientTape() as tape:
            # print('X')
            target_image_list = []
            for target_image_index in batch_memory[:, -self.n_features:]:
                target_image_list.append(get_cur_image(episode_image, [int(x) for x in target_image_index]))
            q_next = target_DQN.call(tf.convert_to_tensor(target_image_list), training=False)
            q_eval = main_DQN.call(tf.convert_to_tensor(target_image_list), training=True)

            # eval_image_list = []
            # for eval_image_index in batch_memory[:, :self.n_features]:
            #     eval_image_list.append(get_cur_image(episode_image, [int(x) for x in eval_image_index]))
            # q_eval = main_DQN.call(tf.convert_to_tensor(eval_image_list), training=True)

            q_target = q_eval.numpy().copy()
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            eval_act_index = batch_memory[:, self.n_features].astype(int)

            reward = batch_memory[:, self.n_features + 1]
            q_target[batch_index, 0] = reward + self.gamma * np.max(q_next, axis=1)
            loss = tf.reduce_mean(tf.square(q_target - q_eval))
            # loss = tf.keras.losses.Huber()(q_target, q_eval)
            # print(q_target)
            # print(q_eval)
            # print(loss)
            # cur_action = np.array(batch_memory[:, self.n_features], dtype=int)
            # q_eval = np.array(q_eval)
            # self.memory[sample_index, -1] = abs(q_target[:, cur_action] - q_eval[:, cur_action])
            # for index in range(len(sample_index)):
            #     x1 = q_target[index, 0]
            #     x2 = q_eval[index, 0]
            #     self.memory[sample_index[index], -1] = abs(x1 - x2)

        grads = tape.gradient(loss, main_DQN.trainable_variables)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE, rho=0.9)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        optimizer.apply_gradients(zip(grads, main_DQN.trainable_variables))
        self.cost.append(loss)

        # increasing epsilon
        # self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        self.learn_step_counter += 1

        if self.learn_step_counter % 100 == 0:
            print("step:", self.learn_step_counter, "loss:", float(loss))
            # print(main_DQN.get_weights())
            # with summary_writer.as_default():
            #     tf.summary.scalar("loss", float(loss), step=len(self.cost))


def run_maze():
    step = 0
    np.random.seed(1)
    train_id_list = np.arange(len(train_y))
    np.random.shuffle(train_id_list)
    for i in range(TRAIN_NUMBER):
        RL.empty_memory()
        if i > 300:
            global SAMPLE_ACTION_NUMBER
            SAMPLE_ACTION_NUMBER = 28
            STOP_STEP = 50
            global LEARNING_RATE
            LEARNING_RATE = 1e-5
        elif i > 100:
            SAMPLE_ACTION_NUMBER = 14
            STOP_STEP = 100
            LEARNING_RATE = 1e-4
        else:
            SAMPLE_ACTION_NUMBER = 10
            STOP_STEP = 400
            LEARNING_RATE = 1e-4
        episode_i = train_id_list[i % len(train_y)]
        y_true = onehot_2_index(train_y[episode_i])
        init_index_list = env.reset(hori_list[episode_i], vert_list[episode_i], cate_list[episode_i],
                                    train_x[episode_i], y_true=y_true)

        observation = copy.deepcopy(init_index_list)
        total_reward = 0.
        p_step = 0

        while True:
            # RL choose action based on observation
            action = RL.choose_action(observation, hori_list[episode_i], vert_list[episode_i],
                                      cate_list[episode_i], train_x[episode_i])
            # RL take action and get next observation and reward
            observation_, reward, done, cate, _, _ = env.step(action)
            total_reward += reward
            RL.store_transition(observation, action, reward, observation_)

            # if (step > 200) and (step % 5 == 0):
            # if (step <= 10) or (step % 5 == 0) or done:
            if (step % 5 == 4):
                RL.train(train_x[episode_i])

            # swap observation
            observation = copy.deepcopy(observation_)

            # break while loop when end of this episode
            if done:
                RL.train(train_x[episode_i])
                print(str(i), "\tsuccess\tstep length: ", str(p_step))
                # target_DQN.set_weights(main_DQN.get_weights())
                break
            elif p_step >= STOP_STEP - 1:
                print(str(i), "\tunsuccess\tstep length: ", str(p_step))
                break
            step += 1
            p_step += 1
        total_reward_list.append(total_reward / (p_step + 1))
        print("Episode Id: ", str(episode_i), "\tTotal Reward: ", total_reward / (p_step + 1))

        # update epsilon
        RL.epsilon = RL.epsilon * RL.epsilon_decay if RL.epsilon > RL.epsilon_min else RL.epsilon_min
        # print(RL.epsilon)

        # show_plot()

    # end of game
    print('game over')


def evaluate_test_set():
    success = 0
    whole_cate = 0
    whole_hori = 0
    whole_vert = 0
    sequence_list = []
    test_id_list = np.arange(len(test_y))
    if IS_TEST_SHUFFLE:
        np.random.seed(2)
        np.random.shuffle(test_id_list)
    for i in range(TEST_NUMBER):
        episode = test_id_list[i]
        y_true = onehot_2_index(test_y[episode])
        index_list = env.reset(test_hori_list[episode], test_vert_list[episode],
                               test_cate_list[episode], test_x[episode], y_true=y_true)
        observation = copy.deepcopy(index_list)
        # observation, _, _, cate, hori, vert = env.step(28)
        obs_img = get_cur_image(test_x[episode], observation)
        best_reward = target_DQN.predict(np.expand_dims(obs_img, axis=0))[0, 0]
        # best_reward = 0.
        step = 0
        # step, cate, hori, vert = 0, 0, 0, 0
        done = False
        while True:
            if step >= TEST_STEP:
                observation_, _, done_, cate_, hori_, vert_ = env.step(28)
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
                action = RL.choose_action(observation, test_hori_list[episode], test_vert_list[episode],
                                          test_cate_list[episode], test_x[episode])

                # RL take action and get next observation and reward
                _, _, done, cate, hori, vert = env.step(28)
                observation_, _, done_, cate_, hori_, vert_ = env.step(action)
                obs_img_ = get_cur_image(test_x[episode], observation_)
                rl_reward = target_DQN.predict(np.expand_dims(obs_img_, axis=0))[0, 0]
                # rl_reward = sum(observation_)

                if rl_reward > best_reward:
                    best_reward = rl_reward
                    observation = copy.deepcopy(observation_)
                    done = done_
                # break while loop when end of this episode
                else:
                    if done:
                        success += 1
                        print(str(i), "\tsuccess\tstep length: ", str(step), "\tsuccess rate\t",
                              str(success / (i + 1)))
                    else:
                        print(str(i), "\tunsuccess\tstep length: ", str(step))
                    # _, _, _, cate, hori, vert = env.step(28)
                    whole_cate += cate
                    whole_hori += hori
                    whole_vert += vert
                    break
            step += 1

    print('Success Number', success)
    print('Success Cate', whole_cate)
    print('Success Hori', whole_hori)
    print('Success Vert', whole_vert)
    # np.save('resnet2f_05_sequence_1.npy', sequence_list)


if __name__ == "__main__":
    # maze game
    env = Puzzle_Env()
    # SAMPLE_ACTION_NUMBER = 28
    RL = Agent(reward_decay=0.995,
               replace_target_iter=2,
               memory_size=200000,
               epsilon=EPSILON_MAX,
               epsilon_min=EPSILON_MIN,
               epsilon_decay=GAMMA
               )  # 主循环，必须为最后一条语句
    run_maze()
    target_DQN.save_weights('RL_weight/RL_SD2RL_3_512feature_model_weight_1')

    TEST_STEP = 10  # the step used in evaluation
    SAMPLE_ACTION_NUMBER = 10
    # target_DQN = FC_Model()
    target_DQN.load_weights('RL_weight/RL_SD2RL_3_512feature_model_weight_1')
    # print(target_DQN.weights)
    RL = Agent(reward_decay=0.995,
               memory_size=200000,
               )
    # start_time = time.time()
    evaluate_test_set()
    # end_time = time.time()
    # print(end_time - start_time)
