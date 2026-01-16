import numpy as np
import copy
import gc
import os

class Evaluator:
    def __init__(self, data_y, puzzle_size, data_cate=None, data_hori=None, data_vert=None, data_puzzleLet=None):
        self.puzzle_size = puzzle_size
        self.data_cate = data_cate
        self.data_hori = data_hori
        self.data_vert = data_vert
        self.data_y = data_y
        self.data_puzzleLet = data_puzzleLet


    def set_index(self, index):
        # self.data_cate_i = self.data_cate[index]
        # self.data_y_i = self.data_y[index]
        self.index = index

    def evaluate(self, s, eval_method='pairwise_info'):
        if self.puzzle_size == 3:
            if eval_method == 'pairwise_info':
                return self.evaluate_33(s)
            elif eval_method == 'puzzleLet_info':
                return self.evaluate_puzzleLet_33(s)
        elif self.puzzle_size == 5:
            if eval_method == 'pairwise_info':
                return self.evaluate_55(s)
            elif eval_method == 'puzzleLet_info':
                return self.evaluate_puzzleLet_55(s)
        elif self.puzzle_size == 70:
            if eval_method == 'pairwise_info':
                return self.evaluate_70(s)
            elif eval_method == 'puzzleLet_info':
                return self.evaluate_puzzleLet_70(s)
        elif self.puzzle_size == 150:
            if eval_method == 'pairwise_info':
                return self.evaluate_150(s)
            elif eval_method == 'puzzleLet_info':
                return self.evaluate_puzzleLet_70(s)
        else:
            print('wrong size')
            exit(-1)

    def evaluate_33(self, s):
        cur_permutation = copy.deepcopy(s)
        for cur_reassemble_i in range(len(s)):
            if cur_permutation[cur_reassemble_i] >= 4:
                cur_permutation[cur_reassemble_i] += 1
        data_cate_i = self.data_cate[self.index]
        data_hori_i = self.data_hori[self.index]
        data_vert_i = self.data_vert[self.index]
        data_y_i = self.data_y[self.index]

        # score = data_cate_i[cur_permutation[0], 0] + data_cate_i[cur_permutation[1], 1] + \
        #         data_cate_i[cur_permutation[2], 2] + data_cate_i[cur_permutation[3], 3] + \
        #         data_cate_i[cur_permutation[4], 4] + data_cate_i[cur_permutation[5], 5] + \
        #         data_cate_i[cur_permutation[6], 6] + data_cate_i[cur_permutation[7], 7]

        reward_0 = data_hori_i[cur_permutation[0]][cur_permutation[1]] * data_cate_i[s[1]][1] + \
                   data_vert_i[cur_permutation[0]][cur_permutation[3]] * data_cate_i[s[3]][3]
        reward_1 = data_hori_i[cur_permutation[0]][cur_permutation[1]] * data_cate_i[s[0]][0] + \
                   data_vert_i[cur_permutation[1]][4] * 1 + \
                   data_hori_i[cur_permutation[1]][cur_permutation[2]] * data_cate_i[s[2]][2]
        reward_2 = data_hori_i[cur_permutation[1]][cur_permutation[2]] * data_cate_i[s[1]][1] + \
                   data_vert_i[cur_permutation[2]][cur_permutation[5]] * data_cate_i[s[4]][4]
        reward_3 = data_vert_i[cur_permutation[0]][cur_permutation[3]] * data_cate_i[s[0]][0] + \
                   data_hori_i[cur_permutation[3]][4] * 1 + \
                   data_vert_i[cur_permutation[3]][cur_permutation[6]] * data_cate_i[s[5]][5]
        reward_5 = data_vert_i[cur_permutation[2]][cur_permutation[4]] * data_cate_i[s[2]][2] + \
                   data_hori_i[4][cur_permutation[4]] * 1 + \
                   data_vert_i[cur_permutation[4]][cur_permutation[7]] * data_cate_i[s[7]][7]
        reward_6 = data_hori_i[cur_permutation[5]][cur_permutation[6]] * data_cate_i[s[6]][6] + \
                   data_vert_i[cur_permutation[3]][cur_permutation[5]] * data_cate_i[s[3]][3]
        reward_7 = data_hori_i[cur_permutation[5]][cur_permutation[6]] * data_cate_i[s[5]][5] + \
                   data_vert_i[4][cur_permutation[6]] * 1 + \
                   data_hori_i[cur_permutation[6]][cur_permutation[7]] * data_cate_i[s[7]][7]
        reward_8 = data_hori_i[cur_permutation[6]][cur_permutation[7]] * data_cate_i[s[6]][6] + \
                   data_vert_i[cur_permutation[4]][cur_permutation[7]] * data_cate_i[s[4]][4]
        score = reward_0 + reward_1 + reward_2 + reward_3 + reward_5 + reward_6 + reward_7 + reward_8

        true_matrix = []
        for data_y_i_i in data_y_i:
            true_matrix.append(np.argmax(data_y_i_i))

        p_s = [8] * 8
        for i in range(len(p_s)):
            p_s[s[i]] = i

        s_hori_pairwise = [(p_s[0], p_s[1]), (p_s[1], p_s[2]), (p_s[3], 4), (4, p_s[4]), (p_s[5], p_s[6]),
                           (p_s[6], p_s[7])]
        s_vert_pairwise = [(p_s[0], p_s[3]), (p_s[1], 4), (p_s[2], p_s[4]), (p_s[3], p_s[5]), (4, p_s[6]),
                           (p_s[5], p_s[7])]

        y_true_hori_pairwise = [(true_matrix[0], true_matrix[1]), (true_matrix[1], true_matrix[2]), (true_matrix[3], 4),
                                (4, true_matrix[4]), (true_matrix[5], true_matrix[6]), (true_matrix[6], true_matrix[7])]
        y_true_vert_pairwise = [(true_matrix[0], true_matrix[3]), (true_matrix[1], 4), (true_matrix[2], true_matrix[4]),
                                (true_matrix[3], true_matrix[5]), (4, true_matrix[6]), (true_matrix[5], true_matrix[7])]

        greedy_whole_cate = 0
        for k in range(len(p_s)):
            if p_s[k] == true_matrix[k]:
                greedy_whole_cate += 1

        greedy_whole_perfect = 0
        if p_s == true_matrix:
            greedy_whole_perfect = 1

        greedy_whole_hori = len(set(s_hori_pairwise) & set(y_true_hori_pairwise))
        greedy_whole_vert = len(set(s_vert_pairwise) & set(y_true_vert_pairwise))

        return score, (greedy_whole_perfect, greedy_whole_cate, greedy_whole_hori, greedy_whole_vert)

    def evaluate_puzzleLet_33(self, s):
        cur_permutation = copy.deepcopy(s)

        # data_cate_i = self.data_cate[self.index]
        # data_hori_i = self.data_hori[self.index]
        # data_vert_i = self.data_vert[self.index]
        data_y_i = self.data_y[self.index]
        true_matrix = []
        for data_y_i_i in data_y_i:
            true_matrix.append(np.argmax(data_y_i_i))

        for cur_permutation_i in range(len(cur_permutation)):
            if cur_permutation[cur_permutation_i] >= 4:
                cur_permutation[cur_permutation_i] += 1
        cur_permutation.insert(4, 4)

        puzzlelet_score_list = []

        for cur_permutation_i in range(len(cur_permutation)):
            if (cur_permutation_i % 3 < 2) and (cur_permutation_i // 3 < 2):
                score_index = 8 * 7 * 6 * cur_permutation[cur_permutation_i]

                if cur_permutation[cur_permutation_i + 1] > cur_permutation[cur_permutation_i]:
                    score_index += 7 * 6 * (cur_permutation[cur_permutation_i + 1] - 1)
                else:
                    score_index += 7 * 6 * cur_permutation[cur_permutation_i + 1]

                if (cur_permutation[cur_permutation_i + 3] > cur_permutation[cur_permutation_i]) and \
                        (cur_permutation[cur_permutation_i + 3] > cur_permutation[cur_permutation_i + 1]):
                    score_index += 6 * (cur_permutation[cur_permutation_i + 3] - 2)
                elif (cur_permutation[cur_permutation_i + 3] > cur_permutation[cur_permutation_i]) or \
                        (cur_permutation[cur_permutation_i + 3] > cur_permutation[cur_permutation_i + 1]):
                    score_index += 6 * (cur_permutation[cur_permutation_i + 3] - 1)
                else:
                    score_index += 6 * cur_permutation[cur_permutation_i + 3]

                if (cur_permutation[cur_permutation_i + 4] > cur_permutation[cur_permutation_i]) and \
                        (cur_permutation[cur_permutation_i + 4] > cur_permutation[cur_permutation_i + 1]) and \
                        (cur_permutation[cur_permutation_i + 4] > cur_permutation[cur_permutation_i + 3]):
                    score_index += cur_permutation[cur_permutation_i + 4] - 3
                elif ((cur_permutation[cur_permutation_i + 4] > cur_permutation[cur_permutation_i]) and \
                      (cur_permutation[cur_permutation_i + 4] > cur_permutation[cur_permutation_i + 1])) or \
                        ((cur_permutation[cur_permutation_i + 4] > cur_permutation[cur_permutation_i]) and \
                         (cur_permutation[cur_permutation_i + 4] > cur_permutation[cur_permutation_i + 3])) or \
                        ((cur_permutation[cur_permutation_i + 4] > cur_permutation[cur_permutation_i + 1]) and \
                         (cur_permutation[cur_permutation_i + 4] > cur_permutation[cur_permutation_i + 3])):
                    score_index += cur_permutation[cur_permutation_i + 4] - 2
                elif (cur_permutation[cur_permutation_i + 4] > cur_permutation[cur_permutation_i]) or \
                        (cur_permutation[cur_permutation_i + 4] > cur_permutation[cur_permutation_i + 1]) or \
                        (cur_permutation[cur_permutation_i + 4] > cur_permutation[cur_permutation_i + 3]):
                    score_index += cur_permutation[cur_permutation_i + 4] - 1
                else:
                    score_index += cur_permutation[cur_permutation_i + 4]

                puzzlelet_score_list.append(self.data_puzzleLet[self.index, score_index])
        score = sum(puzzlelet_score_list)[0]

        # reward_0 = data_hori_i[cur_permutation[0]][cur_permutation[1]] * data_cate_i[s[1]][1] + \
        #            data_vert_i[cur_permutation[0]][cur_permutation[3]] * data_cate_i[s[3]][3]
        # reward_1 = data_hori_i[cur_permutation[0]][cur_permutation[1]] * data_cate_i[s[0]][0] + \
        #            data_vert_i[cur_permutation[1]][4] * 1 + \
        #            data_hori_i[cur_permutation[1]][cur_permutation[2]] * data_cate_i[s[2]][2]
        # reward_2 = data_hori_i[cur_permutation[1]][cur_permutation[2]] * data_cate_i[s[1]][1] + \
        #            data_vert_i[cur_permutation[2]][cur_permutation[5]] * data_cate_i[s[5]][4]
        # reward_3 = data_vert_i[cur_permutation[0]][cur_permutation[3]] * data_cate_i[s[0]][0] + \
        #            data_hori_i[cur_permutation[3]][4] * 1 + \
        #            data_vert_i[cur_permutation[3]][cur_permutation[6]] * data_cate_i[s[6]][5]
        # reward_5 = data_vert_i[cur_permutation[2]][cur_permutation[4]] * data_cate_i[s[2]][2] + \
        #            data_hori_i[4][cur_permutation[4]] * 1 + \
        #            data_vert_i[cur_permutation[4]][cur_permutation[7]] * data_cate_i[s[7]][7]
        # reward_6 = data_hori_i[cur_permutation[5]][cur_permutation[6]] * data_cate_i[s[6]][6] + \
        #            data_vert_i[cur_permutation[3]][cur_permutation[5]] * data_cate_i[s[3]][3]
        # reward_7 = data_hori_i[cur_permutation[5]][cur_permutation[6]] * data_cate_i[s[5]][5] + \
        #            data_vert_i[4][cur_permutation[6]] * 1 + \
        #            data_hori_i[cur_permutation[6]][cur_permutation[7]] * data_cate_i[s[7]][7]
        # reward_8 = data_hori_i[cur_permutation[6]][cur_permutation[7]] * data_cate_i[s[6]][6] + \
        #            data_vert_i[cur_permutation[4]][cur_permutation[7]] * data_cate_i[s[4]][4]
        # score = score + reward_0 + reward_1 + reward_2 + reward_3 + reward_5 + reward_6 + reward_7 + reward_8

        p_s = [8] * 8
        for i in range(len(p_s)):
            p_s[s[i]] = i

        s_hori_pairwise = [(p_s[0], p_s[1]), (p_s[1], p_s[2]), (p_s[3], 4), (4, p_s[4]), (p_s[5], p_s[6]),
                           (p_s[6], p_s[7])]
        s_vert_pairwise = [(p_s[0], p_s[3]), (p_s[1], 4), (p_s[2], p_s[4]), (p_s[3], p_s[5]), (4, p_s[6]),
                           (p_s[5], p_s[7])]

        y_true_hori_pairwise = [(true_matrix[0], true_matrix[1]), (true_matrix[1], true_matrix[2]), (true_matrix[3], 4),
                                (4, true_matrix[4]), (true_matrix[5], true_matrix[6]), (true_matrix[6], true_matrix[7])]
        y_true_vert_pairwise = [(true_matrix[0], true_matrix[3]), (true_matrix[1], 4), (true_matrix[2], true_matrix[4]),
                                (true_matrix[3], true_matrix[5]), (4, true_matrix[6]), (true_matrix[5], true_matrix[7])]

        greedy_whole_cate = 0
        for k in range(len(p_s)):
            if p_s[k] == true_matrix[k]:
                greedy_whole_cate += 1

        greedy_whole_perfect = 0
        if p_s == true_matrix:
            greedy_whole_perfect = 1

        greedy_whole_hori = len(set(s_hori_pairwise) & set(y_true_hori_pairwise))
        greedy_whole_vert = len(set(s_vert_pairwise) & set(y_true_vert_pairwise))
        return score, (greedy_whole_perfect, greedy_whole_cate, greedy_whole_hori, greedy_whole_vert)

    def get_hori_55_feature(self, s):
        cur_permutation = copy.deepcopy(s)
        data_hori_i = self.data_hori[self.index]
        hori_feature = []

        for cur_permutation_i in range(len(cur_permutation)):
            if cur_permutation[cur_permutation_i] >= 12:
                cur_permutation[cur_permutation_i] += 1
        cur_permutation.insert(12, 12)

        for cur_permutation_i in range(len(cur_permutation)):
            if cur_permutation_i != 12:
                if cur_permutation_i % 5 != 0:
                    hori_feature.append(data_hori_i[cur_permutation[cur_permutation_i - 1],
                                                    cur_permutation[cur_permutation_i]])
                if cur_permutation_i % 5 != 4:
                    hori_feature.append(data_hori_i[cur_permutation[cur_permutation_i],
                                                    cur_permutation[cur_permutation_i + 1]])
        return hori_feature

    def get_vert_55_feature(self, s):
        cur_permutation = copy.deepcopy(s)
        data_vert_i = self.data_vert[self.index]
        vert_feature = []

        for cur_permutation_i in range(len(cur_permutation)):
            if cur_permutation[cur_permutation_i] >= 12:
                cur_permutation[cur_permutation_i] += 1
        cur_permutation.insert(12, 12)

        for cur_permutation_i in range(len(cur_permutation)):
            if cur_permutation_i != 12:
                if cur_permutation_i % 5 != 0:
                    vert_feature.append(data_vert_i[cur_permutation[cur_permutation_i - 1],
                                                    cur_permutation[cur_permutation_i]])
                if cur_permutation_i % 5 != 4:
                    vert_feature.append(data_vert_i[cur_permutation[cur_permutation_i],
                                                    cur_permutation[cur_permutation_i + 1]])
        return vert_feature

    def evaluate_55(self, s):
        cur_permutation = copy.deepcopy(s)

        data_y_i = self.data_y[self.index]
        target_loc = []
        for data_y_i_i in data_y_i:
            target_loc.append(np.argmax(data_y_i_i))
        true_matrix = [24] * 24
        for i in range(len(target_loc)):
            true_matrix[target_loc[i]] = i

        # data_cate_i = self.data_cate[self.index]
        data_cate_i = np.insert(self.data_cate[self.index], 12, [0] * 24, axis=0)
        data_cate_i = np.insert(data_cate_i, 12, [0] * 25, axis=1)
        data_cate_i[12][12] = 1
        data_hori_i = self.data_hori[self.index]
        data_vert_i = self.data_vert[self.index]

        score = 0.0

        for cur_permutation_i in range(len(cur_permutation)):
            if cur_permutation[cur_permutation_i] >= 12:
                cur_permutation[cur_permutation_i] += 1
        cur_permutation.insert(12, 12)

        for cur_permutation_i in range(len(cur_permutation)):
            if cur_permutation_i != 12:
                if cur_permutation_i % 5 != 0:
                    score += data_hori_i[cur_permutation[cur_permutation_i - 1], cur_permutation[cur_permutation_i]] * \
                             data_cate_i[cur_permutation[cur_permutation_i], cur_permutation_i]
                if cur_permutation_i % 5 != 4:
                    score += data_hori_i[cur_permutation[cur_permutation_i], cur_permutation[cur_permutation_i + 1]] * \
                             data_cate_i[cur_permutation[cur_permutation_i], cur_permutation_i]
                if cur_permutation_i // 5 != 0:
                    score += data_vert_i[cur_permutation[cur_permutation_i - 5], cur_permutation[cur_permutation_i]] * \
                             data_cate_i[cur_permutation[cur_permutation_i], cur_permutation_i]
                if cur_permutation_i // 5 != 4:
                    score += data_vert_i[cur_permutation[cur_permutation_i], cur_permutation[cur_permutation_i + 5]] * \
                             data_cate_i[cur_permutation[cur_permutation_i], cur_permutation_i]

            '''
            # for the JPwwLEG-5 with 4 fixed fragment in 4-corner
            index_0 = true_matrix.index(0)
            index_4 = true_matrix.index(4)
            index_20 = true_matrix.index(19)
            index_24 = true_matrix.index(23)
    
            cur_permutation.remove(index_0)
            cur_permutation.remove(index_4)
            cur_permutation.remove(index_20)
            cur_permutation.remove(index_24)
    
            score = data_cate_i[cur_permutation[0], 1] + data_cate_i[cur_permutation[1], 2] + \
                    data_cate_i[cur_permutation[2], 3] + data_cate_i[cur_permutation[3], 5] + \
                    data_cate_i[cur_permutation[4], 6] + data_cate_i[cur_permutation[5], 7] + data_cate_i[
                        cur_permutation[6], 8] + \
                    data_cate_i[cur_permutation[7], 9] + data_cate_i[cur_permutation[8], 10] + data_cate_i[
                        cur_permutation[9], 11] + \
                    data_cate_i[cur_permutation[10], 12] + data_cate_i[cur_permutation[11], 13] + data_cate_i[
                        cur_permutation[12], 14] + \
                    data_cate_i[cur_permutation[13], 15] + data_cate_i[cur_permutation[14], 16] + data_cate_i[
                        cur_permutation[15], 17] + \
                    data_cate_i[cur_permutation[16], 18] + data_cate_i[cur_permutation[17], 20] + \
                    data_cate_i[cur_permutation[18], 21] + data_cate_i[cur_permutation[19], 22]
    
            score = data_cate_i[cur_permutation[0], 0] + data_cate_i[cur_permutation[1], 1] + data_cate_i[
                        cur_permutation[2], 2] + \
                    data_cate_i[cur_permutation[3], 3] + data_cate_i[cur_permutation[4], 4] + data_cate_i[
                        cur_permutation[5], 5] + \
                    data_cate_i[cur_permutation[6], 6] + data_cate_i[cur_permutation[7], 7] + data_cate_i[
                        cur_permutation[8], 8] + \
                    data_cate_i[cur_permutation[9], 9] + data_cate_i[cur_permutation[10], 10] + data_cate_i[
                        cur_permutation[11], 11] + \
                    data_cate_i[cur_permutation[12], 12] + data_cate_i[cur_permutation[13], 13] + data_cate_i[
                        cur_permutation[14], 14] + \
                    data_cate_i[cur_permutation[15], 15] + data_cate_i[cur_permutation[16], 16] + data_cate_i[
                        cur_permutation[17], 17] + \
                    data_cate_i[cur_permutation[17], 18] + data_cate_i[cur_permutation[19], 19] + data_cate_i[
                        cur_permutation[20], 20] + \
                    data_cate_i[cur_permutation[21], 21] + data_cate_i[cur_permutation[22], 22] + data_cate_i[
                        cur_permutation[23], 23]
                        
            p_s = [24] * 24
            p_s[index_0] = true_matrix[index_0]
            p_s[cur_permutation[0]] = 1
            p_s[cur_permutation[1]] = 2
            p_s[cur_permutation[2]] = 3
            p_s[index_4] = true_matrix[index_4]
            p_s[cur_permutation[3]] = 5
            p_s[cur_permutation[4]] = 6
            p_s[cur_permutation[5]] = 7
            p_s[cur_permutation[6]] = 8
            p_s[cur_permutation[7]] = 9
            p_s[cur_permutation[8]] = 10
            p_s[cur_permutation[9]] = 11
            p_s[cur_permutation[10]] = 12
            p_s[cur_permutation[11]] = 13
            p_s[cur_permutation[12]] = 14
            p_s[cur_permutation[13]] = 15
            p_s[cur_permutation[14]] = 16
            p_s[cur_permutation[15]] = 17
            p_s[cur_permutation[16]] = 18
            p_s[index_20] = true_matrix[index_20]
            p_s[cur_permutation[17]] = 20
            p_s[cur_permutation[18]] = 21
            p_s[cur_permutation[19]] = 22
            p_s[index_24] = true_matrix[index_24]
            '''
        p_s = copy.deepcopy(s)

        s_hori_pairwise = [(p_s[0], p_s[1]), (p_s[1], p_s[2]), (p_s[3], p_s[4]), (p_s[4], p_s[5]),
                           (p_s[5], p_s[6]), (p_s[6], p_s[7]), (p_s[7], p_s[8]), (p_s[8], p_s[9]),
                           (p_s[10], p_s[11]), (p_s[11], 12), (12, p_s[12]), (p_s[12], p_s[13]),
                           (p_s[14], p_s[15]), (p_s[15], p_s[16]), (p_s[16], p_s[17]), (p_s[17], p_s[18]),
                           (p_s[19], p_s[20]), (p_s[20], p_s[21]), (p_s[21], p_s[22]), (p_s[22], p_s[23])]
        s_vert_pairwise = [(p_s[0], p_s[5]), (p_s[1], p_s[6]), (p_s[2], p_s[7]), (p_s[3], p_s[8]), (p_s[4], p_s[9]),
                           (p_s[5], p_s[10]), (p_s[6], p_s[11]), (p_s[7], 12), (p_s[8], p_s[12]), (p_s[9], p_s[13]),
                           (p_s[10], p_s[14]), (p_s[11], p_s[15]), (12, p_s[16]), (p_s[12], p_s[17]),
                           (p_s[13], p_s[18]),
                           (p_s[14], p_s[19]), (p_s[15], p_s[20]), (p_s[16], p_s[21]), (p_s[17], p_s[22]),
                           (p_s[18], p_s[23])]

        y_true_hori_pairwise = [(true_matrix[0], true_matrix[1]), (true_matrix[1], true_matrix[2]),
                                (true_matrix[3], true_matrix[4]), (true_matrix[4], true_matrix[5]),
                                (true_matrix[5], true_matrix[6]), (true_matrix[6], true_matrix[7]),
                                (true_matrix[7], true_matrix[8]), (true_matrix[8], true_matrix[9]),
                                (true_matrix[10], true_matrix[11]), (true_matrix[11], 12), (12, true_matrix[12]),
                                (true_matrix[12], true_matrix[13]),
                                (true_matrix[14], true_matrix[15]), (true_matrix[15], true_matrix[16]),
                                (true_matrix[16], true_matrix[17]), (true_matrix[17], true_matrix[18]),
                                (true_matrix[19], true_matrix[20]), (true_matrix[20], true_matrix[21]),
                                (true_matrix[21], true_matrix[22]), (true_matrix[22], true_matrix[23])]
        y_true_vert_pairwise = [(true_matrix[0], true_matrix[5]), (true_matrix[1], true_matrix[6]),
                                (true_matrix[2], true_matrix[7]), (true_matrix[3], true_matrix[8]),
                                (true_matrix[4], true_matrix[9]),
                                (true_matrix[5], true_matrix[10]), (true_matrix[6], true_matrix[11]),
                                (true_matrix[7], 12), (true_matrix[8], true_matrix[12]),
                                (true_matrix[9], true_matrix[13]),
                                (true_matrix[10], true_matrix[14]), (true_matrix[11], true_matrix[15]),
                                (12, true_matrix[16]), (true_matrix[12], true_matrix[17]),
                                (true_matrix[13], true_matrix[18]),
                                (true_matrix[14], true_matrix[19]), (true_matrix[15], true_matrix[20]),
                                (true_matrix[16], true_matrix[21]), (true_matrix[17], true_matrix[22]),
                                (true_matrix[18], true_matrix[23])]
        y_true_33 = [true_matrix[6], true_matrix[7], true_matrix[8], true_matrix[11],
                     true_matrix[12], true_matrix[16], true_matrix[17], true_matrix[18]]
        p_s_33 = [p_s[6], p_s[7], p_s[8], p_s[11], p_s[12], p_s[16], p_s[17], p_s[18]]

        greedy_whole_cate = 0
        for k in range(len(p_s)):
            if p_s[k] == true_matrix[k]:
                greedy_whole_cate += 1

        greedy_whole_33 = 0
        if y_true_33 == p_s_33:
            greedy_whole_33 = 1

        greedy_whole_perfect = 0
        if p_s == true_matrix:
            greedy_whole_perfect = 1

        greedy_whole_hori = len(set(s_hori_pairwise) & set(y_true_hori_pairwise))
        greedy_whole_vert = len(set(s_vert_pairwise) & set(y_true_vert_pairwise))

        return score, (greedy_whole_perfect, greedy_whole_cate, greedy_whole_hori, greedy_whole_vert, greedy_whole_33, p_s)

    def get_puzzleLet_55_feature(self, s):
        cur_permutation = copy.deepcopy(s)

        for cur_permutation_i in range(len(cur_permutation)):
            if cur_permutation[cur_permutation_i] >= 12:
                cur_permutation[cur_permutation_i] += 1
        cur_permutation.insert(12, 12)

        puzzlelet_score_list = []

        for cur_permutation_i in range(len(cur_permutation)):
            if (cur_permutation_i % 5 < 4) and (cur_permutation_i // 5 < 4):
                score_index = 24 * 23 * 22 * cur_permutation[cur_permutation_i]

                if cur_permutation[cur_permutation_i + 1] > cur_permutation[cur_permutation_i]:
                    score_index += 23 * 22 * (cur_permutation[cur_permutation_i + 1] - 1)
                else:
                    score_index += 23 * 22 * cur_permutation[cur_permutation_i + 1]

                if (cur_permutation[cur_permutation_i + 5] > cur_permutation[cur_permutation_i]) and \
                        (cur_permutation[cur_permutation_i + 5] > cur_permutation[cur_permutation_i + 1]):
                    score_index += 22 * (cur_permutation[cur_permutation_i + 5] - 2)
                elif (cur_permutation[cur_permutation_i + 5] > cur_permutation[cur_permutation_i]) or \
                        (cur_permutation[cur_permutation_i + 5] > cur_permutation[cur_permutation_i + 1]):
                    score_index += 22 * (cur_permutation[cur_permutation_i + 5] - 1)
                else:
                    score_index += 22 * cur_permutation[cur_permutation_i + 5]

                if (cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i]) and \
                        (cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i + 1]) and \
                        (cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i + 5]):
                    score_index += cur_permutation[cur_permutation_i + 6] - 3
                elif ((cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i]) and \
                      (cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i + 1])) or \
                        ((cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i]) and \
                         (cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i + 5])) or \
                        ((cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i + 1]) and \
                         (cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i + 5])):
                    score_index += cur_permutation[cur_permutation_i + 6] - 2
                elif (cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i]) or \
                        (cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i + 1]) or \
                        (cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i + 5]):
                    score_index += cur_permutation[cur_permutation_i + 6] - 1
                else:
                    score_index += cur_permutation[cur_permutation_i + 6]

                puzzlelet_score_list.append(1 - self.data_puzzleLet[self.index, score_index])

        return puzzlelet_score_list

    def evaluate_puzzleLet_55(self, s):
        cur_permutation = copy.deepcopy(s)

        data_y_i = self.data_y[self.index]

        target_loc = []
        for data_y_i_i in data_y_i:
            target_loc.append(np.argmax(data_y_i_i))
        true_matrix = [24] * 24
        for i in range(len(target_loc)):
            true_matrix[target_loc[i]] = i

        for cur_permutation_i in range(len(cur_permutation)):
            if cur_permutation[cur_permutation_i] >= 12:
                cur_permutation[cur_permutation_i] += 1
        cur_permutation.insert(12, 12)

        puzzlelet_score_list = []

        for cur_permutation_i in range(len(cur_permutation)):
            if (cur_permutation_i % 5 < 4) and (cur_permutation_i // 5 < 4):
                score_index = 24 * 23 * 22 * cur_permutation[cur_permutation_i]

                if cur_permutation[cur_permutation_i + 1] > cur_permutation[cur_permutation_i]:
                    score_index += 23 * 22 * (cur_permutation[cur_permutation_i + 1] - 1)
                else:
                    score_index += 23 * 22 * cur_permutation[cur_permutation_i + 1]

                if (cur_permutation[cur_permutation_i + 5] > cur_permutation[cur_permutation_i]) and \
                        (cur_permutation[cur_permutation_i + 5] > cur_permutation[cur_permutation_i + 1]):
                    score_index += 22 * (cur_permutation[cur_permutation_i + 5] - 2)
                elif (cur_permutation[cur_permutation_i + 5] > cur_permutation[cur_permutation_i]) or \
                        (cur_permutation[cur_permutation_i + 5] > cur_permutation[cur_permutation_i + 1]):
                    score_index += 22 * (cur_permutation[cur_permutation_i + 5] - 1)
                else:
                    score_index += 22 * cur_permutation[cur_permutation_i + 5]

                if (cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i]) and \
                        (cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i + 1]) and \
                        (cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i + 5]):
                    score_index += cur_permutation[cur_permutation_i + 6] - 3
                elif ((cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i]) and \
                      (cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i + 1])) or \
                        ((cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i]) and \
                         (cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i + 5])) or \
                        ((cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i + 1]) and \
                         (cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i + 5])):
                    score_index += cur_permutation[cur_permutation_i + 6] - 2
                elif (cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i]) or \
                        (cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i + 1]) or \
                        (cur_permutation[cur_permutation_i + 6] > cur_permutation[cur_permutation_i + 5]):
                    score_index += cur_permutation[cur_permutation_i + 6] - 1
                else:
                    score_index += cur_permutation[cur_permutation_i + 6]

                puzzlelet_score_list.append(self.data_puzzleLet[self.index, score_index])

        score = 16 - sum(puzzlelet_score_list)[0]

        p_s = copy.deepcopy(s)

        s_hori_pairwise = [(p_s[0], p_s[1]), (p_s[1], p_s[2]), (p_s[3], p_s[4]), (p_s[4], p_s[5]),
                           (p_s[5], p_s[6]), (p_s[6], p_s[7]), (p_s[7], p_s[8]), (p_s[8], p_s[9]),
                           (p_s[10], p_s[11]), (p_s[11], 12), (12, p_s[12]), (p_s[12], p_s[13]),
                           (p_s[14], p_s[15]), (p_s[15], p_s[16]), (p_s[16], p_s[17]), (p_s[17], p_s[18]),
                           (p_s[19], p_s[20]), (p_s[20], p_s[21]), (p_s[21], p_s[22]), (p_s[22], p_s[23])]
        s_vert_pairwise = [(p_s[0], p_s[5]), (p_s[1], p_s[6]), (p_s[2], p_s[7]), (p_s[3], p_s[8]), (p_s[4], p_s[9]),
                           (p_s[5], p_s[10]), (p_s[6], p_s[11]), (p_s[7], 12), (p_s[8], p_s[12]), (p_s[9], p_s[13]),
                           (p_s[10], p_s[14]), (p_s[11], p_s[15]), (12, p_s[16]), (p_s[12], p_s[17]),
                           (p_s[13], p_s[18]),
                           (p_s[14], p_s[19]), (p_s[15], p_s[20]), (p_s[16], p_s[21]), (p_s[17], p_s[22]),
                           (p_s[18], p_s[23])]

        y_true_hori_pairwise = [(true_matrix[0], true_matrix[1]), (true_matrix[1], true_matrix[2]),
                                (true_matrix[3], true_matrix[4]), (true_matrix[4], true_matrix[5]),
                                (true_matrix[5], true_matrix[6]), (true_matrix[6], true_matrix[7]),
                                (true_matrix[7], true_matrix[8]), (true_matrix[8], true_matrix[9]),
                                (true_matrix[10], true_matrix[11]), (true_matrix[11], 12), (12, true_matrix[12]),
                                (true_matrix[12], true_matrix[13]),
                                (true_matrix[14], true_matrix[15]), (true_matrix[15], true_matrix[16]),
                                (true_matrix[16], true_matrix[17]), (true_matrix[17], true_matrix[18]),
                                (true_matrix[19], true_matrix[20]), (true_matrix[20], true_matrix[21]),
                                (true_matrix[21], true_matrix[22]), (true_matrix[22], true_matrix[23])]
        y_true_vert_pairwise = [(true_matrix[0], true_matrix[5]), (true_matrix[1], true_matrix[6]),
                                (true_matrix[2], true_matrix[7]), (true_matrix[3], true_matrix[8]),
                                (true_matrix[4], true_matrix[9]),
                                (true_matrix[5], true_matrix[10]), (true_matrix[6], true_matrix[11]),
                                (true_matrix[7], 12), (true_matrix[8], true_matrix[12]),
                                (true_matrix[9], true_matrix[13]),
                                (true_matrix[10], true_matrix[14]), (true_matrix[11], true_matrix[15]),
                                (12, true_matrix[16]), (true_matrix[12], true_matrix[17]),
                                (true_matrix[13], true_matrix[18]),
                                (true_matrix[14], true_matrix[19]), (true_matrix[15], true_matrix[20]),
                                (true_matrix[16], true_matrix[21]), (true_matrix[17], true_matrix[22]),
                                (true_matrix[18], true_matrix[23])]
        y_true_33 = [true_matrix[6], true_matrix[7], true_matrix[8], true_matrix[11],
                     true_matrix[12], true_matrix[16], true_matrix[17], true_matrix[18]]
        p_s_33 = [p_s[6], p_s[7], p_s[8], p_s[11], p_s[12], p_s[16], p_s[17], p_s[18]]

        greedy_whole_cate = 0
        for k in range(len(p_s)):
            if p_s[k] == true_matrix[k]:
                greedy_whole_cate += 1

        greedy_whole_33 = 0
        if y_true_33 == p_s_33:
            greedy_whole_33 = 1

        greedy_whole_perfect = 0
        if p_s == true_matrix:
            greedy_whole_perfect = 1

        greedy_whole_hori = len(set(s_hori_pairwise) & set(y_true_hori_pairwise))
        greedy_whole_vert = len(set(s_vert_pairwise) & set(y_true_vert_pairwise))

        return score, (greedy_whole_perfect, greedy_whole_cate, greedy_whole_hori, greedy_whole_vert,
                       greedy_whole_33, p_s)

    def evaluate_70(self, s):
        cur_permutation = copy.deepcopy(s)

        data_y_i = self.data_y[self.index]
        target_loc = []
        for data_y_i_i in data_y_i:
            target_loc.append(np.argmax(data_y_i_i))
        true_matrix = [70] * 70
        for i in range(len(target_loc)):
            true_matrix[target_loc[i]] = i

        data_hori_i = self.data_hori[self.index]
        data_vert_i = self.data_vert[self.index]

        score = 0.0
        best_score = 0.0

        for cur_i in range(len(cur_permutation)):
            if cur_i % 10 != 9:
                score += data_hori_i[cur_permutation[cur_i], cur_permutation[cur_i + 1]]
            if cur_i // 10 != 6:
                score += data_vert_i[cur_permutation[cur_i], cur_permutation[cur_i + 10]]

        p_s = copy.deepcopy(s)

        s_hori_pairwise = []
        s_vert_pairwise = []
        y_true_hori_pairwise = []
        y_true_vert_pairwise = []
        for cur_permutation_i in range(len(cur_permutation)):
            if cur_permutation_i % 10 != 9:
                s_hori_pairwise.append((p_s[cur_permutation_i], p_s[cur_permutation_i + 1]))
                y_true_hori_pairwise.append((true_matrix[cur_permutation_i], true_matrix[cur_permutation_i + 1]))
            if cur_permutation_i // 10 != 6:
                s_vert_pairwise.append((p_s[cur_permutation_i], p_s[cur_permutation_i + 10]))
                y_true_vert_pairwise.append((true_matrix[cur_permutation_i], true_matrix[cur_permutation_i + 10]))

        cate = 0
        for k in range(len(p_s)):
            if p_s[k] == true_matrix[k]:
                cate += 1

        perfect = 0
        if p_s == true_matrix:
            perfect = 1

        hori = len(set(s_hori_pairwise) & set(y_true_hori_pairwise))
        vert = len(set(s_vert_pairwise) & set(y_true_vert_pairwise))
        return score, (perfect, cate, hori, vert, true_matrix)

    def evaluate_puzzleLet_70(self, s):
        cur_permutation = copy.deepcopy(s)

        data_y_i = self.data_y[self.index]
        target_loc = []
        for data_y_i_i in data_y_i:
            target_loc.append(np.argmax(data_y_i_i))
        true_matrix = [70] * 70
        for i in range(len(target_loc)):
            true_matrix[target_loc[i]] = i

        puzzlelet_score_list = []

        for cur_permutation_i in range(len(cur_permutation)):
            if (cur_permutation_i % 10 < 9) and (cur_permutation_i // 10 < 6):
                score_index = 69 * 68 * 67 * cur_permutation[cur_permutation_i]

                if cur_permutation[cur_permutation_i + 1] > cur_permutation[cur_permutation_i]:
                    score_index += 68 * 67 * (cur_permutation[cur_permutation_i + 1] - 1)
                else:
                    score_index += 68 * 67 * cur_permutation[cur_permutation_i + 1]

                if (cur_permutation[cur_permutation_i + 10] > cur_permutation[cur_permutation_i]) and \
                        (cur_permutation[cur_permutation_i + 10] > cur_permutation[cur_permutation_i + 1]):
                    score_index += 67 * (cur_permutation[cur_permutation_i + 10] - 2)
                elif (cur_permutation[cur_permutation_i + 10] > cur_permutation[cur_permutation_i]) or \
                        (cur_permutation[cur_permutation_i + 10] > cur_permutation[cur_permutation_i + 1]):
                    score_index += 67 * (cur_permutation[cur_permutation_i + 10] - 1)
                else:
                    score_index += 67 * cur_permutation[cur_permutation_i + 10]

                if (cur_permutation[cur_permutation_i + 11] > cur_permutation[cur_permutation_i]) and \
                        (cur_permutation[cur_permutation_i + 11] > cur_permutation[cur_permutation_i + 1]) and \
                        (cur_permutation[cur_permutation_i + 11] > cur_permutation[cur_permutation_i + 10]):
                    score_index += cur_permutation[cur_permutation_i + 11] - 3
                elif ((cur_permutation[cur_permutation_i + 11] > cur_permutation[cur_permutation_i]) and \
                      (cur_permutation[cur_permutation_i + 11] > cur_permutation[cur_permutation_i + 1])) or \
                        ((cur_permutation[cur_permutation_i + 11] > cur_permutation[cur_permutation_i]) and \
                         (cur_permutation[cur_permutation_i + 11] > cur_permutation[cur_permutation_i + 10])) or \
                        ((cur_permutation[cur_permutation_i + 11] > cur_permutation[cur_permutation_i + 1]) and \
                         (cur_permutation[cur_permutation_i + 11] > cur_permutation[cur_permutation_i + 10])):
                    score_index += cur_permutation[cur_permutation_i + 11] - 2
                elif (cur_permutation[cur_permutation_i + 11] > cur_permutation[cur_permutation_i]) or \
                        (cur_permutation[cur_permutation_i + 11] > cur_permutation[cur_permutation_i + 1]) or \
                        (cur_permutation[cur_permutation_i + 11] > cur_permutation[cur_permutation_i + 10]):
                    score_index += cur_permutation[cur_permutation_i + 11] - 1
                else:
                    score_index += cur_permutation[cur_permutation_i + 11]

                puzzlelet_score_list.append(self.data_puzzleLet[self.index, score_index])

        score = sum(puzzlelet_score_list)
        p_s = copy.deepcopy(s)

        s_hori_pairwise = []
        s_vert_pairwise = []
        y_true_hori_pairwise = []
        y_true_vert_pairwise = []
        for cur_permutation_i in range(len(cur_permutation)):
            if cur_permutation_i % 10 != 9:
                s_hori_pairwise.append((p_s[cur_permutation_i], p_s[cur_permutation_i + 1]))
                y_true_hori_pairwise.append((true_matrix[cur_permutation_i], true_matrix[cur_permutation_i + 1]))
            if cur_permutation_i // 10 != 6:
                s_vert_pairwise.append((p_s[cur_permutation_i], p_s[cur_permutation_i + 10]))
                y_true_vert_pairwise.append((true_matrix[cur_permutation_i], true_matrix[cur_permutation_i + 10]))

        cate = 0
        for k in range(len(p_s)):
            if p_s[k] == true_matrix[k]:
                cate += 1

        perfect = 0
        if p_s == true_matrix:
            perfect = 1

        hori = len(set(s_hori_pairwise) & set(y_true_hori_pairwise))
        vert = len(set(s_vert_pairwise) & set(y_true_vert_pairwise))
        return score, (perfect, cate, hori, vert, true_matrix)

    def evaluate_150(self, s):
        cur_permutation = copy.deepcopy(s)

        data_y_i = self.data_y[self.index]
        target_loc = []
        for data_y_i_i in data_y_i:
            target_loc.append(np.argmax(data_y_i_i))
        true_matrix = [150] * 150
        for i in range(len(target_loc)):
            true_matrix[target_loc[i]] = i

        data_hori_i = self.data_hori[self.index]
        data_vert_i = self.data_vert[self.index]

        score = 0.0
        best_score = 0.0

        for cur_i in range(len(cur_permutation)):
            if cur_i % 15 != 14:
                score += data_hori_i[cur_permutation[cur_i], cur_permutation[cur_i + 1]]
            if cur_i // 15 != 9:
                score += data_vert_i[cur_permutation[cur_i], cur_permutation[cur_i + 15]]

        p_s = copy.deepcopy(s)

        s_hori_pairwise = []
        s_vert_pairwise = []
        y_true_hori_pairwise = []
        y_true_vert_pairwise = []
        for cur_permutation_i in range(len(cur_permutation)):
            if cur_permutation_i % 15 != 14:
                s_hori_pairwise.append((p_s[cur_permutation_i], p_s[cur_permutation_i + 1]))
                y_true_hori_pairwise.append((true_matrix[cur_permutation_i], true_matrix[cur_permutation_i + 1]))
            if cur_permutation_i // 15 != 9:
                s_vert_pairwise.append((p_s[cur_permutation_i], p_s[cur_permutation_i + 15]))
                y_true_vert_pairwise.append((true_matrix[cur_permutation_i], true_matrix[cur_permutation_i + 15]))

        cate = 0
        for k in range(len(p_s)):
            if p_s[k] == true_matrix[k]:
                cate += 1

        perfect = 0
        if p_s == true_matrix:
            perfect = 1

        hori = len(set(s_hori_pairwise) & set(y_true_hori_pairwise))
        vert = len(set(s_vert_pairwise) & set(y_true_vert_pairwise))
        return score, (perfect, cate, hori, vert, true_matrix)

