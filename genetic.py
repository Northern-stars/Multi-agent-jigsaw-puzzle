# -*- coding:utf-8 -*-
# Auther: Chenglin Yao
# Time: 2022/3/3 下午1:38
# Location: AI Lab
import itertools
import os
import random, time
import numpy as np
import multiprocessing
from Evaluator_dict import Evaluator

random.seed(42)


class GeneticAlgorithm:
    def __init__(self, eval, num_population=64, crossover_rate=0.8, mutation_rate=0.2, variety_decay=0.8, max_iters=20):
        self.eval = eval
        self.num_population = num_population
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.variety_decay = variety_decay
        self.max_iters = max_iters

    def evaluate(self, solution, eval_method='pairwise_info'):
        score, other = self.eval.evaluate(list(solution), eval_method)
        return score, other

    def crossover(self, chrom1, chrom2):
        chrom1 = list(chrom1)
        chrom2 = list(chrom2)
        # print("before:", chrom1, chrom2)
        if random.random() < self.crossover_rate:
            rand_sequence = [i for i in range(len(chrom1))]
            random.shuffle(rand_sequence)
            # if random.random() < 2:
            selected_indices = [rand_sequence[0], rand_sequence[1]]
            selected_numbers = [chrom1[i] for i in selected_indices]
            related_indices = [chrom2.index(i) for i in selected_numbers]

            for i in range(len(selected_indices)):
                swap_index = [selected_indices[i], related_indices[i]]
                if swap_index[0] != swap_index[1]:
                    a, b = chrom1[swap_index[0]], chrom1[swap_index[1]]
                    chrom1[swap_index[0]], chrom1[swap_index[1]] = b, a

                    a, b = chrom2[swap_index[0]], chrom2[swap_index[1]]
                    chrom2[swap_index[0]], chrom2[swap_index[1]] = b, a
            # else:
            #     selected_indices = rand_sequence[:3]
            #     selected_numbers = [chrom1[i] for i in selected_indices]
            #     related_indices = [chrom2.index(i) for i in selected_numbers]
            #
            #     for i in range(len(selected_indices)):
            #         swap_index = [selected_indices[i], related_indices[i]]
            #         if swap_index[0] != swap_index[1]:
            #             chrom1[swap_index[0]] = chrom1[swap_index[0]] + chrom1[swap_index[1]]
            #             chrom1[swap_index[1]] = chrom1[swap_index[0]] - chrom1[swap_index[1]]
            #             chrom1[swap_index[0]] = chrom1[swap_index[0]] - chrom1[swap_index[1]]
            #
            #             chrom2[swap_index[0]] = chrom2[swap_index[0]] + chrom2[swap_index[1]]
            #             chrom2[swap_index[1]] = chrom2[swap_index[0]] - chrom2[swap_index[1]]
            #             chrom2[swap_index[0]] = chrom2[swap_index[0]] - chrom2[swap_index[1]]
            for i in range(len(chrom1)):
                if (i not in chrom1) or (i not in chrom2):
                    print("after:", chrom1, chrom2, selected_indices, selected_numbers, related_indices)
        return np.array(chrom1), np.array(chrom2)

    def mutation(self, chrom):
        if random.random() < self.mutation_rate:
            rand_sequence = [i for i in range(len(chrom))]
            random.shuffle(rand_sequence)
            if random.random() < .2:
                swap_index = [rand_sequence[0], rand_sequence[1]]

                chrom[swap_index[0]] = chrom[swap_index[0]] + chrom[swap_index[1]]
                chrom[swap_index[1]] = chrom[swap_index[0]] - chrom[swap_index[1]]
                chrom[swap_index[0]] = chrom[swap_index[0]] - chrom[swap_index[1]]
            else:
                swap_index = rand_sequence[:3]
                for swap_i in itertools.permutations(swap_index):
                    a = chrom[swap_i[0]]
                    b = chrom[swap_i[1]]
                    c = chrom[swap_i[2]]
                    chrom[swap_i[0]] = b
                    chrom[swap_i[1]] = c
                    chrom[swap_i[2]] = a

        return np.array(chrom)

    def ga_search(self, solution_size, init_solution=[], eval_method='pairwise_info'):
        solution_best = None
        score_best = 0

        # solution initialization
        population = np.zeros((self.num_population, solution_size), dtype='int')
        for i in range(self.num_population):
            if len(init_solution) == 0:
                temp_solution = [j for j in range(solution_size)]
                random.shuffle(temp_solution)
                population[i] = np.array(temp_solution)
            else:
                if i % 10 == 0:
                    temp_solution = [j for j in range(solution_size)]
                    random.shuffle(temp_solution)
                    population[i] = np.array(temp_solution)
                else:
                    population[i] = init_solution
        # GA main body
        for ite in range(self.max_iters):
            # child population initialization
            population_new = np.zeros((self.num_population * 2, solution_size), dtype='int')
            population_new[self.num_population:] = population
            # crossover and mutation operator
            idx_sequence = [i for i in range(self.num_population)]
            random.shuffle(idx_sequence)
            for i in range(0, self.num_population, 2):
                parents = [population[i], population[i + 1]]
                children = list(self.crossover(*parents))
                for child_idx in range(len(children)):
                    children[child_idx] = self.mutation(children[child_idx])
                population_new[i] = children[0]
                population_new[i + 1] = children[1]
            # fitness evaluation with premature prevention
            score_variety = []
            population_scores = np.array([self.evaluate(list(population_new[i]), eval_method)[0] for i in range(self.num_population * 2)])
            for i in range(self.num_population * 2):
                if population_scores[i] in score_variety:
                    population_scores[i] *= self.variety_decay
                score_variety.append(population_scores[i])
            argorder = population_scores.argsort()
            ranking = argorder.argsort()  # ascent order
            for i in range(self.num_population * 2):
                if ranking[i] >= self.num_population:
                    population[ranking[i] - self.num_population] = population_new[i]
                if ranking[i] == self.num_population * 2 - 1 and population_scores[i] > score_best:
                    solution_best = population_new[i]
                    score_best = population_scores[i]
        return solution_best, score_best


def solving_task(eval, solution_size, query_idx_list, result, queue, lock, eval_method='pairwise_info',
                 num_population=64, crossover_rate=0.8, mutation_rate=0.2, variety_decay=0.8, max_iters=15):
    queue.put("")
    for query_idx in query_idx_list:
        solver = GeneticAlgorithm(eval, num_population=num_population, crossover_rate=crossover_rate,
                                  mutation_rate=mutation_rate, variety_decay=variety_decay, max_iters=max_iters)
        eval.set_index(query_idx)
        best_solution, _ = solver.ga_search(solution_size=solution_size)

        score, other = solver.evaluate(best_solution, eval_method)
        print('Puzzle {} Solution {} Score {} Correct {} Other {}'.format(
            query_idx, best_solution, score, other[0] == 1, other))

        lock.acquire()
        for i in range(len(result)):
            result[i] += other[i]
        # if len(other) == 5:
        #     n_33 += other[4]
        lock.release()
    queue.get()


def solving_task_single(eval, solution_size, query_idx_list, result, num_population=64, crossover_rate=0.8,
                        mutation_rate=0.2, variety_decay=0.9, max_iters=10):
    solution_lists = []
    for query_idx in query_idx_list:
        solver = GeneticAlgorithm(eval, num_population=num_population, crossover_rate=crossover_rate,
                                  mutation_rate=mutation_rate, variety_decay=variety_decay, max_iters=20)
        puzzlelet_solver = GeneticAlgorithm(eval, num_population=num_population, crossover_rate=crossover_rate,
                                            mutation_rate=mutation_rate, variety_decay=variety_decay, max_iters=max_iters)

        eval.set_index(query_idx)
        best_solution, _ = solver.ga_search(solution_size)
        best_solution, _ = puzzlelet_solver.ga_search(solution_size, init_solution=best_solution, eval_method='puzzleLet_info')
        # best_solution, _ = puzzlelet_solver.ga_search(solution_size, eval_method='puzzleLet_info')
        solution_lists.append(best_solution)
        score, other = solver.evaluate(best_solution)
        # score, other = puzzlelet_solver.evaluate(best_solution, eval_method='puzzleLet_info')
        print('Puzzle {} Solution {} Score {} Correct {} Other {}'.format(
            query_idx, best_solution, score, other[0] == 1, other))
        for i in range(len(result)):
            result[i] += other[i]
    return solution_lists


if __name__ == "__main__":
    best_list = []
    # data_img = np.load('MET_Dataset/select_image/test_img_12gap_55.npy')[:667]
    # data_img = np.load('MET_Dataset/select_image/test_img_12gap_55.npy')[667:1334]
    # data_img = np.load('MET_Dataset/select_image/test_img_12gap_55.npy')[1334:]

    # data_hori = np.load('test_hori_score.npy')[:667]
    # data_hori = np.load('test_hori_score.npy')[667:1334]
    # data_hori = np.load('test_hori_score.npy')[1334:]
    # data_hori = np.load('test_hori_score.npy')
    # data_hori = np.load('test_hori_score_eff_9_ft.npy')[:667]
    # data_hori = np.load('test_hori_score_eff_9_ft.npy')[667:1334]
    # data_hori = np.load('test_hori_score_eff_9_ft.npy')[1334:]
    # data_hori = np.load('test_hori_score_eff_9_ft.npy')
    # data_hori = np.load('test_hori_score_5_vgg_12gap.npy')[:667]
    # data_hori = np.load('test_hori_score_5_vgg_12gap.npy')[667:1334]
    # data_hori = np.load('test_hori_score_5_vgg_12gap.npy')[1334:]
    data_hori = np.load('test_hori_score_5_vgg_12gap.npy')
    # data_hori = np.load('test_hori_score_5_eff_25class_12gap.npy')[:667]
    # data_hori = np.load('test_hori_score_5_eff_25class_12gap.npy')[667:1334]
    # data_hori = np.load('test_hori_score_5_eff_25class_12gap.npy')[1334:]
    # data_hori = np.load('test_hori_score_5_eff_25class_12gap.npy')
    # data_hori = np.load('valid_hori_score_5_eff_25class_12gap.npy')
    # data_hori = np.ones(data_hori.shape)

    # data_vert = np.load('test_vert_score.npy')[:667]
    # data_vert = np.load('test_vert_score.npy')[667:1334]
    # data_vert = np.load('test_vert_score.npy')[1334:]
    # data_vert = np.load('test_vert_score.npy')
    # data_vert = np.load('test_vert_score_eff_9_ft.npy')[:667]
    # data_vert = np.load('test_vert_score_eff_9_ft.npy')[667:1334]
    # data_vert = np.load('test_vert_score_eff_9_ft.npy')[1334:]
    # data_vert = np.load('test_vert_score_eff_9_ft.npy')
    # data_vert = np.load('test_vert_score_5_vgg_12gap.npy')[:667]
    # data_vert = np.load('test_vert_score_5_vgg_12gap.npy')[667:1334]
    # data_vert = np.load('test_vert_score_5_vgg_12gap.npy')[1334:]
    data_vert = np.load('test_vert_score_5_vgg_12gap.npy')
    # data_vert = np.load('test_vert_score_5_eff_25class_12gap.npy')[:667]
    # data_vert = np.load('test_vert_score_5_eff_25class_12gap.npy')[667:1334]
    # data_vert = np.load('test_vert_score_5_eff_25class_12gap.npy')[1334:]
    # data_vert = np.load('test_vert_score_5_eff_25class_12gap.npy')
    # data_vert = np.load('valid_vert_score_5_eff_25class_12gap.npy')
    # data_vert = np.ones(data_vert.shape)

    # data_cate = np.load('test_cate_score.npy')[:667]
    # data_cate = np.load('test_cate_score.npy')[667:1334]
    # data_cate = np.load('test_cate_score.npy')[1334:]
    # data_cate = np.load('test_cate_score.npy')
    # data_cate = np.load('test_cate_score_eff_9_ft.npy')[:667]
    # data_cate = np.load('test_cate_score_eff_9_ft.npy')[667:1334]
    # data_cate = np.load('test_cate_score_eff_9_ft.npy')[1334:]
    # data_cate = np.load('test_cate_score_eff_9_ft.npy')
    data_cate = np.load('test_cate_score_5_vgg_12gap.npy')
    # data_cate = np.load('test_cate_score_5_vgg_12gap.npy')[:667]
    # data_cate = np.load('test_cate_score_5_vgg_12gap.npy')[667:1334]
    # data_cate = np.load('test_cate_score_5_vgg_12gap.npy')[1334:]
    # data_cate = np.load('test_cate_score_5_eff_2class_12gap.npy')
    # data_cate = np.load('valid_cate_score_5_eff_25class_12gap.npy')
    # data_cate = np.load('test_cate_score_5_eff_2class_12gap.npy')[:667]
    # data_cate = np.load('test_cate_score_5_eff_2class_12gap.npy')[667:1334]
    # data_cate = np.load('test_cate_score_5_eff_2class_12gap.npy')[1334:]

    # data_y = np.load('MET_Dataset/select_image/test_label_no_gap.npy')[:667]
    # data_y = np.load('MET_Dataset/select_image/test_label_no_gap.npy')[667:1334]
    # data_y = np.load('MET_Dataset/select_image/test_label_no_gap.npy')[1334:]
    # data_y = np.load('MET_Dataset/select_image/test_label_no_gap.npy')
    # data_y = np.load('MET_Dataset/select_image/test_label_no_gap_55.npy')[:667]
    # data_y = np.load('MET_Dataset/select_image/test_label_no_gap_55.npy')[667:1334]
    # data_y = np.load('MET_Dataset/select_image/test_label_no_gap_55.npy')[1334:]
    data_y = np.load('MET_Dataset/select_image/test_label_no_gap_55.npy')
    # data_y = np.load('MET_Dataset/select_image/valid_label_no_gap_55.npy')

    # data_puzzlelet = np.load('test_22_score_3_vgg16_12gap.npy')[:667]
    # data_puzzlelet = np.load('test_22_score_3_vgg16_12gap.npy')[667:1334]
    # data_puzzlelet = np.load('test_22_score_3_vgg16_12gap.npy')[1334:]
    # data_puzzlelet = np.load('test_22_score_3_vgg16_12gap.npy')
    # data_puzzlelet = np.load('test_22_score_5_vgg16_12gap.npy')[:667]
    # data_puzzlelet = np.load('test_22_score_5_vgg16_12gap.npy')[667:1334]
    # data_puzzlelet = np.load('test_22_score_5_vgg16_12gap.npy')[1334:]
    data_puzzlelet = np.load('test_22_score_5_vgg16_12gap.npy')

    assert len(data_cate) == len(data_y)
    puzzle_size = 5

    start_time = time.time()

    # eval = Evaluator(data_y, puzzle_size, data_cate, data_hori, data_vert, data_puzzleLet=None)
    eval = Evaluator(data_y, puzzle_size, data_cate, data_hori, data_vert, data_puzzlelet)
    total_puzzle_amount = len(data_cate)  # puzzle number in testing set

    correct_amount = 0
    solution_length = len(data_y[0][0])

    '''
    process_list = []
    result_list = multiprocessing.Manager().list([0, 0, 0, 0])  # [prefect_n, cate_n, hori_n, vert_n]
    lock = multiprocessing.Manager().Lock()
    queue = multiprocessing.Queue(32)
    for i in range(0, total_puzzle_amount, 4):
        indices_query = [i + j for j in range(4)]
        process = multiprocessing.Process(target=solving_task, args=(eval, solution_length,
                                                                     indices_query, result_list, queue, lock))
        process_list.append(process)
    job_count = 0
    for process in process_list:
        process.start()
        job_count += 1
        time.sleep(0.1)
    for process in process_list:
        process.join()
    '''

    result_list = [0, 0, 0, 0]
    indices_query = [i for i in range(total_puzzle_amount)]
    # indices_query = [i for i in range(10)]
    solution_lists = solving_task_single(eval, solution_length, indices_query, result_list, num_population=256,
                                         crossover_rate=0.8, mutation_rate=0.2, variety_decay=0.9, max_iters=40)
    solve_time = time.time() - start_time
    solution_matrix = np.array(solution_lists, dtype="int")
    print(solution_matrix.shape)
    # np.save("GA_sequence_5_eff_valid.npy", solution_matrix)
    np.save("PDN_GA_sequence_5_vgg_2.npy", solution_matrix)
    print('prefect\t', result_list[0], 1.0 * result_list[0] / len(data_y))
    if puzzle_size == 5:
        print('cate\t', result_list[1], 1.0 * result_list[1] / len(data_y) / 24)
        print('hori\t', result_list[2], 1.0 * result_list[2] / len(data_y) / 20)
        print('vert\t', result_list[3], 1.0 * result_list[3] / len(data_y) / 20)
    elif puzzle_size == 3:
        print('cate\t', result_list[1], 1.0 * result_list[1] / len(data_y) / 8)
        print('hori\t', result_list[2], 1.0 * result_list[2] / len(data_y) / 6)
        print('vert\t', result_list[3], 1.0 * result_list[3] / len(data_y) / 6)

    print('solve time\t%.3f' % solve_time)
    # print('3_3n\t', n_33)
