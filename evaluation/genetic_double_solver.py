import numpy as np
import itertools
import os
import random, time
import numpy as np
import multiprocessing
from evaluation.greedy import load_score_matrix,get_local_accuracy,get_score
import copy

MIN_MUTATION=0.2
MUTATION_GAMMA=0.9

class GeneticAlgorithm:
    def __init__(self, hori_score_matrix,vert_score_matrix, num_population=64, crossover_rate=0.8, mutation_rate=0.2, variety_decay=0.8, max_iters=20,decay=False):
        self.num_population = num_population
        self.crossover_rate = crossover_rate
        self.init_mutation_rate = mutation_rate
        self.variety_decay = variety_decay
        self.max_iters = max_iters
        self.hori_score_matrix=hori_score_matrix
        self.vert_score_matrix=vert_score_matrix
        self.data_length=hori_score_matrix.shape[0]


    def evaluate(self, solution,img_id):#rw
        solution_=copy.deepcopy(solution)
        solution_1=solution_[0:len(solution_)//2]
        solution_1.insert(4,4)
        solution_2=solution_[len(solution_)//2:]
        solution_2.insert(4,13)

        score=get_score(img_id,solution_1,self.hori_score_matrix,self.vert_score_matrix)+get_score(img_id,solution_2,self.hori_score_matrix,self.vert_score_matrix)
        return score

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
            
            #     for i in range(len(selected_indices)):
            #         swap_index = [selected_indices[i], related_indices[i]]
            #         if swap_index[0] != swap_index[1]:
            #             chrom1[swap_index[0]] = chrom1[swap_index[0]] + chrom1[swap_index[1]]
            #             chrom1[swap_index[1]] = chrom1[swap_index[0]] - chrom1[swap_index[1]]
            #             chrom1[swap_index[0]] = chrom1[swap_index[0]] - chrom1[swap_index[1]]
            
            #             chrom2[swap_index[0]] = chrom2[swap_index[0]] + chrom2[swap_index[1]]
            #             chrom2[swap_index[1]] = chrom2[swap_index[0]] - chrom2[swap_index[1]]
            #             chrom2[swap_index[0]] = chrom2[swap_index[0]] - chrom2[swap_index[1]]
            
            
            # for i in range(len(chrom1)):
            #     if (i not in chrom1) or (i not in chrom2):
            #         print("after:", chrom1, chrom2, selected_indices, selected_numbers, related_indices)
        
        
        
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

    def ga_search(self, img_id,solution_size, init_solution=[]):
        solution_best = None
        score_best = 0
        self.mutation_rate=self.init_mutation_rate
        # solution initialization
        population = np.zeros((self.num_population, solution_size), dtype='int')
        for i in range(self.num_population):
            if len(init_solution) == 0:
                # temp_solution = [j for j in range(solution_size)]
                temp_solution1 = [j if j<4 else j+1 for j in range(solution_size//2)]
                temp_solution2=[j+9 if j<4 else j+10 for j in range(solution_size//2)]
                temp_solution=temp_solution1+temp_solution2
                random.shuffle(temp_solution)
                population[i] = np.array(temp_solution)
            else:
                if i % 10 == 0:
                    # temp_solution = [j for j in range(solution_size)]
                    temp_solution1 = [j if j<4 else j+1 for j in range(solution_size//2)]
                    temp_solution2=[j+9 if j<4 else j+10 for j in range(solution_size//2)]
                    temp_solution=temp_solution1+temp_solution2
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
            # print(f"Population new type: {type(population_new)}, element: {population_new[0]}")
            population_scores = np.array([self.evaluate(list(population_new[i]),img_id) for i in range(self.num_population * 2)])

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
            if self.mutation_rate>MIN_MUTATION:
                self.mutation_rate*=MUTATION_GAMMA
        return solution_best, score_best





def test(ga_solver:GeneticAlgorithm):
    true_result=list(range(0,18))
    true_result.pop(13)
    true_result.pop(4)
    greedy_right_rate=[]
    right=[]
    hori_acc=[]
    vert_acc=[]
    cate_acc=[]
    perm=[]

    print(true_result)

    for img_idx in range(ga_solver.data_length):
        print(f"\rSearching index: {img_idx}, current avg acc:{np.mean(right) if len(right)>0 else 0 :.4f}, greedy better rate: {np.mean(greedy_right_rate) if len(right)>0 else 0 :.4f}",end="")
        best_perm,greedy_score=ga_solver.ga_search(img_idx,16)
        best_perm=list(best_perm)
        perm.append(best_perm)
        
        
        true_score=ga_solver.evaluate(true_result,img_idx)

        if best_perm==true_result:#acc
            right.append(1)
        else:
            right.append(0)

        if greedy_score>=true_score:#greedy better rate
            greedy_right_rate.append(1)
        else:
            greedy_right_rate.append(0)

        hori_right,vert_right,cate_right=get_local_accuracy(best_perm)
        hori_acc.append(hori_right)
        vert_acc.append(vert_right)
        cate_acc.append(cate_right)
    print(f"accuracy: {np.mean(right):.4f}, greedy better rate: {np.mean(greedy_right_rate):.4f}")
    
    return np.mean(right),np.mean(hori_acc), np.mean(vert_acc), np.mean(cate_acc),right,perm

if __name__=="__main__":
    
    hori_score_matrix,vert_score_matrix=load_score_matrix()
    ga_solver=GeneticAlgorithm(hori_score_matrix,
                               vert_score_matrix,
                               crossover_rate=0.9,
                               max_iters=60,
                               num_population=256,
                               mutation_rate=0.2)
    right,hori,vert,cate,right_record,perm=test(ga_solver)