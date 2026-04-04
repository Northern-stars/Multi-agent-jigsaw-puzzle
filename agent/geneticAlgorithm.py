import numpy as np
import itertools
import os
import random, time
import numpy as np
import multiprocessing
import copy
import torch
from evaluation.greedy import get_score
MIN_MUTATION=0.2
MUTATION_GAMMA=0.9
BATCH_SIZE=128
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

class GeneticAlgorithm:
    def __init__(self, 
                 hori_model,vert_model, 
                    num_population=128,
                    crossover_rate=0.8,
                    mutation_rate=0.2,
                    variety_decay=0.8,
                    max_iters=30,
                    get_score=get_score,
                    decay=False):
        self.num_population = num_population
        self.crossover_rate = crossover_rate
        self.init_mutation_rate = mutation_rate
        self.variety_decay = variety_decay
        self.max_iters = max_iters
        self.hori_model=hori_model.to(DEVICE)
        self.vert_model=vert_model.to(DEVICE)
        self.get_score=get_score
        self.hori_score_matrix=None
        self.vert_score_matrix=None

    def summon_score_matrix(self,img_dict):
        hori_score_matrix=np.ones([1,18,18],dtype=float)
        vert_score_matrix=np.ones([1,18,18],dtype=float)
        # Collect all pairs
        pairs = []
        for i0 in range(18):
            for i1 in range(18):
                if i0 == i1:
                    continue
                pairs.append((i0, i1, img_dict[i0], img_dict[i1]))
        
        # Batch processing
        batch_size = 128
        for start in range(0, len(pairs), batch_size):
            if start+batch_size>=len(pairs):
                batch=pairs[start:]
            else:
                batch = pairs[start:start + batch_size]
            frag0_batch = torch.stack([p[2] for p in batch]).to(DEVICE)
            frag1_batch = torch.stack([p[3] for p in batch]).to(DEVICE)
            
            with torch.no_grad():
                hori_score0_batch = self.hori_model(frag0_batch, frag1_batch)
                hori_score1_batch = self.hori_model(frag1_batch, frag0_batch)
                
                vert_score0_batch = self.vert_model(frag0_batch, frag1_batch)
                vert_score1_batch = self.vert_model(frag1_batch, frag0_batch)
            
            for idx, (i0, i1, _, _) in enumerate(batch):
                hori_score_matrix[0][i0][i1] = float(hori_score0_batch[idx].detach())
                hori_score_matrix[0][i1][i0] = float(hori_score1_batch[idx].detach())
                
                vert_score_matrix[0][i0][i1] = float(vert_score0_batch[idx].detach())
                vert_score_matrix[0][i1][i0] = float(vert_score1_batch[idx].detach())
        self.hori_score_matrix=hori_score_matrix
        self.vert_score_matrix=vert_score_matrix
        return hori_score_matrix,vert_score_matrix

    def evaluate(self, solution):#rw
        solution_=copy.deepcopy(solution)
        solution_1=solution_[0:len(solution_)//2]
        solution_1.insert(4,4)
        solution_2=solution_[len(solution_)//2:]
        solution_2.insert(4,13)

        score=self.get_score(0,solution_1,self.hori_score_matrix,self.vert_score_matrix)+self.get_score(0,solution_2,self.hori_score_matrix,self.vert_score_matrix)
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

    def ga_search(self,solution_size,img_dict, init_solution=[]):
        self.summon_score_matrix(img_dict)
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
                    population[i] = np.array(init_solution)
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
            population_scores = np.array([self.evaluate(list(population_new[i])) for i in range(self.num_population * 2)])

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