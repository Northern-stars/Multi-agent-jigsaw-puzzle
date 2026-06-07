import copy
import itertools
import random

import numpy as np
import torch
import torch.nn as nn

from utils.hierachy_config import ACTOR_LR, ACTOR_SCHEDULAR_STEP, DEVICE


class Local_switcher:
    def __init__(self, model, memory_size, gamma, batch_size, env, action_num, tau=1e-3):
        self.model = model
        self.main_model = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.main_model.parameters(), lr=ACTOR_LR, eps=1e-8)
        self.schedular = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=ACTOR_SCHEDULAR_STEP)
        self.memory_size = memory_size
        self.memory = []
        self.memory_counter = 0
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = env
        self.action_num = action_num
        self.tau = tau

    def epsilon_greedy(self, action):
        if random.random() > self.env.epsilon:
            return action
        return (action + random.randint(1, self.action_num)) % self.action_num

    def recording_memory(self, image_id, image_index, state, action, reward, next_state, done):
        memory = {
            "Image_id": image_id,
            "Image_index": image_index,
            "State": state,
            "Action": action,
            "Reward": reward,
            "Next_state": next_state,
            "Done": done,
        }
        if len(self.memory) < self.memory_size:
            self.memory.append(memory)
        else:
            self.memory[self.memory_counter] = memory
        self.memory_counter = (self.memory_counter + 1) % self.memory_size

    def clean_memory(self):
        self.memory = []
        self.memory_counter = 0

    def permute(self, cur_permutation, action_index):
        new_permutation = copy.deepcopy(cur_permutation)
        action = list(itertools.combinations(list(range(len(cur_permutation))), 2))[action_index]
        value0 = cur_permutation[action[0]]
        value1 = cur_permutation[action[1]]
        new_permutation[action[0]] = value1
        new_permutation[action[1]] = value0
        return new_permutation

    def choose_action(self, permutation, image_index):
        value_list = []
        image_list = []
        mask_list = []

        for i in range(self.action_num):
            perm_ = self.permute(permutation, i)
            mask = self.env.get_mask(perm_)
            image, _ = self.env.get_image(perm_, image_index=image_index)
            image_list.append(copy.deepcopy(image.cpu()))
            mask_list.append(mask)

        i = 0
        with torch.no_grad():
            while i < self.action_num:
                if self.action_num - i < self.batch_size:
                    image = torch.cat(image_list[i:], dim=0).to(DEVICE)
                    mask = mask_list[i:]
                else:
                    image = torch.cat(image_list[i : i + self.batch_size], dim=0).to(DEVICE)
                    mask = mask_list[i : i + self.batch_size]
                value = self.model(image, mask)
                value_list.append(value.squeeze(-1).to("cpu"))
                i += self.batch_size
            value_list = torch.cat(value_list)
            best_action = torch.argmax(value_list).item()
        return int(best_action)

    def update(self, show=False):
        self.model.train()
        self.main_model.train()
        order = list(range(len(self.memory)))
        random.shuffle(order)
        loss_sum = []
        for i in range(self.env.epochs):
            if i * self.batch_size >= len(order):
                break
            if len(order) - i * self.batch_size < self.batch_size:
                sample_dicts = [self.memory[j] for j in order[i * self.batch_size :]]
            else:
                sample_dicts = [self.memory[j] for j in order[i * self.batch_size : (i + 1) * self.batch_size]]

            states = []
            outsider_pieces = []
            actions = []
            next_states = []
            next_outsiders = []
            reward = []
            current_mask = []
            next_mask = []

            for sample in sample_dicts:
                current_image, current_outsider = self.env.request_for_image(
                    image_id=sample["Image_id"],
                    permutation=sample["State"],
                    image_index=sample["Image_index"],
                )
                states.append(current_image)
                current_mask.append(self.env.get_mask(sample["State"]))
                outsider_pieces.append(current_outsider)
                actions.append(sample["Action"])
                next_image, next_outsider_piece = self.env.request_for_image(
                    image_id=sample["Image_id"],
                    permutation=sample["Next_state"],
                    image_index=sample["Image_index"],
                )
                next_states.append(next_image)
                next_mask.append(self.env.get_mask(sample["Next_state"]))
                next_outsiders.append(next_outsider_piece)
                reward.append(sample["Reward"])

            if len(states) == 1:
                self.model.eval()
                self.main_model.eval()
            else:
                self.model.train()
                self.main_model.train()
            state_tensor = torch.cat(states, dim=0)
            next_state_tensor = torch.cat(next_states, dim=0)
            q_next = self.model(next_state_tensor, next_mask).detach()
            q_eval = self.main_model(next_state_tensor, next_mask)
            reward_tensor = torch.tensor(reward, dtype=torch.float32).to(DEVICE).unsqueeze(-1)
            q_target = (reward_tensor + self.gamma * q_next).to(torch.float)
            loss = nn.MSELoss()(q_target, q_eval)
            self.optimizer.zero_grad()
            loss.float().backward()
            self.optimizer.step()
            loss_sum.append(loss.item())

        for target_param, main_param in zip(self.model.parameters(), self.main_model.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1 - self.tau) * target_param.data)
        self.schedular.step()
        if show and loss_sum:
            print(f"Local switcher loss: {np.mean(loss_sum)}")

    def act(self, permutation, image_index):
        self.model.eval()
        action = self.choose_action(permutation=permutation, image_index=image_index)
        action = self.epsilon_greedy(action)
        permutation_ = self.permute(cur_permutation=permutation, action_index=action)
        return permutation_, action
