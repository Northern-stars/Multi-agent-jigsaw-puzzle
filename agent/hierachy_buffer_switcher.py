import copy
import random

import numpy as np
import torch
import torch.nn as nn

from utils.hierachy_config import ACTOR_LR, CLIP_GRAD_NORM, DEVICE


class Buffer_switcher:
    def __init__(self, memory_size, model, env, action_num, batch_size, train_epoch, tau=1e-3):
        self.model = model
        self.main_model = copy.deepcopy(model)
        self.optimizer = torch.optim.Adam(self.main_model.parameters(), lr=ACTOR_LR, eps=1e-8)
        self.schedular = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=self.optimizer,
            T_0=10,
            T_mult=1,
            eta_min=1e-6,
            last_epoch=-1,
        )
        self.memory_size = memory_size
        self.memory = []
        self.memory_counter = 0
        self.trace_start_point = 0
        self.batch_size = batch_size
        self.env = env
        self.action_num = action_num
        self.epochs = train_epoch
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

    def act(self, image, outsider, permutation):
        self.model.eval()
        mask = self.env.get_mask(permutation)
        with torch.no_grad():
            action = torch.argmax(self.model(image, outsider, [mask]), dim=-1)
            action = self.epsilon_greedy(action=action.item())
            swap_index, outsider_index = permutation[action], permutation[-1]
            permutation[action], permutation[-1] = outsider_index, swap_index
        return permutation, action

    def update(self, show=False):
        order = list(range(len(self.memory)))
        random.shuffle(order)
        self.main_model.train()
        self.model.train()
        actor_loss_sum = []
        for i in range(self.epochs):
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
            done = []
            mask = []
            next_mask = []

            for sample in sample_dicts:
                current_image, current_outsider = self.env.request_for_image(
                    image_id=sample["Image_id"],
                    permutation=sample["State"],
                    image_index=sample["Image_index"],
                )
                states.append(current_image)
                mask.append(self.env.get_mask(sample["State"]))
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
                done.append(sample["Done"])

            state_tensor = torch.cat(states, dim=0)
            if state_tensor.size(0) == 1:
                self.model.eval()
                self.main_model.eval()
            else:
                self.model.train()
                self.main_model.train()

            outsider_tensor = torch.cat(outsider_pieces, dim=0)
            next_state_tensor = torch.cat(next_states, dim=0)
            next_outsiders_tensor = torch.cat(next_outsiders, dim=0)
            action_tensor = torch.tensor(actions).to(DEVICE).unsqueeze(-1)
            reward_tensor = torch.tensor(reward, dtype=torch.float32).to(DEVICE).unsqueeze(-1)
            done_tensor = torch.tensor(done, dtype=torch.float32).to(DEVICE).unsqueeze(-1)
            q_main = self.main_model(state_tensor, outsider_tensor, mask).gather(1, action_tensor)
            q_next = self.model(next_state_tensor, next_outsiders_tensor, next_mask).max(1)[0].unsqueeze(-1).detach()
            q_target = reward_tensor + self.env.gamma * q_next * (1 - done_tensor)
            loss = nn.MSELoss()(q_main, q_target)
            actor_loss_sum.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.main_model.parameters(), CLIP_GRAD_NORM)
            self.optimizer.step()
            self.schedular.step()

            for target_param, main_param in zip(self.model.parameters(), self.main_model.parameters()):
                target_param.data.copy_(self.tau * main_param.data + (1 - self.tau) * target_param.data)

        if show and actor_loss_sum:
            print(f"Buffer switcher loss: {np.mean(actor_loss_sum)}")
