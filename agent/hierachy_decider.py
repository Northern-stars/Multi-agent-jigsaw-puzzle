import random

import numpy as np
import torch
import torch.nn as nn

from utils.hierachy_config import (
    ACTOR_LR,
    CLIP_GRAD_NORM,
    CRITIC_LR,
    DEVICE,
    ENTROPY_GAMMA,
    ENTROPY_MIN,
)


class Decider:
    def __init__(
        self,
        memory_size,
        actor,
        critic,
        env,
        action_num,
        batch_size,
        entropy_weight,
        train_epoch,
        lmb=0.98,
    ):
        self.memory = []
        self.memory_size = memory_size
        self.memory_counter = 0
        self.trace_start_point = 0
        self.actor_model = actor
        self.critic_model = critic
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=ACTOR_LR, eps=1e-8)
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr=CRITIC_LR, eps=1e-8)
        self.actor_schedular = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=self.actor_optimizer,
            T_0=10,
            T_mult=1,
            eta_min=1e-6,
            last_epoch=-1,
        )
        self.critic_schedular = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=self.critic_optimizer,
            T_0=10,
            T_mult=1,
            eta_min=1e-5,
            last_epoch=-1,
        )
        self.action_num = action_num
        self.env = env
        self.batch_size = batch_size
        self.entropy_weight = entropy_weight
        self.epochs = train_epoch
        self.lmb = lmb

    def recording_memory(
        self,
        image_id,
        image_index,
        other_image_index,
        state,
        other_state,
        action,
        log_prob,
        reward,
        next_state,
        next_other_state,
        done,
    ):
        memory = {
            "Image_id": image_id,
            "Image_index": image_index,
            "Other_image_index": other_image_index,
            "Log_prob": log_prob,
            "State": state,
            "Other_state": other_state,
            "Action": action,
            "Reward": reward,
            "Next_state": next_state,
            "Next_other_state": next_other_state,
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
        self.trace_start_point = self.memory_counter

    def act(self, current_image, outsider_piece, mask=None):
        if mask is None:
            mask = []
        with torch.no_grad():
            self.actor_model.eval()
            prob = nn.Softmax(dim=-1)(self.actor_model(current_image, outsider_piece, [mask]))
            dist = torch.distributions.Categorical(prob)
            action = dist.sample()
            action_log_prob = dist.log_prob(action).detach()
        return action, action_log_prob

    def update(self, show=False):
        eps = 0.2
        gae = [0 for _ in range(self.env.image_num)]
        r_start = [True for _ in range(self.env.image_num)]
        self.critic_model.eval()
        i = (self.memory_counter - 1) % len(self.memory)
        while i != (self.trace_start_point - 1) % len(self.memory) or all(r_start):
            image_index = self.memory[i]["Image_index"]
            done = self.memory[i]["Done"]
            reward = self.memory[i]["Reward"]
            if r_start[image_index]:
                r_start[image_index] = False

            current_image, current_outsider = self.env.request_for_image(
                image_id=self.memory[i]["Image_id"],
                permutation=self.memory[i]["State"],
                image_index=self.memory[i]["Image_index"],
            )
            next_image, next_outsider = self.env.request_for_image(
                image_id=self.memory[i]["Image_id"],
                permutation=self.memory[i]["Next_state"],
                image_index=self.memory[i]["Image_index"],
            )
            cur_mask = self.env.get_mask(self.memory[i]["State"])
            next_mask = self.env.get_mask(self.memory[i]["Next_state"])
            with torch.no_grad():
                cur_value = self.critic_model(current_image, current_outsider, [cur_mask]).item()
                next_value = self.critic_model(next_image, next_outsider, [next_mask]).item()

            delta = reward + self.env.gamma * next_value * (1 - done) - cur_value
            gae[image_index] = delta + self.env.gamma * self.lmb * (1 - done) * gae[image_index]
            self.memory[i]["Advantage"] = gae[image_index]
            i = (i - 1) % len(self.memory)

        for i in range(len(self.memory)):
            if "Advantage" not in self.memory[i]:
                print("Empty advantage: ", i)
                self.clean_memory()
                return

        order = list(range(len(self.memory)))
        random.shuffle(order)
        self.actor_model.train()
        self.critic_model.train()
        critic_loss_sum = []
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
            old_log_probs = []
            advantage = []
            cur_mask = []
            next_mask = []

            for sample in sample_dicts:
                current_image, current_outsider = self.env.request_for_image(
                    image_id=sample["Image_id"],
                    permutation=sample["State"],
                    image_index=sample["Image_index"],
                )
                states.append(current_image)
                outsider_pieces.append(current_outsider)
                cur_mask.append(self.env.get_mask(sample["State"]))
                next_mask.append(self.env.get_mask(sample["Next_state"]))
                old_log_probs.append(sample["Log_prob"])
                actions.append(sample["Action"])
                next_image, next_outsider = self.env.request_for_image(
                    image_id=sample["Image_id"],
                    permutation=sample["Next_state"],
                    image_index=sample["Image_index"],
                )
                next_states.append(next_image)
                next_outsiders.append(next_outsider)
                reward.append(sample["Reward"])
                done.append(sample["Done"])
                advantage.append(sample["Advantage"])

            state_tensor = torch.cat(states, dim=0)
            if state_tensor.size(0) == 1:
                self.critic_model.eval()
                self.actor_model.eval()
            else:
                self.critic_model.train()
                self.actor_model.train()

            outsider_tensor = torch.cat(outsider_pieces, dim=0)
            next_state_tensor = torch.cat(next_states, dim=0)
            next_outsiders_tensor = torch.cat(next_outsiders, dim=0)
            probs = nn.Softmax(dim=-1)(self.actor_model(state_tensor, outsider_tensor, cur_mask))

            action_tensor = torch.tensor(actions).to(DEVICE).unsqueeze(-1)
            advantage_tensor = torch.tensor(advantage).to(DEVICE).unsqueeze(-1)
            selected_probs = probs.gather(1, action_tensor).clamp(min=1e-8)
            log_probs = torch.log(selected_probs)
            old_log_prob_tensor = torch.cat(old_log_probs, dim=0)
            reward_tensor = torch.tensor(reward, dtype=torch.float32).to(DEVICE).unsqueeze(-1)
            done_tensor = torch.tensor(done, dtype=torch.float32).to(DEVICE).unsqueeze(-1)
            entropy = torch.distributions.Categorical(probs).entropy()

            ratio = torch.exp(log_probs - old_log_prob_tensor)
            actor_loss = -torch.min(
                ratio * advantage_tensor,
                torch.clamp(ratio, 1 - eps, 1 + eps) * advantage_tensor,
            ).mean() - entropy.mean() * self.entropy_weight
            actor_loss_sum.append(actor_loss.item())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), CLIP_GRAD_NORM)
            self.actor_optimizer.step()

            pred_ret = self.critic_model(state_tensor, outsider_tensor, cur_mask)
            bellman_ret = self.env.gamma * (1 - done_tensor) * self.critic_model(
                next_state_tensor,
                next_outsiders_tensor,
                next_mask,
            ) + reward_tensor
            critic_loss = nn.MSELoss()(pred_ret, bellman_ret)
            critic_loss_sum.append(critic_loss.item())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), CLIP_GRAD_NORM)
            self.critic_optimizer.step()

        if show and actor_loss_sum and critic_loss_sum:
            print(f"Decider\nCritic loss: {np.mean(critic_loss_sum)}. Actor_loss: {np.mean(actor_loss_sum)}")

        if self.entropy_weight >= ENTROPY_MIN:
            self.entropy_weight *= ENTROPY_GAMMA
        self.actor_schedular.step()
        self.critic_schedular.step()
