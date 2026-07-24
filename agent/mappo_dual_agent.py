import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class MAPPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    mini_batch_size: int = 16
    learning_rate: float = 3e-4
    intent_align_coef: float = 0.1
    epsilon_greedy: float = 0.1


class EpisodeBuffer:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.board_images: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []
        self.outside_probs: List[torch.Tensor] = []


class DualBoardMAPPOAgent:
    def __init__(
        self,
        model: nn.Module,
        env,
        config: MAPPOConfig,
        model_agent2: Optional[nn.Module] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.models = nn.ModuleList([model.to(device), (model_agent2 or copy.deepcopy(model)).to(device)])
        self.model = self.models[0]
        self.model_agent1 = self.models[0]
        self.model_agent2 = self.models[1]
        self.env = env
        self.config = config
        self.device = device
        self.optimizers = [
            torch.optim.Adam(self.models[0].parameters(), lr=config.learning_rate, eps=1e-8),
            torch.optim.Adam(self.models[1].parameters(), lr=config.learning_rate, eps=1e-8),
        ]
        self.optimizer = self.optimizers[0]
        self.buffer = EpisodeBuffer()

    def _stack_obs(self, observations: Sequence[Dict[str, torch.Tensor]]) -> torch.Tensor:
        return torch.stack([obs["board_image"] for obs in observations], dim=0).unsqueeze(0).to(self.device)

    def _sample_policy(
        self,
        board_images: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        outputs = [model.evaluate_policy(board_images) for model in self.models]
        ptr1_actions = []
        ptr1_log_probs = []
        ptr1_entropies = []

        for agent_index, output in enumerate(outputs):
            logits = output["ptr1_logits"][:, agent_index]
            dist = Categorical(logits=logits)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()
            ptr1_actions.append(action)
            ptr1_log_probs.append(dist.log_prob(action))
            ptr1_entropies.append(dist.entropy())

        ptr1_actions_t = torch.stack(ptr1_actions, dim=1)
        ptr2_actions = []
        ptr2_log_probs = []
        ptr2_entropies = []
        outside_probs = []

        for agent_index, (model, output) in enumerate(zip(self.models, outputs)):
            full_output = model.evaluate_policy(
                board_images=None,
                ptr1_actions=ptr1_actions_t,
                encoded=output["encoded"],
            )
            logits = full_output["ptr2_logits"][:, agent_index]
            dist = Categorical(logits=logits)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()
            ptr2_actions.append(action)
            ptr2_log_probs.append(dist.log_prob(action))
            ptr2_entropies.append(dist.entropy())
            outside_probs.append(full_output["outside_prob"][:, agent_index])

        return {
            "ptr1_actions": ptr1_actions_t,
            "ptr2_actions": torch.stack(ptr2_actions, dim=1),
            "log_prob": torch.stack(ptr1_log_probs, dim=1) + torch.stack(ptr2_log_probs, dim=1),
            "entropy": torch.stack(ptr1_entropies, dim=1) + torch.stack(ptr2_entropies, dim=1),
            "value": torch.stack([output["value"] for output in outputs], dim=1),
            "outside_prob": torch.stack(outside_probs, dim=1),
        }

    def select_actions(
        self,
        observations: Sequence[Dict[str, torch.Tensor]],
        deterministic: bool = False,
    ) -> Tuple[List[Tuple[int, int]], Dict[str, torch.Tensor]]:
        for model in self.models:
            model.eval()
        board_images = self._stack_obs(observations)
        with torch.no_grad():
            policy_sample = self._sample_policy(board_images, deterministic=deterministic)

        ptr1 = policy_sample["ptr1_actions"].squeeze(0).cpu()
        ptr2 = policy_sample["ptr2_actions"].squeeze(0).cpu()
        actions = [(int(ptr1[0].item()), int(ptr2[0].item())), (int(ptr1[1].item()), int(ptr2[1].item()))]
        return actions, {
            "board_images": board_images.squeeze(0).detach().cpu().to(torch.uint8),
            "actions": torch.stack([ptr1, ptr2], dim=-1).to(torch.long),
            "log_prob": policy_sample["log_prob"].squeeze(0).detach().cpu(),
            "value": policy_sample["value"].squeeze(0).detach().cpu(),
            "outside_prob": policy_sample["outside_prob"].squeeze(0).detach().cpu(),
        }

    def _build_mixed_policy(self, logits: torch.Tensor, epsilon: float) -> torch.Tensor:
        policy_probs = torch.softmax(logits, dim=-1)
        uniform = torch.full_like(policy_probs, 1.0 / policy_probs.numel())
        mixed_probs = (1.0 - epsilon) * policy_probs + epsilon * uniform
        return mixed_probs / mixed_probs.sum()

    def record_transition(
        self,
        policy_info: Dict[str, torch.Tensor],
        reward: float,
        done: bool,
    ) -> None:
        self.buffer.board_images.append(policy_info["board_images"])
        self.buffer.actions.append(policy_info["actions"])
        self.buffer.log_probs.append(policy_info["log_prob"])
        self.buffer.values.append(policy_info["value"])
        self.buffer.outside_probs.append(policy_info["outside_prob"])
        self.buffer.rewards.append(torch.tensor(reward, dtype=torch.float32))
        self.buffer.dones.append(torch.tensor(float(done), dtype=torch.float32))

    def _compute_bootstrap_value(self, observations: Sequence[Dict[str, torch.Tensor]], done: bool) -> torch.Tensor:
        if done:
            return torch.zeros(2, dtype=torch.float32)
        for model in self.models:
            model.eval()
        board_images = self._stack_obs(observations)
        with torch.no_grad():
            values = [model.evaluate_policy(board_images)["value"].squeeze(0).detach().cpu() for model in self.models]
        return torch.stack(values).to(torch.float32)

    def _compute_advantages(self, last_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = torch.stack(self.buffer.rewards)
        dones = torch.stack(self.buffer.dones)
        values = torch.stack(self.buffer.values)
        outside_probs = torch.stack(self.buffer.outside_probs)

        intent_bonus = self.config.intent_align_coef * (1.0 - torch.abs(outside_probs[:, 0] - outside_probs[:, 1]))
        rewards = rewards + intent_bonus

        rewards = rewards.unsqueeze(-1).expand_as(values)
        returns = torch.zeros_like(values)
        advantages = torch.zeros_like(values)
        gae = torch.zeros(2, dtype=torch.float32)
        next_value = last_value.to(torch.float32)

        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step].view(1)
            delta = rewards[step] + self.config.gamma * next_value * mask - values[step]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages[step] = gae
            returns[step] = advantages[step] + values[step]
            next_value = values[step]
        std = advantages.std(dim=0, unbiased=False)
        mean = advantages.mean(dim=0)
        advantages = torch.where(std > 1e-8, (advantages - mean) / (std + 1e-8), advantages - mean)
        return returns, advantages

    def update(self, next_observations: Sequence[Dict[str, torch.Tensor]], done: bool, show: bool = False) -> Dict[str, float]:
        if not self.buffer.rewards:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        for model in self.models:
            model.train()
        last_value = self._compute_bootstrap_value(next_observations, done)
        returns, advantages = self._compute_advantages(last_value)

        board_images = torch.stack(self.buffer.board_images).to(self.device)
        actions = torch.stack(self.buffer.actions).to(self.device)
        old_log_probs = torch.stack(self.buffer.log_probs).to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        indices = np.arange(len(self.buffer.rewards))
        loss_log: List[Tuple[int, float, float, float]] = []

        for _ in range(self.config.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.config.mini_batch_size):
                batch_indices = indices[start : start + self.config.mini_batch_size]
                batch_indices_t = torch.tensor(batch_indices, dtype=torch.long, device=self.device)

                batch_board = board_images[batch_indices_t]
                batch_actions = actions[batch_indices_t]
                batch_ptr1 = batch_actions[:, :, 0]
                batch_ptr2 = batch_actions[:, :, 1]

                for agent_index, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
                    output = model.evaluate_policy(
                        batch_board,
                        ptr1_actions=batch_ptr1,
                    )
                    ptr1_dist = Categorical(logits=output["ptr1_logits"][:, agent_index])
                    ptr1_log_prob = ptr1_dist.log_prob(batch_ptr1[:, agent_index])
                    ptr1_entropy = ptr1_dist.entropy()

                    ptr2_dist = Categorical(logits=output["ptr2_logits"][:, agent_index])
                    ptr2_log_prob = ptr2_dist.log_prob(batch_ptr2[:, agent_index])
                    ptr2_entropy = ptr2_dist.entropy()

                    new_log_prob = ptr1_log_prob + ptr2_log_prob
                    entropy = (ptr1_entropy + ptr2_entropy).mean()
                    value = output["value"]

                    ratio = torch.exp(new_log_prob - old_log_probs[batch_indices_t, agent_index])
                    agent_advantages = advantages[batch_indices_t, agent_index]
                    unclipped = ratio * agent_advantages
                    clipped = torch.clamp(
                        ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    ) * agent_advantages

                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = nn.functional.mse_loss(value, returns[batch_indices_t, agent_index])
                    loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                    optimizer.step()

                    loss_log.append((agent_index, float(policy_loss.item()), float(value_loss.item()), float(entropy.item())))

        mean_policy = float(np.mean([item[1] for item in loss_log])) if loss_log else 0.0
        mean_value = float(np.mean([item[2] for item in loss_log])) if loss_log else 0.0
        mean_entropy = float(np.mean([item[3] for item in loss_log])) if loss_log else 0.0
        agent_policy_losses = [
            float(np.mean([item[1] for item in loss_log if item[0] == agent_index]))
            if any(item[0] == agent_index for item in loss_log)
            else 0.0
            for agent_index in range(2)
        ]
        self.buffer.clear()

        if show:
            print(
                f"MAPPO update - policy_loss: {mean_policy:.4f}, "
                f"value_loss: {mean_value:.4f}, entropy: {mean_entropy:.4f}, "
                f"agent1_policy_loss: {agent_policy_losses[0]:.4f}, "
                f"agent2_policy_loss: {agent_policy_losses[1]:.4f}"
            )

        return {
            "policy_loss": mean_policy,
            "value_loss": mean_value,
            "entropy": mean_entropy,
            "agent1_policy_loss": agent_policy_losses[0],
            "agent2_policy_loss": agent_policy_losses[1],
        }
