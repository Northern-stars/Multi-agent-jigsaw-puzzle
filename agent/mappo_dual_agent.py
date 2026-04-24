from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

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
        self.slot_images: List[torch.Tensor] = []
        self.anchor_images: List[torch.Tensor] = []
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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model = model.to(device)
        self.env = env
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=1e-8)
        self.buffer = EpisodeBuffer()

    def _stack_obs(self, observations: Sequence[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        slot_images = torch.stack([obs["slot_images"] for obs in observations], dim=0).unsqueeze(0).to(self.device)
        anchor_images = torch.stack([obs["anchor_image"] for obs in observations], dim=0).unsqueeze(0).to(self.device)
        return slot_images, anchor_images

    def _sample_policy(
        self,
        output: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        ptr1_logits = output["ptr1_logits"]
        ptr1_dist = Categorical(logits=ptr1_logits)

        if deterministic:
            ptr1_actions = ptr1_logits.argmax(dim=-1)
        else:
            ptr1_actions = Categorical(self._build_mixed_policy(ptr1_logits, self.config.epsilon_greedy)).sample()

        full_output = self.model.evaluate_policy(
            slot_images=None,
            anchor_images=None,
            ptr1_actions=ptr1_actions,
            encoded=output["encoded"],
        )
        ptr2_logits = full_output["ptr2_logits"]
        ptr2_dist = Categorical(logits=ptr2_logits)

        if deterministic:
            ptr2_actions = ptr2_logits.argmax(dim=-1)
        else:
            ptr2_actions = Categorical(self._build_mixed_policy(ptr2_logits, self.config.epsilon_greedy)).sample()

        return {
            "ptr1_actions": ptr1_actions,
            "ptr2_actions": ptr2_actions,
            "log_prob": ptr1_dist.log_prob(ptr1_actions) + ptr2_dist.log_prob(ptr2_actions),
            "entropy": ptr1_dist.entropy() + ptr2_dist.entropy(),
            "value": output["value"],
            "outside_prob": full_output["outside_prob"],
        }

    def select_actions(
        self,
        observations: Sequence[Dict[str, torch.Tensor]],
        deterministic: bool = False,
    ) -> Tuple[List[Tuple[int, int]], Dict[str, torch.Tensor]]:
        self.model.eval()
        slot_images, anchor_images = self._stack_obs(observations)
        with torch.no_grad():
            output = self.model.evaluate_policy(slot_images, anchor_images)
            policy_sample = self._sample_policy(output, deterministic=deterministic)

        ptr1 = policy_sample["ptr1_actions"].squeeze(0).cpu()
        ptr2 = policy_sample["ptr2_actions"].squeeze(0).cpu()
        actions = [(int(ptr1[0].item()), int(ptr2[0].item())), (int(ptr1[1].item()), int(ptr2[1].item()))]
        return actions, {
            "slot_images": slot_images.squeeze(0).detach().cpu().to(torch.uint8),
            "anchor_images": anchor_images.squeeze(0).detach().cpu().to(torch.uint8),
            "actions": torch.stack([ptr1, ptr2], dim=-1).to(torch.long),
            "log_prob": policy_sample["log_prob"].squeeze(0).detach().cpu(),
            "value": policy_sample["value"].squeeze(0).detach().cpu(),
            "outside_prob": policy_sample["outside_prob"].squeeze(0).detach().cpu(),
        }

    def _build_mixed_policy(self, logits: torch.Tensor, epsilon: float) -> torch.Tensor:
        policy_probs = torch.softmax(logits, dim=0)
        uniform = torch.full_like(policy_probs, 1.0 / policy_probs.numel())
        mixed_probs = (1.0 - epsilon) * policy_probs + epsilon * uniform
        return mixed_probs / mixed_probs.sum()

    def record_transition(
        self,
        policy_info: Dict[str, torch.Tensor],
        reward: float,
        done: bool,
    ) -> None:
        self.buffer.slot_images.append(policy_info["slot_images"])
        self.buffer.anchor_images.append(policy_info["anchor_images"])
        self.buffer.actions.append(policy_info["actions"])
        self.buffer.log_probs.append(policy_info["log_prob"])
        self.buffer.values.append(policy_info["value"])
        self.buffer.outside_probs.append(policy_info["outside_prob"])
        self.buffer.rewards.append(torch.tensor(reward, dtype=torch.float32))
        self.buffer.dones.append(torch.tensor(float(done), dtype=torch.float32))

    def _compute_bootstrap_value(self, observations: Sequence[Dict[str, torch.Tensor]], done: bool) -> float:
        if done:
            return 0.0
        self.model.eval()
        slot_images, anchor_images = self._stack_obs(observations)
        with torch.no_grad():
            value = self.model.evaluate_policy(slot_images, anchor_images)["value"]
        return float(value.squeeze(0).item())

    def _compute_advantages(self, last_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = torch.stack(self.buffer.rewards)
        dones = torch.stack(self.buffer.dones)
        values = torch.stack(self.buffer.values)
        outside_probs = torch.stack(self.buffer.outside_probs)

        intent_bonus = self.config.intent_align_coef * (1.0 - torch.abs(outside_probs[:, 0] - outside_probs[:, 1]))
        rewards = rewards + intent_bonus

        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0)
        next_value = torch.tensor(last_value, dtype=torch.float32)

        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]
            delta = rewards[step] + self.config.gamma * next_value * mask - values[step]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages[step] = gae
            returns[step] = advantages[step] + values[step]
            next_value = values[step]

        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return returns, advantages

    def update(self, next_observations: Sequence[Dict[str, torch.Tensor]], done: bool, show: bool = False) -> Dict[str, float]:
        if not self.buffer.rewards:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        self.model.train()
        last_value = self._compute_bootstrap_value(next_observations, done)
        returns, advantages = self._compute_advantages(last_value)

        slot_images = torch.stack(self.buffer.slot_images).to(self.device)
        anchor_images = torch.stack(self.buffer.anchor_images).to(self.device)
        actions = torch.stack(self.buffer.actions).to(self.device)
        old_log_probs = torch.stack(self.buffer.log_probs).to(self.device).sum(dim=-1)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        indices = np.arange(len(self.buffer.rewards))
        loss_log: List[Tuple[float, float, float]] = []

        for _ in range(self.config.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.config.mini_batch_size):
                batch_indices = indices[start : start + self.config.mini_batch_size]
                batch_indices_t = torch.tensor(batch_indices, dtype=torch.long, device=self.device)

                batch_slot = slot_images[batch_indices_t]
                batch_anchor = anchor_images[batch_indices_t]
                batch_actions = actions[batch_indices_t]
                batch_ptr1 = batch_actions[:, :, 0]
                batch_ptr2 = batch_actions[:, :, 1]

                output = self.model.evaluate_policy(
                    batch_slot,
                    batch_anchor,
                    ptr1_actions=batch_ptr1,
                )
                ptr1_dist = Categorical(logits=output["ptr1_logits"])
                ptr1_log_prob = ptr1_dist.log_prob(batch_ptr1)
                ptr1_entropy = ptr1_dist.entropy()

                ptr2_logits = output["ptr2_logits"]
                ptr2_dist = Categorical(logits=ptr2_logits)
                ptr2_log_prob = ptr2_dist.log_prob(batch_ptr2)
                ptr2_entropy = ptr2_dist.entropy()

                new_log_prob = (ptr1_log_prob + ptr2_log_prob).sum(dim=-1)
                entropy = (ptr1_entropy + ptr2_entropy).mean()
                value = output["value"]

                ratio = torch.exp(new_log_prob - old_log_probs[batch_indices_t])
                unclipped = ratio * advantages[batch_indices_t]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                ) * advantages[batch_indices_t]

                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(value, returns[batch_indices_t])
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                loss_log.append((float(policy_loss.item()), float(value_loss.item()), float(entropy.item())))

        mean_policy = float(np.mean([item[0] for item in loss_log])) if loss_log else 0.0
        mean_value = float(np.mean([item[1] for item in loss_log])) if loss_log else 0.0
        mean_entropy = float(np.mean([item[2] for item in loss_log])) if loss_log else 0.0
        self.buffer.clear()

        if show:
            print(
                f"MAPPO update - policy_loss: {mean_policy:.4f}, "
                f"value_loss: {mean_value:.4f}, entropy: {mean_entropy:.4f}"
            )

        return {
            "policy_loss": mean_policy,
            "value_loss": mean_value,
            "entropy": mean_entropy,
        }
