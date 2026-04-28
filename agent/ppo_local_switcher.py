import copy
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
ACTOR_LR_MIN = 1e-6
CRITIC_LR_MIN = 1e-6
CLIP_GRAD_NORM = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTOR_SCHEDULAR_STEP = 200
CRITIC_SCHEDULAR_STEP = 200


class PPOLocalSwitcher:
    """PPO version of Local_switcher with the same candidate-next-state workflow."""

    def __init__(
        self,
        model: nn.Module,
        memory_size: int,
        gamma: float,
        batch_size: int,
        env: Any,
        action_num: int,
        tau: float = 1e-3,
        recommand: bool = False,
        recommand_num: int = 10,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        ppo_epochs: int = 4,
    ) -> None:
        self.model = model
        self.main_model = self.model
        self.critic_model = copy.deepcopy(model)

        self.actor_optimizer = torch.optim.Adam(self.model.parameters(), lr=ACTOR_LR, eps=1e-8)
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr=CRITIC_LR, eps=1e-8)
        self.actor_schedular = torch.optim.lr_scheduler.StepLR(
            self.actor_optimizer, step_size=ACTOR_SCHEDULAR_STEP
        )
        self.critic_schedular = torch.optim.lr_scheduler.StepLR(
            self.critic_optimizer, step_size=CRITIC_SCHEDULAR_STEP
        )

        self.tau = tau
        self.memory_size = memory_size
        self.memory: List[Dict[str, Any]] = []
        self.memory_counter = 0
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = env
        self.action_num = action_num

        self.recommand = recommand
        self.recommand_num = recommand_num

        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs

        self._pending_action_info: Optional[Dict[str, Any]] = None

    def clean_memory(self) -> None:
        self.memory = []
        self.memory_counter = 0
        self._pending_action_info = None

    def permute(self, cur_permutation: Sequence[int], action_index: int) -> List[int]:
        return self.env.permute(cur_permutation, action_index)

    def _candidate_actions(self, permutation: Sequence[int], image_index: int) -> List[int]:
        if self.recommand:
            return self.recommanded_action(permutation, image_index)
        return list(range(self.action_num))

    def _batch_model_forward(self, model: nn.Module, image_batches: List[torch.Tensor]) -> torch.Tensor:
        outputs = []
        index = 0
        while index < len(image_batches):
            chunk = image_batches[index : index + self.batch_size]
            image_tensor = torch.cat(chunk, dim=0).to(DEVICE)
            value = model(image_tensor)
            outputs.append(value.squeeze(-1))
            index += self.batch_size
        return torch.cat(outputs, dim=0)

    def _get_state_image(self, permutation: Sequence[int], image_index: int) -> torch.Tensor:
        image, _ = self.env.get_image(permutation, image_index=image_index)
        return image

    def _get_state_image_from_memory(
        self, image_id: Sequence[int], permutation: Sequence[int], image_index: int
    ) -> torch.Tensor:
        image, _ = self.env.request_for_image(image_id=image_id, permutation=permutation, image_index=image_index)
        return image

    def _evaluate_state_value(
        self,
        permutation: Sequence[int],
        image_index: int,
        image_id: Optional[Sequence[int]] = None,
    ) -> float:
        if image_id is None:
            image_tensor = self._get_state_image(permutation, image_index)
        else:
            image_tensor = self._get_state_image_from_memory(image_id, permutation, image_index)

        if image_tensor.size(0) == 1:
            self.critic_model.eval()
        else:
            self.critic_model.train()

        with torch.no_grad():
            value = self.critic_model(image_tensor.to(DEVICE)).squeeze().item()
        return float(value)

    def _evaluate_candidate_logits(
        self,
        permutation: Sequence[int],
        image_index: int,
        candidate_actions: Sequence[int],
        image_id: Optional[Sequence[int]] = None,
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        actor_model = self.model if model is None else model
        candidate_images: List[torch.Tensor] = []
        for action in candidate_actions:
            next_permutation = self.permute(permutation, action)
            if image_id is None:
                image, _ = self.env.get_image(next_permutation, image_index=image_index)
            else:
                image, _ = self.env.request_for_image(image_id=image_id, permutation=next_permutation, image_index=image_index)
            candidate_images.append(image.cpu())

        if len(candidate_images) == 1:
            actor_model.eval()
        else:
            actor_model.train()

        return self._batch_model_forward(actor_model, candidate_images)

    def _build_mixed_policy(self, logits: torch.Tensor, epsilon: float) -> torch.Tensor:
        policy_probs = torch.softmax(logits, dim=0)
        uniform = torch.full_like(policy_probs, 1.0 / policy_probs.numel())
        mixed_probs = (1.0 - epsilon) * policy_probs + epsilon * uniform
        return mixed_probs / mixed_probs.sum()

    def recommanded_action(self, permutation: Sequence[int], image_index: int) -> List[int]:
        score_list = np.zeros(self.action_num)
        for i in range(self.action_num):
            permutation_copy = self.permute(permutation, i)
            score_list[i] = self.env.get_visual_score(permutation_copy, image_index)
        best_action = np.argsort(score_list)[::-1].tolist()
        return best_action[: self.recommand_num]

    def _select_action_with_policy(
        self,
        permutation: Sequence[int],
        image_index: int,
        deterministic: bool = False,
        image_id: Optional[Sequence[int]] = None,
        model: Optional[nn.Module] = None,
        epsilon: Optional[float] = None,
    ) -> Tuple[int, torch.Tensor, float, List[int], torch.Tensor]:
        candidate_actions = self._candidate_actions(permutation, image_index)
        logits = self._evaluate_candidate_logits(
            permutation=permutation,
            image_index=image_index,
            candidate_actions=candidate_actions,
            image_id=image_id,
            model=model,
        )
        rollout_epsilon = self.env.epsilon if epsilon is None else epsilon
        probs = self._build_mixed_policy(logits, rollout_epsilon)
        dist = torch.distributions.Categorical(probs=probs)

        if deterministic:
            selected_index = int(torch.argmax(probs).item())
        else:
            selected_index = int(dist.sample().item())

        action = candidate_actions[selected_index]
        log_prob = dist.log_prob(torch.tensor(selected_index, device=probs.device)).detach().cpu()
        value = self._evaluate_state_value(permutation, image_index, image_id=image_id)
        return action, log_prob, value, candidate_actions, probs.detach().cpu()

    def choose_action(
        self,
        permutation: Sequence[int],
        image_index: int,
        deterministic: bool = False,
    ) -> int:
        action, _, _, _, _ = self._select_action_with_policy(
            permutation=permutation,
            image_index=image_index,
            deterministic=deterministic,
        )
        return action

    def act(self, permutation: Sequence[int], image_index: int, deterministic: bool = False) -> Tuple[List[int], int]:
        self.model.eval()
        action, log_prob, value, candidate_actions, probs = self._select_action_with_policy(
            permutation=permutation,
            image_index=image_index,
            deterministic=deterministic,
        )
        permutation_ = self.permute(cur_permutation=permutation, action_index=action)
        self._pending_action_info = {
            "State": list(permutation),
            "Action": action,
            "Next_state": list(permutation_),
            "Image_index": image_index,
            "Log_prob": log_prob,
            "Value": value,
            "Candidate_actions": list(candidate_actions),
            "Policy_probs": probs,
            "Epsilon": float(self.env.epsilon),
        }
        return permutation_, action

    def recording_memory(
        self,
        image_id,
        image_index,
        state,
        action,
        reward,
        next_state,
        done,
    ) -> None:
        pending = self._pending_action_info
        if (
            pending is None
            or pending["Image_index"] != image_index
            or pending["Action"] != action
            or list(pending["State"]) != list(state)
            or list(pending["Next_state"]) != list(next_state)
        ):
            replay_action, log_prob, value, candidate_actions, probs = self._select_action_with_policy(
                permutation=state,
                image_index=image_index,
                deterministic=True,
                image_id=image_id,
                epsilon=float(self.env.epsilon),
            )
            if replay_action != action:
                candidate_actions = self._candidate_actions(state, image_index)
                logits = self._evaluate_candidate_logits(
                    permutation=state,
                    image_index=image_index,
                    candidate_actions=candidate_actions,
                    image_id=image_id,
                )
                probs = self._build_mixed_policy(logits, float(self.env.epsilon)).detach().cpu()
                selected_index = candidate_actions.index(action)
                log_prob = torch.log(probs[selected_index].clamp(min=1e-8))
            pending = {
                "Log_prob": log_prob,
                "Value": value,
                "Candidate_actions": list(candidate_actions),
                "Policy_probs": probs,
                "Epsilon": float(self.env.epsilon),
            }

        memory = {
            "Image_id": image_id,
            "Image_index": image_index,
            "State": list(state),
            "Action": action,
            "Reward": reward,
            "Next_state": list(next_state),
            "Done": done,
            "Log_prob": pending["Log_prob"],
            "Value": pending["Value"],
            "Candidate_actions": pending["Candidate_actions"],
            "Policy_probs": pending["Policy_probs"],
            "Epsilon": pending["Epsilon"],
        }
        if len(self.memory) < self.memory_size:
            self.memory.append(memory)
        else:
            self.memory[self.memory_counter] = memory
        self.memory_counter = (self.memory_counter + 1) % self.memory_size
        self._pending_action_info = None

    def _compute_next_values(self) -> np.ndarray:
        next_values = np.zeros(len(self.memory), dtype=np.float32)
        for index, mem in enumerate(self.memory):
            if mem["Done"]:
                next_values[index] = 0.0
            else:
                next_values[index] = self._evaluate_state_value(
                    permutation=mem["Next_state"],
                    image_index=mem["Image_index"],
                    image_id=mem["Image_id"],
                )
        return next_values

    def _compute_advantages_and_returns(self) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.array([float(mem["Reward"]) for mem in self.memory], dtype=np.float32)
        dones = np.array([float(mem["Done"]) for mem in self.memory], dtype=np.float32)
        values = np.array([float(mem["Value"]) for mem in self.memory], dtype=np.float32)
        next_values = self._compute_next_values()

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for index in reversed(range(len(self.memory))):
            mask = 1.0 - dones[index]
            delta = rewards[index] + self.gamma * next_values[index] * mask - values[index]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[index] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def _evaluate_action_log_prob_and_entropy(
        self,
        memory: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        candidate_actions = memory["Candidate_actions"]
        logits = self._evaluate_candidate_logits(
            permutation=memory["State"],
            image_index=memory["Image_index"],
            candidate_actions=candidate_actions,
            image_id=memory["Image_id"],
            model=self.model,
        )
        probs = self._build_mixed_policy(logits, float(memory["Epsilon"]))
        dist = torch.distributions.Categorical(probs=probs)
        selected_index = candidate_actions.index(memory["Action"])
        action_tensor = torch.tensor(selected_index, device=probs.device)
        log_prob = dist.log_prob(action_tensor)
        entropy = dist.entropy()
        return log_prob, entropy

    def update(self, show: bool = False) -> None:
        if len(self.memory) == 0:
            return

        advantages, returns = self._compute_advantages_and_returns()
        order = np.arange(len(self.memory))
        actor_loss_sum: List[float] = []
        critic_loss_sum: List[float] = []
        entropy_sum: List[float] = []

        for _ in range(self.ppo_epochs):
            np.random.shuffle(order)
            for start in range(0, len(order), self.batch_size):
                batch_indices = order[start : start + self.batch_size]
                sample_dicts = [self.memory[index] for index in batch_indices]

                batch_states = [
                    self._get_state_image_from_memory(mem["Image_id"], mem["State"], mem["Image_index"])
                    for mem in sample_dicts
                ]
                state_tensor = torch.cat(batch_states, dim=0).to(DEVICE)

                if state_tensor.size(0) == 1:
                    self.model.eval()
                    self.critic_model.eval()
                else:
                    self.model.train()
                    self.critic_model.train()

                new_log_probs = []
                entropies = []
                for mem in sample_dicts:
                    log_prob, entropy = self._evaluate_action_log_prob_and_entropy(mem)
                    new_log_probs.append(log_prob.unsqueeze(0))
                    entropies.append(entropy.unsqueeze(0))

                new_log_probs_tensor = torch.cat(new_log_probs, dim=0).to(DEVICE).unsqueeze(-1)
                entropy_tensor = torch.cat(entropies, dim=0).to(DEVICE)
                old_log_probs_tensor = torch.stack([mem["Log_prob"] for mem in sample_dicts]).to(DEVICE).unsqueeze(-1)
                advantages_tensor = torch.tensor(
                    [advantages[index] for index in batch_indices], dtype=torch.float32, device=DEVICE
                ).unsqueeze(-1)
                returns_tensor = torch.tensor(
                    [returns[index] for index in batch_indices], dtype=torch.float32, device=DEVICE
                ).unsqueeze(-1)

                values_pred = self.critic_model(state_tensor)
                ratio = torch.exp(new_log_probs_tensor - old_log_probs_tensor)
                surr1 = ratio * advantages_tensor
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_tensor
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy_tensor.mean()
                critic_loss = nn.functional.mse_loss(values_pred, returns_tensor)
                loss = actor_loss + self.value_coef * critic_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP_GRAD_NORM)
                torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), CLIP_GRAD_NORM)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                actor_loss_sum.append(float(actor_loss.item()))
                critic_loss_sum.append(float(critic_loss.item()))
                entropy_sum.append(float(entropy_tensor.mean().item()))

        if self.actor_optimizer.state_dict()["param_groups"][0]["lr"] > ACTOR_LR_MIN:
            self.actor_schedular.step()
        if self.critic_optimizer.state_dict()["param_groups"][0]["lr"] > CRITIC_LR_MIN:
            self.critic_schedular.step()

        if show and actor_loss_sum:
            print(
                f"PPO local switcher - actor_loss: {np.mean(actor_loss_sum):.4f}, "
                f"critic_loss: {np.mean(critic_loss_sum):.4f}, entropy: {np.mean(entropy_sum):.4f}"
            )

        self.clean_memory()


PPO_Local_switcher = PPOLocalSwitcher
