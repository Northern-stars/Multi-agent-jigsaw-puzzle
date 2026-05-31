import copy
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from model_code.digger_model import DiggerModel


class DiggerAgent:
    """Masked slot-selection agent for the excavation stage."""

    def __init__(
        self,
        model: DiggerModel,
        gamma: float = 0.99,
        batch_size: int = 16,
        memory_size: int = 2000,
        lr: float = 1e-4,
        epsilon: float = 0.2,
        epsilon_min: float = 0.05,
        epsilon_gamma: float = 0.998,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model = model.to(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_gamma = epsilon_gamma
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-8)
        self.memory: List[Dict[str, object]] = []
        self.memory_counter = 0

    def clean_memory(self) -> None:
        self.memory = []
        self.memory_counter = 0

    def choose_action(self, observation: Dict[str, object], deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        self.model.eval()
        board_image = observation["board_image"].to(self.device)
        empty_mask = observation["empty_mask"].to(self.device)
        valid_mask = ~empty_mask.squeeze(0).bool()
        with torch.no_grad():
            logits = self.model(board_image, empty_mask)
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten().tolist()
        if not valid_indices:
            raise RuntimeError("Digger has no valid slot to excavate.")
        if not deterministic and random.random() < self.epsilon:
            action = random.choice(valid_indices)
        else:
            action = int(torch.argmax(logits.squeeze(0)).item())
        return action, logits.squeeze(0).detach().cpu()

    def recording_memory(
        self,
        observation: Dict[str, object],
        action: int,
        reward: float,
        done: bool,
    ) -> None:
        memory = {
            "board_image": observation["board_image"].detach().cpu(),
            "empty_mask": observation["empty_mask"].detach().cpu(),
            "action": int(action),
            "reward": float(reward),
            "done": bool(done),
        }
        if len(self.memory) < self.memory_size:
            self.memory.append(memory)
        else:
            self.memory[self.memory_counter] = memory
        self.memory_counter = (self.memory_counter + 1) % self.memory_size

    def update(self, train_epochs: int = 1, show: bool = False) -> Optional[float]:
        if not self.memory:
            return None
        losses = []
        order = list(range(len(self.memory)))
        random.shuffle(order)
        self.model.train()
        for _ in range(train_epochs):
            for start in range(0, len(order), self.batch_size):
                batch_indices = order[start : start + self.batch_size]
                batch = [self.memory[idx] for idx in batch_indices]
                board_images = torch.cat([item["board_image"] for item in batch], dim=0).to(self.device)
                empty_masks = torch.cat([item["empty_mask"] for item in batch], dim=0).to(self.device)
                actions = torch.tensor([item["action"] for item in batch], dtype=torch.long, device=self.device)
                rewards = torch.tensor([item["reward"] for item in batch], dtype=torch.float32, device=self.device)

                logits = self.model(board_images, empty_masks)
                ce_loss = nn.CrossEntropyLoss(reduction="none")(logits, actions)
                # Positive dig rewards should strengthen the selected class; negative rewards dampen it.
                weights = torch.clamp(rewards.abs(), min=0.1)
                signed_loss = torch.where(rewards >= 0, ce_loss, -0.1 * ce_loss)
                loss = (signed_loss * weights).mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_gamma)
        mean_loss = float(np.mean(losses)) if losses else None
        if show and mean_loss is not None:
            print(f"Digger loss: {mean_loss:.6f}")
        return mean_loss

    def state_dict(self) -> Dict[str, object]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.model.load_state_dict(copy.deepcopy(state["model"]))
        if "optimizer" in state:
            self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        self.epsilon = float(state.get("epsilon", self.epsilon))
