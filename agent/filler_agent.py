import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from model_code.filler_model import FillerModel


class FillerAgent:
    """Candidate scorer and selector for the fill stage."""

    def __init__(
        self,
        model: FillerModel,
        batch_size: int = 16,
        memory_size: int = 2000,
        lr: float = 1e-4,
        epsilon: float = 0.1,
        epsilon_min: float = 0.02,
        epsilon_gamma: float = 0.998,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model = model.to(device)
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

    def _score_images(self, images: torch.Tensor) -> torch.Tensor:
        scores: List[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, images.size(0), self.batch_size):
                batch = images[start : start + self.batch_size].to(self.device)
                scores.append(self.model(batch).detach().cpu())
        return torch.cat(scores, dim=0)

    def score_candidates(self, candidates: Sequence[Dict[str, object]]) -> torch.Tensor:
        if not candidates:
            return torch.empty(0)
        scores = []
        for candidate in candidates:
            if candidate["type"] == "assignment":
                board_scores = self._score_images(candidate["candidate_images"])
                scores.append(board_scores.sum().view(1))
            else:
                scores.append(self._score_images(candidate["candidate_image"]).view(1))
        return torch.cat(scores, dim=0)

    def choose_candidate(
        self,
        candidates: Sequence[Dict[str, object]],
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor]:
        if not candidates:
            raise RuntimeError("Filler has no candidate to choose from.")
        scores = self.score_candidates(candidates)
        if not deterministic and random.random() < self.epsilon:
            candidate_index = random.randrange(len(candidates))
        else:
            candidate_index = int(torch.argmax(scores).item())
        return candidate_index, scores

    def recording_memory(
        self,
        candidate: Dict[str, object],
        reward: float,
        score: Optional[float] = None,
    ) -> None:
        if candidate["type"] == "assignment":
            image = candidate["candidate_images"].detach().cpu()
        else:
            image = candidate["candidate_image"].detach().cpu()
        memory = {
            "image": image,
            "reward": float(reward),
            "score": float(score) if score is not None else None,
            "type": candidate["type"],
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
                image_chunks = []
                targets = []
                for idx in batch_indices:
                    item = self.memory[idx]
                    image = item["image"]
                    reward = float(item["reward"])
                    if item["type"] == "assignment":
                        image_chunks.append(image)
                        targets.extend([reward / max(1, image.size(0))] * image.size(0))
                    else:
                        image_chunks.append(image)
                        targets.append(reward)

                images = torch.cat(image_chunks, dim=0).to(self.device)
                target_tensor = torch.tensor(targets, dtype=torch.float32, device=self.device)
                pred = self.model(images)
                loss = nn.SmoothL1Loss()(pred, target_tensor)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_gamma)
        mean_loss = float(np.mean(losses)) if losses else None
        if show and mean_loss is not None:
            print(f"Filler loss: {mean_loss:.6f}")
        return mean_loss
