import copy
import itertools
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch


SLOT_TO_GRID_INDEX = [0, 1, 2, 3, 5, 6, 7, 8]
GRID_TO_SLOT_INDEX = {grid_index: slot_index for slot_index, grid_index in enumerate(SLOT_TO_GRID_INDEX)}
HORIZONTAL_GRID_PAIRS = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)]
VERTICAL_GRID_PAIRS = [(0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8)]


@dataclass
class RewardConfig:
    pairwise: float = 0.2
    cate: float = 0.8
    done_reward: float = 1000.0
    step_penalty: float = -1.0


class SD2RLEnv:
    """Single-image SD2RL environment with a fixed center anchor."""

    def __init__(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        piece_num: int = 9,
        epsilon: float = 0.5,
        epsilon_gamma: float = 0.998,
        max_steps: int = 100,
        initial_swap_num: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        reward_config: Optional[RewardConfig] = None,
        epochs: int = 5,
    ) -> None:
        if piece_num != 9:
            raise ValueError("SD2RLEnv currently supports only 3x3 puzzles with a fixed center anchor.")

        self.image = train_x
        self.label = train_y
        self.sample_number = train_x.shape[0]
        self.piece_num = piece_num
        self.image_num = 1
        self.buffer_size = 0
        self.device = device
        self.epsilon = epsilon
        self.epsilon_gamma = epsilon_gamma
        self.max_steps = max_steps
        self.initial_swap_num = initial_swap_num
        self.epochs = epochs
        self.reward_config = reward_config or RewardConfig()
        self.action_space = list(itertools.combinations(range(self.piece_num - 1), 2))
        self.action_num = len(self.action_space)

        self.image_id: List[int] = []
        self.current_permutation: List[int] = []
        self.permutation_list: List[List[int]] = []
        self.step_count = 0
        self.last_metrics: Dict[str, float] = {}

        self._episode_piece_cache: Dict[int, Tuple[List[torch.Tensor], torch.Tensor]] = {}

    def _decode_label(self, image_index: int) -> List[int]:
        permutation_raw = list(np.argmax(self.label[image_index], axis=1))
        for idx, value in enumerate(permutation_raw):
            if value >= 4:
                permutation_raw[idx] += 1
        permutation_raw.insert(4, 4)
        return permutation_raw

    def _load_piece_tensors(self, image_index: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        if image_index in self._episode_piece_cache:
            return self._episode_piece_cache[image_index]

        image_np = np.asarray(self.image[image_index], dtype=np.float32)
        image_raw = torch.tensor(image_np.tolist(), dtype=torch.float32).permute(2, 0, 1).contiguous()
        fragments = [
            image_raw[:, 0:96, 0:96],
            image_raw[:, 0:96, 96:192],
            image_raw[:, 0:96, 192:288],
            image_raw[:, 96:192, 0:96],
            image_raw[:, 96:192, 96:192],
            image_raw[:, 96:192, 192:288],
            image_raw[:, 192:288, 0:96],
            image_raw[:, 192:288, 96:192],
            image_raw[:, 192:288, 192:288],
        ]

        permutation = self._decode_label(image_index)
        ordered_fragments = [None] * self.piece_num
        for grid_index, fragment in zip(permutation, fragments):
            ordered_fragments[grid_index] = fragment

        movable_pieces = [ordered_fragments[grid_index] for grid_index in SLOT_TO_GRID_INDEX]
        anchor_piece = ordered_fragments[self.piece_num // 2]
        self._episode_piece_cache[image_index] = (movable_pieces, anchor_piece)
        return movable_pieces, anchor_piece

    def _normalize_image_id(self, image_id: Optional[Union[int, Sequence[int]]]) -> int:
        if image_id is None:
            return random.randrange(self.sample_number)
        if isinstance(image_id, (list, tuple, np.ndarray)):
            if len(image_id) == 0:
                raise ValueError("image_id sequence cannot be empty.")
            return int(image_id[0])
        return int(image_id)

    def _build_board_image_for_id(self, image_index: int, permutation: Sequence[int]) -> torch.Tensor:
        movable_pieces, anchor_piece = self._load_piece_tensors(image_index)
        image = torch.zeros(3, 288, 288, dtype=torch.float32)
        for slot_index, piece_id in enumerate(permutation):
            grid_index = SLOT_TO_GRID_INDEX[slot_index]
            row, col = divmod(grid_index, 3)
            image[:, row * 96 : (row + 1) * 96, col * 96 : (col + 1) * 96] = movable_pieces[piece_id]

        image[:, 96:192, 96:192] = anchor_piece
        return image

    def _build_full_grid_ids(self, permutation: Sequence[int]) -> List[int]:
        full_grid = [-1] * self.piece_num
        for slot_index, piece_id in enumerate(permutation):
            current_grid_index = SLOT_TO_GRID_INDEX[slot_index]
            correct_grid_index = SLOT_TO_GRID_INDEX[piece_id]
            full_grid[current_grid_index] = correct_grid_index
        full_grid[self.piece_num // 2] = self.piece_num // 2
        return full_grid

    def _compute_pair_counts(self, permutation: Sequence[int]) -> Tuple[int, int]:
        full_grid = self._build_full_grid_ids(permutation)
        hori = sum(int((full_grid[i], full_grid[j]) == (i, j)) for i, j in HORIZONTAL_GRID_PAIRS)
        vert = sum(int((full_grid[i], full_grid[j]) == (i, j)) for i, j in VERTICAL_GRID_PAIRS)
        return hori, vert

    def _compute_cate(self, permutation: Sequence[int]) -> int:
        return sum(int(piece_id == slot_index) for slot_index, piece_id in enumerate(permutation))

    def _make_info(self, permutation: Sequence[int]) -> Dict[str, float]:
        cate = self._compute_cate(permutation)
        hori, vert = self._compute_pair_counts(permutation)
        is_success = float(cate == self.piece_num - 1)
        reward = self.reward_config.done_reward
        if not is_success:
            reward = (
                cate * self.reward_config.cate
                + (hori + vert) * self.reward_config.pairwise
                + self.reward_config.step_penalty
            )

        return {
            "cate": float(cate),
            "hori": float(hori),
            "vert": float(vert),
            "is_success": is_success,
            "cate_accuracy": float(cate) / float(self.piece_num - 1),
            "hori_accuracy": float(hori) / float(len(HORIZONTAL_GRID_PAIRS)),
            "vert_accuracy": float(vert) / float(len(VERTICAL_GRID_PAIRS)),
            "reward": float(reward),
        }

    def reset(
        self,
        image_id: Optional[Union[int, Sequence[int]]] = None,
        swap_num: Optional[int] = None,
    ) -> List[int]:
        sampled_image_id = self._normalize_image_id(image_id)
        self.image_id = [sampled_image_id]
        self._load_piece_tensors(sampled_image_id)

        permutation = list(range(self.piece_num - 1))
        total_swaps = self.initial_swap_num if swap_num is None else int(swap_num)
        for _ in range(total_swaps):
            action_index = random.randrange(self.action_num)
            permutation = self.permute(permutation, action_index)

        self.current_permutation = permutation
        self.permutation_list = [copy.deepcopy(permutation)]
        self.step_count = 0
        self.last_metrics = self.get_metrics()
        return copy.deepcopy(self.current_permutation)

    def summon_permutation_list(self, swap_num: int = 0, id: Optional[Sequence[int]] = None) -> None:
        self.reset(image_id=id, swap_num=swap_num)

    def permute(self, cur_permutation: Sequence[int], action_index: int) -> List[int]:
        if action_index < 0 or action_index >= self.action_num:
            raise ValueError(f"Action index {action_index} out of range 0..{self.action_num - 1}.")

        new_permutation = list(cur_permutation)
        slot_a, slot_b = self.action_space[action_index]
        new_permutation[slot_a], new_permutation[slot_b] = new_permutation[slot_b], new_permutation[slot_a]
        return new_permutation

    def step(self, action: int) -> Tuple[List[int], float, bool, Dict[str, float]]:
        self.current_permutation = self.permute(self.current_permutation, action)
        self.permutation_list = [copy.deepcopy(self.current_permutation)]
        self.step_count += 1

        info = self._make_info(self.current_permutation)
        info["step_count"] = float(self.step_count)
        done = bool(info["is_success"]) or self.step_count >= self.max_steps
        self.last_metrics = info
        return copy.deepcopy(self.current_permutation), float(info["reward"]), done, copy.deepcopy(info)

    def get_metrics(self, permutation: Optional[Sequence[int]] = None) -> Dict[str, float]:
        target_permutation = self.current_permutation if permutation is None else permutation
        return self._make_info(target_permutation)

    def get_reward(self, permutation: Optional[Sequence[int]] = None) -> float:
        return float(self.get_metrics(permutation)["reward"])

    def is_solved(self, permutation: Optional[Sequence[int]] = None) -> bool:
        return bool(self.get_metrics(permutation)["is_success"])

    def get_accuracy(self, permutation: Optional[Sequence[int]] = None) -> Tuple[float, float, float, float]:
        metrics = self.get_metrics(permutation)
        return (
            metrics["is_success"],
            metrics["cate_accuracy"],
            metrics["hori_accuracy"],
            metrics["vert_accuracy"],
        )

    def get_visual_score(self, permutation: Sequence[int], image_index: int = 0) -> float:
        metrics = self.get_metrics(permutation)
        if metrics["is_success"]:
            return self.reward_config.done_reward
        return metrics["cate"] * self.reward_config.cate + (metrics["hori"] + metrics["vert"]) * self.reward_config.pairwise

    def get_image(
        self,
        permutation: Sequence[int],
        image_index: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.image_id:
            raise RuntimeError("Environment has not been reset yet.")

        image_tensor = self._build_board_image_for_id(self.image_id[image_index], permutation)
        outsider = torch.zeros(3, 96, 96, dtype=torch.float32)
        return image_tensor.unsqueeze(0).to(self.device), outsider.unsqueeze(0).to(self.device)

    def request_for_image(
        self,
        image_id: Union[int, Sequence[int]],
        permutation: Sequence[int],
        image_index: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_image_id = self._normalize_image_id(image_id)
        image_tensor = self._build_board_image_for_id(target_image_id, permutation)
        outsider = torch.zeros(3, 96, 96, dtype=torch.float32)
        return image_tensor.unsqueeze(0).to(self.device), outsider.unsqueeze(0).to(self.device)

    def show_image(self, permutation: Optional[Sequence[int]] = None) -> None:
        target_permutation = self.current_permutation if permutation is None else permutation
        image_tensor, _ = self.get_image(target_permutation, image_index=0)
        image = image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
        cv2.imshow("SD2RL Board", image)
        cv2.waitKey(1)
