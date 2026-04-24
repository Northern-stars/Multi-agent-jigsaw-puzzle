import copy
import itertools
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


SLOT_TO_GRID_INDEX = [0, 1, 2, 3, 5, 6, 7, 8]
GRID_TO_SLOT_INDEX = {grid_index: slot_index for slot_index, grid_index in enumerate(SLOT_TO_GRID_INDEX)}
CATEGORY_NAMES = ("painting", "engraving", "artifact")
SLOT_TO_GRID_POS = {slot_index: divmod(grid_index, 3) for slot_index, grid_index in enumerate(SLOT_TO_GRID_INDEX)}


@dataclass
class RewardWeights:
    pairwise: float = 0.2
    cate: float = 0.8
    consistency: float = 0.5
    done_reward: float = 1000.0
    consistency_reward: float = 200.0
    penalty: float = -0.5
    coord_fail: float = 0.0


@dataclass
class PieceRecord:
    global_id: int
    source_board: int
    source_slot: int
    image_id: int
    tensor: torch.Tensor


class DualBoardEnv:
    """2-Mixed dual-board environment aligned with the requirement document.

    The API is intentionally close to the existing project style:
    - `reset()` prepares a new episode and returns local observations
    - `step()` applies two agents' synchronous actions
    - `get_metrics()` exposes puzzle-quality metrics
    - `show_image()` renders the current boards for debugging
    """

    def __init__(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        gamma: float,
        image_num: int = 2,
        buffer_size: int = 1,
        epsilon: float = 0.1,
        epsilon_gamma: float = 0.999,
        piece_num: int = 9,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        reward_weights: Optional[RewardWeights] = None,
        cooldown_steps: int = 5,
        max_steps: int = 100,
        training_mode: bool = True,
        category_ranges: Optional[Dict[str, Sequence[int]]] = None,
    ) -> None:
        if image_num != 2:
            raise ValueError("DualBoardEnv currently supports exactly 2 boards.")
        if piece_num != 9:
            raise ValueError("DualBoardEnv assumes 3x3 puzzles with a fixed center anchor.")

        self.image = train_x
        self.label = train_y
        self.sample_number = train_x.shape[0]
        self.gamma = gamma
        self.image_num = image_num
        self.buffer_size = buffer_size
        self.epsilon = epsilon
        self.epsilon_gamma = epsilon_gamma
        self.piece_num = piece_num
        self.device = device
        self.reward_weights = reward_weights or RewardWeights()
        self.cooldown_steps = cooldown_steps
        self.max_steps = max_steps
        self.training_mode = training_mode

        self.category_to_indices = self._build_category_indices(category_ranges)
        self.category_pairs = list(itertools.combinations(CATEGORY_NAMES, 2))

        self.image_id: List[int] = []
        self.image_categories: List[str] = []
        self.anchor_pieces: List[torch.Tensor] = []
        self.piece_records: Dict[int, PieceRecord] = {}
        self.permutation_list: List[List[int]] = []
        self.step_count = 0
        self.completion_step: Optional[int] = None
        self.last_metrics: Dict[str, float] = {}

    def _build_category_indices(
        self,
        category_ranges: Optional[Dict[str, Sequence[int]]],
    ) -> Dict[str, List[int]]:
        if category_ranges is not None:
            return {key: list(value) for key, value in category_ranges.items()}

        base = self.sample_number // len(CATEGORY_NAMES)
        remainder = self.sample_number % len(CATEGORY_NAMES)
        sizes = [base + (1 if idx < remainder else 0) for idx in range(len(CATEGORY_NAMES))]

        category_to_indices: Dict[str, List[int]] = {}
        start = 0
        for name, size in zip(CATEGORY_NAMES, sizes):
            category_to_indices[name] = list(range(start, start + size))
            start += size
        return category_to_indices

    def _decode_image(self, image_index: int) -> Tuple[List[torch.Tensor], List[int]]:
        # Avoid `torch.from_numpy` so the environment still works in setups where
        # the local PyTorch build cannot bridge the installed NumPy ABI.
        image_np = np.asarray(self.image[image_index], dtype=np.uint8)
        image_raw = torch.tensor(image_np.tolist(), dtype=torch.uint8).permute(2, 0, 1).contiguous()
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

        permutation_raw = list(np.argmax(self.label[image_index], axis=1))
        for idx, value in enumerate(permutation_raw):
            if value >= 4:
                permutation_raw[idx] += 1
        permutation_raw.insert(4, 4)
        return fragments, permutation_raw

    def _sample_episode_ids(self, explicit_ids: Optional[Sequence[int]] = None) -> Tuple[List[int], List[str]]:
        if explicit_ids is not None:
            image_ids = list(explicit_ids)
            categories = [self._find_category(image_id) for image_id in image_ids]
            return image_ids, categories

        category_a, category_b = random.choice(self.category_pairs)
        image_a = random.choice(self.category_to_indices[category_a])
        image_b = random.choice(self.category_to_indices[category_b])
        return [image_a, image_b], [category_a, category_b]

    def _find_category(self, image_index: int) -> str:
        for category_name, indices in self.category_to_indices.items():
            if image_index in indices:
                return category_name
        raise ValueError(f"Image index {image_index} not covered by category ranges.")

    def _build_episode_pieces(self, image_ids: Sequence[int]) -> None:
        self.anchor_pieces = []
        self.piece_records = {}
        next_piece_id = 0

        for board_index, image_index in enumerate(image_ids):
            fragments, permutation = self._decode_image(image_index)
            ordered_fragments = [None] * 9
            for grid_index, fragment in zip(permutation, fragments):
                ordered_fragments[grid_index] = fragment

            self.anchor_pieces.append(ordered_fragments[4])
            movable_grid_indices = [grid_idx for grid_idx in range(9) if grid_idx != 4]
            for slot_index, grid_index in enumerate(movable_grid_indices):
                self.piece_records[next_piece_id] = PieceRecord(
                    global_id=next_piece_id,
                    source_board=board_index,
                    source_slot=slot_index,
                    image_id=image_index,
                    tensor=ordered_fragments[grid_index],
                )
                next_piece_id += 1

    def _build_initial_boards(self) -> None:
        global_piece_ids = list(self.piece_records.keys())
        # random.shuffle(global_piece_ids)
        self.permutation_list = [
            global_piece_ids[:8],
            global_piece_ids[8:16],
        ]

    def _get_slot_tensor(self, piece_id: int) -> torch.Tensor:
        return self.piece_records[piece_id].tensor

    def _build_board_image(self, board_index: int) -> torch.Tensor:
        image = torch.zeros(3, 288, 288, dtype=torch.uint8)
        board_pieces = self.permutation_list[board_index]
        for slot_index, piece_id in enumerate(board_pieces):
            grid_index = SLOT_TO_GRID_INDEX[slot_index]
            row, col = divmod(grid_index, 3)
            image[:, row * 96 : (row + 1) * 96, col * 96 : (col + 1) * 96] = self._get_slot_tensor(piece_id)

        anchor = self.anchor_pieces[board_index]
        image[:, 96:192, 96:192] = anchor
        return image

    def _build_observation(self, board_index: int) -> Dict[str, torch.Tensor]:
        board_pieces = self.permutation_list[board_index]
        slot_images = torch.stack([self._get_slot_tensor(piece_id) for piece_id in board_pieces], dim=0)
        return {
            "slot_images": slot_images.clone(),
            "anchor_image": self.anchor_pieces[board_index].clone(),
            "board_image": self._build_board_image(board_index),
            "piece_ids": torch.tensor(board_pieces, dtype=torch.long),
        }

    def get_observations(self) -> List[Dict[str, torch.Tensor]]:
        return [self._build_observation(0), self._build_observation(1)]

    def reset(self, id_pair: Optional[Sequence[int]] = None) -> List[Dict[str, torch.Tensor]]:
        self.image_id, self.image_categories = self._sample_episode_ids(id_pair)
        self._build_episode_pieces(self.image_id)
        self._build_initial_boards()
        self.step_count = 0
        self.completion_step = None
        self.last_metrics = self.get_metrics()
        return self.get_observations()

    def summon_permutation_list(self, swap_num: int = 0, id: Optional[Sequence[int]] = None) -> List[Dict[str, torch.Tensor]]:
        """Compatibility helper for the legacy training style."""
        self.reset(id_pair=id)
        for _ in range(swap_num):
            board_index = random.randint(0, 1)
            slot_a, slot_b = random.sample(range(8), 2)
            self.permutation_list[board_index][slot_a], self.permutation_list[board_index][slot_b] = (
                self.permutation_list[board_index][slot_b],
                self.permutation_list[board_index][slot_a],
            )
        self.last_metrics = self.get_metrics()
        return self.get_observations()

    def _apply_intra_swap(self, board_index: int, source_slot: int, target_slot: int) -> None:
        board = self.permutation_list[board_index]
        board[source_slot], board[target_slot] = board[target_slot], board[source_slot]

    def _apply_cross_swap(self, source_slot_a: int, source_slot_b: int) -> None:
        board_a = self.permutation_list[0]
        board_b = self.permutation_list[1]
        board_a[source_slot_a], board_b[source_slot_b] = board_b[source_slot_b], board_a[source_slot_a]

    def _exact_count(self, board_index: int) -> int:
        count = 0
        for slot_index, piece_id in enumerate(self.permutation_list[board_index]):
            piece = self.piece_records[piece_id]
            if piece.source_board == board_index and piece.source_slot == slot_index:
                count += 1
        return count

    def _ownership_count(self, board_index: int) -> int:
        count = 0
        for piece_id in self.permutation_list[board_index]:
            piece = self.piece_records[piece_id]
            if piece.source_board == board_index:
                count += 1
        return count

    def _encoded_board_permutation(self, board_index: int) -> List[int]:
        permutation = [-1] * self.piece_num
        for slot_index, piece_id in enumerate(self.permutation_list[board_index]):
            piece = self.piece_records[piece_id]
            current_grid_index = SLOT_TO_GRID_INDEX[slot_index]
            source_grid_index = SLOT_TO_GRID_INDEX[piece.source_slot]
            permutation[current_grid_index] = piece.source_board * self.piece_num + source_grid_index

        permutation[self.piece_num // 2] = board_index * self.piece_num + self.piece_num // 2
        return permutation

    def _absolute_local_score(self, board_index: int) -> float:
        permutation = self._encoded_board_permutation(board_index)
        expected = list(range(board_index * self.piece_num, (board_index + 1) * self.piece_num))
        if permutation == expected:
            return self.reward_weights.done_reward

        local_reward = 0.0
        edge_length = int(self.piece_num**0.5)
        hori_pairs = [(i, i + 1) for i in range(self.piece_num) if i % edge_length != edge_length - 1]
        vert_pairs = [(i, i + edge_length) for i in range(self.piece_num - edge_length)]

        for pair_index in range(len(hori_pairs)):
            hori_pair = (permutation[hori_pairs[pair_index][0]], permutation[hori_pairs[pair_index][1]])
            vert_pair = (permutation[vert_pairs[pair_index][0]], permutation[vert_pairs[pair_index][1]])

            if (
                -1 not in hori_pair
                and (hori_pair[0] % self.piece_num, hori_pair[1] % self.piece_num) in hori_pairs
                and hori_pair[0] // self.piece_num == hori_pair[1] // self.piece_num == board_index
            ):
                local_reward += self.reward_weights.pairwise
            if (
                -1 not in vert_pair
                and (vert_pair[0] % self.piece_num, vert_pair[1] % self.piece_num) in vert_pairs
                and vert_pair[0] // self.piece_num == vert_pair[1] // self.piece_num == board_index
            ):
                local_reward += self.reward_weights.pairwise

        for grid_index, piece in enumerate(permutation):
            if piece != -1 and piece % self.piece_num == grid_index and piece // self.piece_num == board_index:
                local_reward += self.reward_weights.cate

        local_reward += self.reward_weights.penalty
        return local_reward

    def _absolute_consistency_score(self, board_index: int) -> float:
        permutation = self._encoded_board_permutation(board_index)
        correct_board_piece_count = 0
        for piece in permutation:
            if piece != -1 and piece // self.piece_num == board_index:
                correct_board_piece_count += 1

        # if correct_board_piece_count == self.piece_num:
        #     return self.reward_weights.consistency_reward
        return correct_board_piece_count * self.reward_weights.consistency

    def _pair_counts(self, board_index: int) -> Tuple[int, int]:
        board = self.permutation_list[board_index]
        hori_pairs = [(0, 1), (1, 2), (3, 4), (5, 6), (6, 7), (4, 7)]
        vert_pairs = [(0, 3), (1, 4), (2, 5), (3, 5), (4, 6), (5, 7)]

        def is_correct_pair(slot_i: int, slot_j: int, mode: str) -> bool:
            piece_i = self.piece_records[board[slot_i]]
            piece_j = self.piece_records[board[slot_j]]
            if piece_i.source_board != board_index or piece_j.source_board != board_index:
                return False

            row_i, col_i = SLOT_TO_GRID_POS[piece_i.source_slot]
            row_j, col_j = SLOT_TO_GRID_POS[piece_j.source_slot]
            if mode == "h":
                return row_i == row_j and col_j == col_i + 1
            return col_i == col_j and row_j == row_i + 1

        hori = sum(is_correct_pair(i, j, "h") for i, j in hori_pairs)
        vert = sum(is_correct_pair(i, j, "v") for i, j in vert_pairs)
        return hori, vert

    def get_metrics(self) -> Dict[str, float]:
        exact_counts = [self._exact_count(board_idx) for board_idx in range(2)]
        ownership_counts = [self._ownership_count(board_idx) for board_idx in range(2)]
        pair_counts = [self._pair_counts(board_idx) for board_idx in range(2)]

        metrics = {
            "board_a_exact": exact_counts[0],
            "board_b_exact": exact_counts[1],
            "board_a_ownership": ownership_counts[0],
            "board_b_ownership": ownership_counts[1],
            "overall_absolute": float(sum(exact_counts)) / 16.0,
            "ownership_accuracy": float(sum(ownership_counts)) / 16.0,
            "both_perfect": float(exact_counts[0] == 8 and exact_counts[1] == 8),
            "board_a_perfect": float(exact_counts[0] == 8),
            "board_b_perfect": float(exact_counts[1] == 8),
            "board_a_horizontal": pair_counts[0][0] / 6.0,
            "board_a_vertical": pair_counts[0][1] / 6.0,
            "board_b_horizontal": pair_counts[1][0] / 6.0,
            "board_b_vertical": pair_counts[1][1] / 6.0,
        }
        return metrics

    def _compute_rewards(
        self,
        failed_coordination: bool,
    ) -> Tuple[List[float], float, List[float], List[float]]:
        local_rewards = [self._absolute_local_score(board_index) for board_index in range(2)]
        consistency_rewards = [self._absolute_consistency_score(board_index) for board_index in range(2)]
        rewards = [
            local_reward + consistency_reward
            for local_reward, consistency_reward in zip(local_rewards, consistency_rewards)
        ]

        if failed_coordination and self.reward_weights.coord_fail != 0:
            rewards = [reward - self.reward_weights.coord_fail for reward in rewards]

        team_reward = float(sum(rewards) / len(rewards))
        return rewards, team_reward, local_rewards, consistency_rewards

    def step(
        self,
        actions: Sequence[Tuple[int, int]],
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[float], bool, Dict[str, float]]:
        if len(actions) != 2:
            raise ValueError("DualBoardEnv.step expects two agents' actions.")

        (source_a, target_a), (source_b, target_b) = actions
        cross_a = target_a == 8
        cross_b = target_b == 8
        cross_executed = cross_a and cross_b
        failed_coordination = cross_a ^ cross_b

        if cross_executed:
            self._apply_cross_swap(source_a, source_b)
        else:
            if not cross_a and source_a != target_a:
                self._apply_intra_swap(0, source_a, target_a)
            if not cross_b and source_b != target_b:
                self._apply_intra_swap(1, source_b, target_b)

        self.step_count += 1
        new_metrics = self.get_metrics()
        rewards, team_reward, local_rewards, consistency_rewards = self._compute_rewards(failed_coordination)

        if new_metrics["both_perfect"] > 0 and self.completion_step is None:
            self.completion_step = self.step_count

        done = False
        if self.training_mode:
            if self.completion_step is not None and (self.step_count - self.completion_step) >= self.cooldown_steps:
                done = True
            elif self.step_count >= self.max_steps:
                done = True
        else:
            done = new_metrics["both_perfect"] > 0 or self.step_count >= self.max_steps

        self.last_metrics = new_metrics
        info = copy.deepcopy(new_metrics)
        info.update(
            {
                "cross_executed": float(cross_executed),
                "failed_coordination": float(failed_coordination),
                "team_reward": team_reward,
                "board_a_local_reward": local_rewards[0],
                "board_b_local_reward": local_rewards[1],
                "board_a_consistency_reward": consistency_rewards[0],
                "board_b_consistency_reward": consistency_rewards[1],
                "step_count": float(self.step_count),
            }
        )
        return self.get_observations(), rewards, done, info

    def show_image(self, image_permutation_list: Optional[List[List[int]]] = None) -> None:
        if image_permutation_list is not None:
            cached_permutation = self.permutation_list
            self.permutation_list = image_permutation_list
        else:
            cached_permutation = None

        for board_index in range(2):
            image = self._build_board_image(board_index).permute(1, 2, 0).numpy()
            cv2.imshow(f"Board {board_index}", image)
        cv2.waitKey(1)

        if cached_permutation is not None:
            self.permutation_list = cached_permutation
