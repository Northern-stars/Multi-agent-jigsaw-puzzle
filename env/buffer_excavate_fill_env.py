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
class BufferExcavateFillRewardConfig:
    pairwise: float = 0.2
    cate: float = 0.8
    consistency: float = 0.5
    done_reward: float = 1000.0
    consistency_reward: float = 200.0
    digger_remove_wrong: float = 1.0
    digger_remove_right: float = -1.0
    step_penalty: float = -0.05


class BufferExcavateFillEnv:
    """Two-board digger/filler environment for fixed-center 3x3 jigsaw puzzles."""

    def __init__(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        image_num: int = 2,
        piece_num: int = 9,
        dig_per_board: int = 2,
        fill_strategy: str = "parallel",
        initial_swap_num: int = 5,
        epsilon: float = 0.3,
        epsilon_gamma: float = 0.998,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        reward_config: Optional[BufferExcavateFillRewardConfig] = None,
        vacancy_token: Optional[torch.Tensor] = None,
    ) -> None:
        if image_num != 2:
            raise ValueError("BufferExcavateFillEnv currently expects exactly two boards.")
        if piece_num != 9:
            raise ValueError("Only fixed-center 3x3 puzzles are supported.")
        if fill_strategy not in {"seq", "alt", "greedy", "parallel"}:
            raise ValueError("fill_strategy must be one of: seq, alt, greedy, parallel.")

        self.image = train_x
        self.label = train_y
        self.sample_number = train_x.shape[0]
        self.image_num = image_num
        self.piece_num = piece_num
        self.dig_per_board = dig_per_board
        self.buffer_size = image_num * dig_per_board
        self.fill_strategy = fill_strategy
        self.initial_swap_num = initial_swap_num
        self.epsilon = epsilon
        self.epsilon_gamma = epsilon_gamma
        self.device = device
        self.reward_config = reward_config or BufferExcavateFillRewardConfig()
        self.vacancy_token = (
            vacancy_token.detach().float().cpu()
            if vacancy_token is not None
            else torch.zeros(3, 96, 96, dtype=torch.float32)
        )

        self.image_id: List[int] = []
        self.boards: List[List[int]] = []
        self.empty_masks: List[List[bool]] = []
        self.buffer: List[Dict[str, object]] = []
        self.digger_rewards: List[float] = []
        self.filler_rewards: List[float] = []
        self.stage = "init"

        self._piece_cache: Dict[int, torch.Tensor] = {}
        self._anchor_cache: Dict[int, torch.Tensor] = {}
        self._dataset_piece_cache: Dict[int, Tuple[List[torch.Tensor], torch.Tensor]] = {}
        self._last_candidates: List[Dict[str, object]] = []

    def _normalize_image_ids(self, image_ids: Optional[Union[int, Sequence[int]]]) -> List[int]:
        if image_ids is None:
            return random.sample(range(self.sample_number), k=self.image_num)
        if isinstance(image_ids, int):
            first = int(image_ids)
            second = random.randrange(self.sample_number)
            while second == first and self.sample_number > 1:
                second = random.randrange(self.sample_number)
            return [first, second]
        normalized = [int(item) for item in image_ids]
        if len(normalized) != self.image_num:
            raise ValueError(f"Expected {self.image_num} image ids, got {len(normalized)}.")
        return normalized

    def _decode_label(self, dataset_index: int) -> List[int]:
        permutation_raw = list(np.argmax(self.label[dataset_index], axis=1))
        for idx, value in enumerate(permutation_raw):
            if value >= self.piece_num // 2:
                permutation_raw[idx] += 1
        permutation_raw.insert(self.piece_num // 2, self.piece_num // 2)
        return permutation_raw

    def _load_dataset_pieces(self, dataset_index: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        if dataset_index in self._dataset_piece_cache:
            return self._dataset_piece_cache[dataset_index]

        image_np = np.asarray(self.image[dataset_index], dtype=np.float32)
        image_raw = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).contiguous()
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

        permutation = self._decode_label(dataset_index)
        ordered_fragments: List[Optional[torch.Tensor]] = [None] * self.piece_num
        for grid_index, fragment in zip(permutation, fragments):
            ordered_fragments[grid_index] = fragment

        if any(fragment is None for fragment in ordered_fragments):
            raise RuntimeError("Failed to decode all 3x3 puzzle fragments from labels.")

        movable = [ordered_fragments[grid_index] for grid_index in SLOT_TO_GRID_INDEX]
        anchor = ordered_fragments[self.piece_num // 2]
        self._dataset_piece_cache[dataset_index] = (movable, anchor)
        return movable, anchor

    def _prepare_episode_pieces(self) -> None:
        self._piece_cache = {}
        self._anchor_cache = {}
        for board_id, dataset_index in enumerate(self.image_id):
            movable, anchor = self._load_dataset_pieces(dataset_index)
            self._anchor_cache[board_id] = anchor
            for slot_index, grid_index in enumerate(SLOT_TO_GRID_INDEX):
                absolute_piece_id = board_id * self.piece_num + grid_index
                self._piece_cache[absolute_piece_id] = movable[slot_index]

    def reset(
        self,
        image_ids: Optional[Union[int, Sequence[int]]] = None,
        swap_num: Optional[int] = None,
    ) -> Dict[str, object]:
        self.image_id = self._normalize_image_ids(image_ids)
        self._prepare_episode_pieces()

        movable_piece_ids = [
            board_id * self.piece_num + grid_index
            for board_id in range(self.image_num)
            for grid_index in SLOT_TO_GRID_INDEX
        ]
        total_swaps = self.initial_swap_num if swap_num is None else int(swap_num)
        for _ in range(total_swaps):
            idx_a, idx_b = random.sample(range(len(movable_piece_ids)), k=2)
            movable_piece_ids[idx_a], movable_piece_ids[idx_b] = movable_piece_ids[idx_b], movable_piece_ids[idx_a]

        self.boards = [
            movable_piece_ids[board_id * (self.piece_num - 1) : (board_id + 1) * (self.piece_num - 1)]
            for board_id in range(self.image_num)
        ]
        self.empty_masks = [[False for _ in range(self.piece_num - 1)] for _ in range(self.image_num)]
        self.buffer = []
        self.digger_rewards = []
        self.filler_rewards = []
        self.stage = "digger"
        self._last_candidates = []
        return self.get_state()

    def get_state(self) -> Dict[str, object]:
        return {
            "image_id": copy.deepcopy(self.image_id),
            "boards": copy.deepcopy(self.boards),
            "empty_masks": copy.deepcopy(self.empty_masks),
            "buffer": copy.deepcopy(self.buffer),
            "stage": self.stage,
        }

    def _build_board_image(self, board_id: int, board_slots: Optional[Sequence[int]] = None) -> torch.Tensor:
        slots = self.boards[board_id] if board_slots is None else board_slots
        image = torch.zeros(3, 288, 288, dtype=torch.float32)
        for slot_index, piece_id in enumerate(slots):
            grid_index = SLOT_TO_GRID_INDEX[slot_index]
            row, col = divmod(grid_index, 3)
            patch = self.vacancy_token if piece_id == -1 else self._piece_cache[int(piece_id)]
            image[:, row * 96 : (row + 1) * 96, col * 96 : (col + 1) * 96] = patch

        image[:, 96:192, 96:192] = self._anchor_cache[board_id]
        return image

    def get_board_image(self, board_id: int) -> torch.Tensor:
        return self._build_board_image(board_id).unsqueeze(0).to(self.device)

    def get_digger_observation(self, board_id: int) -> Dict[str, object]:
        return {
            "board_id": board_id,
            "board_image": self.get_board_image(board_id),
            "empty_mask": torch.tensor(self.empty_masks[board_id], dtype=torch.bool, device=self.device).unsqueeze(0),
            "slots": copy.deepcopy(self.boards[board_id]),
        }

    def get_digger_mask(self, board_id: int) -> torch.Tensor:
        valid = [not is_empty for is_empty in self.empty_masks[board_id]]
        return torch.tensor(valid, dtype=torch.bool, device=self.device)

    def _piece_correct_at(self, piece_id: int, board_id: int, slot_id: int) -> bool:
        gt_board = piece_id // self.piece_num
        gt_grid = piece_id % self.piece_num
        gt_slot = GRID_TO_SLOT_INDEX.get(gt_grid, None)
        return gt_board == board_id and gt_slot == slot_id

    def step_digger(self, board_id: int, slot_id: int) -> Tuple[Dict[str, object], float, bool, Dict[str, object]]:
        if self.stage not in {"digger", "init"}:
            raise RuntimeError("Digger can only act before the filler stage starts.")
        if slot_id < 0 or slot_id >= self.piece_num - 1:
            raise ValueError(f"slot_id must be in [0, {self.piece_num - 2}].")
        if self.empty_masks[board_id][slot_id]:
            raise ValueError(f"Board {board_id} slot {slot_id} is already empty.")

        piece_id = self.boards[board_id][slot_id]
        gt_board = piece_id // self.piece_num
        gt_grid = piece_id % self.piece_num
        gt_slot = GRID_TO_SLOT_INDEX.get(gt_grid, None)
        is_correct = self._piece_correct_at(piece_id, board_id, slot_id)
        reward = (
            self.reward_config.digger_remove_right
            if is_correct
            else self.reward_config.digger_remove_wrong
        ) + self.reward_config.step_penalty

        self.boards[board_id][slot_id] = -1
        self.empty_masks[board_id][slot_id] = True
        self.buffer.append(
            {
                "piece_id": int(piece_id),
                "piece_image": self._piece_cache[int(piece_id)],
                "source_board": board_id,
                "source_slot": slot_id,
                "gt_board": int(gt_board),
                "gt_slot": gt_slot,
            }
        )
        self.digger_rewards.append(float(reward))

        digger_done = len(self.buffer) >= self.buffer_size
        if digger_done:
            self.stage = "filler"
        info = {
            "piece_id": int(piece_id),
            "is_correct_removed": bool(is_correct),
            "buffer_size": len(self.buffer),
            "digger_done": digger_done,
        }
        return self.get_digger_observation(board_id), float(reward), digger_done, info

    def get_filler_observation(self) -> Dict[str, object]:
        board_images = torch.cat([self.get_board_image(board_id) for board_id in range(self.image_num)], dim=0)
        if self.buffer:
            buffer_pieces = torch.stack([item["piece_image"] for item in self.buffer]).to(self.device)
        else:
            buffer_pieces = torch.empty(0, 3, 96, 96, device=self.device)
        return {
            "board_images": board_images,
            "empty_masks": torch.tensor(self.empty_masks, dtype=torch.bool, device=self.device),
            "buffer_pieces": buffer_pieces,
            "buffer": copy.deepcopy(self.buffer),
            "boards": copy.deepcopy(self.boards),
        }

    def get_empty_slots(self, strategy: Optional[str] = None) -> List[Tuple[int, int]]:
        target_strategy = strategy or self.fill_strategy
        per_board = [
            [(board_id, slot_id) for slot_id, is_empty in enumerate(self.empty_masks[board_id]) if is_empty]
            for board_id in range(self.image_num)
        ]
        if target_strategy == "alt":
            ordered: List[Tuple[int, int]] = []
            max_len = max((len(items) for items in per_board), default=0)
            for idx in range(max_len):
                for board_id in range(self.image_num):
                    if idx < len(per_board[board_id]):
                        ordered.append(per_board[board_id][idx])
            return ordered
        return [slot for board_slots in per_board for slot in board_slots]

    def _candidate_board_slots(self, board_id: int, slot_id: int, buffer_index: int) -> List[int]:
        candidate_slots = copy.deepcopy(self.boards[board_id])
        candidate_slots[slot_id] = int(self.buffer[buffer_index]["piece_id"])
        return candidate_slots

    def _single_candidate(self, board_id: int, slot_id: int, buffer_index: int) -> Dict[str, object]:
        candidate_slots = self._candidate_board_slots(board_id, slot_id, buffer_index)
        candidate_image = self._build_board_image(board_id, candidate_slots).unsqueeze(0).to(self.device)
        return {
            "type": "single",
            "board_id": board_id,
            "slot_id": slot_id,
            "buffer_index": buffer_index,
            "piece_id": int(self.buffer[buffer_index]["piece_id"]),
            "candidate_image": candidate_image,
            "assignment": [{"board_id": board_id, "slot_id": slot_id, "buffer_index": buffer_index}],
        }

    def _assignment_candidate(self, assignment: Sequence[Dict[str, int]]) -> Dict[str, object]:
        candidate_boards = copy.deepcopy(self.boards)
        for item in assignment:
            piece_id = int(self.buffer[item["buffer_index"]]["piece_id"])
            candidate_boards[item["board_id"]][item["slot_id"]] = piece_id

        candidate_images = torch.cat(
            [
                self._build_board_image(board_id, candidate_boards[board_id]).unsqueeze(0)
                for board_id in range(self.image_num)
            ],
            dim=0,
        ).to(self.device)
        return {
            "type": "assignment",
            "assignment": copy.deepcopy(list(assignment)),
            "candidate_images": candidate_images,
        }

    def get_filler_candidates(self, fill_strategy: Optional[str] = None) -> List[Dict[str, object]]:
        target_strategy = fill_strategy or self.fill_strategy
        empty_slots = self.get_empty_slots(target_strategy)
        if not empty_slots or not self.buffer:
            self._last_candidates = []
            return []

        candidates: List[Dict[str, object]] = []
        if target_strategy in {"seq", "alt"}:
            board_id, slot_id = empty_slots[0]
            candidates = [
                self._single_candidate(board_id, slot_id, buffer_index)
                for buffer_index in range(len(self.buffer))
            ]
        elif target_strategy == "greedy":
            candidates = [
                self._single_candidate(board_id, slot_id, buffer_index)
                for board_id, slot_id in empty_slots
                for buffer_index in range(len(self.buffer))
            ]
        elif target_strategy == "parallel":
            if len(empty_slots) != len(self.buffer):
                candidates = [
                    self._single_candidate(board_id, slot_id, buffer_index)
                    for board_id, slot_id in empty_slots
                    for buffer_index in range(len(self.buffer))
                ]
            else:
                for permutation in itertools.permutations(range(len(self.buffer)), len(empty_slots)):
                    assignment = [
                        {"board_id": board_id, "slot_id": slot_id, "buffer_index": buffer_index}
                        for (board_id, slot_id), buffer_index in zip(empty_slots, permutation)
                    ]
                    candidates.append(self._assignment_candidate(assignment))

        self._last_candidates = candidates
        return candidates

    def get_candidate_mask(self) -> torch.Tensor:
        if not self._last_candidates:
            self.get_filler_candidates()
        return torch.ones(len(self._last_candidates), dtype=torch.bool, device=self.device)

    def _apply_assignment(self, assignment: Sequence[Dict[str, int]]) -> None:
        used_buffer_indices = []
        for item in assignment:
            board_id = item["board_id"]
            slot_id = item["slot_id"]
            buffer_index = item["buffer_index"]
            if not self.empty_masks[board_id][slot_id]:
                raise ValueError(f"Board {board_id} slot {slot_id} is not empty.")
            self.boards[board_id][slot_id] = int(self.buffer[buffer_index]["piece_id"])
            self.empty_masks[board_id][slot_id] = False
            used_buffer_indices.append(buffer_index)

        for buffer_index in sorted(set(used_buffer_indices), reverse=True):
            self.buffer.pop(buffer_index)

    def step_filler(self, candidate_index: int) -> Tuple[Dict[str, object], float, bool, Dict[str, object]]:
        if self.stage != "filler":
            raise RuntimeError("Filler can only act after the digger stage is complete.")
        candidates = self._last_candidates if self._last_candidates else self.get_filler_candidates()
        if candidate_index < 0 or candidate_index >= len(candidates):
            raise ValueError(f"candidate_index must be in [0, {len(candidates) - 1}].")

        candidate = candidates[candidate_index]
        before_score = self.get_total_score()
        self._apply_assignment(candidate["assignment"])
        after_score = self.get_total_score()
        reward = after_score - before_score
        self.filler_rewards.append(float(reward))
        self._last_candidates = []

        done = len(self.buffer) == 0 or not self.get_empty_slots()
        if done:
            self.stage = "done"
        info = {
            "candidate": copy.deepcopy({key: value for key, value in candidate.items() if key not in {"candidate_image", "candidate_images"}}),
            "score_before": float(before_score),
            "score_after": float(after_score),
            "done": done,
        }
        return self.get_filler_observation(), float(reward), done, info

    def _board_full_grid(self, board_id: int, slots: Optional[Sequence[int]] = None) -> List[int]:
        board_slots = self.boards[board_id] if slots is None else slots
        full_grid = [-1] * self.piece_num
        for slot_id, piece_id in enumerate(board_slots):
            grid_index = SLOT_TO_GRID_INDEX[slot_id]
            full_grid[grid_index] = int(piece_id) if piece_id != -1 else -1
        full_grid[self.piece_num // 2] = board_id * self.piece_num + self.piece_num // 2
        return full_grid

    def get_board_score(self, board_id: int, slots: Optional[Sequence[int]] = None) -> float:
        full_grid = self._board_full_grid(board_id, slots)
        cate = 0
        consistency = 0.0
        for grid_index, piece_id in enumerate(full_grid):
            if piece_id == -1:
                continue
            is_same_board = piece_id // self.piece_num == board_id
            consistency += float(is_same_board)
            cate += int(is_same_board and piece_id % self.piece_num == grid_index)

        pairwise = 0
        for grid_a, grid_b in HORIZONTAL_GRID_PAIRS + VERTICAL_GRID_PAIRS:
            piece_a, piece_b = full_grid[grid_a], full_grid[grid_b]
            if piece_a == -1 or piece_b == -1:
                continue
            if (
                piece_a // self.piece_num == board_id
                and piece_b // self.piece_num == board_id
                and piece_a % self.piece_num == grid_a
                and piece_b % self.piece_num == grid_b
            ):
                pairwise += 1

        if cate == self.piece_num:
            return self.reward_config.done_reward
        if consistency == self.piece_num:
            consistency_score = self.reward_config.consistency_reward
        else:
            consistency_score = consistency * self.reward_config.consistency
        return cate * self.reward_config.cate + pairwise * self.reward_config.pairwise + consistency_score

    def get_total_score(self) -> float:
        return float(sum(self.get_board_score(board_id) for board_id in range(self.image_num)))

    def get_metrics(self) -> Dict[str, float]:
        cate_counts = []
        consistency_counts = []
        pair_counts = []
        done_counts = []
        for board_id in range(self.image_num):
            full_grid = self._board_full_grid(board_id)
            cate = sum(
                int(piece_id != -1 and piece_id // self.piece_num == board_id and piece_id % self.piece_num == grid_index)
                for grid_index, piece_id in enumerate(full_grid)
            )
            consistency = sum(
                int(piece_id != -1 and piece_id // self.piece_num == board_id)
                for piece_id in full_grid
            )
            pairs = 0
            for grid_a, grid_b in HORIZONTAL_GRID_PAIRS + VERTICAL_GRID_PAIRS:
                piece_a, piece_b = full_grid[grid_a], full_grid[grid_b]
                pairs += int(
                    piece_a != -1
                    and piece_b != -1
                    and piece_a // self.piece_num == board_id
                    and piece_b // self.piece_num == board_id
                    and piece_a % self.piece_num == grid_a
                    and piece_b % self.piece_num == grid_b
                )
            cate_counts.append(cate)
            consistency_counts.append(consistency)
            pair_counts.append(pairs)
            done_counts.append(int(cate == self.piece_num))

        movable_num = self.piece_num - 1
        pair_num = len(HORIZONTAL_GRID_PAIRS) + len(VERTICAL_GRID_PAIRS)
        return {
            "score": self.get_total_score(),
            "done_accuracy": float(np.mean(done_counts)),
            "cate_accuracy": float(np.mean([(cate - 1) / movable_num for cate in cate_counts])),
            "consistency_accuracy": float(np.mean([(count - 1) / movable_num for count in consistency_counts])),
            "pair_accuracy": float(np.mean([count / pair_num for count in pair_counts])),
            "buffer_size": float(len(self.buffer)),
            "empty_slots": float(sum(sum(mask) for mask in self.empty_masks)),
            "digger_reward": float(sum(self.digger_rewards)),
            "filler_reward": float(sum(self.filler_rewards)),
        }

    def show_image(self, window_prefix: str = "BufferExcavateFill") -> None:
        for board_id in range(self.image_num):
            image_tensor = self.get_board_image(board_id)
            image = image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
            cv2.imshow(f"{window_prefix} Board {board_id}", image)
        cv2.waitKey(1)
