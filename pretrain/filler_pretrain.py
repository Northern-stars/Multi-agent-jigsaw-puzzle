import os
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model_code.filler_model import FillerModel


class FillerPretrainDataset(Dataset):
    def __init__(self, train_x: np.ndarray, train_y: np.ndarray, sample_size: int = 5000) -> None:
        self.train_x = train_x
        self.train_y = train_y
        self.sample_size = sample_size
        self.piece_num = 9
        self.slot_map = [0, 1, 2, 3, 5, 6, 7, 8]

    def __len__(self) -> int:
        return self.sample_size

    def _decode(self, idx: int) -> List[int]:
        permutation_raw = list(np.argmax(self.train_y[idx], axis=1))
        for j, value in enumerate(permutation_raw):
            if value >= 4:
                permutation_raw[j] += 1
        permutation_raw.insert(4, 4)
        return permutation_raw

    def __getitem__(self, idx: int):
        image_idx = random.randrange(self.train_x.shape[0])
        image = torch.tensor(self.train_x[image_idx], dtype=torch.float32).permute(2, 0, 1).contiguous()
        fragments = [
            image[:, 0:96, 0:96],
            image[:, 0:96, 96:192],
            image[:, 0:96, 192:288],
            image[:, 96:192, 0:96],
            image[:, 96:192, 96:192],
            image[:, 96:192, 192:288],
            image[:, 192:288, 0:96],
            image[:, 192:288, 96:192],
            image[:, 192:288, 192:288],
        ]
        permutation = self._decode(image_idx)
        ordered = [None] * self.piece_num
        for grid_index, fragment in zip(permutation, fragments):
            ordered[grid_index] = fragment
        slot_order = list(self.slot_map)
        reward = 1.0
        if random.random() < 0.5:
            idx_a, idx_b = random.sample(range(8), k=2)
            slot_order[idx_a], slot_order[idx_b] = slot_order[idx_b], slot_order[idx_a]
            reward = 0.0
        board = torch.zeros(3, 288, 288, dtype=torch.float32)
        for slot_id, grid_index in enumerate(self.slot_map):
            row, col = divmod(grid_index, 3)
            source_grid = slot_order[slot_id]
            board[:, row * 96 : (row + 1) * 96, col * 96 : (col + 1) * 96] = ordered[source_grid]
        board[:, 96:192, 96:192] = ordered[4]
        return board, reward


def train_filler_pretrain(
    train_x: np.ndarray,
    train_y: np.ndarray,
    model_name: str = "modulator",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-4,
    save_path: str = "model/filler_pretrain.pth",
) -> FillerModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = FillerPretrainDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model = FillerModel(model_name=model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for board, reward in loader:
            board = board.to(device)
            reward = reward.to(device)
            score = model(board)
            loss = criterion(score, reward)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return model
