import torch
import numpy as np
import random
from torch.utils.data import Dataset


class SiameseNoiseDataset(Dataset):
    def __init__(self, data_path):
        """
        data_path: .npy 文件路径
        """
        print(f"Loading data from {data_path}...")
        self.data = np.load(data_path)
        self.num_samples = len(self.data)

        print(f"Loaded {self.num_samples} images.")

    def __len__(self):
        return self.num_samples

    def cut_patches(self, image_np):
        """把 288x288 切成 9 个 96x96 的块，并返回 RGB 列表"""
        patches_rgb = []

        for r in range(3):
            for c in range(3):
                patch = image_np[r * 96 : (r + 1) * 96, c * 96 : (c + 1) * 96, :]

                # RGB Tensor: [3, 96, 96], 归一化到 0-1 更有利于噪声提取稳定性
                # 假设原始数据是 0-255 uint8 或 float
                patch_tensor = torch.tensor(patch).permute(2, 0, 1).float()
                if patch_tensor.max() > 1.0:
                    patch_tensor /= 255.0

                patches_rgb.append(patch_tensor)

        return patches_rgb

    def __getitem__(self, index):
        # 1. 确定主图 (Host) 和 异图 (Guest)
        host_idx = index
        guest_idx = random.randint(0, self.num_samples - 1)
        while guest_idx == host_idx:
            guest_idx = random.randint(0, self.num_samples - 1)

        # 2. 切片 (RGB)
        host_rgb = self.cut_patches(self.data[host_idx])
        guest_rgb = self.cut_patches(self.data[guest_idx])

        # 3. 确定 Center (参考基准 = Host 中心块)
        c_rgb = host_rgb[4]

        # 4. 构造正负样本 (50% 概率)
        if random.random() > 0.5:
            # === 正样本 (Label = 1) ===
            cand_idx = random.randint(0, 8)
            x_rgb = host_rgb[cand_idx]
            label = 1.0
        else:
            # === 负样本 (Label = 0) ===
            cand_idx = random.randint(0, 8)
            x_rgb = guest_rgb[cand_idx]
            label = 0.0

        return (
            c_rgb,
            x_rgb,
            torch.tensor([label], dtype=torch.float32),
        )
