import torch
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data import Dataset


class SiameseDataset(Dataset):
    def __init__(self, data_path):
        """
        data_path: .npy 文件路径，shape 应该是 (N, 288, 288, 3)
        假定 train_x 里存的是完整的、未打乱的原图 (Ground Truth)。
        """
        print(f"Loading data from {data_path}...")
        self.data = np.load(data_path)
        self.num_samples = len(self.data)
        print(f"Loaded {self.num_samples} images.")

    def __len__(self):
        return self.num_samples

    def cut_patches(self, image_np):
        """把 288x288 切成 9 个 96x96 的块"""
        patches = []
        for r in range(3):
            for c in range(3):
                patch = image_np[r * 96 : (r + 1) * 96, c * 96 : (c + 1) * 96, :]
                # 转为 Tensor [C, H, W]
                patch_tensor = torch.tensor(patch).permute(2, 0, 1).float()
                patches.append(patch_tensor)
        return patches

    def __getitem__(self, index):

        # 1. 确定主图 (Host) 和 异图 (Guest)
        host_idx = index
        guest_idx = random.randint(0, self.num_samples - 1)
        while guest_idx == host_idx:
            guest_idx = random.randint(0, self.num_samples - 1)

        # 2. 切片
        host_patches = self.cut_patches(self.data[host_idx])
        guest_patches = self.cut_patches(self.data[guest_idx])

        # 3. 确定 Center (参考基准)
        # 固定取 Host 的第 4 号位 (中心)
        center_tensor = host_patches[4]

        # 4. 构造正负样本 (50% 概率)
        if random.random() > 0.5:
            # === 正样本 (Label = 1) ===
            # Candidate 来自 Host 的任意位置 (包括 4 自己也可以，或者强制不选 4)
            # 这里我们选任意位置，教模型：只要是这张图的，都算对
            cand_idx = random.randint(0, 8)
            candidate_tensor = host_patches[cand_idx]
            label = 1.0
        else:
            # === 负样本 (Label = 0) ===
            # Candidate 来自 Guest 的任意位置
            cand_idx = random.randint(0, 8)
            candidate_tensor = guest_patches[cand_idx]
            label = 0.0

        return (
            center_tensor,
            candidate_tensor,
            torch.tensor([label], dtype=torch.float32),
        )
