import torch
import torch.nn.functional as F
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

        # 预定义拉普拉斯核 (用于提取高频噪点)
        kernel = torch.tensor(
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.laplacian_kernel = kernel.repeat(3, 1, 1, 1)  # [3, 1, 3, 3] 适配 RGB
        print(f"Loaded {self.num_samples} images.")

    def __len__(self):
        return self.num_samples

    def get_high_freq(self, image_tensor):
        """
        输入: [3, 96, 96]
        输出: [3, 96, 96] (高频分量/噪点图)
        """
        # 扩展维度以适配 conv2d: [1, 3, 96, 96]
        x = image_tensor.unsqueeze(0)

        # 卷积提取高频 (padding=1 保持尺寸不变)
        # groups=3 表示对每个通道独立卷积
        high_freq = F.conv2d(x, self.laplacian_kernel, padding=1, groups=3)

        # 取绝对值并归一化 (防止数值过大)
        return torch.abs(high_freq).squeeze(0) / 4.0

    def cut_patches(self, image_np):
        """把 288x288 切成 9 个 96x96 的块，并返回 (RGB, Noise) 对"""
        patches_rgb = []
        patches_noise = []

        for r in range(3):
            for c in range(3):
                patch = image_np[r * 96 : (r + 1) * 96, c * 96 : (c + 1) * 96, :]

                # RGB Tensor: [3, 96, 96], 归一化到 0-1 更有利于噪声提取稳定性
                # 假设原始数据是 0-255 uint8 或 float
                patch_tensor = torch.tensor(patch).permute(2, 0, 1).float()
                if patch_tensor.max() > 1.0:
                    patch_tensor /= 255.0

                # 提取噪点
                noise_tensor = self.get_high_freq(patch_tensor)

                patches_rgb.append(patch_tensor)
                patches_noise.append(noise_tensor)

        return patches_rgb, patches_noise

    def __getitem__(self, index):
        # 1. 确定主图 (Host) 和 异图 (Guest)
        host_idx = index
        guest_idx = random.randint(0, self.num_samples - 1)
        while guest_idx == host_idx:
            guest_idx = random.randint(0, self.num_samples - 1)

        # 2. 切片 (同时获取 RGB 和 Noise)
        host_rgb, host_noise = self.cut_patches(self.data[host_idx])
        guest_rgb, guest_noise = self.cut_patches(self.data[guest_idx])

        # 3. 确定 Center (参考基准 = Host 中心块)
        c_rgb = host_rgb[4]
        c_noise = host_noise[4]

        # 4. 构造正负样本 (50% 概率)
        if random.random() > 0.5:
            # === 正样本 (Label = 1) ===
            cand_idx = random.randint(0, 8)
            x_rgb = host_rgb[cand_idx]
            x_noise = host_noise[cand_idx]
            label = 1.0
        else:
            # === 负样本 (Label = 0) ===
            cand_idx = random.randint(0, 8)
            x_rgb = guest_rgb[cand_idx]
            x_noise = guest_noise[cand_idx]
            label = 0.0

        return (
            c_rgb,
            c_noise,
            x_rgb,
            x_noise,
            torch.tensor([label], dtype=torch.float32),
        )
