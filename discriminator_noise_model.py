import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3


class NoiseExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 专门用于提取高频噪点特征的小型 CNN
        # 使用 InstanceNorm 而不是 BatchNorm，保证单张图片的对比度信息不被 Batch 抹平
        self.net = nn.Sequential(
            # Input: [B, 3, 96, 96] (High-Freq Image)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> [B, 32, 48, 48]
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> [B, 64, 24, 24]
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(
                64, 128, kernel_size=3, stride=2, padding=1
            ),  # -> [B, 128, 12, 12]
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # -> [B, 128, 1, 1]
            nn.Flatten(),  # -> [B, 128]
        )

    def forward(self, x):
        return self.net(x)


class DiscriminatorModel(nn.Module):
    def __init__(self, rgb_dim=1024, noise_dim=128, hidden_dim=512):
        super().__init__()

        # 1. RGB 语义流 (EfficientNet)
        self.rgb_encoder = efficientnet_b3(weights="DEFAULT")
        # 修改输出层
        self.rgb_encoder.classifier = nn.Linear(1536, rgb_dim)

        # 2. 噪声流 (Custom CNN)
        self.noise_encoder = NoiseExtractor()  # 输出维度固定为 128

        # 3. 融合后的特征维度
        self.feature_dim = rgb_dim + noise_dim

        # 4. 决策层 (输入是 4 倍特征维度: u, v, |u-v|, u*v)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            # 使用 LayerNorm 归一化特征分布，不依赖 Batch
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward_one(self, rgb, noise):
        # 提取 RGB 特征
        f_rgb = self.rgb_encoder(rgb)
        # 提取 噪声 特征
        f_noise = self.noise_encoder(noise)
        # 拼接
        return torch.cat([f_rgb, f_noise], dim=1)

    def forward(self, center_rgb, center_noise, cand_rgb, cand_noise):
        # 1. 提取中心块特征
        v_center = self.forward_one(center_rgb, center_noise)

        # 2. 提取候选块特征
        v_candidate = self.forward_one(cand_rgb, cand_noise)

        # 3. 特征交互 (Interaction)
        # 绝对差 (关注不同点)
        diff = torch.abs(v_center - v_candidate)
        # 点积 (关注共同点)
        prod = v_center * v_candidate

        # 4. 拼接所有信息
        combined = torch.cat([v_center, v_candidate, diff, prod], dim=1)

        # 5. 分类
        logits = self.classifier(combined)
        return logits
