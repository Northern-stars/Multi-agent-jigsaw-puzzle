import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3


class DiscriminatorModel(nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=512):
        super().__init__()
        # 1. 骨干网络 (共享权重)
        # 我们使用 B3，因为它在你的任务中表现较好，且参数量适中
        self.encoder = efficientnet_b3(weights="DEFAULT")
        # 修改输出层，让它输出我们想要的特征维度
        self.encoder.classifier = nn.Linear(1536, feature_dim)

        # 2. 决策层 (MLP)
        # 输入维度是 feature_dim * 4，因为我们拼接了 [u, v, |u-v|, u*v]
        # 这种显式的特征交互能极大提升相似度判断的准确率
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            # 【核心细节】使用 LayerNorm 而不是 BatchNorm
            # BN 会抹平 batch 内的风格差异，导致模型学不到“属于哪张图”的特征
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),  # 输出 1 个 logit
        )

    def forward_one(self, x):
        """提取单张图片的特征"""
        return self.encoder(x)

    def forward(self, center_piece, candidate_piece):
        # 1. 分别提取特征 (权重共享)
        # center_piece: [B, 3, 96, 96] -> [B, 1024]
        v_center = self.forward_one(center_piece)
        # candidate_piece: [B, 3, 96, 96] -> [B, 1024]
        v_candidate = self.forward_one(candidate_piece)

        # 2. 特征交互 (Interaction)
        # 绝对差 (关注不同点)
        diff = torch.abs(v_center - v_candidate)
        # 点积 (关注相似点/共现特征)
        prod = v_center * v_candidate

        # 3. 拼接所有信息
        # [B, 4096]
        combined = torch.cat([v_center, v_candidate, diff, prod], dim=1)

        # 4. 分类输出 (Logits)
        logits = self.classifier(combined)
        return logits
