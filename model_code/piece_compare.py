import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3,efficientnet_b0
from model_code.fen_model import Modulator
import torch.nn.functional as F

class piece_compare_model_b3(nn.Module):
    def __init__(self,hidden_1=512,hidden_2=512,out=9,dropout=0.3):
        super().__init__()
        self.ef=efficientnet_b3(weights="DEFAULT")
        self.ef.classifier=nn.Linear(1536,hidden_1)
        self.compare_layer=nn.Sequential(
            nn.Linear(2*hidden_1,hidden_1),
            nn.Dropout(dropout),
            # nn.BatchNorm1d(hidden_1),
            nn.ReLU()
        )

        self.out_layer=nn.Sequential(
            nn.Linear(hidden_1,hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2,out)
        )
    def forward(self,center_piece,outsider_piece):
        center_feature=self.ef(center_piece)
        outsider_feature=self.ef(outsider_piece)

        feature=torch.cat([center_feature,outsider_feature],dim=-1)
        feature=self.compare_layer(feature)
        out=self.out_layer(feature)
        return out


class piece_compare_model_b0(nn.Module):
    def __init__(self,hidden_1=512,hidden_2=512,out=9,dropout=0.1):
        super().__init__()
        self.ef=efficientnet_b0(weights="DEFAULT")
        self.ef.classifier=nn.Linear(1280,hidden_1)
        self.compare_layer=nn.Sequential(
            nn.Linear(2*hidden_1,hidden_1),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU()
        )

        self.out_layer=nn.Sequential(
            nn.Linear(hidden_1,hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2,out)
        )
    def forward(self,center_piece,outsider_piece):
        center_feature=self.ef(center_piece)
        outsider_feature=self.ef(outsider_piece)

        feature=torch.cat([center_feature,outsider_feature],dim=-1)
        feature=self.compare_layer(feature)
        out=self.out_layer(feature)
        return out
    
class DeepuzzleModel_pieceStyle(nn.Module):
    def __init__(self, rgb_dim=1024, color_dim=6, hidden_dim=512, out=9):
        super().__init__()

        # 1. 骨干网络 (共享权重)
        self.rgb_encoder = efficientnet_b3(weights="DEFAULT")
        self.rgb_encoder.classifier = nn.Linear(1536, rgb_dim)

        # 2. 色彩统计特征映射
        self.color_mlp = nn.Sequential(
            nn.Linear(color_dim, 64), nn.LayerNorm(64), nn.ReLU()
        )

        # 总特征维度
        self.feature_dim = rgb_dim + 64

        # 3. 决策层 (9分类)
        # 输入维度: [v_c, v_x, diff, prod] -> feature_dim * 4
        #           [cosine] -> 1
        self.out_layer = nn.Sequential(
            nn.Linear(self.feature_dim * 4 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out),  # 输出 9 个 Logits
        )

    def get_color_stats(self, tensor):
        """计算均值和方差 [B, 6]"""
        mean = tensor.mean(dim=[2, 3])
        std = tensor.std(dim=[2, 3])
        return torch.cat([mean, std], dim=1)

    def forward_one(self, x):
        # 提取 RGB 特征
        f_rgb = self.rgb_encoder(x)
        # 提取色彩特征
        f_color = self.color_mlp(self.get_color_stats(x))
        return torch.cat([f_rgb, f_color], dim=1)

    def forward(self, center_piece, candidate_piece):
        # 1. 提取特征
        v_center = self.forward_one(center_piece)
        v_candidate = self.forward_one(candidate_piece)

        # 2. 特征交互 (核心增强)
        diff = torch.abs(v_center - v_candidate)
        prod = v_center * v_candidate
        cosine = F.cosine_similarity(v_center, v_candidate, dim=1).unsqueeze(1)

        # 3. 拼接
        combined = torch.cat([v_center, v_candidate, diff, prod, cosine], dim=1)

        # 4. 分类
        logits = self.out_layer(combined)
        return logits


class piece_compare_modulator(nn.Module):
    def __init__(self,hidden_1=512,hidden_2=512,out=9,dropout=0.1):
        super().__init__()
        self.ef=Modulator(512,hidden_1)
        self.compare_layer=nn.Sequential(
            nn.Linear(2*hidden_1,hidden_1),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU()
        )

        self.out_layer=nn.Sequential(
            nn.Linear(hidden_1,hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2,out)
        )
    def forward(self,center_piece,outsider_piece):
        center_feature=self.ef(center_piece)
        outsider_feature=self.ef(outsider_piece)

        feature=torch.cat([center_feature,outsider_feature],dim=-1)
        feature=self.compare_layer(feature)
        out=self.out_layer(feature)
        return out