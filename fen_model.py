import torch.nn as nn
import torch
from torchvision.models import efficientnet_b0,efficientnet_b3

class fen_model(nn.Module):
    def __init__(self, hidden_size1, hidden_size2):
        super(fen_model, self).__init__()
        self.ef = efficientnet_b3(weights="DEFAULT")
        self.ef.classifier = nn.Linear(1536, 1024)

        self.contrast_fc_hori = nn.Linear(2048, 1024)
        self.contrast_fc_vert = nn.Linear(2048, 1024)

        self.fc1 = nn.Linear(1024*12, hidden_size1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_size1)
        self.do = nn.Dropout1d(p=0.1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)

        # 定义 index 对
        self.hori_set = [(i, i+1) for i in range(9) if i % 3 != 2]
        self.vert_set = [(i, i+3) for i in range(6)]

    def forward(self, image):
        B, C, H, W = image.shape   # 假设输入 (B, 3, 288, 288)

        # ---- 1. 切片并 reshape 成 batch ----
        patches = image.unfold(2, 96, 96).unfold(3, 96, 96)
        # patches: (B, C, 3, 3, 96, 96)
        patches = patches.permute(0,2,3,1,4,5).contiguous()
        patches = patches.view(B*9, C, 96, 96)   # (B*9, 3, 96, 96)

        # ---- 2. 一次性送进 ef ----
        feats = self.ef(patches)   # (B*9, 1024)
        feats = feats.view(B, 9, 1024)  # (B, 9, 1024)

        # ---- 3. 构建横向对 ----
        hori_pairs = torch.stack([torch.cat([feats[:, i, :], feats[:, j, :]], dim=-1) 
                                  for (i,j) in self.hori_set], dim=1)  # (B, #pairs, 1024)
        hori_feats = self.contrast_fc_hori(hori_pairs)  # (B, #pairs, 512)

        # ---- 4. 构建纵向对 ----
        vert_pairs = torch.stack([torch.cat([feats[:, i, :], feats[:, j, :]], dim=-1) 
                                  for (i,j) in self.vert_set], dim=1)  # (B, #pairs, 1024)
        vert_feats = self.contrast_fc_vert(vert_pairs)  # (B, #pairs, 512)

        # ---- 5. 拼接所有特征 ----
        feature_tensor = torch.cat([hori_feats, vert_feats], dim=1)  # (B, 12, 512)
        feature_tensor = feature_tensor.view(B, -1)   # (B, 12*512)

        # ---- 6. 全连接部分 ----
        x = self.do(feature_tensor)
        x = self.fc1(x)
        x = self.do(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class central_fen_model(nn.Module):
    def __init__(self, hidden_size1,hidden_size2,dropout=0.1):
        super().__init__()
        self.ef=efficientnet_b0()
        self.ef.classifier=nn.Linear(1280,hidden_size1)
        self.contract_fc=nn.Linear(2*hidden_size1,hidden_size1)
        self.fc1=nn.Linear(8*hidden_size1,hidden_size1)
        self.relu=nn.ReLU()
        self.bn=nn.BatchNorm1d(hidden_size1)

        self.out_layer=nn.Sequential(
            nn.Linear(hidden_size1,hidden_size2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size2,hidden_size2)
        )
    
    def forward(self, image):
        # 输入: [batch, 3, 288, 288]
        B, C, H, W = image.shape
        # 切分为 9 个碎片 [B, 3, 3, 3, 96, 96]
        patches = image.unfold(2, 96, 96).unfold(3, 96, 96)  # [B, C, 3, 3, 96, 96]
        patches = patches.permute(0,2,3,1,4,5).contiguous()  # [B, 3, 3, C, 96, 96]
        patches = patches.view(B*9, C, 96, 96)  # [B*9, 3, 96, 96]
        # 送入 EfficientNet
        patch_features = self.ef(patches)  # [B*9, hidden_size1]
        patch_features = patch_features.view(B, 9, -1)  # [B, 9, hidden_size1]
        # 中心碎片
        central_tensor = patch_features[:, 4, :]  # [B, hidden_size1]
        # 其他碎片
        other_tensor = torch.cat([patch_features[:, :4, :], patch_features[:, 5:, :]], dim=1)  # [B, 8, hidden_size1]
        # 中心碎片扩展为 [B, 8, hidden_size1]
        central_expanded = central_tensor.unsqueeze(1).expand(-1, 8, -1)
        # 拼接中心与其他碎片 [B, 8, 2*hidden_size1]
        concat_tensor = torch.cat([central_expanded, other_tensor], dim=-1)
        # contract_fc 并行处理
        contracted = self.contract_fc(concat_tensor)  # [B, 8, hidden_size1]
        # 展平成 [B, 8*hidden_size1]
        feature_tensor = contracted.reshape(B, -1)
        feature_tensor = self.fc1(feature_tensor)
        feature_tensor = self.bn(feature_tensor)
        feature_tensor = self.relu(feature_tensor)
        out = self.out_layer(feature_tensor)
        return out