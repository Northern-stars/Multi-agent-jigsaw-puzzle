import torch.nn as nn
import torch
from torchvision.models import efficientnet_b0,efficientnet_b3,efficientnet_v2_m
import torch.nn.functional as F

class fen_model(nn.Module):
    def __init__(self, hidden_size1, hidden_size2,feature_hidden=512,model_name="ef"):
        super(fen_model, self).__init__()
        self.feature_hidden=feature_hidden
        if model_name=="ef":
            self.hori_ef = efficientnet_b3(weights="DEFAULT")
            self.hori_ef.classifier = nn.Linear(1536, feature_hidden)
            self.vert_ef = efficientnet_b3(weights="DEFAULT")
            self.vert_ef.classifier = nn.Linear(1536, feature_hidden)
        elif model_name=="modulator":
            self.hori_ef=Modulator(transformer_dim=feature_hidden,out=feature_hidden)
            self.vert_ef=Modulator(transformer_dim=feature_hidden,out=feature_hidden)

        self.contrast_fc_hori = nn.Linear(feature_hidden*2, hidden_size1)
        self.contrast_fc_vert = nn.Linear(feature_hidden*2, hidden_size1)

        self.fc1 = nn.Linear(hidden_size1*12, hidden_size1)
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
        hori_feats = self.hori_ef(patches)   # (B*9, 1024)
        hori_feats = hori_feats.view(B, 9, self.feature_hidden)  # (B, 9, 1024)
        vert_feats = self.vert_ef(patches)   # (B*9, 1024)
        vert_feats = vert_feats.view(B, 9, self.feature_hidden)  # (B, 9, 1024)

        # ---- 3. 构建横向对 ----
        hori_pairs = torch.stack([torch.cat([hori_feats[:, i, :], hori_feats[:, j, :]], dim=-1) 
                                  for (i,j) in self.hori_set], dim=1)  # (B, #pairs, 1024)
        hori_feats = self.contrast_fc_hori(hori_pairs)  # (B, #pairs, 512)

        # ---- 4. 构建纵向对 ----
        vert_pairs = torch.stack([torch.cat([vert_feats[:, i, :], vert_feats[:, j, :]], dim=-1) 
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
    
class piece_style_fen(nn.Module):
    def __init__(self,hidden1=1024,hidden2=512,color_dim=6,color_out=64,out=9):
        super().__init__()
        self.rgb_encoder=efficientnet_b3(weights="DEFAULT")
        self.rgb_encoder.classifier=nn.Linear(1536,hidden1)
        self.color_mlp = nn.Sequential(
            nn.Linear(color_dim, color_out), nn.LayerNorm(color_out), nn.ReLU()
        )
        self.feature_dim = hidden1 + color_out

        self.out_layer = nn.Sequential(
            nn.Linear((self.feature_dim * 4 + 1)*12, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden2),
            nn.Linear(hidden2, out),
        )

        self.hori_set = [(i, i+1) for i in range(9) if i % 3 != 2]
        self.vert_set = [(i, i+3) for i in range(6)]
        self.register_buffer(
            "hori_idx", torch.tensor(self.hori_set, dtype=torch.long)
        )
        self.register_buffer(
            "vert_idx", torch.tensor(self.vert_set, dtype=torch.long)
        )

    def get_color_stats(self, tensor):
        """计算均值和方差 [B, 6]"""
        mean = tensor.mean(dim=[2, 3])
        std = tensor.std(dim=[2, 3])
        return torch.cat([mean, std], dim=1)
    def encode(self, x):
        # 提取 RGB 特征
        f_rgb = self.rgb_encoder(x)
        # 提取色彩特征
        f_color = self.color_mlp(self.get_color_stats(x))
        return torch.cat([f_rgb, f_color], dim=-1)

    def forward(self, image):
        B, C, H, W = image.shape  # (B, 3, 288, 288)

        # ---- 1. 切 patch ----
        patches = image.unfold(2, 96, 96).unfold(3, 96, 96)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B * 9, C, 96, 96)

        # ---- 2. 编码 ----
        feats = self.encode(patches)                 # (B*9, D)
        feats = feats.view(B, 9, self.feature_dim)   # (B, 9, D)

        # =========================
        # 横向 pair（并行）
        # =========================
        hi, hj = self.hori_idx[:, 0], self.hori_idx[:, 1]  # (Nh,)

        f1 = feats[:, hi, :]   # (B, Nh, D)
        f2 = feats[:, hj, :]   # (B, Nh, D)

        diff = torch.abs(f1 - f2)
        prod = f1 * f2
        cosine = F.cosine_similarity(f1, f2, dim=-1, eps=1e-8).unsqueeze(-1)

        hori_feature = torch.cat([f1, f2, diff, prod, cosine], dim=-1)
        # (B, Nh, 4D+1)

        # =========================
        # 纵向 pair（并行）
        # =========================
        vi, vj = self.vert_idx[:, 0], self.vert_idx[:, 1]  # (Nv,)

        f1 = feats[:, vi, :]
        f2 = feats[:, vj, :]

        diff = torch.abs(f1 - f2)
        prod = f1 * f2
        cosine = F.cosine_similarity(f1, f2, dim=-1, eps=1e-8).unsqueeze(-1)

        vert_feature = torch.cat([f1, f2, diff, prod, cosine], dim=-1)
        # (B, Nv, 4D+1)

        # =========================
        # 拼接 & 输出
        # =========================
        hori_feature = hori_feature.flatten(1)  # (B, Nh*(4D+1))
        vert_feature = vert_feature.flatten(1)  # (B, Nv*(4D+1))

        final_feature = torch.cat([hori_feature, vert_feature], dim=-1)

        out = self.out_layer(final_feature)
        return out



class FeatureMLP(nn.Module):

    def __init__(self, input_dim, output_dim=256):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states

class Modulator(nn.Module):
    def __init__(
        self,
        transformer_dim,
        out,
        hidden_sizes=[128, 256, 512, 1024,1280],
    ):
        super().__init__()

        self.mlp1 = FeatureMLP(
            input_dim=hidden_sizes[-1], output_dim=transformer_dim
        ) 

        self.gmp_branch = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            # nn.Conv2d(transformer_dim // 2, transformer_dim // 2, kernel_size=1),
            nn.Conv2d(transformer_dim, transformer_dim//2, kernel_size=1),
        )
        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(transformer_dim // 2, transformer_dim // 2, kernel_size=1),
            nn.Conv2d(transformer_dim , transformer_dim//2, kernel_size=1),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim * 2),
            nn.ReLU(),
            nn.Linear(transformer_dim * 2,out),
        )

        self.fen=efficientnet_b0(pretrained=True).features
        self.reverse_conv=nn.ConvTranspose2d(1280,1280,5)

    def forward(self, x: torch.Tensor):
        x=self.fen(x)
        x=self.reverse_conv(x)
        bs, c, h, w = x.shape
        # print(f"Feature size: {bs,c,h,w}")
        x = self.mlp1(x) 
        x = x.permute(0, 2, 1).reshape(bs, -1, h, w)

        

        max_channel_attention = self.gmp_branch(x)
        avg_channel_attention = self.gap_branch(x)

        concated_channel_attention = torch.cat(
            [max_channel_attention, avg_channel_attention], dim=1
        ) 

        flatten_channel_attention = concated_channel_attention.flatten(
            1
        )
        fused_channel_attention = self.mlp2(
            flatten_channel_attention
        )


        return fused_channel_attention