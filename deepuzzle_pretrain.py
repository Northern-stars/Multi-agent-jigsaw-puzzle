import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b3
from tqdm import tqdm
import random
import numpy as np
import os

# ================= 配置 =================
BATCH_SIZE = 64  # 根据显存调整
LR = 1e-4
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = "model"
MODEL_NAME = "deepuzzle_best.pth"
CKPT_NAME = "deepuzzle_ckpt.pth"
RESUME = True
FILENAME = "_deepuzzle_9"

# 路径配置
train_x_path = "dataset/train_img_48gap_33-001.npy"
train_y_path = "dataset/train_label_48gap_33.npy"
test_x_path = "dataset/test_img_48gap_33.npy"
test_y_path = "dataset/test_label_48gap_33.npy"


# ================= 模型定义 =================
class DeepuzzleModel(nn.Module):
    def __init__(self, rgb_dim=1024, color_dim=6, hidden_dim=512, output_dim=9):
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
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * 4 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),  # 输出 9 个 Logits
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
        logits = self.classifier(combined)
        return logits


# ================= 数据集定义 =================
class DeepuzzleDataset(Dataset):
    def __init__(self, x_path, y_path):
        print(f"Loading data from {x_path}...")
        self.image = np.load(x_path)
        self.label = np.load(y_path)
        self.num_samples = self.image.shape[0]
        # 假设数据是按某种规律排列的，这里简单处理 sector
        self.sector_num = max(1, self.num_samples // 3)
        print("Data loaded.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # 1. 获取主图 (Host)
        img1_np = self.image[index]

        # 解析 Label (Permutation)
        # label[index] 是 one-hot [8, 8]，转为 index list
        # image1_label[i] 表示第 i 个位置放的是原来的第几号块
        perm_onehot = self.label[index]
        perm_idx = list(np.argmax(perm_onehot, axis=1))
        # 修正 label (因为中间少了4号)
        for i in range(len(perm_idx)):
            if perm_idx[i] >= 4:
                perm_idx[i] += 1
        perm_idx.insert(4, 4)  # 现在的 perm_idx 是 9 个数，对应 0-8 位置的块ID

        # 2. 获取异图 (Guest) - 用于 Outsider
        rand_idx = random.randint(0, self.num_samples - 1)
        while rand_idx // self.sector_num == index // self.sector_num:
            rand_idx = random.randint(0, self.num_samples - 1)
        img2_np = self.image[rand_idx]

        # 3. 切片 (Host)
        # 注意：这里切出来的是视觉上的 9 个格子的内容
        host_patches = []
        for r in range(3):
            for c in range(3):
                patch = (
                    torch.tensor(
                        img1_np[r * 96 : (r + 1) * 96, c * 96 : (c + 1) * 96, :]
                    )
                    .permute(2, 0, 1)
                    .float()
                )
                host_patches.append(patch)

        # 4. 切片 (Guest)
        guest_patches = []
        for r in range(3):
            for c in range(3):
                patch = (
                    torch.tensor(
                        img2_np[r * 96 : (r + 1) * 96, c * 96 : (c + 1) * 96, :]
                    )
                    .permute(2, 0, 1)
                    .float()
                )
                guest_patches.append(patch)

        # 5. 确定 Center
        center_piece = host_patches[4]

        # 6. 构造样本
        # pos: 0-8.
        # 如果 pos < 8: 表示它是 Center 的某个邻居 (0,1,2,3, 5,6,7,8 -> 映射为 label 0-7)
        # 如果 pos == 8: 表示它是 Outsider

        target_class = random.randint(0, 8)  # 0-7: Neighbors, 8: Outsider

        candidate_piece = None

        if target_class == 8:
            # Outsider: 随便选一个 guest patch
            candidate_piece = random.choice(guest_patches)
        else:
            # Neighbor: 我们需要找到“真值”是 pos 的那个块
            # target_class 对应真实的 piece ID (0,1,2,3, 5,6,7,8)
            # 我们需要把 0-7 映射回 piece ID
            target_piece_id = target_class
            if target_piece_id >= 4:
                target_piece_id += 1

            # 现在的 host_patches 是乱序的(scrambled)，perm_idx 告诉我们要找的 piece ID 在哪个位置
            # perm_idx[k] == target_piece_id，那么 host_patches[k] 就是我们要的块
            try:
                current_pos = perm_idx.index(target_piece_id)
                candidate_piece = host_patches[current_pos]
            except ValueError:
                # 容错：如果找不到(极少见)，就给个全黑，label设为8
                candidate_piece = torch.zeros_like(center_piece)
                target_class = 8

        # 归一化到 0-1 (配合 EfficientNet 和 Color Stats)
        if center_piece.max() > 1.0:
            center_piece /= 255.0
        if candidate_piece.max() > 1.0:
            candidate_piece /= 255.0

        return (
            center_piece,
            candidate_piece,
            torch.tensor(target_class, dtype=torch.long),
        )


def save_checkpoint(path, model, optimizer, epoch, best_acc):
    torch.save(
        {
            "epoch": epoch,
            "best_acc": best_acc,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt.get("epoch", -1) + 1
    best_acc = ckpt.get("best_acc", 0.0)
    return start_epoch, best_acc


# ================= 训练流程 =================
def train(resume=False):
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    ckpt_path = os.path.join(MODEL_SAVE_DIR, CKPT_NAME)

    # 数据集
    train_dataset = DeepuzzleDataset(train_x_path, train_y_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    test_dataset = DeepuzzleDataset(test_x_path, test_y_path)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # 模型
    model = DeepuzzleModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    loss_fn = nn.CrossEntropyLoss()

    print(f"Start training Deepuzzle Improved on {DEVICE}...")
    best_acc = 0.0
    start_epoch = 0

    if resume:
        if os.path.exists(ckpt_path):
            start_epoch, best_acc = load_checkpoint(ckpt_path, model, optimizer)
            print(
                f"Resumed from checkpoint: {ckpt_path} (epoch {start_epoch}, best_acc {best_acc:.4f})"
            )
        elif os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path, map_location=DEVICE))
            print(
                f"Checkpoint not found. Loaded best model weights from {save_path}; optimizer reset; starting at epoch 0."
            )
        else:
            print(
                "Resume requested but no checkpoint or best model found. Training from scratch."
            )

    for local_epoch in range(EPOCHS):
        epoch = start_epoch + local_epoch
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + EPOCHS}")
        for center, cand, label in pbar:
            center, cand, label = center.to(DEVICE), cand.to(DEVICE), label.to(DEVICE)

            optimizer.zero_grad()
            logits = model(center, cand)
            loss = loss_fn(logits, label)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == label).sum().item()
            train_total += label.size(0)

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{train_correct/train_total:.4f}",
                }
            )

        # 验证
        model.eval()
        test_correct = 0
        test_total = 0

        # 统计 Outsider 的召回率 (Class 8)
        outsider_correct = 0
        outsider_total = 0

        with torch.no_grad():
            for center, cand, label in test_loader:
                center, cand, label = (
                    center.to(DEVICE),
                    cand.to(DEVICE),
                    label.to(DEVICE),
                )
                logits = model(center, cand)
                preds = logits.argmax(dim=1)

                test_correct += (preds == label).sum().item()
                test_total += label.size(0)

                # 统计 class 8
                mask8 = label == 8
                if mask8.sum() > 0:
                    outsider_total += mask8.sum().item()
                    outsider_correct += (preds[mask8] == 8).sum().item()

        test_acc = test_correct / test_total
        avg_loss = train_loss / len(train_loader)

        outsider_acc = outsider_correct / outsider_total if outsider_total > 0 else 0.0

        print(
            f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Test Acc={test_acc:.4f}, Outsider Acc={outsider_acc:.4f}"
        )

        save_checkpoint(ckpt_path, model, optimizer, epoch, best_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model (Acc: {best_acc:.4f})")


def continue_train():
    train(resume=True)


if __name__ == "__main__":
    if RESUME:
        continue_train()
    else:
        train()
