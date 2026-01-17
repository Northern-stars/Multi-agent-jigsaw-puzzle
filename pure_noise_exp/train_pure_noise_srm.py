import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
from tqdm import tqdm

# 使用新版混合精度 API
from torch.amp import autocast, GradScaler

# ================= 配置区域 =================
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = "pure_noise_model"
MODEL_NAME = "noise_extractor_srm_highres.pth"  # High-Res 代表我们保留了更多细节

# 路径配置
train_x_path = "dataset/train_img_48gap_33-001.npy"
test_x_path = "dataset/test_img_48gap_33.npy"


# ================= 1. SRM 滤波器 (内容抑制器) =================
class SRMConv2d(nn.Module):
    """
    SRM (Steganalysis Rich Models)
    专门设计用于提取图像残差，抑制物体轮廓和颜色。
    """

    def __init__(self):
        super().__init__()
        # Filter 1: KB (基本高通)
        f1 = [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ]

        # Filter 2: KV (异向滤波)
        f2 = [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ]

        # Filter 3: SPAM (像素相关性)
        f3 = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]

        # 归一化系数
        q = [4.0, 12.0, 2.0]
        filters = []
        for f, div in zip([f1, f2, f3], q):
            filters.append(np.array(f, dtype=float) / div)

        filters = np.stack(filters)  # [3, 5, 5]
        self.register_buffer("weight", torch.from_numpy(filters).float().unsqueeze(1))

    def forward(self, x):
        # x: [B, 3, H, W] -> Reshape -> [B*3, 1, H, W]
        b, c, h, w = x.shape
        x_reshape = x.view(b * c, 1, h, w)

        # 卷积
        out = F.conv2d(x_reshape, self.weight, padding=2)

        # Reshape back: [B, 9, H, W]
        out = out.view(b, c * 3, h, w)

        # 【关键】截断：只保留 -3 到 3 之间的微小残差，丢弃强边缘
        out = torch.clamp(out, min=-3.0, max=3.0)

        return out


# ================= 2. 噪点特征提取器 (高保真版) =================
class NoiseExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_process = SRMConv2d()

        self.net = nn.Sequential(
            # Input: 9 channels (RGB * 3 SRM Filters)
            # 【优化点】Stride=1:
            # 为了防止 CNN 这种低通滤波器过早丢弃高频噪点，
            # 第一层我们保持原尺寸，不进行下采样。
            nn.Conv2d(9, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32, affine=True),  # IN 保持单图对比度
            nn.ReLU(),
            # 第二层：开始缓慢下采样
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            # Global Average Pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        # SRM 需要幅度较大的输入 (0-255) 来产生明显的残差
        if x.max() <= 1.0:
            x = x * 255.0
        return self.net(self.pre_process(x))


# ================= 3. 判别器模型 =================
class PureNoiseDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = NoiseExtractor()

        # 决策层
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 + 1, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, img1, img2):
        v1 = self.encoder(img1)
        v2 = self.encoder(img2)

        # 交互特征
        diff = torch.abs(v1 - v2)
        prod = v1 * v2
        cosine = F.cosine_similarity(v1, v2, dim=1).unsqueeze(1)

        combined = torch.cat([v1, v2, diff, prod, cosine], dim=1)
        return self.classifier(combined)


# ================= 4. 数据集 (纯净版) =================
class SiameseNoiseDataset(Dataset):
    def __init__(self, x_path):
        print(f"Loading data from {x_path}...")
        self.image = np.load(x_path)
        self.num_samples = self.image.shape[0]
        print("Data loaded (Clean, No Augmentation).")

    def __len__(self):
        return self.num_samples

    def get_patch(self, img_idx):
        img = self.image[img_idx]
        # 随机切片
        row = random.randint(0, 2)
        col = random.randint(0, 2)
        patch = img[row * 96 : (row + 1) * 96, col * 96 : (col + 1) * 96, :]

        tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
        # 归一化到 0-1 (模型内部会处理回 255 给 SRM)
        if tensor.max() > 1.0:
            tensor /= 255.0
        return tensor

    def __getitem__(self, index):
        # 50% 正样本 (同源), 50% 负样本 (异源)
        is_same = random.random() > 0.5

        img1 = self.get_patch(index)

        if is_same:
            img2 = self.get_patch(index)  # 同一张图的另一个切片
            label = 1.0
        else:
            rand_idx = random.randint(0, self.num_samples - 1)
            while rand_idx == index:
                rand_idx = random.randint(0, self.num_samples - 1)
            img2 = self.get_patch(rand_idx)  # 另一张图的切片
            label = 0.0

        return img1, img2, torch.tensor(label, dtype=torch.float32).unsqueeze(0)


# ================= 5. 训练主程序 =================
def train():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)

    train_dataset = SiameseNoiseDataset(train_x_path)
    test_dataset = SiameseNoiseDataset(test_x_path)

    # Num workers 设为 4 加速 CPU 切片
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = PureNoiseDiscriminator().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = GradScaler("cuda")

    print(f"Start Pure Noise Training (SRM + Stride1) on {DEVICE}...")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for img1, img2, label in pbar:
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)

            optimizer.zero_grad()
            with autocast("cuda"):
                logits = model(img1, img2)
                loss = loss_fn(logits, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == label).sum().item()
            total += label.size(0)

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"}
            )

        # 验证
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for img1, img2, label in test_loader:
                img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
                with autocast("cuda"):
                    logits = model(img1, img2)
                preds = (torch.sigmoid(logits) > 0.5).float()
                test_correct += (preds == label).sum().item()
                test_total += label.size(0)

        test_acc = test_correct / test_total
        print(f"Epoch {epoch+1}: Test Acc = {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Noise Extractor! (Acc: {best_acc:.4f})")


if __name__ == "__main__":
    train()
