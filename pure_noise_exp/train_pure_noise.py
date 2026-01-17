import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# ================= 配置 =================
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = "pure_noise_model"
MODEL_NAME = "noise_extractor_best.pth"

# 路径配置 (请确保路径正确)
train_x_path = "dataset/train_img_48gap_33-001.npy"
# 不需要 label 文件，因为我们是自监督学习 (Self-Supervised)
# train_y_path 也不需要，我们自己生成 0/1 标签
test_x_path = "dataset/test_img_48gap_33.npy"


# ================= 1. 核心组件：噪点提取器 =================
class LaplacianLayer(nn.Module):
    """GPU 上的高通滤波器"""

    def __init__(self):
        super().__init__()
        kernel = torch.tensor(
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel", kernel.repeat(3, 1, 1, 1))

    def forward(self, x):
        return torch.abs(F.conv2d(x, self.kernel, padding=1, groups=3))


class NoiseExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_process = LaplacianLayer()
        # 纯粹的小型 CNN，不依赖任何预训练权重
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(self.pre_process(x))


# ================= 2. 纯噪点判别模型 =================
class PureNoiseDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = NoiseExtractor()

        # 决策层：只看噪点特征
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 + 1, 256),  # u, v, |u-v|, u*v, cosine
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, 1),  # 二分类 logits
        )

    def forward(self, img1, img2):
        v1 = self.encoder(img1)
        v2 = self.encoder(img2)

        # 强特征交互
        diff = torch.abs(v1 - v2)
        prod = v1 * v2
        cosine = F.cosine_similarity(v1, v2, dim=1).unsqueeze(1)

        combined = torch.cat([v1, v2, diff, prod, cosine], dim=1)
        return self.classifier(combined)


# ================= 3. 数据集：构造正负样本 =================
class SiameseNoiseDataset(Dataset):
    def __init__(self, x_path):
        print(f"Loading data from {x_path}...")
        self.image = np.load(x_path)
        self.num_samples = self.image.shape[0]
        print("Data loaded (Clean, No Augmentation).")

    def __len__(self):
        return self.num_samples

    def get_patch(self, img_idx):
        # 随机从一张大图里切一个块
        img = self.image[img_idx]
        row = random.randint(0, 2)
        col = random.randint(0, 2)
        patch = img[row * 96 : (row + 1) * 96, col * 96 : (col + 1) * 96, :]

        tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
        if tensor.max() > 1.0:
            tensor /= 255.0
        return tensor

    def __getitem__(self, index):
        # 50% 概率是正样本 (同一个 index 的图)
        # 50% 概率是负样本 (不同的 index)
        is_same = random.random() > 0.5

        img1 = self.get_patch(index)

        if is_same:
            img2 = self.get_patch(index)  # 再切一块
            label = 1.0
        else:
            rand_idx = random.randint(0, self.num_samples - 1)
            while rand_idx == index:
                rand_idx = random.randint(0, self.num_samples - 1)
            img2 = self.get_patch(rand_idx)
            label = 0.0

        return img1, img2, torch.tensor(label, dtype=torch.float32).unsqueeze(0)


# ================= 4. 训练流程 =================
def train():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)

    # 这里的 Dataset 很简单，不需要 label 文件
    train_dataset = SiameseNoiseDataset(train_x_path)
    test_dataset = SiameseNoiseDataset(test_x_path)

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

    print(f"Start Pure Noise Training on {DEVICE}...")
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
