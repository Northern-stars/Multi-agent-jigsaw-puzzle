import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b3
import numpy as np
import os
import cv2
import random
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# ================= 配置 =================
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = "model_discriminator"
MODEL_NAME = "discriminator_srm_best.pth"
ERROR_SAVE_DIR = "error_analysis_srm"

TRAIN_DATA_PATH = "dataset/train_img_48gap_33-001.npy"
TEST_DATA_PATH = "dataset/test_img_48gap_33.npy"


# ================= 1. SRM 滤波器 (GPU层) =================
class SRMConv2d(nn.Module):
    """
    SRM 滤波器：用于在 GPU 上实时提取残差
    """

    def __init__(self):
        super().__init__()
        # 经典的 SRM 3个核
        f1 = [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ]
        f2 = [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ]
        f3 = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]

        q = [4.0, 12.0, 2.0]
        filters = []
        for f, div in zip([f1, f2, f3], q):
            filters.append(np.array(f, dtype=float) / div)
        filters = np.stack(filters)

        self.register_buffer("weight", torch.from_numpy(filters).float().unsqueeze(1))

    def forward(self, x):
        # x: [B, 3, H, W] (0-1 或 0-255)
        # 建议输入给 SRM 前还原到 0-255 幅度，效果最好
        if x.max() <= 1.0:
            x = x * 255.0

        b, c, h, w = x.shape
        x_reshape = x.view(b * c, 1, h, w)
        out = F.conv2d(x_reshape, self.weight, padding=2)
        out = out.view(b, c * 3, h, w)
        # 截断残差，抑制强边缘
        return torch.clamp(out, min=-3.0, max=3.0)


# ================= 2. 噪点提取器 (Noise Stream) =================
class NoiseExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_process = SRMConv2d()

        self.net = nn.Sequential(
            # Input: 9 channels (RGB * 3 filters)
            # Stride=1 保持高分辨率，不丢失噪点细节
            nn.Conv2d(9, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),  # -> [B, 128]
        )

    def forward(self, x):
        # x 是原始 RGB，内部先过 SRM 再过 CNN
        return self.net(self.pre_process(x))


# ================= 3. 双流融合判别器 (The Dual-Stream Detective) =================
class DualStreamDiscriminator(nn.Module):
    def __init__(self, rgb_dim=1024, noise_dim=128, hidden_dim=512):
        super().__init__()

        # Stream 1: RGB 语义流 (EfficientNet)
        self.rgb_encoder = efficientnet_b3(weights="DEFAULT")
        self.rgb_encoder.classifier = nn.Linear(1536, rgb_dim)

        # Stream 2: SRM 噪点流 (Custom CNN)
        self.noise_encoder = NoiseExtractor()

        self.feature_dim = rgb_dim + noise_dim  # 1024 + 128 = 1152

        # 决策层
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * 4, hidden_dim),  # u, v, |u-v|, u*v
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward_one_branch(self, img):
        # 语义特征
        f_rgb = self.rgb_encoder(img)
        # 噪点特征 (内部自动计算 SRM)
        f_noise = self.noise_encoder(img)
        return torch.cat([f_rgb, f_noise], dim=1)

    def forward(self, img_center, img_cand):
        v_center = self.forward_one_branch(img_center)
        v_cand = self.forward_one_branch(img_cand)

        # 特征交互
        diff = torch.abs(v_center - v_cand)
        prod = v_center * v_cand

        combined = torch.cat([v_center, v_cand, diff, prod], dim=1)
        return self.classifier(combined)


# ================= 4. 数据集 (纯净版，只负责切图) =================
class SiameseDataset(Dataset):
    def __init__(self, x_path):
        print(f"Loading data from {x_path}...")
        self.image = np.load(x_path)
        self.num_samples = self.image.shape[0]
        # 不做任何 Transform，保留原始指纹
        print("Data loaded (Clean).")

    def __len__(self):
        return self.num_samples

    def get_patch(self, img_idx):
        img = self.image[img_idx]
        row = random.randint(0, 2)
        col = random.randint(0, 2)
        patch = img[row * 96 : (row + 1) * 96, col * 96 : (col + 1) * 96, :]
        tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
        if tensor.max() > 1.0:
            tensor /= 255.0
        return tensor

    def __getitem__(self, index):
        # 50% 正样本 (同源), 50% 负样本 (异源)
        if random.random() > 0.5:
            # Positive
            img1 = self.get_patch(index)
            img2 = self.get_patch(index)
            label = 1.0
        else:
            # Negative
            img1 = self.get_patch(index)
            rand_idx = random.randint(0, self.num_samples - 1)
            while rand_idx == index:
                rand_idx = random.randint(0, self.num_samples - 1)
            img2 = self.get_patch(rand_idx)
            label = 0.0

        return img1, img2, torch.tensor([label], dtype=torch.float32)


# ================= 5. 错误分析可视化 =================
def save_error_grid(error_list, epoch, save_dir):
    """保存错误样本的大图"""
    if not error_list:
        return
    os.makedirs(save_dir, exist_ok=True)

    samples = error_list[:64]  # 最多保存64个
    rows, row_imgs = [], []

    for c_img, x_img, lbl, pred in samples:
        # 反归一化 [C, H, W] -> [H, W, C]
        c_np = (c_img.transpose(1, 2, 0) * 255).astype(np.uint8)
        x_np = (x_img.transpose(1, 2, 0) * 255).astype(np.uint8)

        c_bgr = cv2.cvtColor(c_np, cv2.COLOR_RGB2BGR)
        x_bgr = cv2.cvtColor(x_np, cv2.COLOR_RGB2BGR)

        # 拼接
        pair = np.hstack([c_bgr, x_bgr])

        # 标记: L=Label, P=Pred (1=Same, 0=Diff)
        color = (0, 0, 255)  # Red for Error
        cv2.putText(
            pair,
            f"L:{int(lbl)} P:{int(pred)}",
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

        # 加白框
        pair = cv2.copyMakeBorder(
            pair, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        row_imgs.append(pair)

        if len(row_imgs) == 8:
            rows.append(np.hstack(row_imgs))
            row_imgs = []

    if row_imgs:  # 补齐
        while len(row_imgs) < 8:
            row_imgs.append(np.zeros_like(row_imgs[0]))
        rows.append(np.hstack(row_imgs))

    if rows:
        final_grid = np.vstack(rows)
        cv2.imwrite(os.path.join(save_dir, f"epoch_{epoch}_errors.jpg"), final_grid)


# ================= 6. 训练主流程 =================
def train():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)

    train_dataset = SiameseDataset(TRAIN_DATA_PATH)
    test_dataset = SiameseDataset(TEST_DATA_PATH)

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

    model = DualStreamDiscriminator().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = GradScaler("cuda")

    print(f"Start Dual-Stream Training (SRM+EffNet) on {DEVICE}...")
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
                # 模型现在只接收两个 Raw RGB，内部自动分流
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

        # === 验证 & 错误分析 ===
        model.eval()
        test_correct = 0
        test_total = 0
        error_list = []  # 收集错误样本

        with torch.no_grad():
            for img1, img2, label in test_loader:
                img1_d, img2_d, label_d = (
                    img1.to(DEVICE),
                    img2.to(DEVICE),
                    label.to(DEVICE),
                )
                with autocast("cuda"):
                    logits = model(img1_d, img2_d)
                preds = (torch.sigmoid(logits) > 0.5).float()

                test_correct += (preds == label_d).sum().item()
                test_total += label_d.size(0)

                # 收集错误
                incorrect_mask = (preds != label_d).squeeze()
                if incorrect_mask.sum() > 0 and len(error_list) < 64:
                    idxs = torch.nonzero(incorrect_mask, as_tuple=True)[0]
                    for idx in idxs:
                        if len(error_list) >= 64:
                            break
                        # 转回 CPU numpy 用于保存
                        c_np = img1[idx].cpu().numpy()
                        x_np = img2[idx].cpu().numpy()
                        error_list.append(
                            (c_np, x_np, label[idx].item(), preds[idx].item())
                        )

        # 保存错误图
        save_error_grid(error_list, epoch, ERROR_SAVE_DIR)

        test_acc = test_correct / test_total
        print(f"Epoch {epoch+1}: Test Acc = {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model (Acc: {best_acc:.4f})")


if __name__ == "__main__":
    train()
