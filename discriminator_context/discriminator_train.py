import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# 引入上面写的模块
from discriminator_model import DiscriminatorModel
from discriminator_dataset import SiameseDataset

# ================= 配置 =================
BATCH_SIZE = 64  # 如果显存够大，可以开到 128
LR = 1e-4
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = "model"
MODEL_NAME = "discriminator_best.pth"

TRAIN_DATA_PATH = "dataset/train_img_48gap_33-001.npy"
TEST_DATA_PATH = "dataset/test_img_48gap_33.npy"


def train():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)

    # 1. 数据准备
    train_dataset = SiameseDataset(TRAIN_DATA_PATH)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )  # 建议 num_workers > 0

    test_dataset = SiameseDataset(TEST_DATA_PATH)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # 2. 模型与优化器
    model = DiscriminatorModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()  # 配合输出 logits 使用

    print(f"Start training on {DEVICE}...")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        # === 训练阶段 ===
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for center, candidate, label in pbar:
            center, candidate, label = (
                center.to(DEVICE),
                candidate.to(DEVICE),
                label.to(DEVICE),
            )

            optimizer.zero_grad()
            logits = model(center, candidate)
            loss = loss_fn(logits, label)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 计算精度 (Logits > 0 等价于 Sigmoid > 0.5)
            preds = (logits > 0).float()
            train_correct += (preds == label).sum().item()
            train_total += label.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # === 验证阶段 ===
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for center, candidate, label in test_loader:
                center, candidate, label = (
                    center.to(DEVICE),
                    candidate.to(DEVICE),
                    label.to(DEVICE),
                )
                logits = model(center, candidate)
                preds = (logits > 0).float()
                test_correct += (preds == label).sum().item()
                test_total += label.size(0)

        train_acc = train_correct / train_total
        test_acc = test_correct / test_total
        avg_loss = train_loss / len(train_loader)

        print(
            f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}"
        )

        # 保存最优模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model (Acc: {best_acc:.4f}) to {save_path}")


if __name__ == "__main__":
    train()
