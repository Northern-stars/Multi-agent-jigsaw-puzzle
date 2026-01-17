import torch
import torch.nn as nn
import os
import numpy as np
import cv2  # 需要 opencv-python
from torch.utils.data import DataLoader
from tqdm import tqdm

# 引入上面写的模块
from discriminator_noise_model import DiscriminatorModel
from discriminator_dataset_noise import SiameseNoiseDataset

# ================= 配置 =================
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = "model"
MODEL_NAME = "discriminator_noise_best.pth"
ERROR_SAVE_DIR = "error_analysis"  # 错误样本保存路径

TRAIN_DATA_PATH = "dataset/train_img_48gap_33-001.npy"
TEST_DATA_PATH = "dataset/test_img_48gap_33.npy"


def save_error_grid(error_list, epoch, save_dir):
    """
    将错误样本拼成大图保存
    error_list: [(center_img, cand_img, label, pred), ...]
    """
    if not error_list:
        return

    os.makedirs(save_dir, exist_ok=True)

    # 限制最大保存数量，比如只看前 64 个错误
    max_samples = 64
    samples = error_list[:max_samples]

    rows = []
    row_imgs = []
    cols_per_row = 8

    for c_img, x_img, lbl, pred in samples:
        # c_img, x_img 是 numpy array (H, W, 3), RGB
        # 转换数据类型和颜色空间
        if c_img.max() <= 1.0:
            c_img = (c_img * 255).astype(np.uint8)
        if x_img.max() <= 1.0:
            x_img = (x_img * 255).astype(np.uint8)
        else:
            c_img = c_img.astype(np.uint8)
            x_img = x_img.astype(np.uint8)

        c_bgr = cv2.cvtColor(c_img, cv2.COLOR_RGB2BGR)
        x_bgr = cv2.cvtColor(x_img, cv2.COLOR_RGB2BGR)

        # 拼接：左边是中心块，右边是待选块
        pair = np.hstack([c_bgr, x_bgr])

        # 在图片上写字：L=真值 P=预测
        # L:1 P:0 (漏报, FN) | L:0 P:1 (误报, FP)
        text = f"L:{int(lbl)} P:{int(pred)}"
        # 红色字体
        cv2.putText(pair, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 加个白框隔开
        pair = cv2.copyMakeBorder(
            pair, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

        row_imgs.append(pair)

        if len(row_imgs) == cols_per_row:
            rows.append(np.hstack(row_imgs))
            row_imgs = []

    # 补齐最后一行
    if row_imgs:
        while len(row_imgs) < cols_per_row:
            blank = np.zeros_like(row_imgs[0])
            row_imgs.append(blank)
        rows.append(np.hstack(row_imgs))

    if rows:
        final_grid = np.vstack(rows)
        save_path = os.path.join(save_dir, f"epoch_{epoch}_errors.png")
        cv2.imwrite(save_path, final_grid)
        # print(f"Saved error visualization to {save_path}")


def train():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)

    # 1. 数据准备
    train_dataset = SiameseNoiseDataset(TRAIN_DATA_PATH)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    test_dataset = SiameseNoiseDataset(TEST_DATA_PATH)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # 2. 模型与优化器
    model = DiscriminatorModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"Start training Noise-Aware Discriminator on {DEVICE}...")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        # === 训练阶段 ===
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for c_rgb, x_rgb, label in pbar:
            c_rgb = c_rgb.to(DEVICE)
            x_rgb = x_rgb.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()
            logits = model(c_rgb, x_rgb)
            loss = loss_fn(logits, label)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = (logits > 0).float()
            train_correct += (preds == label).sum().item()
            train_total += label.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # === 验证阶段 ===
        model.eval()
        test_correct = 0
        test_total = 0

        # 收集这一轮所有的错误样本
        error_samples = []

        with torch.no_grad():
            for c_rgb, x_rgb, label in tqdm(test_loader, desc="[Eval]"):
                c_rgb_d = c_rgb.to(DEVICE)
                x_rgb_d = x_rgb.to(DEVICE)
                label_d = label.to(DEVICE)

                logits = model(c_rgb_d, x_rgb_d)
                preds = (logits > 0).float()

                test_correct += (preds == label_d).sum().item()
                test_total += label_d.size(0)

                # --- 收集错误样本 ---
                # 找出预测错误的索引
                incorrect_mask = (preds != label_d).squeeze()
                if incorrect_mask.sum() > 0:
                    # 获取错误的下标
                    idxs = torch.nonzero(incorrect_mask, as_tuple=True)[0]

                    for idx in idxs:
                        # 只需要收集够画图的数量就行了 (比如 64 个)
                        if len(error_samples) >= 64:
                            continue

                        # 转回 CPU numpy 用于画图
                        # permute(1, 2, 0) 把 [C, H, W] -> [H, W, C]
                        c_img_np = c_rgb[idx].permute(1, 2, 0).numpy()
                        x_img_np = x_rgb[idx].permute(1, 2, 0).numpy()
                        lbl_val = label[idx].item()
                        pred_val = preds[idx].item()

                        error_samples.append((c_img_np, x_img_np, lbl_val, pred_val))

        # 保存本轮错误拼图
        if len(error_samples) > 0:
            save_error_grid(error_samples, epoch, ERROR_SAVE_DIR)

        train_acc = train_correct / train_total
        test_acc = test_correct / test_total
        avg_loss = train_loss / len(train_loader)

        print(
            f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model (Acc: {best_acc:.4f}) to {save_path}")


if __name__ == "__main__":
    train()
