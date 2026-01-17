import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import random

# ================= 配置 =================
DATA_PATH = "dataset/train_img_48gap_33-001.npy"
SAMPLES_TO_SHOW = 4  # 一次看几张图


def get_laplacian_noise(img_tensor):
    """
    模拟模型中的 LaplacianLayer
    Input: [1, 3, H, W]
    Output: [1, 3, H, W]
    """
    kernel = torch.tensor(
        [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32
    ).view(1, 1, 3, 3)
    # 重复卷积核以适应 RGB 3通道
    kernel = kernel.repeat(3, 1, 1, 1)

    # 卷积提取高频
    noise = F.conv2d(img_tensor, kernel, padding=1, groups=3)
    return torch.abs(noise)


def visualize():
    if not os.path.exists(DATA_PATH):
        print(f"Error: 找不到文件 {DATA_PATH}")
        return

    print(f"Loading data from {DATA_PATH}...")
    data = np.load(DATA_PATH)
    total_imgs = data.shape[0]
    print(f"Loaded {total_imgs} images. Selecting random samples...")

    # 创建画布
    fig, axes = plt.subplots(SAMPLES_TO_SHOW, 3, figsize=(12, 4 * SAMPLES_TO_SHOW))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    fig.suptitle(f"Camera Noise Visualization (Laplacian Filter)", fontsize=16)

    for i in range(SAMPLES_TO_SHOW):
        # 1. 随机选一张图
        idx = random.randint(0, total_imgs - 1)
        full_img = data[idx]  # (288, 288, 3)

        # 2. 随机切一个 96x96 的 Patch (模拟训练时的切片)
        row = random.randint(0, 2)
        col = random.randint(0, 2)
        patch = full_img[row * 96 : (row + 1) * 96, col * 96 : (col + 1) * 96, :]

        # 3. 转 Tensor 并归一化
        tensor = (
            torch.from_numpy(patch).permute(2, 0, 1).float().unsqueeze(0)
        )  # [1, 3, 96, 96]
        if tensor.max() > 1.0:
            tensor /= 255.0

        # 4. 计算噪点
        noise_tensor = get_laplacian_noise(tensor)

        # --- 准备显示数据 ---

        # A. 原图
        img_display = tensor.squeeze(0).permute(1, 2, 0).numpy()

        # B. 原始噪点 (模型真正看到的输入)
        # 通常很黑，因为数值很小
        noise_display = noise_tensor.squeeze(0).permute(1, 2, 0).numpy()

        # C. 增强噪点 (为了让人眼看清)
        # 乘以 5 倍亮度，并截断到 1.0
        noise_enhanced = np.clip(noise_display * 5.0, 0, 1.0)

        # --- 绘图 ---
        ax_row = axes[i] if SAMPLES_TO_SHOW > 1 else axes

        # 原图
        ax_row[0].imshow(img_display)
        ax_row[0].set_title(f"Original Patch (Idx: {idx})")
        ax_row[0].axis("off")

        # 原始噪点
        ax_row[1].imshow(noise_display)
        ax_row[1].set_title("Raw Noise (What Model Sees)")
        ax_row[1].axis("off")

        # 增强噪点
        ax_row[2].imshow(noise_enhanced)
        ax_row[2].set_title("Enhanced Noise (x5 Contrast)")
        ax_row[2].axis("off")

    print("Showing plot window...")
    plt.show()


if __name__ == "__main__":
    visualize()
