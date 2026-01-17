import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import random

# ================= 配置 =================
DATA_PATH = "dataset/train_img_48gap_33-001.npy"  # 确保路径正确
SAMPLES_TO_SHOW = 5  # 展示几组样本
DEVICE = "cpu"  # 可视化用 CPU 足够了


# ================= SRM 滤波器定义 (与训练脚本保持一致) =================
class SRMConv2d(nn.Module):
    """
    SRM (Steganalysis Rich Models) 滤波器
    专门设计用于抑制 RGB 内容，提取残差。
    输出: [B, 9, H, W] (3个滤波器 * 3个RGB通道)
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

        # Stack -> [3, 5, 5] -> Unsqueeze -> [3, 1, 5, 5]
        filters = np.stack(filters)
        self.register_buffer("weight", torch.from_numpy(filters).float().unsqueeze(1))

    def forward(self, x):
        # 输入 x 预期范围: 0-255
        b, c, h, w = x.shape
        # Reshape: [B*3, 1, H, W] 以独立处理每个 RGB 通道
        x_reshape = x.view(b * c, 1, h, w)

        # 卷积
        out = F.conv2d(x_reshape, self.weight, padding=2)

        # Reshape back: [B, 9, H, W]
        # 通道顺序: [R_f1, G_f1, B_f1, R_f2, G_f2, B_f2, R_f3, G_f3, B_f3]
        out = out.view(b, c * 3, h, w)

        # 【关键】截断：只保留微小残差
        out = torch.clamp(out, min=-3.0, max=3.0)

        return out


# ================= 可视化逻辑 =================
def normalize_for_display(img_data, boost=1.0):
    """将残差数据归一化以便显示，并可选择增强对比度"""
    # 取绝对值，因为残差有正有负
    img_data = np.abs(img_data)
    # 增强对比度
    img_data = img_data * boost
    # 截断到 0-1
    img_data = np.clip(img_data, 0, 1.0)
    return img_data


def visualize():
    if not os.path.exists(DATA_PATH):
        print(f"Error: 找不到文件 {DATA_PATH}")
        return

    print(f"Loading data from {DATA_PATH}...")
    # 使用 mmap_mode='r' 快速加载大文件
    data = np.load(DATA_PATH, mmap_mode="r")
    total_imgs = data.shape[0]
    print(f"Data loaded. Total images: {total_imgs}")

    # 初始化 SRM 层
    srm_layer = SRMConv2d().to(DEVICE)
    srm_layer.eval()

    # 创建画布：每行显示 原图 + 3个滤波器的代表性输出 + 平均热力图
    fig, axes = plt.subplots(SAMPLES_TO_SHOW, 5, figsize=(16, 3.5 * SAMPLES_TO_SHOW))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    fig.suptitle(
        "SRM Filter Residual Visualization (Content Suppression)", fontsize=16, y=0.95
    )

    cols = [
        "Original Patch",
        "Filter 1 (KB) Residual",
        "Filter 2 (KV) Residual",
        "Filter 3 (SPAM) Residual",
        "Avg Residual Heatmap",
    ]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=10)

    print("Processing and plotting samples...")
    with torch.no_grad():
        for i in range(SAMPLES_TO_SHOW):
            # 1. 随机切片
            img_idx = random.randint(0, total_imgs - 1)
            full_img = data[img_idx]
            row = random.randint(0, 2)
            col = random.randint(0, 2)
            patch = full_img[row * 96 : (row + 1) * 96, col * 96 : (col + 1) * 96, :]

            # 2. 准备输入 Tensor (0-1 float)
            tensor_01 = (
                torch.from_numpy(patch).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
            )
            if tensor_01.max() > 1.0:
                tensor_01 /= 255.0

            # 3. 转换为 0-255 输入给 SRM (重要!)
            tensor_255 = tensor_01 * 255.0

            # 4. 通过 SRM 层 -> [1, 9, 96, 96]
            residuals = srm_layer(tensor_255)
            res_np = residuals.squeeze(0).cpu().numpy()  # [9, 96, 96]

            # 5. 准备绘图数据
            # 原图
            img_display = tensor_01.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # 选取代表性通道：我们展示每个滤波器处理绿色(Green)通道的结果
            # 通道索引: 1 (G_f1), 4 (G_f2), 7 (G_f3)
            # 增强系数 boost=30.0，因为残差非常小(-3到3)，需要大幅增强才能看见
            BOOST = 30.0
            res_f1_display = normalize_for_display(res_np[1], boost=BOOST)
            res_f2_display = normalize_for_display(res_np[4], boost=BOOST)
            res_f3_display = normalize_for_display(res_np[7], boost=BOOST)

            # 平均热力图：计算所有9个通道残差的平均绝对值，展示整体噪声活跃度
            avg_heatmap = np.mean(np.abs(res_np), axis=0)
            # 热力图不需要截断到 1.0，保留原始分布
            avg_heatmap = (
                avg_heatmap / avg_heatmap.max()
                if avg_heatmap.max() > 0
                else avg_heatmap
            )

            # --- 绘图 ---
            ax_row = axes[i] if SAMPLES_TO_SHOW > 1 else axes

            # Col 1: 原图
            ax_row[0].imshow(img_display)
            ax_row[0].axis("off")
            if i == 0:
                ax_row[0].set_ylabel("Sample 1")

            # Col 2-4: 单个滤波器的残差 (灰度图显示)
            ax_row[1].imshow(res_f1_display, cmap="gray", vmin=0, vmax=1)
            ax_row[1].axis("off")

            ax_row[2].imshow(res_f2_display, cmap="gray", vmin=0, vmax=1)
            ax_row[2].axis("off")

            ax_row[3].imshow(res_f3_display, cmap="gray", vmin=0, vmax=1)
            ax_row[3].axis("off")

            # Col 5: 平均热力图 (使用热力图配色)
            im = ax_row[4].imshow(avg_heatmap, cmap="inferno")
            ax_row[4].axis("off")

    print("Done. Showing plot.")
    plt.show()


if __name__ == "__main__":
    visualize()
