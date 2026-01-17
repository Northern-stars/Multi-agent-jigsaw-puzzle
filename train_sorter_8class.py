import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b3
import numpy as np
import os
import random
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# ================= 配置 =================
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = "model_sorter"
MODEL_NAME = "sorter_8class_best.pth"

# 你的预训练权重路径 (用于初始化骨干网络)
PRETRAINED_PATH = ""

# 路径配置
TRAIN_DATA_PATH = "dataset/train_img_48gap_33-001.npy"
TRAIN_LABEL_PATH = "dataset/train_label_48gap_33.npy"
TEST_DATA_PATH = "dataset/test_img_48gap_33.npy"
TEST_LABEL_PATH = "dataset/test_label_48gap_33.npy"


# ================= 1. 排序模型定义 (纯 RGB 分类) =================
class SorterModel(nn.Module):
    def __init__(self):
        super().__init__()
        # --- RGB Backbone ---
        base_model = efficientnet_b3(weights="DEFAULT")
        self.rgb_features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1536 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 8),  # 8分类：代表 0-7 个相邻位置
        )

    def forward_one_branch(self, img):
        f_rgb = self.rgb_features(img)  # [B, 1536, H, W]
        f_rgb = self.pool(f_rgb).flatten(1)  # [B, 1536]
        return f_rgb

    def forward(self, img_center, img_cand):
        feat_center = self.forward_one_branch(img_center)  # [B, 1536]
        feat_cand = self.forward_one_branch(img_cand)  # [B, 1536]
        combined = torch.cat([feat_center, feat_cand], dim=1)  # [B, 3072]
        return self.classifier(combined)


# ================= 3. 数据集 =================
class NeighborOnlyDataset(Dataset):
    def __init__(self, x_path, y_path):
        print(f"Loading data from {x_path}...")
        self.image = np.load(x_path)
        self.label = np.load(y_path)
        self.num_samples = self.image.shape[0]
        print(f"Data loaded: {self.num_samples} samples.")

    def __len__(self):
        return self.num_samples

    def get_patch(self, img, row, col):
        patch = img[row * 96 : (row + 1) * 96, col * 96 : (col + 1) * 96, :]
        tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
        if tensor.max() > 1.0:
            tensor /= 255.0
        return tensor

    def __getitem__(self, index):
        host_idx = index
        perm_onehot = self.label[index]
        perm_idx = list(np.argmax(perm_onehot, axis=1))

        # 还原 3x3 索引 (跳过中心点 4)
        final_perm = []
        for p in perm_idx:
            final_perm.append(p + 1 if p >= 4 else p)
        final_perm.insert(4, 4)

        # 随机选择一个邻居进行训练 (0-7 分类)
        target_class = random.randint(0, 7)
        target_id = target_class if target_class < 4 else target_class + 1

        current_pos = final_perm.index(target_id)
        cand_row, cand_col = current_pos // 3, current_pos % 3

        c_img = self.get_patch(self.image[host_idx], 1, 1)  # 中心图
        x_img = self.get_patch(self.image[host_idx], cand_row, cand_col)

        return c_img, x_img, torch.tensor(target_class, dtype=torch.long)


# ================= 4. 训练流程 =================
def train():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)

    model = SorterModel().to(DEVICE)

    # 加载预训练骨干 (如果存在)
    if os.path.exists(PRETRAINED_PATH):
        print(f"Loading pretrained backbone from {PRETRAINED_PATH}...")
        state = torch.load(PRETRAINED_PATH, map_location=DEVICE)
        state = state["state_dict"] if "state_dict" in state else state

        model_state = model.state_dict()
        # 仅加载 matching 的 encoder 层
        pretrained_dict = {
            k: v for k, v in state.items() if k in model_state and "classifier" not in k
        }
        model_state.update(pretrained_dict)
        model.load_state_dict(model_state, strict=False)
        print(f"Successfully initialized {len(pretrained_dict)} layers.")

    train_loader = DataLoader(
        NeighborOnlyDataset(TRAIN_DATA_PATH, TRAIN_LABEL_PATH),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        NeighborOnlyDataset(TEST_DATA_PATH, TEST_LABEL_PATH),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler("cuda")

    print(f"Starting Training on {DEVICE}...")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for c, x, lbl in pbar:
            c, x, lbl = c.to(DEVICE), x.to(DEVICE), lbl.to(DEVICE)
            optimizer.zero_grad()

            with autocast("cuda"):
                logits = model(c, x)
                loss = loss_fn(logits, lbl)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = logits.argmax(1)
            correct += (preds == lbl).sum().item()
            total += lbl.size(0)
            pbar.set_postfix(
                {"acc": f"{correct/total:.4f}", "loss": f"{loss.item():.4f}"}
            )

        # 验证
        model.eval()
        t_correct, t_total = 0, 0
        with torch.no_grad():
            for c, x, lbl in test_loader:
                c, x, lbl = c.to(DEVICE), x.to(DEVICE), lbl.to(DEVICE)
                with autocast("cuda"):
                    logits = model(c, x)
                t_correct += (logits.argmax(1) == lbl).sum().item()
                t_total += lbl.size(0)

        acc = t_correct / t_total
        print(f"Epoch {epoch+1} Test Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print("--- Best Model Saved ---")


if __name__ == "__main__":
    train()
