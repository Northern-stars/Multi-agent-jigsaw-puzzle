import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b3
from tqdm import tqdm
import random
import numpy as np
import os
from torch.amp import autocast, GradScaler

# ================= 配置 =================
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = "model"
MODEL_NAME = "deepuzzle_best.pth"
CKPT_NAME = "deepuzzle_ckpt.pth"
AUX_WEIGHT = 1.0  # 辅助损失的权重，越大越强迫模型学噪点

# 路径配置
train_x_path = "dataset/train_img_48gap_33-001.npy"
train_y_path = "dataset/train_label_48gap_33.npy"
test_x_path = "dataset/test_img_48gap_33.npy"
test_y_path = "dataset/test_label_48gap_33.npy"


# ================= 1. 辅助函数 =================
def save_checkpoint(path, model, optimizer, scaler, epoch, best_acc):
    state = {
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_acc": best_acc,
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer, scaler):
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    if "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return state["epoch"], state["best_acc"]


# ================= 2. 噪点处理层 =================
class LaplacianLayer(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor(
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel", kernel.repeat(3, 1, 1, 1))

    def forward(self, x):
        high_freq = F.conv2d(x, self.kernel, padding=1, groups=3)
        # 移除 /4.0，让数值大一点，更容易被网络捕捉
        return torch.abs(high_freq)


class NoiseExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_process = LaplacianLayer()
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
        noise_map = self.pre_process(x)
        return self.net(noise_map)


# ================= 3. 带辅助损失的模型 =================
class DeepuzzleModel(nn.Module):
    def __init__(
        self, rgb_dim=1024, noise_dim=128, color_dim=6, hidden_dim=512, output_dim=9
    ):
        super().__init__()
        # RGB流
        self.rgb_encoder = efficientnet_b3(weights="DEFAULT")
        self.rgb_encoder.classifier = nn.Linear(1536, rgb_dim)

        # 噪点流
        self.noise_encoder = NoiseExtractor()

        # 色彩流
        self.color_mlp = nn.Sequential(
            nn.Linear(color_dim, 64), nn.LayerNorm(64), nn.ReLU()
        )
        self.feature_dim = rgb_dim + noise_dim + 64

        # 【新增】辅助分类器：专门检查噪点是否一致
        # 输入：两个噪点向量的 diff 和 prod
        # 输出：1 (Same Image) / 0 (Different Image)
        self.aux_classifier = nn.Sequential(
            nn.Linear(noise_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1)  # 输出 Logits
        )

        # 主分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * 4 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def get_color_stats(self, tensor):
        mean = tensor.mean(dim=[2, 3])
        std = tensor.std(dim=[2, 3])
        return torch.cat([mean, std], dim=1)

    def forward_one(self, img):
        f_rgb = self.rgb_encoder(img)
        f_noise = self.noise_encoder(img)
        f_color = self.color_mlp(self.get_color_stats(img))
        return f_rgb, f_noise, f_color

    def forward(self, center_img, candidate_img):
        # 分别提取特征
        rgb_c, noise_c, color_c = self.forward_one(center_img)
        rgb_x, noise_x, color_x = self.forward_one(candidate_img)

        # 1. 组合主特征 (Main Branch)
        v_center = torch.cat([rgb_c, noise_c, color_c], dim=1)
        v_candidate = torch.cat([rgb_x, noise_x, color_x], dim=1)

        diff = torch.abs(v_center - v_candidate)
        prod = v_center * v_candidate
        cosine = F.cosine_similarity(v_center, v_candidate, dim=1).unsqueeze(1)
        combined = torch.cat([v_center, v_candidate, diff, prod, cosine], dim=1)

        main_logits = self.classifier(combined)

        # 2. 计算辅助任务 (Aux Branch) - 只看噪点！
        # 强制模型比较 noise_c 和 noise_x
        noise_diff = torch.abs(noise_c - noise_x)
        noise_prod = noise_c * noise_x
        aux_input = torch.cat([noise_diff, noise_prod], dim=1)

        aux_logits = self.aux_classifier(aux_input)

        return main_logits, aux_logits


# ================= 4. 数据集 (纯净版) =================
class DeepuzzleDataset(Dataset):
    def __init__(self, x_path, y_path, is_train=False):
        print(f"Loading data from {x_path}...")
        self.image = np.load(x_path)
        self.label = np.load(y_path)
        self.num_samples = self.image.shape[0]
        self.sector_num = max(1, self.num_samples // 3)
        self.is_train = is_train
        print("Data loaded (Clean).")

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
        rand_idx = random.randint(0, self.num_samples - 1)
        while rand_idx == host_idx:
            rand_idx = random.randint(0, self.num_samples - 1)

        perm_onehot = self.label[index]
        perm_idx = list(np.argmax(perm_onehot, axis=1))

        final_perm = []
        for p in perm_idx:
            if p >= 4:
                final_perm.append(p + 1)
            else:
                final_perm.append(p)
        final_perm.insert(4, 4)

        center_row, center_col = 1, 1
        target_class = random.randint(0, 8)
        cand_row, cand_col = 0, 0
        use_host = True

        if target_class == 8:  # Outsider
            k = random.randint(0, 8)
            cand_row, cand_col = k // 3, k % 3
            use_host = False
        else:  # Neighbor
            target_id = target_class if target_class < 4 else target_class + 1
            try:
                current_pos = final_perm.index(target_id)
                cand_row, cand_col = current_pos // 3, current_pos % 3
                use_host = True
            except:
                target_class = 8
                use_host = False

        c_img = self.get_patch(self.image[host_idx], center_row, center_col)

        if use_host:
            x_img = self.get_patch(self.image[host_idx], cand_row, cand_col)
        else:
            x_img = self.get_patch(self.image[rand_idx], cand_row, cand_col)

        return c_img, x_img, torch.tensor(target_class, dtype=torch.long)


# ================= 5. 训练主函数 =================
def train(resume=False):
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    ckpt_path = os.path.join(MODEL_SAVE_DIR, CKPT_NAME)

    train_dataset = DeepuzzleDataset(train_x_path, train_y_path, is_train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = DeepuzzleDataset(test_x_path, test_y_path, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = DeepuzzleModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # 两个损失函数
    loss_main_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss_aux_fn = nn.BCEWithLogitsLoss()  # 二分类

    scaler = GradScaler("cuda")

    print(f"Start training with Aux Loss (Weight={AUX_WEIGHT}) on {DEVICE}...")
    best_acc = 0.0
    start_epoch = 0

    if resume and os.path.exists(ckpt_path):
        start_epoch, best_acc = load_checkpoint(ckpt_path, model, optimizer, scaler)
        print(f"Resumed from epoch {start_epoch}")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-6, last_epoch=start_epoch - 1
        )

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for c_img, x_img, label in pbar:
            c_img, x_img, label = c_img.to(DEVICE), x_img.to(DEVICE), label.to(DEVICE)

            # 准备 Aux Label: 0-7是同源(1), 8是异源(0)
            # 注意：BCE 需要 float 类型的 target
            aux_label = (label < 8).float().unsqueeze(1)

            optimizer.zero_grad()

            with autocast("cuda"):
                logits_main, logits_aux = model(c_img, x_img)

                loss_main = loss_main_fn(logits_main, label)
                loss_aux = loss_aux_fn(logits_aux, aux_label)

                # 总损失 = 主任务 + 辅助任务
                loss = loss_main + AUX_WEIGHT * loss_aux

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            preds = logits_main.argmax(dim=1)
            train_correct += (preds == label).sum().item()
            train_total += label.size(0)

            pbar.set_postfix(
                {
                    "L": f"{loss.item():.2f}",
                    "Lm": f"{loss_main.item():.2f}",
                    "La": f"{loss_aux.item():.2f}",
                    "Acc": f"{train_correct/train_total:.2f}",
                }
            )

        scheduler.step()

        # 验证
        model.eval()
        test_correct = 0
        test_total = 0
        outsider_correct = 0
        outsider_total = 0

        with torch.no_grad():
            for c_img, x_img, label in test_loader:
                c_img, x_img, label = (
                    c_img.to(DEVICE),
                    x_img.to(DEVICE),
                    label.to(DEVICE),
                )

                with autocast("cuda"):
                    logits_main, _ = model(c_img, x_img)

                preds = logits_main.argmax(dim=1)
                test_correct += (preds == label).sum().item()
                test_total += label.size(0)

                mask8 = label == 8
                if mask8.sum() > 0:
                    outsider_total += mask8.sum().item()
                    outsider_correct += (preds[mask8] == 8).sum().item()

        test_acc = test_correct / test_total
        avg_loss = train_loss / len(train_loader)
        outsider_acc = outsider_correct / outsider_total if outsider_total > 0 else 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Test Acc={test_acc:.4f}, Outsider Acc={outsider_acc:.4f}, LR={current_lr:.6e}"
        )

        save_checkpoint(ckpt_path, model, optimizer, scaler, epoch, best_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model (Acc: {best_acc:.4f})")


if __name__ == "__main__":
    train(resume=False)
