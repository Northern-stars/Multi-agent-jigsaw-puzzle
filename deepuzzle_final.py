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
LR = 1e-3  # 因为只训练最后一层，学习率可以稍微大一点
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 路径配置
PRETRAINED_DISCRIMINATOR_PATH = (
    "model_discriminator/discriminator_srm_best.pth"  # 你的二分类权重路径
)
MODEL_SAVE_DIR = "model_jigsaw"
MODEL_NAME = "jigsaw_frozen_best.pth"

TRAIN_DATA_PATH = "dataset/train_img_48gap_33-001.npy"
TRAIN_LABEL_PATH = "dataset/train_label_48gap_33.npy"
TEST_DATA_PATH = "dataset/test_img_48gap_33.npy"
TEST_LABEL_PATH = "dataset/test_label_48gap_33.npy"

# ================= 1. 复刻模型结构 (必须与权重文件完全一致) =================
# 为了加载权重，我们需要定义一模一样的类结构


class SRMConv2d(nn.Module):
    def __init__(self):
        super().__init__()
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
        if x.max() <= 1.0:
            x = x * 255.0
        b, c, h, w = x.shape
        x_reshape = x.view(b * c, 1, h, w)
        out = F.conv2d(x_reshape, self.weight, padding=2)
        out = out.view(b, c * 3, h, w)
        return torch.clamp(out, min=-3.0, max=3.0)


class NoiseExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_process = SRMConv2d()
        self.net = nn.Sequential(
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
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(self.pre_process(x))


class DualStreamDiscriminator(nn.Module):
    def __init__(self, rgb_dim=1024, noise_dim=128, hidden_dim=512, noise_weight=3.0):
        super().__init__()
        self.noise_weight = noise_weight
        self.rgb_encoder = efficientnet_b3(weights="DEFAULT")
        self.rgb_encoder.classifier = nn.Linear(1536, 256)  # RGB降维
        self.noise_encoder = NoiseExtractor()
        self.noise_projector = nn.Sequential(
            nn.Linear(128, 256), nn.LayerNorm(256), nn.ReLU()
        )

        self.feature_dim = 256 + 256

        # 这里的 Classifier 是旧的二分类头，之后会被我们换掉
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward_one_branch(self, img):
        # 即使在训练新头时，这里我们也只用 eval 模式，防止 BN 层统计量漂移
        f_rgb = self.rgb_encoder(img)
        raw_noise = self.noise_encoder(img)
        f_noise = self.noise_projector(raw_noise)
        f_noise = f_noise * self.noise_weight
        return torch.cat([f_rgb, f_noise], dim=1)

    def forward(self, img_center, img_cand):
        # 提取特征 (使用冻结的权重)
        v_center = self.forward_one_branch(img_center)
        v_cand = self.forward_one_branch(img_cand)

        # 特征交互
        diff = torch.abs(v_center - v_cand)
        prod = v_center * v_cand

        # 拼接 -> 进入新的分类头
        combined = torch.cat([v_center, v_cand, diff, prod], dim=1)
        return self.classifier(combined)  # 这里的 classifier 将是新的 9 分类头


# ================= 2. 拼图数据集 (9分类逻辑) =================
class JigsawDataset(Dataset):
    def __init__(self, x_path, y_path, is_train=False):
        print(f"Loading data from {x_path}...")
        self.image = np.load(x_path)
        self.label = np.load(y_path)
        self.num_samples = self.image.shape[0]
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
        # 随机找一个异源图片索引 (为了生成 Class 8 Outsider)
        rand_idx = random.randint(0, self.num_samples - 1)
        while rand_idx == host_idx:
            rand_idx = random.randint(0, self.num_samples - 1)

        # 解析 Label 获得拼图排列
        perm_onehot = self.label[index]
        perm_idx = list(np.argmax(perm_onehot, axis=1))

        final_perm = []
        for p in perm_idx:
            if p >= 4:
                final_perm.append(p + 1)
            else:
                final_perm.append(p)
        final_perm.insert(4, 4)  # 还原 9 宫格位置

        center_row, center_col = 1, 1  # 物理中心
        target_class = random.randint(0, 8)  # 0-7: Neighbors, 8: Outsider

        cand_row, cand_col = 0, 0
        use_host = True

        if target_class == 8:  # Outsider
            k = random.randint(0, 8)
            cand_row, cand_col = k // 3, k % 3
            use_host = False
        else:  # Neighbor (0-7)
            # 找到对应逻辑编号 target_class 在物理图中的位置
            target_id = target_class if target_class < 4 else target_class + 1
            try:
                current_pos = final_perm.index(target_id)
                cand_row, cand_col = current_pos // 3, current_pos % 3
                use_host = True
            except:
                # 容错
                target_class = 8
                use_host = False

        c_img = self.get_patch(self.image[host_idx], center_row, center_col)

        if use_host:
            x_img = self.get_patch(self.image[host_idx], cand_row, cand_col)
        else:
            x_img = self.get_patch(self.image[rand_idx], cand_row, cand_col)

        return c_img, x_img, torch.tensor(target_class, dtype=torch.long)


# ================= 3. 训练流程 =================
def train():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)

    # 1. 实例化模型 (结构必须和预训练时完全一样)
    # 注意：noise_weight 也要保持一致
    print("Building model...")
    model = DualStreamDiscriminator(noise_weight=3.0)

    # 2. 加载预训练权重
    if os.path.exists(PRETRAINED_DISCRIMINATOR_PATH):
        print(f"Loading frozen weights from {PRETRAINED_DISCRIMINATOR_PATH}...")
        state_dict = torch.load(PRETRAINED_DISCRIMINATOR_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(
            f"Cannot find pretrained model at {PRETRAINED_DISCRIMINATOR_PATH}"
        )

    # 3. 【核心步骤】冻结所有层
    for param in model.parameters():
        param.requires_grad = False
    print("Backbone frozen.")

    # 4. 【核心步骤】替换分类头 (Surgical Replacement)
    # 这一步会自动创建一个新的 Classifier，它的参数默认是 requires_grad=True 的
    hidden_dim = 512
    model.classifier = nn.Sequential(
        nn.Linear(model.feature_dim * 4, hidden_dim),  # 输入维度不变
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, 9),  # 输出改为 9 (0-7位置 + 8外来者)
    )
    print("New classifier head initialized (9 classes).")

    model = model.to(DEVICE)

    # 5. 优化器：只优化 requires_grad=True 的参数 (也就是只有新头)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )

    # 6. 数据准备
    train_dataset = JigsawDataset(TRAIN_DATA_PATH, TRAIN_LABEL_PATH, is_train=True)
    test_dataset = JigsawDataset(TEST_DATA_PATH, TEST_LABEL_PATH, is_train=False)

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

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler("cuda")

    print(f"Start Training Jigsaw Head on {DEVICE}...")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()  # 这里这会让新头进入训练模式，但 backbone 的 BN 层可能会受影响
        # 为了保险起见，强制 backbone 进入 eval 模式 (可选，看情况)
        model.rgb_encoder.eval()
        model.noise_encoder.eval()

        train_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for c_img, x_img, label in pbar:
            c_img, x_img, label = c_img.to(DEVICE), x_img.to(DEVICE), label.to(DEVICE)

            optimizer.zero_grad()
            with autocast("cuda"):
                logits = model(c_img, x_img)  # 此时调用的是新头
                loss = loss_fn(logits, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"}
            )

        # Validation
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
                    logits = model(c_img, x_img)
                preds = logits.argmax(dim=1)
                test_correct += (preds == label).sum().item()
                test_total += label.size(0)

                # Check Outsider Acc
                mask8 = label == 8
                if mask8.sum() > 0:
                    outsider_total += mask8.sum().item()
                    outsider_correct += (preds[mask8] == 8).sum().item()

        test_acc = test_correct / test_total
        outsider_acc = outsider_correct / outsider_total if outsider_total > 0 else 0.0

        print(
            f"Epoch {epoch+1}: Test Acc={test_acc:.4f}, Outsider Acc={outsider_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Jigsaw Model! (Acc: {best_acc:.4f})")


if __name__ == "__main__":
    train()
