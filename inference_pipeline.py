import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# 引入模型定义
from train_sorter_8class import SorterModel

try:
    from discriminator_context.train_discriminator_final_weighted_noise import (
        DualStreamDiscriminator,
    )
except Exception:
    from discriminator_context.train_discriminator_final import DualStreamDiscriminator


class MultiPuzzleSolver:
    def __init__(self, gatekeeper_path, sorter_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 1. 加载门卫
        print("Loading Gatekeeper...")
        self.gatekeeper = DualStreamDiscriminator(noise_weight=3.0).to(self.device)
        try:
            self.gatekeeper.load_state_dict(
                torch.load(gatekeeper_path, map_location=self.device)
            )
        except:
            self.gatekeeper.load_state_dict(
                torch.load(gatekeeper_path, map_location=self.device), strict=False
            )
        self.gatekeeper.eval()

        # 2. 加载排序工
        print("Loading Sorter...")
        self.sorter = SorterModel().to(self.device)
        self.sorter.load_state_dict(torch.load(sorter_path, map_location=self.device))
        self.sorter.eval()

        # 物理位置映射 (0-7 -> 0-8)
        self.pos_map = [0, 1, 2, 3, 5, 6, 7, 8]

    def preprocess(self, patch_list):
        """把 patch list 转成 batch tensor"""
        tensors = [torch.tensor(p).permute(2, 0, 1).float() for p in patch_list]
        batch = torch.stack(tensors)
        if batch.max() > 1.0:
            batch /= 255.0
        return batch.to(self.device)

    def solve(self, center_A, center_B, mixed_candidates):
        """
        输入:
            center_A: (96,96,3) numpy
            center_B: (96,96,3) numpy
            mixed_candidates: list of 16 numpy arrays (96,96,3)
        输出:
            pred_grid_A: list of 9 patches
            pred_grid_B: list of 9 patches
        """
        num_candidates = len(mixed_candidates)

        # 1. 预处理数据（中心固定，候选为两张图除中心外的全部块，已打乱）
        c_A_tensor = self.preprocess([center_A]).repeat(
            num_candidates, 1, 1, 1
        )  # [16, 3, 96, 96]
        c_B_tensor = self.preprocess([center_B]).repeat(
            num_candidates, 1, 1, 1
        )  # [16, 3, 96, 96]
        x_tensor = self.preprocess(mixed_candidates)  # [16, 3, 96, 96]

        with torch.no_grad():
            # ==========================================
            # Step 1: 归属权争夺 (Gatekeeper Competition)
            # ==========================================
            # 计算属于 A 的概率
            logits_A = self.gatekeeper(c_A_tensor, x_tensor)
            probs_A = torch.sigmoid(logits_A).squeeze()  # [16]

            # 计算属于 B 的概率
            logits_B = self.gatekeeper(c_B_tensor, x_tensor)
            probs_B = torch.sigmoid(logits_B).squeeze()  # [16]

            # 记录分配结果
            assignments = (
                []
            )  # 存 dict: {'owner': 'A'/'B', 'patch_idx': i, 'gate_score': float}

            for i in range(num_candidates):
                score_a = probs_A[i].item()
                score_b = probs_B[i].item()

                # 谁的分高跟谁走 (Winner Takes All)
                if score_a > score_b:
                    owner = "A"
                    score = score_a
                else:
                    owner = "B"
                    score = score_b

                assignments.append(
                    {"patch_idx": i, "owner": owner, "gate_score": score}
                )

            # ==========================================
            # Step 2: 组内排序 (Sorting)
            # ==========================================
            # 分组
            group_A_indices = [
                item["patch_idx"] for item in assignments if item["owner"] == "A"
            ]
            group_B_indices = [
                item["patch_idx"] for item in assignments if item["owner"] == "B"
            ]

            # 准备结果容器 (9个空位, 中间填好)
            grid_A = [None] * 9
            grid_A[4] = center_A
            grid_B = [None] * 9
            grid_B[4] = center_B
            used_indices = set()

            # --- 处理 Group A ---
            if group_A_indices:
                # 选出属于 A 的 candidate tensors
                valid_x_A = x_tensor[group_A_indices]
                # 对应的 Center A (只需数量匹配)
                valid_c_A = c_A_tensor[: len(group_A_indices)]

                # 预测位置 0-7
                sort_logits = self.sorter(valid_c_A, valid_x_A)
                # 获取位置和置信度
                probs = torch.softmax(sort_logits, dim=1)
                best_class = probs.argmax(dim=1)
                confidences = probs.max(dim=1).values

                # 按置信度排序再填充，减少冲突
                order = torch.argsort(confidences, descending=True)
                for ord_i in order:
                    idx = group_A_indices[ord_i]
                    cls_id = best_class[ord_i]
                    pos = self.pos_map[cls_id.item()]
                    if grid_A[pos] is None:
                        grid_A[pos] = mixed_candidates[idx]
                        used_indices.add(idx)
                    else:
                        # 冲突！为了演示简单，找个空位塞进去，或者丢弃
                        # 实际工业级代码需要匈牙利算法(Hungarian Algorithm)
                        for k in range(9):
                            if grid_A[k] is None:
                                grid_A[k] = mixed_candidates[idx]
                                used_indices.add(idx)
                                break

            # --- 处理 Group B ---
            if group_B_indices:
                valid_x_B = x_tensor[group_B_indices]
                valid_c_B = c_B_tensor[: len(group_B_indices)]

                sort_logits = self.sorter(valid_c_B, valid_x_B)
                probs = torch.softmax(sort_logits, dim=1)
                best_class = probs.argmax(dim=1)
                confidences = probs.max(dim=1).values

                order = torch.argsort(confidences, descending=True)
                for ord_i in order:
                    idx = group_B_indices[ord_i]
                    cls_id = best_class[ord_i]
                    pos = self.pos_map[cls_id.item()]
                    if grid_B[pos] is None:
                        grid_B[pos] = mixed_candidates[idx]
                        used_indices.add(idx)
                    else:
                        for k in range(9):
                            if grid_B[k] is None:
                                grid_B[k] = mixed_candidates[idx]
                                used_indices.add(idx)
                                break

            # 将未被放置的碎片补到空位，确保显示门卫分错的结果而不是黑块
            remaining_indices = [
                i for i in range(num_candidates) if i not in used_indices
            ]
            if remaining_indices:
                # 先补 A
                for k in range(9):
                    if k == 4:
                        continue
                    if grid_A[k] is None and remaining_indices:
                        idx = remaining_indices.pop(0)
                        grid_A[k] = mixed_candidates[idx]

                # 再补 B
                for k in range(9):
                    if k == 4:
                        continue
                    if grid_B[k] is None and remaining_indices:
                        idx = remaining_indices.pop(0)
                        grid_B[k] = mixed_candidates[idx]

        return grid_A, grid_B


def stitch_image(patches, size=96):
    """把9个patch拼成一张大图"""
    canvas = np.zeros((size * 3, size * 3, 3), dtype=np.uint8)
    for i, patch in enumerate(patches):
        if patch is None:
            continue  # 留黑
        r, c = divmod(i, 3)
        canvas[r * size : (r + 1) * size, c * size : (c + 1) * size, :] = patch
    return canvas


# ================= 主程序 =================
if __name__ == "__main__":
    GATE_PATH = "model_discriminator/discriminator_srm_best.pth"
    SORT_PATH = "model_sorter/sorter_8class_best.pth"

    # 验证集路径
    VALID_X = "dataset/test_img_48gap_33.npy"
    VALID_Y = "dataset/test_label_48gap_33.npy"

    if os.path.exists(VALID_X) and os.path.exists(VALID_Y):
        # 1. 准备数据
        data = np.load(VALID_X)
        labels = np.load(VALID_Y)

        # 随机抽两张不同的图
        idx1, idx2 = np.random.choice(len(data), 2, replace=False)
        img1 = data[idx1]
        img2 = data[idx2]
        label1 = labels[idx1]
        label2 = labels[idx2]

        print(f"Mixing Puzzle A (Idx {idx1}) and Puzzle B (Idx {idx2})...")

        # 切片函数
        def slice_img(img):
            patches = []
            for r in range(3):
                for c in range(3):
                    patches.append(img[r * 96 : (r + 1) * 96, c * 96 : (c + 1) * 96])
            return patches

        patches1 = slice_img(img1)
        patches2 = slice_img(img2)

        def reconstruct_gt(img, label):
            """使用标签将打乱图像重排为 Ground Truth"""
            patches = slice_img(img)
            perm_idx = list(np.argmax(label, axis=1))
            for i in range(len(perm_idx)):
                if perm_idx[i] >= 4:
                    perm_idx[i] += 1
            perm_idx.insert(4, 4)

            gt = np.zeros_like(img)
            for cur_pos in range(9):
                target_pos = perm_idx[cur_pos]
                r, c = divmod(target_pos, 3)
                gt[r * 96 : (r + 1) * 96, c * 96 : (c + 1) * 96, :] = patches[cur_pos]
            return gt

        # 提取中心
        center_A = patches1[4]
        center_B = patches2[4]

        # 混合剩下的 16 个块 (8 from A, 8 from B)，中心不动
        candidates_A = [p for i, p in enumerate(patches1) if i != 4]
        candidates_B = [p for i, p in enumerate(patches2) if i != 4]
        mixed_bag = candidates_A + candidates_B

        # 打乱混合包 (模拟真实情况)
        np.random.shuffle(mixed_bag)
        print(f"Total candidates to sort: {len(mixed_bag)}")

        # 2. 求解
        solver = MultiPuzzleSolver(GATE_PATH, SORT_PATH)
        pred_A_list, pred_B_list = solver.solve(center_A, center_B, mixed_bag)

        # 3. 拼图展示
        res_A = stitch_image(pred_A_list)
        res_B = stitch_image(pred_B_list)
        gt_A = reconstruct_gt(img1, label1)
        gt_B = reconstruct_gt(img2, label2)

        # 4. 画图
        plt.figure(figsize=(10, 8))

        plt.subplot(2, 2, 1)
        plt.imshow(gt_A)
        plt.title(f"Ground Truth A (Idx {idx1})")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.imshow(gt_B)
        plt.title(f"Ground Truth B (Idx {idx2})")
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.imshow(res_A)
        plt.title("Recovered Puzzle A")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.imshow(res_B)
        plt.title("Recovered Puzzle B")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    else:
        print("Dataset not found.")
