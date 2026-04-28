import os
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_b0
from tqdm import tqdm


import sys

sys.path.append(".")
sys.path.append("..")

import model_code.Vit as Vit
from model_code.fen_model import fen_model
from model_code.fen_model import attention_fen_model



TRAIN_X_PATH = "dataset/train_img_48gap_33-001.npy"
TRAIN_Y_PATH = "dataset/train_label_48gap_33.npy"
TEST_X_PATH = "dataset/test_img_48gap_33.npy"
TEST_Y_PATH = "dataset/test_label_48gap_33.npy"

MODEL_NAME = "attn_fen"
MODEL_PATH = os.path.join("model", f"swap_classifier_pretrain_{MODEL_NAME}.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 25
LEARNING_RATE = 1e-4
EPOCHS = 50
NUM_WORKERS = 0
LOAD_MODEL = False
TEST_ONLY = False

CLASS_NAMES = ["ordered", "inner_swap", "outer_replace"]
MOVABLE_INDICES = [0, 1, 2, 3, 5, 6, 7, 8]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode_label(label_8x8: np.ndarray) -> List[int]:
    piece_order = list(np.argmax(label_8x8, axis=1))
    for idx, value in enumerate(piece_order):
        if value >= 4:
            piece_order[idx] = value + 1
    piece_order.insert(4, 4)
    return piece_order


def split_image_to_pieces(image: torch.Tensor) -> List[torch.Tensor]:
    return [
        image[:, 0:96, 0:96],
        image[:, 0:96, 96:192],
        image[:, 0:96, 192:288],
        image[:, 96:192, 0:96],
        image[:, 96:192, 96:192],
        image[:, 96:192, 192:288],
        image[:, 192:288, 0:96],
        image[:, 192:288, 96:192],
        image[:, 192:288, 192:288],
    ]


def rebuild_image_from_pieces(pieces: Sequence[torch.Tensor]) -> torch.Tensor:
    image = torch.zeros(3, 288, 288, dtype=pieces[0].dtype)
    for piece_idx, piece in enumerate(pieces):
        row, col = divmod(piece_idx, 3)
        image[:, row * 96 : (row + 1) * 96, col * 96 : (col + 1) * 96] = piece
    return image


def build_ordered_pieces(image_np: np.ndarray, label_8x8: np.ndarray) -> List[torch.Tensor]:
    # Keep the cache in uint8 to reduce memory usage; normalization happens in the model.
    image = torch.tensor(image_np, dtype=torch.float).permute(2, 0, 1)
    raw_pieces = split_image_to_pieces(image)
    piece_order = decode_label(label_8x8)

    ordered_pieces: List[torch.Tensor] = [torch.zeros_like(raw_pieces[0]) for _ in range(9)]
    for shuffled_slot, target_slot in enumerate(piece_order):
        ordered_pieces[target_slot] = raw_pieces[shuffled_slot]
    return ordered_pieces


def apply_inner_swap(pieces: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    swapped = [piece.clone() for piece in pieces]
    idx_a, idx_b = random.sample(MOVABLE_INDICES, 2)
    swapped[idx_a], swapped[idx_b] = swapped[idx_b], swapped[idx_a]
    return swapped


def apply_outer_replace(
    pieces: Sequence[torch.Tensor],
    guest_pieces: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    replaced = [piece.clone() for piece in pieces]
    local_idx = random.choice(MOVABLE_INDICES)
    guest_idx = random.choice(MOVABLE_INDICES)
    replaced[local_idx] = guest_pieces[guest_idx].clone()
    return replaced


class PuzzleTypeClassificationDataset(Dataset):
    def __init__(self, data: np.ndarray, label: np.ndarray) -> None:
        self.data = data
        self.label = label
        self.ordered_piece_cache = [None for _ in range(self.data.shape[0])]

    def _preload_ordered_pieces(self) -> List[Tuple[torch.Tensor, ...]]:
        ordered_cache: List[Tuple[torch.Tensor, ...]] = []
        for index in tqdm(range(len(self.data)), desc="preload_ordered_pieces"):
            ordered_pieces = build_ordered_pieces(self.data[index], self.label[index])
            ordered_cache.append(tuple(ordered_pieces))
        return ordered_cache

    def __len__(self) -> int:
        return len(self.data)
    
    def _get_cached_pieces(self,index):
        if self.ordered_piece_cache[index] is None:
            self.ordered_piece_cache[index]=tuple(build_ordered_pieces(self.data[index],self.label[index]))
        return self.ordered_piece_cache[index]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        ordered_pieces =self._get_cached_pieces(index)
        task_type = random.randint(0, 2)

        if task_type == 0:
            transformed_pieces = [piece.clone() for piece in ordered_pieces]
        elif task_type == 1:
            transformed_pieces = apply_inner_swap(ordered_pieces)
        else:
            guest_idx = random.randint(0, len(self.data) - 1)
            while guest_idx == index:
                guest_idx = random.randint(0, len(self.data) - 1)
            guest_pieces = self._get_cached_pieces(guest_idx)
            transformed_pieces = apply_outer_replace(ordered_pieces, guest_pieces)

        transformed_image = rebuild_image_from_pieces(transformed_pieces)
        return transformed_image.float(), torch.tensor(task_type)


class PuzzleTypeClassifier(nn.Module):
    def __init__(self, hidden_size: int = 512, model_name: str = "ef", dropout: float = 0.3) -> None:
        super().__init__()
        self.model_name = model_name

        if model_name == "ef_sole":
            self.backbone = efficientnet_b0(weights="DEFAULT")
            self.backbone.classifier = nn.Linear(1280, hidden_size)
            feature_dim = hidden_size
        elif model_name=="ef":
            self.backbone=fen_model(hidden_size1=hidden_size,hidden_size2=hidden_size,model_name=model_name)
            feature_dim=hidden_size
        elif model_name == "modulator":
            self.backbone = fen_model(hidden_size1=hidden_size,hidden_size2=hidden_size,model_name=model_name)
            feature_dim = hidden_size
        elif model_name == "vit":
            self.backbone = Vit.VisionTransformer(
                picture_size=[1, 3, 288, 288],
                patch_size=16,
                encoder_hidden=hidden_size,
                out_size=hidden_size,
                n_head=4,
                encoder_layer_num=4,
                unet_hidden=hidden_size,
                output_channel=3,
            )
            feature_dim = hidden_size
        elif model_name == "attn_fen":
            self.backbone = attention_fen_model(
                embed_dim=256,
                num_heads=4,
                num_layers=5,
                dropout=dropout,
                single_output=True,
                project_hidden=hidden_size
                )
            
            feature_dim = hidden_size
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float() / 255.0
        feature = self.backbone(image)
        logits = self.classifier(feature)
        return logits


def build_dataloaders() -> Tuple[DataLoader, DataLoader]:
    train_x = np.load(TRAIN_X_PATH)
    train_y = np.load(TRAIN_Y_PATH)
    test_x = np.load(TEST_X_PATH)
    test_y = np.load(TEST_Y_PATH)

    train_dataset = PuzzleTypeClassificationDataset(train_x, train_y)
    test_dataset = PuzzleTypeClassificationDataset(test_x, test_y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    model.train()
    loss_sum = 0.0
    sample_count = 0

    for images, labels in tqdm(dataloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        loss_sum += loss.item() * batch_size
        sample_count += batch_size

    return loss_sum / max(sample_count, 1)


def evaluate_classification(model: nn.Module, dataloader: DataLoader) -> Dict[str, object]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(DEVICE)
            logits = model(images)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    labels_np = np.array(all_labels)
    preds_np = np.array(all_preds)

    accuracy = float((labels_np == preds_np).mean())
    macro_f1 = float(f1_score(labels_np, preds_np, average="macro"))
    macro_precision = float(precision_score(labels_np, preds_np, average="macro", zero_division=0))
    macro_recall = float(recall_score(labels_np, preds_np, average="macro", zero_division=0))
    cm = confusion_matrix(labels_np, preds_np, labels=[0, 1, 2])

    per_class_precision = precision_score(
        labels_np,
        preds_np,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    per_class_recall = recall_score(
        labels_np,
        preds_np,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    per_class_f1 = f1_score(
        labels_np,
        preds_np,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    result = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "confusion_matrix": cm,
        "per_class": {},
    }

    for class_idx, class_name in enumerate(CLASS_NAMES):
        result["per_class"][class_name] = {
            "precision": float(per_class_precision[class_idx]),
            "recall": float(per_class_recall[class_idx]),
            "f1": float(per_class_f1[class_idx]),
        }
    return result


def print_metrics(metrics: Dict[str, object]) -> None:
    print("\n" + "=" * 60)
    print("Swap Classifier Evaluation")
    print("=" * 60)
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro F1:        {metrics['macro_f1']:.4f}")
    print("\nPer-class metrics:")
    for class_name in CLASS_NAMES:
        class_metric = metrics["per_class"][class_name]
        print(
            f"{class_name:>14} | "
            f"precision={class_metric['precision']:.4f} "
            f"recall={class_metric['recall']:.4f} "
            f"f1={class_metric['f1']:.4f}"
        )
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])
    print("=" * 60)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_acc: float) -> None:
    os.makedirs("model", exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "model_name": MODEL_NAME,
        },
        MODEL_PATH,
    )


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer = None) -> float:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return float(checkpoint.get("best_acc", 0.0))


def train_classification(epoch_num: int = EPOCHS, load: bool = False) -> None:
    train_loader, test_loader = build_dataloaders()
    model = PuzzleTypeClassifier(hidden_size=512, model_name=MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    if load and os.path.exists(MODEL_PATH):
        best_acc = load_checkpoint(model, optimizer)

    print("start training swap classifier pretrain")
    for epoch in range(epoch_num):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        metrics = evaluate_classification(model, test_loader)

        print(
            f"Epoch {epoch + 1}/{epoch_num} | "
            f"train_loss={train_loss:.4f} | "
            f"test_acc={metrics['accuracy']:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

        if metrics["accuracy"] >= best_acc:
            best_acc = float(metrics["accuracy"])
            save_checkpoint(model, optimizer, epoch, best_acc)
            print(f"best model updated, acc={best_acc:.4f}")

    print("training finished")


def test_classification(load: bool = True) -> Dict[str, object]:
    _, test_loader = build_dataloaders()
    model = PuzzleTypeClassifier(hidden_size=512, model_name=MODEL_NAME).to(DEVICE)

    if load:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")
        load_checkpoint(model)

    metrics = evaluate_classification(model, test_loader)
    print_metrics(metrics)
    return metrics


if __name__ == "__main__":
    set_seed(42)
    if TEST_ONLY:
        test_classification(load=True)
    else:
        train_classification(epoch_num=EPOCHS, load=LOAD_MODEL)
        test_classification(load=True)
