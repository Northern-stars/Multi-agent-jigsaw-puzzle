import torch
import torch.nn as nn

from model_code.fen_model import fen_model,dualstem_fen_model


class DiggerModel(nn.Module):
    """Slot classifier for selecting which non-center piece should be excavated."""

    def __init__(
        self,
        hidden_size: int = 512,
        feature_hidden: int = 512,
        model_name: str = "modulator",
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.freeze_backbone = freeze_backbone
        # self.fen_model = fen_model(
        #     hidden_size1=hidden_size,
        #     hidden_size2=hidden_size,
        #     feature_hidden=feature_hidden,
        #     model_name=model_name,
        # )
        self.fen_model=dualstem_fen_model(
            hidden_size,hidden_size,feature_hidden,model_name=model_name
        )
        self.slot_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size*2,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2,8)
        )
        if self.freeze_backbone:
            for parameter in self.fen_model.parameters():
                parameter.requires_grad = False

    def forward(self, board_image: torch.Tensor, empty_mask: torch.Tensor = None) -> torch.Tensor:
        features = self.fen_model(board_image)
        logits = self.slot_head(features)
        if empty_mask is not None:
            logits = logits.masked_fill(empty_mask.bool(), -1e9)
        return logits

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.fen_model.eval()
        return self
