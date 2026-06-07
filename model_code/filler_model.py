import torch
import torch.nn as nn

from model_code.fen_model import fen_model,dualstem_fen_model


class FillerModel(nn.Module):
    """Candidate board scorer used by the filler assignment strategies."""

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
        self.score_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size*2,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2,1)
        )
        if self.freeze_backbone:
            for parameter in self.fen_model.parameters():
                parameter.requires_grad = False

    def forward(self, candidate_complete_image: torch.Tensor, empty_mask: torch.Tensor = None) -> torch.Tensor:
        features = self.fen_model(candidate_complete_image)
        score = self.score_head(features).squeeze(-1)
        if empty_mask is not None:
            score = score.masked_fill(empty_mask.bool().view(-1), -1e9)
        return score

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.fen_model.eval()
        return self
