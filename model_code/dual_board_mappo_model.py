from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torchvision.models import efficientnet_b0


MASK_VALUE = -1e9


class PieceEncoder(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.backbone = efficientnet_b0(weights="DEFAULT")
        self.backbone.classifier=nn.Linear(1280,embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images.float()
        x = self.backbone(x)

        return x


class BoardStateEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_layers: int = 3, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.position_embedding = nn.Parameter(torch.randn(1, 9, embed_dim) * 0.02)
        self.summary_norm = nn.LayerNorm(embed_dim)

    def forward(self, slot_tokens: torch.Tensor, anchor_token: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = torch.cat([slot_tokens[:, :4], anchor_token.unsqueeze(1), slot_tokens[:, 4:]], dim=1)
        encoded = self.transformer(tokens + self.position_embedding)
        slot_context = torch.cat([encoded[:, :4], encoded[:, 5:]], dim=1)
        board_summary = self.summary_norm(encoded.mean(dim=1))
        return {"slot_context": slot_context, "board_summary": board_summary}


class PointerActor(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.ptr1_head = nn.Linear(embed_dim, 1)
        self.query_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.outside_proj = nn.Linear(embed_dim, embed_dim)
        self.slot_key = nn.Linear(embed_dim, embed_dim)

    def pointer1_logits(self, slot_context: torch.Tensor) -> torch.Tensor:
        return self.ptr1_head(slot_context).squeeze(-1)

    def pointer2_logits(
        self,
        slot_context: torch.Tensor,
        board_summary: torch.Tensor,
        selected_source: torch.Tensor,
        other_message: torch.Tensor,
        source_index: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query_fusion(torch.cat([selected_source, board_summary], dim=-1))
        slot_keys = self.slot_key(slot_context)
        outside_key = self.outside_proj(other_message).unsqueeze(1)
        candidate_keys = torch.cat([slot_keys, outside_key], dim=1)
        logits = torch.einsum("bd,bnd->bn", query, candidate_keys)

        source_mask = torch.zeros_like(logits, dtype=torch.bool)
        source_mask.scatter_(1, source_index.unsqueeze(1), True)
        logits = logits.masked_fill(source_mask, MASK_VALUE)
        return logits


class CentralizedCritic(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, board_summary_a: torch.Tensor, board_summary_b: torch.Tensor) -> torch.Tensor:
        return self.value_head(torch.cat([board_summary_a, board_summary_b], dim=-1)).squeeze(-1)


class DualBoardMAPPOModel(nn.Module):
    def __init__(self, embed_dim: int = 128, num_layers: int = 3, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.piece_encoder = PieceEncoder(embed_dim)
        self.board_encoder = BoardStateEncoder(embed_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout)
        self.actor = PointerActor(embed_dim)
        self.critic = CentralizedCritic(embed_dim)

    def encode_boards(self, slot_images: torch.Tensor, anchor_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = slot_images.size(0)
        slot_embeddings = self.piece_encoder(slot_images.view(batch_size * 2 * 8, 3, 96, 96)).view(batch_size, 2, 8, -1)
        anchor_embeddings = self.piece_encoder(anchor_images.view(batch_size * 2, 3, 96, 96)).view(batch_size, 2, -1)

        board_outputs = []
        for board_index in range(2):
            board_outputs.append(self.board_encoder(slot_embeddings[:, board_index], anchor_embeddings[:, board_index]))

        return {
            "slot_context": torch.stack([board_outputs[0]["slot_context"], board_outputs[1]["slot_context"]], dim=1),
            "board_summary": torch.stack([board_outputs[0]["board_summary"], board_outputs[1]["board_summary"]], dim=1),
        }

    def evaluate_policy(
        self,
        slot_images: torch.Tensor,
        anchor_images: torch.Tensor,
        ptr1_actions: Optional[torch.Tensor] = None,
        ptr2_actions: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode_boards(slot_images, anchor_images)
        slot_context = encoded["slot_context"]
        board_summary = encoded["board_summary"]

        ptr1_logits = torch.stack(
            [
                self.actor.pointer1_logits(slot_context[:, 0]),
                self.actor.pointer1_logits(slot_context[:, 1]),
            ],
            dim=1,
        )
        ptr1_dist = Categorical(logits=ptr1_logits)

        if ptr1_actions is None:
            if deterministic:
                ptr1_actions = ptr1_logits.argmax(dim=-1)
            else:
                ptr1_actions = ptr1_dist.sample()

        selected_source = torch.gather(
            slot_context,
            2,
            ptr1_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, slot_context.size(-1)),
        ).squeeze(2)

        ptr2_logits = torch.stack(
            [
                self.actor.pointer2_logits(
                    slot_context[:, 0],
                    board_summary[:, 0],
                    selected_source[:, 0],
                    selected_source[:, 1],
                    ptr1_actions[:, 0],
                ),
                self.actor.pointer2_logits(
                    slot_context[:, 1],
                    board_summary[:, 1],
                    selected_source[:, 1],
                    selected_source[:, 0],
                    ptr1_actions[:, 1],
                ),
            ],
            dim=1,
        )
        ptr2_dist = Categorical(logits=ptr2_logits)

        if ptr2_actions is None:
            if deterministic:
                ptr2_actions = ptr2_logits.argmax(dim=-1)
            else:
                ptr2_actions = ptr2_dist.sample()

        log_prob_1 = ptr1_dist.log_prob(ptr1_actions)
        log_prob_2 = ptr2_dist.log_prob(ptr2_actions)
        entropy = ptr1_dist.entropy() + ptr2_dist.entropy()
        value = self.critic(board_summary[:, 0], board_summary[:, 1])
        outside_prob = ptr2_dist.probs[..., 8]

        return {
            "ptr1_logits": ptr1_logits,
            "ptr2_logits": ptr2_logits,
            "ptr1_actions": ptr1_actions,
            "ptr2_actions": ptr2_actions,
            "log_prob": log_prob_1 + log_prob_2,
            "entropy": entropy,
            "value": value,
            "outside_prob": outside_prob,
        }
