from typing import Dict, Optional

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import sys
sys.path.append("..")
sys.path.append(".")
from model_code.fen_model import attention_fen_model as fen_model

MASK_VALUE = -1e9

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
        self.board_fen = fen_model(embed_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout)
        self.actor = PointerActor(embed_dim)
        self.critic = CentralizedCritic(embed_dim)

    def encode_boards(self, board_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = board_images.size(0)
        flat_board_images = board_images.view(batch_size * 2, 3, 288, 288)
        encoded_boards = self.board_fen(flat_board_images)
        return {
            "token_context": encoded_boards["token_context"].view(batch_size, 2, 9, -1),
            "slot_context": encoded_boards["slot_context"].view(batch_size, 2, 8, -1),
            "board_summary": encoded_boards["board_summary"].view(batch_size, 2, -1),
        }

    def evaluate_policy(
        self,
        board_images: Optional[torch.Tensor],
        ptr1_actions: Optional[torch.Tensor] = None,
        encoded: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if encoded is None:
            if board_images is None:
                raise ValueError("board_images must be provided when encoded features are not supplied.")
            encoded = self.encode_boards(board_images)
        slot_context = encoded["slot_context"]
        board_summary = encoded["board_summary"]

        ptr1_logits = torch.stack(
            [
                self.actor.pointer1_logits(slot_context[:, 0]),
                self.actor.pointer1_logits(slot_context[:, 1]),
            ],
            dim=1,
        )
        ptr1_probs = torch.softmax(ptr1_logits, dim=-1)
        value = self.critic(board_summary[:, 0], board_summary[:, 1])

        result = {
            "encoded": encoded,
            "ptr1_logits": ptr1_logits,
            "ptr1_probs": ptr1_probs,
            "value": value,
        }
        if ptr1_actions is None:
            return result

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
        ptr2_probs = torch.softmax(ptr2_logits, dim=-1)
        result.update(
            {
                "ptr2_logits": ptr2_logits,
                "ptr2_probs": ptr2_probs,
                "outside_prob": ptr2_probs[..., 8],
            }
        )
        return result
