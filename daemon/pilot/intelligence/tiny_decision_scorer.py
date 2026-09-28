"""Compact byte-level listwise scorer adapted from Minimal Labs' jevlike (MIT)."""

from __future__ import annotations


def build_tiny_decision_scorer(*, width: int = 64, rank: int = 64, context_tokens: int = 192):
    """Create the optional-Torch model only for local research/evaluation runs."""
    import torch
    from torch import nn

    from .listwise_decision_head import build_listwise_decision_head

    if min(width, rank, context_tokens) < 1:
        raise ValueError("tiny scorer dimensions must be positive")

    class TinyDecisionScorer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.context_tokens = context_tokens
            self.embedding = nn.Embedding(257, width, padding_idx=0)
            self.position = nn.Embedding(context_tokens, width)
            self.head = build_listwise_decision_head(width, rank)

        def forward(
            self,
            context_ids: torch.Tensor,
            context_mask: torch.Tensor,
            option_ids: torch.Tensor,
            option_token_mask: torch.Tensor,
            option_mask: torch.Tensor,
        ) -> torch.Tensor:
            if context_ids.ndim != 2 or option_ids.ndim != 3:
                raise ValueError("byte inputs must be batch-by-context and batch-by-option-by-token")
            if context_ids.shape[1] > self.context_tokens:
                raise ValueError("context byte sequence exceeds the configured position table")
            positions = torch.arange(context_ids.shape[1], device=context_ids.device)
            context = self.embedding(context_ids) + self.position(positions)[None, :, :]
            option_tokens = self.embedding(option_ids)
            weights = option_token_mask.unsqueeze(-1).to(option_tokens.dtype)
            options = (option_tokens * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1)
            return self.head(context, options, context_mask, option_mask)

    return TinyDecisionScorer()
