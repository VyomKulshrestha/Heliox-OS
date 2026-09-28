"""Small cross-attention head for frozen-context, listwise decisions.

The option-to-context attention pattern is adapted from Dasein Labs' Open-Jev
AttentionHead (MIT; copyright notice is in benchmarks/OPEN_JEV_LICENSE.txt).
"""

from __future__ import annotations


def build_listwise_decision_head(hidden_size: int, rank: int = 512):
    """Build the optional-Torch module only when a local research run asks for it."""
    import math

    import torch
    from torch import nn

    class ListwiseDecisionHead(nn.Module):
        """Score padded or unpadded option sets against ordered context features."""

        def __init__(self) -> None:
            super().__init__()
            if hidden_size < 1 or rank < 1:
                raise ValueError("hidden_size and rank must be positive")
            self.hidden_size = hidden_size
            self.rank = rank
            self.context_norm = nn.LayerNorm(hidden_size)
            self.option_norm = nn.LayerNorm(hidden_size)
            self.query = nn.Linear(hidden_size, rank, bias=False)
            self.key = nn.Linear(hidden_size, rank, bias=False)
            self.value = nn.Linear(hidden_size, rank, bias=False)

        def forward(
            self,
            context: torch.Tensor,
            options: torch.Tensor,
            context_mask: torch.Tensor | None = None,
            option_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            squeeze = context.ndim == 2 and options.ndim == 2
            if squeeze:
                context = context.unsqueeze(0)
                options = options.unsqueeze(0)
                if context_mask is not None:
                    context_mask = context_mask.unsqueeze(0)
                if option_mask is not None:
                    option_mask = option_mask.unsqueeze(0)
            if context.ndim != 3 or options.ndim != 3:
                raise ValueError("context and options must be batch-by-token-by-hidden tensors")
            if context.shape[-1] != self.hidden_size or options.shape[-1] != self.hidden_size:
                raise ValueError("context and option hidden sizes must match the head")
            if context.shape[1] < 1 or options.shape[1] < 2 or context.shape[0] != options.shape[0]:
                raise ValueError("each decision needs context and at least two options")

            if context_mask is None:
                context_mask = torch.ones(context.shape[:2], dtype=torch.bool, device=context.device)
            else:
                context_mask = context_mask.to(device=context.device, dtype=torch.bool)
            if option_mask is None:
                option_mask = torch.ones(options.shape[:2], dtype=torch.bool, device=options.device)
            else:
                option_mask = option_mask.to(device=options.device, dtype=torch.bool)
            if context_mask.shape != context.shape[:2] or option_mask.shape != options.shape[:2]:
                raise ValueError("padding masks must match batch and token dimensions")
            if not bool(context_mask.any(dim=-1).all()) or bool((option_mask.sum(dim=-1) < 2).any()):
                raise ValueError("each row needs at least one context token and two valid options")

            context = self.context_norm(context.float())
            options = self.option_norm(options.float())
            query = self.query(options)
            key = self.key(context)
            value = self.value(context)
            scores = (query @ key.transpose(-1, -2)) / math.sqrt(rank)
            scores = scores.masked_fill(~context_mask[:, None, :], torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1)
            attended = weights @ value
            logits = (query * attended).sum(dim=-1) / math.sqrt(rank)
            logits = logits.masked_fill(~option_mask, torch.finfo(logits.dtype).min)
            return logits[0] if squeeze else logits

    return ListwiseDecisionHead()
