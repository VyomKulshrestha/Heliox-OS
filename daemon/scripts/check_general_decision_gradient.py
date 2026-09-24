"""Check CUDA candidate replay against direct full-batch autograd."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from scripts.train_general_decision import decision_logits, gradient_row, scalar_score


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(12, 4)

    def forward(self, input_ids, **_kwargs):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids).cumsum(dim=1))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for kind, target in (("noul", [0.25, 0.75]),
                         ("choice", [0.1, 0.7, 0.2]),
                         ("score", [0.15, 0.5, 0.35])):
        torch.manual_seed(17)
        backbone = TinyBackbone().cuda()
        head = nn.Linear(4, 1).cuda()
        sequences = [[1, 2, 3]] if kind == "noul" else [[1, 2], [3, 4], [5, 6]]
        item = {"row": {"kind": kind, "target": target}, "sequences": sequences}
        gradient_row(backbone, head, item, brier_weight=0.1, scale=1.0)
        replayed = [parameter.grad.detach().clone()
                    for parameter in (*backbone.parameters(), *head.parameters())]

        backbone.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        logits = decision_logits(kind, torch.stack([scalar_score(backbone, head, ids) for ids in sequences]))
        expected = torch.tensor(target, device="cuda")
        predicted = logits.softmax(-1)
        loss = -(expected * logits.log_softmax(-1)).sum() + 0.1 * (predicted - expected).square().sum()
        loss.backward()
        for actual, parameter in zip(replayed, (*backbone.parameters(), *head.parameters()), strict=True):
            torch.testing.assert_close(actual, parameter.grad, rtol=1e-5, atol=1e-6)
        print(f"{kind}: replay matches direct gradient")


if __name__ == "__main__":
    main()
