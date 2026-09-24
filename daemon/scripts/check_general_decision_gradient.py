"""Check CUDA candidate replay against direct full-batch autograd."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from scripts.train_general_decision import (
    decision_logits,
    gradient_row,
    load_checkpoint,
    save_checkpoint,
    scalar_score,
)


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

    optimizer = torch.optim.AdamW([*backbone.parameters(), *head.parameters()])
    optimizer.step()
    expected = [parameter.detach().clone() for parameter in (*backbone.parameters(), *head.parameters())]
    expected_moments = [state["exp_avg"].detach().clone() for state in optimizer.state.values()]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pt"
        save_checkpoint(path, backbone, head, optimizer, signature="tiny-run",
                        next_position=4, elapsed_seconds=1.25, peak_vram_gib=0.1)
        with torch.no_grad():
            for parameter in (*backbone.parameters(), *head.parameters()):
                parameter.add_(1)
            for state in optimizer.state.values():
                state["exp_avg"].add_(1)
        try:
            load_checkpoint(path, backbone, head, optimizer, signature="wrong-run", total=8)
        except ValueError:
            pass
        else:
            raise AssertionError("Checkpoint signature mismatch was accepted")
        assert load_checkpoint(path, backbone, head, optimizer,
                               signature="tiny-run", total=8) == (4, 1.25, 0.1)
        for actual, restored in zip(expected, (*backbone.parameters(), *head.parameters()), strict=True):
            torch.testing.assert_close(actual, restored)
        for actual, state in zip(expected_moments, optimizer.state.values(), strict=True):
            torch.testing.assert_close(actual, state["exp_avg"])
    print("checkpoint: trainable parameters and optimizer restored")


if __name__ == "__main__":
    main()
