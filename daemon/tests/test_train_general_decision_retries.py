"""Nonfinite GPU observations never skip a decision or double-step weights."""

from __future__ import annotations

import pytest

from scripts import train_general_decision as trainer

torch = pytest.importorskip("torch")


def _models():
    backbone = torch.nn.Linear(1, 1, bias=False)
    head = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW([*backbone.parameters(), *head.parameters()], lr=0.01)
    items = [{"row": {"id": "first"}}, {"row": {"id": "second"}}]
    return backbone, head, optimizer, items


def test_retry_restarts_entire_group_before_one_optimizer_step(monkeypatch) -> None:
    backbone, head, optimizer, items = _models()
    calls = []

    def intermittent(model, output, item, *, scale, **_kwargs):
        calls.append(item["row"]["id"])
        if len(calls) == 2:
            raise FloatingPointError("transient nonfinite")
        loss = (model.weight.sum() + output.weight.sum()).square()
        (loss / scale).backward()
        return float(loss.detach())

    monkeypatch.setattr(trainer, "gradient_row", intermittent)
    losses, retries = trainer.train_group(
        backbone, head, optimizer, items, [0, 1], brier_weight=0.1, ordinal_weight=0.0
    )
    assert retries == 1
    assert len(losses) == 2
    assert calls == ["first", "second", "first", "second"]
    assert all(int(state["step"]) == 1 for state in optimizer.state.values())


def test_repeated_nonfinite_group_fails_without_changing_weights(monkeypatch) -> None:
    backbone, head, optimizer, items = _models()
    previous = [parameter.detach().clone() for parameter in (*backbone.parameters(), *head.parameters())]

    def nonfinite(*_args, **_kwargs):
        raise FloatingPointError("still nonfinite")

    monkeypatch.setattr(trainer, "gradient_row", nonfinite)
    with pytest.raises(FloatingPointError, match="after 3 attempts") as failure:
        trainer.train_group(backbone, head, optimizer, items, [0], brier_weight=0.1, ordinal_weight=0.0)
    assert "first" in str(failure.value)
    assert not optimizer.state
    for before, after in zip(previous, (*backbone.parameters(), *head.parameters()), strict=True):
        torch.testing.assert_close(before, after)


def test_group_weights_reach_each_candidate_gradient(monkeypatch) -> None:
    backbone, head, optimizer, items = _models()
    observed = []

    def weighted(model, output, item, *, scale, weight, **_kwargs):
        observed.append((item["row"]["id"], weight))
        loss = (model.weight.sum() + output.weight.sum()).square()
        (loss * weight / scale).backward()
        return float(loss.detach()) * weight

    monkeypatch.setattr(trainer, "gradient_row", weighted)
    trainer.train_group(
        backbone, head, optimizer, items, [0, 1], brier_weight=0.1, ordinal_weight=0.0, row_weights=[0.25, 1.75]
    )
    assert observed == [("first", 0.25), ("second", 1.75)]
    with pytest.raises(ValueError, match="align"):
        trainer.train_group(
            backbone, head, optimizer, items, [0], brier_weight=0.1, ordinal_weight=0.0, row_weights=[1.0]
        )
