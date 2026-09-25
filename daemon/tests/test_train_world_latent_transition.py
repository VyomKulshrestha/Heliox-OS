"""Contract tests for the research-only latent transition probe."""

import numpy as np
import pytest

from scripts.train_world_latent_transition import (
    encode_rows,
    parse_row,
    score_predictions,
    task_group_split,
    text_vector,
)


def _row(group: str, state: str = "screen", action: str = "click",
         target: str = "window") -> dict:
    return {
        "group_sha256": group,
        "messages": [
            {"role": "system", "content": "predict"},
            {"role": "user", "content": f"Current state:\n{state}\n\nAction:\n{action}"
             "\n\nNext observed screen state:"},
            {"role": "assistant", "content": target},
        ],
    }


def test_parser_requires_unambiguous_observed_transition() -> None:
    assert parse_row(_row("a")) == ("screen", "click", "window")
    with pytest.raises(ValueError):
        parse_row(_row("a", action=""))
    with pytest.raises(ValueError):
        parse_row(_row("a", state="screen\n\nAction:\nother"))


def test_shared_text_space_and_group_disjoint_validation() -> None:
    assert np.array_equal(text_vector("Window opened", 64), text_vector("window OPENED", 64))
    rows = [_row("a"), _row("a"), _row("b"), _row("c")]
    fit, validation = task_group_split(rows, seed=7)
    assert {row["group_sha256"] for row in fit}.isdisjoint(
        {row["group_sha256"] for row in validation}
    )
    assert len(fit) + len(validation) == len(rows)
    state, action, target = encode_rows(rows[:1], 64)
    assert state.shape == action.shape == target.shape == (1, 64)


def test_retrieval_metric_compares_same_heldout_targets() -> None:
    targets = np.eye(2, dtype=np.float32)
    scores = score_predictions(targets, targets[::-1], targets)
    assert scores["unchanged_state"]["retrieval_at_1"] == 0
    assert scores["trained_transition"]["retrieval_at_1"] == 1
