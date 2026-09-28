from __future__ import annotations

import numpy as np
import pytest

from scripts.train_cross_attention_typed_decision import (
    _context_text,
    _pool_context,
    _read_feature_cache,
    _write_feature_cache,
)
from scripts.train_tiny_typed_decision import _bytes, _jev_metrics


def test_context_text_excludes_candidates_labels_and_provenance() -> None:
    row = {
        "id": "private-id",
        "kind": "choice",
        "question": "Which window is active?",
        "state_json": '{"window":"Settings"}',
        "options": ["A-secret-candidate", "B-secret-candidate"],
        "target": [1.0, 0.0],
        "source": "private-source",
    }
    rendered = _context_text(row)
    assert "Which window is active?" in rendered
    assert "Settings" in rendered
    assert "A-secret-candidate" not in rendered
    assert "B-secret-candidate" not in rendered
    assert "private-id" not in rendered
    assert "private-source" not in rendered
    assert "1.0" not in rendered


def test_context_pooling_preserves_sequence_order_and_bounds_cache_size() -> None:
    values = np.arange(20, dtype=np.float32).reshape(10, 2)
    pooled = _pool_context(values, 5)
    assert pooled.shape == (5, 2)
    assert pooled.dtype == np.float16
    assert np.all(np.diff(pooled[:, 0]) > 0)
    assert np.array_equal(_pool_context(values[:3], 5), values[:3].astype(np.float16))


def test_byte_encoding_is_utf8_bounded_and_zero_remains_padding() -> None:
    assert _bytes("A", 4) == [66]
    assert _bytes("é", 1) == [196]
    assert _bytes("long", 2) == [109, 112]


def test_expected_calibration_error_is_not_normalized_twice() -> None:
    rows = [
        {"source": "test", "kind": "choice", "target": [1.0, 0.0]},
        {"source": "test", "kind": "choice", "target": [1.0, 0.0]},
    ]
    result = _jev_metrics(rows, [np.asarray([2.0, 0.0])] * 2, 1.0)
    expected = 1.0 - (1.0 / (1.0 + np.exp(-2.0)))
    assert result["ece_10bin"] == pytest.approx(expected)
    assert result["calibration_bins"] == [
        {
            "count": 2,
            "mean_confidence": pytest.approx(1.0 - expected),
            "accuracy": 1.0,
        }
    ]


def test_uniform_scores_use_expected_random_choice_accuracy() -> None:
    rows = [
        {
            "source": "test",
            "kind": "choice",
            "options": ["a", "b", "c", "d"],
            "target": [0.0, 0.0, 1.0, 0.0],
        }
    ]
    result = _jev_metrics(rows, [np.zeros(4)], 1.0)
    assert result["top1_accuracy"] == pytest.approx(0.25)
    assert result["top3_accuracy"] == pytest.approx(0.75)
    assert result["top1_accuracy_interpretation"] == "uniform_expected"
    assert result["by_kind"]["choice"]["top1_accuracy"] == pytest.approx(0.25)


def test_cross_attention_head_supports_variable_options_and_is_permutation_equivariant() -> None:
    torch = pytest.importorskip("torch")
    from pilot.intelligence.listwise_decision_head import build_listwise_decision_head

    torch.manual_seed(9)
    head = build_listwise_decision_head(hidden_size=12, rank=7).eval()
    context = torch.randn(5, 12)
    options = torch.randn(4, 12)
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        base = head(context, options)
        permuted = head(context, options[permutation])
        smaller = head(context[:2], options[:2])
    assert base.shape == (4,)
    assert smaller.shape == (2,)
    assert torch.allclose(permuted, base[permutation], atol=1e-6)
    with pytest.raises(ValueError, match="at least two"):
        head(context, options[:1])

    contexts = torch.zeros((2, 5, 12))
    contexts[0] = context
    contexts[1, :2] = context[:2]
    options_batch = torch.zeros((2, 4, 12))
    options_batch[0] = options
    options_batch[1, :2] = options[:2]
    context_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.bool)
    option_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    with torch.no_grad():
        batched = head(contexts, options_batch, context_mask, option_mask)
    assert batched.shape == (2, 4)
    assert torch.allclose(batched[0], base, atol=1e-6)
    assert torch.allclose(batched[1, :2], smaller, atol=1e-6)
    assert torch.isfinite(batched).all()


def test_independent_feature_cache_round_trips_variable_context_and_option_counts(tmp_path) -> None:
    first_row = {"id": "one", "group_id": "g1", "kind": "choice", "options": ["a", "b"], "target": [1.0, 0.0]}
    second_row = {
        "id": "two",
        "group_id": "g2",
        "kind": "score",
        "options": ["low", "mid", "high"],
        "target": [0.0, 1.0, 0.0],
    }
    encoded = {
        "train": [
            {
                "row": first_row,
                "context": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
                "options": np.asarray([[5.0, 6.0], [7.0, 8.0]]),
                "target": np.asarray([1.0, 0.0]),
                "answer_keys": ("0", "1"),
            },
            {
                "row": second_row,
                "context": np.asarray([[9.0, 10.0]]),
                "options": np.asarray([[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]),
                "target": np.asarray([0.0, 1.0, 0.0]),
                "answer_keys": ("0", "1", "2"),
            },
        ]
    }
    cache = tmp_path / "independent-features.npz"
    digest = _write_feature_cache(
        cache,
        "signature",
        encoded,
        {"train": []},
        {"train": 3},
        {"train": True},
        2,
    )
    loaded, skipped, truncated, complete, hidden_size = _read_feature_cache(
        cache, "signature", {"train": [first_row, second_row]}
    )

    assert len(digest) == 64
    assert loaded["train"][0]["context"].shape == (2, 2)
    assert loaded["train"][1]["options"].shape == (3, 2)
    assert loaded["train"][1]["target"].tolist() == [0.0, 1.0, 0.0]
    assert skipped == {"train": []}
    assert truncated == {"train": 3}
    assert complete == {"train": True}
    assert hidden_size == 2


def test_tiny_scorer_accepts_varying_option_counts_and_masks_padding() -> None:
    torch = pytest.importorskip("torch")
    from pilot.intelligence.tiny_decision_scorer import build_tiny_decision_scorer

    torch.manual_seed(5)
    model = build_tiny_decision_scorer(width=8, rank=4, context_tokens=8).eval()
    context_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    context_mask = context_ids.ne(0)
    option_ids = torch.tensor(
        [
            [[6, 7], [8, 0], [9, 10]],
            [[11, 0], [12, 0], [0, 0]],
        ]
    )
    option_token_mask = option_ids.ne(0)
    option_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    with torch.no_grad():
        logits = model(context_ids, context_mask, option_ids, option_token_mask, option_mask)
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    assert logits[1, 2] < -1e30
