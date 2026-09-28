from __future__ import annotations

import numpy as np
import pytest

from pilot.intelligence import packed_typed_decision as packed
from scripts.train_packed_typed_decision import (
    _read_feature_cache,
    _shuffled_context_pairs,
    _write_feature_cache,
)


class FakeTokenizer:
    unk_token_id = 0

    def __init__(self) -> None:
        self.ids = {token: index + 1 for index, token in enumerate(packed.SPECIAL_TOKENS)}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.ids.get(token, self.unk_token_id)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        result = []
        cursor = 0
        while cursor < len(text):
            marker = next((token for token in self.ids if text.startswith(token, cursor)), None)
            if marker is not None:
                result.append(self.ids[marker])
                cursor += len(marker)
            else:
                result.append(ord(text[cursor]) + 100)
                cursor += 1
        return result


def record(kind: str, options: list[str], keys: list[str], *, question_id: str = "q") -> dict[str, object]:
    return {
        "id": question_id,
        "kind": kind,
        "question": "Pick one",
        "options": options,
        "answer_keys": keys,
    }


def test_packed_attention_keeps_questions_and_candidates_isolated() -> None:
    tokenizer = FakeTokenizer()
    result = packed.pack_typed_questions(
        tokenizer,
        {"screen": "Settings"},
        [
            record("choice", ["alpha", "beta"], ["a", "b"]),
            record("choice", ["gamma", "delta"], ["c", "d"], question_id="q2"),
        ],
        max_sequence_length=256,
    )
    first, second = result.questions
    end_option_id = tokenizer.ids[packed.SPECIAL_TOKENS[3]]
    first_start = result.state_token_count
    second_start = first.branch_end_index
    first_option_starts = [
        index
        for index in range(first_start, first.branch_end_index)
        if result.input_ids[index] == tokenizer.ids[packed.SPECIAL_TOKENS[2]]
    ]
    first_option_ends = [
        index for index in range(first_start, first.branch_end_index) if result.input_ids[index] == end_option_id
    ]
    assert len(first_option_starts) == len(first_option_ends) == 2

    option_spans = list(zip(first_option_starts, first_option_ends, strict=True))
    for candidate_index, (candidate_start, candidate_end) in enumerate(option_spans):
        candidate_rows = result.attention_mask[candidate_start : candidate_end + 1]
        for other_index, (other_start, other_end) in enumerate(option_spans):
            if other_index != candidate_index:
                assert not candidate_rows[:, other_start : other_end + 1].any()
    assert not result.attention_mask[first.query_index, first_option_starts].any()
    assert not result.attention_mask[first.query_index, second_start:].any()
    assert not result.attention_mask[second.query_index, first_start : first.branch_end_index].any()
    assert result.attention_mask[second.query_index, : result.state_token_count].all()


def test_choice_candidates_are_canonical_but_ordered_scores_are_not() -> None:
    tokenizer = FakeTokenizer()
    choice_a = packed.pack_typed_questions(
        tokenizer,
        "state",
        [record("choice", ["zulu", "alpha"], ["z", "a"])],
        max_sequence_length=128,
    )
    choice_b = packed.pack_typed_questions(
        tokenizer,
        "state",
        [record("choice", ["alpha", "zulu"], ["a", "z"])],
        max_sequence_length=128,
    )
    assert choice_a.input_ids == choice_b.input_ids
    assert choice_a.questions[0].answer_keys == ("a", "z")

    score_a = packed.pack_typed_questions(
        tokenizer,
        "state",
        [record("score", ["low", "high"], ["0", "1"])],
        max_sequence_length=128,
    )
    score_b = packed.pack_typed_questions(
        tokenizer,
        "state",
        [record("score", ["high", "low"], ["1", "0"])],
        max_sequence_length=128,
    )
    assert score_a.input_ids != score_b.input_ids
    assert score_a.questions[0].answer_keys == ("0", "1")
    assert score_b.questions[0].answer_keys == ("1", "0")


def test_user_text_cannot_inject_packed_control_tokens() -> None:
    tokenizer = FakeTokenizer()
    result = packed.pack_typed_questions(
        tokenizer,
        {"note": "literal <|fim_suffix|>"},
        [record("choice", ["<|box_end|>", "ordinary"], ["x", "y"])],
        max_sequence_length=128,
    )
    assert result.input_ids.count(tokenizer.ids[packed.SPECIAL_TOKENS[4]]) == 0
    assert result.input_ids.count(tokenizer.ids[packed.SPECIAL_TOKENS[3]]) == 2


def test_sequence_and_attention_mask_memory_limits_are_checked_first(monkeypatch) -> None:
    tokenizer = FakeTokenizer()
    with pytest.raises(ValueError, match="model limit"):
        packed.pack_typed_questions(
            tokenizer,
            "state",
            [record("choice", ["long enough", "candidate"], ["a", "b"])],
            max_sequence_length=4,
        )

    monkeypatch.setattr(packed.np, "zeros", lambda *args, **kwargs: pytest.fail("mask allocated before budget check"))
    with pytest.raises(ValueError, match="memory budget"):
        packed.pack_typed_questions(
            tokenizer,
            "s" * 4_100,
            [record("choice", ["a", "b"], ["a", "b"])],
            max_sequence_length=5_000,
        )


def test_typed_record_validation_rejects_malformed_fields() -> None:
    invalid = record("other", ["a", "b"], ["a", "b"])
    with pytest.raises(ValueError, match="fields are invalid"):
        packed.pack_typed_questions(FakeTokenizer(), "state", [invalid], max_sequence_length=32)

    duplicate_keys = record("choice", ["a", "b"], ["same", "same"])
    with pytest.raises(ValueError, match="one answer key"):
        packed.pack_typed_questions(FakeTokenizer(), "state", [duplicate_keys], max_sequence_length=32)


def test_feature_cache_round_trips_with_protocol_signature(tmp_path) -> None:
    row = {
        "id": "public-row",
        "group_id": "group-1",
        "kind": "choice",
        "source": "public-source",
        "options": ["left", "right"],
        "target": [1.0, 0.0],
    }
    encoded = {
        "train": [
            {
                "row": row,
                "state": np.asarray([1.0, 2.0]),
                "question": np.asarray([3.0, 4.0]),
                "options": np.asarray([[5.0, 6.0], [7.0, 8.0]]),
                "target": np.asarray([0.0, 1.0]),
                "answer_keys": ("1", "0"),
            }
        ]
    }
    cache = tmp_path / "packed-features.npz"
    digest = _write_feature_cache(cache, "protocol-signature", encoded, {"train": []}, {"train": True}, 2)
    loaded, skipped, complete, hidden_size = _read_feature_cache(cache, "protocol-signature", {"train": [row]})

    assert len(digest) == 64
    assert loaded["train"][0]["answer_keys"] == ("1", "0")
    assert loaded["train"][0]["target"].tolist() == [0.0, 1.0]
    assert loaded["train"][0]["options"].tolist() == [[5.0, 6.0], [7.0, 8.0]]
    assert skipped == {"train": []}
    assert complete == {"train": True}
    assert hidden_size == 2
    with pytest.raises(ValueError, match="does not match"):
        _read_feature_cache(cache, "different-signature", {"train": [row]})


def test_shuffled_context_pairs_are_deterministic_and_cross_group() -> None:
    rows = [
        {
            "id": f"row-{index}",
            "group_id": f"group-{index}",
            "kind": "choice",
            "state_json": f'{{"screen": {index}}}',
        }
        for index in range(4)
    ]

    pairs = _shuffled_context_pairs(rows, seed=17)
    repeated = _shuffled_context_pairs(rows, seed=17)

    assert [(row["id"], donor["id"]) for row, donor in pairs] == [
        (row["id"], donor["id"]) for row, donor in repeated
    ]
    assert len(pairs) == 4
    assert {donor["id"] for _, donor in pairs} == {row["id"] for row in rows}
    assert all(
        row["id"] != donor["id"]
        and row["group_id"] != donor["group_id"]
        and row["state_json"] != donor["state_json"]
        and row["kind"] == donor["kind"]
        for row, donor in pairs
    )


def test_shuffled_context_pairs_skip_rows_without_a_distinct_context() -> None:
    rows = [
        {"id": "one", "group_id": "one", "kind": "choice", "state_json": '{"screen": 1}'},
        {"id": "two", "group_id": "two", "kind": "noul", "state_json": '{"screen": 2}'},
        {"id": "three", "group_id": "three", "kind": "noul", "state_json": '{"screen": 2}'},
    ]

    assert _shuffled_context_pairs(rows, seed=0) == []
