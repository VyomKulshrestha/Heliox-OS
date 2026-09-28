from __future__ import annotations

import json

import pytest

from scripts.evaluate_base_option_likelihood import (
    encode_candidates,
    longest_common_prefix,
    mean_token_logprob,
    render_prompt,
)


def test_longest_common_prefix_preserves_every_candidate_suffix():
    prefix, suffixes = longest_common_prefix([[1, 2, 3, 7], [1, 2, 3, 8], [1, 2, 3, 9]])

    assert prefix == [1, 2, 3]
    assert suffixes == [[7], [8], [9]]


def test_longest_common_prefix_rejects_empty_or_prefix_only_candidate():
    with pytest.raises(ValueError, match="nonempty"):
        longest_common_prefix([])
    with pytest.raises(ValueError, match="at least one token"):
        longest_common_prefix([[1], [1, 2]])


def test_candidate_normalization_is_explicit():
    assert mean_token_logprob(-6.0, 3, normalization="mean") == -2.0
    assert mean_token_logprob(-6.0, 3, normalization="sum") == -6.0
    with pytest.raises(ValueError):
        mean_token_logprob(-6.0, 0, normalization="mean")


def test_prompt_uses_runtime_question_and_options_without_target_or_provenance():
    row = {
        "state_json": json.dumps({"visible": "save dialog"}),
        "question": "Which control saves the document?",
        "options": ["Save", "Cancel"],
        "target": [1.0, 0.0],
        "id": "private-row-id",
        "metadata_json": '{"secret":"not an input"}',
    }

    prompt = render_prompt(row)

    assert "Which control saves the document?" in prompt
    assert "1. Save" in prompt and "2. Cancel" in prompt
    assert "private-row-id" not in prompt
    assert "not an input" not in prompt
    assert "1.0" not in prompt


def test_encoded_options_share_one_tokenized_prompt_prefix():
    class Tokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return f"<user>{messages[0]['content']}<assistant>"

        def __call__(self, text, **_kwargs):
            return {"input_ids": [ord(character) for character in text]}

    row = {
        "state_json": json.dumps({"visible": "save dialog"}),
        "question": "Which control saves the document?",
        "options": ["Save", "Cancel"],
    }

    prompt_length, sequences = encode_candidates(Tokenizer(), row)
    prefix, suffixes = longest_common_prefix(sequences)

    assert prefix
    assert prompt_length <= len(prefix)
    assert all(sequence[:prompt_length] == sequences[0][:prompt_length] for sequence in sequences)
    assert len(suffixes) == 2
    assert all(suffix for suffix in suffixes)
