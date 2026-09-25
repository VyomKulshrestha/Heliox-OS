"""The transition probe trains only on source training outcomes."""

import json

import pytest

from scripts.train_world_transition_lora import encode_rows, load_rows


def _row(split, context):
    return {"split": split, "context_sha256": context, "dataset": "fixture",
            "messages": [{"role": "system", "content": "state"},
                         {"role": "user", "content": "action"},
                         {"role": "assistant", "content": "result"}]}


def test_load_rows_rejects_cross_split_context_leakage(tmp_path):
    source = tmp_path / "rows.jsonl"
    source.write_text("\n".join(json.dumps(_row(split, "same"))
                                for split in ("train", "eval")), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate context"):
        load_rows(source)


def test_load_rows_rejects_cross_split_task_leakage(tmp_path):
    source = tmp_path / "rows.jsonl"
    train = {**_row("train", "a"), "group_sha256": "f" * 64}
    evaluation = {**_row("eval", "b"), "group_sha256": "f" * 64}
    source.write_text("\n".join(json.dumps(row) for row in (train, evaluation)), encoding="utf-8")

    with pytest.raises(ValueError, match="share a task group"):
        load_rows(source)


def test_encode_rows_masks_state_and_action_tokens():
    class FakeTokenizer:
        eos_token_id = 9

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return " ".join(message["content"] for message in messages)

        def __call__(self, text, **kwargs):
            return {"input_ids": [{"state": 1, "action": 2, "result": 3}[text]]
                    if text in ("state", "action", "result") else [1, 2]}

    rows, excluded = encode_rows(FakeTokenizer(), {
        "train": [_row("train", "a")], "eval": [_row("eval", "b")],
    }, max_input_tokens=2, max_target_tokens=1)
    assert excluded == {"train": 0, "eval": 0}
    assert rows["train"][0]["ids"] == [1, 2, 3, 9]
    assert rows["train"][0]["labels"] == [-100, -100, 3, 9]


def test_encode_rows_keeps_group_id_for_eligible_rows():
    class FakeTokenizer:
        eos_token_id = 9

        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

        def __call__(self, text, **kwargs):
            return {"input_ids": [1]}

    training = {**_row("train", "a"), "group_sha256": "a" * 64}
    evaluation = {**_row("eval", "b"), "group_sha256": "b" * 64}
    rows, _ = encode_rows(FakeTokenizer(), {"train": [training], "eval": [evaluation]},
                          max_input_tokens=2, max_target_tokens=2)

    assert rows["train"][0]["group_sha256"] == "a" * 64
