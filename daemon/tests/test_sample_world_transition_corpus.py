"""World-transition samples retain source integrity and split boundaries."""

import io
import json
from urllib.error import HTTPError

import pytest

from scripts import sample_world_transition_corpus as sampler


def _entry(index=4, content="<tool_response>done</tool_response>"):
    return {"row_idx": index, "row": {"dataset": "toolathlon", "messages": [
        {"role": "system", "content": "state"},
        {"role": "user", "content": "action"},
        {"role": "assistant", "content": content},
    ]}, "truncated_cells": []}


def test_extract_record_keeps_provenance_without_relabeling():
    record, reason = sampler.extract_record(_entry(), config="train", split="train",
                                             revision="a" * 40, max_input_chars=20,
                                             max_target_chars=100)
    assert reason == "accepted"
    assert record["dataset"] == "toolathlon"
    assert record["messages"][-1]["content"] == "<tool_response>done</tool_response>"
    assert len(record["context_sha256"]) == 64


def test_extract_record_rejects_truncated_and_overlong():
    entry = _entry()
    entry["truncated_cells"] = ["messages"]
    assert sampler.extract_record(entry, config="train", split="train", revision="a" * 40,
                                  max_input_chars=20, max_target_chars=100)[1] == "truncated_source"
    assert sampler.extract_record(_entry(), config="train", split="train", revision="a" * 40,
                                  max_input_chars=1, max_target_chars=100)[1] == "overlong"


def test_sample_split_fails_if_revision_changes(monkeypatch):
    calls = iter([
        ({"num_rows_total": 10, "rows": [_entry(0)]}, "a" * 40),
        ({"num_rows_total": 10, "rows": [_entry(4)]}, "b" * 40),
    ])
    monkeypatch.setattr(sampler, "fetch_page", lambda *args: next(calls))
    with pytest.raises(ValueError, match="revision"):
        sampler.sample_split(config="train", split="train", pages=1, page_length=1,
                             rng=__import__("random").Random(0), expected_revision=None,
                             max_input_chars=20, max_target_chars=100)


def test_token_filter_excludes_overlong_targets_without_truncation(monkeypatch):
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return " ".join(message["content"] for message in messages)

        def __call__(self, text, **kwargs):
            assert kwargs["add_special_tokens"] is False
            return {"input_ids": text.split()}

    import huggingface_hub
    from transformers import AutoTokenizer

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: "cached-snapshot")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: FakeTokenizer())
    records = [{"messages": _entry(content="one two")["row"]["messages"]},
               {"messages": _entry(content="one two three")["row"]["messages"]}]
    accepted, counts = sampler.token_filter(records, max_input_tokens=2,
                                             max_target_tokens=2)
    assert len(accepted) == 1
    assert accepted[0]["input_tokens"] == 2
    assert accepted[0]["target_tokens"] == 2
    assert counts == {"accepted": 1, "overlong_tokens": 1}


def test_fetch_page_stops_on_rate_limit(monkeypatch):
    def rate_limited(*args, **kwargs):
        raise HTTPError("https://example.invalid", 429, "too many", {}, None)

    monkeypatch.setattr(sampler, "urlopen", rate_limited)
    with pytest.raises(RuntimeError, match="rate limit"):
        sampler.fetch_page("train", "train", 0, 1)


def test_fetch_page_retries_transient_server_error(monkeypatch):
    response = io.BytesIO(json.dumps({"rows": []}).encode())
    response.headers = {"x-revision": "a" * 40}
    calls = iter([HTTPError("https://example.invalid", 502, "bad gateway", {}, None),
                  response])

    def fetch(*args, **kwargs):
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(sampler, "urlopen", fetch)
    monkeypatch.setattr(sampler.time, "sleep", lambda seconds: None)
    assert sampler.fetch_page("train", "train", 0, 1) == ({"rows": []}, "a" * 40)
