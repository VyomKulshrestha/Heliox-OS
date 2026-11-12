"""The fast model's stable system prompt carries one cache breakpoint (the same words), and the session runner warms
the fast model's connection before each dispatch, as the daemon does at a wake or a follow-up."""

from __future__ import annotations

import asyncio

from benchmarks.conversation_eval.sessions import run_session
from pilot.conversation.fast_model import claude_payload
from pilot.conversation.style import SYSTEM_PROMPT


def test_the_system_prompt_is_sent_with_one_cache_breakpoint_and_unchanged_words():
    body = claude_payload("claude-haiku-5-5", 400, SYSTEM_PROMPT, [{"role": "user", "content": "hi"}], "low", True)
    assert body["system"] == [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    assert body["messages"] == [{"role": "user", "content": "hi"}]  # the turn's own context stays after the breakpoint


def test_the_session_runner_warms_the_connection_before_each_dispatch(monkeypatch):
    warmed: list[int] = []
    from pilot.conversation import fast_model

    original = fast_model.HttpFastModel.warm

    async def counting(self):
        warmed.append(1)
        return await original(self)

    monkeypatch.setattr(fast_model.HttpFastModel, "warm", counting)
    session = {
        "name": "warm",
        "turns": [
            {"say": "Hey Heliox, what's a good name for a cat?", "model": "Whiskers."},
            {"say": "Hey Heliox, and a dog?", "gap": 3, "model": "Biscuit."},
        ],
    }
    record, failures = asyncio.run(run_session(session, "claude"))
    assert failures == [], failures
    assert len(warmed) == 2  # one per dispatch
    assert sum(record.model_calls.values()) == 2  # the warm is not counted as a model call
