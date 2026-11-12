"""The further eval checks, and the eval replayed through each provider's wire format (no network, no cost)."""

from __future__ import annotations

import asyncio

import pytest

from benchmarks.conversation_eval.cases import CONVERSATIONS
from benchmarks.conversation_eval.checks import TurnRecord, summarise
from benchmarks.conversation_eval.more import MORE_CONVERSATIONS, more_checks, replay_model, run_conversations


def talk(text):
    return TurnRecord(text="q", route="conversation", done=True, spoken=[text], raw=text)


def failed(text):
    return sorted(o.check for o in more_checks(talk(text), {}) if not o.passed)


def test_two_questions_a_long_reply_and_unsayable_numbers_fail():
    assert failed("Do you want to plan the week? Or just today?") == ["one_question"]
    assert failed(" ".join(["word"] * 40) + ".") == ["short_enough"]
    for text in ("It was modified on 12/11/2026.", "The file is 16GB.", "Build 2026111201 failed.", "Use v1.2.3."):
        assert failed(text) == ["numbers_said_well"], text
    assert failed("It's sixteen gigabytes, from the twelfth of November.") == []


@pytest.mark.parametrize("provider", ["openai", "gemini", "claude"])
def test_the_recorded_run_replays_identically_over_each_providers_stream_format(provider):
    model, budget = replay_model(provider)
    results = asyncio.run(run_conversations(CONVERSATIONS, model, extra_checks=False))
    summary = summarise(results)
    assert summary["handover"]["precision"] == 1.0 and summary["handover"]["recall"] == 1.0
    # The recorded Haiku run's only failure: two honest "I can't..." answers in a row.
    assert [f["check"] for f in summary["failures"]] == ["fresh_opening"]
    assert len(budget.records) == len(model.calls) and all(r[-1] == "reported" for r in budget.records)


def test_the_further_cases_are_well_formed():
    assert sum(len(c["turns"]) for c in MORE_CONVERSATIONS) >= 10
    assert any(c.get("heliox_first") for c in MORE_CONVERSATIONS)
