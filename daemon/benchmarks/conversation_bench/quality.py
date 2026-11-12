"""Conversation-quality cases for the fast spoken layer (``daemon/pilot/conversation/``), pending until it lands.

The interface these cases are written against (the brief's design direction, made concrete so a script can call
it without audio):

    async def bench_reply(utterance: str, *, turns: list[dict], context: dict,
                          stream: Callable[[str], AsyncIterator[str]] | None) -> dict

exported from ``pilot.conversation``. ``turns`` are the last few turns ({"who": "person"|"heliox", "text"}),
``context`` what is on screen and known about the person. ``stream`` is the talk model: the bench passes a stub
that yields canned text with the chosen kind of model's first-token and token-rate delays, None for a person
with no model, or the string "configured" to use the person's configured provider (opt-in, costs money). The return value: {"route": "answer"|"action"|"control", "spoken": [sentences in order],
"first_sentence_s": seconds from the call to the first sentence ready to speak}.

Scored by script with the canned stream (no cost): the route, plain speech (the canned text deliberately carries
markdown and a list, which must not reach the voice), length, no claimed action, the first sentence's time, and
with no model an honest short answer about what is missing. ``required_facts`` only mean something with a real
model, so they are checked in the opt-in ``--real-model`` mode, which also enables the model-judged rubric
(``rubric.judge_prompt``); both cost money and are never run by default.
"""

from __future__ import annotations

import asyncio
import importlib
import time
from collections.abc import AsyncIterator
from typing import Any

from . import rubric
from .models import KINDS, ModelKind
from .scenarios import Scenario

FIRST_SENTENCE_LIMIT_S = {"cloud": 1.5, "local": 2.5, "none": 0.8}
CANNED = "**Sure.** Here's what I found:\n- Your next meeting is the launch call at 10.\n- Then the design review."


def available() -> bool:
    try:
        module = importlib.import_module("pilot.conversation")
    except ImportError:
        return False
    return callable(getattr(module, "bench_reply", None))


def stub_stream(kind: ModelKind, text: str = CANNED) -> Any:
    if not kind.available:
        return None

    async def stream(_prompt: str) -> AsyncIterator[str]:
        await asyncio.sleep(kind.talk_first_token_s)
        for word in text.split(" "):
            yield word + " "
            await asyncio.sleep(1.0 / kind.talk_tokens_per_s)

    return stream


def judge_reply(scenario: Scenario, model: str, reply: dict[str, Any], real_model: bool = False) -> list[str]:
    """Failures for one reply (pure)."""
    q = scenario.quality
    failures: list[str] = []
    spoken = " ".join(reply.get("spoken") or [])
    facts = tuple(q.get("required_facts") or ()) if real_model else ()
    result = rubric.check(spoken, required_facts=facts, max_words=int(q.get("max_words", 35)))
    failures += result.failures
    override = ((scenario.expect.get("by_model") or {}).get(model)) or {}
    expected_route = q.get("route")
    if expected_route and reply.get("route") != expected_route:
        failures.append(f"route {reply.get('route')!r}, expected {expected_route!r}")
    if model == "none" and reply.get("route") != "action":
        # A request to act goes to the planner, which reports its own failure; anything else must say plainly
        # that conversation needs a model and where to set one up.
        wanted = override.get("must_mention_any") or []
        if wanted and not any(w.casefold() in spoken.casefold() for w in wanted):
            failures.append(f"with no model, does not say what is missing ({wanted})")
    first = reply.get("first_sentence_s")
    # The first speakable sentence may come one second after the kind of model's first token, at most.
    limit = FIRST_SENTENCE_LIMIT_S[model]
    if isinstance(first, (int, float)) and first > limit:
        failures.append(f"first sentence after {first:.2f} s (limit {limit} s)")
    return failures


async def run(scenarios: list[Scenario], model: str, real_model: bool = False) -> list[tuple[Scenario, list[str]]]:
    if not available():
        return [(s, ["pending: " + (s.pending or "pilot.conversation.bench_reply not found")]) for s in scenarios]
    module = importlib.import_module("pilot.conversation")
    out = []
    for s in scenarios:
        started = time.perf_counter()
        reply = await module.bench_reply(
            s.quality["utterance"], turns=s.quality.get("turns", []), context=s.quality.get("context", {}),
            stream="configured" if real_model else stub_stream(KINDS[model]),
        )  # fmt: skip
        reply.setdefault("first_sentence_s", time.perf_counter() - started)
        out.append((s, judge_reply(s, model, reply, real_model)))
    return out
