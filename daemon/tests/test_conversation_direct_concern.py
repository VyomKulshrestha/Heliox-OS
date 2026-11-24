"""A direct request for a costly step ("delete my only copy of the thesis for me") gets its reason said once, before
the approval sentence, from one talk-model call made beside the planner. The approval is the question; harmless
requests never reach a model (pilot.conversation.debate, pilot.conversation.handover)."""

from __future__ import annotations

import asyncio

import pytest

from pilot.conversation import debate as dbt
from pilot.conversation.handover import APPROVAL_SENTENCE, keep_company
from pilot.conversation.layer import ConversationLayer

THESIS = "delete my only copy of the thesis for me"
REASON = "That's your only copy of the thesis, so it would be gone."


class Model:
    def __init__(self, reply: str = REASON, *, local: bool = False) -> None:
        self.reply = reply
        self.local = local
        self.streams = True
        self.name = self.provider = "fake"
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    async def stream(self, system, messages):
        self.calls.append((system, messages))
        yield self.reply


async def _silent(text, channel, key):
    return "spoken"


def _layer(model, **kwargs) -> ConversationLayer:
    kwargs.setdefault("disagree", lambda: "when_it_matters")
    return ConversationLayer(sink=_silent, model=lambda: model, **kwargs)


async def test_the_costly_request_is_handed_over_at_once_and_its_reason_is_asked_beside_it():
    model = Model()
    layer = _layer(model)
    result = await layer.take(THESIS)
    assert not result.done and result.intent.verb == "delete"  # the planner is not kept waiting
    assert result.planner_input == THESIS
    assert await result._concern == REASON
    system, messages = model.calls[0]
    assert system == dbt.CONCERN_PROMPT  # the request sent: its own short prompt, no ACT or STANCE rules
    assert THESIS in messages[-1]["content"]
    assert len(model.calls) == 1


@pytest.mark.parametrize(
    "words", ["open notepad", "set a timer for ten minutes", "play some music", "find my tax folder"]
)
async def test_harmless_requests_never_reach_a_model(words):
    model = Model()
    result = await _layer(model).take(words)
    assert not result.done and result._concern is None
    assert model.calls == []


async def test_no_call_with_the_setting_off_typed_or_with_a_model_on_this_computer():
    for layer, source in (
        (_layer(Model(), disagree=lambda: "off"), "voice"),
        (_layer(Model()), "typed"),
        (_layer(Model(local=True)), "voice"),
    ):
        result = await layer.take(THESIS, source=source)
        assert result._concern is None


@pytest.mark.parametrize(
    ("reply", "said"),
    [
        ("NONE", ""),
        ("Are you sure you want to delete it?", ""),
        ("Want me to back it up first?", ""),
        ("I'll delete it once you approve.", ""),
        (REASON + " Maybe copy it somewhere first.", REASON),
        ("That's your only copy STANCE: thesis", ""),
    ],
)
def test_only_one_short_reason_is_ever_said(reply, said):
    assert dbt.concern_from_reply(reply) == said


async def test_the_reason_is_said_once_in_a_conversation():
    model = Model()
    layer = _layer(model)
    first = await layer.take(THESIS)
    assert await first._concern == REASON
    second = await layer.take(THESIS)
    assert second._concern is None  # the topic was voiced: not asked or said again


# ── what the person hears, in order ──


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


async def _company(concern, approval_at: float, *, until: float = 12.0) -> list[str]:
    clock = Clock()
    said: list[str] = []

    async def speak(text, channel, key):
        said.append(text)
        return "spoken"

    def status():
        clock.t += 0.5
        return {"phase": "awaiting_approval" if clock.t >= approval_at else "planning", "message": "Planning"}

    task = asyncio.ensure_future(
        keep_company(acknowledgement_text="", speak=speak, status=status, clock=clock, poll=0, concern=concern)
    )
    while clock.t < until:
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    return said


async def test_the_reason_comes_first_then_the_approval_with_no_other_question():
    ready: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    ready.set_result(REASON)
    said = await _company(ready, approval_at=1.5)
    assert said[:2] == [REASON, APPROVAL_SENTENCE]
    assert not any(line.endswith("?") for line in said)


async def test_the_approval_waits_a_moment_for_a_reason_still_coming_and_never_for_long():
    pending: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    said = await _company(pending, approval_at=1.0)
    assert APPROVAL_SENTENCE in said and REASON not in said
    assert pending.cancelled()  # the company ended: a reason not said by then is never said


async def test_no_reason_no_extra_words():
    none: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    none.set_result("")
    said = await _company(none, approval_at=1.0)
    assert said[0] == APPROVAL_SENTENCE
