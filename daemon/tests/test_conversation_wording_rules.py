"""Wording a person would find wrong, seen in the spoken runs, and the lines said while an approval waits: the app in
front of them is context, not news; no made-up specifics; a concern said to them, as "you", with no judgment; no
offer tucked into a question; nothing after giving way; and while an approval waits, the pointer and then silence
until something changes, with one closing line when they say no."""

from __future__ import annotations

import asyncio
import json

import httpx

from pilot.config import PilotConfig
from pilot.conversation import debate as dbt
from pilot.conversation import layer_for
from pilot.conversation.context import ContextSources, ConversationMemory, Screen, build_messages
from pilot.conversation.fast_model import HttpFastModel
from pilot.conversation.handover import APPROVAL_SENTENCE, keep_company
from pilot.conversation.layer import ConversationLayer
from pilot.conversation.style import SYSTEM_PROMPT
from pilot.server import PendingConfirmation, PilotServer
from pilot.system.companion_speech import SpeechChannel, SpeechOutcome
from pilot.system.interaction import InteractionPhase
from pilot.system.voice import voice_turn_addressed


def _claude(sent: list[dict]) -> HttpFastModel:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content.decode("utf-8")))
        event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Okay."}}
        return httpx.Response(
            200, content=f"data: {json.dumps(event)}\n\n".encode(), headers={"content-type": "text/event-stream"}
        )

    async def key() -> str:
        return "k"

    return HttpFastModel(
        provider="claude",
        name="m",
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        api_key=key,
    )


async def _silent(text, channel, key):
    return "spoken"


async def test_the_request_body_carries_the_rules_and_no_example_date():
    sent: list[dict] = []
    model = _claude(sent)
    layer = ConversationLayer(sink=_silent, model=lambda: model, disagree=lambda: "when_it_matters")
    layer.sources = ContextSources(screen=lambda: Screen(app="Quillpad"))
    await layer.take("I'm going to delete my only copy of the thesis to free up space.")
    body = sent[0]
    system = json.dumps(body["system"])
    assert "twelfth of November" not in system and "twelfth of November" not in SYSTEM_PROMPT
    assert "About the person's own things (their files, backups, dates, plans, people), name only what" in system
    assert 'never \\"their\\"' in system and "you may not have weighed" in system  # said to them, no judgment
    assert "may not have seen it" not in system
    context = json.dumps(body["messages"])
    assert "The app in front of them: Quillpad (mention it only if they ask" in context
    assert "The person is in" not in context


def test_the_concern_prompt_speaks_to_them_and_a_third_person_or_judging_reply_is_not_said():
    assert 'as "you"' in dbt.CONCERN_PROMPT and "may not have seen" not in dbt.CONCERN_PROMPT
    assert dbt.concern_from_reply("That would be permanent, and it's their only copy, so I'd push back.") == ""
    assert dbt.concern_from_reply("It's a risk you may not have fully weighed.") == ""
    assert dbt.concern_from_reply("That's your only copy of the thesis.") == "That's your only copy of the thesis."


class Replies:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.local, self.streams, self.name, self.provider = False, True, "fake", "fake"

    async def stream(self, system, messages):
        yield self.replies.pop(0)


async def test_an_offer_tucked_into_a_question_is_not_said_and_nothing_follows_giving_way():
    said: list[str] = []

    async def sink(text, channel, key):
        said.append(text)
        return "spoken"

    model = Replies(
        "Wait, that's your only copy, so deleting it can't be undone.\nSTANCE: only copy of the thesis",
        "Okay, your call. But a cloud copy or an external drive would free the space without losing it.",
    )
    layer = ConversationLayer(sink=sink, model=lambda: model, disagree=lambda: "when_it_matters")
    await layer.take("I'm going to delete my only copy of the thesis to free up space.")
    assert layer.debate.stance is not None
    await layer.take("Yes.")
    assert said[-1] == "Okay, your call."  # the "but" after giving way is not said

    model = Replies("Do you have a backup, or should I look for space elsewhere first?")
    said.clear()
    await ConversationLayer(sink=sink, model=lambda: model).take("I'm going to delete my only copy of the thesis.")
    assert said == ["Do you have a backup?"]


# ── while an approval waits ──


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


async def test_after_the_approval_pointer_nothing_until_the_plan_runs():
    clock = Clock()
    said: list[tuple[float, str]] = []

    def status():
        clock.t += 0.5
        t = clock.t
        if t < 3:
            return {"phase": "planning", "message": "Planning the safest useful action"}
        if 20 <= t < 22:
            return {"phase": "speaking", "message": "Speaking"}  # Heliox's own refusal of a spoken "yes"
        if t < 40:
            return {"phase": "awaiting_approval" if t < 20 or t >= 22 else "speaking", "message": "Waiting"}
        return {"phase": "acting", "message": "Running step 1 of 2"}

    async def speak(text, channel, key):
        said.append((clock.t, text))
        return "spoken"

    task = asyncio.ensure_future(keep_company(acknowledgement_text="", speak=speak, status=status, clock=clock, poll=0))
    while clock.t < 45:
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    lines = [text for _t, text in said]
    waiting = [text for t, text in said if 3 <= t < 40]
    assert waiting == [APPROVAL_SENTENCE]  # one pointer, then silence while it waits
    assert lines[-1] not in (APPROVAL_SENTENCE,)  # once the approved plan runs, the company speaks again


async def test_saying_no_to_a_waiting_approval_gets_one_closing_line():
    server = PilotServer(PilotConfig())
    heard: list[str] = []

    async def speak(text, channel, dedupe_key="", recorder=None):
        heard.append(text)
        return SpeechOutcome("spoken", SpeechChannel(str(channel)), "")

    async def nowhere(*_args, **_kwargs):
        return None

    async def reserved(params, ws):
        """The product's own _handle_execute around it; this part stands for planning to an approval."""
        pending = PendingConfirmation(plan_id="p1", event=asyncio.Event())
        server._pending_confirms["p1"] = pending
        server._active_plan_id = "p1"
        await server._interaction_runtime.transition(InteractionPhase.AWAITING_APPROVAL, message="Waiting")
        await pending.event.wait()
        server._pending_confirms.pop("p1", None)
        return {"status": "cancelled", "message": "Cancelled before execution."}

    server._speech_coordinator.speak = speak
    server._broadcast_notification = nowhere
    server._handle_execute_reserved = reserved
    server._model_router = object()
    layer_for(server)._model = lambda: None
    voice_turn_addressed.set("wake")
    request = asyncio.ensure_future(server._voice_command_dispatch("delete my only copy of the thesis for me"))
    while "p1" not in server._pending_confirms:
        await asyncio.sleep(0)
    voice_turn_addressed.set("follow_up")
    await server._voice_command_dispatch("No")
    await request
    closing = [line for line in heard if "won't" in line or "Cancelled" in line or "Nothing ran" in line]
    assert len(closing) == 1, heard
