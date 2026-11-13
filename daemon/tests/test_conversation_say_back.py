"""A destructive step that came in as the answer to Heliox's own question is said back before its dialog opens
(pilot.conversation.say_back). The real path, for every kind of person, is the personal session "a destructive answer
to Heliox's question is said back; a spoken yes opens the dialog, never approves"."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from pilot.actions import Action
from pilot.config import PilotConfig
from pilot.conversation import say_back
from pilot.server import ANSWERED_HELIOX_QUESTION, PilotServer

DELETE = Action.model_validate(
    {"action_type": "file_delete", "target": "C:/Users/me/reports", "parameters": {"path": "C:/Users/me/reports"}}
)
OPEN = Action.model_validate(
    {"action_type": "open_application", "target": "notepad", "parameters": {"name": "notepad"}}
)


def test_what_is_destructive_comes_from_the_planners_classification():
    assert say_back.destructive([DELETE]) and not say_back.destructive([OPEN])


@pytest.mark.parametrize(
    ("heard", "said"),
    [
        ("Delete them all.", "Delete reports, yes?"),  # only pointed at: the target is named
        ("delete the reports folder", "Delete the reports folder, yes?"),  # the person's own words
    ],
)
def test_said_back_in_the_persons_words_with_the_target_named(heard, said):
    assert say_back.sentence(heard, [DELETE]) == said


def _server(monkeypatch):
    server = PilotServer(PilotConfig())
    server.config.voice.follow_up_window_seconds = 0.05
    said: list[str] = []

    async def speak(text, **_kwargs):
        said.append(text)
        return False

    monkeypatch.setattr(server, "_speak_voice_response", speak)
    return server, said


async def _answer_after(server, words, delay=0.01):
    await asyncio.sleep(delay)
    assert server._answer_say_back(words)


@pytest.mark.asyncio
async def test_a_spoken_yes_opens_the_dialog_and_is_never_an_approval(monkeypatch):
    """True here means "open the dialog": the dialog itself (_wait_for_confirmation's pending confirmation) still
    needs the click, and nothing in this path resolves it."""
    server, said = _server(monkeypatch)
    token = ANSWERED_HELIOX_QUESTION.set("Delete them all.")
    try:
        yes = asyncio.ensure_future(_answer_after(server, "Yes."))
        assert await server._said_back_and_confirmed(SimpleNamespace(actions=[DELETE])) is True
        await yes
    finally:
        ANSWERED_HELIOX_QUESTION.reset(token)
    assert said == ["Delete reports, yes?"]
    assert server._pending_confirms == {}  # nothing approved, nothing pending: the dialog opens after this


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["No.", "And now, the weather.", None])  # None: silence past the window
async def test_anything_but_yes_drops_it_with_one_short_sentence(monkeypatch, answer):
    server, said = _server(monkeypatch)
    token = ANSWERED_HELIOX_QUESTION.set("Delete them all.")
    try:
        other = asyncio.ensure_future(_answer_after(server, answer)) if answer else None
        assert await server._said_back_and_confirmed(SimpleNamespace(actions=[DELETE])) is False
        if other:
            await other
    finally:
        ANSWERED_HELIOX_QUESTION.reset(token)
    assert said == ["Delete reports, yes?", say_back.DROPPED]


@pytest.mark.asyncio
async def test_the_same_answer_again_unconfirmed_is_dropped_without_a_word(monkeypatch):
    server, said = _server(monkeypatch)
    token = ANSWERED_HELIOX_QUESTION.set("Delete them all.")
    try:
        assert await server._said_back_and_confirmed(SimpleNamespace(actions=[DELETE])) is False  # silence
        said.clear()
        assert await server._said_back_and_confirmed(SimpleNamespace(actions=[DELETE])) is False
    finally:
        ANSWERED_HELIOX_QUESTION.reset(token)
    assert said == []


@pytest.mark.asyncio
async def test_a_request_started_with_the_wake_phrase_or_a_harmless_step_is_not_said_back(monkeypatch):
    server, said = _server(monkeypatch)
    assert await server._said_back_and_confirmed(SimpleNamespace(actions=[DELETE])) is True  # not an answer
    token = ANSWERED_HELIOX_QUESTION.set("Open Notepad.")
    try:
        assert await server._said_back_and_confirmed(SimpleNamespace(actions=[OPEN])) is True
    finally:
        ANSWERED_HELIOX_QUESTION.reset(token)
    assert said == []
