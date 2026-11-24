"""A card Heliox Live raised in one conversation is not brought up in the next, and within a conversation it reaches
the talk model only when the person's words bear on it (pilot.live.conversation_link)."""

from __future__ import annotations

from types import SimpleNamespace

from pilot.config import PilotConfig
from pilot.conversation import layer_for
from pilot.conversation.adapters import drop_live_hold
from pilot.conversation.context import ContextSources, ConversationMemory, build_messages
from pilot.live.conversation_link import LiveConversationLink, bears_on
from pilot.server import PilotServer

CARD = {
    "id": "c1",
    "kind": "commitment",
    "title": "Want a reminder today at four p m?",
    "message": "You said you'd call your mom at five.",
    "evidence": "I'll call my mom at five",
    "source": "speech",
    "spoken": False,
}


class _Live:
    def __init__(self, entry):
        self.entry = entry

    def last_raised(self):
        return dict(self.entry)


def _link(now: list[float], entry=None) -> LiveConversationLink:
    live = _Live({**CARD, "at": 100.0, **(entry or {})})
    layer = SimpleNamespace(memory=ConversationMemory(clock=lambda: now[0]))
    return LiveConversationLink(
        live=lambda: live, layer=lambda: layer, setting=lambda: "auto", model_local=lambda: False, clock=lambda: now[0]
    )


def test_the_card_bears_on_words_about_it_and_on_a_reply_to_it():
    assert bears_on("Actually I'll call mom at six instead", CARD)
    assert bears_on("why?", CARD)
    assert bears_on("yes please", CARD)
    assert bears_on("what reminder?", CARD)
    assert not bears_on("I'm going to wear the blue shirt tomorrow", CARD)
    assert not bears_on("what's the capital of Australia?", CARD)


def test_a_new_conversation_starts_clean():
    now = [124.0]
    link = _link(now)
    assert "call your mom" in link.context_line("I'll call mom later then")
    link.new_conversation()
    now[0] = 130.0
    assert link.context_line("I'll call mom later then") == ""  # raised before the new conversation
    assert link.context_line() == ""


def test_within_a_conversation_only_when_the_words_bear_on_it():
    now = [124.0]
    link = _link(now)
    assert link.context_line("Blue shirt tomorrow, I think.") == ""
    line = link.context_line("why did you say that")
    assert "never repeat it or offer it again" in line  # the model is not to offer it a second time


def test_the_line_goes_in_the_request_only_when_it_bears(monkeypatch):
    now = [124.0]
    link = _link(now)
    sources = ContextSources(extra=[link.context_line])
    memory = ConversationMemory(clock=lambda: now[0])
    unrelated = build_messages("I'll wear the blue shirt tomorrow.", sources, memory, screen=None)
    assert "Heliox Live" not in str(unrelated)
    related = build_messages("Can you remind me about mom at six instead?", sources, memory, screen=None)
    assert "Heliox Live" in str(related)


def test_the_servers_new_conversation_call_clears_the_link():
    server = PilotServer(PilotConfig())
    from pilot.personal.wiring import link_for

    link = link_for(server)
    link._clock = lambda: 124.0
    server._live = _Live({**CARD, "at": 100.0})
    layer_for(server)
    assert link.context_line("call mom") != ""
    drop_live_hold(server)  # what _handle_voice_new_conversation calls
    assert link.context_line("call mom") == ""
