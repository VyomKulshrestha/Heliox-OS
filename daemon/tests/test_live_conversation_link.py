"""A spoken Live remark recorded in the conversation is marked as Live's own (``source="live"``)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from pilot.live.conversation_link import LiveConversationLink


def _entry():
    return {
        "id": "c1",
        "kind": "weekday_date",
        "title": "14 November is a Saturday",
        "message": "You said Tuesday.",
        "evidence": "Tuesday 14 November",
        "source": "speech",
        "app": "",
        "spoken": True,
        "at": 0.0,
    }


def _link(memory):
    live = MagicMock()
    live.last_spoken_entries.return_value = [_entry()]
    return LiveConversationLink(
        live=lambda: live,
        layer=lambda: SimpleNamespace(memory=memory),
        setting=lambda: "auto",
        model_local=lambda: False,
        clock=lambda: 0.0,
    )


def test_marked_as_lives_own_when_the_memory_takes_a_source():
    calls = []

    class Memory:
        def add_heliox(self, text, *, cut_off=False, source=""):
            calls.append((text, cut_off, source))

    assert _link(Memory()).record_spoken("14 November is a Saturday.", "interrupted")
    assert calls == [("14 November is a Saturday.", True, "live")]


def test_still_recorded_by_a_memory_without_a_source():
    calls = []

    class Memory:
        def add_heliox(self, text, *, cut_off=False):
            calls.append((text, cut_off))

    assert _link(Memory()).record_spoken("14 November is a Saturday.", "spoken")
    assert calls == [("14 November is a Saturday.", False)]
