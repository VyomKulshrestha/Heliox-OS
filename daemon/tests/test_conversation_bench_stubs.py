"""A bench run tells the talk model a made-up foreground app, never the real one, unless it asks; and it has the
stores the server's code handlers answer from, in its own scratch folder (benchmarks.conversation_bench.stubs)."""

from __future__ import annotations

from types import SimpleNamespace

from benchmarks.conversation_bench import stubs
from pilot.config import PilotConfig
from pilot.conversation import adapters
from pilot.conversation.context import ContextSources, ConversationMemory, build_messages
from pilot.server import PilotServer


async def test_the_model_is_told_a_made_up_app(monkeypatch):
    monkeypatch.setattr(adapters, "_active_window", lambda: ("SomeRealApp", "a private window title"))
    assert stubs.stub_foreground() == stubs.BENCH_APP
    server = SimpleNamespace(_live=None, config=SimpleNamespace(live=SimpleNamespace(enabled=False)))
    focus = await adapters.FocusSource(server).focus()
    assert focus.app == stubs.BENCH_APP and focus.title == ""
    screen = adapters.screen_for_model(
        focus, setting="auto", model_local=False, excluded_apps=[], label=lambda name: name
    )
    sent = str(build_messages("hello", ContextSources(), ConversationMemory(), screen=screen))
    assert f"The app in front of them: {stubs.BENCH_APP} (" in sent and "SomeRealApp" not in sent


def test_a_run_can_ask_for_the_real_one(monkeypatch):
    real = lambda: ("SomeRealApp", "")  # noqa: E731
    monkeypatch.setattr(adapters, "_active_window", real)
    assert stubs.stub_foreground(real=True) == "real"
    assert adapters._active_window is real


def test_the_stores_are_made_in_the_scratch_folder(tmp_path):
    server = PilotServer(PilotConfig())
    assert stubs.give_stores(server, tmp_path) == ["todos", "reminders", "profile"]
    server._todos.add("finish the report")
    assert (tmp_path / "todos.json").exists()
    assert stubs.give_stores(server, tmp_path) == []  # a store already there is kept
