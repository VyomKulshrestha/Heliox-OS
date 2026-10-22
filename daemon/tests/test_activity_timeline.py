"""Activity timeline: spans of app and window title, exclusions, pausing, and answers composed in code."""

from __future__ import annotations

import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig, TimelineConfig
from pilot.workflows.activity_timeline import ActivityTimeline, parse_when

NOON = datetime(2026, 10, 23, 12, 0, 0)


def _at(hour: int, minute: int = 0, day: int = 23) -> float:
    return datetime(2026, 10, day, hour, minute).timestamp()


@pytest.fixture
def timeline(tmp_path) -> ActivityTimeline:
    return ActivityTimeline(tmp_path / "timeline" / "timeline.db", config=TimelineConfig(enabled=True))


def _hold(timeline: ActivityTimeline, app: str, title: str, start: float, minutes: int) -> None:
    for second in range(0, minutes * 60 + 1, 30):
        timeline.observe(app, title, start + second)


def test_consecutive_observations_merge_into_one_span(timeline):
    _hold(timeline, "Code", "server.py - Heliox", _at(9), 20)
    _hold(timeline, "Chrome", "Pull requests", _at(9, 20, 23) + 30, 10)

    spans = timeline.spans(_at(8), _at(11))

    assert [(s["app"], s["title"]) for s in spans] == [("Code", "server.py - Heliox"), ("Chrome", "Pull requests")]
    assert spans[0]["end"] - spans[0]["start"] == 20 * 60


def test_a_long_gap_starts_a_new_span(timeline):
    timeline.observe("Code", "a.py", _at(9))
    timeline.observe("Code", "a.py", _at(9, 10))

    assert len(timeline.spans(_at(8), _at(10))) == 2


def test_password_managers_and_private_windows_are_never_recorded(timeline):
    timeline.observe("1Password", "Vault", _at(9))
    timeline.observe("msedge", "Bank - InPrivate", _at(9, 1))
    timeline.observe("chrome", "Taxes - Incognito", _at(9, 2))

    assert timeline.spans(_at(8), _at(10)) == []


def test_nothing_is_recorded_while_off_or_paused(tmp_path):
    config = TimelineConfig(enabled=False)
    timeline = ActivityTimeline(tmp_path / "t.db", config=lambda: config)
    timeline.observe("Code", "a.py", _at(9))
    assert not (tmp_path / "t.db").exists()

    config.enabled = True
    timeline.pause(60)
    timeline.observe("Code", "a.py")
    assert timeline.spans(0, 2e10) == []
    timeline.pause(0)
    timeline.observe("Code", "a.py")
    assert len(timeline.spans(0, 2e10)) == 1


def test_answers_what_was_i_working_on(timeline):
    _hold(timeline, "Code", "server.py - Heliox", _at(9), 40)
    _hold(timeline, "Chrome", "Pull requests", _at(10), 15)

    answer = timeline.answer("What was I working on this morning?", now=NOON)

    assert answer.startswith("This morning you used Code for 40 min")
    assert '"server.py - Heliox"' in answer and "Chrome for 15 min" in answer


def test_answers_when_did_i_last_open(timeline):
    _hold(timeline, "Chrome", "Quarterly report - Google Docs", _at(16, 0, 22), 5)

    answer = timeline.answer("when did I last open the quarterly report", now=NOON)

    assert answer == 'You last had "Quarterly report - Google Docs" open in Chrome yesterday at 4:05 PM.'


def test_other_requests_are_not_timeline_questions(timeline):
    assert timeline.answer("open the quarterly report") is None
    assert timeline.answer("what is my cpu usage") is None


def test_off_timeline_says_how_to_turn_it_on(tmp_path):
    timeline = ActivityTimeline(tmp_path / "t.db", config=TimelineConfig(enabled=False))

    assert "turn it on in Settings" in timeline.answer("what was I doing yesterday")


@pytest.mark.parametrize(
    ("phrase", "start", "end", "label"),
    [
        ("yesterday afternoon", (12, 22), (17, 22), "yesterday afternoon"),
        ("this morning", (5, 23), (12, 23), "this morning"),
        ("at 3 pm yesterday", (14, 22, 30), (15, 22, 30), "yesterday around 3:00 PM"),
        ("an hour ago", (10, 23, 45), (11, 23, 15), "an hour ago"),
    ],
)
def test_parse_when(phrase, start, end, label):
    def ts(parts):
        hour, day, minute = (parts + (0,))[:3]
        return datetime(2026, 10, day, hour, minute).timestamp()

    got_start, got_end, got_label = parse_when(phrase, NOON)

    assert (got_start, got_end, got_label) == (ts(start), ts(end), label)


def test_prune_and_clear(timeline):
    _hold(timeline, "Code", "old.py", _at(9, 0, 1), 5)
    _hold(timeline, "Code", "new.py", _at(9), 5)

    assert timeline.prune(now=_at(12)) == 1
    assert [s["title"] for s in timeline.spans(0, 2e10)] == ["new.py"]
    timeline.clear()
    assert timeline.spans(0, 2e10) == []


def test_search_treats_wildcards_literally(timeline):
    _hold(timeline, "Code", "100% done.md", _at(9), 1)
    _hold(timeline, "Code", "notes.md", _at(9, 5), 1)

    assert [s["title"] for s in timeline.search("100%")] == ["100% done.md"]


def test_timeline_store_is_off_limits_to_actions(monkeypatch, tmp_path):
    import pilot.config
    from pilot.security.validator import inside_heliox_private_dirs

    monkeypatch.setattr(pilot.config, "DATA_DIR", tmp_path / "data")

    assert inside_heliox_private_dirs(str(tmp_path / "data" / "timeline" / "timeline.db"))


@pytest.mark.asyncio
async def test_voice_question_is_answered_without_planning(tmp_path):
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server.config.timeline.enabled = True
    server._activity_timeline = ActivityTimeline(tmp_path / "t.db", config=lambda: server.config.timeline)
    now = time.time()
    server._activity_timeline.observe("Code", "server.py", now - 60)
    server._activity_timeline.observe("Code", "server.py", now - 30)
    server._broadcast_notification = AsyncMock()
    server._speak_voice_response = AsyncMock()
    server._queue_interactive_request = AsyncMock()

    await server._voice_command_dispatch("what was I working on in the last hour")

    method, payload = server._broadcast_notification.await_args.args
    assert method == "voice_result" and payload["message"].startswith("In the last hour you used Code")
    server._speak_voice_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_timeline_rpcs(tmp_path):
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server.config.timeline.enabled = True
    server._activity_timeline = ActivityTimeline(tmp_path / "t.db", config=lambda: server.config.timeline)
    server._activity_timeline.observe("Code", "server.py")

    assert (await server._handle_timeline_search({"query": "server"}, MagicMock()))["spans"][0]["app"] == "Code"
    summary = await server._handle_timeline_summary({"when": "today"}, MagicMock())
    assert summary["status"] == "ok" and summary["label"] == "today"
    assert (await server._handle_timeline_pause({"minutes": 5000}, MagicMock()))["status"] == "error"
    assert (await server._handle_timeline_clear({}, MagicMock()))["status"] == "error"
    assert (await server._handle_timeline_clear({"confirmed": True}, MagicMock()))["status"] == "ok"
    bad = await server._handle_update_config(
        {"section": "timeline", "values": {"excluded_apps": ["", "x"]}}, MagicMock()
    )
    assert bad["status"] == "error"
