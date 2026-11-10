"""Heliox Live only checks what the person writes: page text, text that was already there, and on-screen OCR
matches of key-like text do not raise cards."""

from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace

import pytest

from pilot.live.engine import TICK_SECONDS, LiveCompanion, Observation
from pilot.live.focused_text import parse_reply

MONDAY = date(2026, 11, 9)
CLAUDE = {"app": "claude.exe", "title": "Claude", "source": "focused_field"}


# ── the helper's answer ──


def test_only_a_writable_field_or_terminal_passes_its_text_on():
    page = parse_reply({"ok": True, "type": "ControlType.Document", "pid": 7, "editable": False, "kind": ""})
    assert page is not None and not page.editable and page.text == "" and page.control_type == "Document"
    # Even if a reply carried text for a read-only element, it is dropped.
    leaked = parse_reply({"ok": True, "type": "ControlType.Text", "editable": False, "text": "Work Work"})
    assert leaked is not None and leaked.text == ""
    field = parse_reply(
        {"ok": True, "type": "ControlType.Edit", "editable": True, "kind": "field", "field": "42.7", "text": "hi"}
    )
    assert field is not None and field.editable and field.text == "hi" and field.field == "42.7"
    term = parse_reply({"ok": True, "type": "ControlType.Text", "editable": True, "kind": "terminal", "text": "$ ls"})
    assert term is not None and term.editable and term.kind == "terminal"
    password = parse_reply(
        {"ok": True, "type": "ControlType.Edit", "password": True, "editable": True, "kind": "field", "text": "x"}
    )
    assert password is not None and password.password and not password.editable and password.text == ""
    unknown = parse_reply({"ok": True, "type": "ControlType.Edit", "editable": True, "kind": "maybe", "text": "x"})
    assert unknown is not None and not unknown.editable and unknown.text == ""
    assert parse_reply({"ok": False}) is None


def test_the_helper_decides_from_value_pattern_read_only_and_never_reads_other_elements():
    from pilot.live.focused_text import _HELPER

    # The decision comes before any text is read, and only Edit or Document controls can be fields.
    assert _HELPER.index("$vp.Current.IsReadOnly") < _HELPER.index("GetVisibleRanges")
    assert "'ControlType.Edit'" in _HELPER and "'ControlType.Document'" in _HELPER
    assert "IsPassword" in _HELPER and "TermControl" in _HELPER


# ── the engine ──


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def build(tmp_path, obs: list[Observation | None], window_exists=lambda handle: True, **settings_kw):
    clock = Clock()
    events: list[tuple[str, dict]] = []
    settings = SimpleNamespace(enabled=True, level="balanced", listen_to_speech=True, muted_kinds=[], **settings_kw)

    async def observe():
        return obs[0]

    async def emit(event, payload):
        events.append((event, payload))

    async def speak(message, urgent):
        return None

    live = LiveCompanion(
        settings=lambda: settings,
        observe=observe,
        emit=emit,
        speak=speak,
        relevance_path=tmp_path / "rel.json",
        clock=clock,
        today=lambda: MONDAY,
        window_exists=window_exists,
    )
    return live, clock, events, settings


async def run_for(live, clock, seconds: float) -> None:
    for _ in range(int(seconds / TICK_SECONDS)):
        clock.t += TICK_SECONDS
        await live.tick()


def cards(events) -> list[dict]:
    return [p for e, p in events if e == "live_interjection"]


@pytest.mark.asyncio
async def test_page_text_with_focus_is_read_as_reading_not_writing(tmp_path):
    # A doubled word and a rewrite-like shrink in a page that has focus: no card of any kind.
    page = Observation(text="Work Work. Thanks Don't mention these " * 3, editable=False, **CLAUDE)
    obs: list[Observation | None] = [page]
    live, clock, events, _ = build(tmp_path, obs)
    await run_for(live, clock, 4)
    for shorter in ("Work Work", "Thanks", ""):
        obs[0] = Observation(text=shorter, editable=False, **CLAUDE)
        await run_for(live, clock, 3)
    assert cards(events) == []
    assert live.status()["watching"] == "claude.exe"


@pytest.mark.asyncio
async def test_text_already_in_a_field_is_not_the_persons_slip(tmp_path):
    obs: list[Observation | None] = [Observation(text="Notes for today", field="1", **CLAUDE)]
    live, clock, events, _ = build(tmp_path, obs)
    await run_for(live, clock, 3)  # the first field after Live is turned on counts as the person's own draft
    # Focus moves to another field that already holds a doubled word; the person adds to it.
    obs[0] = Observation(text="Work Work is the heading", field="2", **CLAUDE)
    await run_for(live, clock, 3)
    obs[0] = Observation(text="Work Work is the heading and the plan is fine", field="2", **CLAUDE)
    await run_for(live, clock, 6)
    assert cards(events) == []
    # Their own doubled word in the same field is still caught.
    obs[0] = Observation(text="Work Work is the heading and the the plan is fine", field="2", **CLAUDE)
    await run_for(live, clock, 8)
    [card] = cards(events)
    assert card["kind"] == "repeated_word" and card["evidence"] == "the the"


@pytest.mark.asyncio
async def test_a_slip_typed_again_counts_even_when_the_same_words_were_there(tmp_path):
    obs: list[Observation | None] = [Observation(text="start", field="1", **CLAUDE)]
    live, clock, events, _ = build(tmp_path, obs)
    await run_for(live, clock, 2)
    obs[0] = Observation(text="Quoted: we can sign sign now", field="2", **CLAUDE)
    await run_for(live, clock, 3)
    obs[0] = Observation(text="Quoted: we can sign sign now\nMe: fine, sign sign", field="2", **CLAUDE)
    await run_for(live, clock, 8)
    assert [c["kind"] for c in cards(events)] == ["repeated_word"]


@pytest.mark.asyncio
async def test_a_sent_message_clears_the_baseline(tmp_path):
    obs: list[Observation | None] = [Observation(text="start", field="1", **CLAUDE)]
    live, clock, events, _ = build(tmp_path, obs)
    await run_for(live, clock, 2)
    obs[0] = Observation(text="I think the the plan works", field="2", **CLAUDE)
    await run_for(live, clock, 6)
    assert cards(events) == []  # already there when focus arrived
    obs[0] = Observation(text="", field="2", **CLAUDE)  # sent
    await run_for(live, clock, 2)
    obs[0] = Observation(text="I think the the plan works", field="2", **CLAUDE)  # typed fresh
    await run_for(live, clock, 8)
    assert [c["kind"] for c in cards(events)] == ["repeated_word"]


@pytest.mark.asyncio
async def test_ocr_risk_notices_never_become_cards(tmp_path):
    live, clock, events, _ = build(tmp_path, [Observation(text="nothing here", **CLAUDE)])
    for _ in range(5):
        await live.notice_risk("credential_exposure", "ocr")
        await run_for(live, clock, 60)
    assert cards(events) == []
    assert live.status()["recent"] == []


@pytest.mark.asyncio
async def test_a_typed_risk_notice_is_one_normal_card_and_never_repeats(tmp_path):
    live, clock, events, _ = build(tmp_path, [Observation(text="nothing here", **CLAUDE)])
    for _ in range(4):
        await live.notice_risk("credential_exposure", "keystroke")
        await run_for(live, clock, 70)  # past the old once-a-minute key
    [card] = cards(events)
    assert card["kind"] == "secret_in_text" and card["urgency"] == "normal" and card["source"] == "notice"
    assert card["message"].startswith("What you just typed") and card["app"] == ""


@pytest.mark.asyncio
async def test_a_typed_risk_notice_respects_mutes_and_defers_to_the_field_card(tmp_path):
    live, clock, events, settings = build(tmp_path, [Observation(text="nothing here", **CLAUDE)])
    settings.muted_kinds = ["risky_command"]
    await live.notice_risk("destructive_shell_command", "keystroke")
    await run_for(live, clock, 5)
    assert cards(events) == []

    settings.muted_kinds = []
    key = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
    live2, clock2, events2, _ = build(
        tmp_path, [Observation(text=f"here is the key {key}", app="slack.exe", title="Slack", source="focused_field")]
    )
    await run_for(live2, clock2, 3)
    assert [c["kind"] for c in cards(events2)] == ["secret_in_text"]
    await live2.notice_risk("credential_exposure", "keystroke")  # the same thing, with no evidence
    await run_for(live2, clock2, 5)
    assert len(cards(events2)) == 1


# ── editors that put the text in the window title (Windows 11 Notepad, VS Code's unsaved dot) ──

NOTEPAD = {"app": "Notepad", "source": "focused_field", "field": "42.30540886"}


def notepad(text: str) -> Observation:
    # Windows 11 Notepad titles an untitled tab with its first line, so the title changes on every keystroke.
    return Observation(text=text, title=f"*{text.splitlines()[0][:40]} - Notepad", **NOTEPAD)


async def type_out(live, clock, obs, text: str, start: str = "", per_char: float = 0.12) -> None:
    for i in range(len(start) + 1, len(text) + 1):
        obs[0] = notepad(text[:i])
        clock.t += per_char
        await live.tick()


@pytest.mark.asyncio
async def test_a_slip_typed_where_the_title_follows_the_text_is_caught_and_its_fix_seen(tmp_path):
    # Live first saw another field (the Heliox prompt box, say), so every later field gets a baseline.
    obs: list[Observation | None] = [Observation(text="hello", field="9", **CLAUDE)]
    live, clock, events, _ = build(tmp_path, obs)
    await run_for(live, clock, 2)
    obs[0] = None  # a new, empty Notepad window: nothing to read yet
    await run_for(live, clock, 2)
    await type_out(live, clock, obs, "Launch call is Monday 12 Nov at 10am")
    await run_for(live, clock, 10)
    [card] = cards(events)
    assert card["title"] == "12 November is a Thursday" and card["app"] == "Notepad"
    # They retype the weekday; the title changes with every key, and the card still sees its slip go.
    for partial in ("Launch call is Monda 12 Nov at 10am", "Launch call is  12 Nov at 10am"):
        obs[0] = notepad(partial)
        await run_for(live, clock, 0.6)
    for i in range(1, len("Thursday") + 1):
        obs[0] = notepad(f"Launch call is {'Thursday'[:i]} 12 Nov at 10am")
        await run_for(live, clock, 0.2)
    await run_for(live, clock, 3)
    for _ in range(3):
        await asyncio.sleep(0)  # the withdrawal is sent from a task
    assert ("live_interjection_withdrawn", {"id": card["id"], "reason": "fixed"}) in events
    assert len(cards(events)) == 1


@pytest.mark.asyncio
async def test_text_already_there_stays_quiet_when_the_title_changes_as_they_type(tmp_path):
    obs: list[Observation | None] = [Observation(text="hello", field="9", **CLAUDE)]
    live, clock, events, _ = build(tmp_path, obs)
    await run_for(live, clock, 2)
    # An opened file that already says Monday 12 Nov, and Work Work; the person adds a line under it.
    before = "Kickoff is Monday 12 Nov.\nWork Work is the heading.\n"
    obs[0] = Observation(text=before, title="plan.txt - Notepad", **NOTEPAD)
    await run_for(live, clock, 3)
    await type_out(live, clock, obs, before + "Bring the slides.", start=before)
    await run_for(live, clock, 10)
    assert cards(events) == []


@pytest.mark.asyncio
async def test_another_tab_in_the_same_editor_control_is_not_the_persons_text(tmp_path):
    obs: list[Observation | None] = [Observation(text="hello", field="9", **CLAUDE)]
    live, clock, events, _ = build(tmp_path, obs)
    await run_for(live, clock, 2)
    await type_out(live, clock, obs, "Shopping list")
    await run_for(live, clock, 2)
    # A tab switch that keeps the same editor element: its whole text changes in one look.
    obs[0] = Observation(text="Old notes: the the review is Monday 12 Nov", title="old.txt - Notepad", **NOTEPAD)
    await run_for(live, clock, 10)
    assert cards(events) == []


@pytest.mark.asyncio
async def test_a_cleared_field_typed_again_is_checked_even_when_empty_reads_never_arrive(tmp_path):
    # The daemon passes no observation for an empty field, so "it was emptied" is never seen directly.
    obs: list[Observation | None] = [Observation(text="hello", field="9", **CLAUDE)]
    live, clock, events, _ = build(tmp_path, obs)
    await run_for(live, clock, 2)
    chat = {"app": "slack", "title": "Slack", "source": "focused_field", "field": "7"}
    obs[0] = Observation(text="I think the the plan works", **chat)
    await run_for(live, clock, 6)
    assert cards(events) == []  # already there when focus arrived
    obs[0] = None  # sent: the box is empty
    await run_for(live, clock, 1)
    message = "I think the the plan works"
    for i in range(1, len(message) + 1):
        obs[0] = Observation(text=message[:i], **chat)
        clock.t += 0.12
        await live.tick()
    await run_for(live, clock, 8)
    assert [c["kind"] for c in cards(events)] == ["repeated_word"]


def test_replaced_tells_a_swap_from_an_edit():
    from pilot.live.engine import replaced

    assert not replaced("Launch call", "Launch call is M")  # typing
    assert not replaced("Launch call is Monday 12 Nov", "Launch call is  12 Nov")  # deleting a word
    assert not replaced("Launch call is Monday 12 Nov", "Launch call is Thursday 12 Nov")  # retyping it
    assert not replaced("Hi Bob", "Hi Tom")  # short fields need at least a few characters gone
    assert not replaced("A" * 300, "A" * 150)  # a long cut that leaves most of the text
    assert replaced("I think the the plan works", "I t")  # sent, then typing again
    assert replaced("Shopping list", "Old notes: the the review is Monday")  # another tab
    assert not replaced("same", "same")


def test_explain_says_plainly_when_there_is_nothing_to_explain(tmp_path):
    live, _, _, _ = build(tmp_path, [None])
    assert live.explain() == "Nothing to explain: Live hasn't raised anything in the last minute and a half."
    assert live.explain("abc") == "That correction is no longer available."


# ── a card whose window was closed ──


def test_window_handle_reads_the_window_from_a_runtime_id():
    from pilot.live.focused_text import window_handle

    assert window_handle("42.30540886") == 30540886
    assert window_handle("42.88540834.4.7.4.212") == 88540834
    assert window_handle("42.-5") == 2**32 - 5  # printed as a signed number
    assert window_handle("7.1234") is None  # not hosted in a window
    assert window_handle("42") is None and window_handle("") is None and window_handle("42.x") is None
    assert window_handle("42.0") is None


def test_window_exists_cannot_say_no_off_windows(monkeypatch):
    from pilot.live import focused_text

    monkeypatch.setattr(focused_text.os, "name", "posix")
    assert focused_text.window_exists(1234) is True


@pytest.mark.asyncio
async def test_a_card_retires_when_its_window_is_closed_but_not_when_it_loses_focus(tmp_path):
    open_windows = {30540886}
    obs: list[Observation | None] = [Observation(text="hello", field="9", **CLAUDE)]
    live, clock, events, _ = build(tmp_path, obs, window_exists=lambda handle: handle in open_windows)
    await run_for(live, clock, 2)
    await type_out(live, clock, obs, "Invoice due 31 November")
    await run_for(live, clock, 15)
    [card] = cards(events)
    obs[0] = Observation(text="hello", field="9", **CLAUDE)  # focus moves away; Notepad is still open
    await run_for(live, clock, 20)
    assert not [p for e, p in events if e == "live_interjection_withdrawn"]
    open_windows.clear()  # closed without saving
    await run_for(live, clock, 2)
    for _ in range(3):
        await asyncio.sleep(0)
    assert ("live_interjection_withdrawn", {"id": card["id"], "reason": "closed"}) in events
    assert live.relevance.score("weekday_date") == pytest.approx(live.relevance.score("never_seen_kind"))
