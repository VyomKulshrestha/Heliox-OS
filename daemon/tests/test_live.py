"""Heliox Live: specific detectors, considerate timing, and the micro-turn engine end to end."""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import pytest

from pilot.live.detectors import TextContext, detect
from pilot.live.engine import TICK_SECONDS, LiveCompanion, Observation
from pilot.live.policy import LEVELS, SHOW, SPEAK, WAIT, Activity, Pending, Relevance, decide

MONDAY = date(2026, 11, 9)


def ctx(**kw) -> TextContext:
    return TextContext(today=MONDAY, people={"manager": "Priya", "friend": "Sam"}, **kw)


def kinds(text: str, **kw) -> list[str]:
    return [c.kind for c in detect(text, ctx(**kw))]


# ── detectors ──


def test_weekday_date_names_the_real_day_and_offers_both_readings():
    [c] = detect("Let's meet on Tuesday, 14 November to review.", ctx(app="chrome", title="Gmail"))
    assert c.message == "14 November is a Saturday, not a Tuesday."
    assert c.fix == "Saturday, 14 November"
    assert "17 November" in c.detail  # the nearest Tuesday, if the date was the slip
    assert detect("Meeting Tue Nov 17 works", ctx()) == []
    assert detect("See you Monday, 9 November", ctx()) == []
    assert "There's no 31 February" in detect("Friday 31 February 2027", ctx())[0].title


def test_arithmetic_checks_sums_and_percentages_but_not_code():
    [c] = detect("The total is 12 x 15 = 170 so we order more", ctx(app="winword"))
    assert c.message == "12 × 15 is 180, not 170." and c.fix == "12 x 15 = 180"
    assert detect("20% of 150 = 35", ctx())[0].message == "20% of 150 is 30, not 35."
    assert detect("2 + 2 = 4 and 10 / 4 = 2.5", ctx()) == []
    assert detect("total = 5 + 3", ctx(app="Code.exe", title="main.py - Visual Studio Code")) == []


def test_risky_commands_are_urgent_and_carry_the_safer_command():
    [c] = detect("PS C:\\repo> git push origin main --force", ctx(app="WindowsTerminal", title="PowerShell"))
    assert c.urgency == "urgent" and c.fix == "git push origin main --force-with-lease"
    assert c.evidence == "git push origin main --force"
    assert kinds("$ rm -rf /", app="WindowsTerminal", title="bash") == ["risky_command"]
    assert kinds("DELETE FROM users;", app="dbeaver") == ["risky_command"]
    for safe in ("$ rm -rf /tmp/build", "git push --force-with-lease", "DELETE FROM users WHERE id = 4;"):
        assert kinds(safe, app="WindowsTerminal", title="bash") == [], safe


def test_old_terminal_history_is_not_the_command_about_to_run():
    history = "$ git reset --hard\n" + "\n".join(f"line {i}" for i in range(10)) + "\n$ ls"
    assert kinds(history, app="WindowsTerminal", title="bash") == []


def test_secrets_in_messages_are_redacted_and_ignored_where_they_belong():
    [c] = detect("here is the key sk-proj-abcdefghijklmnopqrstuvwxyz123456", ctx(app="slack", title="Slack"))
    assert c.urgency == "urgent" and "sk-proj-abc" not in c.evidence and c.evidence.startswith("sk-p")
    assert (
        kinds(
            "OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz123456", app="Code.exe", title=".env - Visual Studio Code"
        )
        == []
    )
    assert kinds("sk-proj-abcdefghijklmnopqrstuvwxyz123456", app="1Password") == []


def test_profile_facts_only_for_people_you_have_one_of():
    [c] = detect("I will email my manager Sarah tomorrow", ctx())
    assert c.message == "You told me your manager is Priya, not Sarah." and c.fix == "my manager Priya"
    assert detect("lunch with my manager Priya", ctx()) == []
    assert detect("drinks with my friend Alex", ctx()) == []  # you can have many friends


def test_errors_on_screen_come_with_the_specific_fix():
    [c] = detect("Traceback...\nModuleNotFoundError: No module named 'cv2'", ctx(app="WindowsTerminal"))
    assert c.fix == "pip install opencv-python" and c.fix_kind == "command"
    [port] = detect("Error: listen EADDRINUSE: address already in use :::3000", ctx(app="WindowsTerminal"))
    assert port.fix_kind == "ask" and "3000" in port.fix
    [missing] = detect("'ffmpeg' is not recognized as an internal or external command", ctx(app="cmd.exe"))
    assert "ffmpeg" in missing.title


def test_prose_checks_stay_in_prose_and_speech_gets_only_the_spoken_checks():
    assert kinds("Hi team, please find attached the report.", app="chrome", title="Gmail - Compose") == [
        "missing_attachment"
    ]
    assert kinds("I think the the plan is good", app="notepad") == ["repeated_word"]
    assert kinds("I had had enough", app="notepad") == []
    spoken = detect("book lunch on Tuesday 14 November", ctx(source="speech"))
    assert [c.kind for c in spoken] == ["weekday_date"]
    assert detect("rm -rf /", ctx(source="speech")) == []


# ── policy ──


def pending(text: str, source: str = "focused_field", first_seen: float = 0.0, **kw) -> Pending:
    [c] = detect(text, ctx(source=source, **kw))
    return Pending(c, source, first_seen, first_seen, window="app|title")


def test_policy_waits_for_a_pause_and_never_talks_over_the_person():
    item = pending("12 x 15 = 170 units")
    level = LEVELS["balanced"]
    typing = Activity(now=10.0, last_text_change=9.8)
    assert decide(item, typing, level, 0.67, frozenset()) == WAIT
    paused = Activity(now=10.0, last_text_change=8.0)
    assert decide(item, paused, level, 0.67, frozenset()) == SPEAK
    talking = Activity(now=10.0, last_text_change=8.0, user_speaking=True)
    assert decide(item, talking, level, 0.67, frozenset()) == WAIT


def test_a_mistake_at_the_end_of_the_text_gets_a_longer_pause_and_stale_text_none():
    item = pending("12 x 15 = 17")  # maybe they are still typing 170
    level = LEVELS["balanced"]
    assert decide(item, Activity(now=10.0, last_text_change=8.0), level, 0.67, frozenset()) == WAIT
    assert decide(item, Activity(now=10.0, last_text_change=6.9), level, 0.67, frozenset()) == SPEAK
    item.missing_since = 9.0  # the text it was about has gone
    assert decide(item, Activity(now=10.0, last_text_change=1.0), level, 0.67, frozenset()) == WAIT


def test_urgent_lands_at_the_first_short_pause_even_in_focus():
    item = pending("$ rm -rf /", app="WindowsTerminal", title="bash")
    act = Activity(now=10.0, last_text_change=9.3, focus_active=True, last_spoken_at=9.9)
    assert decide(item, act, LEVELS["quiet"], 0.1, frozenset()) == SPEAK
    assert (
        decide(item, Activity(now=10.0, last_text_change=9.3, heliox_speaking=True), LEVELS["quiet"], 1, frozenset())
        == SHOW
    )


def test_focus_rate_limits_relevance_and_mutes_hold_back_ordinary_corrections():
    item = pending("12 x 15 = 170")
    level = LEVELS["balanced"]
    base = dict(now=500.0, last_text_change=490.0)
    assert decide(item, Activity(**base, focus_active=True), level, 0.9, frozenset()) == WAIT
    assert decide(item, Activity(**base, last_spoken_at=450.0), level, 0.9, frozenset()) == SHOW  # spoke recently
    assert decide(item, Activity(**base), level, 0.4, frozenset()) == SHOW  # not welcome enough to speak
    assert decide(item, Activity(**base), level, 0.1, frozenset()) == "drop"
    assert decide(item, Activity(**base), level, 0.9, frozenset({"arithmetic"})) == "drop"
    assert (
        decide(pending("I think the the plan", app="notepad"), Activity(**base), LEVELS["quiet"], 0.9, frozenset())
        == "drop"
    )


def test_relevance_learns_and_persists(tmp_path):
    path = tmp_path / "live_relevance.json"
    rel = Relevance(path)
    assert rel.score("arithmetic") == pytest.approx(2 / 3)
    rel.update("arithmetic", "dismissed")
    rel.update("arithmetic", "dismissed")
    assert rel.score("arithmetic") == pytest.approx(0.4)
    assert Relevance(path).score("arithmetic") == pytest.approx(0.4)
    rel.update("weekday_date", "fixed")
    assert rel.score("weekday_date") > 2 / 3


# ── the engine, one micro-turn at a time ──


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def build(tmp_path, text: list[str], **overrides):
    clock = Clock()
    events: list[tuple[str, dict]] = []
    spoken: list[tuple[str, bool]] = []
    settings = SimpleNamespace(enabled=True, level="balanced", listen_to_speech=True, muted_kinds=[])
    flags = {"speaking": False, "focus": False}

    async def observe():
        return Observation(text=text[0], app="notepad.exe", title="notes.txt - Notepad", source="focused_field")

    async def emit(event, payload):
        events.append((event, payload))

    async def speak(message, urgent):
        spoken.append((message, urgent))

    async def mute(kind):
        settings.muted_kinds.append(kind)

    live = LiveCompanion(
        settings=lambda: settings,
        observe=observe,
        emit=emit,
        speak=speak,
        user_speaking=lambda: flags["speaking"],
        focus_active=lambda: flags["focus"],
        people=lambda: {"manager": "Priya"},
        mute_kind=mute,
        relevance_path=tmp_path / "rel.json",
        clock=clock,
        today=lambda: MONDAY,
        **overrides,
    )
    return live, clock, events, spoken, settings, flags


async def run_for(live, clock, seconds: float) -> None:
    for _ in range(int(seconds / TICK_SECONDS)):
        clock.t += TICK_SECONDS
        await live.tick()


async def settle():
    import asyncio

    for _ in range(3):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_engine_waits_for_the_pause_speaks_once_and_withdraws_when_fixed(tmp_path):
    text = ["Invoice total: 12 x 15 = 1"]
    live, clock, events, spoken, _, _ = build(tmp_path, text)
    await run_for(live, clock, 0.8)
    text[0] = "Invoice total: 12 x 15 = 17"
    await run_for(live, clock, 0.6)
    text[0] = "Invoice total: 12 x 15 = 170"  # still typing: nothing yet
    await run_for(live, clock, 0.6)
    assert not any(e == "live_interjection" for e, _ in events)
    await run_for(live, clock, 1.6)  # a pause, but the sum is the last thing typed: they may not be done
    assert not any(e == "live_interjection" for e, _ in events)
    await run_for(live, clock, 2.0)  # a longer pause
    await settle()
    cards = [p for e, p in events if e == "live_interjection"]
    assert len(cards) == 1 and cards[0]["kind"] == "arithmetic" and cards[0]["spoken"]
    assert spoken == [("12 × 15 is 180, not 170.", False)]

    await run_for(live, clock, 5)  # the mistake is still there: never repeated
    assert len([e for e, _ in events if e == "live_interjection"]) == 1

    text[0] = "Invoice total: 12 x 15 = 180"  # they fixed it after being told
    await run_for(live, clock, 3)
    await settle()
    assert ("live_interjection_withdrawn", {"id": cards[0]["id"], "reason": "fixed"}) in events
    assert live.relevance.score("arithmetic") > 2 / 3


@pytest.mark.asyncio
async def test_engine_holds_still_while_the_person_talks_and_catches_what_they_say(tmp_path):
    live, clock, events, spoken, _, flags = build(tmp_path, ["nothing to see here"])
    flags["speaking"] = True
    live.offer_speech("put lunch with my manager Sarah on Tuesday 14 November")
    await run_for(live, clock, 1)
    assert spoken == []
    flags["speaking"] = False
    await run_for(live, clock, 1)
    await settle()
    # Both catches from one sentence come back as one reply, and both get a card.
    assert spoken == [
        ("14 November is a Saturday, not a Tuesday. You told me your manager is Priya, not Sarah.", False)
    ]
    assert sorted(p["kind"] for e, p in events if e == "live_interjection") == ["profile_fact", "weekday_date"]


@pytest.mark.asyncio
async def test_replies_thanks_why_fix_and_mute(tmp_path):
    text = ["Error: listen EADDRINUSE: address already in use :::3000"]
    live, clock, events, spoken, settings, _ = build(tmp_path, text)
    await run_for(live, clock, 8)  # errors get a moment to be read first
    await settle()
    [card] = [p for e, p in events if e == "live_interjection"]
    assert card["kind"] == "error_on_screen"
    why = await live.handle_reply("why")
    assert "port 3000" in why.reply
    fix = await live.handle_reply("fix it")
    assert fix.delegate == "Show me which program is using port 3000"
    assert await live.handle_reply("what time is it") is None
    assert (await live.feedback(card["id"], "mute"))["status"] == "ok"
    assert settings.muted_kinds == ["error_on_screen"]


@pytest.mark.asyncio
async def test_disabled_or_paused_does_nothing_and_stop_withdraws(tmp_path):
    text = ["I think the the plan is good"]
    live, clock, events, spoken, settings, _ = build(tmp_path, text)
    settings.enabled = False
    await run_for(live, clock, 4)
    assert not [e for e, _ in events if e == "live_interjection"]
    settings.enabled = True
    live.set_paused(True)
    await run_for(live, clock, 4)
    assert not [e for e, _ in events if e == "live_interjection"]
    live.set_paused(False)
    await run_for(live, clock, 4)
    await settle()
    [card] = [p for e, p in events if e == "live_interjection"]
    assert card["kind"] == "repeated_word" and not card["spoken"]  # minor: a card, not a voice
    await live.stop()
    assert ("live_interjection_withdrawn", {"id": card["id"], "reason": "off"}) in events
    assert live.status()["enabled"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("hour,expected", [(4, 0), (5, 1)])
async def test_engine_passes_its_own_hour_to_the_relative_day_rule(tmp_path, hour, expected):
    text = ["Standup is tomorrow, Thursday at 9."]  # the engine's day is Monday, so tomorrow is Tuesday
    live, clock, events, _, _, _ = build(tmp_path, text, wallclock=lambda: datetime(2026, 11, 9, hour, 30))
    await run_for(live, clock, 8)
    await settle()
    assert live._context(None, "focused_field").hour == hour
    cards = [p for e, p in events if e == "live_interjection" and p["kind"] == "weekday_date"]
    assert len(cards) == expected
