"""Live says once, quietly, when it cannot see where the person types (pilot.live.blind_spots)."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from pilot.live.blind_spots import BLIND_AFTER_SECONDS, BlindSpots, advice, kind_of, sentence
from pilot.live.engine import LiveCompanion, Observation


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def look_for(spots, clock, app, seconds, readable=False, **kw):
    said = []
    for _ in range(int(seconds / 0.6)):
        clock.t += 0.6
        out = spots.note(app, readable, **kw)
        if out:
            said.append(out)
    return said


# ── what is said ──


def test_the_table_names_what_would_let_live_see():
    mac = advice("Notes", platform="darwin", reader_state="needs_permission")
    assert "Accessibility for Heliox-OS in System Settings" in mac.fix
    assert "remote control" in advice("kitty", platform="linux").fix
    assert "terminal check" in advice("alacritty", platform="darwin").fix
    assert advice("SomeGame", platform="win32").fix == ""
    assert kind_of("Code.exe") == "electron" and kind_of("Google Chrome.app") == "chromium"


@pytest.mark.parametrize("app", ["google-chrome", "chromium", "slack", "discord", "code"])
@pytest.mark.parametrize("switch_off", [True, False])
def test_chrome_and_electron_on_linux_get_what_the_runs_showed(app, switch_off):
    """CI on 9aa37d8 and 49bd35f: read with the desktop switch on and the app started after it, no flag; started
    before the switch, unread until restarted. The flag is never the step."""
    found = advice(app, platform="linux", reader_message="toolkit accessibility is off" if switch_off else "")
    if switch_off:
        assert found.fix == (
            "turn on accessibility for apps (gsettings set org.gnome.desktop.interface toolkit-accessibility true), "
            "then restart it"
        )
    else:
        assert found.fix == "restart it, so it starts with accessibility for apps on"
    assert "9aa37d8" in found.shown and "--force-renderer-accessibility" not in found.fix


def test_qt_on_linux():
    assert advice("telegram-desktop", platform="linux", reader_message="off").fix == (
        "turn on accessibility for apps (gsettings set org.gnome.desktop.interface toolkit-accessibility true), "
        "then restart it"
    )
    assert "usually can" in advice("telegram-desktop", platform="linux").fix  # switch on: read by default


@pytest.mark.parametrize(("app", "platform"), [("firefox", "linux"), ("Safari", "darwin")])
def test_apps_read_by_default_are_only_named_if_the_reader_really_gives_nothing(app, platform):
    """Firefox on X11 and Safari on macOS gave fields by default; if one ever gives nothing, the sentence says so."""
    for message in ("", "toolkit accessibility is off"):
        found = advice(app, platform=platform, reader_message=message)
        assert "usually can" in found.fix and found.shown
    clock = Clock()
    spots = BlindSpots(None, clock)
    look_for(spots, clock, app, 1.2, readable=True)  # a field, as by default
    assert look_for(spots, clock, app, 300, platform=platform) == []  # never named once a field was seen


def test_a_cell_a_run_shows_working_becomes_the_advice(monkeypatch):
    """The first set-up a run showed reading is the step; here a kind only its own flag reads."""
    from pilot.live import blind_spots

    cells = {
        ("off", "off"): blind_spots.Finding(False, "a run"),
        ("off", "on"): blind_spots.Finding(True, "a later run"),
    }
    monkeypatch.setitem(blind_spots.LINUX_FINDINGS, "qt", cells)
    found = advice("telegram-desktop", platform="linux", reader_message="off")
    assert found.fix == "start it with QT_LINUX_ACCESSIBILITY_ALWAYS_ON=1, then restart it"
    assert found.shown == "a later run"


def test_the_sentence():
    assert sentence("google-chrome", advice("google-chrome", platform="linux")) == (
        "Live can't see what you type in Google Chrome: restart it, so it starts with accessibility for apps on."
    )
    assert sentence("python3", advice("python3", platform="linux")) == "Live doesn't work in Python3 yet."
    assert sentence("SomeGame", advice("SomeGame", platform="win32")) == "Live doesn't work in SomeGame yet."


# ── when it is said ──


def test_said_once_after_enough_time_in_front_with_nothing_to_read(tmp_path):
    clock = Clock()
    spots = BlindSpots(tmp_path / "b.json", clock)
    assert look_for(spots, clock, "slack", BLIND_AFTER_SECONDS - 2, platform="linux") == []
    said = look_for(spots, clock, "slack", 4, platform="linux")
    assert len(said) == 1 and "Slack" in said[0] and spots.current == said[0]
    assert look_for(spots, clock, "slack", 120, platform="linux") == []  # never again for that app
    look_for(spots, clock, "gedit", 1.2, readable=True)
    assert spots.current == ""  # another app in front: the first-time sentence is not shown for it
    look_for(spots, clock, "slack", 3, platform="linux")
    assert spots.current == ""  # back in Slack: not said again
    assert BlindSpots(tmp_path / "b.json", clock).notices() == [{"app": "slack", "sentence": said[0]}]  # remembered


def test_visits_add_up_but_time_away_does_not(tmp_path):
    clock = Clock()
    spots = BlindSpots(None, clock)
    for _ in range(3):
        look_for(spots, clock, "slack", 14)
        clock.t += 60  # away for a minute: not counted
    assert spots.notices() == []
    assert look_for(spots, clock, "slack", 10) != []  # about 3 x 13 s + 9 s in front


def test_an_app_where_live_has_seen_a_field_is_never_called_blind(tmp_path):
    clock = Clock()
    spots = BlindSpots(None, clock)
    look_for(spots, clock, "firefox", 2, readable=True)
    assert look_for(spots, clock, "firefox", 300) == []  # the person is only between fields


def test_seeing_a_field_later_takes_the_line_away(tmp_path):
    clock = Clock()
    spots = BlindSpots(tmp_path / "b.json", clock)
    look_for(spots, clock, "slack", BLIND_AFTER_SECONDS + 2)
    assert spots.notices()
    look_for(spots, clock, "slack", 1.2, readable=True)  # turned on: Live sees into it now
    assert spots.notices() == [] and BlindSpots(tmp_path / "b.json", clock).notices() == []


def test_windows_nobody_types_in_are_never_called_blind():
    clock = Clock()
    spots = BlindSpots(None, clock)
    for app in ("Finder", "explorer.exe", "vlc", "Unknown", ""):
        assert look_for(spots, clock, app, 300) == []


def test_asking_again_from_settings_says_it_again():
    clock = Clock()
    spots = BlindSpots(None, clock)
    look_for(spots, clock, "slack", BLIND_AFTER_SECONDS + 2)
    spots.forget("slack")
    assert look_for(spots, clock, "slack", 2) != []  # the stretch in front already counted


# ── in Live, and through the daemon's own observe ──


def build(tmp_path, enabled=True):
    settings = SimpleNamespace(enabled=enabled, level="balanced", listen_to_speech=True, muted_kinds=[])
    events = []

    async def observe():
        return None

    async def emit(event, payload):
        events.append((event, payload))

    async def speak(*_):
        return "spoken"

    clock = Clock()
    live = LiveCompanion(settings=lambda: settings, observe=observe, emit=emit, speak=speak,
                         relevance_path=tmp_path / "r.json", clock=clock, today=lambda: date(2026, 11, 12))  # fmt: skip
    return live, clock, settings, events


def test_live_off_or_paused_says_nothing_and_the_status_carries_it(tmp_path):
    live, clock, settings, _ = build(tmp_path, enabled=False)
    for _ in range(100):
        clock.t += 0.6
        assert live.note_foreground("slack", False) == ""
    settings.enabled = True
    live.set_paused(True)
    clock.t += 0.6
    assert live.note_foreground("slack", False) == ""
    live.set_paused(False)
    said = ""
    for _ in range(100):
        clock.t += 0.6
        said = said or live.note_foreground("slack", False)
    assert said and live.status()["blind_spots"] == [{"app": "slack", "sentence": said}]
    assert live.status()["blind_note"] == said
    assert (tmp_path / "live_blind_spots.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("app", "title", "noted"),
    [
        ("Slack", "general - Team", True),
        ("Notepad", "notes.txt - Notepad", False),  # excluded by the person
        ("Bitwarden", "My Vault", False),  # a password manager
        ("Heliox-OS", "Heliox-OS", False),  # Heliox itself
    ],
)
async def test_only_windows_past_the_never_read_rules_are_counted(monkeypatch, app, title, noted):
    from pilot.agents import screen_vision
    from pilot.config import PilotConfig
    from pilot.server import PilotServer

    monkeypatch.setattr(screen_vision, "_get_active_window", lambda: (app, title))
    seen = []
    config = PilotConfig()
    config.timeline.excluded_apps = ["Notepad"]

    async def read():
        return None  # the reader gives nothing for this window

    reader = SimpleNamespace(read=read, status=lambda: SimpleNamespace(state="ready", message=""))
    live = SimpleNamespace(note_foreground=lambda *args: seen.append(args) or "")
    server = SimpleNamespace(config=config, _live_reader=reader, _live_last_ocr=0.0, _live=live)
    assert await PilotServer._live_observe(server) is None
    assert bool(seen) is noted
    if noted:
        assert seen[0][:2] == ("Slack", False)


def test_a_terminal_live_reads_directly_counts_as_seen_into():
    from pilot.personal.wiring import live_blind_note

    seen = []
    terminals = SimpleNamespace(status=lambda: {"kitty": {"found": True, "ready": True, "needs": ""}})
    server = SimpleNamespace(_live=SimpleNamespace(note_foreground=lambda *a: seen.append(a)), _terminal_text=terminals)
    live_blind_note(server, "kitty", None)
    assert seen[0][1] is True
    terminals.status = lambda: {"kitty": {"found": True, "ready": False, "needs": "add remote control"}}
    live_blind_note(server, "kitty", None)
    assert seen[1][1] is False


def test_nothing_new_is_captured():
    """The only inputs are the app's name, whether the reader gave anything, and the reader's own status."""
    import inspect

    from pilot.live import blind_spots

    source = inspect.getsource(blind_spots)
    assert "pynput" not in source and "screen_ocr" not in source and "keyboard" not in source
    assert Observation  # the observation itself is never passed in
