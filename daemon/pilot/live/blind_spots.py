"""Telling the person, once and quietly, when Live cannot see where they type in an app.

A person who types in an app Live cannot read gets silence, and cannot tell "nothing was wrong" from "Live is blind
here". This notices it from what Live already has, and nothing new is captured:

* the foreground app and title, after the never-read rules (Heliox's own window, password managers, excluded apps
  never reach this: the caller returns before);
* the reader's answer for that window: a field (or a password field, or page text in focus) means Live can see into
  the app; nothing at all means it cannot;
* time: the app has to have been the window in front for ``BLIND_AFTER_SECONDS`` in all (visits joined up) with the
  reader never giving anything for it. If the reader has ever given a field for that app in this run, the person is
  only between fields and nothing is said.

Then one sentence names the app and what would let Live see it there (``ADVICE``, per system and toolkit), or says
that Live does not work in that app. It is said once per app: in the Live pill's tooltip the first time, and as a
standing line in Live's settings (asking there is asking again). Never spoken, never a card. It is remembered across
restarts (``path``), and forgotten for an app as soon as Live does see a field there.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pilot.system.app_names import app_label

logger = logging.getLogger("pilot.live.blind_spots")

BLIND_AFTER_SECONDS = 45.0  # in front this long, all visits together, with nothing to read
VISIT_GAP_SECONDS = 5.0  # looks further apart than this do not add up as one stretch in front

# Windows where nobody types, or that are the system's own: nothing is said about them.
NOT_TYPED_IN = frozenset(
    {
        "", "unknown", "explorer", "finder", "dock", "loginwindow", "systemuiserver", "controlcenter",
        "gnome-shell", "plasmashell", "xfdesktop", "nautilus", "dolphin", "thunar", "nemo", "pcmanfm",
        "vlc", "mpv", "totem", "quicktime player", "preview", "photos", "eog", "loupe", "gwenview",
        "system settings", "system preferences", "gnome-control-center", "lockscreen", "screensaverengine",
    }
)  # fmt: skip


@dataclass(frozen=True)
class Advice:
    """What lets Live see into an app of one kind on one system. ``shown`` says what has proved it so far (the
    platform checks fill these in): "" for nothing yet, else the check and run that showed it."""

    fix: str  # completes "Live can't see what you type in {app}: ..."; "" when there is nothing to turn on
    shown: str = ""


# Known kinds of app by their process name (case-folded, without .exe or .app). The platform agent's checks say per
# app whether the default gives a field and what turns it on; each finding goes into this table.
CHROMIUM = frozenset(
    {"chrome", "google-chrome", "google chrome", "chromium", "chromium-browser", "brave", "brave-browser",
     "microsoft-edge", "msedge", "microsoft edge", "vivaldi", "opera"}
)  # fmt: skip
ELECTRON = frozenset(
    {"slack", "discord", "code", "code - oss", "teams", "microsoft teams", "signal", "signal-desktop", "obsidian",
     "element", "whatsapp", "notion", "spotify", "cursor"}
)  # fmt: skip
QT = frozenset({"telegram-desktop", "telegram", "kate", "okular", "krita", "qbittorrent", "kdenlive", "konsole"})
GECKO = frozenset({"firefox", "firefox-esr", "thunderbird", "librewolf"})
TERMINALS = frozenset(
    {"kitty", "alacritty", "wezterm", "wezterm-gui", "foot", "xterm", "urxvt", "ghostty", "hyper", "tabby", "warp"}
)
TOOLKIT_SWITCH = "gsettings set org.gnome.desktop.interface toolkit-accessibility true"


def kind_of(app: str) -> str:
    name = app.strip().casefold().removesuffix(".exe").removesuffix(".app")
    for kind, names in (
        ("chromium", CHROMIUM),
        ("electron", ELECTRON),
        ("qt", QT),
        ("gecko", GECKO),
        ("terminal", TERMINALS),
    ):
        if name in names:
            return kind
    return ""


@dataclass(frozen=True)
class Finding:
    """What a real run showed for one set-up of one kind of app: ``reads`` True (a field was read), False (shown not
    read), None (not run yet); ``shown`` names the run and check."""

    reads: bool | None = None
    shown: str = ""


# Linux: per kind of app, the four set-ups a person can choose: the desktop's accessibility switch (TOOLKIT_SWITCH)
# off or on, and the app's own flag (LINUX_FLAGS) off or on. The platform agent's runs fill in each cell. Live offers a
# fix only when a cell with something turned on shows a field read; with none, the sentence is that Live does not work
# in the app yet.
R22 = "platform-readers on 7f553b9 (ci_r22)"
# CI on 9aa37d8 and 49bd35f (runs 43295213661 and 43337605746, apps_fields_check.py), after fa393b8 asks each window
# for its attributes: with the desktop switch on and the app started after it, Chrome, Electron and Qt fields are read
# with no flag, passwords refused 17 of 17. Started before the switch was turned on ("late"), Chrome and Electron stay
# unread until they are restarted. The app's flag without the switch does not read Chrome or Electron (Qt's does).
# ACCESSIBILITY_ENABLED=1 in the app's environment reads Chrome and Electron without the switch: something Heliox could
# set for an app it starts itself, not a step to tell a person.
R9A = "platform-readers on 9aa37d8 and 49bd35f"
LINUX_FINDINGS: dict[str, dict[tuple[str, str], Finding]] = {
    "chromium": {
        ("off", "off"): Finding(False, f"{R9A}: Chrome not read by default"),
        ("on", "off"): Finding(
            True, f"{R9A}: Chrome read with the switch on, started after it; password refused 17 of 17"
        ),
        ("off", "on"): Finding(False, f"{R9A}: --force-renderer-accessibility alone, not read"),
        ("on", "on"): Finding(True, f"{R9A}: read with the switch on and the flag"),
    },
    "electron": {
        ("off", "off"): Finding(False, f"{R9A}: not read by default"),
        ("on", "off"): Finding(True, f"{R9A}: read with the switch on, started after it; password refused 17 of 17"),
        ("off", "on"): Finding(False, f"{R9A}: --force-renderer-accessibility alone, not read"),
        ("on", "on"): Finding(True, f"{R9A}: read with the switch on and the flag"),
    },
    "gecko": {("off", "off"): Finding(True, f"{R22}: Firefox read by default, password refused 17 of 17")},
    "qt": {
        ("off", "off"): Finding(False, f"{R9A}: not read by default"),
        ("on", "off"): Finding(True, f"{R9A}: read with the switch on; password refused 17 of 17"),
        ("off", "on"): Finding(True, f"{R9A}: read with QT_LINUX_ACCESSIBILITY_ALWAYS_ON=1"),
    },
}
# Kinds a run showed unread when the switch was turned on after the app started: restarting the app is the step.
LINUX_LATE_UNREAD: dict[str, str] = {
    "chromium": f"{R9A}: the switch turned on after Chrome started, not read",
    "electron": f"{R9A}: the switch turned on after the app started, not read",
}
LINUX_FLAGS = {
    "chromium": "--force-renderer-accessibility",
    "electron": "--force-renderer-accessibility",
    "qt": "QT_LINUX_ACCESSIBILITY_ALWAYS_ON=1",
}
# macOS: apps a run showed read with the default set-up (Accessibility granted to Heliox-OS).
MAC_READ_BY_DEFAULT: dict[str, str] = {"safari": f"{R22}: Safari read by default, password refused 14 of 14"}


def _usually_read(app: str, shown: str) -> Advice:
    """An app Live does read with the default set-up, giving nothing right now."""
    return Advice("it usually can, so restarting it may bring it back", shown)


def advice(app: str, *, platform: str = sys.platform, reader_state: str = "ready", reader_message: str = "") -> Advice:
    """What would let Live see into ``app`` here, from what real runs have shown (``LINUX_FINDINGS``,
    ``MAC_READ_BY_DEFAULT``). Never a fix that no run has shown to work, except for kinds not run yet."""
    kind = kind_of(app)
    name = _key(app)
    mac, linux = platform == "darwin", platform.startswith("linux")
    if mac and reader_state == "needs_permission":
        return Advice("turn on Accessibility for Heliox-OS in System Settings, Privacy & Security, Accessibility")
    if kind == "terminal":
        if name == "kitty":
            return Advice(
                "turn on kitty's remote control, or add the zsh or fish terminal check, in Settings, Live, "
                "Terminal check",
                "kitty read directly: platform-readers, run 43219012295",
            )
        return Advice("add the zsh or fish terminal check in Settings, Live, Terminal check",
                      "shell-check run 43209835182 (zsh and fish)")  # fmt: skip
    if mac:
        shown = MAC_READ_BY_DEFAULT.get(name)
        return _usually_read(app, shown) if shown else Advice("")
    if not linux or kind not in LINUX_FINDINGS:
        return Advice("")
    cells = LINUX_FINDINGS[kind]
    switch = "off" if reader_message else "on"  # the reader's note says the desktop switch is off
    # Read with no flag: with the desktop switch as it is, or with it off (turning it on cannot take an app off the bus).
    default = cells.get((switch, "off"), Finding())
    if not default.reads and switch == "on":
        default = cells.get(("off", "off"), Finding())
    if default.reads:
        if switch == "on" and kind in LINUX_LATE_UNREAD:
            # Read when it starts after the switch; one started before stays unread. Live cannot tell when the app
            # started relative to the switch, so one step covers both: restart it.
            return Advice("restart it, so it starts with accessibility for apps on", LINUX_LATE_UNREAD[kind])
        return _usually_read(app, default.shown)
    working = [(cell, found) for cell, found in cells.items() if found.reads]
    if working:
        (need_switch, need_flag), found = working[0]
        steps = []
        if need_switch == "on" and switch == "off":
            steps.append(f"turn on accessibility for apps ({TOOLKIT_SWITCH})")
        if need_flag == "on":
            steps.append(f"start it with {LINUX_FLAGS[kind]}")
        return Advice(" and ".join(steps) + ", then restart it", found.shown) if steps else Advice("", found.shown)
    tried = [found.shown for found in cells.values() if found.reads is False]
    if tried:
        return Advice("", "; ".join(tried))  # the set-ups a person can choose that were tried showed nothing
    # Not run yet (Qt): what is documented, not shown.
    if switch == "off":
        return Advice(f"turn on accessibility for apps ({TOOLKIT_SWITCH}), then restart it")
    flag = LINUX_FLAGS.get(kind)
    return Advice(f"start it with {flag} set, then it can be read") if flag else Advice("")


def sentence(app: str, found: Advice) -> str:
    label = app_label(app) or app
    if label.islower():  # "google-chrome" as the process is named: "Google Chrome" as people say it
        label = " ".join(word[:1].upper() + word[1:] for word in label.replace("-", " ").split())
    if found.fix:
        return f"Live can't see what you type in {label}: {found.fix}."
    return f"Live doesn't work in {label} yet."


def _key(app: str) -> str:
    return app.strip().casefold().removesuffix(".exe").removesuffix(".app")


class BlindSpots:
    """Counts time in front with nothing to read, per app, and says so once per app."""

    def __init__(self, path: Path | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        self._path = path
        self._clock = clock
        self.after = BLIND_AFTER_SECONDS
        self._unread: dict[str, float] = {}
        self._seen_field: set[str] = set()
        self._last: tuple[str, float] | None = None
        self.told: dict[str, str] = self._load()  # app key -> the sentence said
        self.current = ""  # the sentence for the app in front, the first time it is said; "" otherwise

    def _load(self) -> dict[str, str]:
        if self._path is None:
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self.told, indent=1), encoding="utf-8")
        except OSError:
            logger.debug("Could not keep the apps Live cannot see", exc_info=True)

    def note(self, app: str, readable: bool, *, platform: str = sys.platform, reader_state: str = "ready",
             reader_message: str = "") -> str:  # fmt: skip
        """One look at the window in front (already past the never-read rules). Returns the sentence when it is said
        for the first time, else ""."""
        key = _key(app)
        now = self._clock()
        if key in NOT_TYPED_IN:
            self._last, self.current = None, ""
            return ""
        if readable:
            self._seen_field.add(key)
            self._unread.pop(key, None)
            self._last, self.current = (key, now), ""
            if self.told.pop(key, None) is not None:  # Live sees into it now: the line goes
                self._save()
            return ""
        if self._last is not None and self._last[0] == key and now - self._last[1] <= VISIT_GAP_SECONDS:
            self._unread[key] = self._unread.get(key, 0.0) + (now - self._last[1])
        elif self._last is None or self._last[0] != key:
            self.current = ""  # another app in front: the first-time sentence is not for it
        self._last = (key, now)
        if key in self._seen_field or key in self.told or self._unread.get(key, 0.0) < self.after:
            return ""
        said = sentence(app, advice(app, platform=platform, reader_state=reader_state, reader_message=reader_message))
        self.told[key] = said
        self.current = said
        self._save()
        return said

    def notices(self) -> list[dict[str, str]]:
        return [{"app": key, "sentence": said} for key, said in sorted(self.told.items())]

    def forget(self, app: str = "") -> None:
        """Say it again (for one app, or all): asked from Settings."""
        if app:
            self.told.pop(_key(app), None)
        else:
            self.told.clear()
        self.current = ""
        self._save()


__all__ = ["ADVICE_KINDS", "BLIND_AFTER_SECONDS", "Advice", "BlindSpots", "advice", "kind_of", "sentence"]
ADVICE_KINDS = ("chromium", "electron", "qt", "gecko", "terminal")
