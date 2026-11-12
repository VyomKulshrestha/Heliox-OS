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


def advice(app: str, *, platform: str = sys.platform, reader_state: str = "ready", reader_message: str = "") -> Advice:
    """What would let Live see into ``app`` here, from what is known so far."""
    kind = kind_of(app)
    mac, linux = platform == "darwin", platform.startswith("linux")
    if mac and reader_state == "needs_permission":
        return Advice("turn on Accessibility for Heliox-OS in System Settings, Privacy & Security, Accessibility")
    if kind == "terminal":
        if app.strip().casefold() == "kitty":
            return Advice(
                "turn on kitty's remote control, or add the zsh or fish terminal check, in Settings, Live, "
                "Terminal check",
                "kitty read directly: platform-readers, run 43219012295",
            )
        return Advice("add the zsh or fish terminal check in Settings, Live, Terminal check",
                      "shell-check run 43209835182 (zsh and fish)")  # fmt: skip
    if linux and reader_message and kind in ("chromium", "electron", "qt", "gecko"):
        # The session reports toolkit accessibility off: these toolkits only join the accessibility bus with it on.
        return Advice(f"turn on accessibility for apps ({TOOLKIT_SWITCH}), then restart it")
    if linux and kind == "chromium":
        return Advice("start it with --force-renderer-accessibility, then it can be read")
    if linux and kind == "electron":
        return Advice("start it with ACCESSIBILITY_ENABLED=1 set, then it can be read")
    if linux and kind == "qt":
        return Advice("start it with QT_LINUX_ACCESSIBILITY_ALWAYS_ON=1 set, then it can be read")
    return Advice("")


def sentence(app: str, found: Advice) -> str:
    label = app_label(app) or app
    if label.islower():
        label = label[:1].upper() + label[1:]  # "slack" as the process is named: "Slack" as people say it
    if found.fix:
        return f"Live can't see what you type in {label}: {found.fix}."
    return f"Live can't see what you type in {label}, and there is nothing to turn on for it yet."


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
