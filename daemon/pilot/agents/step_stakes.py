"""What is at stake in each kind of step, for deciding whether a plan waits for the companion's review.

Every action type is in exactly one class, with the reason (tests/test_review_timing.py walks the catalog, so a new
action type without an entry fails):

- ``read``: it only reads (``READ_ONLY_ACTIONS``, and an HTTP request only while its method reads).
- ``nothing_at_stake``: it changes only the device's or the view's own state (a level, which window or tab is in
  front, where a page or the pointer is) and doing the opposite puts it back. No data, no other person and no other
  program's content is involved, so it runs beside its review whatever the undo journal keeps.
- ``journal_keeps``: a write, copy or move of one file that runs beside its review only while the undo journal will
  keep the earlier state (pilot.agents.review_timing checks that before the plan runs, and the executor stops the
  step if the journal then cannot keep its copy).
- ``waits``: anything else. Where it is unclear whether a step can lose something, it waits.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pilot.actions import READ_ONLY_ACTIONS, ActionType


class Stake(StrEnum):
    READ = "read"
    NOTHING_AT_STAKE = "nothing_at_stake"
    JOURNAL_KEEPS = "journal_keeps"
    WAITS = "waits"


_LEVEL = "a level of this device; setting it again puts it back"
_VIEW = "only which window or tab is in front, or where the view is; doing the opposite puts it back"
_NAVIGATION = "a navigation (a page loaded by address, not a form sent); going back puts it back"
_INPUT = "goes into whatever program has focus, which can change or send that program's content"
_SENDS = "sends something to another person or service, which cannot be called back"
_SYSTEM = "changes the system or a setting that lasts past a restart"
_RUNS = "runs code or a command whose effects are not known before it runs"
_CLOSES = "closes something that may hold unsaved work"
_DELETES = "deletes or removes something"
_POWER = "locks, sleeps, restarts or turns off the computer"
_SCHEDULE = "changes what Heliox does later on its own"
_GIT = "changes a repository's history or what others can see"

_EXPLICIT: dict[ActionType, tuple[Stake, str]] = {
    # Reads whose permission tier depends on their parameters (checked again when the plan runs).
    ActionType.API_REQUEST: (Stake.READ, "an HTTP request reads only while its method is GET or HEAD"),
    ActionType.API_GITHUB: (Stake.READ, "a GitHub API call reads only while its method is GET or HEAD"),
    # Nothing at stake.
    ActionType.VOLUME_SET: (Stake.NOTHING_AT_STAKE, _LEVEL),
    ActionType.VOLUME_MUTE: (Stake.NOTHING_AT_STAKE, "mute or unmute; doing it again puts it back"),
    ActionType.BRIGHTNESS_SET: (Stake.NOTHING_AT_STAKE, _LEVEL),
    ActionType.WINDOW_FOCUS: (Stake.NOTHING_AT_STAKE, _VIEW),
    ActionType.WINDOW_MINIMIZE: (Stake.NOTHING_AT_STAKE, _VIEW),
    ActionType.WINDOW_MAXIMIZE: (Stake.NOTHING_AT_STAKE, _VIEW),
    ActionType.MOUSE_MOVE: (Stake.NOTHING_AT_STAKE, "only where the pointer is; nothing is pressed"),
    ActionType.BROWSER_NAVIGATE: (Stake.NOTHING_AT_STAKE, _NAVIGATION),
    ActionType.BROWSER_BACK: (Stake.NOTHING_AT_STAKE, _NAVIGATION),
    ActionType.BROWSER_FORWARD: (Stake.NOTHING_AT_STAKE, _NAVIGATION),
    ActionType.BROWSER_SWITCH_TAB: (Stake.NOTHING_AT_STAKE, _VIEW),
    ActionType.BROWSER_NEW_TAB: (Stake.NOTHING_AT_STAKE, "opens an empty tab beside the others"),
    ActionType.BROWSER_SCROLL: (Stake.NOTHING_AT_STAKE, _VIEW),
    ActionType.BROWSER_HOVER: (Stake.NOTHING_AT_STAKE, "only where the pointer rests on the page; nothing is pressed"),
    # Kept by the undo journal (and checked against it).
    ActionType.FILE_WRITE: (Stake.JOURNAL_KEEPS, "the journal keeps the earlier file or deletes a new one"),
    ActionType.FILE_COPY: (Stake.JOURNAL_KEEPS, "the journal keeps the file it lands on or deletes the new copy"),
    ActionType.FILE_MOVE: (Stake.JOURNAL_KEEPS, "the journal keeps the file it lands on and moves it back"),
    # Waits.
    ActionType.MOUSE_SCROLL: (Stake.WAITS, "a wheel over a field or slider in a program changes its value"),
    ActionType.MOUSE_CLICK: (Stake.WAITS, _INPUT),
    ActionType.MOUSE_DOUBLE_CLICK: (Stake.WAITS, _INPUT),
    ActionType.MOUSE_RIGHT_CLICK: (Stake.WAITS, _INPUT),
    ActionType.MOUSE_DRAG: (Stake.WAITS, _INPUT),
    ActionType.KEYBOARD_TYPE: (Stake.WAITS, _INPUT),
    ActionType.KEYBOARD_PRESS: (Stake.WAITS, _INPUT),
    ActionType.KEYBOARD_HOTKEY: (Stake.WAITS, _INPUT),
    ActionType.KEYBOARD_HOLD: (Stake.WAITS, _INPUT),
    ActionType.BROWSER_CLICK: (Stake.WAITS, "a click on a page can send a form or buy something"),
    ActionType.BROWSER_CLICK_TEXT: (Stake.WAITS, "a click on a page can send a form or buy something"),
    ActionType.BROWSER_TYPE: (Stake.WAITS, "types into a page, which can change what it sends"),
    ActionType.BROWSER_SELECT: (Stake.WAITS, "changes a choice on a page, which can change what it sends"),
    ActionType.BROWSER_FILL_FORM: (Stake.WAITS, "fills and may send a form"),
    ActionType.BROWSER_EXECUTE_JS: (Stake.WAITS, _RUNS),
    ActionType.BROWSER_REFRESH: (Stake.WAITS, "a reload can send a form again or lose what was typed"),
    ActionType.BROWSER_CLOSE_TAB: (Stake.WAITS, _CLOSES),
    ActionType.BROWSER_CLOSE: (Stake.WAITS, _CLOSES),
    ActionType.WINDOW_CLOSE: (Stake.WAITS, _CLOSES),
    ActionType.PROCESS_KILL: (Stake.WAITS, _CLOSES),
    ActionType.CLIPBOARD_WRITE: (
        Stake.WAITS,
        "the earlier clipboard content can be lost (an image, or after a restart)",
    ),
    ActionType.FILE_DELETE: (Stake.WAITS, _DELETES),
    ActionType.FILE_PERMISSIONS: (Stake.WAITS, _SYSTEM),
    ActionType.DOWNLOAD_FILE: (Stake.WAITS, "fetches a file from outside and writes it here"),
    ActionType.GIT_RESOLVE: (Stake.WAITS, "rewrites a file in a conflict; which side wins is the person's choice"),
    ActionType.GIT_BRANCH: (Stake.WAITS, _GIT),
    ActionType.GIT_STAGE: (Stake.WAITS, _GIT),
    ActionType.GIT_COMMIT: (Stake.WAITS, _GIT),
    ActionType.GIT_PUSH: (Stake.WAITS, _SENDS),
    ActionType.PACKAGE_INSTALL: (Stake.WAITS, _SYSTEM),
    ActionType.PACKAGE_REMOVE: (Stake.WAITS, _DELETES),
    ActionType.PACKAGE_UPDATE: (Stake.WAITS, _SYSTEM),
    ActionType.SERVICE_START: (Stake.WAITS, _SYSTEM),
    ActionType.SERVICE_STOP: (Stake.WAITS, _SYSTEM),
    ActionType.SERVICE_RESTART: (Stake.WAITS, _SYSTEM),
    ActionType.SERVICE_ENABLE: (Stake.WAITS, _SYSTEM),
    ActionType.SERVICE_DISABLE: (Stake.WAITS, _SYSTEM),
    ActionType.GNOME_SETTING_WRITE: (Stake.WAITS, _SYSTEM),
    ActionType.REGISTRY_WRITE: (Stake.WAITS, _SYSTEM),
    ActionType.ENV_SET: (Stake.WAITS, _SYSTEM),
    ActionType.WIFI_CONNECT: (Stake.WAITS, _SYSTEM),
    ActionType.WIFI_DISCONNECT: (Stake.WAITS, _SYSTEM),
    ActionType.DISK_MOUNT: (Stake.WAITS, _SYSTEM),
    ActionType.DISK_UNMOUNT: (Stake.WAITS, _SYSTEM),
    ActionType.DBUS_CALL: (Stake.WAITS, _RUNS),
    ActionType.SHELL_COMMAND: (Stake.WAITS, _RUNS),
    ActionType.SHELL_SCRIPT: (Stake.WAITS, _RUNS),
    ActionType.PTY_EXEC: (Stake.WAITS, _RUNS),
    ActionType.CODE_EXECUTE: (Stake.WAITS, _RUNS),
    ActionType.CODE_GENERATE_AND_RUN: (Stake.WAITS, _RUNS),
    ActionType.WASM_CALL: (Stake.WAITS, _RUNS),
    ActionType.PLUGIN_CALL: (Stake.WAITS, _RUNS),
    ActionType.SKILL_RUN: (Stake.WAITS, _RUNS),
    ActionType.MCP_CALL_TOOL: (Stake.WAITS, "a tool on another server can send, post or delete"),
    ActionType.POWER_SHUTDOWN: (Stake.WAITS, _POWER),
    ActionType.POWER_RESTART: (Stake.WAITS, _POWER),
    ActionType.POWER_SLEEP: (Stake.WAITS, _POWER),
    ActionType.POWER_LOCK: (Stake.WAITS, _POWER),
    ActionType.POWER_LOGOUT: (Stake.WAITS, _POWER),
    ActionType.SCHEDULE_CREATE: (Stake.WAITS, _SCHEDULE),
    ActionType.SCHEDULE_DELETE: (Stake.WAITS, _SCHEDULE),
    ActionType.TRIGGER_CREATE: (Stake.WAITS, _SCHEDULE),
    ActionType.TRIGGER_DELETE: (Stake.WAITS, _SCHEDULE),
    ActionType.TRIGGER_START: (Stake.WAITS, _SCHEDULE),
    ActionType.TRIGGER_STOP: (Stake.WAITS, _SCHEDULE),
    ActionType.CALENDAR_SYNC: (Stake.WAITS, _SENDS),
    ActionType.CALENDAR_CREATE_EVENT: (Stake.WAITS, _SENDS),
    ActionType.CALENDAR_DELETE_EVENT: (Stake.WAITS, _DELETES),
    ActionType.API_SEND_EMAIL: (Stake.WAITS, _SENDS),
    ActionType.API_WEBHOOK: (Stake.WAITS, _SENDS),
    ActionType.API_SLACK: (Stake.WAITS, _SENDS),
    ActionType.API_DISCORD: (Stake.WAITS, _SENDS),
    ActionType.EMAIL_REPLY: (Stake.WAITS, _SENDS),
    ActionType.SSH_COMMAND: (Stake.WAITS, _RUNS),
    ActionType.SSH_SCRIPT: (Stake.WAITS, _RUNS),
}

STAKES: dict[ActionType, tuple[Stake, str]] = {
    **{action_type: (Stake.READ, "only reads") for action_type in READ_ONLY_ACTIONS},
    **_EXPLICIT,
}


def stake_of(action_type: ActionType) -> Stake:
    """The class of an action type; an action type without an entry waits."""
    return STAKES.get(ActionType(action_type), (Stake.WAITS, ""))[0]


# Keys that only control what is playing or how loud: play or pause, next, previous, stop, volume up, down and mute
# (pyautogui's names). A key press made only of these goes to the media player, not into the program in front.
MEDIA_KEYS = frozenset({"playpause", "nexttrack", "prevtrack", "stop", "volumeup", "volumedown", "volumemute"})


def media_keys_only(action: Any) -> bool:
    """A key press or hotkey whose keys are all media or volume keys."""
    if action.action_type not in (ActionType.KEYBOARD_PRESS, ActionType.KEYBOARD_HOTKEY):
        return False
    params = action.parameters
    keys = [str(key) for key in (getattr(params, "keys", None) or [])]
    if action.action_type is ActionType.KEYBOARD_PRESS:
        keys = [str(getattr(params, "key", "") or "")]
    keys = [key.strip().lower() for key in keys if key and key.strip()]
    return bool(keys) and all(key in MEDIA_KEYS for key in keys)


def step_stake(action: Any) -> Stake:
    """The class of one step: its action type's, except that a key press made only of media or volume keys has
    nothing at stake (every other key press waits, as its type does)."""
    if media_keys_only(action):
        return Stake.NOTHING_AT_STAKE
    return stake_of(action.action_type)


__all__ = ["MEDIA_KEYS", "STAKES", "Stake", "media_keys_only", "stake_of", "step_stake"]
