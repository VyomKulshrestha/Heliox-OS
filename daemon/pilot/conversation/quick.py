"""The quick road: one harmless action, built in code, run through the normal pipeline without the planner call.

A simple spoken request ("open Notepad", "mute", "next song", "how much disk space do I have") does not need a
model to write a plan. This module turns it into a one-action ``ActionPlan`` when, and only when:

* the action type is on a fixed allow-list of actions the permission tiers never ask approval for and that are
  harmless or trivially undone (``ALLOWED``);
* its parameters pass the checks below (no arguments to a launched program except a known folder opened by the
  system's file manager; web addresses only; media keys only, never typing);
* ``PermissionChecker``'s rules say it needs no confirmation and is not irreversible.

Anything else returns None and the request goes to the full planner. The plan then runs through the same
``_handle_execute_inner`` as a planned one: the ActionValidator and protected-path rules (in the executor), the
world model's risk gate, the executor, the verifier, the Activity log, the timeline and the undo journal. The
server skips only the planner call and the model reviewers; if the permission check or the world model would ask
for approval, it drops the quick plan and plans normally instead (it never creates an approval from here).

Nothing writes, deletes, moves, sends, installs, runs a shell command or types into another app on this road.
Timers, reminders and to-dos are answered in code before this layer is asked, so they need no road here.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from pilot.actions import (
    Action,
    ActionPlan,
    ActionType,
    KeyboardParams,
    OpenApplicationParams,
    OpenUrlParams,
    PermissionTier,
    ProcessParams,
    SystemInfoParams,
    VolumeParams,
    WindowParams,
)
from pilot.conversation.router import MEDIA_PATTERNS as _MEDIA
from pilot.conversation.router import VOLUME_LEVEL as _VOLUME_LEVEL

ALLOWED = frozenset(
    {
        ActionType.OPEN_APPLICATION,
        ActionType.OPEN_URL,
        ActionType.WINDOW_FOCUS,
        ActionType.VOLUME_SET,
        ActionType.VOLUME_MUTE,
        ActionType.VOLUME_GET,
        ActionType.KEYBOARD_PRESS,  # media keys only, checked below
        ActionType.SYSTEM_INFO,
        ActionType.BATTERY_INFO,
        ActionType.DISK_USAGE,
        ActionType.MEMORY_USAGE,
        ActionType.CPU_USAGE,
        ActionType.PROCESS_LIST,
    }
)
MEDIA_KEYS = frozenset({"playpause", "nexttrack", "prevtrack", "volumeup", "volumedown", "volumemute"})
# Programs that run commands or scripts. Never launched from this road, with or without arguments.
_INTERPRETERS = frozenset(
    {
        "cmd",
        "command prompt",
        "powershell",
        "pwsh",
        "windows powershell",
        "bash",
        "sh",
        "zsh",
        "fish",
        "wsl",
        "python",
        "python3",
        "py",
        "pythonw",
        "node",
        "deno",
        "ruby",
        "perl",
        "php",
        "wscript",
        "cscript",
        "mshta",
        "rundll32",
        "regsvr32",
        "msiexec",
        "reg",
        "regedit",
        "schtasks",
        "wmic",
        "certutil",
        "bitsadmin",
        "osascript",
        "sudo",
        "su",
        "doas",
        "terminal",
        "windows terminal",
        "wt",
        "iterm",
        "konsole",
        "gnome-terminal",
        "xterm",
        "explorer",
        "finder",
        "xdg-open",
        "open",
        "start",
        "control",
        "mmc",
        "taskkill",
    }
)
KNOWN_FOLDERS = {
    "desktop": "Desktop",
    "documents": "Documents",
    "downloads": "Downloads",
    "pictures": "Pictures",
    "photos": "Pictures",
    "music": "Music",
    "videos": "Videos",
}
DEFAULT_SEARCH_URL = "https://www.google.com/search?q={query}"


def valid_search_url(url: str) -> str:
    """``conversation.search_url`` when it is an http(s) address with a "{query}" placeholder, else the default."""
    text = (url or "").strip()
    if "{query}" in text and re.match(r"^https?://[^\s/$.?#][^\s]*$", text.replace("{query}", "q"), re.IGNORECASE):
        return text
    return DEFAULT_SEARCH_URL


_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}(?:/\S*)?$")
# Words that make "open X" something other than an app: the planner works those out.
_NOT_AN_APP = re.compile(
    r"\b(?:file|files|folder|folders|document|documents|tab|tabs|window|windows|new|last|latest|recent|it|that|"
    r"this|those|them|report|invoice|email|emails|message|messages|project|repo|link|page|settings for|and|with|"
    r"in|from|on|at|to|for)\b"
)
_STATE = (
    (re.compile(r"\bbattery\b"), ActionType.BATTERY_INFO),
    (re.compile(r"\b(?:disk|space|storage|drive)\b"), ActionType.DISK_USAGE),
    (re.compile(r"\b(?:memory|ram)\b"), ActionType.MEMORY_USAGE),
    (re.compile(r"\bcpu\b"), ActionType.CPU_USAGE),
)


@dataclass(frozen=True, slots=True)
class QuickPlan:
    plan: ActionPlan
    source: str  # "router" or "model"
    summary: str  # what is being done, for the log


def _platform() -> str:
    return sys.platform


def _action(action_type: ActionType, target: str, parameters: Any) -> Action:
    return Action(action_type=action_type, target=target, parameters=parameters)


def admissible(action: Action, *, platform: str | None = None, home: Path | None = None) -> tuple[bool, str]:
    """Whether one action may take the quick road. Pure apart from the folder existing; tested on every OS."""
    platform = platform or _platform()
    if action.action_type not in ALLOWED:
        return False, f"{action.action_type.value} is not on the quick road's allow-list"
    if action.requires_root or action.destructive or action.rollback_action is not None or action.use_previous_output:
        return False, "the action asks for more than a plain step"
    if action.permission_tier > PermissionTier.USER_WRITE or action.requires_confirmation or action.is_irreversible:
        return False, "the permission tiers would ask for approval"
    params = action.parameters
    if action.action_type is ActionType.OPEN_APPLICATION:
        if not isinstance(params, OpenApplicationParams):
            return False, "wrong parameters"
        name = " ".join(params.name.casefold().split())
        if params.args:
            return _folder_opener_ok(params, platform, home)
        if (
            not name
            or name in _INTERPRETERS
            or re.search(r"[\\/:;&|<>`$%]", name)
            or name.endswith((".exe", ".bat", ".cmd", ".ps1", ".sh"))
        ):
            return False, "only an app by name, never a program that runs commands"
        return True, ""
    if action.action_type is ActionType.OPEN_URL:
        url = getattr(params, "url", "") if isinstance(params, OpenUrlParams) else ""
        if not url or not re.match(r"^https?://[^\s/$.?#][^\s]*$", url, re.IGNORECASE):
            return False, "only http and https addresses"
        return True, ""
    if action.action_type is ActionType.KEYBOARD_PRESS:
        if not isinstance(params, KeyboardParams) or params.text or params.keys or params.key not in MEDIA_KEYS:
            return False, "only media keys, never typing"
        if not 1 <= params.presses <= 10:
            return False, "too many presses"
        return True, ""
    if action.action_type is ActionType.VOLUME_SET:
        level = getattr(params, "level", None)
        if not (isinstance(level, int) and 0 <= level <= 100):
            return False, "a volume level from 0 to 100"
        return True, ""
    return True, ""


def _folder_opener(platform: str) -> str | None:
    """Windows: Explorer. Linux: xdg-open. macOS: none, since ``launch_application`` uses ``open -a`` there."""
    if platform == "win32":
        return "explorer"
    if platform.startswith("linux"):
        return "xdg-open"
    return None


def _folder_opener_ok(params: OpenApplicationParams, platform: str, home: Path | None) -> tuple[bool, str]:
    """The one launch with an argument: the system's file manager showing a known folder of the person's."""
    opener = _folder_opener(platform)
    if opener is None or params.name != opener or len(params.args) != 1:
        return False, "only a known folder, through the system's file manager"
    base = home or Path.home()
    allowed = {str(base / name) for name in KNOWN_FOLDERS.values()} | {str(base)}
    return (params.args[0] in allowed), "only a known folder, through the system's file manager"


def _folder_plan(words: list[str], platform: str, home: Path | None) -> Action | None:
    rest = [w for w in words if w not in {"my", "the", "folder", "directory", "please"}]
    if len(rest) != 1:
        return None
    key = rest[0]
    if key in {"home"}:
        path = home or Path.home()
    elif key in KNOWN_FOLDERS:
        path = (home or Path.home()) / KNOWN_FOLDERS[key]
    else:
        return None
    opener = _folder_opener(platform)
    if opener is None:
        return None  # macOS: "open -a" cannot show a folder through launch_application; the planner does it
    return _action(ActionType.OPEN_APPLICATION, str(path), OpenApplicationParams(name=opener, args=[str(path)]))


_TLDS = "com|org|net|io|dev|ai|co|in|de|uk|edu|gov|app|me|tv|fr|es|it|nl|ca|au|jp"


def _address(target: str) -> str:
    """A web address as the recogniser wrote it: "youtube.com", "youtube com", "youtube dot com"."""
    text = target.replace(" dot ", ".").strip()
    spaced = re.fullmatch(rf"([a-z0-9-]+) ({_TLDS})", text)
    if spaced:
        text = f"{spaced.group(1)}.{spaced.group(2)}"
    return text if _DOMAIN.match(text) else ""


def _app_name(target: str) -> str | None:
    words = [w for w in target.split() if w not in {"the", "app", "application", "program", "up", "please", "my"}]
    name = " ".join(words)
    if not name or len(words) > 3 or _NOT_AN_APP.search(name) or re.search(r"[\\/:]", name):
        return None
    return name


def from_words(
    normalized: str,
    verb: str,
    target: str,
    *,
    platform: str | None = None,
    home: Path | None = None,
    search_url: str = DEFAULT_SEARCH_URL,
) -> QuickPlan | None:
    """A one-action plan from the router's parse, or None when the request is not plainly one allowed action."""
    platform = platform or _platform()
    text = normalized.strip()
    action: Action | None = None
    for pattern, key in _MEDIA:
        if pattern.search(text):
            action = _action(
                ActionType.KEYBOARD_PRESS, key, KeyboardParams(key=key, presses=5 if "volume" in key else 1)
            )
            break
    sound = {"", "the sound", "the volume", "sound", "volume", "the audio", "audio", "it", "everything"}
    if action is None and verb in {"mute", "unmute"} and target.strip() in sound:
        # "mute", "mute the sound", and the same said politely ("would you mind muting the sound?")
        action = _action(ActionType.VOLUME_MUTE, "", VolumeParams(mute=verb == "mute"))
    if action is None and (level := _VOLUME_LEVEL.match(text)):
        action = _action(ActionType.VOLUME_SET, "", VolumeParams(level=min(100, int(level.group(1)))))
    if action is None and verb == "check":
        for pattern, action_type in _STATE:
            if pattern.search(text):
                params = ProcessParams() if action_type is ActionType.PROCESS_LIST else SystemInfoParams()
                action = _action(action_type, "", params)
                break
    if action is None and verb in {"open", "launch"} and target:
        words = target.split()
        action = _folder_plan(words, platform, home)
        address = _address(target)
        if action is None and address:
            action = _action(
                ActionType.OPEN_URL, address, OpenUrlParams(url=address if "://" in address else f"https://{address}")
            )
        if action is None and (name := _app_name(target)):
            action = _action(ActionType.OPEN_APPLICATION, name, OpenApplicationParams(name=name))
    if action is None and verb in {"search", "look"}:
        query = re.sub(r"^(?:up |for |the web for |online for |google for |the internet for )", "", target).strip()
        if query and not re.search(
            r"\b(?:file|files|folder|computer|drive|downloads|documents|desktop|email|emails|disk)\b", query
        ):
            url = search_url.replace("{query}", quote_plus(query))
            action = _action(ActionType.OPEN_URL, url, OpenUrlParams(url=url))
    if action is None and verb in {"switch", "focus"}:
        name = re.sub(r"^(?:to |over to |back to )", "", target).strip()
        if name and (app := _app_name(name)):
            action = _action(ActionType.WINDOW_FOCUS, app, WindowParams(title=app))
    if action is None:
        return None
    ok, _reason = admissible(action, platform=platform, home=home)
    if not ok:
        return None
    return QuickPlan(
        ActionPlan(actions=[action], explanation="", raw_input=normalized), "router", action.action_type.value
    )


def from_model(proposal: Any, *, words: str, platform: str | None = None, home: Path | None = None) -> QuickPlan | None:
    """One action the fast model proposed in the planner's own schema, used only if it is on the allow-list."""
    if not isinstance(proposal, dict):
        return None
    try:
        action_type = ActionType(str(proposal.get("action_type") or "").strip().casefold())
    except ValueError:
        return None
    if action_type not in ALLOWED:
        return None
    raw = proposal.get("parameters") or {}
    if not isinstance(raw, dict):
        return None
    try:
        from pilot.agents.planner import Planner

        parameters = Planner._parse_parameters(action_type, dict(raw))
        action = Action(action_type=action_type, target=str(proposal.get("target") or "")[:300], parameters=parameters)
    except Exception:
        return None
    if action_type is ActionType.OPEN_APPLICATION and isinstance(action.parameters, OpenApplicationParams):
        if action.parameters.args:
            return None  # a model never passes arguments on this road, not even a folder
    ok, _reason = admissible(action, platform=platform, home=home)
    if not ok:
        return None
    return QuickPlan(ActionPlan(actions=[action], explanation="", raw_input=words), "model", action_type.value)


def speakable_state(result: str) -> str:
    """Battery, disk and memory readings as a sentence; "" when the text is not one of them."""
    text = result or ""
    charge = re.search(r"Charge:\s*(\d+)%", text)
    if charge:
        status = re.search(r"Status:\s*([A-Za-z ]+?)(?=\s*(?:Time remaining|$)|\n|\s{2})", text)
        tail = f", {status.group(1).strip().lower()}" if status and status.group(1).strip() not in ("", "N/A") else ""
        return f"The battery is at {charge.group(1)} percent{tail}."
    drives = re.findall(r"([A-Za-z]:\\|/[^\s(]*)[^\n]*?([\d.]+) GB total, ([\d.]+) GB used", text)
    if drives:
        parts = []
        for name, total, used in drives[:3]:
            free = max(0.0, float(total) - float(used))
            label = f"drive {name[0].upper()}" if name[:1].isalpha() else (name or "the main disk")
            parts.append(f"{free:.0f} gigabytes free on {label}")
        return "You have " + ", and ".join(parts) + "."
    memory = re.search(r"Total:\s*([\d.]+) GB\s+Used:\s*([\d.]+) GB \(([\d.]+)%\)", text)
    if memory:
        return f"{memory.group(2)} of {memory.group(1)} gigabytes of memory are in use, {float(memory.group(3)):.0f} percent."
    cpu = re.search(r"(?:CPU|Usage)[^\n]*?([\d.]+)%", text)
    if cpu and "=== CPU" in text:
        return f"The processor is at {float(cpu.group(1)):.0f} percent."
    return ""


def _title(name: str) -> str:
    from pilot.conversation.handover import _title_app

    return _title_app(name)


def _original_words(words: str, lowered: str) -> str:
    """``lowered`` as the person's words had it (case kept), when it can be found there."""
    index = words.casefold().find(lowered.casefold())
    return words[index : index + len(lowered)] if index >= 0 else lowered


def quick_acknowledgement(plan: QuickPlan, words: str) -> str:
    """What the person hears at once, from the action itself: "Opening Notepad.", "Searching the web for ..."."""
    action = plan.plan.actions[0]
    params = action.parameters
    kind = action.action_type
    if kind is ActionType.OPEN_APPLICATION:
        if getattr(params, "args", None):
            return f"Opening your {Path(params.args[0]).name or 'home'} folder."
        return f"Opening {_title(params.name)}."
    if kind is ActionType.OPEN_URL:
        url = getattr(params, "url", "")
        query = re.search(r"[?&]q=([^&]+)", url)
        if query:
            from urllib.parse import unquote_plus

            return f"Searching the web for {_original_words(words, unquote_plus(query.group(1)))}."
        host = re.sub(r"^https?://(?:www\.)?", "", url).split("/")[0]
        return f"Opening {host}."
    if kind is ActionType.WINDOW_FOCUS:
        return f"Switching to {_title(getattr(params, 'title', '') or 'it')}."
    if kind is ActionType.VOLUME_MUTE:
        return "Muting." if getattr(params, "mute", True) else "Unmuting."
    if kind is ActionType.VOLUME_SET:
        return f"Setting the volume to {params.level}."
    if kind is ActionType.KEYBOARD_PRESS:
        from pilot.conversation.handover import MEDIA_ACKS

        return MEDIA_ACKS.get(getattr(params, "key", ""), "Okay.")
    return "I'll check."


def quick_result_sentence(plan: QuickPlan, result: str, status: str) -> str:
    """The result in one sentence; a failure says what failed."""
    action = plan.plan.actions[0]
    if status != "success":
        detail = next((line.strip() for line in (result or "").splitlines() if line.strip()), "")
        detail = re.sub(r"^(?:error|failed)[:\s]*", "", detail, flags=re.IGNORECASE)[:160]
        return f"That didn't work: {detail}" if detail else "That didn't work."
    state = speakable_state(result)
    if state:
        return state
    if action.action_type is ActionType.OPEN_APPLICATION and not getattr(action.parameters, "args", None):
        return f"{_title(action.parameters.name)} is open."
    if action.action_type is ActionType.OPEN_URL:
        return "It's open in your browser."
    return "Done."
