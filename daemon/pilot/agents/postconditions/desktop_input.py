"""Post-conditions of desktop input and window actions: focus, typing, keys, the pointer, window state, starting an app.

Desktop state is fleeting and the verifier runs after the whole plan, when later steps (or the person) have moved
the focus and the pointer again. So the readings that matter are made right after each step, inside the executor's
step, by ``pilot.system.desktop_observe`` through a different call than the one that acted (GetForegroundWindow
after AppActivate, xprop after wmctrl, GetCursorPos after pyautogui's SendInput, and so on); a wrong window or a
pointer in the wrong place fails that step at once, so the plan's next keystrokes do not run. Here each verifier
reports that reading (``recalled``), and reads afresh only where the state lasts (held keys, window state, the
window list).

What cannot be observed is said, never passed: a field that does not expose its text, a password field (never
read), Wayland's rules (no program may ask for the focused window, the pointer or the keys; AT-SPI answers for the
focused window when accessibility is on), macOS without the Accessibility permission, a click whose effect on its
target has no state Heliox knows to expect.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from pilot.actions import ActionResult, ActionType
from pilot.agents.postconditions import Outcome, mismatch, register, unobserved, verified

APP_WINDOW_WAIT_SECONDS = 10.0
APP_POLL_SECONDS = 0.5
MEDIA_KEYS = frozenset({"playpause", "nexttrack", "prevtrack"})

NO_READING = "No reading was made right after the step"


async def _poll_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


_sleep = _poll_sleep


def _from_record(result: ActionResult, *, verified_suffix: str = "") -> Outcome | None:
    """The outcome of the reading made right after the step, or None when there is none."""
    from pilot.system.desktop_observe import recalled

    record = recalled(result.action)
    if record is None:
        return None
    if record.outcome == "match":
        return verified(record.detail + verified_suffix)
    if record.outcome == "mismatch":
        return mismatch(record.detail)
    return unobserved(f"Could not verify: {record.detail}")


# ── window_focus ──


@register(ActionType.WINDOW_FOCUS, method="focused_window_postcondition", timeout=10.0)
async def focused_window(result: ActionResult) -> Outcome:
    """The window asked for had the keyboard focus right after the step, read through a different call than the one
    that focused it. Without that reading (the step ran outside the executor), the window in front is read now."""
    from pilot.system import desktop_observe as observe

    recorded = _from_record(result, verified_suffix=", read right after the step")
    if recorded is not None:
        return recorded
    params = result.action.parameters
    window_id = getattr(params, "window_id", None)
    title = getattr(params, "title", None)
    process_name = getattr(params, "process_name", None)
    try:
        found = await observe.foreground()
        pids = await observe._threaded(lambda: observe.title_pids_now(title or "")) if title else frozenset()
    except observe.Unobservable as reason:
        return unobserved(f"Could not verify the focus: {reason}")
    verdict = observe.focus_matches(found, window_id=window_id, title=title, process_name=process_name, title_pids=pids)
    if verdict:
        return verified(f"{found.describe()} is in front ({found.source})")
    return unobserved(
        f"{NO_READING}; {found.describe()} is in front now, and later steps or the person may have moved the focus"
    )


# ── keyboard_type ──


@register(ActionType.KEYBOARD_TYPE, method="conditional_typed_text_postcondition", timeout=10.0)
async def typed_text(result: ActionResult) -> Outcome:
    """The window that had the focus kept it while typing, and the focused field holds the typed text, where the
    field shows its text through UI Automation, the Accessibility API or AT-SPI. Typed text is never logged or put
    in this result: it is compared in memory right after the step, and only lengths are said."""
    recorded = _from_record(result)
    if recorded is not None:
        return recorded
    length = len(getattr(result.action.parameters, "text", "") or "")
    return unobserved(f"{NO_READING}, so where the {length} typed characters went cannot be said")


# ── keyboard_press, keyboard_hotkey, keyboard_hold ──


def _keys_of(result: ActionResult) -> list[str]:
    params = result.action.parameters
    if result.action.action_type == ActionType.KEYBOARD_HOTKEY:
        return [str(key) for key in getattr(params, "keys", []) or []]
    key = str(getattr(params, "key", "") or "")
    return [key] if key else []


@register(
    ActionType.KEYBOARD_PRESS,
    ActionType.KEYBOARD_HOTKEY,
    ActionType.KEYBOARD_HOLD,
    method="keys_released_postcondition",
    timeout=10.0,
)
async def keys_released(result: ActionResult) -> Outcome:
    """No key of the step is still held down. A stuck modifier (the real danger of keyboard_hold) changes every key
    after it. Whether the app in front acted on the keys has no state Heliox can read, and the result says so."""
    from pilot.system import desktop_observe as observe

    keys = _keys_of(result)
    tail = "; whether the app in front acted on them cannot be read"
    recorded = _from_record(result, verified_suffix=tail)
    if recorded is not None:
        return recorded
    if result.action.action_type == ActionType.KEYBOARD_PRESS and keys and keys[0] in MEDIA_KEYS:
        return unobserved("The media player was asked through MPRIS or a media key; no key state shows what it did")
    if not keys:
        return unobserved("The step named no key to read")
    try:
        held, unknown = await observe.keys_down(keys)
    except observe.Unobservable as reason:
        return unobserved(f"Could not verify the keys were released: {reason}")
    if held:
        return mismatch(f"{'+'.join(sorted(held))} is still held down")
    if len(unknown) == len(keys):
        return unobserved(f"The state of {'+'.join(unknown)} cannot be read here")
    return verified(f"{'+'.join(k for k in keys if k not in unknown)} released{tail}")


# ── the mouse ──

_POINTER_TAILS = {
    ActionType.MOUSE_CLICK: "; what the target did with the click cannot be read",
    ActionType.MOUSE_DOUBLE_CLICK: "; what the target did with the double click cannot be read",
    ActionType.MOUSE_RIGHT_CLICK: "; whether a menu opened cannot be read",
    ActionType.MOUSE_DRAG: "; whether the dragged item moved cannot be read",
    ActionType.MOUSE_SCROLL: "; how far the content moved cannot be read",
    ActionType.MOUSE_MOVE: "",
}


@register(
    ActionType.MOUSE_CLICK,
    ActionType.MOUSE_DOUBLE_CLICK,
    ActionType.MOUSE_RIGHT_CLICK,
    ActionType.MOUSE_DRAG,
    ActionType.MOUSE_SCROLL,
    method="conditional_pointer_postcondition",
    timeout=10.0,
)
async def pointer_where_sent(result: ActionResult) -> Outcome:
    """Right after the step the pointer was where the step sent it (the end point for a drag) and the button it
    pressed was up again. That is what a click leaves behind that can be read; what the target did with it cannot."""
    recorded = _from_record(result, verified_suffix=_POINTER_TAILS[result.action.action_type])
    if recorded is not None:
        return recorded
    output = str(result.output or "")
    if output.startswith("Pressed "):
        return unobserved(
            "The control was pressed through the accessibility interface and the pointer did not move; its state "
            "after a press is not something Heliox knows to expect"
        )
    if "through xdotool" in output:
        return unobserved("The click went through xdotool, after which the pointer was not read")
    params = result.action.parameters
    if result.action.action_type == ActionType.MOUSE_SCROLL and not (
        getattr(params, "x", 0) and getattr(params, "y", 0)
    ):
        return unobserved("The scroll went to wherever the pointer was; how far the content moved cannot be read")
    return unobserved(NO_READING)


@register(ActionType.MOUSE_MOVE, method="pointer_position_postcondition", timeout=10.0)
async def pointer_moved(result: ActionResult) -> Outcome:
    """The pointer was where it was sent, read right after the step (GetCursorPos, Quartz, X11)."""
    recorded = _from_record(result)
    if recorded is not None:
        return recorded
    if getattr(result.action.parameters, "relative", False):
        return unobserved(f"{NO_READING}; a relative move has no fixed point to compare with later")
    return unobserved(f"{NO_READING}; later steps or the person may have moved the pointer since")


# ── window_minimize, window_maximize ──


def _window_target(result: ActionResult) -> str:
    params = result.action.parameters
    return str(getattr(params, "process_name", None) or getattr(params, "title", None) or "")


@register(ActionType.WINDOW_MINIMIZE, ActionType.WINDOW_MAXIMIZE, method="window_state_postcondition", timeout=10.0)
async def window_state(result: ActionResult) -> Outcome:
    """The window's state read back: minimised (IsIconic, AXMinimized, _NET_WM_STATE_HIDDEN) or maximised (IsZoomed,
    the window filling the screen's visible area, both _NET_WM_STATE_MAXIMIZED atoms)."""
    from pilot.system import desktop_observe as observe

    minimise = result.action.action_type == ActionType.WINDOW_MINIMIZE
    verb = "minimised" if minimise else "maximised"
    if result.skip_reason == "already_absent":
        return unobserved(
            str(result.notice or result.output or f"Nothing was open under that name, so nothing was {verb}")
        )
    target = _window_target(result)
    if not target:
        raise ValueError(f"The {verb} window has no title or process name to observe")
    try:
        states = await observe.window_states(target)
    except observe.Unobservable as reason:
        return unobserved(f"Could not verify the window state: {reason}")
    if not states:
        return mismatch(f"Expected a {verb} window matching {target!r}; no window matching it is open")
    if minimise:
        hit = any(minimised for minimised, _maximised in states)
    else:
        hit = any(maximised and not minimised for minimised, maximised in states)
    if hit:
        return verified(f"A window matching {target!r} is {verb}")
    return mismatch(f"Expected a window matching {target!r} to be {verb}; {len(states)} found, none {verb}")


# ── open_application ──


def app_candidates(name: str, output: str) -> list[str]:
    """What "that application" is called: the name asked for, the name the launcher resolved (Start-menu label,
    bundle, desktop entry) and a path's file name. Pure."""
    found = [name.strip()]
    launched = re.match(r"Launched (.+?) \((?:resolved via|file path)", output or "")
    if launched:
        found.append(launched.group(1).strip())
    bundle = re.search(r"resolved via (/[^)]*?\.app)\)", output or "")
    if bundle:
        found.append(Path(bundle.group(1)).stem)
    if "/" in name or "\\" in name:
        found.append(Path(name.replace("\\", "/")).stem)
    return [candidate for candidate in dict.fromkeys(found) if candidate]


def _tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[^0-9a-z]+", text.casefold()) if len(token) >= 3]


def process_matches(process: str, candidates: list[str]) -> bool:
    """A process belongs to the app when its name is the app's name or one of its words, or starts with one of its
    words of four letters or more ("CalculatorApp" for "Calculator", "code" for "Visual Studio Code"). Pure."""
    from pilot.system.desktop_observe import bare_name

    bare = bare_name(process).replace(" ", "")
    if not bare:
        return False
    for candidate in candidates:
        whole = bare_name(candidate).replace(" ", "")
        if bare == whole:
            return True
        for token in _tokens(candidate):
            if bare == token or (len(token) >= 4 and bare.startswith(token)):
                return True
    return False


def window_matches(row: Any, candidates: list[str]) -> bool:
    title = (row.title or "").casefold()
    return process_matches(row.name, candidates) or any(
        candidate.casefold() in title for candidate in candidates if len(candidate) >= 3
    )


@register(ActionType.OPEN_APPLICATION, method="conditional_application_window_postcondition", timeout=20.0)
async def application_window(result: ActionResult) -> Outcome:
    """A window of the application is on screen within a bounded time, read from the window list (EnumWindows,
    Quartz, wmctrl), not from the launcher's exit code. An app that shows only a tray icon has a process and no
    window: that is "could not verify", not a failure."""
    from pilot.system import desktop_observe as observe

    name = str(getattr(result.action.parameters, "name", "") or "")
    candidates = app_candidates(name, str(result.output or ""))
    if not candidates:
        raise ValueError("open_application has no application name to observe")
    waited = 0.0
    listing_reason = ""
    while True:
        try:
            rows = await observe.window_rows()
        except observe.Unobservable as reason:
            listing_reason = str(reason)
            break
        found = next((row for row in rows if window_matches(row, candidates)), None)
        if found is not None:
            label = f'"{found.title}"' if found.title else "a window"
            return verified(f"{label} of {found.name or candidates[0]} is open (the window list)")
        if waited >= APP_WINDOW_WAIT_SECONDS:
            break
        await _sleep(APP_POLL_SECONDS)
        waited += APP_POLL_SECONDS
    try:
        processes = await observe.running_process_names()
    except observe.Unobservable as reason:
        return unobserved(f"Could not verify that {candidates[0]} opened: {listing_reason or reason}")
    running = any(process_matches(process, candidates) for process in processes)
    if listing_reason:
        if running:
            return unobserved(f"{candidates[0]} is running; its windows cannot be listed here: {listing_reason}")
        return mismatch(f"Expected {candidates[0]} to be running; no process of it was found")
    if running:
        return unobserved(
            f"{candidates[0]} is running but shows no window after {APP_WINDOW_WAIT_SECONDS:g} s: an app that "
            "lives in the tray, or one still starting"
        )
    return mismatch(
        f"Expected a window of {candidates[0]}; no window or process of it appeared within "
        f"{APP_WINDOW_WAIT_SECONDS:g} s"
    )
