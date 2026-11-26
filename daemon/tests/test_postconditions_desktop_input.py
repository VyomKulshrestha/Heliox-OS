"""Post-conditions of desktop input and window actions (postconditions/desktop_input.py, system/desktop_observe.py).

Every platform interface is a fake: no window is focused, no key is pressed, the pointer does not move and no
application starts. pyautogui is a stand-in module named "pyautogui" (so the input functions take it for real input
and read the result back), and every reading goes through faked ``desktop_observe`` readers.
"""

from __future__ import annotations

import re
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import (
    RECONCILE_NOT_RETRY_ACTIONS,
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    KeyboardParams,
    MouseParams,
    OpenApplicationParams,
    WindowParams,
)
from pilot.agents.postconditions import desktop_input
from pilot.agents.verifier import POSTCONDITION_VERIFIERS, Verifier
from pilot.system import desktop_observe as observe
from pilot.system import input_control, window_mgr
from pilot.system.desktop_observe import DesktopStepMismatch, FieldReading, Foreground, Unobservable
from pilot.system.platform_detect import Platform

NOTEPAD = Foreground(("notepad.exe",), "notes.txt - Notepad", 42, 1001, "GetForegroundWindow")
BROWSER = Foreground(("msedge.exe",), "Inbox - Mail", 77, 2002, "GetForegroundWindow")
SECRET = "correct horse battery staple"
REAL_FOCUSED_FIELD = observe.focused_field


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch):
    observe.forget_all()
    # No reading may reach the real desktop: every reader is replaced, and a test that forgets one fails loudly.
    for name in (
        "read_foreground_now",
        "read_pointer_now",
        "read_buttons_down_now",
        "read_keys_down_now",
        "window_states_now",
        "window_rows_now",
        "read_clipboard_now",
    ):
        monkeypatch.setattr(observe, name, MagicMock(side_effect=AssertionError(f"real {name} reached")))
    monkeypatch.setattr(observe, "title_pids_now", lambda _title: frozenset())
    monkeypatch.setattr(observe, "focused_field", AsyncMock(side_effect=AssertionError("real field read reached")))
    monkeypatch.setattr(observe, "_sleep", AsyncMock())
    monkeypatch.setattr(desktop_input, "_sleep", AsyncMock())
    yield
    observe.forget_all()


def fake_pyautogui() -> types.ModuleType:
    module = types.ModuleType("pyautogui")
    for name in ("click", "moveTo", "moveRel", "drag", "scroll", "hscroll", "press", "hotkey", "typewrite", "write"):
        setattr(module, name, MagicMock())
    module.hold = MagicMock()  # a MagicMock is a context manager
    return module


@pytest.fixture
def pag(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = fake_pyautogui()
    monkeypatch.setattr(input_control, "_ensure_pyautogui", lambda: module)
    return module


async def verify(action: Action, *, success: bool = True, output: str = "", **extra):
    result = ActionResult(action=action, success=success, output=output, error=None if success else "boom", **extra)
    return await Verifier(MagicMock()).verify(ActionPlan(actions=[action]), [result])


def label(verification) -> str:
    return verification.details[0].split(": ", 1)[1]


async def said(action: Action, *, output: str = "", **extra) -> str:
    """The verifier's own sentence (a VERIFIED result shows only its label)."""
    from pilot.agents.postconditions import registrations

    result = ActionResult(action=action, success=True, output=output, **extra)
    return (await registrations()[action.action_type].check(result)).detail


def reads(monkeypatch: pytest.MonkeyPatch, name: str, *values):
    fake = MagicMock(side_effect=list(values)) if len(values) > 1 else MagicMock(return_value=values[0])
    monkeypatch.setattr(observe, name, fake)
    return fake


def raising(monkeypatch: pytest.MonkeyPatch, name: str, reason: str):
    monkeypatch.setattr(observe, name, MagicMock(side_effect=Unobservable(reason)))


def test_every_desktop_action_is_registered() -> None:
    assert POSTCONDITION_VERIFIERS[ActionType.WINDOW_FOCUS] == "focused_window_postcondition"
    assert POSTCONDITION_VERIFIERS[ActionType.KEYBOARD_TYPE] == "conditional_typed_text_postcondition"
    for action in (ActionType.KEYBOARD_PRESS, ActionType.KEYBOARD_HOTKEY, ActionType.KEYBOARD_HOLD):
        assert POSTCONDITION_VERIFIERS[action] == "keys_released_postcondition"
    for action in (
        ActionType.MOUSE_CLICK,
        ActionType.MOUSE_DOUBLE_CLICK,
        ActionType.MOUSE_RIGHT_CLICK,
        ActionType.MOUSE_DRAG,
        ActionType.MOUSE_SCROLL,
    ):
        assert POSTCONDITION_VERIFIERS[action] == "conditional_pointer_postcondition"
    assert POSTCONDITION_VERIFIERS[ActionType.MOUSE_MOVE] == "pointer_position_postcondition"
    assert POSTCONDITION_VERIFIERS[ActionType.WINDOW_MINIMIZE] == "window_state_postcondition"
    assert POSTCONDITION_VERIFIERS[ActionType.OPEN_APPLICATION] == "conditional_application_window_postcondition"
    assert ActionType.OPEN_URL not in POSTCONDITION_VERIFIERS


def test_open_url_is_listed_as_unobservable() -> None:
    """open_url has a reason in postconditions/_unobservable.py and no verifier."""
    from pilot.agents.postconditions import _unobservable as unobservable

    assert unobservable.UNOBSERVABLE_POSTCONDITION[ActionType.OPEN_URL]
    assert not set(unobservable.UNOBSERVABLE_POSTCONDITION) & set(POSTCONDITION_VERIFIERS)


@pytest.mark.parametrize(
    "action",
    [
        Action(action_type=ActionType.WINDOW_FOCUS, parameters=WindowParams(title="Notepad")),
        Action(action_type=ActionType.KEYBOARD_TYPE, parameters=KeyboardParams(text="x")),
        Action(action_type=ActionType.KEYBOARD_HOLD, parameters=KeyboardParams(key="shift")),
        Action(action_type=ActionType.MOUSE_CLICK, parameters=MouseParams(x=5, y=5)),
        Action(action_type=ActionType.WINDOW_MAXIMIZE, parameters=WindowParams(title="Notepad")),
        Action(action_type=ActionType.OPEN_APPLICATION, parameters=OpenApplicationParams(name="Notepad")),
    ],
)
async def test_a_failed_handler_is_a_failed_action_and_its_verifier_is_not_called(action: Action) -> None:
    verification = await verify(action, success=False)

    assert not verification.passed and label(verification).startswith("FAILED")


# ── pure matching rules ──


def test_focus_matching_follows_the_handlers_rules() -> None:
    assert observe.focus_matches(NOTEPAD, process_name="Notepad") is True
    assert observe.focus_matches(NOTEPAD, process_name="notepad.EXE") is True
    assert observe.focus_matches(NOTEPAD, process_name="note") is False
    assert observe.focus_matches(NOTEPAD, title="NOTEPAD") is True
    assert observe.focus_matches(BROWSER, title="Notepad") is False
    assert observe.focus_matches(BROWSER, title="Notepad", title_pids=frozenset({77})) is True  # its dialog
    assert observe.focus_matches(NOTEPAD, window_id="0x3e9") is True
    assert observe.focus_matches(NOTEPAD, window_id="1002") is False
    no_title = Foreground(("TextEdit",), None, 9, None, "LaunchServices")
    assert observe.focus_matches(no_title, title="Untitled") is None
    assert observe.focus_matches(Foreground(("x",), "", 1, None), window_id="0x10") is None


def test_x11_answers_are_parsed() -> None:
    assert observe.parse_xprop_active("_NET_ACTIVE_WINDOW(WINDOW): window id # 0x3a00007\n") == 0x3A00007
    assert observe.parse_xprop_active("_NET_ACTIVE_WINDOW:  not found.\n") is None
    pid, title = observe.parse_xprop_window(
        '_NET_WM_PID(CARDINAL) = 4242\n_NET_WM_NAME(UTF8_STRING) = "Caf\\303\\251 \\"notes\\""\nWM_NAME(STRING) = "x"\n'
    )
    assert pid == 4242 and "notes" in title
    assert observe.parse_xprop_states("_NET_WM_STATE(ATOM) = _NET_WM_STATE_HIDDEN, _NET_WM_STATE_SKIP_TASKBAR\n") == {
        "_NET_WM_STATE_HIDDEN",
        "_NET_WM_STATE_SKIP_TASKBAR",
    }
    assert observe.parse_wmctrl_lp("0x03a00007  0 4242 host  Notes - Editor\nbad line\n") == [
        (0x3A00007, 4242, "Notes - Editor")
    ]
    assert observe.parse_xdotool_location("X=10\nY=20\nSCREEN=0\nWINDOW=1\n") == (10, 20)


def test_key_tables() -> None:
    assert observe.windows_virtual_key("ctrl") == 0x11
    assert observe.windows_virtual_key("A") == ord("A")
    assert observe.windows_virtual_key("f5") == 0x74
    assert observe.windows_virtual_key("playpause") is None


# ── reading the window in front, per platform ──


def test_wayland_reads_the_focus_through_atspi_or_says_it_cannot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()  # this test reaches the dispatcher itself, with its platform readers faked
    monkeypatch.setattr(observe, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(observe, "wayland", lambda: True)
    monkeypatch.setattr(observe, "_x11_foreground", MagicMock(side_effect=AssertionError("X11 under Wayland")))
    monkeypatch.setattr(observe, "_atspi_foreground", lambda: None)
    with pytest.raises(Unobservable, match="Wayland does not let a program"):
        observe.read_foreground_now()
    monkeypatch.setattr(observe, "_atspi_foreground", lambda: Foreground(("gedit",), "notes", 0, None, "AT-SPI"))
    assert observe.read_foreground_now().source == "AT-SPI"
    for reader in (observe.read_pointer_now, observe.read_clipboard_now):
        with pytest.raises(Unobservable, match="Wayland"):
            reader()
    with pytest.raises(Unobservable, match="Wayland"):
        observe.read_keys_down_now(["shift"])


# ── window_focus ──


async def _focus(monkeypatch: pytest.MonkeyPatch, **params):
    action = Action(action_type=ActionType.WINDOW_FOCUS, parameters=WindowParams(**params))
    monkeypatch.setattr(window_mgr, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(window_mgr, "run_powershell", AsyncMock(return_value=(0, "42", "")))
    with observe.step(action):
        output = await window_mgr.window_focus(params.get("window_id"), params.get("title"), params.get("process_name"))
    return action, output


async def test_focus_is_read_back_right_after_the_step_and_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    reads(monkeypatch, "read_foreground_now", NOTEPAD)

    action, output = await _focus(monkeypatch, process_name="notepad")
    verification = await verify(action, output=output)

    assert verification.passed and verification.independently_observed_actions == [0]
    assert "VERIFIED" in label(verification)


async def test_a_different_window_in_front_fails_the_step_and_is_a_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    reads(monkeypatch, "read_foreground_now", BROWSER)

    with pytest.raises(DesktopStepMismatch, match="no keys went to the wrong window"):
        await _focus(monkeypatch, title="Notepad")

    # It waited for the window manager, counted in steps rather than measured in time.
    assert observe._sleep.await_count == round(observe.FOCUS_SETTLE_SECONDS / observe.SETTLE_STEP_SECONDS)
    action = observe._RECORDS[next(iter(observe._RECORDS))].action
    verification = await verify(action)
    assert not verification.passed and "MISMATCH" in label(verification)
    assert "msedge" in verification.reasons[0] and "Notepad" in verification.reasons[0]
    assert "Inbox - Mail" not in verification.reasons[0]  # a window title never reaches a result


async def test_focus_that_arrives_during_the_settle_time_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    reads(monkeypatch, "read_foreground_now", BROWSER, BROWSER, NOTEPAD)

    action, _ = await _focus(monkeypatch, title="Notepad")

    assert observe._sleep.await_count == 2
    assert (await verify(action)).passed


async def test_focus_that_cannot_be_read_is_reported_not_passed_or_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    raising(monkeypatch, "read_foreground_now", observe.WAYLAND_FOREGROUND)

    action, _ = await _focus(monkeypatch, title="Notepad")
    verification = await verify(action)

    assert verification.passed and verification.independently_observed_actions == []
    assert "REPORTED" in label(verification) and "Wayland" in verification.reasons[0]


async def test_macos_without_accessibility_cannot_compare_a_title(monkeypatch: pytest.MonkeyPatch) -> None:
    reads(monkeypatch, "read_foreground_now", Foreground(("TextEdit",), None, 9, None, "LaunchServices"))

    action, _ = await _focus(monkeypatch, title="Untitled")
    verification = await verify(action)

    assert verification.passed and verification.independently_observed_actions == []
    assert "title cannot be read" in verification.reasons[0]


async def test_focus_outside_a_step_reads_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(window_mgr, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(window_mgr, "run_powershell", AsyncMock(return_value=(0, "42", "")))

    assert await window_mgr.window_focus(title="Notepad") == "Focused window: Notepad"
    observe.read_foreground_now.assert_not_called()


async def test_focus_without_a_reading_is_read_now_and_never_failed_on_a_later_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = Action(action_type=ActionType.WINDOW_FOCUS, parameters=WindowParams(title="Notepad"))
    reads(monkeypatch, "read_foreground_now", NOTEPAD)
    assert "VERIFIED" in label(await verify(action))
    reads(monkeypatch, "read_foreground_now", BROWSER)
    later = await verify(action)
    assert later.passed and "REPORTED" in label(later) and "may have moved" in later.reasons[0]


# ── keyboard_type ──


@pytest.fixture
def clipboard_module(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    fake = types.SimpleNamespace(paste=MagicMock(return_value="earlier"), copy=MagicMock())
    monkeypatch.setitem(__import__("sys").modules, "pyperclip", fake)
    monkeypatch.setattr(input_control.time, "sleep", lambda _seconds: None)
    return fake


async def _type(monkeypatch, text: str, field: FieldReading | Exception, *windows: Foreground, target=None):
    action = Action(action_type=ActionType.KEYBOARD_TYPE, parameters=KeyboardParams(text=text, window_title=target))
    reads(monkeypatch, "read_foreground_now", *windows)
    reader = AsyncMock(side_effect=field) if isinstance(field, Exception) else AsyncMock(return_value=field)
    monkeypatch.setattr(observe, "focused_field", reader)
    with observe.step(action):
        output = await input_control.keyboard_type(text)
    return action, output


def _field(text: str | None, *, secure: bool = False, editable: bool = True) -> FieldReading:
    return FieldReading(editable, secure, text, "UI Automation")


async def test_typed_text_found_in_the_focused_field_is_verified_and_never_repeated(
    monkeypatch, pag, clipboard_module
) -> None:
    action, output = await _type(monkeypatch, SECRET, _field(f"Dear Sam, {SECRET}"), NOTEPAD, NOTEPAD)
    verification = await verify(action, output="Typed exactly")

    assert verification.passed and verification.independently_observed_actions == [0]
    assert "holds the 28 typed characters" in await said(action)
    assert SECRET not in " ".join(verification.details) and SECRET not in str(
        [(r.detail, r.facts) for r in observe._RECORDS.values()]
    )


async def test_the_window_changing_while_typing_stops_the_plan(monkeypatch, pag, clipboard_module) -> None:
    sentence = "The text may have been typed into msedge instead of notepad."
    with pytest.raises(DesktopStepMismatch, match=re.escape(sentence)):
        await _type(monkeypatch, SECRET, _field(SECRET), NOTEPAD, BROWSER)

    action = next(iter(observe._RECORDS.values())).action
    verification = await verify(action)
    assert not verification.passed and "MISMATCH" in label(verification)
    assert SECRET not in verification.reasons[0] and "Inbox - Mail" not in verification.reasons[0]
    # The run stops here instead of typing again: the failed step's own record says where the keys may have gone.
    failed = ActionResult(action=action, success=False, error=sentence)
    assert observe.input_went_elsewhere([failed]) == sentence
    assert ActionType.KEYBOARD_TYPE in RECONCILE_NOT_RETRY_ACTIONS


async def test_a_window_change_after_enter_is_expected(monkeypatch, pag, clipboard_module) -> None:
    action, _ = await _type(monkeypatch, "notepad\n", _field("x"), NOTEPAD, BROWSER)

    verification = await verify(action)
    assert verification.passed and "Enter often opens or closes a window" in verification.reasons[0]


@pytest.mark.parametrize(
    ("field", "outcome", "words"),
    [
        (_field(None, secure=True, editable=False), "REPORTED", "password field, which is never read"),
        (_field(None, editable=False), "REPORTED", "does not expose its text"),
        (_field("", editable=True), "REPORTED", "does not show the 28 typed characters"),
        (_field("Dear Sam, correct horse battery Staple."), "REPORTED", "complete or reformat text"),
        (Unobservable("not verified: the Accessibility permission is not granted"), "REPORTED", "not granted"),
    ],
)
async def test_what_the_field_shows_decides_the_outcome(
    monkeypatch, pag, clipboard_module, field, outcome, words
) -> None:
    if outcome == "MISMATCH":  # nothing landed in the field: the step fails before the plan goes on
        with pytest.raises(DesktopStepMismatch, match=words):
            await _type(monkeypatch, SECRET, field, NOTEPAD, NOTEPAD)
        action = next(iter(observe._RECORDS.values())).action
    else:
        action, _ = await _type(monkeypatch, SECRET, field, NOTEPAD, NOTEPAD)

    verification = await verify(action)
    assert outcome in label(verification) and words in verification.reasons[0]
    assert SECRET not in verification.reasons[0]


async def test_password_managers_are_never_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()
    with pytest.raises(Unobservable, match="never reads"):
        await observe.focused_field(Foreground(("1Password.exe",), "Vault", 5, 1, ""))


def test_field_comparison_keeps_only_lengths_and_answers() -> None:
    facts = observe.field_holds("line one\r\nline two", "one\nline two")
    assert facts == {"field_length": 18, "typed_length": 12, "contains": True, "ends_with": True}


def test_macos_pastes_with_command_v() -> None:
    assert input_control.paste_keys("darwin") == ("command", "v")
    assert input_control.paste_keys("win32") == ("ctrl", "v")
    assert input_control.paste_keys("linux") == ("ctrl", "v")


async def test_typing_without_a_reading_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    action = Action(action_type=ActionType.KEYBOARD_TYPE, parameters=KeyboardParams(text=SECRET))
    verification = await verify(action)
    assert "REPORTED" in label(verification) and "28 typed characters" in verification.reasons[0]


async def test_a_named_window_not_in_front_gets_no_key(monkeypatch, pag, clipboard_module) -> None:
    monkeypatch.setattr(window_mgr, "window_set_text", AsyncMock(side_effect=RuntimeError("no UI Automation editor")))
    monkeypatch.setattr(window_mgr, "window_focus", AsyncMock(return_value="Focused window: Notepad"))
    reads(monkeypatch, "read_foreground_now", BROWSER)
    action = Action(
        action_type=ActionType.KEYBOARD_TYPE, parameters=KeyboardParams(text=SECRET, window_title="Notepad")
    )

    with (
        observe.step(action),
        pytest.raises(DesktopStepMismatch, match="No key was sent: msedge is in front, not Notepad"),
    ):
        await input_control.keyboard_type(SECRET, target_window="Notepad")

    pag.hotkey.assert_not_called()
    pag.typewrite.assert_not_called()
    clipboard_module.copy.assert_not_called()
    # Nothing was typed, so the step may be tried again (after focusing): it is not a "went elsewhere" stop.
    assert observe.input_went_elsewhere([ActionResult(action=action, success=False, error="x")]) is None


@pytest.mark.parametrize(
    ("kind", "params", "run"),
    [
        (
            ActionType.KEYBOARD_PRESS,
            {"key": "enter"},
            lambda: input_control.keyboard_press("enter", target_window="Notepad"),
        ),
        (
            ActionType.KEYBOARD_HOTKEY,
            {"keys": ["ctrl", "s"]},
            lambda: input_control.keyboard_hotkey("ctrl", "s", target_window="Notepad"),
        ),
        (
            ActionType.KEYBOARD_HOLD,
            {"key": "shift"},
            lambda: input_control.keyboard_hold("shift", 0, target_window="Notepad"),
        ),
    ],
)
async def test_keys_for_a_named_window_are_refused_when_another_is_in_front(
    monkeypatch, pag, kind, params, run
) -> None:
    reads(monkeypatch, "read_foreground_now", BROWSER)
    action = Action(action_type=kind, parameters=KeyboardParams(window_title="Notepad", **params))

    with observe.step(action), pytest.raises(DesktopStepMismatch, match="No key was sent"):
        await run()

    pag.press.assert_not_called()
    pag.hotkey.assert_not_called()
    pag.hold.assert_not_called()


async def test_keys_for_a_named_window_in_front_go(monkeypatch, pag) -> None:
    reads(monkeypatch, "read_foreground_now", NOTEPAD)
    reads(monkeypatch, "read_keys_down_now", (set(), []))
    action = Action(
        action_type=ActionType.KEYBOARD_PRESS, parameters=KeyboardParams(key="enter", window_title="Notepad")
    )
    with observe.step(action):
        await input_control.keyboard_press("enter", target_window="Notepad")
    pag.press.assert_called_once_with("enter", presses=1)


def test_a_focus_mismatch_stays_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    action = Action(action_type=ActionType.WINDOW_FOCUS, parameters=WindowParams(title="Notepad"))
    observe.remember(observe.StepRecord(action, "mismatch", "Expected Notepad in front", {"elsewhere": False}))
    assert ActionType.WINDOW_FOCUS not in RECONCILE_NOT_RETRY_ACTIONS
    assert observe.input_went_elsewhere([ActionResult(action=action, success=False, error="x")]) is None


# ── reading the focused field cheaply on Windows ──


@pytest.fixture
def windows_field(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(observe, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(observe, "focused_field", REAL_FOCUSED_FIELD)
    monkeypatch.setattr(observe, "_FIELD_READER", None)
    started = MagicMock(side_effect=AssertionError("no PowerShell process may be started for a field read"))
    monkeypatch.setattr(observe.asyncio, "create_subprocess_exec", started)
    return started


async def test_lives_running_helper_is_reused_for_the_field(windows_field) -> None:
    from pilot.live.focused_text_policy import FocusedText

    reader = types.SimpleNamespace(
        _proc=types.SimpleNamespace(returncode=None),
        read=AsyncMock(return_value=FocusedText("hello", "Edit", 42, False, True, "field", "42.1")),
    )
    observe.set_field_reader(reader)

    reading = await observe.focused_field(NOTEPAD)

    assert reading.text == "hello" and reading.editable


async def test_without_a_running_helper_the_field_is_not_read_and_nothing_is_started(windows_field) -> None:
    observe.set_field_reader(types.SimpleNamespace(_proc=None, read=AsyncMock(side_effect=AssertionError("stopped"))))
    with pytest.raises(Unobservable, match="the window was right; the field's content was not read"):
        await observe.focused_field(NOTEPAD)
    windows_field.assert_not_called()


async def test_typing_without_the_field_read_is_reported_with_the_window_right(
    monkeypatch, pag, clipboard_module, windows_field
) -> None:
    reads(monkeypatch, "read_foreground_now", NOTEPAD)
    action = Action(action_type=ActionType.KEYBOARD_TYPE, parameters=KeyboardParams(text=SECRET))
    with observe.step(action):
        await input_control.keyboard_type(SECRET)

    verification = await verify(action)
    assert verification.passed and verification.independently_observed_actions == []
    assert "the window was right; the field's content was not read" in verification.reasons[0]


# ── keyboard_press, keyboard_hotkey, keyboard_hold ──


async def test_released_keys_are_verified_and_a_stuck_modifier_stops_the_plan(monkeypatch, pag) -> None:
    hotkey = Action(action_type=ActionType.KEYBOARD_HOTKEY, parameters=KeyboardParams(keys=["ctrl", "c"]))
    reads(monkeypatch, "read_keys_down_now", (set(), []))
    with observe.step(hotkey):
        await input_control.keyboard_hotkey("ctrl", "c")
    verification = await verify(hotkey)
    assert verification.passed and "ctrl+c released" in await said(hotkey)

    hold = Action(action_type=ActionType.KEYBOARD_HOLD, parameters=KeyboardParams(key="shift"))
    reads(monkeypatch, "read_keys_down_now", ({"shift"}, []))
    with observe.step(hold), pytest.raises(DesktopStepMismatch, match="shift is still held down"):
        await input_control.keyboard_hold("shift", 0)
    pag.hold.assert_called_once_with("shift")  # the stand-in, never a second import of the real pyautogui
    verification = await verify(hold)
    assert not verification.passed and "still held down" in verification.reasons[0]


async def test_keys_read_afresh_when_no_reading_was_made(monkeypatch: pytest.MonkeyPatch) -> None:
    press = Action(action_type=ActionType.KEYBOARD_PRESS, parameters=KeyboardParams(key="enter"))
    reads(monkeypatch, "read_keys_down_now", (set(), []))
    assert "VERIFIED" in label(await verify(press))
    reads(monkeypatch, "read_keys_down_now", ({"enter"}, []))
    assert "MISMATCH" in label(await verify(press))
    raising(monkeypatch, "read_keys_down_now", observe.WAYLAND_KEYS)
    assert "REPORTED" in label(await verify(press))
    media = Action(action_type=ActionType.KEYBOARD_PRESS, parameters=KeyboardParams(key="playpause"))
    assert "MPRIS" in (await verify(media)).reasons[0]


# ── the mouse ──


async def test_click_where_sent_with_the_button_up_is_verified(monkeypatch, pag) -> None:
    click = Action(action_type=ActionType.MOUSE_CLICK, parameters=MouseParams(x=300, y=200))
    reads(monkeypatch, "read_pointer_now", (301, 199))
    reads(monkeypatch, "read_buttons_down_now", set())
    with observe.step(click):
        await input_control.mouse_click(300, 200)
    verification = await verify(click)
    assert verification.passed and "what the target did with the click cannot be read" in await said(click)


async def test_a_click_whose_pointer_is_not_at_its_target_is_not_made(monkeypatch, pag) -> None:
    click = Action(action_type=ActionType.MOUSE_CLICK, parameters=MouseParams(x=300, y=200))
    reads(monkeypatch, "read_pointer_now", (640, 10))
    with observe.step(click), pytest.raises(DesktopStepMismatch, match=re.escape("No click was made")):
        await input_control.mouse_click(300, 200)
    pag.moveTo.assert_called_once_with(300, 200)
    pag.click.assert_not_called()
    # Nothing was clicked, so this is not a "went elsewhere" stop: the step may be tried again.
    assert observe.input_went_elsewhere([ActionResult(action=click, success=False, error="x")]) is None


async def test_a_pointer_moved_after_the_click_is_not_verified_not_a_stop(monkeypatch, pag) -> None:
    click = Action(action_type=ActionType.MOUSE_CLICK, parameters=MouseParams(x=300, y=200))
    reads(monkeypatch, "read_pointer_now", (300, 200), (640, 10))  # at the event, then after it
    reads(monkeypatch, "read_buttons_down_now", set())
    with observe.step(click):
        await input_control.mouse_click(300, 200)
    pag.click.assert_called_once_with(button="left", clicks=1, interval=0.1)
    verification = await verify(click)
    assert verification.passed and "the mouse may have been moved since" in verification.reasons[0]


async def test_a_button_still_held_after_the_click_stops_the_run(monkeypatch, pag) -> None:
    click = Action(action_type=ActionType.MOUSE_CLICK, parameters=MouseParams(x=300, y=200))
    reads(monkeypatch, "read_pointer_now", (300, 200))
    reads(monkeypatch, "read_buttons_down_now", {"left"})
    with observe.step(click), pytest.raises(DesktopStepMismatch, match="left mouse button is still held"):
        await input_control.mouse_click(300, 200)
    failed = ActionResult(action=click, success=False, error="held")
    assert observe.input_went_elsewhere([failed]) == "held"


async def test_relative_move_is_compared_with_where_it_started(monkeypatch, pag) -> None:
    move = Action(action_type=ActionType.MOUSE_MOVE, parameters=MouseParams(x=5, y=10, relative=True))
    reads(monkeypatch, "read_pointer_now", (10, 10), (15, 20))
    with observe.step(move):
        await input_control.mouse_move(5, 10, relative=True)
    assert "VERIFIED" in label(await verify(move))


async def test_pointer_that_cannot_be_read_is_reported(monkeypatch, pag) -> None:
    drag = Action(action_type=ActionType.MOUSE_DRAG, parameters=MouseParams(x=1, y=1, end_x=50, end_y=50))
    raising(monkeypatch, "read_pointer_now", observe.WAYLAND_POINTER)
    with observe.step(drag):
        await input_control.mouse_drag(1, 1, 50, 50)
    verification = await verify(drag)
    assert verification.passed and verification.independently_observed_actions == []
    assert "Wayland" in verification.reasons[0]


@pytest.mark.parametrize(
    ("action", "output", "words"),
    [
        (
            Action(action_type=ActionType.MOUSE_CLICK, parameters=MouseParams(element_ref="e3", snapshot_id="s1")),
            'Pressed Button "OK" (e3) through UI Automation (Invoke); the cursor did not move',
            "pointer did not move",
        ),
        (Action(action_type=ActionType.MOUSE_SCROLL, parameters=MouseParams(amount=-3)), "Scrolled", "wherever"),
        (Action(action_type=ActionType.MOUSE_MOVE, parameters=MouseParams(x=4, y=4)), "Moved", "No reading"),
    ],
)
async def test_mouse_steps_without_a_reading_say_why(action: Action, output: str, words: str) -> None:
    verification = await verify(action, output=output)
    assert "REPORTED" in label(verification) and words in verification.reasons[0]


async def test_input_through_a_stand_in_is_never_read_back(monkeypatch: pytest.MonkeyPatch) -> None:
    stand_in = MagicMock()
    monkeypatch.setattr(input_control, "_ensure_pyautogui", lambda: stand_in)
    click = Action(action_type=ActionType.MOUSE_CLICK, parameters=MouseParams(x=3, y=3))
    with observe.step(click):
        await input_control.mouse_click(3, 3)
    observe.read_pointer_now.assert_not_called()
    assert observe.recalled(click) is None


# ── window_minimize, window_maximize ──


@pytest.mark.parametrize(
    ("kind", "states", "outcome"),
    [
        (ActionType.WINDOW_MINIMIZE, [(True, False)], "VERIFIED"),
        (ActionType.WINDOW_MINIMIZE, [(False, False), (False, True)], "MISMATCH"),
        (ActionType.WINDOW_MAXIMIZE, [(False, True)], "VERIFIED"),
        (ActionType.WINDOW_MAXIMIZE, [(True, True)], "MISMATCH"),
        (ActionType.WINDOW_MAXIMIZE, [], "MISMATCH"),
    ],
)
async def test_window_state_is_read_back(monkeypatch: pytest.MonkeyPatch, kind, states, outcome) -> None:
    reads(monkeypatch, "window_states_now", states)
    action = Action(action_type=kind, parameters=WindowParams(title="Notepad"))

    assert outcome in label(await verify(action))
    observe.window_states_now.assert_called_once_with("Notepad")


async def test_window_state_unreadable_or_nothing_open(monkeypatch: pytest.MonkeyPatch) -> None:
    raising(monkeypatch, "window_states_now", "not verified: the Accessibility permission is not granted to Heliox")
    action = Action(action_type=ActionType.WINDOW_MINIMIZE, parameters=WindowParams(process_name="TextEdit"))
    verification = await verify(action)
    assert "REPORTED" in label(verification) and "Accessibility" in verification.reasons[0]
    absent = await verify(
        action, output="TextEdit was not open, so nothing was minimized", skip_reason="already_absent"
    )
    assert "REPORTED" in label(absent) and "was not open" in absent.reasons[0]


def test_windows_window_state_matches_title_then_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()
    monkeypatch.setattr(observe, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(
        observe, "_win_windows", lambda: [(1, 42, "notes.txt - Notepad", True, False), (2, 7, "Mail", False, True)]
    )
    monkeypatch.setattr(observe, "process_name", lambda pid: {42: "notepad.exe", 7: "olk.exe"}[pid])
    assert observe.window_states_now("notepad") == [(True, False)]
    assert observe.window_states_now("olk") == [(False, True)]
    assert observe.frames_match((0, 25, 1440, 875), (0, 25, 1440, 875)) and not observe.frames_match(
        (100, 100, 800, 600), (0, 25, 1440, 875)
    )


# ── open_application ──


def test_what_that_application_is_called() -> None:
    assert desktop_input.app_candidates("code", "Launched Visual Studio Code (resolved via Start menu).") == [
        "code",
        "Visual Studio Code",
    ]
    assert "Safari" in desktop_input.app_candidates(
        "safari", "Launched Safari (resolved via /Applications/Safari.app)."
    )
    assert desktop_input.process_matches("CalculatorApp.exe", ["Calculator"])
    assert desktop_input.process_matches("Code.exe", ["Visual Studio Code"])
    assert not desktop_input.process_matches("explorer.exe", ["Calculator"])


async def test_a_window_of_the_app_appearing_is_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    reads(
        monkeypatch,
        "window_rows_now",
        [],
        [observe.WindowRow(11, "ApplicationFrameHost.exe", "Calculator")],
    )
    action = Action(action_type=ActionType.OPEN_APPLICATION, parameters=OpenApplicationParams(name="calculator"))

    verification = await verify(action, output="Launched Calculator (resolved via Windows app registry).")

    assert verification.passed and verification.independently_observed_actions == [0]
    assert desktop_input._sleep.await_count == 1


async def test_an_app_whose_window_never_shows_is_not_verified_never_a_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads(monkeypatch, "window_rows_now", [observe.WindowRow(3, "explorer.exe", "Downloads", 30)])
    action = Action(action_type=ActionType.OPEN_APPLICATION, parameters=OpenApplicationParams(name="Dropbox"))

    tray = await verify(action, output="Launched Dropbox (resolved via Start menu).")

    assert tray.passed and "REPORTED" in label(tray) and "tray" in tray.reasons[0]
    polls = round(desktop_input.APP_WINDOW_WAIT_SECONDS / desktop_input.APP_POLL_SECONDS)
    assert desktop_input._sleep.await_count == polls == 6  # 3 s at most, counted in steps


async def test_app_whose_windows_cannot_be_listed_is_said_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    raising(monkeypatch, "window_rows_now", "native Wayland windows are not listed to other programs")
    action = Action(action_type=ActionType.OPEN_APPLICATION, parameters=OpenApplicationParams(name="gnome-calculator"))

    verification = await verify(action, output="Launched gnome-calculator (resolved via PATH).")

    assert verification.passed and "REPORTED" in label(verification)
    assert "could not be looked for here" in verification.reasons[0]
    desktop_input._sleep.assert_not_awaited()


@pytest.mark.parametrize(
    ("name", "output", "programs", "row"),
    [
        # Windows Terminal: title "Windows PowerShell", process WindowsTerminal, started from its Store app id.
        (
            "Terminal",
            "Launched Terminal (resolved via Windows app registry).",
            ["Microsoft.WindowsTerminal_8wekyb3d8bbwe!App"],
            observe.WindowRow(5, "WindowsTerminal.exe", "Windows PowerShell", 51),
        ),
        # GNOME Terminal: title "user@host: ~", process gnome-terminal-server (cut to 15 characters by the kernel).
        (
            "terminal",
            "Launched Terminal (resolved via its desktop entry org.gnome.Terminal.desktop).",
            ["/usr/bin/gnome-terminal", "org.gnome.Terminal"],
            observe.WindowRow(6, "gnome-terminal-", "user@host: ~", 61),
        ),
        # Files: title "Home", process nautilus, desktop entry org.gnome.Nautilus.
        (
            "files",
            "Launched Files (resolved via its desktop entry org.gnome.Nautilus.desktop).",
            ["/usr/bin/nautilus", "org.gnome.Nautilus"],
            observe.WindowRow(7, "nautilus", "Home", 71),
        ),
        (
            "calculator",
            "Launched Calculator (resolved via PATH).",
            [],
            observe.WindowRow(8, "gnome-calculator", "", 81),
        ),
        # Edge: a zero-width space in its title, process msedge.
        (
            "Microsoft Edge",
            "Launched Microsoft Edge (resolved via Start menu).",
            [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"],
            observe.WindowRow(9, "msedge.exe", "New tab - Microsoft\u200b Edge", 91),
        ),
    ],
)
async def test_everyday_apps_are_found_by_their_process_or_program(monkeypatch, name, output, programs, row) -> None:
    reads(monkeypatch, "window_rows_now", [observe.WindowRow(3, "explorer.exe", "Downloads", 30), row])
    action = Action(action_type=ActionType.OPEN_APPLICATION, parameters=OpenApplicationParams(name=name))
    observe.remember(observe.StepRecord(action, "launch", "", {"programs": programs}))

    verification = await verify(action, output=output)

    assert verification.passed and verification.independently_observed_actions == [0]
    desktop_input._sleep.assert_not_awaited()


async def test_a_new_window_after_the_launch_is_evidence_whatever_its_title(monkeypatch) -> None:
    reads(
        monkeypatch,
        "window_rows_now",
        [observe.WindowRow(3, "explorer.exe", "Downloads", 30), observe.WindowRow(44, "launcher-x", "Welcome", 440)],
    )
    action = Action(action_type=ActionType.OPEN_APPLICATION, parameters=OpenApplicationParams(name="Zed"))
    observe.remember(observe.StepRecord(action, "launch", "", {"windows_before": [30]}))

    verification = await verify(action, output="Launched Zed (resolved via PATH).")

    assert verification.passed and verification.independently_observed_actions == [0]
    assert "new window" in await said(action, output="Launched Zed (resolved via PATH).")


async def test_a_repair_round_never_opens_the_app_a_second_time(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.agents.executor import Executor
    from pilot.intelligence.experience import experience_scope
    from pilot.system import applications

    launch = AsyncMock(return_value="Launched Notepad (resolved via PATH).")
    monkeypatch.setattr(applications, "launch_application", launch)
    monkeypatch.setattr(observe, "window_rows_now", MagicMock(return_value=[]))
    observe._LAUNCHED.clear()
    action = Action(action_type=ActionType.OPEN_APPLICATION, parameters=OpenApplicationParams(name="Notepad"))

    with experience_scope(task_id="task-1"):
        first = await Executor._exec_open_application(None, action)
        again = await Executor._exec_open_application(None, action)
    with experience_scope(task_id="task-2"):
        other_task = await Executor._exec_open_application(None, action)

    assert first.startswith("Launched") and other_task.startswith("Launched")
    assert again == "Notepad was opened a moment ago in this task, so it was not opened again"
    assert launch.await_count == 2
    assert (await verify(action, output=again)).passed
    observe._LAUNCHED.clear()


def test_the_launcher_names_the_program_it_started() -> None:
    from pilot.system.applications import launched_program

    assert launched_program(["explorer.exe", r"shell:AppsFolder\Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"]) == (
        "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"
    )
    assert launched_program(["/usr/bin/nautilus", "--new-window"]) == "/usr/bin/nautilus"


# ── fix10desk: everyday false alarms ──


START_MENU = Foreground(("StartMenuExperienceHost.exe",), "Start", 900, 9001, "GetForegroundWindow")
SEARCH = Foreground(("SearchHost.exe",), "Search", 901, 9002, "GetForegroundWindow")
TASKBAR = Foreground(("explorer.exe",), "", 4, 4001, "GetForegroundWindow")


def test_the_shell_moving_focus_to_its_search_window_is_not_a_window_change() -> None:
    assert observe.same_place(START_MENU, SEARCH)
    assert observe.same_place(TASKBAR, SEARCH)
    assert observe.same_place(NOTEPAD, Foreground(("notepad.exe",), "Save as", 42, 1009, ""))  # its own dialog
    assert not observe.same_place(NOTEPAD, BROWSER)
    assert not observe.same_place(TASKBAR, BROWSER)
    assert not observe.same_place(NOTEPAD, SEARCH)


async def test_typing_into_the_start_menu_goes_on(monkeypatch, pag, clipboard_module) -> None:
    action, _ = await _type(monkeypatch, "notepad", _field(None, editable=False), START_MENU, SEARCH)
    verification = await verify(action)
    assert verification.passed and "does not expose its text" in verification.reasons[0]


async def test_a_field_that_shows_the_paste_late_is_read_again(monkeypatch, pag, clipboard_module) -> None:
    action = Action(action_type=ActionType.KEYBOARD_TYPE, parameters=KeyboardParams(text=SECRET))
    reads(monkeypatch, "read_foreground_now", NOTEPAD)
    reader = AsyncMock(side_effect=[_field(""), _field(""), _field(f"Hi {SECRET}")])
    monkeypatch.setattr(observe, "focused_field", reader)
    with observe.step(action):
        await input_control.keyboard_type(SECRET)
    assert reader.await_count == 3 and observe._sleep.await_count == 2
    assert "VERIFIED" in label(await verify(action))


async def test_a_field_that_stays_empty_is_not_verified_and_the_run_goes_on(monkeypatch, pag, clipboard_module) -> None:
    action, _ = await _type(monkeypatch, SECRET, _field(""), NOTEPAD, NOTEPAD)
    assert observe.focused_field.await_count == observe.FIELD_READS
    assert observe._sleep.await_count == observe.FIELD_READS - 1  # about 300 ms, counted in steps
    verification = await verify(action)
    assert verification.passed and "REPORTED" in label(verification)
    assert observe.input_went_elsewhere([ActionResult(action=action, success=False, error="x")]) is None


async def test_background_typing_is_not_read_again_and_never_stops(monkeypatch, pag) -> None:
    monkeypatch.setattr(window_mgr, "window_set_text", AsyncMock(return_value="ok"))
    monkeypatch.setattr(window_mgr, "window_read_text", AsyncMock(side_effect=AssertionError("no second process")))
    action = Action(action_type=ActionType.KEYBOARD_TYPE, parameters=KeyboardParams(text=SECRET, window_title="Notes"))
    with observe.step(action):
        await input_control.keyboard_type(SECRET, target_window="Notes")
    verification = await verify(action)
    assert verification.passed and "compared it in the same call" in verification.reasons[0]


def test_a_linux_name_cut_to_fifteen_characters_still_matches() -> None:
    cut = Foreground(("gnome-terminal-",), "user@host: ~", 7, 70, "xprop")
    assert observe.focus_matches(cut, process_name="gnome-terminal-server") is True
    assert observe.focus_matches(cut, process_name="gnome-text-editor") is False


def test_window_titles_stay_out_of_results() -> None:
    assert BROWSER.describe() == "msedge"
    assert Foreground((), "Re: salary review", 1, 2).describe() == "a window"


def test_pbpaste_runs_under_a_utf8_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()
    calls = []
    monkeypatch.setattr(observe, "CURRENT_PLATFORM", Platform.MACOS)
    monkeypatch.setattr(observe, "_run", lambda args, **kw: calls.append((args, kw)) or "caf\u00e9")
    assert observe.read_clipboard_now() == ("caf\u00e9", "pbpaste")
    assert calls == [(["pbpaste"], {"utf8_locale": True})]
