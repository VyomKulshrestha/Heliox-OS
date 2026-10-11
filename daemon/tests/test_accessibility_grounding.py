"""Accessibility-tree grounding: snapshot parsing, stale-ref refusal and click resolution."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import AccessibilitySnapshotParams, ActionType, MouseParams
from pilot.agents.autonomous import AutonomousExecutor
from pilot.agents.executor import Executor
from pilot.agents.planner import Planner
from pilot.system import accessibility
from pilot.system.accessibility import (
    AccessibilityUnavailable,
    SnapshotStore,
    StaleElementRef,
    WindowIdentity,
    parse_snapshot,
)

WINDOW = WindowIdentity(handle=42, rect=(0, 0, 800, 600))


def _payload(*elements: dict, title: str = "Untitled - Notepad", truncated: bool = False) -> str:
    items: object = list(elements)
    if len(elements) == 1:
        items = elements[0]  # ConvertTo-Json collapses a one-item list
    return json.dumps({"title": title, "process": 7, "elements": items, "truncated": truncated})


SAVE = {"t": "ControlType.Button", "n": "Save", "a": "save", "x": 10, "y": 20, "w": 40, "h": 20, "e": True, "f": False}
EDIT = {
    "t": "ControlType.Edit",
    "n": "  Search\n box ",
    "a": "",
    "x": 100,
    "y": 50,
    "w": 200,
    "h": 30,
    "e": True,
    "f": True,
}
OFF = {"t": "ControlType.Button", "n": "Delete", "a": "", "x": 0, "y": 0, "w": 10, "h": 10, "e": False, "f": False}


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _store(window: WindowIdentity | None = WINDOW, clock: _Clock | None = None) -> tuple[SnapshotStore, dict]:
    state = {"window": window}
    store = SnapshotStore(clock=clock or _Clock(), window_probe=lambda: state["window"])
    return store, state


def _executor_stub() -> SimpleNamespace:
    return SimpleNamespace(
        _accessibility_target=Executor._accessibility_target,
        _press_element_in_background=Executor._press_element_in_background,
    )


def test_parse_snapshot_assigns_refs_and_cleans_names():
    snapshot = parse_snapshot(_payload(SAVE, EDIT), WINDOW, captured_at=5.0, snapshot_id="ax1")

    assert [element.ref for element in snapshot.elements] == ["e1", "e2"]
    assert snapshot.elements[0].role == "Button"
    assert snapshot.elements[1].name == "Search box"
    assert snapshot.elements[1].focused is True
    assert snapshot.elements[0].center == (30, 30)
    assert snapshot.title == "Untitled - Notepad"


def test_parse_snapshot_accepts_a_collapsed_single_element():
    snapshot = parse_snapshot(_payload(SAVE), WINDOW, captured_at=0.0)

    assert len(snapshot.elements) == 1
    assert snapshot.elements[0].name == "Save"


def test_describe_lists_refs_and_how_to_click_them():
    snapshot = parse_snapshot(_payload(SAVE, OFF), WINDOW, captured_at=0.0, snapshot_id="ax9")
    text = snapshot.describe()

    assert 'e1 Button "Save" at (30,30)' in text
    assert "[disabled]" in text
    assert '"snapshot_id": "ax9"' in text
    assert "untrusted screen data" in text


def test_resolve_returns_the_element_while_the_window_is_unchanged():
    store, _ = _store()
    store.add(parse_snapshot(_payload(SAVE, EDIT), WINDOW, captured_at=store.now(), snapshot_id="ax1"))

    assert store.resolve("ax1", "e2").name == "Search box"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("other_window", "different window"),
        ("moved", "moved or was resized"),
        ("expired", "older than"),
        ("unknown_snapshot", "unknown or expired"),
        ("unknown_ref", "not in snapshot"),
    ],
)
def test_resolve_refuses_stale_refs(change, message):
    clock = _Clock()
    store, state = _store(clock=clock)
    store.add(parse_snapshot(_payload(SAVE, EDIT), WINDOW, captured_at=clock(), snapshot_id="ax1"))
    snapshot_id, ref = "ax1", "e1"
    if change == "other_window":
        state["window"] = WindowIdentity(handle=99, rect=WINDOW.rect)
    elif change == "moved":
        state["window"] = WindowIdentity(handle=42, rect=(10, 0, 810, 600))
    elif change == "expired":
        clock.now += accessibility.SNAPSHOT_TTL_SECONDS + 1
    elif change == "unknown_snapshot":
        snapshot_id = "ax404"
    else:
        ref = "e99"

    with pytest.raises(StaleElementRef, match=message):
        store.resolve(snapshot_id, ref)


def test_resolve_refuses_a_disabled_element():
    store, _ = _store()
    store.add(parse_snapshot(_payload(SAVE, OFF), WINDOW, captured_at=store.now(), snapshot_id="ax1"))

    with pytest.raises(ValueError, match="disabled"):
        store.resolve("ax1", "e2")


def test_store_keeps_only_recent_snapshots():
    store = SnapshotStore(capacity=2, window_probe=lambda: WINDOW)
    for index in range(3):
        store.add(parse_snapshot(_payload(SAVE), WINDOW, captured_at=store.now(), snapshot_id=f"ax{index}"))

    with pytest.raises(StaleElementRef):
        store.resolve("ax0", "e1")
    assert store.resolve("ax2", "e1").name == "Save"


@pytest.mark.asyncio
async def test_capture_is_unavailable_off_windows(monkeypatch):
    monkeypatch.setattr(accessibility, "CURRENT_PLATFORM", object())

    with pytest.raises(AccessibilityUnavailable, match="Windows only"):
        await accessibility.capture_snapshot()


@pytest.mark.asyncio
async def test_capture_parses_ui_automation_output(monkeypatch):
    from pilot.system.platform_detect import Platform

    store, _ = _store()
    monkeypatch.setattr(accessibility, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(accessibility, "foreground_window", lambda: WINDOW)
    run = AsyncMock(return_value=(0, _payload(SAVE, EDIT), ""))
    monkeypatch.setattr(accessibility, "run_powershell", run)

    snapshot = await accessibility.capture_snapshot(max_elements=999, store=store)

    script = run.await_args.args[0]
    assert "FromHandle([IntPtr]42)" in script
    assert f"$found.Count -lt {accessibility.MAX_ELEMENTS_LIMIT}" in script
    assert store.resolve(snapshot.snapshot_id, "e1").name == "Save"


@pytest.mark.asyncio
async def test_mouse_click_uses_a_fresh_element_ref(monkeypatch):
    store, _ = _store()
    store.add(parse_snapshot(_payload(SAVE, EDIT), WINDOW, captured_at=store.now(), snapshot_id="ax1"))
    monkeypatch.setattr(accessibility, "_default_store", store)
    click = AsyncMock(return_value="Clicked at (200, 65)")
    monkeypatch.setattr("pilot.system.input_control.mouse_click", click)
    monkeypatch.setattr(accessibility, "press_in_background", AsyncMock(return_value=None))
    action = SimpleNamespace(parameters=MouseParams(element_ref="e2", snapshot_id="ax1"))

    output = await Executor._exec_mouse_click(_executor_stub(), action)

    click.assert_awaited_once_with(200, 65, "left", 1)
    assert 'Edit "Search box" (e2)' in output


@pytest.mark.asyncio
async def test_mouse_click_refuses_a_stale_ref_without_clicking(monkeypatch):
    store, state = _store()
    store.add(parse_snapshot(_payload(SAVE), WINDOW, captured_at=store.now(), snapshot_id="ax1"))
    state["window"] = WindowIdentity(handle=7, rect=WINDOW.rect)
    monkeypatch.setattr(accessibility, "_default_store", store)
    click = AsyncMock()
    monkeypatch.setattr("pilot.system.input_control.mouse_click", click)
    action = SimpleNamespace(parameters=MouseParams(element_ref="e1", snapshot_id="ax1"))

    with pytest.raises(StaleElementRef, match="re-snapshot required"):
        await Executor._exec_mouse_click(_executor_stub(), action)
    click.assert_not_awaited()


def test_element_ref_without_snapshot_id_is_rejected():
    with pytest.raises(ValueError, match="requires the snapshot_id"):
        Executor._accessibility_target(MouseParams(element_ref="e1"))


def test_planner_maps_ref_synonyms_and_element_limit():
    planner = Planner(MagicMock(), MagicMock())

    click = planner._parse_parameters(ActionType.MOUSE_CLICK, {"ref": "e3", "snapshot_id": "ax2"})
    snapshot = planner._parse_parameters(ActionType.SCREEN_ACCESSIBILITY_SNAPSHOT, {"max_elements": 40})

    assert isinstance(click, MouseParams) and click.element_ref == "e3" and click.snapshot_id == "ax2"
    assert isinstance(snapshot, AccessibilitySnapshotParams) and snapshot.element_limit == 40


@pytest.mark.asyncio
async def test_autonomous_observation_includes_the_accessibility_snapshot():
    screen = SimpleNamespace(
        observe_now=AsyncMock(
            return_value=SimpleNamespace(screen_hash="h", active_app="Notepad", active_window_title="x")
        ),
        get_context_for_planner=MagicMock(return_value="Viewing Notepad"),
    )
    observer = AsyncMock(return_value='Accessibility snapshot ax3 ...\ne1 Button "Save" at (30,30)')
    autonomous = AutonomousExecutor(
        planner=None,
        executor=None,
        verifier=None,
        decomposer=MagicMock(),
        screen_vision=screen,
        accessibility_observer=observer,
    )

    context, *_ = await autonomous._observe_screen_context()

    assert 'e1 Button "Save" at (30,30)' in context
    observer.assert_awaited_once()


RID = {**SAVE, "r": "42.1001.4.7"}


@pytest.mark.asyncio
async def test_single_left_click_on_a_ref_presses_in_the_background(monkeypatch):
    store, _ = _store()
    store.add(parse_snapshot(_payload(RID, EDIT), WINDOW, captured_at=store.now(), snapshot_id="ax1"))
    monkeypatch.setattr(accessibility, "_default_store", store)
    press = AsyncMock(return_value="invoke")
    monkeypatch.setattr(accessibility, "press_in_background", press)
    click = AsyncMock()
    monkeypatch.setattr("pilot.system.input_control.mouse_click", click)
    action = SimpleNamespace(parameters=MouseParams(element_ref="e1", snapshot_id="ax1"))

    output = await Executor._exec_mouse_click(_executor_stub(), action)

    assert press.await_args.args[1].runtime_id == "42.1001.4.7"
    click.assert_not_awaited()
    assert "cursor did not move" in output and 'Button "Save" (e1)' in output


@pytest.mark.asyncio
async def test_right_and_double_clicks_never_use_background_press(monkeypatch):
    press = AsyncMock(return_value="invoke")
    monkeypatch.setattr(accessibility, "press_in_background", press)

    for params in (
        MouseParams(element_ref="e1", snapshot_id="ax1", button="right"),
        MouseParams(element_ref="e1", snapshot_id="ax1", clicks=2),
    ):
        assert await Executor._press_element_in_background(params) is None
    press.assert_not_awaited()


@pytest.mark.asyncio
async def test_background_press_reports_a_vanished_element_as_stale(monkeypatch):
    from pilot.system.platform_detect import Platform

    monkeypatch.setattr(accessibility, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(accessibility, "run_powershell", AsyncMock(return_value=(0, "missing\n", "")))
    element = parse_snapshot(_payload(RID), WINDOW, captured_at=0.0).elements[0]

    with pytest.raises(StaleElementRef, match="no longer in the window"):
        await accessibility.press_in_background(WINDOW, element)


@pytest.mark.asyncio
async def test_background_press_falls_back_when_no_pattern_applies(monkeypatch):
    from pilot.system.platform_detect import Platform

    monkeypatch.setattr(accessibility, "CURRENT_PLATFORM", Platform.WINDOWS)
    run = AsyncMock(return_value=(0, "none\n", ""))
    monkeypatch.setattr(accessibility, "run_powershell", run)
    element = parse_snapshot(_payload(RID), WINDOW, captured_at=0.0).elements[0]

    assert await accessibility.press_in_background(WINDOW, element) is None
    assert "[int[]]@(42,1001,4,7)" in run.await_args.args[0]


def test_malformed_runtime_ids_are_dropped():
    snapshot = parse_snapshot(_payload({**SAVE, "r": "1; Remove-Item"}), WINDOW, captured_at=0.0)

    assert snapshot.elements[0].runtime_id == ""


def test_snapshot_script_keeps_named_button_like_panes_but_not_layout_panes():
    script = accessibility._snapshot_script(4242, 50)

    # WinForms buttons reach UI Automation as a named Pane leaf whose window class says BUTTON.
    assert "$t -eq 'ControlType.Pane' -or $t -eq 'ControlType.Custom'" in script
    assert r"$c.ClassName -match '(^|\.)BUTTON(\.|$)'" in script
    assert "$t = 'ControlType.Button'" in script
    # Only leaves count: a panel that contains other controls is layout, not a target.
    assert "$leaf = ($walker.GetFirstChild($element) -eq $null)" in script
    # Other named leaves qualify only when they still offer an action pattern.
    assert "'Invoke|Toggle|ExpandCollapse|SelectionItem'" in script
    assert "if ($visible -and $keep)" in script


def test_a_winforms_button_reported_as_a_pane_parses_as_a_clickable_button():
    window = WindowIdentity(7, (0, 0, 300, 200))
    payload = json.dumps(
        {
            "title": "Heliox WinForms probe",
            "process": 1,
            "elements": [
                {"t": "ControlType.Button", "n": "Save report", "x": 10, "y": 10, "w": 80, "h": 24, "r": "42.1"}
            ],
        }
    )

    snapshot = accessibility.parse_snapshot(payload, window, captured_at=0.0)

    [element] = snapshot.elements
    assert (element.role, element.name, element.runtime_id) == ("Button", "Save report", "42.1")
    assert "Save report" in snapshot.describe(10)


async def test_win32_buttons_without_a_pattern_are_pressed_with_bm_click(monkeypatch):
    from pilot.system.platform_detect import Platform

    monkeypatch.setattr(accessibility, "CURRENT_PLATFORM", Platform.WINDOWS)
    run = AsyncMock(return_value=(0, "bm_click\n", ""))
    monkeypatch.setattr(accessibility, "run_powershell", run)
    element = parse_snapshot(_payload(RID), WINDOW, captured_at=0.0).elements[0]

    assert await accessibility.press_in_background(WINDOW, element) == "bm_click"
    script = run.await_args.args[0]
    # Posted (not sent) to the button's own window, and only for BUTTON-class windows.
    assert "PostMessage([IntPtr]$hwnd, 0x00F5" in script
    assert r"ClassName -match '(^|\.)BUTTON(\.|$)'" in script
    assert script.index("InvokePattern") < script.index("0x00F5")  # UI Automation patterns are tried first
