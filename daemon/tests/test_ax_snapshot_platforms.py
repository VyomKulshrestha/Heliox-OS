"""Accessibility snapshots on macOS and Linux: role mapping, never-read windows, the AT-SPI walk and the dispatch,
with fake platform layers on every system (the real runs are in the platform-readers CI job)."""

from __future__ import annotations

import json

import pytest

from pilot.live import focused_text_linux as L
from pilot.system import accessibility
from pilot.system import ax_snapshot_platforms as P
from pilot.system.accessibility import AccessibilityUnavailable
from pilot.system.platform_detect import Platform


def test_roles_map_to_the_control_types_the_planner_knows():
    assert P.control_type("AXButton", P.MAC_ROLES) == "Button"
    assert P.control_type("AXTextField", P.MAC_ROLES) == "Edit"
    assert P.control_type("AXGroup", P.MAC_ROLES) is None
    assert P.control_type("push button", P.ATSPI_ROLES) == "Button"
    assert P.control_type("password text", P.ATSPI_ROLES) == "Edit"
    assert P.control_type("filler", P.ATSPI_ROLES) is None
    assert P.keep("Button", "") and P.keep("Text", "Name") and not P.keep("Text", "  ") and not P.keep(None, "x")
    assert P.window_key("a") == P.window_key("a") != P.window_key("b")


def test_never_read_windows_are_refused():
    assert P.refuse_window("1Password", "Vault", ())
    assert P.refuse_window("Heliox-OS", "Heliox-OS", ())
    assert P.refuse_window("Slack", "general", ["slack"])
    assert P.refuse_window("Keychain Access", "login", ())
    assert P.refuse_window("gedit", "notes.txt", ()) == ""


class FakeClient:
    """An AT-SPI window with a button, a password entry, a label and a hidden button."""

    def __init__(self) -> None:
        self.nodes = {
            "/win": (
                "frame",
                "Heliox AT-SPI fixture",
                [L.STATE_ACTIVE, L.STATE_SHOWING],
                ["/btn", "/pw", "/lbl", "/hid"],
                (0, 0, 400, 300),
            ),
            "/btn": ("push button", "Press me", [L.STATE_SHOWING, 8, 24], [], (10, 10, 80, 24)),
            "/pw": ("password text", "password", [L.STATE_SHOWING, L.STATE_FOCUSED, 8], [], (10, 50, 200, 24)),
            "/lbl": ("label", "Type here", [L.STATE_SHOWING], [], (10, 90, 100, 20)),
            "/hid": ("push button", "Hidden", [], [], (0, 0, 10, 10)),
        }
        outer = self

        class T:
            def call(self, bus, path, interface, method, signature="", body=()):
                assert method == "GetExtents"
                return (outer.nodes[path][4],)

        self._t = T()

    def focused(self):
        return (":1.5", "/pw")

    def _role(self, ref):
        return self.nodes[ref[1]][0]

    def _get(self, ref, interface, prop):
        if prop == "Name":
            return self.nodes[ref[1]][1]
        if prop == "Parent":
            return (":1.5", "/win")
        raise AssertionError(prop)

    def _state(self, ref):
        return set(self.nodes[ref[1]][2])

    def _children(self, ref):
        return [(":1.5", child) for child in self.nodes[ref[1]][3]]

    def pid(self, bus):
        return 4242


def test_linux_capture_lists_showing_controls_and_never_values(monkeypatch):
    shared = L._Shared()
    shared.client = FakeClient()
    monkeypatch.setattr(L, "SHARED", shared)
    monkeypatch.setattr(L, "comm_name", lambda pid: "python3")
    payload, (key, rect) = P.capture_linux(50)
    names = [(e["t"], e["n"]) for e in payload["elements"]]
    assert names == [("Button", "Press me"), ("Edit", "password"), ("Text", "Type here")]
    assert all("value" not in e for e in payload["elements"])
    assert payload["title"] == "Heliox AT-SPI fixture" and rect == (0, 0, 400, 300)
    assert payload["elements"][1]["f"] is True and payload["elements"][0]["e"] is True
    with pytest.raises(RuntimeError, match="does not read this window"):
        P.capture_linux(50, excluded=["python3"])


@pytest.mark.asyncio
async def test_capture_snapshot_dispatches_to_the_platform_reader(monkeypatch):
    monkeypatch.setattr(accessibility, "CURRENT_PLATFORM", Platform.MACOS)
    element = {"t": "Button", "n": "Save", "a": "", "x": 5, "y": 6, "w": 30, "h": 10, "e": True, "f": False}
    monkeypatch.setattr(
        P,
        "capture_macos",
        lambda limit, excluded: ({"title": "Doc", "process": 7, "elements": [element]}, (99, (0, 0, 9, 9))),
    )
    store = accessibility.SnapshotStore(window_probe=lambda: accessibility.WindowIdentity(99, (0, 0, 9, 9)))
    snapshot = await accessibility.capture_snapshot(store=store)
    assert snapshot.elements[0].label == 'Button "Save"' and snapshot.window.handle == 99
    assert store.resolve(snapshot.snapshot_id, "e1").center == (20, 11)

    def refused(limit, excluded):
        raise RuntimeError("Heliox does not read this window: it is a password manager.")

    monkeypatch.setattr(P, "capture_macos", refused)
    with pytest.raises(AccessibilityUnavailable, match="password manager"):
        await accessibility.capture_snapshot(store=store)


@pytest.mark.asyncio
async def test_windows_snapshots_of_a_password_manager_are_refused(monkeypatch):
    from unittest.mock import AsyncMock

    window = accessibility.WindowIdentity(handle=42, rect=(0, 0, 800, 600))
    payload = json.dumps({"title": "Vault", "process": 9, "elements": [], "truncated": False})
    monkeypatch.setattr(accessibility, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(accessibility, "foreground_window", lambda: window)
    monkeypatch.setattr(accessibility, "run_powershell", AsyncMock(return_value=(0, payload, "")))
    monkeypatch.setattr(accessibility, "_process_name", lambda pid: "1Password")
    store = accessibility.SnapshotStore(window_probe=lambda: window)
    with pytest.raises(AccessibilityUnavailable, match="does not read this window"):
        await accessibility.capture_snapshot(store=store)


# ── the person's excluded apps reach the snapshot through the real callers ──


def _executor():
    from pilot.agents.executor import Executor
    from pilot.config import PilotConfig

    config = PilotConfig()
    config.timeline.excluded_apps = ["python3", "Notes", "Slack"]
    executor = Executor.__new__(Executor)
    executor._config = config
    return executor


def _snapshot_action():
    from pilot.actions import AccessibilitySnapshotParams, Action, ActionType

    return Action(
        action_type=ActionType.SCREEN_ACCESSIBILITY_SNAPSHOT,
        target="screen",
        parameters=AccessibilitySnapshotParams(element_limit=20),
    )


@pytest.mark.asyncio
async def test_an_excluded_apps_window_is_refused_through_the_executor_on_linux(monkeypatch):
    shared = L._Shared()
    shared.client = FakeClient()
    monkeypatch.setattr(L, "SHARED", shared)
    monkeypatch.setattr(L, "comm_name", lambda pid: "python3")
    monkeypatch.setattr(accessibility, "CURRENT_PLATFORM", Platform.LINUX)
    with pytest.raises(AccessibilityUnavailable, match="excluded"):
        await _executor()._exec_screen_accessibility_snapshot(_snapshot_action())


@pytest.mark.asyncio
async def test_an_excluded_apps_window_is_refused_through_the_executor_on_macos(monkeypatch):
    seen = {}

    def capture(limit, excluded):  # the pyobjc walk itself needs a Mac; the refusal rule is the shared one
        seen["excluded"] = excluded
        reason = P.refuse_window("Notes", "Shopping list", excluded)
        if reason:
            raise RuntimeError(reason)
        return {"title": "", "process": 0, "elements": []}, (1, (0, 0, 1, 1))

    monkeypatch.setattr(accessibility, "CURRENT_PLATFORM", Platform.MACOS)
    monkeypatch.setattr(P, "capture_macos", capture)
    with pytest.raises(AccessibilityUnavailable, match="excluded"):
        await _executor()._exec_screen_accessibility_snapshot(_snapshot_action())
    assert "Notes" in seen["excluded"]


@pytest.mark.asyncio
async def test_an_excluded_apps_window_is_refused_through_the_executor_on_windows(monkeypatch):
    from unittest.mock import AsyncMock

    window = accessibility.WindowIdentity(handle=42, rect=(0, 0, 800, 600))
    payload = json.dumps({"title": "general", "process": 9, "elements": [], "truncated": False})
    monkeypatch.setattr(accessibility, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(accessibility, "foreground_window", lambda: window)
    monkeypatch.setattr(accessibility, "run_powershell", AsyncMock(return_value=(0, payload, "")))
    monkeypatch.setattr(accessibility, "_process_name", lambda pid: "slack")
    with pytest.raises(AccessibilityUnavailable, match="excluded"):
        await _executor()._exec_screen_accessibility_snapshot(_snapshot_action())


@pytest.mark.asyncio
async def test_the_autonomous_observer_reads_nothing_from_an_excluded_app(monkeypatch):
    from pilot import server

    shared = L._Shared()
    shared.client = FakeClient()
    monkeypatch.setattr(L, "SHARED", shared)
    monkeypatch.setattr(L, "comm_name", lambda pid: "python3")
    monkeypatch.setattr(accessibility, "CURRENT_PLATFORM", Platform.LINUX)
    assert await server._accessibility_observation(excluded=("python3",)) == ""
    assert "Press me" in await server._accessibility_observation()
