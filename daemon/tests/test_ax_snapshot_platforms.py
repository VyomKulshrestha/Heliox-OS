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


def test_mac_labels_fall_back_in_order_and_never_read_a_fields_value():
    def getter(values):
        return lambda key: values.get(key)

    assert P.mac_label(getter({"AXTitle": "Save", "AXDescription": "x"}), "AXButton") == "Save"
    assert P.mac_label(getter({"AXTitle": " ", "AXDescription": "Close"}), "AXButton") == "Close"
    assert P.mac_label(getter({"AXPlaceholderValue": "Search", "AXValue": "my secret query"}), "AXTextField") == (
        "Search"
    )
    assert P.mac_label(getter({"AXHelp": "Send the  message"}), "AXButton") == "Send the message"
    assert P.mac_label(getter({"AXValue": "typed text"}), "AXTextField") == ""
    assert P.mac_label(getter({"AXValue": "Total: 3"}), "AXStaticText") == "Total: 3"
    # The window's own buttons are named by their subrole; a plain button with nothing says nothing.
    assert P.mac_label(getter({"AXSubrole": "AXCloseButton"}), "AXButton") == "close window"
    assert P.mac_label(getter({"AXSubrole": "AXMinimizeButton"}), "AXButton") == "minimise window"
    assert P.mac_label(getter({"AXSubrole": "AXSomethingElse"}), "AXButton") == ""


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

    def is_password(self, ref, role=None):
        return self.nodes[ref[1]][0] == "password text" or ref[1] in getattr(self, "password_paths", ())


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


# ── pressing a control from a snapshot: AXPress on macOS, the AT-SPI action on Linux ──


class PressClient(FakeClient):
    """FakeClient whose button offers AT-SPI actions and records presses."""

    def __init__(self, actions=("click",), title="Heliox AT-SPI fixture") -> None:
        super().__init__()
        self.nodes["/win"] = (self.nodes["/win"][0], title, *self.nodes["/win"][2:])
        self.actions = list(actions)
        self.pressed: list[tuple[str, int]] = []
        outer = self

        class T:
            def call(self, bus, path, interface, method, signature="", body=()):
                if method == "GetExtents":
                    return (outer.nodes[path][4],)
                if method == "GetName":
                    return (outer.actions[body[0]],)
                if method == "DoAction":
                    outer.pressed.append((path, body[0]))
                    return (True,)
                raise AssertionError(method)

        self._t = T()

    def retitle(self, title: str) -> None:
        self.nodes["/win"] = (self.nodes["/win"][0], title, *self.nodes["/win"][2:])

    def _get(self, ref, interface, prop):
        if prop == "NActions":
            return len(self.actions) if ref[1] == "/btn" else 0
        return super()._get(ref, interface, prop)


def _linux(monkeypatch, client):
    shared = L._Shared()
    shared.client = client
    monkeypatch.setattr(L, "SHARED", shared)
    monkeypatch.setattr(L, "comm_name", lambda pid: "gedit")
    monkeypatch.setattr(accessibility, "CURRENT_PLATFORM", Platform.LINUX)


def test_press_action_choice_and_remembered_controls():
    assert P.choose_action(["Press"]) == 0
    assert P.choose_action(["activate", "click"]) == 1
    assert P.choose_action(["expand or contract"]) is None
    for n in range(10):
        P.remember(f"cap{n}", [f"h{n}a", f"h{n}b"])
    assert P.handle_for("cap0", "e1") is None, "only the last few snapshots are kept"
    assert P.handle_for("cap9", "e2") == "h9b"
    assert P.handle_for("cap9", "e3") is None and P.handle_for("cap9", "x1") is None


def test_linux_press_uses_the_controls_own_action(monkeypatch):
    client = PressClient(actions=("activate", "click"))
    _linux(monkeypatch, client)
    assert P.press_linux((":1.5", "/btn")) == ("click", "")
    assert client.pressed == [("/btn", 1)]
    client.actions = []
    assert P.press_linux((":1.5", "/btn")) == ("none", "")
    client.nodes["/btn"] = ("push button", "Press me", [L.STATE_SHOWING], [], (10, 10, 80, 24))
    assert P.press_linux((":1.5", "/btn"))[0] == "disabled"


def test_linux_press_checks_the_window_again_and_presses_nothing_in_a_refused_one(monkeypatch):
    client = PressClient(title="Bitwarden - vault")
    _linux(monkeypatch, client)
    outcome, detail = P.press_linux((":1.5", "/btn"))
    assert outcome == "refused" and "does not read this window" in detail
    client = PressClient()
    _linux(monkeypatch, client)
    assert P.press_linux((":1.5", "/btn"), excluded=("gedit",))[0] == "refused"
    assert client.pressed == []


def test_linux_press_refuses_a_password_field_and_gtk4_controls_are_named(monkeypatch):
    client = PressClient()
    _linux(monkeypatch, client)
    client.password_paths = {"/btn"}  # e.g. a GTK 4 password entry: known only by its numeric role
    outcome, detail = P.press_linux((":1.5", "/btn"))
    assert outcome == "refused" and "password field" in detail and client.pressed == []
    assert P.control_type("text box", P.ATSPI_ROLES) == "Edit" and P.control_type("button", P.ATSPI_ROLES) == "Button"


def _click_action(snapshot_id: str, ref: str):
    from types import SimpleNamespace

    from pilot.actions import MouseParams

    return SimpleNamespace(parameters=MouseParams(element_ref=ref, snapshot_id=snapshot_id))


async def _linux_snapshot(monkeypatch, client):
    _linux(monkeypatch, client)
    store = accessibility.SnapshotStore(window_probe=lambda: accessibility.WindowIdentity(*P.front_window_linux()))
    monkeypatch.setattr(accessibility, "_default_store", store)
    snapshot = await accessibility.capture_snapshot(store=store)
    button = next(e for e in snapshot.elements if e.name == "Press me")
    return snapshot, button


@pytest.mark.asyncio
async def test_a_click_on_a_ref_presses_through_at_spi_and_never_moves_the_mouse(monkeypatch):
    from unittest.mock import AsyncMock

    client = PressClient()
    snapshot, button = await _linux_snapshot(monkeypatch, client)
    click = AsyncMock()
    monkeypatch.setattr("pilot.system.input_control.mouse_click", click)
    output = await _executor()._exec_mouse_click(_click_action(snapshot.snapshot_id, button.ref))
    assert client.pressed == [("/btn", 0)] and "through AT-SPI (click)" in output
    click.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_refused_window_gets_neither_a_press_nor_a_click(monkeypatch):
    from unittest.mock import AsyncMock

    client = PressClient()
    snapshot, button = await _linux_snapshot(monkeypatch, client)
    client.retitle("KeePassXC")  # the same window, now one Heliox never reads
    click = AsyncMock()
    monkeypatch.setattr("pilot.system.input_control.mouse_click", click)
    with pytest.raises(accessibility.AccessibilityRefused):
        await _executor()._exec_mouse_click(_click_action(snapshot.snapshot_id, button.ref))
    assert client.pressed == []
    click.assert_not_awaited()


@pytest.mark.asyncio
async def test_without_a_press_action_x11_clicks_the_centre_and_wayland_says_why_not(monkeypatch):
    from unittest.mock import AsyncMock

    client = PressClient(actions=())
    snapshot, button = await _linux_snapshot(monkeypatch, client)
    click = AsyncMock(return_value="Clicked")
    monkeypatch.setattr("pilot.system.input_control.mouse_click", click)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    await _executor()._exec_mouse_click(_click_action(snapshot.snapshot_id, button.ref))
    click.assert_awaited_once_with(50, 22, "left", 1)

    click.reset_mock()
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    with pytest.raises(ValueError, match="on Wayland Heliox cannot move the pointer"):
        await _executor()._exec_mouse_click(_click_action(snapshot.snapshot_id, button.ref))
    click.assert_not_awaited()


class FakeAX:
    """Just enough of ApplicationServices for press_macos: one button in one window."""

    def __init__(self, title="Draft", actions=("AXPress",), enabled=True) -> None:
        self.title, self.actions, self.enabled = title, list(actions), enabled
        self.performed: list[tuple[str, str]] = []

    def AXIsProcessTrusted(self):  # noqa: N802
        return True

    def AXUIElementCopyAttributeValue(self, element, name, _out):  # noqa: N802
        values = {("button", "AXWindow"): "window", ("window", "AXTitle"): self.title}
        values[("button", "AXEnabled")] = self.enabled
        return (0, values[(element, name)]) if (element, name) in values else (-25205, None)

    def AXUIElementGetPid(self, _element, _out):  # noqa: N802
        return 0, 77

    def AXUIElementCopyActionNames(self, _element, _out):  # noqa: N802
        return 0, self.actions

    def AXUIElementPerformAction(self, element, action):  # noqa: N802
        self.performed.append((element, action))
        return 0


def _fake_mac(monkeypatch, fake: FakeAX, app_name: str = "TextEdit"):
    import sys
    import types

    appkit = types.ModuleType("AppKit")
    running = types.SimpleNamespace(localizedName=lambda: app_name)
    appkit.NSRunningApplication = types.SimpleNamespace(runningApplicationWithProcessIdentifier_=lambda pid: running)
    monkeypatch.setitem(sys.modules, "AppKit", appkit)
    monkeypatch.setitem(sys.modules, "ApplicationServices", fake)


def test_macos_press_uses_axpress_after_checking_the_window(monkeypatch):
    fake = FakeAX()
    _fake_mac(monkeypatch, fake)
    assert P.press_macos("button") == ("AXPress", "")
    assert fake.performed == [("button", "AXPress")]

    fake = FakeAX(title="1Password 8")
    _fake_mac(monkeypatch, fake)
    assert P.press_macos("button")[0] == "refused" and fake.performed == []
    fake = FakeAX()
    _fake_mac(monkeypatch, fake, app_name="Notes")
    assert P.press_macos("button", excluded=("Notes",))[0] == "refused" and fake.performed == []

    _fake_mac(monkeypatch, FakeAX(actions=("AXShowMenu",)))
    assert P.press_macos("button") == ("none", "")
    _fake_mac(monkeypatch, FakeAX(enabled=False))
    assert P.press_macos("button")[0] == "disabled"


@pytest.mark.asyncio
async def test_macos_press_outcomes_reach_the_caller(monkeypatch):
    element = accessibility.AccessibleElement("e1", "Button", "Save", "", 0, 0, 10, 10, True, False)
    P.remember("axm", ["button"])
    monkeypatch.setattr(accessibility, "CURRENT_PLATFORM", Platform.MACOS)
    window = accessibility.WindowIdentity(1, (0, 0, 9, 9))
    for outcome, expected in (("AXPress", "AXPress"), ("none", None)):
        monkeypatch.setattr(P, "press_macos", lambda handle, excluded, o=outcome: (o, ""))
        assert await accessibility.press_in_background(window, element, snapshot_id="axm") == expected
    monkeypatch.setattr(P, "press_macos", lambda handle, excluded: ("refused", "a password manager"))
    with pytest.raises(accessibility.AccessibilityRefused, match="password manager"):
        await accessibility.press_in_background(window, element, snapshot_id="axm")
    monkeypatch.setattr(P, "press_macos", lambda handle, excluded: ("failed", "AXPress returned -25204"))
    with pytest.raises(RuntimeError, match="-25204"):
        await accessibility.press_in_background(window, element, snapshot_id="axm")
    assert await accessibility.press_in_background(window, element, snapshot_id="unknown") is None
