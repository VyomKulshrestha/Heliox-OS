"""The XDG portal client (Wayland screenshots and global shortcuts) against a fake session bus, on every system."""

from __future__ import annotations

import pytest

from pilot.system import portal
from pilot.system.portal import PortalError


class FakePortal(portal.Transport):
    unique_name = ":1.42"

    def __init__(self, responses: dict[str, tuple], missing: bool = False) -> None:
        self.responses = responses
        self.missing = missing
        self.watched: list[tuple[str, str, str]] = []
        self.calls: list[tuple[str, str, str, tuple, str]] = []
        self.pending: list[tuple[str, str, str, tuple]] = []

    def call(self, interface, method, signature, body, path=portal.PORTAL_PATH):
        if self.missing:
            raise PortalError("org.freedesktop.DBus.Error.UnknownMethod: No such interface")
        self.calls.append((interface, method, signature, body, path))
        options = body[-1] if body and isinstance(body[-1], dict) else {}
        handle = options.get("handle_token", ("s", ""))[1]
        if method in self.responses:
            request = portal.request_path(self.unique_name, handle)
            # The portal answers only on a path the caller subscribed to before calling.
            assert (portal.REQUEST, "Response", request) in self.watched
            self.pending.append((portal.REQUEST, "Response", request, self.responses[method]))
        return (portal.request_path(self.unique_name, handle),)

    def watch(self, interface, member, path):
        self.watched.append((interface, member, path))

    def next_signal(self, interface, member, path, timeout):
        for item in self.pending:
            if item[0] == interface and item[1] == member and (path is None or item[2] == path):
                self.pending.remove(item)
                return item[3]
        return None


def test_paths_messages_and_wayland_detection():
    assert portal.request_path(":1.42", "heliox_ab") == "/org/freedesktop/portal/desktop/request/1_42/heliox_ab"
    assert portal.session_path(":1.7", "t") == "/org/freedesktop/portal/desktop/session/1_7/t"
    assert portal.file_from_uri("file:///tmp/Screenshot%20from%20now.png") == "/tmp/Screenshot from now.png"
    with pytest.raises(PortalError):
        portal.file_from_uri("https://example.com/a.png")
    assert "answer was no" in portal.explain(1, "take a screenshot")
    refused = portal.explain(2, "take a screenshot")
    assert "allow screenshots for Heliox when it asks" in refused and "grim" in refused
    assert "portal response 2" in portal.explain(2, "bind shortcuts")
    assert "xdg-desktop-portal" in portal.no_portal("take a screenshot", "org.freedesktop.DBus.Error.ServiceUnknown")
    assert portal.wayland_session({"WAYLAND_DISPLAY": "wayland-0"})
    assert portal.wayland_session({"XDG_SESSION_TYPE": "wayland"})
    assert not portal.wayland_session({"DISPLAY": ":0", "XDG_SESSION_TYPE": "x11"})
    assert portal.tauri_to_xdg_trigger("Ctrl+Space") == "CTRL+space"
    assert portal.tauri_to_xdg_trigger("Alt+Shift+H") == "ALT+SHIFT+h"
    assert portal.tauri_to_xdg_trigger("Super+Space") == "LOGO+space"


def test_screenshot_returns_the_file_the_desktop_wrote():
    fake = FakePortal({"Screenshot": (0, {"uri": ("s", "file:///tmp/shot.png")})})
    assert portal.screenshot(fake) == "/tmp/shot.png"
    interface, method, signature, body, _path = fake.calls[0]
    assert (interface, method, signature) == (portal.SCREENSHOT, "Screenshot", "sa{sv}")
    assert body[1]["interactive"] == ("b", False)


def test_screenshot_refused_missing_or_silent_says_why():
    with pytest.raises(PortalError, match="answer was no"):
        portal.screenshot(FakePortal({"Screenshot": (1, {})}))
    with pytest.raises(PortalError, match="does not offer the portal"):
        portal.screenshot(FakePortal({}, missing=True))
    with pytest.raises(PortalError, match="did not answer"):
        portal.screenshot(FakePortal({}), timeout=0)


def test_global_shortcut_session_binds_and_reports_activations():
    fake = FakePortal(
        {
            "CreateSession": (0, {"session_handle": ("s", "/org/freedesktop/portal/desktop/session/1_42/s1")}),
            "BindShortcuts": (
                0,
                {"shortcuts": ("a(sa{sv})", [("toggle", {"trigger_description": ("s", "Ctrl+Space")})])},
            ),
        }
    )
    dialogs = []
    session = portal.bind_shortcuts(
        {"toggle": ("Show or hide Heliox", "CTRL+space")}, fake, on_dialog=lambda: dialogs.append(1)
    )
    assert session.session.endswith("/s1") and session.shortcuts == {"toggle": "Ctrl+Space"} and dialogs == [1]
    bind = next(call for call in fake.calls if call[1] == "BindShortcuts")
    assert bind[2] == "oa(sa{sv})sa{sv}" and bind[3][1][0][0] == "toggle"
    assert bind[3][1][0][1]["preferred_trigger"] == ("s", "CTRL+space")
    fake.pending.append((portal.SHORTCUTS, "Activated", portal.PORTAL_PATH, (session.session, "toggle", 1, {})))
    assert session.wait(1) == "toggle"
    fake.pending.append((portal.SHORTCUTS, "Activated", portal.PORTAL_PATH, ("/other/session", "toggle", 2, {})))
    assert session.wait(1) is None  # another app's session


def test_global_shortcuts_missing_on_this_desktop():
    with pytest.raises(PortalError, match="does not offer the portal"):
        portal.bind_shortcuts({"toggle": ("x", "CTRL+space")}, FakePortal({}, missing=True))


def test_capture_stops_asking_after_the_person_says_no(monkeypatch, tmp_path):
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"png-bytes")
    asked = []

    def first_yes_then_no():
        asked.append(1)
        if len(asked) == 1:
            return str(shot)
        raise PortalError(portal.explain(1, "take a screenshot"))

    monkeypatch.setattr(portal, "screenshot", first_yes_then_no)
    monkeypatch.setattr(portal, "_screenshot_refused", "")
    assert portal.capture_png() == b"png-bytes" and not shot.exists()
    assert portal.capture_png() is None
    assert portal.capture_png() is None and len(asked) == 2  # not asked again after "no"


def test_vision_uses_the_portal_on_wayland(monkeypatch):
    import asyncio

    from pilot.system import vision

    monkeypatch.setattr(vision.sys, "platform", "linux")
    monkeypatch.setattr(portal, "wayland_session", lambda env=None: True)
    monkeypatch.setattr(portal, "capture_png", lambda region=None: b"from-portal")
    assert asyncio.run(vision._capture_screenshot_bytes()) == b"from-portal"


def test_wayland_shortcut_service_reports_activations_and_missing_portals(monkeypatch):
    import threading

    from pilot.system import wayland_shortcut

    sent: list[tuple[str, dict]] = []
    done = threading.Event()

    class Session:
        session = "/s"
        shortcuts = {"toggle": "Ctrl+Space"}
        calls = 0

        def wait(self, timeout):
            Session.calls += 1
            if Session.calls == 1:
                return "toggle"
            done.set()
            return None

        def close(self):
            pass

    monkeypatch.setattr(portal, "bind_shortcuts", lambda wanted, **_kw: Session())
    service = wayland_shortcut.WaylandShortcut(lambda method, params: sent.append((method, params))).start()
    assert done.wait(5)
    service.stop()
    assert ("global_shortcut", {"id": "toggle"}) in sent
    assert service.status == "ready" and service.keys == "Ctrl+Space"

    def missing(wanted, **_kw):
        raise PortalError(portal.no_portal("use a global shortcut", "org.freedesktop.DBus.Error.ServiceUnknown"))

    monkeypatch.setattr(portal, "bind_shortcuts", missing)
    quiet = wayland_shortcut.WaylandShortcut(lambda *_a: None)
    quiet._run()
    assert quiet.status == "unavailable" and "xdg-desktop-portal" in quiet.message


def test_wayland_shortcut_starts_only_in_a_linux_wayland_session(monkeypatch):
    from pilot.system import wayland_shortcut

    monkeypatch.setattr(portal, "wayland_session", lambda env=None: False)
    assert wayland_shortcut.start_if_wayland(lambda *_a: None) is None


def test_screenshot_portal_needs_an_access_backend_and_says_so():
    # xdg-desktop-portal 1.18 exports Screenshot only when some backend implements Access (the consent dialog).
    message = portal.no_portal("take a screenshot", "org.freedesktop.DBus.Error.UnknownMethod")
    assert "xdg-desktop-portal-gtk" in message and "consent dialog" in message
    assert "xdg-desktop-portal-gtk" not in portal.no_portal("use a global shortcut", "ServiceUnknown")
    names = ["org.freedesktop.impl.portal.desktop.wlr", "org.freedesktop.portal.Desktop"]
    assert portal.missing_access_backend(names, ["org.freedesktop.portal.ScreenCast"])
    assert not portal.missing_access_backend(names, [portal.SCREENSHOT])
    assert not portal.missing_access_backend(["org.freedesktop.portal.Desktop"], [])
