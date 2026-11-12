"""Real runs of the XDG portal client for the ``platform-readers`` CI job.

    python linux_portal_check.py real   # inside a sway session with xdg-desktop-portal and xdg-desktop-portal-wlr
    python linux_portal_check.py fake   # on an empty session bus: a fake GlobalShortcuts portal served here

"real" asks the desktop for a screenshot through the Screenshot portal and checks a PNG came back; it also asks
for a global shortcut, which xdg-desktop-portal-wlr does not implement, and checks that Heliox says so plainly.
"fake" serves org.freedesktop.portal.Desktop on the bus with GlobalShortcuts, as GNOME 48 and KDE do, and runs
Heliox's client against it over real D-Bus messages: create a session, bind, receive an activation. Real desktops'
shortcut dialogs are not proven by this. Each check prints one CHECK line; the exit code counts failures.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.system import portal  # noqa: E402

failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)


def real() -> None:
    start = time.perf_counter()
    try:
        path = portal.screenshot(timeout=30)
        data = Path(path).read_bytes()
        check(
            "portal/screenshot",
            data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 1000,
            f"{len(data)} bytes in {(time.perf_counter() - start) * 1000:.0f} ms from {path}",
        )
    except Exception as exc:
        check("portal/screenshot", False, f"{type(exc).__name__}: {exc}")
    try:
        portal.bind_shortcuts({"toggle": ("Show or hide Heliox", "CTRL+space")}, timeout=10)
        check("portal/global-shortcuts-on-wlr", False, "bound, but xdg-desktop-portal-wlr has no GlobalShortcuts")
    except portal.PortalError as exc:
        check("portal/global-shortcuts-missing-is-explained", "does not offer the portal" in str(exc), str(exc))


def serve_fake(ready: threading.Event, stop: threading.Event) -> None:
    from jeepney import DBusAddress, HeaderFields, MessageType, new_method_return, new_signal
    from jeepney.bus_messages import message_bus
    from jeepney.io.blocking import open_dbus_connection

    conn = open_dbus_connection("SESSION")
    conn.send_and_get_reply(message_bus.RequestName(portal.PORTAL), timeout=5)
    ready.set()
    while not stop.is_set():
        try:
            msg = conn.receive(timeout=0.5)
        except TimeoutError:
            continue
        if msg.header.message_type != MessageType.method_call:
            continue
        member = msg.header.fields.get(HeaderFields.member)
        sender = msg.header.fields.get(HeaderFields.sender)
        options = msg.body[-1] if msg.body and isinstance(msg.body[-1], dict) else {}
        handle = options.get("handle_token", ("s", "t"))[1]
        request = portal.request_path(sender, handle)
        conn.send(new_method_return(msg, "o", (request,)))
        if member == "CreateSession":
            session = portal.session_path(sender, options["session_handle_token"][1])
            results = {"session_handle": ("s", session)}
        elif member == "BindShortcuts":
            session = msg.body[0]
            bound = [(key, {"trigger_description": ("s", "Ctrl+Space")}) for key, _props in msg.body[1]]
            results = {"shortcuts": ("a(sa{sv})", bound)}
        else:
            results = {}
        conn.send(new_signal(DBusAddress(request, interface=portal.REQUEST), "Response", "ua{sv}", (0, results)))
        if member == "BindShortcuts":
            time.sleep(0.3)
            conn.send(
                new_signal(
                    DBusAddress(portal.PORTAL_PATH, interface=portal.SHORTCUTS),
                    "Activated",
                    "osta{sv}",
                    (session, "toggle", int(time.time()), {}),
                )
            )
    conn.close()


def fake() -> None:
    ready, stop = threading.Event(), threading.Event()
    server = threading.Thread(target=serve_fake, args=(ready, stop), daemon=True)
    server.start()
    ready.wait(10)
    try:
        session = portal.bind_shortcuts({"toggle": ("Show or hide Heliox", "CTRL+space")}, timeout=10)
        check(
            "portal-fake/bind", session.shortcuts == {"toggle": "Ctrl+Space"}, f"{session.session} {session.shortcuts}"
        )
        check("portal-fake/activated", session.wait(5) == "toggle")
        session.close()
    except Exception as exc:
        check("portal-fake/bind", False, f"{type(exc).__name__}: {exc}")
    finally:
        stop.set()


if __name__ == "__main__":
    {"real": real, "fake": fake}[sys.argv[1] if len(sys.argv) > 1 else "real"]()
    sys.exit(failures)
