"""Accessibility snapshots of the front window on macOS (Accessibility API) and Linux (AT-SPI).

The same shape as the Windows snapshot in ``pilot.system.accessibility``: the front window's interactive elements
(role, name, position, size, enabled, focused) as the raw dicts ``parse_snapshot`` reads ("t", "n", "a", "x", "y",
"w", "h", "e", "f"), plus the window's identity so a click by ``element_ref`` is refused when another window is in
front or the window moved. Only roles, names and positions are read: a field's value is never read, and a password
field appears only as its role and label. Windows Heliox never reads (its own window, password managers, the
system credential stores, apps excluded from the timeline) are refused before anything in them is read.

Role names are mapped to the Windows control types the planner already knows (Button, Edit, CheckBox, ...), by
pure tables tested on every system.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable
from typing import Any

MAX_TREE_DEPTH = 12
MAX_NODE_VISITS = 2500

# macOS AXRole (and subrole) to the planner's control types.
MAC_ROLES = {
    "AXButton": "Button",
    "AXMenuButton": "Button",
    "AXDisclosureTriangle": "Button",
    "AXPopUpButton": "ComboBox",
    "AXComboBox": "ComboBox",
    "AXCheckBox": "CheckBox",
    "AXRadioButton": "RadioButton",
    "AXTextField": "Edit",
    "AXTextArea": "Edit",
    "AXSearchField": "Edit",
    "AXLink": "Hyperlink",
    "AXMenuItem": "MenuItem",
    "AXMenuBarItem": "MenuItem",
    "AXSlider": "Slider",
    "AXIncrementor": "Spinner",
    "AXCell": "ListItem",
    "AXRow": "ListItem",
    "AXTab": "TabItem",
    "AXStaticText": "Text",
}

# AT-SPI role names to the planner's control types.
ATSPI_ROLES = {
    "push button": "Button",
    "toggle button": "Button",
    "check box": "CheckBox",
    "check menu item": "MenuItem",
    "radio button": "RadioButton",
    "radio menu item": "MenuItem",
    "combo box": "ComboBox",
    "entry": "Edit",
    "text": "Edit",
    "password text": "Edit",
    "spin button": "Spinner",
    "link": "Hyperlink",
    "menu item": "MenuItem",
    "page tab": "TabItem",
    "list item": "ListItem",
    "table cell": "ListItem",
    "tree item": "TreeItem",
    "slider": "Slider",
    "scroll bar": "ScrollBar",
    "label": "Text",
}


def control_type(role: str, table: dict[str, str]) -> str | None:
    """The planner's control type for a platform role, or None when the element is not worth listing."""
    return table.get(role)


def keep(kind: str | None, name: str) -> bool:
    """Interactive elements always; text only when it has a name (it labels what is around it)."""
    return kind is not None and (kind != "Text" or bool(name.strip()))


def window_key(text: str) -> int:
    """A stable number for a window from its platform identity (Windows uses the real handle)."""
    return (zlib.crc32(text.encode("utf-8")) & 0x3FFFFFFF) | 0x40000000


def raw_element(kind: str, name: str, box: tuple[int, int, int, int], enabled: bool, focused: bool) -> dict:
    x, y, w, h = box
    return {"t": kind, "n": name, "a": "", "x": x, "y": y, "w": w, "h": h, "e": enabled, "f": focused}


def refuse_window(app: str, title: str, excluded: Iterable[str]) -> str:
    """Why this window must not be read, or "" when it may be."""
    from pilot.live.focused_text_policy import never_read
    from pilot.workflows.manual_workflow_hook import _is_heliox_window_title

    if never_read(app, title, excluded, is_heliox=_is_heliox_window_title(title)):
        return "Heliox does not read this window: it is Heliox itself, a password manager, or an app you excluded."
    return ""


# ── macOS ──


def capture_macos(
    max_elements: int, excluded: Iterable[str] = ()
) -> tuple[dict, tuple[int, tuple[int, int, int, int]]]:
    """(payload, (window key, rect)) for the front window, through the Accessibility API. Raises RuntimeError."""
    import AppKit
    import ApplicationServices as ax  # noqa: N813

    if not ax.AXIsProcessTrusted():
        raise RuntimeError(
            "Heliox needs Accessibility access to list a window's controls: System Settings > Privacy & Security > "
            "Accessibility, turn on Heliox-OS."
        )

    def attr(element: Any, name: str) -> Any:
        err, value = ax.AXUIElementCopyAttributeValue(element, name, None)
        return value if err == 0 else None

    def point(value: Any, kind: int) -> tuple[float, float]:
        if value is None:
            return (0.0, 0.0)
        ok, result = ax.AXValueGetValue(value, kind, None)
        if not ok:
            return (0.0, 0.0)
        return (
            (float(result.x), float(result.y))
            if kind == ax.kAXValueCGPointType
            else (
                float(result.width),
                float(result.height),
            )
        )

    def box(element: Any) -> tuple[int, int, int, int]:
        x, y = point(attr(element, "AXPosition"), ax.kAXValueCGPointType)
        w, h = point(attr(element, "AXSize"), ax.kAXValueCGSizeType)
        return int(x), int(y), int(w), int(h)

    system = ax.AXUIElementCreateSystemWide()
    ax.AXUIElementSetMessagingTimeout(system, 0.5)
    app = attr(system, "AXFocusedApplication")
    window = attr(app, "AXFocusedWindow") if app is not None else None
    if window is None:
        raise RuntimeError("No window is in front to read.")
    err, pid = ax.AXUIElementGetPid(window, None)
    running = AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
    app_name = str(running.localizedName() or "") if running is not None else ""
    title = str(attr(window, "AXTitle") or "")
    reason = refuse_window(app_name, title, excluded)
    if reason:
        raise RuntimeError(reason)
    elements: list[dict] = []
    queue: list[tuple[Any, int]] = [(window, 0)]
    visits = 0
    while queue and len(elements) < max_elements and visits < MAX_NODE_VISITS:
        element, depth = queue.pop(0)
        visits += 1
        role = str(attr(element, "AXRole") or "")
        subrole = str(attr(element, "AXSubrole") or "")
        kind = control_type(role, MAC_ROLES)
        if role == "AXRadioButton" and subrole == "AXTabButton":
            kind = "TabItem"
        # The label, never the value of a field: AXValue is read only for static text, which is a label itself.
        name = str(attr(element, "AXTitle") or attr(element, "AXDescription") or "")
        if not name and role == "AXStaticText":
            name = str(attr(element, "AXValue") or "")
        bounds = box(element)
        if element is not window and keep(kind, name) and bounds[2] > 0 and bounds[3] > 0:
            elements.append(
                raw_element(
                    kind or "",
                    name,
                    bounds,
                    bool(attr(element, "AXEnabled") is not False),
                    bool(attr(element, "AXFocused")),
                )
            )
        if depth < MAX_TREE_DEPTH:
            queue.extend((child, depth + 1) for child in (attr(element, "AXChildren") or []))
    import CoreFoundation

    wx, wy, ww, wh = box(window)
    key = window_key(f"{pid}:{CoreFoundation.CFHash(window)}")
    payload = {"title": title, "process": int(pid), "elements": elements, "truncated": bool(queue)}
    return payload, (key, (wx, wy, wx + ww, wy + wh))


def front_window_macos() -> tuple[int, tuple[int, int, int, int]] | None:
    try:
        import ApplicationServices as ax  # noqa: N813
        import CoreFoundation
    except ImportError:
        return None
    if not ax.AXIsProcessTrusted():
        return None
    system = ax.AXUIElementCreateSystemWide()
    err, app = ax.AXUIElementCopyAttributeValue(system, "AXFocusedApplication", None)
    if err != 0 or app is None:
        return None
    err, window = ax.AXUIElementCopyAttributeValue(app, "AXFocusedWindow", None)
    if err != 0 or window is None:
        return None
    err, pid = ax.AXUIElementGetPid(window, None)
    rect = [0, 0, 0, 0]
    for name, kind, slots in (
        ("AXPosition", ax.kAXValueCGPointType, (0, 1)),
        ("AXSize", ax.kAXValueCGSizeType, (2, 3)),
    ):
        err, value = ax.AXUIElementCopyAttributeValue(window, name, None)
        if err == 0 and value is not None:
            ok, result = ax.AXValueGetValue(value, kind, None)
            if ok:
                pair = (result.x, result.y) if slots == (0, 1) else (result.width, result.height)
                rect[slots[0]], rect[slots[1]] = int(pair[0]), int(pair[1])
    key = window_key(f"{pid}:{CoreFoundation.CFHash(window)}")
    return key, (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])


# ── Linux ──


def _atspi_window(client: Any) -> Any:
    from pilot.live import focused_text_linux as L

    ref = client.focused()
    if ref is None:
        raise RuntimeError("No window is in front to read (nothing has keyboard focus on the accessibility bus).")
    for _ in range(40):
        if client._role(ref) in L.WINDOW_ROLES:
            return ref
        bus, path = client._get(ref, L.ACCESSIBLE, "Parent")
        ref = (str(bus), str(path))
    raise RuntimeError("The focused element's window was not found.")


def _extents(client: Any, ref: Any) -> tuple[int, int, int, int]:
    try:
        x, y, w, h = client._t.call(ref[0], ref[1], "org.a11y.atspi.Component", "GetExtents", "u", (0,))[0]
        return int(x), int(y), int(w), int(h)
    except Exception:
        return (0, 0, 0, 0)


def capture_linux(
    max_elements: int, excluded: Iterable[str] = ()
) -> tuple[dict, tuple[int, tuple[int, int, int, int]]]:
    """(payload, (window key, rect)) for the window that has focus, through AT-SPI. Raises RuntimeError."""
    from pilot.live import focused_text_linux as L

    with L.SHARED.lock:
        client = L.SHARED.get()
        if client is None:
            raise RuntimeError(L.SHARED.error or "The accessibility bus is not reachable.")
        window = _atspi_window(client)
        title = str(client._get(window, L.ACCESSIBLE, "Name") or "")
        app_name = L.comm_name(client.pid(window[0]))
        reason = refuse_window(app_name, title, excluded)
        if reason:
            raise RuntimeError(reason)
        elements: list[dict] = []
        queue: list[tuple[Any, int]] = [(window, 0)]
        visits = 0
        while queue and len(elements) < max_elements and visits < MAX_NODE_VISITS:
            ref, depth = queue.pop(0)
            visits += 1
            try:
                state = client._state(ref)
                role = client._role(ref)
                name = str(client._get(ref, L.ACCESSIBLE, "Name") or "")
            except Exception:
                continue
            if L.STATE_SHOWING not in state and ref != window:
                continue
            kind = control_type(role, ATSPI_ROLES)
            box = _extents(client, ref)
            if ref != window and keep(kind, name) and box[2] > 0 and box[3] > 0:
                elements.append(raw_element(kind or "", name, box, 8 in state or 24 in state, L.STATE_FOCUSED in state))
            if depth < MAX_TREE_DEPTH and L.STATE_MANAGES_DESCENDANTS not in state:
                try:
                    queue.extend((child, depth + 1) for child in client._children(ref))
                except Exception:
                    pass
        wx, wy, ww, wh = _extents(client, window)
        payload = {"title": title, "process": client.pid(window[0]), "elements": elements, "truncated": bool(queue)}
        return payload, (window_key(f"{window[0]}{window[1]}"), (wx, wy, wx + ww, wy + wh))


def front_window_linux() -> tuple[int, tuple[int, int, int, int]] | None:
    from pilot.live import focused_text_linux as L

    if not L.SHARED.lock.acquire(timeout=1.0):
        return None
    try:
        client = L.SHARED.get()
        if client is None:
            return None
        try:
            window = _atspi_window(client)
        except Exception:
            return None
        x, y, w, h = _extents(client, window)
        return window_key(f"{window[0]}{window[1]}"), (x, y, x + w, y + h)
    finally:
        L.SHARED.lock.release()
