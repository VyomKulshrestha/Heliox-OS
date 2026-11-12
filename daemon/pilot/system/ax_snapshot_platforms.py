"""Accessibility snapshots of the front window on macOS (Accessibility API) and Linux (AT-SPI).

The same shape as the Windows snapshot in ``pilot.system.accessibility``: the front window's interactive elements
(role, name, position, size, enabled, focused) as the raw dicts ``parse_snapshot`` reads ("t", "n", "a", "x", "y",
"w", "h", "e", "f"), plus the window's identity so a click by ``element_ref`` is refused when another window is in
front or the window moved. Only roles, names and positions are read: a field's value is never read, and a password
field appears only as its role and label. Windows Heliox never reads (its own window, password managers, the
system credential stores, apps excluded from the timeline) are refused before anything in them is read.

Role names are mapped to the Windows control types the planner already knows (Button, Edit, CheckBox, ...), by
pure tables tested on every system.

A control from a snapshot can be pressed without the mouse: AXPress on macOS, the AT-SPI Action interface on
Linux, when the control offers it. The press checks the control's window again first, since a window can become
one Heliox never reads after the snapshot was taken (a title change, a password manager taking its place), and
then nothing is pressed or clicked. Each press returns an outcome word, as the Windows script does: the action
used, "none" (the control offers no press; the caller may click it), "missing", "disabled", "refused" or "failed".
"""

from __future__ import annotations

import zlib
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

MAX_TREE_DEPTH = 12
MAX_NODE_VISITS = 2500
MAX_REMEMBERED_SNAPSHOTS = 8

# AT-SPI action names that press a control, best first (GTK buttons offer "click", Qt "Press", menus "activate").
PRESS_ACTIONS = ("click", "press", "activate", "toggle", "jump")

# kAXErrorInvalidUIElement: the control is gone.
AX_INVALID_ELEMENT = -25202

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
    # GTK 4 names its roles after ARIA in GetRoleName ("button", "text box"; CI on 008f1a0).
    "button": "Button",
    "text box": "Edit",
    "search box": "Edit",
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


# ── the controls of recent snapshots, for pressing them ──

_handles: OrderedDict[str, list[Any]] = OrderedDict()


def remember(snapshot_id: str, handles: list[Any], capacity: int = MAX_REMEMBERED_SNAPSHOTS) -> None:
    """Keep a snapshot's native controls (AXUIElements, AT-SPI refs) in element order, for the last few snapshots."""
    _handles[snapshot_id] = list(handles)
    _handles.move_to_end(snapshot_id)
    while len(_handles) > capacity:
        _handles.popitem(last=False)


def handle_for(snapshot_id: str, ref: str) -> Any | None:
    """The native control behind ``ref`` ("e3") in a remembered snapshot, or None."""
    handles = _handles.get(snapshot_id)
    if not handles or not ref.startswith("e") or not ref[1:].isdigit():
        return None
    index = int(ref[1:]) - 1
    return handles[index] if 0 <= index < len(handles) else None


def choose_action(names: Iterable[str]) -> int | None:
    """The index of the action that presses a control, or None when none of its actions does. Pure."""
    folded = [str(name).strip().casefold() for name in names]
    for wanted in PRESS_ACTIONS:
        if wanted in folded:
            return folded.index(wanted)
    return None


# ── macOS ──

# Where a control's label is, in order. A field's AXValue is never one of them; a placeholder is the app's own
# hint text, not what the person typed. (CI, 12 November: with AXTitle and AXDescription alone, two text fields
# and three buttons of a real window came back unnamed.)
MAC_LABEL_ATTRIBUTES = ("AXTitle", "AXDescription", "AXPlaceholderValue", "AXHelp")


# The window's own buttons say what they are only in their subrole (CI on 0a8edad: two bare "Button" entries).
MAC_WINDOW_BUTTONS = {
    "AXCloseButton": "close window",
    "AXMinimizeButton": "minimise window",
    "AXZoomButton": "zoom window",
    "AXFullScreenButton": "full screen",
}


def mac_label(get: Any, role: str) -> str:
    """A control's label from ``get(attribute)``, never a field's value; static text is a label itself, and a
    window's own button is named by its subrole. Pure."""
    for key in MAC_LABEL_ATTRIBUTES:
        value = get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    subrole = get("AXSubrole")
    if isinstance(subrole, str) and subrole in MAC_WINDOW_BUTTONS:
        return MAC_WINDOW_BUTTONS[subrole]
    if role == "AXStaticText":
        value = get("AXValue")
        return " ".join(value.split()) if isinstance(value, str) else ""
    return ""


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
    handles: list[Any] = []
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
        name = mac_label(lambda key, element=element: attr(element, key), role)
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
            handles.append(element)
        if depth < MAX_TREE_DEPTH:
            queue.extend((child, depth + 1) for child in (attr(element, "AXChildren") or []))
    import CoreFoundation

    wx, wy, ww, wh = box(window)
    key = window_key(f"{pid}:{CoreFoundation.CFHash(window)}")
    payload = {"title": title, "process": int(pid), "elements": elements, "truncated": bool(queue)}
    payload["_handles"] = handles  # taken out before the payload is parsed; see accessibility.capture_snapshot
    return payload, (key, (wx, wy, wx + ww, wy + wh))


def press_macos(element: Any, excluded: Iterable[str] = ()) -> tuple[str, str]:
    """AXPress on a control from a snapshot, after checking its window again. (outcome, detail)."""
    import AppKit
    import ApplicationServices as ax  # noqa: N813

    if not ax.AXIsProcessTrusted():
        return "failed", "Heliox needs Accessibility access to press a control."
    err, window = ax.AXUIElementCopyAttributeValue(element, "AXWindow", None)
    if err == AX_INVALID_ELEMENT:
        return "missing", ""
    if err != 0 or window is None:
        # A control whose window cannot be named is not pressed: its window may be one Heliox never reads.
        return "refused", "Heliox cannot tell which window this control is in, so it does not press it."
    err, pid = ax.AXUIElementGetPid(window, None)
    running = AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid) if err == 0 else None
    app_name = str(running.localizedName() or "") if running is not None else ""
    err, title = ax.AXUIElementCopyAttributeValue(window, "AXTitle", None)
    reason = refuse_window(app_name, str(title or "") if err == 0 else "", excluded)
    if reason:
        return "refused", reason
    err, enabled = ax.AXUIElementCopyAttributeValue(element, "AXEnabled", None)
    if err == 0 and enabled is False:
        return "disabled", ""
    err, names = ax.AXUIElementCopyActionNames(element, None)
    if err == AX_INVALID_ELEMENT:
        return "missing", ""
    if err != 0 or "AXPress" not in list(names or []):
        return "none", ""
    err = ax.AXUIElementPerformAction(element, "AXPress")
    if err == 0:
        return "AXPress", ""
    if err == AX_INVALID_ELEMENT:
        return "missing", ""
    # Cannot-complete often means the app ran the action and is now busy (a sheet opened): never click again.
    return "failed", f"AXPress returned {err}"


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


def _atspi_window(client: Any, start: Any = None) -> Any:
    from pilot.live import focused_text_linux as L

    ref = start if start is not None else client.focused()
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
        handles: list[Any] = []
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
                handles.append(ref)
            if depth < MAX_TREE_DEPTH and L.STATE_MANAGES_DESCENDANTS not in state:
                try:
                    queue.extend((child, depth + 1) for child in client._children(ref))
                except Exception:
                    pass
        wx, wy, ww, wh = _extents(client, window)
        payload = {"title": title, "process": client.pid(window[0]), "elements": elements, "truncated": bool(queue)}
        payload["_handles"] = handles  # taken out before the payload is parsed; see accessibility.capture_snapshot
        return payload, (window_key(f"{window[0]}{window[1]}"), (wx, wy, wx + ww, wy + wh))


ATSPI_ACTION = "org.a11y.atspi.Action"


def press_linux(ref: Any, excluded: Iterable[str] = ()) -> tuple[str, str]:
    """DoAction on a control from a snapshot, after checking its window again. (outcome, detail)."""
    from pilot.live import focused_text_linux as L

    if not L.SHARED.lock.acquire(timeout=2.0):
        return "failed", "The accessibility bus is busy."
    try:
        client = L.SHARED.get()
        if client is None:
            return "failed", L.SHARED.error or "The accessibility bus is not reachable."
        try:
            state = client._state(ref)
            window = _atspi_window(client, ref)
            title = str(client._get(window, L.ACCESSIBLE, "Name") or "")
            app_name = L.comm_name(client.pid(window[0]))
        except Exception as error:
            if L.gone_from_error(str(error)):
                return "missing", ""
            # A control whose window cannot be named is not pressed: its window may be one Heliox never reads.
            return "refused", "Heliox cannot tell which window this control is in, so it does not press it."
        reason = refuse_window(app_name, title, excluded)
        if reason:
            return "refused", reason
        if client.is_password(ref):
            # A password field, or the text inside one (GTK 4 marks it only in the entry's numeric role).
            return "refused", "Heliox does not act inside a password field."
        if L.STATE_DEFUNCT in state:
            return "missing", ""
        if not (8 in state or 24 in state):  # ENABLED or SENSITIVE
            return "disabled", ""
        try:
            count = int(client._get(ref, ATSPI_ACTION, "NActions") or 0)
            names = [str(client._t.call(ref[0], ref[1], ATSPI_ACTION, "GetName", "i", (i,))[0]) for i in range(count)]
        except Exception as error:
            return ("missing", "") if L.gone_from_error(str(error)) else ("none", "")
        index = choose_action(names)
        if index is None:
            return "none", ""
        try:
            done = bool(client._t.call(ref[0], ref[1], ATSPI_ACTION, "DoAction", "i", (index,))[0])
        except Exception as error:
            if L.gone_from_error(str(error)):
                return "missing", ""
            return "failed", f"DoAction failed: {type(error).__name__}"
        return (names[index], "") if done else ("failed", f"DoAction({names[index]!r}) returned false")
    finally:
        L.SHARED.lock.release()


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
