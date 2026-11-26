"""Mouse & Keyboard Control — full desktop automation.

Cross-platform input simulation using pyautogui.
Click, type, drag, scroll, hotkeys — control ANY application.
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger("pilot.system.input_control")


def _ensure_pyautogui():
    """Lazy import pyautogui with safety settings."""
    import pyautogui

    pyautogui.FAILSAFE = True  # Move mouse to corner to abort
    pyautogui.PAUSE = 0.05  # Small pause between actions
    return pyautogui


def _real_input(pag: object) -> bool:
    """Whether the input went to the real desktop through pyautogui (not a stand-in), so its effect can be read
    back right after the step (``pilot.system.desktop_observe``)."""
    return getattr(pag, "__name__", "") == "pyautogui"


def paste_keys(platform: str | None = None) -> tuple[str, str]:
    """The paste shortcut: command+V on macOS, where control+V pastes nothing in most apps; control+V elsewhere."""
    import sys

    return ("command", "v") if (platform or sys.platform) == "darwin" else ("ctrl", "v")


# ── Mouse ────────────────────────────────────────────────────────────


# pyautogui reaches X11 through python-xlib, which refuses to connect when no X authority file can be found
# ($XAUTHORITY unset and no ~/.Xauthority), even when the X server asks for none (CI on 008f1a0, under Xvfb). A
# person meets it when Heliox's daemon runs outside the desktop session's environment (a systemd user service
# started without XAUTHORITY, for one). libX11, which xdotool uses, reads the same settings without that failure.
X11_NO_AUTHORITY = (
    "Heliox could not reach the X display to click: python-xlib found no X authority file ($XAUTHORITY is not set "
    "and ~/.Xauthority does not exist). Install xdotool, or start Heliox from your desktop session so it gets the "
    "session's XAUTHORITY."
)
XDOTOOL_BUTTONS = {"left": "1", "middle": "2", "right": "3"}


def x_display_unreachable(error: BaseException) -> bool:
    """Whether pyautogui failed for want of an X connection (no authority file, no display). Pure."""
    text = f"{type(error).__name__}: {error}"
    return any(mark in text for mark in ("XauthError", "Xauthority", "DisplayConnectionError", "DisplayNameError"))


def x11_session(platform: str | None = None, env: dict[str, str] | None = None) -> bool:
    import os
    import sys

    env = dict(os.environ) if env is None else env
    return (
        (platform or sys.platform).startswith("linux") and bool(env.get("DISPLAY")) and not env.get("WAYLAND_DISPLAY")
    )


async def _xdotool_click(x: int, y: int, button: str, clicks: int, interval: float, cause: BaseException) -> str:
    import shutil

    from pilot.system.platform_detect import run_command

    if not shutil.which("xdotool"):
        raise RuntimeError(f"{X11_NO_AUTHORITY} ({type(cause).__name__})") from cause
    command = ["xdotool", "mousemove", "--sync", str(int(x)), str(int(y)), "click"]
    command += ["--repeat", str(max(1, int(clicks))), "--delay", str(max(1, int(interval * 1000)))]
    command.append(XDOTOOL_BUTTONS.get(button, "1"))
    code, _out, err = await run_command(command, timeout=10)
    if code != 0:
        raise RuntimeError(f"{X11_NO_AUTHORITY} xdotool could not click either: {(err or '').strip()[:160]}")
    logger.info("Clicked through xdotool: pyautogui could not reach the X display (%s)", type(cause).__name__)
    return f"Clicked ({button}, {clicks}x) at ({x}, {y}) through xdotool"


async def mouse_click(
    x: int,
    y: int,
    button: str = "left",
    clicks: int = 1,
    interval: float = 0.1,
) -> str:
    """Click at screen coordinates. On X11, when pyautogui cannot reach the display, xdotool clicks instead."""
    try:
        pag = _ensure_pyautogui()

        from pilot.system import desktop_observe

        watched = _real_input(pag) and desktop_observe.current_step() is not None

        def _do() -> tuple[int, int] | None:
            if not watched:
                pag.click(x=x, y=y, button=button, clicks=clicks, interval=interval)
                return None
            # Read the pointer right at the event, between the move and the press: a click whose pointer is not at
            # its target is not made. A pointer moved after the press is the person's and only "not verified".
            pag.moveTo(x, y)
            off = desktop_observe.pointer_off_target_now((x, y))
            if off is None:
                pag.click(button=button, clicks=clicks, interval=interval)
            return off

        off_target = await asyncio.to_thread(_do)
    except Exception as error:
        if not (x11_session() and x_display_unreachable(error)):
            raise
        return await _xdotool_click(x, y, button, clicks, interval, error)
    if off_target is not None:
        desktop_observe.refuse_click(off_target, (x, y))
    if _real_input(pag):
        from pilot.system.desktop_observe import confirm_pointer

        await confirm_pointer((x, y), released=(button,))
    return f"Clicked ({button}, {clicks}x) at ({x}, {y})"


async def mouse_double_click(x: int, y: int) -> str:
    """Double-click at coordinates."""
    return await mouse_click(x, y, clicks=2)


async def mouse_right_click(x: int, y: int) -> str:
    """Right-click at coordinates."""
    return await mouse_click(x, y, button="right")


async def mouse_move(
    x: int,
    y: int,
    duration: float = 0.3,
    relative: bool = False,
) -> str:
    """Move mouse to coordinates (or relative offset)."""
    pag = _ensure_pyautogui()
    target: tuple[int, int] | None = (x, y)
    if relative:
        target = None
        if _real_input(pag):
            from pilot.system.desktop_observe import current_step, pointer

            if current_step() is not None:
                try:
                    start = await pointer()
                    target = (start[0] + x, start[1] + y)
                except Exception:
                    target = None

    def _do():
        if relative:
            pag.moveRel(x, y, duration=duration)
        else:
            pag.moveTo(x, y, duration=duration)

    await asyncio.to_thread(_do)
    if target is not None and _real_input(pag):
        from pilot.system.desktop_observe import confirm_pointer

        await confirm_pointer(target)
    mode = "relative" if relative else "absolute"
    return f"Moved mouse to ({x}, {y}) [{mode}]"


async def mouse_drag(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    duration: float = 0.5,
    button: str = "left",
) -> str:
    """Drag from one position to another."""
    pag = _ensure_pyautogui()

    def _do():
        pag.moveTo(start_x, start_y, duration=0.1)
        pag.drag(end_x - start_x, end_y - start_y, duration=duration, button=button)

    await asyncio.to_thread(_do)
    if _real_input(pag):
        from pilot.system.desktop_observe import confirm_pointer

        await confirm_pointer((end_x, end_y), released=(button,))
    return f"Dragged from ({start_x},{start_y}) to ({end_x},{end_y})"


async def mouse_scroll(
    amount: int,
    x: int | None = None,
    y: int | None = None,
    horizontal: bool = False,
) -> str:
    """Scroll the mouse wheel. Positive = up, negative = down."""
    pag = _ensure_pyautogui()

    def _do():
        if x is not None and y is not None:
            pag.moveTo(x, y)
        if horizontal:
            pag.hscroll(amount)
        else:
            pag.scroll(amount)

    await asyncio.to_thread(_do)
    if x is not None and y is not None and _real_input(pag):
        from pilot.system.desktop_observe import confirm_pointer

        await confirm_pointer((x, y))
    direction = "horizontal" if horizontal else "vertical"
    return f"Scrolled {direction} by {amount}"


async def mouse_position() -> str:
    """Get current mouse position."""
    pag = _ensure_pyautogui()
    pos = pag.position()
    return f"Mouse at ({pos.x}, {pos.y})"


# ── Keyboard ─────────────────────────────────────────────────────────


async def keyboard_type(
    text: str,
    interval: float = 0.03,
    *,
    target_window: str | None = None,
) -> str:
    """Enter exact text at the current cursor position.

    Key-by-key synthesis is affected by Caps Lock and the active keyboard
    layout. Prefer a clipboard paste for deterministic case and Unicode, then
    restore the user's prior clipboard value immediately afterward.
    """
    pag = _ensure_pyautogui()
    observed = _real_input(pag)

    if target_window:
        try:
            from pilot.system.window_mgr import window_set_text

            await window_set_text(text, title=target_window)
            if observed:
                from pilot.system.desktop_observe import confirm_typing

                await confirm_typing(text, None, target_window=target_window, background=True)
            preview = text[:80] + "..." if len(text) > 80 else text
            return f"Typed exactly in {target_window}: {preview}"
        except Exception as error:
            logger.info(
                "Background text entry unavailable for %r; attempting verified foreground entry: %s",
                target_window,
                error,
            )
            from pilot.system.window_mgr import window_focus

            # Fail closed if Windows refuses to re-acquire the requested app;
            # never paste into whichever unrelated window happens to be active.
            await window_focus(title=target_window)

    before = None
    if observed:
        from pilot.system.desktop_observe import refuse_unless_in_front, typing_started

        # A named window that is not in front gets no key at all.
        before = await refuse_unless_in_front(target_window) if target_window else await typing_started()
    typed = await _type_into_focus(pag, text, interval)
    if observed:
        from pilot.system.desktop_observe import confirm_typing

        await confirm_typing(text, before, target_window=target_window, background=False)
    return typed


async def _type_into_focus(pag, text: str, interval: float) -> str:
    try:
        import pyperclip

        keys = paste_keys()

        def _paste_exact() -> None:
            previous = pyperclip.paste()
            try:
                pyperclip.copy(text)
                pag.hotkey(*keys)
                # Give the foreground application time to consume WM_PASTE
                # before restoring the clipboard that belongs to the user.
                time.sleep(0.05)
            finally:
                pyperclip.copy(previous)

        await asyncio.to_thread(_paste_exact)
        preview = text[:80] + "..." if len(text) > 80 else text
        return f"Typed exactly: {preview}"
    except (ImportError, RuntimeError):
        logger.debug("Exact clipboard typing unavailable; using key synthesis", exc_info=True)

    def _do() -> None:
        pag.typewrite(text, interval=interval) if text.isascii() else pag.write(text)

    await asyncio.to_thread(_do)
    preview = text[:80] + "..." if len(text) > 80 else text
    return f"Typed: {preview}"


async def keyboard_press(key: str, presses: int = 1, *, target_window: str | None = None) -> str:
    """Press a single key (enter, tab, escape, f1, etc.). With ``target_window``, no key is sent when another
    window is in front."""
    pag = _ensure_pyautogui()
    await _refuse_unless_in_front(pag, target_window)
    await asyncio.to_thread(lambda: pag.press(key, presses=presses))
    await _keys_released(pag, [key])
    return f"Pressed '{key}' x{presses}"


async def _refuse_unless_in_front(pag: object, target_window: str | None) -> None:
    if target_window and _real_input(pag):
        from pilot.system.desktop_observe import refuse_unless_in_front

        await refuse_unless_in_front(target_window)


async def _keys_released(pag: object, keys: list[str]) -> None:
    """Read right after a key step that none of its keys is still held down (a stuck modifier changes every key
    after it); a held key fails the step so the plan's later steps do not run."""
    if _real_input(pag):
        from pilot.system.desktop_observe import confirm_keys_released

        await confirm_keys_released(keys)


async def keyboard_hotkey(*keys: str, target_window: str | None = None) -> str:
    """Press a keyboard shortcut (e.g., ctrl+c, alt+tab, win+d). With ``target_window``, no key is sent when another
    window is in front.

    Examples: ('ctrl', 'c'), ('alt', 'tab'), ('win', 'd'), ('ctrl', 'shift', 'esc')
    """
    pag = _ensure_pyautogui()
    await _refuse_unless_in_front(pag, target_window)
    await asyncio.to_thread(lambda: pag.hotkey(*keys))
    await _keys_released(pag, list(keys))
    return f"Pressed hotkey: {'+'.join(keys)}"


async def keyboard_hold(key: str, duration: float = 0.5, *, target_window: str | None = None) -> str:
    """Hold a key down for a duration. The key goes through the same pyautogui as every other key (it was imported
    a second time here, so a stand-in for pyautogui still held a real key). With ``target_window``, no key is sent
    when another window is in front."""
    pag = _ensure_pyautogui()
    await _refuse_unless_in_front(pag, target_window)

    def _do():
        with pag.hold(key):
            time.sleep(duration)

    await asyncio.to_thread(_do)
    await _keys_released(pag, [key])
    return f"Held '{key}' for {duration}s"


# ── Screen Info ──────────────────────────────────────────────────────


async def screen_size() -> str:
    """Get screen resolution."""
    pag = _ensure_pyautogui()
    w, h = pag.size()
    return f"Screen resolution: {w}x{h}"


async def pixel_color(x: int, y: int) -> str:
    """Get the color of a pixel at coordinates."""
    pag = _ensure_pyautogui()
    r, g, b = pag.pixel(x, y)
    return f"Pixel at ({x},{y}): RGB({r},{g},{b}) / #{r:02x}{g:02x}{b:02x}"


async def locate_on_screen(image_path: str, confidence: float = 0.8) -> str:
    """Find an image on screen (template matching).

    Returns the center coordinates if found.
    """
    pag = _ensure_pyautogui()

    def _do():
        try:
            location = pag.locateOnScreen(image_path, confidence=confidence)
            if location:
                center = pag.center(location)
                return f"Found at ({center.x}, {center.y}), region: {location}"
            return "Image not found on screen"
        except Exception as e:
            return f"Image search failed: {e}"

    return await asyncio.to_thread(_do)
