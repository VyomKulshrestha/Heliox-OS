"""Explicitly consented desktop input recording for workflow drafts.

The hook never buffers or stores typed characters. It sees only that text entry
occurred and creates an editable placeholder. It also avoids recording Heliox
injected input via the store's execution suppression scope.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable

from pilot.workflows.recorded_workflows import RecordedWorkflowStore


def _is_heliox_foreground() -> bool:
    """Best-effort exclusion of the Heliox UI (including its browser tab)."""
    if sys.platform != "win32":
        return False
    import ctypes

    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    user32.GetWindowTextW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int)
    title = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(user32.GetForegroundWindow(), title, len(title))
    return "heliox os" in title.value.lower()


class ManualWorkflowHook:
    def __init__(
        self, store: RecordedWorkflowStore, *, ignore_foreground: Callable[[], bool] = _is_heliox_foreground
    ) -> None:
        self._store = store
        self._ignore_foreground = ignore_foreground
        self._keyboard: Any = None
        self._mouse: Any = None
        self._pressed: tuple[int, int, str, float] | None = None
        self._modifiers: set[str] = set()

    def start(self) -> None:
        if self._keyboard is not None:
            raise RuntimeError("Manual workflow hook is already running")
        try:
            from pynput import keyboard, mouse
        except ImportError as exc:
            raise RuntimeError("Manual desktop recording requires pynput") from exc
        keyboard_listener = keyboard.Listener(on_press=self._on_key_press, on_release=self._on_key_release)
        mouse_listener = mouse.Listener(on_click=self._on_click, on_scroll=self._on_scroll)
        try:
            keyboard_listener.start()
            mouse_listener.start()
            if not keyboard_listener.is_alive() or not mouse_listener.is_alive():
                raise RuntimeError("Manual desktop input hook did not start")
        except Exception:
            keyboard_listener.stop()
            mouse_listener.stop()
            raise
        self._keyboard, self._mouse = keyboard_listener, mouse_listener

    def stop(self) -> None:
        keyboard, mouse = self._keyboard, self._mouse
        self._keyboard = self._mouse = None
        self._pressed = None
        self._modifiers.clear()
        if keyboard is not None:
            keyboard.stop()
        if mouse is not None:
            mouse.stop()

    def healthy(self) -> bool:
        return bool(self._keyboard and self._mouse and self._keyboard.is_alive() and self._mouse.is_alive())

    @staticmethod
    def _name(key: object) -> str:
        return str(getattr(key, "name", "") or "").lower()

    def _on_key_press(self, key: object) -> None:
        if self._ignore_foreground():
            return
        name = self._name(key)
        if name.startswith(("ctrl", "alt", "cmd", "shift")):
            self._modifiers.add(name.split("_")[0])
            return
        # Modifier combinations can trigger arbitrary actions; a guessed
        # replay would be unsafe. Deliberately do not read their characters.
        if self._modifiers - {"shift"}:
            self._store.record_manual_event("unsupported")
        elif name in {"enter", "tab", "esc"}:
            self._store.record_manual_event("key", name=name)
        elif getattr(key, "char", None) is not None or name in {"space", "backspace", "delete"}:
            self._store.record_manual_event("text")
        elif name:
            self._store.record_manual_event("unsupported")

    def _on_key_release(self, key: object) -> None:
        name = self._name(key)
        if name.startswith(("ctrl", "alt", "cmd", "shift")):
            self._modifiers.discard(name.split("_")[0])

    def _on_click(self, x: object, y: object, button: object, pressed: bool) -> None:
        if self._ignore_foreground():
            self._pressed = None
            return
        button_name = str(getattr(button, "name", "") or "").lower()
        if pressed:
            self._pressed = (x, y, button_name, time.monotonic()) if isinstance(x, int) and isinstance(y, int) else None
            return
        origin = self._pressed
        self._pressed = None
        if origin is None or origin[2] != button_name or not isinstance(x, int) or not isinstance(y, int):
            self._store.record_manual_event("unsupported")
        elif abs(x - origin[0]) > 5 or abs(y - origin[1]) > 5:
            self._store.record_manual_event(
                "drag",
                x=origin[0],
                y=origin[1],
                end_x=x,
                end_y=y,
                button=button_name,
                duration=time.monotonic() - origin[3],
            )
        else:
            self._store.record_manual_event("click", x=origin[0], y=origin[1], button=button_name)

    def _on_scroll(self, x: object, y: object, dx: object, dy: object) -> None:
        if self._ignore_foreground():
            return
        if isinstance(dx, int) and dx:
            self._store.record_manual_event("scroll", x=x, y=y, amount=dx, horizontal=True)
        if isinstance(dy, int) and dy:
            self._store.record_manual_event("scroll", x=x, y=y, amount=dy, horizontal=False)
