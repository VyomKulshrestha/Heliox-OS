"""Human takeover: the person can take the desktop back from Heliox at any time.

While the person holds control, desktop input that Heliox starts on its own
(autonomous jobs, MCP-submitted work) is refused, not queued, and autonomous
jobs pause until control is handed back. The person's own typed, spoken or
gestured commands still run.

Control moves to the person explicitly (the ``take_control`` RPC, or the agent
asking for help) or automatically when physical mouse or keyboard input arrives
while the gate is armed during autonomous work. On Windows, synthetic input,
including Heliox's own clicks and keystrokes, carries the OS "injected" flag
and is ignored. Only the fact that physical input happened is used: keys,
characters and pointer coordinates are never read or stored.

Handing control back (``release_control``) starts a new generation and
discards earlier accessibility snapshots, so the agent observes again before
acting.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from enum import StrEnum

from pilot.actions import ActionType

logger = logging.getLogger("pilot.security.human_control")

HUMAN_CONTROL_WAIT_SECONDS = 15 * 60

# Actions that drive the person's input devices or change which window they see.
HUMAN_CONTROLLED_ACTIONS = frozenset(
    {
        ActionType.MOUSE_CLICK,
        ActionType.MOUSE_DOUBLE_CLICK,
        ActionType.MOUSE_RIGHT_CLICK,
        ActionType.MOUSE_MOVE,
        ActionType.MOUSE_DRAG,
        ActionType.MOUSE_SCROLL,
        ActionType.KEYBOARD_TYPE,
        ActionType.KEYBOARD_PRESS,
        ActionType.KEYBOARD_HOTKEY,
        ActionType.KEYBOARD_HOLD,
        ActionType.WINDOW_FOCUS,
        ActionType.WINDOW_MINIMIZE,
        ActionType.WINDOW_MAXIMIZE,
        ActionType.WINDOW_CLOSE,
        ActionType.OPEN_APPLICATION,
        ActionType.CLIPBOARD_WRITE,
    }
)

_agent_initiated: ContextVar[bool] = ContextVar("heliox_agent_initiated", default=False)


@contextmanager
def agent_initiated() -> Iterator[None]:
    """Mark work Heliox started on its own; it is subject to human takeover."""
    token = _agent_initiated.set(True)
    try:
        yield
    finally:
        _agent_initiated.reset(token)


def is_agent_initiated() -> bool:
    return _agent_initiated.get()


class ControlHolder(StrEnum):
    AGENT = "agent"
    HUMAN = "human"


@dataclass(frozen=True, slots=True)
class HumanControlStatus:
    holder: ControlHolder
    reason: str
    since: float
    generation: int
    armed: bool
    physical_input_detection: bool

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["holder"] = self.holder.value
        return data


class HumanControlRefused(PermissionError):
    """Heliox may not drive the desktop while the person holds control."""


class PhysicalInputWatcher:
    """Report physical clicks, wheel turns and key presses; ignore synthetic input.

    Windows only: the low-level hooks expose an "injected" flag for input sent
    by programs (SendInput, pyautogui). Elsewhere this returns False from
    ``start`` and takeover stays explicit.
    """

    _KEY_INJECTED = 0x10 | 0x02  # LLKHF_INJECTED | LLKHF_LOWER_IL_INJECTED
    _MOUSE_INJECTED = 0x01 | 0x02  # LLMHF_INJECTED | LLMHF_LOWER_IL_INJECTED
    _MOUSE_BUTTON_MESSAGES = frozenset({0x0201, 0x0204, 0x0207, 0x020A, 0x020B, 0x020E})

    def __init__(self, on_physical_input: Callable[[], None]) -> None:
        self._on_physical_input = on_physical_input
        self._listeners: list[object] = []

    @property
    def running(self) -> bool:
        return bool(self._listeners)

    def start(self) -> bool:
        if self._listeners:
            return True
        if sys.platform != "win32":
            return False
        try:
            from pynput import keyboard, mouse
        except ImportError:
            logger.info("pynput is not installed; physical-input takeover is unavailable")
            return False

        self._listeners = [
            keyboard.Listener(win32_event_filter=self.keyboard_filter),
            mouse.Listener(win32_event_filter=self.mouse_filter),
        ]
        for listener in self._listeners:
            listener.start()
        return True

    def keyboard_filter(self, _message: int, data: object) -> bool:
        """Low-level keyboard hook filter: only the injected flag is read."""
        if not getattr(data, "flags", 0) & self._KEY_INJECTED:
            self._on_physical_input()
        return True

    def mouse_filter(self, message: int, data: object) -> bool:
        """Low-level mouse hook filter: button and wheel messages only; no coordinates."""
        if message in self._MOUSE_BUTTON_MESSAGES and not getattr(data, "flags", 0) & self._MOUSE_INJECTED:
            self._on_physical_input()
        return True

    def stop(self) -> None:
        for listener in self._listeners:
            listener.stop()
        self._listeners = []


class HumanControlGate:
    """Track who holds the desktop and let autonomous work wait for the person."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        watcher_factory: Callable[[Callable[[], None]], PhysicalInputWatcher] | None = PhysicalInputWatcher,
    ) -> None:
        self._clock = clock
        self._holder = ControlHolder.AGENT
        self._reason = ""
        self._since = clock()
        self._generation = 0
        self._armed = 0
        self._released = asyncio.Event()
        self._released.set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pending_input = threading.Event()
        self._watcher = watcher_factory(self._on_physical_input) if watcher_factory else None
        self._listeners: list[Callable[[HumanControlStatus], Awaitable[None] | None]] = []

    def add_listener(self, callback: Callable[[HumanControlStatus], Awaitable[None] | None]) -> None:
        self._listeners.append(callback)

    def holds_control(self) -> bool:
        return self._holder == ControlHolder.HUMAN

    def status(self) -> HumanControlStatus:
        return HumanControlStatus(
            holder=self._holder,
            reason=self._reason,
            since=self._since,
            generation=self._generation,
            armed=self._armed > 0,
            physical_input_detection=bool(self._watcher and self._watcher.running),
        )

    def take(self, reason: str) -> bool:
        """Give the desktop to the person. Returns False if they already had it."""
        if self._holder == ControlHolder.HUMAN:
            return False
        self._holder = ControlHolder.HUMAN
        self._reason = reason.strip()[:200] or "requested"
        self._since = self._clock()
        self._released.clear()
        logger.info("Desktop control moved to the person: %s", self._reason)
        self._notify()
        return True

    def release(self) -> bool:
        """Hand the desktop back to Heliox; earlier screen snapshots become stale."""
        if self._holder == ControlHolder.AGENT:
            return False
        self._holder = ControlHolder.AGENT
        self._reason = ""
        self._since = self._clock()
        self._generation += 1
        from pilot.system.accessibility import default_store

        default_store().clear()
        self._released.set()
        logger.info("Desktop control returned to Heliox (generation %d)", self._generation)
        self._notify()
        return True

    async def wait_for_release(self, timeout: float = HUMAN_CONTROL_WAIT_SECONDS) -> bool:
        if not self.holds_control():
            return True
        try:
            await asyncio.wait_for(self._released.wait(), timeout)
        except TimeoutError:
            return False
        return True

    def check_agent_may_act(self, action_type: ActionType) -> None:
        if self.holds_control() and is_agent_initiated() and action_type in HUMAN_CONTROLLED_ACTIONS:
            raise HumanControlRefused(
                f"The person has control of the desktop ({self._reason}); "
                f"{action_type.value} was not run. Hand control back to Heliox to continue."
            )

    @contextmanager
    def armed(self) -> Iterator[None]:
        """Watch for physical input while autonomous work runs."""
        self._loop = asyncio.get_running_loop()
        self._armed += 1
        if self._armed == 1 and self._watcher is not None:
            try:
                self._watcher.start()
            except Exception:
                logger.warning("Physical-input watcher could not start", exc_info=True)
        try:
            yield
        finally:
            self._armed -= 1
            if self._armed == 0 and self._watcher is not None:
                self._watcher.stop()

    def _on_physical_input(self) -> None:
        """Hook-thread callback: stay O(1) and hand off to the event loop once."""
        loop = self._loop
        if loop is None or self._pending_input.is_set() or self._holder == ControlHolder.HUMAN:
            return
        self._pending_input.set()
        loop.call_soon_threadsafe(self._take_after_physical_input)

    def _take_after_physical_input(self) -> None:
        self._pending_input.clear()
        if self._armed > 0:
            self.take("physical mouse or keyboard input")

    def _notify(self) -> None:
        status = self.status()
        for callback in list(self._listeners):
            try:
                result = callback(status)
                if inspect.isawaitable(result):
                    asyncio.ensure_future(result)
            except Exception:
                logger.debug("Human-control listener failed", exc_info=True)
