"""A real press of the fixture's "Press me" button through the click path the planner uses, for the macOS and
Linux checks.

The button is pressed with ``mouse_click`` on its snapshot ref, through ``Executor._exec_mouse_click``: the press
must go through AXPress (macOS) or the AT-SPI action (Linux), the button's own handler must run (the fixture
counts it), and the cursor must not be used. Then the same window is retitled as a password manager's, which
Heliox never reads or acts in, and the same click must be refused with nothing pressed and nothing clicked: if the
refusal failed, the click path would fall back to a real mouse click on the button's centre, and the fixture
would count it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from types import SimpleNamespace

REFUSED_TITLE = "Bitwarden - Heliox CI"


def _executor_stub():
    from pilot.agents.executor import Executor

    stub = SimpleNamespace(
        _accessibility_target=Executor._accessibility_target,
        _press_element_in_background=Executor._press_element_in_background,
        _detected_click_target=None,
        _config=None,
    )
    return Executor, stub


def _click(snapshot_id: str, ref: str) -> str:
    from pilot.actions import MouseParams

    executor, stub = _executor_stub()
    action = SimpleNamespace(parameters=MouseParams(element_ref=ref, snapshot_id=snapshot_id))
    return asyncio.run(executor._exec_mouse_click(stub, action))


def _wait_for_count(presses: Callable[[], int], wanted: int, timeout: float) -> int:
    deadline = time.monotonic() + timeout
    count = presses()
    while count < wanted and time.monotonic() < deadline:
        time.sleep(0.3)
        count = presses()
    return count


def press_checks(
    prefix: str,
    check: Callable[..., bool],
    presses: Callable[[], int],
    retitle: Callable[[str], None],
    title_after: str,
) -> None:
    from pilot.system import accessibility

    try:
        snapshot = asyncio.run(accessibility.capture_snapshot(max_elements=60))
    except Exception as exc:
        check(f"{prefix}/press-snapshot", False, f"{type(exc).__name__}: {exc}")
        return
    button = next((e for e in snapshot.elements if e.role == "Button" and e.name == "Press me"), None)
    if not check(f"{prefix}/press-finds-the-button", button is not None):
        return
    before = presses()
    try:
        output = _click(snapshot.snapshot_id, button.ref)
    except Exception as exc:
        output = f"{type(exc).__name__}: {exc}"
    after = _wait_for_count(presses, before + 1, 5.0)
    check(f"{prefix}/press-ran-the-buttons-handler", after == before + 1, f"presses {before} -> {after}; {output}")
    check(f"{prefix}/press-did-not-use-the-cursor", "cursor did not move" in output, output)

    retitle(REFUSED_TITLE)
    time.sleep(0.5)
    before = presses()
    try:
        output = _click(snapshot.snapshot_id, button.ref)
        refused = False
    except accessibility.AccessibilityRefused as exc:
        output, refused = f"refused: {exc}", True
    except Exception as exc:
        output, refused = f"{type(exc).__name__}: {exc}", False
    time.sleep(1.5)
    after = presses()
    check(f"{prefix}/press-refused-in-a-password-managers-window", refused, output)
    check(f"{prefix}/nothing-pressed-or-clicked-when-refused", after == before, f"presses {before} -> {after}")
    retitle(title_after)
    time.sleep(0.5)
