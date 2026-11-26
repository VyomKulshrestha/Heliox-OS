"""Actions that change something whose result cannot be observed when the check would run, each with the reason.

The catalog generator shows these as ``verification.method = "unobservable_at_run_time"`` with the reason, instead
of the plain "no check registered". The action's own result says what was requested, never that it happened. An
action listed here must not also register a verifier.
"""

from __future__ import annotations

from pilot.actions import ActionType

UNOBSERVABLE_POSTCONDITION: dict[ActionType, str] = {
    ActionType.POWER_SLEEP: (
        "A sleeping computer runs nothing, so it cannot look at its own sleep: on Linux and macOS the command returns "
        "before the computer sleeps, on Windows after it wakes. The result says sleep was requested."
    ),
}
