"""The ``notify`` action's real result, through the executor as the planner calls it, for ``platform-readers``.

    python notify_check.py missing    # Linux, before notify-send is installed: the result says so, never "sent"
    python notify_check.py no-daemon  # Linux, notify-send installed, no notification daemon on the bus
    python notify_check.py shown      # Linux with dunst running (it records what it showed); macOS: osascript

"shown" on Linux checks dunst's own history for the notification, so "sent" means a daemon really took it. On
macOS only osascript's exit status is known: the runner gives no way to see the banner.

Each check prints one ``CHECK`` line; the exit code is the number of failed checks.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

failures = 0
SUMMARY = "Heliox CI notice"
BODY = "The build finished."


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


def notify() -> str:
    from pilot.actions import NotifyParams
    from pilot.agents.executor import Executor

    action = SimpleNamespace(parameters=NotifyParams(summary=SUMMARY, body=BODY))
    try:
        return asyncio.run(Executor._exec_notify(SimpleNamespace(), action))
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def dunst_history() -> str:
    """The summaries dunst has shown; a notification enters its history once closed, so close them first."""
    subprocess.run(["dunstctl", "close-all"], capture_output=True, timeout=10, check=False)
    time.sleep(0.5)
    done = subprocess.run(["dunstctl", "history"], capture_output=True, text=True, timeout=10, check=False)
    try:
        data = json.loads(done.stdout)
        return json.dumps([item.get("summary", {}).get("data") for item in data.get("data", [[]])[0]])
    except ValueError:
        return done.stdout[:300]


def main(argv: list[str]) -> int:
    mode = argv[0] if argv else "shown"
    result = notify()
    platform = "macos" if sys.platform == "darwin" else "linux"
    if mode == "missing":
        check(
            "notify-linux/missing-notify-send-is-said",
            result.startswith("Could not show a desktop notification") and "notify-send is not installed" in result,
            result,
        )
    elif mode == "no-daemon":
        check(
            "notify-linux/no-notification-daemon-is-said",
            result.startswith("Could not show a desktop notification") and BODY in result,
            result,
        )
    elif platform == "linux":
        time.sleep(1.0)
        history = dunst_history()
        check(
            "notify-linux/sent-only-when-a-daemon-took-it",
            result == f"Notification sent: {SUMMARY}" and SUMMARY in history,
            f"{result}; dunst history {history}",
        )
    else:
        check("notify-macos/osascript-result", result == f"Notification sent: {SUMMARY}", result)
    return failures


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
