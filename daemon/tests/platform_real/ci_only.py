"""The guard for checks that change something outside their own temporary folder: a keychain setting, a mount, the
audio defaults, Heliox's own config, data or credentials in the home folder. They run only on a CI runner, never on
a person's computer, and there is no override.

Each such check calls ``require_runner`` first, before any import or call that has an effect. Off a runner it prints
one sentence saying what the check would change and exits with NOT_RUN (3): neither a pass (0) nor a failure (1).
Importing this module has no effect.
"""

from __future__ import annotations

import os
import sys

NOT_RUN = 3


def on_runner() -> bool:
    """A GitHub Actions runner: GITHUB_ACTIONS is "true" and RUNNER_TEMP is set."""
    return os.environ.get("GITHUB_ACTIONS") == "true" and bool(os.environ.get("RUNNER_TEMP", "").strip())


def require_runner(changes: str) -> None:
    """Return on a CI runner; otherwise say what would change and exit with NOT_RUN."""
    if on_runner():
        return
    print(
        f"NOT RUN: this check {changes}, so it runs only on a CI runner (GITHUB_ACTIONS=true and RUNNER_TEMP set) "
        "and changed nothing here.",
        flush=True,
    )
    sys.exit(NOT_RUN)
