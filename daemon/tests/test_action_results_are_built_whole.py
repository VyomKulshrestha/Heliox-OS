"""Every ActionResult the daemon builds has what validation requires, on every path, tested or not.

In a live run an HTTP GET that answered 404 ended as RPC error -32603 "1 validation error for ActionResult" with no
reply: the RSS agent built ``ActionResult(action_type=..., success=False, output=...)`` on a branch no test reached,
and pydantic refused it (``action`` is required; ``action_type`` is not a field and was dropped). This reads the
source of ``pilot/`` and checks each call to ``ActionResult(...)`` by its words, so a branch nobody runs is checked too.
"""

from __future__ import annotations

import ast
from pathlib import Path

from pilot.actions import ActionResult

PILOT = Path(__file__).resolve().parents[1] / "pilot"
FIELDS = set(ActionResult.model_fields)
REQUIRED = [name for name, field in ActionResult.model_fields.items() if field.is_required()]


def _calls() -> list[tuple[str, int, ast.Call]]:
    found = []
    for path in sorted(PILOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            if name == "ActionResult":
                found.append((str(path.relative_to(PILOT.parent)).replace("\\", "/"), node.lineno, node))
    return found


def _problems(call: ast.Call) -> list[str]:
    if any(keyword.arg is None for keyword in call.keywords):
        return []  # ActionResult(**fields): checked where the fields are made
    given = {keyword.arg for keyword in call.keywords}
    given |= set(REQUIRED[: len(call.args)])  # positional: action, success
    problems = [f"missing {name}" for name in REQUIRED if name not in given]
    problems += [f"{name} is not a field" for name in sorted(given - FIELDS)]
    for keyword in call.keywords:
        if keyword.arg in {"output", "error", "notice", "skip_reason"} and isinstance(
            keyword.value, (ast.Dict, ast.List, ast.Set, ast.Tuple)
        ):
            problems.append(f"{keyword.arg} is given a {type(keyword.value).__name__.lower()}, not text")
    return problems


def test_the_daemon_builds_action_results_in_many_places() -> None:
    assert REQUIRED == ["action", "success"]
    # The walk found the calls (95 on 27 November 2026): a broken walk would pass the next test vacuously.
    assert len(_calls()) > 50


def test_every_action_result_the_daemon_builds_would_validate() -> None:
    broken = [f"{path}:{line}: {', '.join(problems)}" for path, line, call in _calls() if (problems := _problems(call))]
    assert broken == []


def test_the_check_catches_the_call_that_crashed_the_404() -> None:
    crashed = ast.parse(
        "ActionResult(action_type=ActionType.API_REQUEST, success=False, output='RSS Agent: unrecognized request.')"
    ).body[0]
    assert isinstance(crashed, ast.Expr) and isinstance(crashed.value, ast.Call)
    assert _problems(crashed.value) == ["missing action", "action_type is not a field"]
