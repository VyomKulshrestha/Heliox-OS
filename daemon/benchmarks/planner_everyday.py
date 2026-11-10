"""Acceptance checks for the everyday planner eval (benchmarks/planner_everyday_v1.json).

Each case gives a request and what a correct plan must look like: the action types it may use,
the steps it must contain (with their key parameters, in order), a step limit, and which shell,
code or non-undoable steps the job genuinely needs. Equivalent correct plans pass: a case can
list alternative step sequences, extra read-only steps can be allowed, and parameters are
matched by rule (a path ending, a word the text contains) instead of by exact value.

Six rules are scored per plan, each pass, fail or not applicable (None):

- types: every step uses an action type the case allows.
- params: every required step is present with the right key parameters.
- order: the required steps appear in the order given, and a step that consumes the previous
  step's output comes right after a step that produces it.
- max_steps: the plan has no more steps than the case allows.
- no_code: no shell or code step unless the case says the job needs one (no typed action does it),
  and none of the case's forbidden steps (such as a shell command deleting what file_delete can).
- undo: every step that changes something can be reversed by the undo journal, unless the case
  says no undoable action can do that job.

The checker reads plans only; it never executes anything.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RULES = ("types", "params", "order", "max_steps", "no_code", "undo")

CODE_TYPES = frozenset(
    {
        "shell_command",
        "shell_script",
        "pty_exec",
        "code_execute",
        "code_generate_and_run",
        "browser_execute_js",
        "ssh_command",
        "ssh_script",
    }
)

# Read-only steps a case may allow as harmless extras with "allow_lookup": true.
LOOKUP_TYPES = frozenset(
    {
        "file_read",
        "file_list",
        "file_search",
        "directory_summary",
        "directory_size",
        "window_list",
        "process_list",
        "volume_get",
        "brightness_get",
        "clipboard_read",
    }
)

# Steps that change nothing. undo_support() files some of them (file_read is in the user_write
# tier) under "not undoable"; for this rule they are no change at all.
READ_ONLY_TYPES = LOOKUP_TYPES | frozenset(
    {
        "file_parse",
        "file_hash",
        "file_compare",
        "file_search_content",
        "screen_ocr",
        "screen_find_text",
        "screen_analyze",
        "screen_element_map",
        "screen_accessibility_snapshot",
        "screen_detect_elements",
        "browser_page_info",
        "browser_snapshot",
        "browser_extract",
        "disk_list",
        "system_info",
        "package_search",
    }
)

DEFAULT_CASES = Path(__file__).with_name("planner_everyday_v1.json")
_PLACEHOLDER = re.compile(r"\{\{?[A-Z][A-Z0-9_]*\}\}?")


@dataclass
class CheckResult:
    """The rule outcomes for one plan against one case."""

    case_id: str
    rules: dict[str, bool | None]
    failures: list[str] = field(default_factory=list)
    asked: bool = False
    error: str = ""

    @property
    def passed(self) -> bool:
        return all(value is not False for value in self.rules.values())


def load_cases(path: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
    data = json.loads(Path(path or DEFAULT_CASES).read_text(encoding="utf-8"))
    return list(data["cases"])


def expand_home(value: str, home: str | None = None) -> str:
    home = home or str(Path.home())
    return value.replace("{home}", home)


def norm_path(value: Any) -> str:
    """Lower-case, forward slashes, no trailing slash: for comparing paths as text."""
    text = str(value or "").strip().strip("\"'")
    text = os.path.expanduser(text) if text.startswith("~") else text
    return text.replace("\\", "/").rstrip("/").lower()


# --- turning a plan into plain step dicts ---------------------------------------------------------


def _parsed(raw: dict[str, Any]) -> Any | None:
    """The Action the daemon would build from a raw action dict, or None if it would reject it."""
    from pilot.agents.planner import Planner

    try:
        return Planner.__new__(Planner)._parse_action(json.loads(json.dumps(raw)))
    except Exception:
        return None


def _step_dict(action: Any) -> dict[str, Any]:
    if isinstance(action, dict):
        parsed = _parsed(action)
        if parsed is not None:
            return _step_dict(parsed)
        raw = dict(action)
        kind = raw.get("action_type") or raw.get("type") or ""
        params = raw.get("parameters") or {}
        return {
            "action_type": str(getattr(kind, "value", kind)),
            "target": str(raw.get("target") or ""),
            "parameters": dict(params) if isinstance(params, dict) else {},
            "use_previous_output": bool(raw.get("use_previous_output", False)),
        }
    params = action.parameters.model_dump() if hasattr(action.parameters, "model_dump") else {}
    return {
        "action_type": action.action_type.value,
        "target": action.target or "",
        "parameters": params,
        "use_previous_output": bool(action.use_previous_output),
    }


def plan_steps(plan: Any) -> list[dict[str, Any]]:
    """Steps from an ActionPlan, a dict with "actions", or a list of actions or dicts."""
    if plan is None:
        return []
    actions = plan.get("actions", []) if isinstance(plan, dict) else getattr(plan, "actions", plan)
    return [_step_dict(action) for action in actions or []]


def _undo_kind(step: dict[str, Any]) -> str:
    """'undoable', 'not_undoable' or 'no_change', as the undo journal classifies the step."""
    from pilot.actions import Action, ActionType
    from pilot.agents.planner import Planner
    from pilot.workflows.undo_journal import undo_support

    try:
        action = Planner.__new__(Planner)._parse_action(
            {"action_type": step["action_type"], "target": step["target"], "parameters": dict(step["parameters"])}
        )
    except Exception:
        try:
            ActionType(step["action_type"])
        except ValueError:
            return "not_undoable"  # an unknown type cannot be journalled
        action = Action.model_construct(action_type=ActionType(step["action_type"]), target=step["target"])
    return undo_support(action).kind


# --- parameter matching ---------------------------------------------------------------------------


def _param(step: dict[str, Any], name: str) -> Any:
    if name == "target":
        return step["target"]
    params = step["parameters"]
    value = params.get(name)
    if (value is None or value == "") and name == "path":
        value = step["target"]
    return value


def _match_value(value: Any, spec: Any, step: dict[str, Any], home: str) -> bool:
    if isinstance(spec, list):  # shorthand for any_of
        return any(_match_value(value, item, step, home) for item in spec)
    if not isinstance(spec, dict):
        spec = {"equals": spec}
    for op, expected in spec.items():
        if op == "any_of":
            if not any(_match_value(value, item, step, home) for item in expected):
                return False
        elif op == "equals":
            if isinstance(expected, str) and isinstance(value, str):
                if value.strip().lower() != expected.strip().lower():
                    return False
            elif isinstance(expected, bool) or expected is None:
                if value is not expected:
                    return False
            elif value != expected:
                return False
        elif op == "endswith":
            if value is None or not norm_path(value).endswith(norm_path(expand_home(expected, home))):
                return False
        elif op == "path":  # the exact path, after expanding {home}
            if value is None or norm_path(value) != norm_path(expand_home(expected, home)):
                return False
        elif op == "contains":
            words = expected if isinstance(expected, list) else [expected]
            text = str(value or "").lower()
            if not all(str(word).lower() in text for word in words):
                return False
        elif op == "not_contains":
            words = expected if isinstance(expected, list) else [expected]
            text = str(value or "").lower()
            if any(str(word).lower() in text for word in words):
                return False
        elif op == "regex":
            if value is None or not re.search(expected, str(value), re.IGNORECASE):
                return False
        elif op == "range":
            low, high = expected
            if isinstance(value, bool) or not isinstance(value, int | float) or not low <= value <= high:
                return False
        elif op == "same_dir_as":
            other = _param(step, expected)
            if value is None or other is None:
                return False
            if norm_path(value).rsplit("/", 1)[0] != norm_path(other).rsplit("/", 1)[0]:
                return False
        elif op == "has_placeholder":
            if bool(_PLACEHOLDER.search(str(value or ""))) is not bool(expected):
                return False
        elif op == "present":
            if bool(value not in (None, "", [], {})) is not bool(expected):
                return False
        else:
            raise ValueError(f"unknown matcher {op!r}")
    return True


def step_matches(step: dict[str, Any], spec: dict[str, Any], home: str | None = None) -> bool:
    home = home or str(Path.home())
    kinds = spec.get("type")
    kinds = [kinds] if isinstance(kinds, str) else list(kinds or [])
    if kinds and step["action_type"] not in kinds:
        return False
    for name, rule in (spec.get("params") or {}).items():
        if not _match_value(_param(step, name), rule, step, home):
            return False
    if "use_previous_output" in spec and step["use_previous_output"] is not spec["use_previous_output"]:
        return False
    return True


def _assign(steps: list[dict[str, Any]], specs: list[dict[str, Any]], home: str) -> list[int] | None:
    """Indexes of distinct plan steps matching each spec, preferring an in-order assignment."""
    # In order first (a greedy left-to-right scan finds one whenever one exists).
    ordered: list[int] = []
    start = 0
    for spec in specs:
        index = next((i for i in range(start, len(steps)) if step_matches(steps[i], spec, home)), None)
        if index is None:
            break
        ordered.append(index)
        start = index + 1
    if len(ordered) == len(specs):
        return ordered
    # Any order (small plans: simple backtracking).
    used: list[int] = []

    def place(position: int) -> bool:
        if position == len(specs):
            return True
        for i, step in enumerate(steps):
            if i not in used and step_matches(step, specs[position], home):
                used.append(i)
                if place(position + 1):
                    return True
                used.pop()
        return False

    return list(used) if place(0) else None


# --- the rules ------------------------------------------------------------------------------------


def _consumes_previous(step: dict[str, Any]) -> bool:
    if step["use_previous_output"]:
        return True
    return any(isinstance(v, str) and _PLACEHOLDER.search(v) for v in step["parameters"].values())


def _producers() -> frozenset[str]:
    return LOOKUP_TYPES | {
        "browser_extract",
        "browser_extract_table",
        "browser_extract_links",
        "screen_ocr",
        "screen_analyze",
        "screen_find_text",
        "screen_detect_elements",
        "screen_accessibility_snapshot",
        "screen_element_map",
        "browser_snapshot",
        "file_parse",
        "api_scrape",
        "code_execute",
        "shell_command",
        "shell_script",
    }


def _check_alternative(
    steps: list[dict[str, Any]], case: dict[str, Any], specs: list[dict[str, Any]], home: str
) -> tuple[dict[str, bool | None], list[str]]:
    failures: list[str] = []
    allowed = set(case.get("allowed") or [])
    for spec in specs:
        kinds = spec.get("type")
        allowed.update([kinds] if isinstance(kinds, str) else kinds or [])
    code_ok = set(case.get("code_ok") or [])
    allowed |= code_ok
    if case.get("allow_lookup"):
        allowed |= LOOKUP_TYPES

    rules: dict[str, bool | None] = {}

    bad_types = [s["action_type"] for s in steps if s["action_type"] not in allowed]
    rules["types"] = not bad_types
    if bad_types:
        failures.append("types: " + ", ".join(dict.fromkeys(bad_types)))

    assignment = _assign(steps, specs, home)
    rules["params"] = assignment is not None
    if assignment is None:
        missing = [
            spec.get("label") or str(spec.get("type"))
            for spec in specs
            if not any(step_matches(s, spec, home) for s in steps)
        ]
        failures.append("params: no step matches " + "; ".join(missing or ["the required steps together"]))

    order_ok: bool | None = None
    if assignment is not None:
        order_ok = assignment == sorted(assignment)
        if not order_ok:
            failures.append("order: required steps out of order")
    for index, step in enumerate(steps):
        if _consumes_previous(step) and step["action_type"] not in CODE_TYPES | {"notify"}:
            previous = steps[index - 1]["action_type"] if index else ""
            if previous not in _producers():
                order_ok = False
                failures.append(f"order: step {index + 1} ({step['action_type']}) uses output no earlier step made")
    rules["order"] = order_ok

    limit = case.get("max_steps")
    rules["max_steps"] = None if limit is None else len(steps) <= int(limit)
    if rules["max_steps"] is False:
        failures.append(f"max_steps: {len(steps)} > {limit}")

    code_steps = [s["action_type"] for s in steps if s["action_type"] in CODE_TYPES and s["action_type"] not in code_ok]
    forbidden = [
        spec.get("label") or str(spec.get("type"))
        for spec in case.get("forbid") or []
        if any(step_matches(s, spec, home) for s in steps)
    ]
    rules["no_code"] = not code_steps and not forbidden
    if code_steps:
        failures.append("no_code: " + ", ".join(dict.fromkeys(code_steps)))
    if forbidden:
        failures.append("no_code: " + "; ".join(forbidden))

    exempt = set(case.get("not_undoable_ok") or []) | code_ok
    not_undoable = []
    for step in steps:
        kind = "no_change" if step["action_type"] in READ_ONLY_TYPES else _undo_kind(step)
        if kind == "not_undoable" and step["action_type"] not in exempt:
            not_undoable.append(step["action_type"])
    rules["undo"] = not not_undoable
    if not_undoable:
        failures.append("undo: cannot undo " + ", ".join(dict.fromkeys(not_undoable)))
    return rules, failures


def check_plan(
    case: dict[str, Any],
    plan: Any,
    *,
    error: str = "",
    explanation: str = "",
    home: str | None = None,
) -> CheckResult:
    """Score one plan (ActionPlan, dict or action list) against one case."""
    home = home or str(Path.home())
    case_id = str(case.get("id") or case.get("request", ""))
    if error:
        return CheckResult(case_id, dict.fromkeys(RULES, False), [f"planner error: {error[:200]}"], error=error)
    steps = plan_steps(plan)
    if not steps:
        if case.get("allow_ask") and (explanation or getattr(plan, "explanation", "")):
            return CheckResult(case_id, dict.fromkeys(RULES), [], asked=True)
        return CheckResult(case_id, dict.fromkeys(RULES, False), ["no steps: the planner did not act"])

    alternatives = [case.get("steps") or []] + list(case.get("alternatives") or [])
    best: tuple[dict[str, bool | None], list[str]] | None = None
    for specs in alternatives:
        rules, failures = _check_alternative(steps, case, specs, home)
        score = sum(value is not False for value in rules.values())
        if best is None or score > sum(value is not False for value in best[0].values()):
            best = (rules, failures)
    assert best is not None
    return CheckResult(case_id, best[0], best[1])


def summarise(results: Iterable[CheckResult]) -> dict[str, Any]:
    """Pass rate per rule (applicable plans only) and overall."""
    results = list(results)
    summary: dict[str, Any] = {"plans": len(results), "overall": sum(r.passed for r in results)}
    for rule in RULES:
        scored = [r.rules[rule] for r in results if r.rules.get(rule) is not None]
        summary[rule] = {"passed": sum(bool(v) for v in scored), "scored": len(scored)}
    summary["asked"] = sum(r.asked for r in results)
    return summary
