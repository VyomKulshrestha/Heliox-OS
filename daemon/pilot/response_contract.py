"""Truthful, user-facing summaries for terminal execution states.

Planning text describes intent.  These helpers describe what actually happened
after execution and verification, so callers never reuse a proposed plan as a
success or failure result.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pilot.actions import (
    READ_ONLY_ACTIONS,
    ActionPlan,
    ActionResult,
    ActionType,
    PermissionTier,
    VerificationResult,
)

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_EFFECTFUL_READ_ONLY_CATALOG_ENTRIES = {
    ActionType.OPEN_URL,
    ActionType.OPEN_APPLICATION,
    ActionType.NOTIFY,
}


def exact_labeled_finding_count(plan: ActionPlan) -> int | None:
    """Return the requested exact finding count, when the user specified one."""
    raw_input = str(plan.raw_input or "")
    match = re.search(
        r"\bexactly\s+(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
        r"\s+(?:distinct\s+)?(?:labeled\s+)?(?:findings?|results?|items?)\b",
        raw_input,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = match.group("count").lower()
    return int(value) if value.isdigit() else _NUMBER_WORDS[value]


def success_message(
    plan: ActionPlan,
    results: Sequence[ActionResult],
    verification: VerificationResult,
    *,
    dry_run: bool,
) -> str:
    """Build a terminal success message from verified execution state."""
    if dry_run:
        intent = _clean(plan.explanation)
        return (
            f"Dry run completed; no changes were made. Planned: {intent}"
            if intent
            else "Dry run completed; no changes were made."
        )

    observed = set(verification.independently_observed_actions)
    unverified_effects = sum(
        result.success
        and index not in observed
        and (result.action.action_type not in READ_ONLY_ACTIONS
             or result.action.action_type in _EFFECTFUL_READ_ONLY_CATALOG_ENTRIES)
        and result.action.permission_tier > PermissionTier.READ_ONLY
        for index, result in enumerate(results)
    )
    caveat = (
        "\nNote: " + ("1 action effect was" if unverified_effects == 1
                       else f"{unverified_effects} action effects were")
        + " reported successful but not independently verified."
        if unverified_effects else ""
    )

    exact_count = exact_labeled_finding_count(plan)
    if exact_count is not None:
        sections = [section for result in results for section in _extract_labeled_sections(result.output)]
        if len(sections) == exact_count:
            empty_finding = "Reported." if unverified_effects else "Verified."
            return "\n".join(
                f"{index}. {label}: {content or empty_finding}"
                for index, (label, content) in enumerate(sections, start=1)
            ) + caveat

    if exact_count is not None and exact_count == len(results):
        findings = []
        empty_finding = "Reported." if unverified_effects else "Verified."
        for index, result in enumerate(results, start=1):
            label = _clean(result.action.target) or result.action.action_type.value.replace("_", " ").title()
            output = _matching_labeled_section(result.output, label)
            findings.append(f"{index}. {label}: {output or empty_finding}")
        return "\n".join(findings) + caveat

    successful_outputs = [
        _bound(_clean(result.output), 1200) for result in results if result.success and _clean(result.output)
    ]
    if len(results) == 1 and successful_outputs:
        return successful_outputs[0] + caveat

    if successful_outputs:
        lines = []
        for result in results:
            output = _bound(_clean(result.output), 500)
            if not result.success or not output:
                continue
            label = _clean(result.action.target) or result.action.action_type.value.replace("_", " ").title()
            lines.append(f"- {label}: {output}")
        if lines:
            heading = "Execution results" if unverified_effects else "Verified results"
            return heading + ":\n" + "\n".join(lines) + caveat

    count = len(results)
    noun = "action" if count == 1 else "actions"
    intent = _clean(plan.explanation)
    prefix = f"Completed {count} {noun}." if unverified_effects else f"Completed and verified {count} {noun}."
    if intent:
        return f"{prefix} {intent}{caveat}"
    if verification.details:
        return f"{prefix} {_clean(verification.details[0])}{caveat}"
    return prefix + caveat


def partial_failure_message(
    results: Sequence[ActionResult],
    verification: VerificationResult | None,
) -> str:
    """Build a failure summary without presenting the original intent as fact."""
    total = len(results)
    failed_indices = set(verification.failed_actions if verification else [])
    failed_indices.update(i for i, result in enumerate(results) if not result.success)
    failed = len(failed_indices)

    if total:
        noun = "action" if total == 1 else "actions"
        prefix = (
            f"Task did not complete successfully: {failed or total} of {total} {noun} failed or could not be verified."
        )
    else:
        prefix = "Task did not complete successfully: no action result was verified."

    issues = [result.error for result in results if result.error]
    if verification:
        issues.extend(
            detail
            for detail in verification.details
            if "FAILED" in detail or "MISMATCH" in detail or "error" in detail.lower()
        )
    first_issue = next((_clean(issue) for issue in issues if _clean(issue)), "")
    return f"{prefix} {first_issue}" if first_issue else prefix


def _clean(value: object) -> str:
    return " ".join(str(value).split())


def _bound(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def _extract_labeled_sections(output: str) -> list[tuple[str, str]]:
    """Extract ``=== Label ===`` tool sections as deterministic findings."""
    matches = list(re.finditer(r"(?m)^===\s*(?P<label>[^=\r\n]+?)\s*===\s*$", output))
    if not matches:
        return []

    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(output)
        sections.append((_clean(match.group("label")), _clean(output[start:end])))
    return sections


def _matching_labeled_section(output: str, target: str) -> str:
    """Return only the composite tool section requested by an action target."""
    sections = _extract_labeled_sections(output)
    if not sections:
        return _clean(output)

    target_words = set(re.findall(r"[a-z0-9]+", target.lower()))
    aliases = {
        "os": {"operating", "system"},
        "release": {"operating", "system"},
        "ram": {"memory"},
        "storage": {"disk"},
    }
    expanded_target = set(target_words)
    for word in target_words:
        expanded_target.update(aliases.get(word, set()))

    best_content = ""
    best_score = 0
    for label, content in sections:
        label_words = set(re.findall(r"[a-z0-9]+", label.lower()))
        score = len(expanded_target & label_words)
        if score > best_score:
            best_content = content
            best_score = score
    return best_content or _clean(output)
