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
from pilot.result_sentences import Said

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
    ActionType.REQUEST_HUMAN_HELP,
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
    message = _success_message_body(plan, results, verification, dry_run=dry_run)
    # A notice that is the step's whole output (an absent target's sentence) is already said by the step's sentence.
    notices = (
        []
        if dry_run
        else [
            _clean(result.notice)
            for result in results
            if result.success and result.notice and _clean(result.notice) != _clean(result.output)
        ]
    )
    if not dry_run:
        found = getattr(verification, "notices", None) or {}
        notices.extend(_clean(found[index]) for index in sorted(found) if found[index])
    # A notice already in the message (an absent target's sentence is also its output) is not repeated.
    lines = [message, *(f"Note: {notice}" for notice in notices if notice and notice not in message)]
    if not dry_run:
        line = unconfirmed_line(results, verification)
        if line:
            lines.append(line)
    return "\n".join(lines)


def unconfirmed_line(results: Sequence[ActionResult], verification: VerificationResult) -> str:
    """The one line about steps that were not confirmed (pilot.agents.unconfirmed.window_line), or "".

    The verifier's reasons are not in it: they go to the receipt, masked."""
    from pilot.agents.unconfirmed import window_line

    kinds = getattr(verification, "reason_kinds", None) or {}
    steps = [
        (result.action.action_type, kinds[index])
        for index, result in enumerate(results)
        if index in kinds and result.success
    ]
    return window_line(steps)


def _success_message_body(
    plan: ActionPlan,
    results: Sequence[ActionResult],
    verification: VerificationResult,
    *,
    dry_run: bool,
) -> str:
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
        and (
            result.action.action_type not in READ_ONLY_ACTIONS
            or result.action.action_type in _EFFECTFUL_READ_ONLY_CATALOG_ENTRIES
        )
        and result.action.permission_tier > PermissionTier.READ_ONLY
        for index, result in enumerate(results)
    )
    # What was not confirmed is said once, by unconfirmed_line: a step that by nature cannot be read (a volume step, a
    # scroll, typing where no reader exists) adds nothing, so an everyday reply carries no caveat a person tires of.
    caveat = ""

    exact_count = exact_labeled_finding_count(plan)
    if exact_count is not None:
        sections = [section for result in results for section in _extract_labeled_sections(result.output)]
        if len(sections) == exact_count:
            empty_finding = "Reported." if unverified_effects else "Verified."
            return (
                "\n".join(
                    f"{index}. {label}: {content or empty_finding}"
                    for index, (label, content) in enumerate(sections, start=1)
                )
                + caveat
            )

    if exact_count is not None and exact_count == len(results):
        findings = []
        empty_finding = "Reported." if unverified_effects else "Verified."
        for index, result in enumerate(results, start=1):
            label = _clean(result.action.target) or result.action.action_type.value.replace("_", " ").title()
            output = _matching_labeled_section(result.output, label)
            findings.append(f"{index}. {label}: {output or empty_finding}")
        return "\n".join(findings) + caveat

    # Each step's own sentence (pilot.result_sentences), never the handler's raw output: a read's answer, or what was
    # done to what. A plan of several steps gets one or two sentences about its outcome; the window's detail view and
    # the receipt keep every step's full output.
    if any(result.success for result in results):
        from pilot.result_sentences import plan_said

        return plan_said(results).shown + caveat

    count = len(results)
    noun = "action" if count == 1 else "actions"
    intent = _clean(plan.explanation)
    prefix = f"Completed {count} {noun}." if unverified_effects else f"Completed and verified {count} {noun}."
    if intent:
        return f"{prefix} {intent}{caveat}"
    if verification.details:
        from pilot.agents.unconfirmed import mask

        return f"{prefix} {_clean(mask(verification.details[0], files=False))}{caveat}"
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

    # A failed step's own sentence (pilot.failure_sentences): a common system failure in plain words that name the
    # thing, any other with the error as the system wrote it, backslashes as written.
    from pilot.failure_sentences import failure_sentence

    issues = [failure_sentence(result.action, result.error)[0] for result in results if result.error]
    if verification:
        from pilot.agents.unconfirmed import mask

        # What a check read can name a window by its title (a mail subject, a document): the title is masked here.
        issues.extend(
            mask(detail, files=False)
            for detail in verification.details
            if "FAILED" in detail or "MISMATCH" in detail or "error" in detail.lower()
        )
    first_issue = next((_clean(issue) for issue in issues if _clean(issue)), "")
    return f"{prefix} {first_issue}" if first_issue else prefix


def success_reply(
    plan: ActionPlan,
    results: Sequence[ActionResult],
    verification: VerificationResult,
    *,
    dry_run: bool,
) -> Said:
    """The success reply in both forms: ``shown`` is ``success_message``; ``spoken`` is the outcome's sentence
    (pilot.result_sentences) and the fixed phrase of a doubt worth saying (pilot.agents.unconfirmed)."""
    from pilot.agents.unconfirmed import spoken_from_window_line
    from pilot.reply_rules import for_ear
    from pilot.result_sentences import plan_said

    shown = success_message(plan, results, verification, dry_run=dry_run)
    if dry_run:
        spoken = "That was a dry run; nothing was changed."
    elif exact_labeled_finding_count(plan) is not None and any(result.success for result in results):
        spoken = "The findings are on screen."
    elif any(result.success for result in results):
        spoken = plan_said(results).spoken
    else:
        spoken = for_ear(shown.splitlines()[0] if shown else "") or "Done."
    _rest, phrase = spoken_from_window_line(shown)
    if phrase and phrase.casefold() not in spoken.casefold():
        spoken = for_ear(f"{spoken} {phrase}.") or spoken
    return Said(shown=shown, spoken=spoken)


def failure_reply(
    results: Sequence[ActionResult],
    verification: VerificationResult | None,
    *,
    follow_up: str = "",
) -> Said:
    """The failure reply in both forms: ``shown`` is ``partial_failure_message``; ``spoken`` is the failed step's own
    sentence (never the system's raw text), or, when a check found something else, that the step did not check out,
    and how many other steps were done. ``follow_up``: a repair round's answer in prose (it asks which file was
    meant, say), shown after the failure; its first question is also said."""
    from pilot.agents.unconfirmed import SPOKEN_PHRASES, Kind
    from pilot.failure_sentences import failure_sentence
    from pilot.reply_rules import for_ear, name_of

    shown = partial_failure_message(results, verification)
    prose = " ".join(str(follow_up or "").split())
    if prose:
        shown = f"{shown}\n{prose}"
    failed_by_check = set(verification.failed_actions if verification else [])
    spoken = ""
    for index, result in enumerate(results):
        if result.error:
            spoken = failure_sentence(result.action, result.error)[1]
            break
        if index in failed_by_check:
            phrase = SPOKEN_PHRASES[Kind.MISMATCH]
            what = name_of(str(getattr(result.action.parameters, "destination", "") or result.action.target or ""))
            lead = f"That didn't work for {what}" if what else "That didn't work"
            spoken = for_ear(f"{lead}: {phrase[:1].lower()}{phrase[1:]}.")
            break
    if not spoken:
        spoken = "That didn't work. The details are on screen."
    others = sum(1 for index, result in enumerate(results) if result.success and index not in failed_by_check)
    if others:
        spoken = for_ear(f"{spoken} {plural_steps(others)} done.") or spoken
    question = next((part for part in re.split(r"(?<=[.!?])\s+", prose) if part.endswith("?")), "")
    if question:
        spoken = for_ear(f"{spoken} {question}") or spoken
    return Said(shown=shown, spoken=spoken)


def plural_steps(count: int) -> str:
    return "The other step is" if count == 1 else f"The other {count} steps are"


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
