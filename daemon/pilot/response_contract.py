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

        return plan_said(results, plan.raw_input).shown + caveat

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
    """The failure reply as shown: the failed step's sentence first (pilot.failure_sentences), for a plan of several
    steps one plain count after it ("1 of 3 steps failed; the other 2 are done."), then on lines of their own what the
    system said or what the check found. Nothing in it is read for meaning: the reply's status and results carry that.
    The first line alone is the sentence."""
    return failure_parts(results, verification)[0]


def failure_parts(
    results: Sequence[ActionResult],
    verification: VerificationResult | None,
) -> tuple[str, str]:
    """(shown, spoken) of a failed plan, both built from its results and the check, never from each other."""
    from pilot.agents.unconfirmed import SPOKEN_PHRASES, Kind, mask
    from pilot.failure_sentences import SYSTEM_SAID, failure_sentence
    from pilot.reply_rules import for_ear, name_of

    total = len(results)
    failed_by_check = {i for i in (verification.failed_actions if verification else []) if isinstance(i, int)}
    failed = sorted(failed_by_check | {i for i, result in enumerate(results) if not result.success})
    sentence, spoken, detail = "", "", ""
    for index in failed:
        if index >= total:
            continue
        result = results[index]
        if result.error:
            shown_sentence, spoken = failure_sentence(result.action, result.error)
            sentence, _newline, detail = shown_sentence.partition("\n")
            break
        if index in failed_by_check:
            phrase = SPOKEN_PHRASES[Kind.MISMATCH]
            what = name_of(str(getattr(result.action.parameters, "destination", "") or result.action.target or ""))
            lead = f"That didn't work for {what}" if what else "That didn't work"
            sentence = f"{lead}: {phrase[:1].lower()}{phrase[1:]}."
            spoken = for_ear(sentence)
            # What the check read, a window title masked (it can be a mail subject or a document's name).
            found = next(
                (
                    mask(item, files=False)
                    for item in (verification.details if verification else [])
                    if item.startswith(f"Action {index} ") and ("MISMATCH" in item or "FAILED" in item)
                ),
                "",
            )
            reason = re.split(r"\s(?:MISMATCH|FAILED)\s\S\s", found, maxsplit=1)[-1].strip() if found else ""
            detail = f"What the check found: {reason}" if reason else ""
            break
    if not sentence:
        sentence = "That didn't work." if total else "Nothing was done, so nothing could be checked."
        spoken = sentence
        reasons = [
            mask(item, files=False)
            for item in (verification.details if verification else [])
            if "FAILED" in item or "MISMATCH" in item or "error" in item.lower()
        ]
        if reasons:
            detail = f"{SYSTEM_SAID}{_clean(reasons[0])}"
    count = ""
    if total > 1:
        others = sum(1 for i, result in enumerate(results) if result.success and i not in failed_by_check)
        count = (
            f"{len(failed)} of {total} steps failed; "
            + ("the other is done." if others == 1 else f"the other {others} are done.")
            if others
            else f"{len(failed)} of {total} steps failed."
        )
    first = f"{sentence} {count}".strip()
    shown = "\n".join(line for line in (first, detail) if line)
    return shown, (for_ear(f"{spoken} {count}") if count else spoken) or spoken


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
        spoken = plan_said(results, plan.raw_input).spoken
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
    asked: tuple[Sequence[ActionResult], VerificationResult | None] | None = None,
) -> Said:
    """The failure reply in both forms (``failure_parts``): the failed step's own sentence, never the system's raw
    text aloud. ``follow_up``: a repair round's answer in prose (it asks which file was meant, say), shown after the
    failure; its first question is also said. ``asked``: the results and check of the first attempt of the plan the
    person asked for. When it failed, the reply leads with its failure and counts its steps: a repair step that
    failed afterwards (a command planned to work around it) is never what the person hears first."""
    from pilot.reply_rules import for_ear

    retry = ""
    if asked is not None:
        asked_results, asked_check = asked
        if any(not result.success for result in asked_results) or (asked_check and asked_check.failed_actions):
            retry = repair_clause(asked_results, results)
            results, verification = asked_results, asked_check
    shown, spoken = failure_parts(results, verification)
    if retry:
        shown, spoken = f"{shown}\n{retry}", for_ear(f"{spoken} {retry}") or spoken
    prose = " ".join(str(follow_up or "").split())
    if prose:
        shown = f"{shown}\n{prose}"
    question = next((part for part in re.split(r"(?<=[.!?])\s+", prose) if part.endswith("?")), "")
    if question:
        spoken = for_ear(f"{spoken} {question}") or spoken
    return Said(shown=shown, spoken=spoken)


def repair_clause(asked: Sequence[ActionResult], final: Sequence[ActionResult]) -> str:
    """What happened to the retry, in one clause, when a repair round ran after the plan the person asked for failed:
    refused before it ran ("I did not try again: I could not keep a copy of the earlier file, so I did not replace
    it.") or tried and failed too. "" when no repair step ran."""
    from pilot.agents.request_match import REFUSED_REPAIR
    from pilot.failure_sentences import failure_sentence
    from pilot.reply_rules import for_ear

    seen = {id(result) for result in asked}
    later = [result for result in final if id(result) not in seen and not result.success]
    if not later:
        return ""
    first = later[0]
    error = " ".join(str(first.error or "").split())
    earlier = next((" ".join(str(r.error or "").split()) for r in asked if not r.success and r.error), "")
    if first.executed and not first.skip_reason and error and error == earlier:
        return "I tried again, and it failed the same way."
    # Heliox's own sentences start with "I " and are said as they are; a system's error gets its plain sentence.
    reason = error if error.startswith("I ") else failure_sentence(first.action, error)[1] if error else ""
    reason = reason.rstrip(".") + "." if reason else ""
    if not first.executed or first.skip_reason:
        return for_ear(f"{REFUSED_REPAIR}: {reason[:1].lower() + reason[1:] if reason[:2] != 'I ' else reason}")
    return for_ear(
        f"I tried again, and that failed too: {reason[:1].lower() + reason[1:] if reason[:2] != 'I ' else reason}"
    )


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
