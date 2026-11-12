"""Scoring: word error rate, tolerance matching, each scenario against its expectations, and the scorecard (pure).

The scorecard is kept apart per voice ("real" never mixed with "synthetic"), per condition (as recorded, or a
derived one such as a Bluetooth call-mode headset or an open office) and per kind of model (cloud, local, none).
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from . import rubric
from .scenarios import Scenario
from .targets import BY_ID, LINES, passes
from .timeline import Measures

_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]  # fmt: skip
_TENS = ["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
        "twelve": "twelfth", "twenty": "twentieth", "thirty": "thirtieth"}  # fmt: skip
_SAME = {"ok": "okay", "heliox's": "heliox", "todo": "to do", "to-do": "to do", "o'clock": "oclock",
         "i'll": "i will", "what's": "what is", "it's": "it is", "don't": "do not", "can't": "cannot",
         "won't": "will not", "i'm": "i am", "let's": "let us", "that's": "that is"}  # fmt: skip


def number_words(n: int, ordinal: bool = False) -> str:
    """0..99 in words ("thirty one", "twelfth"), the way a recogniser may write a spoken number."""
    if n < 20:
        word = _UNITS[n]
        return (_ORD.get(word) or word + "th") if ordinal else word
    tens, unit = divmod(n, 10)
    if unit == 0:
        return (_ORD.get(_TENS[tens]) or _TENS[tens] + "th") if ordinal else _TENS[tens]
    return f"{_TENS[tens]} {number_words(unit, ordinal)}"


def normalise(text: str) -> list[str]:
    text = text.casefold().replace("-", " ").replace("’", "'")
    out: list[str] = []
    for token in re.findall(r"[a-z0-9']+", text):
        token = token.strip("'")
        match = re.fullmatch(r"(\d{1,2})(st|nd|rd|th)?", token)
        if match:
            token = number_words(int(match.group(1)), ordinal=bool(match.group(2)))
        token = _SAME.get(token, token)
        out.extend(token.split())
    return out


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    row = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, row[0] = row[0], i
        for j in range(1, len(hyp) + 1):
            cur = min(row[j] + 1, row[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev, row[j] = row[j], cur
    return row[len(hyp)]


def wer(reference: str, hypothesis: str) -> float:
    ref = normalise(reference)
    return edit_distance(ref, normalise(hypothesis)) / max(1, len(ref))


def within(reference: str, hypothesis: str, tolerance: float) -> bool:
    """The tolerance match: the heard words are within ``tolerance`` word error rate of what was said."""
    return wer(reference, hypothesis) <= tolerance + 1e-9


WAKE = re.compile(r"^\s*(hey|hi|okay|ok)?[\s,]*heliox\b", re.IGNORECASE)


def has_wake_phrase(text: str) -> bool:
    return bool(WAKE.match(text))


@dataclass
class Outcome:
    """One scenario run under one condition and kind of model."""

    scenario: Scenario
    condition: str
    model: str
    measures: Measures
    passed: bool
    failures: list[str] = field(default_factory=list)
    contributions: list[tuple[str, Any]] = field(default_factory=list)
    heard: str = ""


def _route_ok(expected: Any, got: str) -> bool:
    options = expected if isinstance(expected, list) else [expected]
    return "any" in options or got in options


def _spoken_ok(expected: Any, spoken: list[str]) -> bool:
    if expected == "any" or expected is None:
        return True
    if expected is True:
        return bool(spoken)
    if expected is False:
        return not spoken
    return any(str(expected).casefold() in s.casefold() for s in spoken)


def expectations_for(scenario: Scenario, model: str) -> dict[str, Any]:
    """The scenario's expectations with any per-model override (``expect_by_model``) applied."""
    expect = dict(scenario.expect)
    expect.update((scenario.expect.get("by_model") or {}).get(model, {}))
    expect.pop("by_model", None)
    return expect


def score(scenario: Scenario, measures: Measures, *, condition: str = "as-recorded", model: str = "cloud") -> Outcome:
    expect = expectations_for(scenario, model)
    m = measures
    failures: list[str] = []
    contrib: list[tuple[str, Any]] = []
    heard = " ".join(m.transcripts)

    if "woken" in expect and bool(m.woken or m.follow_up) != bool(expect["woken"]):
        failures.append(f"woken={m.woken or m.follow_up}, expected {expect['woken']}")
    if "route" in expect and not _route_ok(expect["route"], m.route):
        failures.append(f"route={m.route}, expected {expect['route']}")
    if "spoken" in expect and not _spoken_ok(expect["spoken"], m.spoken):
        failures.append(f"spoken={m.spoken[:2]}, expected {expect['spoken']}")
    if "card" in expect and m.card != bool(expect["card"]):
        failures.append(f"card={m.card}, expected {expect['card']}")
    if expect.get("stopped") and m.stop_time is None:
        failures.append("Heliox's voice was not cut")
    if expect.get("stopped") is False and m.stop_time is not None:
        failures.append("Heliox's voice was cut by a sound that was not the person speaking to it")
    if expect.get("approved") is False and "confirmed=True" in m.approval:
        failures.append("approved by voice")
    if expect.get("denied") and "confirmed=False resolved=True" not in m.approval and m.approval != "gone":
        failures.append(f"not denied ({m.approval or 'no state'})")
    tolerance = float(expect.get("wer_max", 0.08))
    said = scenario.said
    if said and scenario.reference_confirmed and expect.get("transcript", True) and scenario.set == "understanding":
        if not within(said, heard, tolerance):
            failures.append(f"heard {heard!r} (WER {wer(said, heard):.2f} > {tolerance})")
        ref = normalise(said)
        contrib.append(("2.1a", (edit_distance(ref, normalise(heard)), len(ref))))
    if expect.get("woken") and has_wake_phrase(said):
        contrib.append(("2.1b", bool(m.woken)))
    if expect.get("follow_up"):
        contrib.append(("2.1c", m.follow_up and m.route not in ("ignored", "overheard")))
    if scenario.voice == "real" and said and scenario.set in ("understanding", "turn_taking"):
        sentences = sum(1 for s in scenario.steps if s.say)
        contrib.append(("1.3", len(m.transcripts) > sentences))
    elif scenario.set == "turn_taking" and expect.get("one_turn"):
        contrib.append(("1.3", len(m.transcripts) > 1))
    if "card" in expect and expect["card"]:
        contrib.append(("3.1", m.card))
    addressed = scenario.set != "not_addressed"  # speech for someone else is scored by 2.2 alone
    if addressed:
        contrib.append(("1.2a", m.talk_over))
    if expect.get("stopped"):
        contrib.append(("1.2b", m.stop_time if m.stop_time is not None else float("inf")))
        contrib.append(("1.2c", m.interrupted_heard))
    elif expect.get("stopped") is False:
        contrib.append(("1.2d", m.stop_time is None))
    if expect.get("reply") and m.reply_start is not None:
        contrib.append(("1.1b" if m.used_model else "1.1a", m.reply_start))
        if m.used_model:
            contrib.append(("1.1c", m.reply_start))
    elif expect.get("reply"):
        failures.append("no spoken reply")
        contrib.append(("1.1b" if m.used_model else "1.1a", float("inf")))
    if m.silent_gap is not None and expect.get("reply"):
        contrib.append(("1.1d", m.silent_gap))
    if expect.get("under_way") and scenario.set != "not_addressed":
        contrib.append(("4.1", m.under_way if m.under_way is not None else float("inf")))
    previous = ""
    for line in m.spoken if addressed else []:
        result = rubric.check(line, previous=previous, action_happened=True)
        contrib.append(("1.4a", result.passed))
        previous = line
    if addressed:
        contrib.append(("1.4b", rubric.repeats_running(m.spoken)))
    if scenario.set == "not_addressed":
        acted = len(m.dispatched) + len(m.spoken) + m.stored
        contrib.append(("2.2", acted))
        if acted:
            failures.append(f"acted on {m.dispatched[:3]} spoke {m.spoken[:2]} stored {m.stored}")
    passed = not failures
    if scenario.set == "control":
        contrib.append(("4.2", passed))
    return Outcome(scenario, condition, model, m, passed, failures, contrib, heard)


@dataclass
class Row:
    line: str
    voice: str
    condition: str
    model: str
    value: float | None
    n: int
    ok: bool | None
    detail: str = ""


def _combine(line_id: str, values: list[Any]) -> tuple[float | None, str]:
    line = BY_ID[line_id]
    if not values:
        return None, ""
    if line.statistic == "wer":
        edits = sum(v[0] for v in values)
        words = sum(v[1] for v in values)
        return (edits / words if words else None), f"{edits} edits / {words} words"
    if line.statistic in ("share", "rate"):
        hits = sum(1 for v in values if v)
        return hits / len(values), f"{hits}/{len(values)}"
    if line.statistic == "count":
        return float(sum(values)), f"over {len(values)} scenarios"
    finite = [float(v) for v in values]
    if line.statistic == "median":
        return statistics.median(finite), f"n={len(finite)}"
    return max(finite), f"n={len(finite)}"


def scorecard(outcomes: list[Outcome], extra: list[Row] | None = None) -> list[Row]:
    groups: dict[tuple[str, str, str, str], list[Any]] = defaultdict(list)
    for o in outcomes:
        for line_id, value in o.contributions:
            groups[(line_id, o.scenario.voice, o.condition, o.model)].append(value)
    rows: list[Row] = []
    order = {line.id: i for i, line in enumerate(LINES)}
    for (line_id, voice, condition, model), values in sorted(
        groups.items(), key=lambda kv: (order.get(kv[0][0], 99), kv[0][1], kv[0][2], kv[0][3])
    ):
        value, detail = _combine(line_id, values)
        rows.append(Row(line_id, voice, condition, model, value, len(values), passes(BY_ID[line_id], value), detail))
    return rows + list(extra or [])


def worst(outcomes: list[Outcome], limit: int = 10) -> list[Outcome]:
    """Failed scenarios first, then the slowest replies."""

    def key(o: Outcome) -> tuple[int, float]:
        return (0 if not o.passed else 1, -(o.measures.reply_start or 0.0))

    return sorted(outcomes, key=key)[:limit]
