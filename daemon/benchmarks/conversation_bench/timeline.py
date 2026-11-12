"""A run's timeline and what can be read off it (pure).

Events are ``{"t": seconds, "event": kind, "detail": text, "data": {...}}`` on one clock. The driver writes them;
voice_probe.py's JSON (``{"events": [...]}``, no "data") parses too, with its older event names mapped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Event kinds the driver emits.
SCENARIO = "scenario"
PERSON_STARTS = "person starts"
FIRST_WORD = "person first word"
LAST_WORD = "person last word"
PERSON_STOPS = "person stops"
TRANSCRIPT = "transcript"
WAKE = "wake"
FOLLOW_UP = "follow-up"
AMBIENT_IGNORED = "ambient ignored"
DISPATCH = "dispatch"
OVERHEARD = "overheard"
PLANNER = "planner call"
PLANNER_CANCELLED = "planner cancelled"
WITHDRAW = "withdraw"
INTERJECT = "interject call"
RESULT = "voice result"
SPEAKS = "heliox speaks"
AUDIO_STARTS = "heliox audio starts"
DONE = "heliox done"
CUT_OFF = "heliox cut off"
BARGE_IN = "barge-in detected"
CARD = "card"
APPROVAL = "approval state"
STORED = "stored"
SETUP_SPEECH = "setup speech"

_PROBE_NAMES = {
    "person's last word ends": LAST_WORD,
    "command heard": DISPATCH,
    "overheard (to Live)": OVERHEARD,
    "planner (stubbed)": PLANNER,
    "plan correction (stubbed)": INTERJECT,
    "live_interjection": CARD,
    "voice_result": RESULT,
    "=== scenario": SCENARIO,
}


@dataclass(frozen=True)
class Event:
    t: float
    event: str
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def parse(raw: Any) -> list[Event]:
    """Events from the driver's or the probe's JSON (a dict with "events", a list, or a JSON string)."""
    if isinstance(raw, str):
        raw = json.loads(raw)
    items = raw.get("events", []) if isinstance(raw, dict) else raw
    out = []
    for item in items:
        kind = str(item.get("event", ""))
        out.append(
            Event(
                t=float(item.get("t", 0.0)),
                event=_PROBE_NAMES.get(kind, kind),
                detail=str(item.get("detail", "")),
                data=dict(item.get("data") or {}),
            )
        )
    return sorted(out, key=lambda e: e.t)


def to_json(events: list[Event]) -> list[dict[str, Any]]:
    return [
        {"t": round(e.t, 3), "event": e.event, "detail": e.detail, **({"data": e.data} if e.data else {})}
        for e in events
    ]


def first(events: list[Event], kind: str, after: float = float("-inf")) -> Event | None:
    return next((e for e in events if e.event == kind and e.t >= after), None)


def all_of(events: list[Event], kind: str) -> list[Event]:
    return [e for e in events if e.event == kind]


@dataclass
class Measures:
    """What one scenario's timeline shows. Times are seconds; None when the moment never happened."""

    last_word: float | None = None
    reply_start: float | None = None  # Heliox's voice after the person's last word
    under_way: float | None = None  # planner, interject, local answer or control result after the last word
    silent_gap: float | None = None  # longest stretch after the last word with nothing said while work goes on
    talk_over: int = 0
    stop_time: float | None = None  # Heliox's voice cut after the person's first word over it
    interrupted_heard: bool = False
    transcripts: list[str] = field(default_factory=list)
    woken: bool = False
    follow_up: bool = False
    route: str = "ignored"
    spoken: list[str] = field(default_factory=list)
    dispatched: list[str] = field(default_factory=list)
    card: bool = False
    approval: str = ""
    stored: int = 0
    used_model: bool = False


def measure(events: list[Event]) -> Measures:
    """Read one scenario's events (already cut to that scenario)."""
    m = Measures()
    words = all_of(events, LAST_WORD)
    firsts = all_of(events, FIRST_WORD)
    m.last_word = words[-1].t if words else None
    setup_end = max((e.t for e in events if e.event == SETUP_SPEECH), default=float("-inf"))
    audio = [e for e in all_of(events, AUDIO_STARTS) if e.t > setup_end or not e.data.get("setup")]
    own_audio = [e for e in audio if not e.data.get("setup")]

    # Starting to speak while the person is mid-sentence (setup speech that was already playing is barge-in).
    spans = list(zip(firsts, words, strict=False))
    for start in own_audio:
        if any(a.t < start.t < b.t for a, b in spans):
            m.talk_over += 1

    if m.last_word is not None:
        reply = next((e for e in own_audio if e.t >= m.last_word), None)
        m.reply_start = None if reply is None else reply.t - m.last_word
        markers = [
            e
            for e in events
            if e.t >= m.last_word and e.event in (PLANNER, INTERJECT, RESULT, AUDIO_STARTS) and not e.data.get("setup")
        ]
        m.under_way = (markers[0].t - m.last_word) if markers else None
        m.silent_gap = _silent_gap(events, m.last_word)

    # Heliox's voice stopped: cancelled, or ended by the barge-in path (which can return normally).
    cut = first(events, CUT_OFF)
    barge = first(events, BARGE_IN)
    if barge is not None:
        ended = next((e for e in events if e.event in (CUT_OFF, DONE) and e.t >= barge.t), None)
        if ended is not None and (cut is None or ended.t < cut.t):
            cut = ended
    if cut is not None and firsts:
        over = [f for f in firsts if f.t <= cut.t]
        if over:
            m.stop_time = cut.t - over[-1].t
    m.interrupted_heard = bool(all_of(events, BARGE_IN))
    m.transcripts = [e.detail for e in all_of(events, TRANSCRIPT) if e.detail.strip()]
    m.woken = bool(all_of(events, WAKE))
    m.follow_up = bool(all_of(events, FOLLOW_UP))
    m.spoken = [e.detail for e in all_of(events, SPEAKS) if not e.data.get("setup")]
    m.dispatched = [e.detail for e in all_of(events, DISPATCH)]
    m.card = bool(all_of(events, CARD))
    approvals = all_of(events, APPROVAL)
    m.approval = approvals[-1].detail if approvals else ""
    m.stored = sum(int(e.data.get("rows", 0)) for e in all_of(events, STORED))
    m.used_model = bool(all_of(events, PLANNER))
    m.route = _route(events)
    return m


def _route(events: list[Event]) -> str:
    if all_of(events, INTERJECT):
        return "interject"
    if all_of(events, PLANNER):
        return "planner"
    results = all_of(events, RESULT)
    statuses = [str(e.data.get("status", e.detail)) for e in results]
    if "approval_required" in statuses:
        return "refuse_approval"
    messages = " ".join(str(e.data.get("message", "")) for e in results).casefold()
    if any(e.data.get("plan_id") for e in results) and "won't run" in messages:
        return "deny"
    commands = " ".join(e.detail for e in all_of(events, DISPATCH)).casefold()
    if results and ("stopped" in messages or commands.strip(" .") in {"stop", "wait", "cancel"}):
        return "control"
    if results or all_of(events, DISPATCH):
        return "local"
    if all_of(events, OVERHEARD):
        return "overheard"
    return "ignored"


def _silent_gap(events: list[Event], since: float) -> float | None:
    """Longest stretch after ``since`` with no voice while Heliox still owed an answer (until its last result)."""
    ends = [e.t for e in events if e.event in (RESULT, DONE, CUT_OFF) and e.t >= since]
    if not ends:
        return None
    until = max(ends)
    voiced: list[tuple[float, float]] = []
    start = None
    for e in events:
        if e.t < since or e.data.get("setup"):
            continue
        if e.event == AUDIO_STARTS and start is None:
            start = e.t
        elif e.event in (DONE, CUT_OFF) and start is not None:
            voiced.append((start, e.t))
            start = None
    if start is not None:
        voiced.append((start, until))
    gap, cursor = 0.0, since
    for a, b in voiced:
        gap = max(gap, a - cursor)
        cursor = max(cursor, b)
    if cursor < until:
        gap = max(gap, until - cursor)
    return gap


def slice_scenarios(events: list[Event]) -> dict[str, list[Event]]:
    """Cut a whole run into scenarios at each SCENARIO marker (detail is the scenario id)."""
    out: dict[str, list[Event]] = {}
    current = ""
    for e in events:
        if e.event == SCENARIO:
            current = e.detail
            out.setdefault(current, [])
        elif current:
            out[current].append(e)
    return out
