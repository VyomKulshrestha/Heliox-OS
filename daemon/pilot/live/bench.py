"""Heliox Live Bench: replay typing and speech, and score when the live layer spoke up.

An interruption is only useful if it is right and on time. This bench runs the real ``LiveCompanion`` (same
detectors, same policy, same personal timing model) against scripted sessions on a simulated clock, and counts:

- **catches**: expected corrections that were delivered;
- **missed**: expected corrections that never came (or came after the person pressed Enter);
- **false alarms**: anything delivered that the scenario did not expect, including slips the person was about
  to fix themselves;
- **cut-offs**: moments Live spoke up while the person was still typing, thinking mid-sentence, or talking
  (two cards arriving together are one interruption, so they count once);
- **wait**: seconds from the moment the person really stopped to the delivery (lower is better, when on time).

Each scenario is typed by three personas with different rhythms (``quick``, ``steady``, ``thinker``), so a policy
that works only for fast typists shows up. Before the scored scenarios each persona types a warm-up of plain
sentences, which is when the personal timing model learns their rhythm; the warm-up is not scored. The simulated
person reacts like a person: a card that arrives at a real break gets read and acted on, and a card that cut in
gets waved away when they are done, which is the feedback the timing model learns from.

Scenarios are JSON (``pilot/live/bench_scenarios/*.json``; the format is in ``docs/live-bench.md``) and contain no
personal data. Run it with ``python -m pilot.live.bench``; add ``--scenarios my_pack.json`` to include more.
Everything is local and deterministic (seeded jitter), so two runs on any machine give the same numbers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pilot.live.detectors import KINDS
from pilot.live.engine import TICK_SECONDS, LiveCompanion, Observation

SCENARIO_DIR = Path(__file__).with_name("bench_scenarios")
POLICIES = ("rules", "personal")
BUSY = frozenset({"typing", "think", "speaking"})
REACTION_SECONDS = 1.2  # reading a card and acting on it
SETTLE_SECONDS = 3.0  # quiet time after the last step, so late deliveries still count
FIX_TYPING_SECONDS = 0.6


@dataclass(frozen=True)
class Persona:
    name: str
    chars_per_second: float
    think_seconds: float  # a pause mid-sentence to find the next words
    done_seconds: float  # a pause after a finished thought
    jitter: float = 0.25


PERSONAS = {
    "quick": Persona("quick", 8.0, 0.8, 5.0),
    "steady": Persona("steady", 5.5, 1.5, 6.0),
    "thinker": Persona("thinker", 4.0, 2.6, 7.0),
}

WARM_UP = [
    ("I read through the draft this morning and", " the second section reads much better now."),
    ("The numbers from last week look", " about the same as the month before."),
    ("We should ask the design team whether", " they have time for one more round."),
    ("My plan for the afternoon is to", " finish the summary and send it around."),
    ("If the build passes again tonight", " we can tag the release tomorrow."),
    ("The meeting notes are in the shared folder", " under this week's date."),
    ("I'm not sure the second option", " is worth the extra cost."),
    ("Could you check with the venue whether", " they still have the larger room?"),
    ("Most of the feedback so far", " has been about the onboarding flow."),
    ("Let's keep the scope small and", " ship the first part on its own."),
    ("The new laptop arrived but", " it still needs the usual setup."),
    ("I'll write up the decision so", " nobody has to dig through the thread."),
]


@dataclass
class Step:
    at: float
    until: float
    label: str
    kind: str  # "type", "pause", "replace", "set", "paste", "send", "say"
    text: str = ""
    old: str = ""
    seconds: float = 0.0


@dataclass
class Delivery:
    scenario: str
    kind: str
    at: float
    label: str
    expected: bool
    wait: float | None
    spoken: bool
    source: str


@dataclass
class Score:
    policy: str
    persona: str
    expected: int = 0
    catches: int = 0
    missed: int = 0
    false_alarms: int = 0
    cut_offs: int = 0
    deliveries: int = 0
    waits: list[float] = field(default_factory=list)
    learned_mid_sentence_seconds: float = 0.0
    details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.catches / self.deliveries if self.deliveries else 1.0

    @property
    def recall(self) -> float:
        return self.catches / self.expected if self.expected else 1.0

    @property
    def cut_off_rate(self) -> float:
        return self.cut_offs / self.deliveries if self.deliveries else 0.0

    @property
    def median_wait(self) -> float:
        return round(statistics.median(self.waits), 2) if self.waits else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "persona": self.persona,
            "expected": self.expected,
            "catches": self.catches,
            "missed": self.missed,
            "false_alarms": self.false_alarms,
            "cut_offs": self.cut_offs,
            "deliveries": self.deliveries,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "cut_off_rate": round(self.cut_off_rate, 3),
            "median_wait_seconds": self.median_wait,
            "learned_mid_sentence_seconds": self.learned_mid_sentence_seconds,
            "details": self.details,
        }


# ── scenarios ──


def load_scenarios(extra: list[Path] | None = None) -> list[dict[str, Any]]:
    packs = sorted(SCENARIO_DIR.glob("*.json")) + list(extra or [])
    scenarios: list[dict[str, Any]] = []
    for pack in packs:
        for scenario in json.loads(Path(pack).read_text(encoding="utf-8")):
            validate(scenario, str(pack))
            scenarios.append(scenario)
    return scenarios


def validate(scenario: dict[str, Any], where: str = "") -> None:
    name = scenario.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{where}: every scenario needs a name")
    for kind in scenario.get("expect", []):
        if kind not in KINDS:
            raise ValueError(f"{where}: {name}: unknown correction kind {kind!r}")
    steps = scenario.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{where}: {name}: needs steps")
    allowed = {
        "type", "pause", "replace", "paste", "send", "say", "label", "seconds", "switch", "text", "delete", "notice",
        "source",
    }  # fmt: skip
    for step in steps:
        if not isinstance(step, dict) or not set(step) <= allowed:
            raise ValueError(f"{where}: {name}: bad step {step!r}")


def expand(scenario: dict[str, Any], persona: Persona, rng: random.Random) -> list[Step]:
    """Turn a scenario's steps into a timed script for one persona."""

    def vary(seconds: float) -> float:
        return round(seconds * rng.uniform(1 - persona.jitter, 1 + persona.jitter), 2)

    t = 0.0
    script: list[Step] = []
    for raw in scenario["steps"]:
        if "type" in raw:
            text = raw["type"]
            seconds = len(text) / persona.chars_per_second
            script.append(Step(t, t + seconds, "typing", "type", text=text, seconds=seconds))
        elif "paste" in raw:
            script.append(Step(t, t + 0.2, "typing", "paste", text=raw["paste"], seconds=0.2))
            seconds = 0.2
        elif "replace" in raw:
            old, new = raw["replace"]
            script.append(Step(t, t + FIX_TYPING_SECONDS, "typing", "replace", text=new, old=old))
            seconds = FIX_TYPING_SECONDS
        elif "send" in raw:
            script.append(Step(t, t + 0.4, "idle", "send", seconds=0.4))
            seconds = 0.4
        elif "delete" in raw:
            # Backspacing: twice typing speed, one character at a time.
            count = int(raw["delete"])
            seconds = count / (persona.chars_per_second * 2)
            script.append(Step(t, t + seconds, "typing", "delete", seconds=seconds, text="x" * count))
        elif "switch" in raw:
            # Another window comes to the front: {"switch": "app|title", "text": "what its field holds"}.
            script.append(Step(t, t + 0.6, "idle", "switch", text=raw["switch"], old=raw.get("text", ""), seconds=0.6))
            seconds = 0.6
        elif "notice" in raw:
            # Supervision matched a risk rule: {"notice": "credential_exposure", "source": "ocr" or "keystroke"}.
            script.append(Step(t, t, "idle", "notice", text=raw["notice"], old=raw.get("source", "ocr")))
            seconds = 0.0
        elif "say" in raw:
            seconds = float(raw.get("seconds", 2.5))
            script.append(Step(t, t + seconds, "speaking", "say", text=raw["say"], seconds=seconds))
        elif "pause" in raw:
            pause = raw["pause"]
            if pause == "think":
                seconds, label = vary(persona.think_seconds), "think"
            elif pause == "done":
                seconds, label = vary(persona.done_seconds), "done"
            else:
                seconds, label = float(pause), str(raw.get("label", "done"))
            script.append(Step(t, t + seconds, label, "pause", seconds=seconds))
        else:
            raise ValueError(f"bad step {raw!r}")
        t += seconds
    script.append(Step(t, t + SETTLE_SECONDS, "done", "pause", seconds=SETTLE_SECONDS))
    return script


def warm_up(persona: Persona) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    for first, rest in WARM_UP + WARM_UP[::-1]:
        steps += [{"type": first}, {"pause": "think"}, {"type": rest}, {"pause": 2.0}]
    return {"name": f"warm-up ({persona.name})", "app": "notepad.exe", "title": "warm-up.txt - Notepad", "steps": steps}


# ── one simulated person ──


class Session:
    """The document, what the person is doing, and how they react to cards, on the simulated clock."""

    def __init__(self, scenario: dict[str, Any], script: list[Step], start: float) -> None:
        self.scenario = scenario
        self.script = script
        self.start = start
        self.end = start + script[-1].until
        self.window = f"{scenario['app']}|{scenario['title']}"
        self.docs: dict[str, str] = {self.window: scenario.get("start_text", "")}
        self.index = 0
        self.typed = 0  # characters of the current typing step already in the document
        self._edit_at = -1
        self.reactions: list[tuple[float, dict[str, Any]]] = []
        self.said: list[str] = []
        self.notices: list[tuple[str, str]] = []
        # False: focus is on page text (a web page, a chat transcript), which the person reads but does not write.
        self.editable = bool(scenario.get("editable", True))

    @property
    def doc(self) -> str:
        return self.docs[self.window]

    @doc.setter
    def doc(self, text: str) -> None:
        self.docs[self.window] = text

    def step_at(self, now: float) -> Step:
        rel = now - self.start
        for step in self.script:
            if step.at <= rel < step.until:
                return step
        return self.script[-1]

    def label(self, now: float) -> str:
        return self.step_at(now).label

    def advance(self, now: float) -> None:
        rel = now - self.start
        while self.index < len(self.script):
            step = self.script[self.index]
            if rel < step.at:
                break
            if step.kind == "type":
                count = min(len(step.text), int((rel - step.at) * len(step.text) / max(step.seconds, 1e-6)) + 1)
                if count > self.typed:
                    self.doc += step.text[self.typed : count]
                    self.typed = count
                if self.typed < len(step.text):
                    break
            elif step.kind == "delete":
                count = min(len(step.text), int((rel - step.at) * len(step.text) / max(step.seconds, 1e-6)) + 1)
                if count > self.typed:
                    self.doc = self.doc[: max(0, len(self.doc) - (count - self.typed))]
                    self.typed = count
                if self.typed < len(step.text):
                    break
            elif step.kind == "paste":
                self.doc += step.text
            elif step.kind == "replace":
                # Delete the old words at once (the text changes as soon as they start editing), retype after.
                if not self.typed:
                    head, sep, tail = self.doc.rpartition(step.old)
                    self._edit_at = len(head) if sep else -1
                    if sep:
                        self.doc = head + tail
                    self.typed = 1
                if rel < step.until:
                    break
                if self._edit_at >= 0:
                    self.doc = self.doc[: self._edit_at] + step.text + self.doc[self._edit_at :]
            elif step.kind == "send":
                self.doc = ""
            elif step.kind == "switch":
                self.window = step.text
                self.docs.setdefault(self.window, step.old)
            elif step.kind == "notice":
                self.notices.append((step.text, step.old))
            elif step.kind == "say":
                if rel < step.until:
                    break
                self.said.append(step.text)
            self.index += 1
            self.typed = 0
        for due, card in list(self.reactions):
            if now >= due:
                self.reactions.remove((due, card))
                self._act_on(card)

    def _act_on(self, card: dict[str, Any]) -> None:
        evidence, fix = card.get("evidence", ""), card.get("fix", "")
        if evidence and evidence in self.doc:
            self.doc = self.doc.replace(evidence, fix if card.get("fix_kind") in ("replace", "command") else "", 1)

    def speaking(self, now: float) -> bool:
        return self.label(now) == "speaking"

    def stopped_at(self, now: float) -> float:
        """When the current quiet stretch began."""
        rel = now - self.start
        began = rel
        for step in reversed(self.script):
            if step.at > rel:
                continue
            if step.label in BUSY:
                break
            began = step.at
        return self.start + began


async def run(
    policy: str,
    persona: Persona,
    scenarios: list[dict[str, Any]],
    *,
    level: str = "balanced",
    seed: int = 7,
) -> Score:
    # Each scenario has its own seed, so adding a scenario never changes the pauses in another one.
    clock = {"t": 1000.0}
    state: dict[str, Any] = {"session": None}
    settings = SimpleNamespace(
        enabled=True,
        level=level,
        listen_to_speech=True,
        muted_kinds=[],
        personal_timing=policy == "personal",
    )
    deliveries: list[tuple[dict[str, Any], float]] = []

    async def observe() -> Observation | None:
        session = state["session"]
        if session is None:
            return None
        app, _, title = session.window.partition("|")
        # Page text is passed on here (the real reader never sends it) to show the engine ignores it anyway.
        return Observation(text=session.doc, app=app, title=title, source="focused_field", editable=session.editable)

    async def emit(event: str, payload: dict[str, Any]) -> None:
        if event == "live_interjection":
            deliveries.append((payload, clock["t"]))

    async def speak(message: str, urgent: bool) -> None:
        return None

    def people() -> dict[str, str]:
        session = state["session"]
        return dict(session.scenario.get("people", {})) if session else {}

    def today() -> date:
        session = state["session"]
        raw = session.scenario.get("today", "2026-11-09") if session else "2026-11-09"
        return date.fromisoformat(raw)

    live = LiveCompanion(
        settings=lambda: settings,
        observe=observe,
        emit=emit,
        speak=speak,
        user_speaking=lambda: bool(state["session"] and state["session"].speaking(clock["t"])),
        people=people,
        clock=lambda: clock["t"],
        today=today,
        # Wall-clock time for deadlines: 11:00 on the scenario's day, plus the simulated seconds.
        wallclock=lambda: (
            datetime.combine(today(), datetime.min.time()) + timedelta(hours=11, seconds=clock["t"] - 1000)
        ),
    )
    score = Score(policy, persona.name)

    async def play(scenario: dict[str, Any], scored: bool) -> None:
        rng = random.Random(f"{seed}:{persona.name}:{scenario['name']}")
        session = Session(scenario, expand(scenario, persona, rng), clock["t"])
        state["session"] = session
        seen = len(deliveries)
        expected = list(scenario.get("expect", []))
        cut_in: set[str] = set()
        cut_moments: set[float] = set()
        while clock["t"] < session.end:
            session.advance(clock["t"])
            for text in session.said:
                live.offer_speech(text)
            session.said.clear()
            for pattern, source in session.notices:
                await live.notice_risk(pattern, source)
            session.notices.clear()
            await live.tick()
            await asyncio.sleep(0)
            for payload, at in deliveries[seen:]:
                seen += 1
                label = session.label(at)
                busy = label in BUSY
                if busy:
                    cut_in.add(payload["id"])
                elif payload.get("source") != "speech":
                    session.reactions.append((at + REACTION_SECONDS, payload))
                hit = payload["kind"] in expected
                if hit:
                    expected.remove(payload["kind"])
                if scored:
                    score.deliveries += 1
                    score.catches += int(hit)
                    score.false_alarms += int(not hit)
                    if busy and at not in cut_moments:
                        cut_moments.add(at)
                        score.cut_offs += 1
                    wait = None if busy else round(at - session.stopped_at(at), 2)
                    if wait is not None and hit:
                        score.waits.append(wait)
                    score.details.append(
                        Delivery(
                            scenario["name"],
                            payload["kind"],
                            round(at - session.start, 2),
                            label,
                            hit,
                            wait,
                            bool(payload.get("spoken")),
                            str(payload.get("source", "")),
                        ).__dict__
                    )
            clock["t"] = round(clock["t"] + TICK_SECONDS, 3)
        if scored:
            score.expected += len(scenario.get("expect", []))
            score.missed += len(expected)
            for kind in expected:
                score.details.append({"scenario": scenario["name"], "kind": kind, "missed": True})
        # The person is done: cards that cut in are waved away, the rest were read.
        for card_id in list(live._visible):
            await live.feedback(card_id, "dismissed" if card_id in cut_in else "helpful")
        live._pending.clear()
        state["session"] = None
        clock["t"] += 2.0
        await live.tick()

    await play(warm_up(persona), scored=False)
    for scenario in scenarios:
        await play(scenario, scored=True)
    score.learned_mid_sentence_seconds = live.timing.mid_sentence_pause(1.2) if policy == "personal" else 1.2
    return score


async def run_all(
    scenarios: list[dict[str, Any]] | None = None,
    *,
    personas: list[str] | None = None,
    policies: tuple[str, ...] = POLICIES,
    level: str = "balanced",
) -> list[Score]:
    scenarios = scenarios if scenarios is not None else load_scenarios()
    results = []
    for policy in policies:
        for name in personas or list(PERSONAS):
            results.append(await run(policy, PERSONAS[name], scenarios, level=level))
    return results


def table(scores: list[Score]) -> str:
    head = (
        f"{'policy':<9} {'persona':<8} {'caught':>7} {'missed':>6} {'false':>6} {'cut-offs':>8} "
        f"{'precision':>9} {'recall':>6} {'wait s':>6}"
    )
    lines = [head, "-" * len(head)]
    for s in scores:
        lines.append(
            f"{s.policy:<9} {s.persona:<8} {s.catches:>3}/{s.expected:<3} {s.missed:>6} {s.false_alarms:>6} "
            f"{s.cut_offs:>8} {s.precision:>9.2f} {s.recall:>6.2f} {s.median_wait:>6.2f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pilot.live.bench", description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenarios", type=Path, action="append", default=[], help="an extra scenario pack (JSON)")
    parser.add_argument("--persona", choices=sorted(PERSONAS), action="append", help="only these personas")
    parser.add_argument("--policy", choices=POLICIES, action="append", help="only these policies")
    parser.add_argument("--level", choices=("quiet", "balanced", "proactive"), default="balanced")
    parser.add_argument("--json", type=Path, help="write the full results, every delivery included, here")
    args = parser.parse_args(argv)
    scenarios = load_scenarios(args.scenarios)
    scores = asyncio.run(
        run_all(scenarios, personas=args.persona, policies=tuple(args.policy or POLICIES), level=args.level)
    )
    print(f"Heliox Live Bench: {len(scenarios)} scenarios, level {args.level}")
    print(table(scores))
    if args.json:
        args.json.write_text(json.dumps([s.as_dict() for s in scores], indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
