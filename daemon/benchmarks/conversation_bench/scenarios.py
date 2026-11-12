"""Scenario files: what is said, in what state, and what must happen (pure; JSON in ``scenarios/``).

A scenario::

    {"id": "tt-wait-stop-real-2241", "set": "turn_taking", "voice": "real", "mic": "desk-usb",
     "lines": ["1.2b", "1.2c"],
     "setup": {"heliox_says": "Here is your morning...", "pending_approval": false, "task_running": false,
               "conversation_open": false},
     "steps": [{"say": "real/take_2241.wav#24.0-26.0", "when": "heliox_audio", "delay": 1.5,
                "text": "Wait, stop."}],
     "expect": {"woken": true, "route": "control", "spoken": "any", "stopped": true}}

``voice`` is "real" (the owner's recorded voice) or "synthetic" (local Kokoro); they are scored apart and never
mixed. ``models`` limits a scenario to some kinds of model (default: all three). ``pending`` marks a scenario
written against an interface that does not exist yet (it is listed, not run).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audio import parse_ref

SETS = ("understanding", "turn_taking", "not_addressed", "control", "conversation_quality")
VOICES = ("real", "synthetic")
ROUTES = ("local", "planner", "control", "deny", "refuse_approval", "interject", "overheard", "ignored", "any")
MODEL_KINDS = ("cloud", "local", "none")
SCENARIO_DIR = Path(__file__).parent / "scenarios"


class ScenarioError(ValueError):
    pass


@dataclass(frozen=True)
class Step:
    say: str = ""  # audio reference
    text: str = ""  # what is said (the reference transcript); "" when unknown
    when: str = "start"  # "start" | "heliox_audio" (after Heliox's voice starts) | "previous_done"
    delay: float = 0.0
    set_state: dict[str, Any] = field(default_factory=dict)
    wait: float = 0.0


@dataclass(frozen=True)
class Scenario:
    id: str
    set: str
    voice: str
    lines: tuple[str, ...]
    steps: tuple[Step, ...]
    expect: dict[str, Any]
    setup: dict[str, Any] = field(default_factory=dict)
    mic: str = ""
    models: tuple[str, ...] = MODEL_KINDS
    pending: str = ""
    settle: float = 8.0
    note: str = ""
    reference_confirmed: bool = True
    quality: dict[str, Any] = field(default_factory=dict)

    @property
    def said(self) -> str:
        return " ".join(step.text for step in self.steps if step.say and step.text)


def _step(raw: dict[str, Any], where: str) -> Step:
    unknown = set(raw) - {"say", "text", "when", "delay", "set", "wait"}
    if unknown:
        raise ScenarioError(f"{where}: unknown step keys {sorted(unknown)}")
    if raw.get("say"):
        parse_ref(str(raw["say"]))
    when = str(raw.get("when", "start"))
    if when not in ("start", "heliox_audio", "previous_done"):
        raise ScenarioError(f"{where}: unknown 'when' {when!r}")
    return Step(
        say=str(raw.get("say", "")),
        text=str(raw.get("text", "")),
        when=when,
        delay=float(raw.get("delay", 0.0)),
        set_state=dict(raw.get("set", {})),
        wait=float(raw.get("wait", 0.0)),
    )


def parse_scenario(raw: dict[str, Any]) -> Scenario:
    sid = str(raw.get("id") or "")
    if not sid:
        raise ScenarioError("scenario without an id")
    if raw.get("set") not in SETS:
        raise ScenarioError(f"{sid}: set must be one of {SETS}")
    if raw.get("voice") not in VOICES:
        raise ScenarioError(f"{sid}: voice must be one of {VOICES}")
    expect = dict(raw.get("expect") or {})
    route = expect.get("route", "any")
    routes = route if isinstance(route, list) else [route]
    if any(r not in ROUTES for r in routes):
        raise ScenarioError(f"{sid}: unknown route {route!r}")
    models = tuple(raw.get("models") or MODEL_KINDS)
    if any(m not in MODEL_KINDS for m in models):
        raise ScenarioError(f"{sid}: unknown model kind in {models}")
    steps = tuple(_step(s, f"{sid} step {i}") for i, s in enumerate(raw.get("steps") or []))
    if not steps and not raw.get("pending"):
        raise ScenarioError(f"{sid}: no steps")
    return Scenario(
        id=sid,
        set=raw["set"],
        voice=raw["voice"],
        lines=tuple(raw.get("lines") or ()),
        steps=steps,
        expect=expect,
        setup=dict(raw.get("setup") or {}),
        mic=str(raw.get("mic", "")),
        models=models,
        pending=str(raw.get("pending", "")),
        settle=float(raw.get("settle", 8.0)),
        note=str(raw.get("note", "")),
        reference_confirmed=bool(raw.get("reference_confirmed", True)),
        quality=dict(raw.get("quality") or {}),
    )


def load_file(path: Path) -> list[Scenario]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["scenarios"] if isinstance(data, dict) else data
    out = [parse_scenario(item) for item in items]
    ids = [s.id for s in out]
    if len(ids) != len(set(ids)):
        raise ScenarioError(f"{path.name}: duplicate scenario ids")
    return out


def load_all(directory: Path = SCENARIO_DIR) -> list[Scenario]:
    out: list[Scenario] = []
    for path in sorted(directory.glob("*.json")):
        out.extend(load_file(path))
    ids = [s.id for s in out]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ScenarioError(f"duplicate scenario ids across files: {sorted(dupes)}")
    return out
