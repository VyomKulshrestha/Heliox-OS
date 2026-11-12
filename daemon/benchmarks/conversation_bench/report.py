"""Printing the scorecard and saving a run's results (pure apart from writing files)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import timeline as tl
from .scoring import Outcome, Row
from .targets import BY_ID


def fmt(value: float | None, unit: str, statistic: str) -> str:
    if value is None:
        return "-"
    if value == float("inf"):
        return "never"
    if statistic in ("share", "rate", "wer"):
        return f"{value * 100:.1f}%"
    if statistic == "count":
        return f"{value:.0f}"
    return f"{value:.2f} {unit}".strip()


def target_text(line_id: str) -> str:
    line = BY_ID.get(line_id)
    if line is None:
        return ""
    if line.direction == "zero":
        return "0"
    sign = "<=" if line.direction == "max" else ">="
    shown = fmt(line.target, line.unit, line.statistic)
    return f"{sign} {shown}"


def table(rows: list[Row]) -> str:
    out = [f"{'line':<6}{'voice':<10}{'condition':<13}{'model':<7}{'value':>10}  {'target':<11}{'':<5}detail"]
    out.append("-" * 96)
    for r in rows:
        line = BY_ID.get(r.line)
        value = fmt(r.value, line.unit if line else "", line.statistic if line else "share") if line else r.detail
        verdict = {True: "pass", False: "FAIL", None: "-"}[r.ok]
        detail = r.detail if line else ""
        out.append(
            f"{r.line:<6}{r.voice:<10}{r.condition:<13}{r.model:<7}{value:>10}  {target_text(r.line):<11}"
            f"{verdict:<5}{detail}"
        )
    return "\n".join(out)


def legend() -> str:
    return "\n".join(f"  {line.id:<5} {line.brief}" for line in BY_ID.values())


def outcome_json(o: Outcome) -> dict[str, Any]:
    m = o.measures
    return {
        "scenario": o.scenario.id,
        "set": o.scenario.set,
        "voice": o.scenario.voice,
        "mic": o.scenario.mic,
        "condition": o.condition,
        "model": o.model,
        "passed": o.passed,
        "failures": o.failures,
        "said": o.scenario.said,
        "heard": o.heard,
        "route": m.route,
        "reply_start_s": m.reply_start,
        "under_way_s": m.under_way,
        "silent_gap_s": m.silent_gap,
        "stop_time_s": m.stop_time,
        "spoken": m.spoken,
        "dispatched": m.dispatched,
        "card": m.card,
        "approval": m.approval,
    }


def save_outcome(folder: Path, o: Outcome, events: list[tl.Event]) -> Path:
    path = folder / "timelines" / o.scenario.set / f"{o.scenario.id}__{o.condition}__{o.model}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"outcome": outcome_json(o), "events": tl.to_json(events)}, indent=1), encoding="utf-8")
    return path


def save_scorecard(folder: Path, rows: list[Row], outcomes: list[Outcome], worst: list[Outcome]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "scorecard.txt").write_text(table(rows) + "\n\nLines:\n" + legend() + "\n", encoding="utf-8")
    (folder / "scorecard.json").write_text(
        json.dumps(
            {
                "rows": [r.__dict__ for r in rows],
                "outcomes": [outcome_json(o) for o in outcomes],
                "worst": [outcome_json(o) for o in worst],
            },
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
