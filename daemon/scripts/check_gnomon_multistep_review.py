"""Multi-step review check: does Gnomon leave correct multi-step plans alone and still catch near misses?

Runs the plans of ``benchmarks/gnomon_multistep_review_v1.json`` (2-6 typed actions for multi-part everyday
requests) through the daemon's ``GnomonReviewer`` with the model loaded locally (CUDA). A correct plan Gnomon
escalates is a false alarm; a bad plan it does not escalate is a miss. Undo is on, as in the default config.

    python scripts/check_gnomon_multistep_review.py --artifact-dir <dir> --report <training report>
        [--sets dev,recorded] [--json out.json] [--old]

``--old`` reviews with the reviewer as it was before 10 November 2026 (whole plan against the whole request, no
undo rule), for the before/after comparison. The heldout set is for reporting only: never tune on it.
This is a measurement, not a promotion gate; docs/GNOMON_TRAINING.md has the numbers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pilot.actions import Action, ActionPlan  # noqa: E402
from pilot.intelligence import gnomon_reviewer  # noqa: E402
from pilot.intelligence.gnomon_reviewer import GnomonReviewer, describe_plan  # noqa: E402

CASES = Path(__file__).resolve().parents[1] / "benchmarks" / "gnomon_multistep_review_v1.json"


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_set: dict[str, Any] = {}
    for part in sorted({r["set"] for r in rows}):
        correct = [r for r in rows if r["set"] == part and r["label"] == "correct"]
        bad = [r for r in rows if r["set"] == part and r["label"] == "bad"]
        by_set[part] = {
            "false_alarms": sum(r["escalate"] for r in correct),
            "correct": len(correct),
            "misses": sum(not r["escalate"] for r in bad),
            "bad": len(bad),
            "incomplete": sum(not r["complete"] for r in correct + bad),
            "false_alarm_requests": [r["request"] for r in correct if r["escalate"]],
            "missed": [f"{r['category']}: {r['request']}" for r in bad if not r["escalate"]],
        }
    return by_set


class _OldReviewer(GnomonReviewer):
    """The reviewer before multi-step support: the whole plan against the whole request."""

    async def review_plan(self, goal, plan, untrusted_context="", **_ignored):
        proposed = describe_plan(plan)
        request = gnomon_reviewer._clip(goal, 600)
        verdict = gnomon_reviewer.GnomonVerdict()
        in_scope = await self._yes(
            {"proposed_action": proposed, "user_message": request}, gnomon_reviewer.SCOPE_QUESTION
        )
        if in_scope is not None:
            verdict.scores["in_scope"] = in_scope
            if in_scope < gnomon_reviewer.SCOPE_MIN_YES:
                verdict.escalate = True
                verdict.reasons.append(f"Gnomon judged this may go beyond what you asked ({in_scope:.0%} in scope).")
        if not all(a.action_type.value in gnomon_reviewer.LOCAL_ONLY_ACTIONS for a in plan.actions):
            harm = await self._score(
                {"proposed_action": proposed}, "score", gnomon_reviewer.HARM_QUESTION, gnomon_reviewer.HARM_LEVELS
            )
            if harm:
                probabilities = harm.get("probabilities") or {}
                severe = sum(float(probabilities.get(level, 0.0)) for level in gnomon_reviewer.HARM_LEVELS[2:])
                verdict.scores["severe_harm"] = severe
                if severe >= gnomon_reviewer.HARM_ESCALATE_PROBABILITY:
                    verdict.escalate = True
                    verdict.reasons.append(f"Gnomon graded the harm if wrong as: {harm.get('most_likely')}.")
        if untrusted_context.strip():
            permitted = await self._yes(
                {
                    "content_source": "tool or screen output",
                    "observed_text": gnomon_reviewer._clip(untrusted_context, 1500),
                    "proposed_action": proposed,
                    "request": request,
                },
                gnomon_reviewer.UNTRUSTED_QUESTION,
            )
            if permitted is not None:
                verdict.scores["permitted_by_user"] = permitted
                if permitted < gnomon_reviewer.UNTRUSTED_MIN_YES:
                    verdict.escalate = True
                    verdict.reasons.append("Gnomon judged this follows instructions found in tool or screen content.")
        return verdict


async def review_all(engine: Any, cases: list[dict[str, Any]], *, old: bool = False) -> list[dict[str, Any]]:
    reviewer = (_OldReviewer if old else GnomonReviewer)(lambda: engine, call_timeout=60.0)
    reviewer.warm()
    while reviewer.status()["state"] == "loading":
        await asyncio.sleep(0.5)
    if reviewer.status()["state"] != "ready":
        raise RuntimeError(f"Gnomon did not load: {reviewer.status()['last_error']}")
    rows = []
    try:
        for case in cases:
            plan = ActionPlan(actions=[Action.model_validate(a) for a in case["actions"]], explanation="", raw_input="")
            started = time.perf_counter()
            verdict = await reviewer.review_plan(
                case["request"], plan, case.get("untrusted_context", ""), undo_enabled=True
            )
            if verdict is None:
                raise RuntimeError("Gnomon returned no verdict while ready")
            rows.append(
                {
                    **{k: case.get(k, "") for k in ("set", "label", "category", "request", "source")},
                    "actions": len(case["actions"]),
                    "escalate": verdict.escalate,
                    "complete": verdict.complete,
                    "reasons": verdict.reasons,
                    "scores": {k: round(v, 4) for k, v in verdict.scores.items()},
                    "notes": verdict.notes,
                    "scope_reviewed": verdict.scope_reviewed,
                    "seconds": round(time.perf_counter() - started, 2),
                }
            )
    finally:
        await reviewer.close()
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--sets", default="dev,recorded", help="comma-separated: dev, heldout, recorded")
    parser.add_argument("--old", action="store_true", help="review with the pre-multi-step reviewer")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    from pilot.intelligence.typed_decision import LocalTypedDecisionModel

    wanted = set(args.sets.split(","))
    cases = [c for c in json.loads(CASES.read_text(encoding="utf-8"))["cases"] if c["set"] in wanted]
    artifact, report = Path(args.artifact_dir).resolve(), Path(args.report).resolve()
    rows = asyncio.run(review_all(LocalTypedDecisionModel(artifact, report), cases, old=args.old))
    result = {"artifact_dir": args.artifact_dir, "reviewer": "old" if args.old else "current", "by_set": score(rows)}
    result["rows"] = rows
    for row in rows:
        mark = "FLAG" if row["escalate"] else "ok  "
        print(f"{row['set']:8} {row['label']:7} {mark} {row['seconds']:5.2f}s {row['request'][:60]}")
    for part, numbers in result["by_set"].items():
        print(
            f"{part}: {numbers['false_alarms']}/{numbers['correct']} false alarms, "
            f"{numbers['misses']}/{numbers['bad']} misses, {numbers['incomplete']} incomplete"
        )
    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
