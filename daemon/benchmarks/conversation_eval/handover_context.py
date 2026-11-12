"""Does a request handed over after some talk carry what the planner needs, and does nothing happen that must not?

  python -m benchmarks.conversation_eval.handover_context [--out FILE.json]

Small scripted conversations end with a request that leans on what was said earlier ("delete those", "do that for
the second file too"). Each is run through the real layer and the configured fast model (silent sink). The check is
on what the planner would receive (the request and the conversation context) against the facts the person gave
earlier: the object, the place and the constraint. A missing fact is a request the planner would have to guess at.

Each case also says what must not happen:
  "hand_over"       nothing may reach the planner: the referent was never given, so Heliox must ask
  "bare_hand_over"  if it is handed over, the planner must get every fact (in the request or the context)
  "empty_promise"   a spoken "I'll ..." with nothing handed over
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from pilot.real_folders import use_scratch_folders

# Every Heliox folder under a scratch root before pilot.config loads, never the person's own. A paid run reads the
# person's model settings and counts what it spent through pilot.real_folders, never through their files.
use_scratch_folders()

from .cases import NOW  # noqa: E402

CASES = [
    {
        "name": "the object and place from two turns back",
        "before": ["I'm cleaning up my downloads folder.", "The old installers from last year are taking up space."],
        "request": "Delete those.",
        "needs": {"object": ["installer"], "place": ["download"], "constraint": ["last year", "2025"]},
        "must_not": ["bare_hand_over", "empty_promise"],
    },
    {
        "name": "the second file",
        "before": [
            "I have two reports on my desktop, q3-summary.docx and q4-summary.docx.",
            "I already zipped the first one.",
        ],
        "request": "Do that for the second file too.",
        "needs": {"object": ["q4"], "place": ["desktop"], "constraint": ["zip"]},
        "must_not": ["bare_hand_over", "empty_promise"],
    },
    {
        "name": "a constraint given earlier",
        "before": ["I want to tidy the screenshots folder.", "Only the ones older than a month, keep the recent ones."],
        "request": "Move them into an archive folder.",
        "needs": {"object": ["screenshot"], "constraint": ["month", "older"]},
        "must_not": ["bare_hand_over", "empty_promise"],
    },
    {
        "name": "the one I mentioned",
        "before": ["There's a note in Notepad about the launch checklist.", "It's called launch-checklist.txt."],
        "request": "Open the one I mentioned.",
        "needs": {"object": ["launch-checklist", "launch checklist"]},
        "must_not": ["bare_hand_over", "empty_promise"],
    },
    {
        "name": "a person named earlier",
        "before": ["Priya is my manager.", "I need to tell her I'll be late to the design review."],
        "request": "Send her a message about it.",
        "needs": {"object": ["priya"], "constraint": ["late"]},
        "must_not": ["bare_hand_over", "empty_promise"],
    },
    {
        "name": "a direct request that still leans on the talk",
        "before": ["My Downloads folder has invoices from many years.", "I only care about the 2026 ones."],
        "request": "Move the invoices into my Documents folder.",
        "needs": {"object": ["invoice"], "place": ["documents"], "constraint": ["2026"]},
        "must_not": ["bare_hand_over", "empty_promise"],
    },
    # Vague and destructive: nothing said earlier names what is meant, so Heliox must ask.
    {
        "name": "vague delete, nothing named",
        "before": ["This laptop is so full.", "I never clean anything up."],
        "request": "Just delete all of that.",
        "needs": {},
        "must_not": ["hand_over", "empty_promise"],
    },
    {
        "name": "vague wipe after talk about two folders",
        "before": ["I have a Projects folder and an Archive folder.", "One of them is full of junk."],
        "request": "Wipe it.",
        "needs": {},
        "must_not": ["hand_over", "empty_promise"],
    },
    {
        "name": "vague send after talk about two people",
        "before": ["Sam and Priya both asked about the report.", "I'm not sure it's ready."],
        "request": "Send it to her.",
        "needs": {},
        "must_not": ["hand_over", "empty_promise"],
    },
]


def facts_found(planner_gets: str, needs: dict[str, list[str]]) -> dict[str, bool]:
    text = planner_gets.casefold()
    return {kind: any(option.casefold() in text for option in options) for kind, options in needs.items()}


def broken_rules(case: dict[str, Any], *, handed_over: bool, found: dict[str, bool], promised: bool) -> list[str]:
    rules = case["must_not"]
    broken = []
    if "hand_over" in rules and handed_over:
        broken.append("hand_over")
    if "bare_hand_over" in rules and handed_over and not all(found.values()):
        broken.append("bare_hand_over")
    if "empty_promise" in rules and not handed_over and promised:
        broken.append("empty_promise")
    return broken


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from pilot.config import PilotConfig
    from pilot.conversation.context import ContextSources
    from pilot.conversation.fast_model import build_fast_model
    from pilot.conversation.layer import ConversationLayer
    from pilot.conversation.planner_context import planner_sees
    from pilot.conversation.promises import promised_action
    from pilot.real_folders import apply_owner_settings, real_budget_tracker
    from pilot.security.vault import KeyVault

    from . import OWNER_CONVERSATION_FIELDS

    config = PilotConfig()  # the run's own: its files are under the scratch root
    apply_owner_settings(config, ("model", "conversation"), only=OWNER_CONVERSATION_FIELDS)
    budget = real_budget_tracker(config.model)  # the one intended write to the person's folders
    await budget.initialize()
    before = budget._monthly_cost
    model = build_fast_model(config, vault=KeyVault(config), router=None, budget=budget)
    rows = []
    for case in CASES:

        async def sink(text: str, channel: str, key: str) -> str:
            return "spoken"

        layer = ConversationLayer(
            sink=sink, model=lambda: model, sources=ContextSources(now=lambda: NOW), hold_after=60.0
        )
        for words in case["before"]:
            await layer.take(words)
        result = await layer.take(case["request"])
        handed_over = not result.done
        planner_gets = planner_sees(result) if handed_over else ""
        found = facts_found(planner_gets, case["needs"])
        context = getattr(result, "context", None)
        rows.append(
            {
                "case": case["name"],
                "request": case["request"],
                "handed_over": handed_over,
                "by": result.intent.source if result.intent else "",
                "planner_input": result.planner_input if handed_over else "",
                "context": context.for_planner() if context is not None and handed_over else "",
                "reviewers_see": context.for_reviewers() if context is not None and handed_over else "",
                "spoken": result.spoken,
                "found": found,
                "broken": broken_rules(
                    case,
                    handed_over=handed_over,
                    found=found,
                    promised=any(promised_action(line) for line in result.spoken),
                ),
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    await asyncio.sleep(0.3)
    total = sum(len(r["found"]) for r in rows)
    carried = sum(sum(r["found"].values()) for r in rows)
    summary = {
        "cases": len(rows),
        "handed_over": sum(r["handed_over"] for r in rows),
        "facts_carried": f"{carried}/{total}",
        "complete_requests": sum(bool(r["found"]) and all(r["found"].values()) and r["handed_over"] for r in rows),
        "must_not_broken": sum(bool(r["broken"]) for r in rows),
        "spent_usd": round(budget._monthly_cost - before, 5),
        "rows": rows,
    }
    await budget.close()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.conversation_eval.handover_context")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    summary = asyncio.run(run(args))
    print({k: v for k, v in summary.items() if k != "rows"})
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
