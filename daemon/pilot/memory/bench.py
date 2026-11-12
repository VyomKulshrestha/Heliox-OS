"""Heliox Memory Bench: does the right memory reach the model, within its budget, after weeks of use?

Long-term memory is only useful if the right piece of it is in front of the model when it matters. This bench
replays a few weeks of scripted use through Heliox's real memory path (``UserProfile`` for "remember that…",
``MemoryStore.record`` for every planned turn, the temporal memory graph underneath) and then asks probe questions
through the real ``Planner``. A stand-in model records the exact messages the planner would send and returns an
empty plan; no model runs and nothing leaves the computer.

A probe passes when every expected phrase is somewhere in those messages and no forbidden phrase is. Forbidden
phrases are facts that have since changed ("I live in Mumbai" after "I live in Pune now"): a memory that brings back
the old fact is worse than one that forgets.

    python -m pilot.memory.bench [--json results.json]

The script is ``memory_bench.json`` next to this file. It runs in a temporary folder with the vector index off, so
the result depends only on code and is the same on every machine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from pilot.actions import ActionPlan
from pilot.config import PilotConfig
from pilot.memory.sliding_window import get_token_count
from pilot.memory.store import MemoryStore
from pilot.memory.temporal import MemoryProvenance, MemoryScope
from pilot.workflows.user_profile import UserProfile

SCRIPT = Path(__file__).with_name("memory_bench.json")

# Pressure: months of ordinary use around the script, so the memory budget has to choose. Generated, not random,
# so every run is the same.
_FILLER_FACTS = [
    "I like jazz", "I am learning Spanish", "I cycle to work on Tuesdays", "I drink oat milk",
    "I support Arsenal", "I read before bed", "I use a standing desk", "I keep a paper diary",
    "I am allergic to peanuts", "I grew up near the sea", "I play chess online", "I collect vinyl records",
    "I prefer window seats", "I volunteer at a library", "I am saving for a car", "I bake bread on Sundays",
    "I have a sourdough starter called Bob", "I wake up at six", "I use dark mode everywhere",
    "I listen to podcasts while cooking", "I am vegetarian on weekdays", "I go swimming on Fridays",
    "I sketch in the evenings", "I water the plants on Mondays",
]  # fmt: skip
_FILLER_TASKS = [
    ("Open the monthly invoice spreadsheet", "Opened invoices-{n}.xlsx."),
    ("Check how much RAM is free", "{n} GB of RAM is free."),
    ("Find photos from the {n}th", "Found {n} photos in Pictures."),
    ("Turn on do not disturb", "Do not disturb is on for an hour."),
    ("What's on my calendar this afternoon?", "You have {n} meetings this afternoon."),
    ("Summarize the email from the flour mill", "The mill raised prices by {n} percent from next month."),
    ("Rename the screenshots on the desktop", "Renamed {n} screenshots."),
    ("Play some focus music", "Playing a focus playlist."),
    ("Draft a reply thanking the landlord", "Drafted a short thank-you reply."),
    ("Clean up the downloads folder", "Moved {n} old installers to the bin."),
]


_FOODS = ("ramen", "pho", "dosa", "paella", "falafel", "pierogi", "bibimbap", "tagine", "ceviche", "laksa", "goulash",
          "arepas")  # fmt: skip
_PLACES = ("Goa", "Hampi", "Kyoto", "Lisbon", "Oslo", "Cusco", "Tbilisi", "Hanoi", "Zanzibar", "Bergen", "Porto",
           "Seville")  # fmt: skip


def _pressure(script: dict[str, Any], level: int = 1) -> dict[str, Any]:
    facts = list(_FILLER_FACTS)
    if level >= 2:  # twice the profile and twice the history
        facts += [f"I once ate {food} in {place}" for food, place in zip(_FOODS, _PLACES, strict=True)]
        facts += [f"I want to visit {place} someday" for place in reversed(_PLACES)]
    sessions = [{"id": "filler-facts", "turns": [{"say": f"Remember that {fact}"} for fact in facts]}]
    for week in range(8 * level):
        turns = []
        for i in range(10):
            asked, answered = _FILLER_TASKS[(week * 3 + i) % len(_FILLER_TASKS)]
            n = 3 + week * 10 + i
            turns.append({"say": asked.format(n=n), "reply": answered.format(n=n)})
        sessions.append({"id": f"filler-week-{week}", "turns": turns})
    # The filler comes first, so the scripted facts are not simply the most recent ones.
    return {**script, "sessions": sessions + script["sessions"]}


class RecordingModel:
    """Stands in for the model router: keeps the messages the planner sends and answers with an empty plan."""

    def __init__(self, config: PilotConfig) -> None:
        self._config = config
        self.messages: list[dict[str, Any]] = []

    async def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.messages = [dict(message) for message in messages]
        return json.dumps({"explanation": "ok", "actions": []})

    def __getattr__(self, name: str) -> Any:  # anything else the planner asks of the router
        async def nothing(*args: Any, **kwargs: Any) -> None:
            return None

        return nothing


async def run(
    script: dict[str, Any] | None = None,
    *,
    mode: str = "lightweight",
    pressure: int = 0,
    budget: int | None = None,
) -> dict[str, Any]:
    from pilot.agents.planner import Planner

    script = script or json.loads(SCRIPT.read_text(encoding="utf-8"))
    if pressure:
        script = _pressure(script, int(pressure))
    with tempfile.TemporaryDirectory(prefix="heliox-memory-bench-") as folder:
        root = Path(folder)
        config = PilotConfig()
        config.model.mode = mode
        if budget:
            config.memory.max_context_tokens = budget  # the memory share is a third of this
        memory = MemoryStore(
            checkpoint_interval_seconds=0,
            pruning_interval_seconds=0,
            db_file=root / "memory.db",
            semantic_search=False,
        )
        await memory.initialize()

        async def remember(predicate: str, value: str) -> str | None:
            fact = await memory.remember_fact(
                subject="user",
                predicate=predicate,
                value=value,
                scope=MemoryScope.USER,
                confidence=1.0,
                provenance=MemoryProvenance.EXPLICIT_USER,
                evidence_payload={"source": "user_profile"},
            )
            return fact.fact_id if fact is not None else None

        async def forget(fact_id: str) -> None:
            try:
                await memory.erase_fact(fact_id, reason="Removed from your profile")
            except KeyError:
                pass

        profile = UserProfile(root / "user_profile.json", remember=remember, forget=forget)
        memory.set_profile_source(profile.context_lines)  # as the server wires it
        memory.set_people_source(profile.people_by_relation)
        try:
            for session in script["sessions"]:
                for turn in session["turns"]:
                    if "reply" not in turn:
                        handled = await profile.answer_async(turn["say"])
                        if handled is None:
                            raise ValueError(f"the profile did not take {turn['say']!r}")
                        continue  # answered in code, like the server, so not in the planner's history
                    plan = ActionPlan(actions=[], explanation=turn["reply"], raw_input=turn["say"])
                    await memory.record(turn["say"], plan, [], session_id=session["id"])

            model = RecordingModel(config)
            planner = Planner(model, memory)
            rows = []
            for probe in script["probes"]:
                session_id = probe["session"] if probe["session"] != "new" else f"probe-{probe['id']}"
                model.messages = []
                await planner.plan(probe["ask"], session_id=session_id, force_model=True)
                seen = "\n".join(str(message.get("content", "")) for message in model.messages)
                folded = seen.casefold()
                missing = [phrase for phrase in probe.get("expect", []) if phrase.casefold() not in folded]
                stale = [phrase for phrase in probe.get("forbid", []) if phrase.casefold() in folded]
                rows.append(
                    {
                        "id": probe["id"],
                        "ask": probe["ask"],
                        "why": probe.get("why", ""),
                        "pass": not missing and not stale and bool(model.messages),
                        "missing": missing,
                        "stale": stale,
                        "prompt_tokens": get_token_count(seen) if seen else 0,
                    }
                )
        finally:
            await memory.close()
    passed = sum(row["pass"] for row in rows)
    return {
        "mode": mode,
        "pressure": pressure,
        "budget": budget,
        "probes": len(rows),
        "passed": passed,
        "recall": round(sum(not row["missing"] for row in rows) / len(rows), 3) if rows else 1.0,
        "stale": sum(bool(row["stale"]) for row in rows),
        "median_prompt_tokens": sorted(row["prompt_tokens"] for row in rows)[len(rows) // 2] if rows else 0,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pilot.memory.bench", description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=("lightweight", "full"), default="lightweight")
    parser.add_argument(
        "--pressure", type=int, choices=(0, 1, 2), default=0, help="add months (1) or twice that (2) of ordinary use"
    )
    parser.add_argument("--budget", type=int, help="the context budget in tokens (memory gets a third)")
    parser.add_argument("--json", type=Path, help="write every probe's result here")
    args = parser.parse_args(argv)
    result = asyncio.run(run(mode=args.mode, pressure=args.pressure, budget=args.budget))
    for row in result["rows"]:
        mark = "ok  " if row["pass"] else "FAIL"
        detail = ""
        if row["missing"]:
            detail += f" missing {row['missing']}"
        if row["stale"]:
            detail += f" stale {row['stale']}"
        print(f"{mark} {row['id']:10} {row['prompt_tokens']:5} tokens  {row['why']}{detail}")
    print(
        f"\nHeliox Memory Bench ({result['mode']}): {result['passed']}/{result['probes']} probes, "
        f"recall {result['recall']}, {result['stale']} stale, median {result['median_prompt_tokens']} prompt tokens"
    )
    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
