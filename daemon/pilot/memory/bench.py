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


async def run(script: dict[str, Any] | None = None, *, mode: str = "lightweight") -> dict[str, Any]:
    from pilot.agents.planner import Planner

    script = script or json.loads(SCRIPT.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="heliox-memory-bench-") as folder:
        root = Path(folder)
        config = PilotConfig()
        config.model.mode = mode
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
                await memory.retract_fact(fact_id, reason="Removed from your profile")
            except KeyError:
                pass

        profile = UserProfile(root / "user_profile.json", remember=remember, forget=forget)
        memory.set_profile_source(profile.context_lines)  # as the server wires it
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
    parser.add_argument("--json", type=Path, help="write every probe's result here")
    args = parser.parse_args(argv)
    result = asyncio.run(run(mode=args.mode))
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
