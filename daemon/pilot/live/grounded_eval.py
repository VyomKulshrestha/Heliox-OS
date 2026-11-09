"""Measure grounded corrections (pilot.live.grounded) against a fixed week of calendar, to-dos and reminders.

    python -m pilot.live.grounded_eval --model qwen3.5:4b [--json results.json]

Each case is one sentence and either the fact it contradicts or nothing. A case passes when the checker raises a
correction about exactly that fact, or raises nothing for a sentence that contradicts nothing. Precision is the
share of raised corrections that were right; recall is the share of contradictions caught. The model runs in the
local Ollama; ``--model none`` measures the code-only clash check by itself.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

from pilot.live.grounded import Complete, GroundedChecker, build_facts, ollama_complete

CASES = Path(__file__).with_name("grounded_cases.json")


def load(path: Path = CASES) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


async def evaluate(complete: Complete | None, data: dict[str, Any] | None = None) -> dict[str, Any]:
    data = data or load()
    today = date.fromisoformat(data["today"])
    facts = build_facts(today=today, events=data["events"], todos=data["todos"], reminders=data["reminders"])
    timings: list[float] = []

    async def timed(system: str, prompt: str) -> str:
        started = time.perf_counter()
        try:
            return await complete(system, prompt)
        finally:
            timings.append(time.perf_counter() - started)

    rows = []
    for case in data["cases"]:
        checker = GroundedChecker(timed if complete else None, lambda: facts, today=lambda: today)
        calls_before = len(timings)
        found = await checker.check(case["text"])
        raised = [c.key.split(":")[1] + ":" + c.key.split(":")[2] for c in found]
        expect = case["expect"]
        ok = raised == [expect] if expect else not raised
        rows.append(
            {
                "text": case["text"],
                "expect": expect,
                "raised": raised,
                "message": found[0].message if found else "",
                "verdict": checker.last_verdict,
                "model_called": len(timings) > calls_before,
                "pass": ok,
            }
        )
    raised_total = sum(1 for row in rows if row["raised"])
    right = sum(1 for row in rows if row["raised"] and row["pass"])
    positives = [row for row in rows if row["expect"]]
    return {
        "cases": len(rows),
        "passed": sum(row["pass"] for row in rows),
        "precision": round(right / raised_total, 3) if raised_total else 1.0,
        "recall": round(sum(row["pass"] for row in positives) / len(positives), 3) if positives else 1.0,
        "false_alarms": sum(1 for row in rows if row["raised"] and not row["pass"]),
        "model_calls": len(timings),
        "median_model_seconds": round(sorted(timings)[len(timings) // 2], 2) if timings else 0.0,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pilot.live.grounded_eval", description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="qwen3.5:4b", help='an Ollama model, or "none" for the code-only check')
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--json", type=Path, help="write every case's result here")
    args = parser.parse_args(argv)
    complete = None if args.model == "none" else ollama_complete(args.model, args.base_url)
    result = asyncio.run(evaluate(complete))
    result["model"] = args.model
    for row in result["rows"]:
        mark = "ok  " if row["pass"] else "MISS" if row["expect"] else "FALSE"
        print(f"{mark:5} {row['text'][:60]:60} -> {', '.join(row['raised']) or '-'} ({row['verdict'] or 'no call'})")
    print(
        f"\n{args.model}: {result['passed']}/{result['cases']} cases, precision {result['precision']}, "
        f"recall {result['recall']}, {result['false_alarms']} false alarms, {result['model_calls']} model calls, "
        f"median {result['median_model_seconds']} s per call"
    )
    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
