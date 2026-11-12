"""Conversation eval: scripted conversations through the real fast layer and the real configured model.

  python -m benchmarks.conversation_eval [--provider claude|openai|openrouter|gemini|meta|ollama] [--model NAME]
                                         [--out FILE.json] [--only TEXT]

Each turn runs through ConversationLayer.take with a silent sink; its checks are decided in code (checks.py).
Prints the pass rate per check, the hand-over precision and recall, every failure quoted, the cost recorded in the
budget tracker, and the median and worst first-sentence time, which are only quoted when the machine is not busy
(timings measured under load are reported as owed). Real calls: about 40 short turns, under a cent on Claude
Haiku 5.5. Results go to --out (outside the repository: they quote what the model said).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from .cases import CONVERSATIONS, NOW
from .checks import TurnRecord, check_turn, summarise


def machine_busy(cpu_limit: float = 70.0, free_gb: float = 1.5) -> str:
    """Why timings would not be trustworthy now, or "" when the machine is quiet enough to quote them."""
    try:
        import psutil
    except ImportError:
        return ""
    cpu = psutil.cpu_percent(interval=1.0)
    free = psutil.virtual_memory().available / 2**30
    reasons = []
    if cpu > cpu_limit:
        reasons.append(f"CPU at {cpu:.0f}%")
    if free < free_gb:
        reasons.append(f"{free:.1f} GB of memory free")
    return ", ".join(reasons)


class RecordingModel:
    """Wraps the real fast model to keep the raw text of each reply (before cleaning)."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self.raw: list[str] = []
        for name in ("name", "provider", "streams", "local", "calls"):
            setattr(self, name, getattr(model, name, None))

    async def stream(self, system: str, messages: list[dict[str, str]]):
        self.raw.append("")
        async for piece in self._model.stream(system, messages):
            self.raw[-1] += piece
            yield piece

    def hold(self, seconds: float, reason: str) -> None:
        hold = getattr(self._model, "hold", None)
        if callable(hold):
            hold(seconds, reason)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from pilot.config import DB_FILE, PilotConfig
    from pilot.conversation.context import ContextSources, Screen
    from pilot.conversation.fast_model import build_fast_model
    from pilot.conversation.layer import ConversationLayer
    from pilot.models.budget_tracker import BudgetTracker
    from pilot.security.vault import KeyVault

    config = PilotConfig.load()
    config.save = lambda: None  # never write the person's config
    if args.provider:
        if args.provider == "ollama":
            config.model.provider = "ollama"
        else:
            config.model.provider, config.model.cloud_provider = "cloud", args.provider
    if args.model:
        setattr(config.conversation, f"{args.provider or config.model.cloud_provider or 'ollama'}_model", args.model)
    budget = BudgetTracker(config.model, str(DB_FILE))
    await budget.initialize()
    spent_before = budget._monthly_cost
    base = build_fast_model(config, vault=KeyVault(config), router=None, budget=budget)
    if base is None:
        raise SystemExit("No fast model is configured; set one under [model] or pass --provider.")
    busy_before = machine_busy()

    results: list[tuple[str, TurnRecord, list]] = []
    timings: list[float] = []
    for conversation in CONVERSATIONS:
        if args.only and args.only.casefold() not in conversation["name"].casefold():
            continue
        model = RecordingModel(base)
        screen = conversation.get("screen")
        sources = ContextSources(
            profile_lines=lambda c=conversation: list(c.get("profile", [])),
            task_state=lambda c=conversation: dict(c.get("task", {})),
            screen=lambda s=screen: Screen(app=s["app"], excerpt=s.get("excerpt", "")) if s else None,
            now=lambda: NOW,
        )
        spoken: list[str] = []

        async def sink(text: str, channel: str, key: str, spoken: list[str] = spoken) -> str:
            spoken.append(text)
            return "spoken"

        layer = ConversationLayer(sink=sink, model=lambda m=model: m, sources=sources, hold_after=60.0)
        previous: TurnRecord | None = None
        for text, spec in conversation["turns"]:
            spoken.clear()
            calls_before = len(model.raw)
            result = await layer.take(text)
            record = TurnRecord(
                text=text,
                route=result.route.kind.value if result.route else "",
                done=result.done,
                spoken=[s for s in spoken],
                raw=model.raw[-1] if len(model.raw) > calls_before else "",
                planner_input="" if result.done else result.planner_input,
                quick_types=[a.action_type.value for a in result.quick.plan.actions] if result.quick else [],
                first_sentence_s=(result.timings.get("first_sentence_ms") or 0) / 1000 or None,
            )
            if record.first_sentence_s and record.done:
                timings.append(record.first_sentence_s)
            outcomes = check_turn(record, spec, previous)
            results.append((f"{conversation['name']}: {text}", record, outcomes))
            print(
                f"{'ok ' if all(o.passed for o in outcomes) else 'BAD'} {text!r} -> {record.route} "
                f"{'talk' if record.done else 'act'} {' '.join(spoken)[:110]!r}",
                flush=True,
            )
            previous = record
            await asyncio.sleep(0.2)
    await asyncio.sleep(0.5)
    summary = summarise(results)
    busy_after = machine_busy()
    busy = busy_before or busy_after
    summary["model"] = f"{getattr(base, 'provider', '')}:{getattr(base, 'name', '')}"
    summary["turns"] = len(results)
    summary["spent_usd"] = round(budget._monthly_cost - spent_before, 5)
    summary["first_sentence_s"] = (
        {"owed": f"not quoted: the machine was busy ({busy})"}
        if busy
        else {"median": round(statistics.median(timings), 3), "worst": round(max(timings), 3), "n": len(timings)}
        if timings
        else {}
    )
    summary["records"] = [
        {
            "turn": name,
            "route": r.route,
            "talk": r.done,
            "spoken": r.spoken,
            "raw": r.raw,
            "planner_input": r.planner_input,
            "quick": r.quick_types,
        }
        for name, r, _ in results
    ]
    await budget.close()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.conversation_eval")
    parser.add_argument("--provider", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--only", default="")
    args = parser.parse_args(argv)
    started = time.time()
    summary = asyncio.run(run(args))
    print("\nmodel:", summary["model"], "turns:", summary["turns"], "cost: $", summary["spent_usd"])
    for name, rate in sorted(summary["checks"].items()):
        print(f"  {name:24} {rate['passed']}/{rate['of']}")
    print("  hand-over precision", summary["handover"]["precision"], "recall", summary["handover"]["recall"])
    print("  first sentence:", summary["first_sentence_s"])
    for failure in summary["failures"]:
        print(f"  FAIL {failure['check']}: {failure['turn']} | {failure['detail'][:200]}")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"({time.time() - started:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
