"""A gated tuning loop for memory retrieval: propose settings, measure them on the Memory Bench, keep only wins.

This is the self-improvement loop in its safe form. It does not let Heliox rewrite itself. It changes a handful of
retrieval settings (how much room the profile gets, how much goes to earlier turns of a long session, how many
opening turns are always kept, how strongly the memory search trades relevance against variety), runs the Memory
Bench under pressure at several context budgets for each candidate, and recommends a candidate only if:

- it scores higher than the current settings on the held-out probes, which the search never optimised for;
- it scores at least as well on the probes it did optimise for;
- no probe that passes with the current settings fails with it;
- it brings back nothing stale.

It never applies its recommendation. It prints it and, with ``--json``, writes the evidence, so a person can decide
and change the constants in code, where review and CI see the change.

    python -m pilot.memory.tune [--candidates 24] [--json tune.json]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import itertools
import json
import random
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pilot.memory import assembler, bench, session_recall, store

# (module, name, values to try). The first value of each is not special: the current value is read at run time.
SPACE: tuple[tuple[Any, str, tuple[Any, ...]], ...] = (
    (store, "PROFILE_TOKENS", (200, 300, 400, 500)),
    (store, "PROFILE_RANK_BY_QUERY", (True, False)),
    (session_recall, "EARLIER_SHARE", (0.3, 0.4, 0.5, 0.6)),
    (session_recall, "FIRST_TURNS", (1, 2, 3)),
    (assembler, "MMR_RELEVANCE_WEIGHT", (0.5, 0.6, 0.7, 0.8, 0.9)),
)
BUDGETS = (1500, 2400, 4000)


def current() -> dict[str, Any]:
    return {f"{module.__name__}.{name}": getattr(module, name) for module, name, _ in SPACE}


@contextlib.contextmanager
def applied(settings: dict[str, Any]) -> Iterator[None]:
    saved = current()
    try:
        for module, name, _ in SPACE:
            key = f"{module.__name__}.{name}"
            if key in settings:
                setattr(module, name, settings[key])
        yield
    finally:
        for module, name, _ in SPACE:
            setattr(module, name, saved[f"{module.__name__}.{name}"])


def split(probe_ids: list[str]) -> tuple[set[str], set[str]]:
    """Every other probe, in name order, is held out from the search."""
    ordered = sorted(probe_ids)
    return set(ordered[0::2]), set(ordered[1::2])


async def score(settings: dict[str, Any], pressure: int) -> dict[str, Any]:
    """Pass/fail per (budget, probe) and stale count, with these settings in force."""
    passed: dict[str, bool] = {}
    stale = 0
    with applied(settings):
        for budget in BUDGETS:
            result = await bench.run(pressure=pressure, budget=budget)
            for row in result["rows"]:
                passed[f"{budget}:{row['id']}"] = row["pass"]
                stale += bool(row["stale"])
    return {"passed": passed, "stale": stale}


def _total(passed: dict[str, bool], probes: set[str]) -> int:
    return sum(ok for key, ok in passed.items() if key.split(":", 1)[1] in probes)


def candidates(count: int, seed: int = 11) -> list[dict[str, Any]]:
    grid = list(itertools.product(*(values for _, _, values in SPACE)))
    random.Random(seed).shuffle(grid)
    keys = [f"{module.__name__}.{name}" for module, name, _ in SPACE]
    return [dict(zip(keys, values, strict=True)) for values in grid[:count]]


async def tune(count: int = 24, pressure: int = 2) -> dict[str, Any]:
    baseline_settings = current()
    baseline = await score(baseline_settings, pressure)
    probe_ids = sorted({key.split(":", 1)[1] for key in baseline["passed"]})
    train, held_out = split(probe_ids)
    report = {
        "pressure": pressure,
        "budgets": list(BUDGETS),
        "train_probes": sorted(train),
        "held_out_probes": sorted(held_out),
        "baseline": {
            "settings": baseline_settings,
            "train": _total(baseline["passed"], train),
            "held_out": _total(baseline["passed"], held_out),
            "stale": baseline["stale"],
        },
        "tried": [],
        "recommendation": None,
    }
    best = None
    for settings in candidates(count):
        if settings == baseline_settings:
            continue
        result = await score(settings, pressure)
        regressions = sorted(k for k, ok in baseline["passed"].items() if ok and not result["passed"].get(k, False))
        entry = {
            "settings": settings,
            "train": _total(result["passed"], train),
            "held_out": _total(result["passed"], held_out),
            "stale": result["stale"],
            "regressions": regressions,
        }
        entry["accepted"] = (
            entry["held_out"] > report["baseline"]["held_out"]
            and entry["train"] >= report["baseline"]["train"]
            and not regressions
            and entry["stale"] == 0
        )
        report["tried"].append(entry)
        if entry["accepted"] and (
            best is None or (entry["held_out"], entry["train"]) > (best["held_out"], best["train"])
        ):
            best = entry
    report["recommendation"] = best
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pilot.memory.tune", description=__doc__.split("\n\n")[0])
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--pressure", type=int, choices=(1, 2), default=2)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    report = asyncio.run(tune(args.candidates, args.pressure))
    base = report["baseline"]
    total = len(report["train_probes"] + report["held_out_probes"]) * len(BUDGETS)
    print(
        f"Current settings: {base['train'] + base['held_out']}/{total} probe runs pass "
        f"(held-out {base['held_out']}), {base['stale']} stale"
    )
    accepted = [entry for entry in report["tried"] if entry["accepted"]]
    print(f"Tried {len(report['tried'])} candidates; {len(accepted)} passed the gate.")
    if report["recommendation"]:
        changed = {k: v for k, v in report["recommendation"]["settings"].items() if base["settings"].get(k) != v}
        print(f"Recommended (not applied): {changed}, held-out {report['recommendation']['held_out']}")
    else:
        print("No candidate beat the current settings on the held-out probes without a regression; keep them.")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
