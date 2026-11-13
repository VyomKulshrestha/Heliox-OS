"""Re-baseline with the conversation layer up, real-voice lines split by microphone, and timings owed on a busy CPU.

  python -m benchmarks.conversation_bench.rebaseline --audio <folder> [--results DIR] [--conditions ...]
         [--blocks N] [--wait-quiet MINUTES]

Differences from ``run``: the server's model stack is marked up and the fast model is a stub per kind of model
(``stack.py``), so the conversation layer answers as it does in the product; real-voice rows are labelled by
microphone (desk, headset) because the headset records in narrowband call mode; the load is sampled before the run
and before every scenario, and any timing line (``machine.TIMING_LINES``) measured above the busy threshold is
printed as owed instead of as a number.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from . import audio as audio_refs
from . import machine, report, scoring, stack
from . import timeline as tl
from .__main__ import jobs
from .bench_root import chosen_root
from .models import KINDS
from .scenarios import load_all
from .scoring import Row

MIC_LABEL = {"ptron-usb-desk": "desk", "bluetooth-headset-call-mode": "headset"}


def relabel(outcome: scoring.Outcome) -> scoring.Outcome:
    """Real-voice rows carry their microphone in the condition column (pure)."""
    if outcome.scenario.voice == "real":
        mic = MIC_LABEL.get(outcome.scenario.mic, outcome.scenario.mic or "?")
        outcome.condition = f"{outcome.condition}|{mic}"
    return outcome


def owe_timings(rows: list[Row], busy: dict[tuple[str, str, str], bool]) -> list[Row]:
    """Timing rows measured on a busy machine lose their number (pure)."""
    out = []
    for r in rows:
        if r.line in machine.TIMING_LINES and busy.get((r.voice, r.condition, r.model), True):
            out.append(Row(r.line, r.voice, r.condition, r.model, None, r.n, None, "owed, machine busy"))
        else:
            out.append(r)
    return out


async def run(args: argparse.Namespace, results: Path) -> list[scoring.Outcome]:
    from .driver import Harness

    harness = Harness(audio_root=args.audio, data_dir=chosen_root(args.root, results), model=KINDS["cloud"],
                      owner_settings=True)  # fmt: skip
    await harness.build()
    fast_models = stack.enable(harness)
    scenarios = load_all()
    only = "u-real,c-,tt-,u-syn-follow-up,na-real,na-syn-open" + "".join(
        f",na-block-{b:02d}" for b in range(args.blocks)
    )
    todo = jobs(scenarios, profile="baseline", sets=["understanding", "control", "turn_taking", "not_addressed"],
                conditions=args.conditions.split(","), models=[], only=only, babble=True, na_blocks=args.blocks)  # fmt: skip
    print(f"{len(todo)} runs", flush=True)
    outcomes = []
    try:
        for i, (scenario, condition, model) in enumerate(todo, 1):
            stack.select(harness, KINDS[model])
            load = machine.cpu_percent(1.0)
            events = await harness.run(scenario, condition)
            outcome = scoring.score(scenario, tl.measure(events), condition=condition, model=model)
            report.save_outcome(results, outcome, events)  # file names keep the plain condition
            outcome = relabel(outcome)
            outcome.contributions.append(("_load", load))
            outcomes.append(outcome)
            print(f"[{i}/{len(todo)}] {'ok ' if outcome.passed else 'BAD'} {scenario.id} ({outcome.condition}, {model}) "
                  f"cpu={load} route={outcome.measures.route} heard={outcome.heard[:50]!r} "
                  f"{'; '.join(outcome.failures)[:100]}", flush=True)  # fmt: skip
    finally:
        await harness.close(session_to=results)
        calls = {name: len(m.calls) for name, m in fast_models.items()}
        (results / "fast_model_calls.json").write_text(json.dumps(calls), encoding="utf-8")
        print("fast-model calls by kind:", calls, flush=True)
    return outcomes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.conversation_bench.rebaseline")
    parser.add_argument("--audio", default=None)
    parser.add_argument("--results", default=None)
    parser.add_argument("--root", default=None, help="the bench's own Heliox folders, kept between runs")
    parser.add_argument("--conditions", default="as-recorded,bt-call,voice-10db,fan-10db")
    parser.add_argument(
        "--blocks", type=int, default=0, help="five-minute not-addressed blocks (the replay covers 2.2)"
    )
    parser.add_argument("--wait-quiet", type=float, default=0.0, help="minutes to wait for a quiet machine")
    args = parser.parse_args(argv)
    args.audio = audio_refs.audio_root(args.audio)
    results = Path(args.results) if args.results else (
        Path.home() / "heliox-conversation-bench-results" / f"rebaseline-{time.strftime('%Y%m%d-%H%M%S')}"
    )  # fmt: skip
    results.mkdir(parents=True, exist_ok=True)
    quiet, load = machine.wait_for_quiet(args.wait_quiet * 60)
    print(f"machine load before the run: {load}% ({'quiet' if quiet else 'busy: timing lines will be owed'})")
    outcomes = asyncio.run(run(args, results))
    busy: dict[tuple[str, str, str], bool] = {}
    for o in outcomes:
        loads = [v for k, v in o.contributions if k == "_load"]
        key = (o.scenario.voice, o.condition, o.model)
        busy[key] = busy.get(key, False) or not quiet or any(machine.is_busy(v) for v in loads)
        o.contributions = [(k, v) for k, v in o.contributions if k != "_load"]
    rows = owe_timings(scoring.scorecard(outcomes), busy)
    report.save_scorecard(results, rows, outcomes, scoring.worst(outcomes))
    (results / "run.json").write_text(json.dumps({"load_before": load, "quiet": quiet}), encoding="utf-8")
    print("\n" + report.table(rows))
    print(f"\nSaved: {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
