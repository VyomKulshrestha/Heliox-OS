"""Conversation Bench command line.

  python -m benchmarks.conversation_bench run    [--audio DIR] [--results DIR] [--profile baseline|full]
                                                 [--sets ...] [--conditions ...] [--models ...] [--only TEXT]
  python -m benchmarks.conversation_bench score  RESULTS_DIR          (re-score saved timelines, no audio)
  python -m benchmarks.conversation_bench soak   [--minutes 120] [--audio DIR] [--results DIR]
  python -m benchmarks.conversation_bench list   [--audio DIR]        (scenarios and whether their audio is there)

``run`` prints the scorecard (each line per voice, condition and kind of model, its number, pass or fail) and
saves every scenario's timeline, the scorecard and the worst cases under the results folder, which defaults to
``~/heliox-conversation-bench-results/<time>`` (outside the repository: timelines quote what was said).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import audio as audio_refs
from . import conditions as conds
from . import quality, report, scoring
from . import timeline as tl
from .bench_root import chosen_root
from .platform_scan import scan, unavailable_messages
from .scenarios import MODEL_KINDS, SETS, Scenario, load_all, load_local
from .scoring import Row

DAEMON_DIR = Path(__file__).resolve().parents[2]
NOISE_FOR_NOT_ADDRESSED = ("as-recorded", "room-20db", "fan-10db", "voice-10db")


def default_results() -> Path:
    return Path.home() / "heliox-conversation-bench-results" / time.strftime("%Y%m%d-%H%M%S")


def audio_present(root: str, scenario: Scenario) -> bool:
    for step in scenario.steps:
        if step.say and not Path(audio_refs.resolve(root, audio_refs.parse_ref(step.say))).exists():
            return False
    return True


def jobs(
    scenarios: list[Scenario], *, profile: str, sets: list[str], conditions: list[str], models: list[str],
    only: str, babble: bool, na_blocks: int | None,
) -> list[tuple[Scenario, str, str]]:  # fmt: skip
    """(scenario, condition, model) to run, in an order that keeps like with like."""
    usable = conds.available(babble)
    out = []
    for s in scenarios:
        wanted_ids = [o for o in only.split(",") if o]
        if s.pending or s.set == "conversation_quality" or s.set not in sets:
            continue
        if wanted_ids and not any(o in s.id for o in wanted_ids):
            continue
        if s.set == "not_addressed" and na_blocks is not None and s.id.startswith("na-block-"):
            if int(s.id.rsplit("-", 1)[1]) >= na_blocks:
                continue
        if s.set == "understanding" and s.voice == "real":
            wanted = conditions or (usable if profile in ("full", "baseline") else ["as-recorded"])
        elif s.set == "not_addressed" and s.id.startswith("na-block-"):
            wanted = conditions or list(NOISE_FOR_NOT_ADDRESSED if profile == "full" else ["as-recorded"])
        else:
            wanted = ["as-recorded"]  # derived conditions apply to real sentences and not-addressed blocks
        for condition in wanted:
            if condition not in usable:
                continue
            for model in s.models:
                if models and model not in models:
                    continue
                out.append((s, condition, model))
    return out


def external_benches() -> list[Row]:
    """The Live Bench (precision and recall) and the Memory Bench, run as they are (no model, no audio)."""
    rows: list[Row] = []
    python = sys.executable
    try:
        live = subprocess.run(
            [python, "-m", "pilot.live.bench"], cwd=DAEMON_DIR, capture_output=True, text=True, timeout=900
        ).stdout
        for line in live.splitlines():
            parts = line.split()
            if len(parts) >= 8 and parts[0] in ("rules", "personal") and "/" in parts[2]:
                precision, recall, false = float(parts[6]), float(parts[7]), int(parts[4])
                rows.append(Row("3.1L", "text", parts[1], parts[0], precision, 48, None,
                                f"Live Bench precision {precision:.2f} recall {recall:.2f} false {false}"))  # fmt: skip
    except Exception as error:
        rows.append(Row("3.1L", "text", "-", "-", None, 0, None, f"Live Bench did not run: {error}"))
    try:
        memory = subprocess.run(
            [python, "-m", "pilot.memory.bench"], cwd=DAEMON_DIR, capture_output=True, text=True, timeout=900
        ).stdout
        match = re.search(r"(\d+)/(\d+) probes", memory)
        if match:
            got, total = int(match.group(1)), int(match.group(2))
            rows.append(Row("5M", "text", "-", "-", got / total, total, got == total, f"Memory Bench {got}/{total}"))
    except Exception as error:
        rows.append(Row("5M", "text", "-", "-", None, 0, None, f"Memory Bench did not run: {error}"))
    return rows


async def run_jobs(args: argparse.Namespace, todo: list[tuple[Scenario, str, str]], results: Path) -> list:
    from .driver import Harness, model_kind

    harness = Harness(audio_root=args.audio, data_dir=chosen_root(args.root, results), model=model_kind("cloud"),
                      pace=args.pace, owner_settings=True)  # fmt: skip
    await harness.build()
    outcomes = []
    cpu: list[float] = []
    try:
        for index, (scenario, condition, model) in enumerate(todo, 1):
            harness.model = model_kind(model)
            events = await harness.run(scenario, condition)
            marker = next((e for e in events if e.event == tl.SCENARIO), None)
            if marker is not None and marker.data.get("cpu_percent") is not None:
                cpu.append(marker.data["cpu_percent"])
            outcome = scoring.score(scenario, tl.measure(events), condition=condition, model=model)
            report.save_outcome(results, outcome, events)
            outcomes.append(outcome)
            verdict = "ok " if outcome.passed else "BAD"
            reply = outcome.measures.reply_start
            print(f"[{index}/{len(todo)}] {verdict} {scenario.id} ({condition}, {model}) route={outcome.measures.route} "
                  f"reply={'-' if reply is None else f'{reply:.2f}s'} heard={outcome.heard[:60]!r} "
                  f"{'; '.join(outcome.failures)[:120]}", flush=True)  # fmt: skip
    finally:
        await harness.close(session_to=results)
        (results / "run.json").write_text(
            json.dumps(
                {
                    "worker_spawns": harness.worker_spawns,
                    "voice_load_s": harness.voice_load_s,
                    "pump_lag_s": harness.pump.max_lag if harness.pump else 0,
                    "machine_cpu_percent_median": sorted(cpu)[len(cpu) // 2] if cpu else None,
                }
            ),
            encoding="utf-8",
        )
    return outcomes


def quality_rows(cases: list[Scenario], real_model: bool) -> list[Row]:
    """The fast spoken layer's cases per kind of model; one pending row when the layer is not there yet."""
    if not cases:
        return []
    if not quality.available():
        return [Row("Q", "synthetic", "-", "-", None, len(cases), None,
                    f"{len(cases)} cases pending: {cases[0].pending}")]  # fmt: skip
    rows = []
    for model in MODEL_KINDS:
        judged = asyncio.run(quality.run(cases, model, real_model))
        ok = sum(1 for _, failures in judged if not failures)
        worst = "; ".join(f"{s.id}: {f[0]}" for s, f in judged if f)[:200]
        rows.append(Row("Q", "synthetic", "text", model, ok / len(judged), len(judged), ok == len(judged), worst))
    return rows


def finish(results: Path, outcomes: list, with_benches: bool, extra_rows: list[Row]) -> None:
    extra = (external_benches() if with_benches else []) + extra_rows
    rows = scoring.scorecard(outcomes, extra)
    worst = scoring.worst(outcomes)
    report.save_scorecard(results, rows, outcomes, worst)
    print("\n" + report.table(rows))
    print("\nWorst cases (timelines in " + str(results / "timelines") + "):")
    for o in worst:
        print(f"  {o.scenario.id} ({o.condition}, {o.model}): {'; '.join(o.failures) or 'passed'}")
    said = [line for o in outcomes for line in o.measures.spoken] + [e for o in outcomes for e in o.measures.dispatched]
    platform = {"unavailable_messages": unavailable_messages(said), "platform_branches": scan(DAEMON_DIR)}
    (results / "platform.json").write_text(json.dumps(platform, indent=1), encoding="utf-8")
    print("\nPlatform-specific code on the spoken path:")
    for module, branches in platform["platform_branches"].items():
        print(f"  {module}: " + ", ".join(f"{name} (line {line})" for name, line, _ in branches))
    print(f"\nSaved: {results}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.conversation_bench")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "soak", "list"):
        p = sub.add_parser(name)
        p.add_argument("--audio", default=None)
        p.add_argument("--results", default=None)
        p.add_argument("--pace", type=float, default=1.0)
        p.add_argument("--root", default=None, help="the bench's own Heliox folders, kept between runs")
    run_p = sub.choices["run"]
    run_p.add_argument("--profile", choices=("baseline", "full", "quick"), default="baseline")
    run_p.add_argument("--sets", default=",".join(SETS))
    run_p.add_argument(
        "--conditions", default="", help="for real sentences and not-addressed blocks; the rest run as recorded"
    )
    run_p.add_argument("--models", default="")
    run_p.add_argument("--only", default="")
    run_p.add_argument("--na-blocks", type=int, default=None)
    run_p.add_argument("--no-benches", action="store_true")
    run_p.add_argument("--real-model", action="store_true", help="opt-in, costs money: quality cases on a real model")
    sub.choices["soak"].add_argument("--minutes", type=float, default=120.0)
    score_p = sub.add_parser("score")
    score_p.add_argument("results")
    args = parser.parse_args(argv)

    scenarios = load_all()
    if args.command == "score":
        folder = Path(args.results)
        by_id = {s.id: s for s in scenarios}
        outcomes = []
        for path in sorted((folder / "timelines").rglob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            o = data["outcome"]
            if o["scenario"] in by_id:
                events = tl.parse(data)
                outcomes.append(scoring.score(by_id[o["scenario"]], tl.measure(events), condition=o["condition"],
                                              model=o["model"]))  # fmt: skip
        finish(folder, outcomes, False, [])
        return 0

    args.audio = audio_refs.audio_root(args.audio)
    scenarios = scenarios + load_local(args.audio)  # the person's own recordings, if any
    if args.command == "list":
        for s in scenarios:
            state = "pending" if s.pending else ("ok" if audio_present(args.audio, s) else "AUDIO MISSING")
            print(f"{s.set:<21}{s.voice:<10}{s.id:<48}{state}")
        return 0

    results = Path(args.results) if args.results else default_results()
    results.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    babble = (Path(args.audio) / "synthetic" / "babble.wav").exists()
    if args.command == "soak":
        from . import soak
        from .driver import Harness, model_kind

        pool = [
            s for s in scenarios if not s.pending and audio_present(args.audio, s) and s.voice in ("real", "synthetic")
        ]

        async def go() -> None:
            harness = Harness(audio_root=args.audio, data_dir=chosen_root(args.root, results),
                              model=model_kind("cloud"), pace=args.pace, owner_settings=True)  # fmt: skip
            await harness.build()
            try:
                result = await soak.run(harness, pool, args.minutes)
            finally:
                await harness.close(session_to=results)
            text = soak.summary(result)
            print(text)
            (results / "soak.txt").write_text(text + "\n", encoding="utf-8")
            (results / "soak.json").write_text(
                json.dumps(result, default=lambda o: o.__dict__, indent=1), encoding="utf-8"
            )

        asyncio.run(go())
        return 0

    sets = [s for s in args.sets.split(",") if s]
    todo = jobs(
        scenarios,
        profile=args.profile,
        sets=sets,
        conditions=[c for c in args.conditions.split(",") if c],
        models=[m for m in args.models.split(",") if m in MODEL_KINDS],
        only=args.only,
        babble=babble,
        na_blocks=args.na_blocks if args.profile != "quick" else min(args.na_blocks or 1, 1),
    )
    missing = sorted({s.id for s, _, _ in todo if not audio_present(args.audio, s)})
    if missing:
        print(f"Skipping {len(missing)} scenarios whose audio is not in {args.audio}: {missing[:8]}")
        todo = [j for j in todo if j[0].id not in missing]
    print(f"{len(todo)} runs; audio from {args.audio}; results in {results}")
    outcomes = asyncio.run(run_jobs(args, todo, results)) if todo else []
    cases = [s for s in scenarios if s.set == "conversation_quality" and "conversation_quality" in sets]
    finish(results, outcomes, not args.no_benches, quality_rows(cases, args.real_model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
