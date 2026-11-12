"""The Conversation Bench's timing lines on a quiet machine with no audio device: a CI runner, synthetic voice only.

  python -m benchmarks.conversation_bench.ci_timing --audio <folder with synthetic/> [--results DIR] [--summary FILE]

Runs the synthetic scenarios behind the timing lines (1.1a local answer starts, 1.1b/c model answer starts with
the stub fast model, 1.1d longest silence, 1.2b voice stops after barge-in, 4.1 simple request under way) through
the real listener and dispatch with the bench's file-fed microphone and silent output, the conversation layer up
with the stub fast model (``stack.py``). Before each scenario the machine's load must be under the busy threshold;
a scenario that cannot get a quiet machine is skipped and its lines are owed. Prints the scorecard as CHECK lines,
each labelled "synthetic voice, GitHub runner (N cores), CPU only", writes a Markdown summary with the stage
breakdown on the wall clock and on the audio clock, and saves every timeline for the artifact.

Real voice never runs here: scenarios with ``voice: real`` are refused, so no recording of the owner can be needed
or uploaded.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from . import machine, report, scoring, stack, stages
from . import timeline as tl
from .models import KINDS
from .scenarios import Scenario, load_all
from .targets import BY_ID

TIMING_LINES = ("1.1a", "1.1b", "1.1c", "1.1d", "1.2b", "4.1")
SCENARIO_PREFIXES = ("tt-syn-reply-", "tt-syn-bargein-", "c-syn-stop", "c-syn-wait", "u-syn-hey_heliox")
GROUPS = {"local reply": ("tt-syn-reply-local", "u-syn-hey_heliox_what_can"),
          "model reply": ("tt-syn-reply-model",), "barge-in": ("tt-syn-bargein-", "c-syn-stop", "c-syn-wait")}  # fmt: skip


def chosen(scenarios: list[Scenario]) -> list[Scenario]:
    """The synthetic scenarios behind the timing lines; real voice is refused outright (pure)."""
    out = [s for s in scenarios if s.id.startswith(SCENARIO_PREFIXES) and not s.pending]
    if any(s.voice != "synthetic" for s in out):
        raise ValueError("ci_timing runs synthetic voice only")
    return out


def target_text(line_id: str) -> str:
    line = BY_ID[line_id]
    return f"{'<=' if line.direction == 'max' else '>='}{line.target:g}{line.unit}"


def check_lines(rows: list[scoring.Row], note: str, owed: set[str]) -> list[str]:
    """CHECK lines for the timing lines, one per kind of model (pure)."""
    out = []
    for r in rows:
        if r.line not in TIMING_LINES:
            continue
        if r.line in owed:
            out.append(stages.check_line(r.line, None, target_text(r.line), None, f"{note}; {r.model}; machine busy"))
            continue
        value = None if r.value is None or r.value == float("inf") else round(float(r.value), 2)
        ok = r.ok if value is not None else False
        out.append(stages.check_line(r.line, value, target_text(r.line), ok, f"{note}; {r.model} model; n={r.n}"))
    return out


def stage_table(groups: dict[str, list[dict[str, Any]]]) -> str:
    lines = ["| scenarios | stage | median, wall clock | median, audio clock | n |", "|---|---|---|---|---|"]
    for name, per in groups.items():
        for stage, v in stages.summarise(per).items():
            audio = "-" if v["audio"] is None else f"{v['audio']:.2f} s"
            lines.append(f"| {name} | {stage} | {v['wall']:.2f} s | {audio} | {v['n']} |")
    return "\n".join(lines)


def configure(config: Any) -> None:
    """The recommended English setup on a fresh machine: Parakeet for the words, Kokoro as the voice."""
    from pilot.system.speech_models import local_path

    folder = local_path("parakeet-tdt-0.6b-v2-int8")
    if folder is not None:
        config.voice.final_pass = "parakeet"
        config.voice.parakeet_model_dir = str(folder)
    config.voice.tts_engine = "kokoro_tts"
    config.voice.tts_voice = "af_heart"


async def run(args: argparse.Namespace, results: Path) -> tuple[list[scoring.Outcome], dict[str, list], set[str]]:
    from .driver import Harness

    data_dir = Path(os.environ.get("XDG_DATA_HOME") or results / "data")
    harness = Harness(audio_root=args.audio, data_dir=data_dir, model=KINDS["cloud"], configure=configure)
    await harness.build()
    stack.enable(harness)
    outcomes: list[scoring.Outcome] = []
    groups: dict[str, list[dict[str, Any]]] = {name: [] for name in GROUPS}
    owed: set[str] = set()
    wanted = [w for w in args.only.split(",") if w]
    todo = [(s, m) for s in chosen(load_all()) for m in s.models if not wanted or any(w in s.id for w in wanted)]
    try:
        for i, (scenario, model) in enumerate(todo, 1):
            quiet, load = machine.wait_for_quiet(args.max_wait, sample=lambda: machine.cpu_percent(2.0),
                                                 threshold=args.threshold)  # fmt: skip
            if not quiet:
                owed.update(scenario.lines)
                print(f"[{i}/{len(todo)}] skipped {scenario.id} ({model}): machine busy ({load}%)", flush=True)
                continue
            stack.select(harness, KINDS[model])
            events = await harness.run(scenario, "as-recorded")
            outcome = scoring.score(scenario, tl.measure(events), condition="as-recorded", model=model)
            report.save_outcome(results, outcome, events)
            outcomes.append(outcome)
            for name, prefixes in GROUPS.items():
                if scenario.id.startswith(prefixes):
                    groups[name].append(stages.breakdown(events))
            print(f"[{i}/{len(todo)}] {scenario.id} ({model}) load {load}% reply={outcome.measures.reply_start} "
                  f"stop={outcome.measures.stop_time}", flush=True)  # fmt: skip
    finally:
        await harness.close()
    return outcomes, groups, owed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.conversation_bench.ci_timing")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--results", default="conversation-timing")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--threshold", type=float, default=machine.BUSY_PERCENT)
    parser.add_argument("--only", default="", help="comma-separated id fragments (a quick check of the wiring)")
    parser.add_argument("--max-wait", type=float, default=120.0, help="seconds to wait for a quiet machine")
    args = parser.parse_args(argv)
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    outcomes, groups, owed = asyncio.run(run(args, results))
    where = "GitHub runner" if os.environ.get("GITHUB_ACTIONS") == "true" else "this machine, not a runner"
    note = stages.label(os.cpu_count(), where)
    rows = scoring.scorecard(outcomes)
    checks = check_lines(rows, note, owed)
    report.save_scorecard(results, rows, outcomes, scoring.worst(outcomes))
    (results / "stages.json").write_text(json.dumps(groups, indent=1), encoding="utf-8")
    for line in checks:
        print(line)
    text = "\n".join([
        f"### Conversation Bench timing lines: {note}",
        "",
        "These numbers are for a small CPU-only machine with nothing else running and synthetic voices. They say "
        "how long Heliox's own stages take on such a machine; they do not say how the owner's laptop or voice "
        "behaves (another CPU, other load, a GPU, a real microphone and speaker, a real voice).",
        "",
        "```", *checks, "```", "",
        "Where the time goes (wall clock = what a person waits; audio clock = seconds of microphone audio fed; "
        "they differ only when the listener falls behind):",
        "",
        stage_table(groups),
        "",
        f"Run took {(time.perf_counter() - started) / 60:.0f} minutes. Timelines are in the artifact.",
        "",
    ])  # fmt: skip
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
