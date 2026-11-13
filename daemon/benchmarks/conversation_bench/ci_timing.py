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

Comparing settings (``--settings parakeet_threads=2,3,4 --repeats 3``): every scenario runs once per setting, back
to back, in an order that rotates each repeat, and the summary adds each setting's difference from the first with its
range over the repeats (compare_settings.py). The scorecard lines are the first setting's, as before. A CALIBRATION
line (the CPU, a fixed Python loop, one Parakeet pass on a fixed clip) is printed beside them on every run.
A comparison runs only the scenarios its setting can move (compare_settings.SCOPE) unless ``--only`` names others,
is refused before anything is built when its estimate does not fit ``--limit-minutes``, and writes summary.md and
compared.json after every repeat, so a run stopped part way still reports the repeats it finished.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from . import compare_settings, machine, report, scoring, stack, stages
from . import timeline as tl
from .models import KINDS
from .scenarios import Scenario, load_all
from .targets import BY_ID

TIMING_LINES = ("1.1a", "1.1b", "1.1c", "1.1d", "1.2b", "4.1")
SCENARIO_PREFIXES = ("tt-syn-reply-", "tt-syn-bargein-", "c-syn-stop", "c-syn-wait", "u-syn-hey_heliox")
GROUPS = {"local reply": ("tt-syn-reply-local", "u-syn-hey_heliox_what_can"),
          "model reply": ("tt-syn-reply-model",), "barge-in": ("tt-syn-bargein-", "c-syn-stop", "c-syn-wait"),
          "spoken requests": ("u-syn-hey_heliox",)}  # fmt: skip


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


CALIBRATION_CLIP = "synthetic/am_michael/1.0/hey_heliox_what_can_i_say.wav"


def calibrate(harness: Any, audio_root: str) -> str:
    """The CALIBRATION line: this runner's CPU, a fixed Python loop and one Parakeet pass on a fixed clip."""
    from . import wavio

    transcriber = harness.listener._parakeet() if harness.listener is not None else None
    clip_path = Path(audio_root) / CALIBRATION_CLIP
    clip = wavio.read(str(clip_path)) if clip_path.exists() else None
    passed = compare_settings.parakeet_pass_seconds(transcriber, clip) if clip is not None else None
    return compare_settings.calibration_line(
        compare_settings.cpu_name(),
        os.cpu_count(),
        compare_settings.python_loop_seconds(),
        passed,
        getattr(transcriber, "threads", None),
    )


def plan(args: argparse.Namespace) -> tuple[str, list[str], int, list[str], list[tuple[Scenario, str]]]:
    """(setting name, its values, repeats, the id fragments chosen, the scenario passes of one setting and repeat).
    A comparison runs only the scenarios its setting can move (compare_settings.SCOPE) unless --only names others."""
    name, values = compare_settings.parse(args.settings)
    repeats = args.repeats or (3 if name else 1)
    wanted = [w for w in args.only.split(",") if w] or list(compare_settings.SCOPE.get(name, ()))
    todo = [(s, m) for s in chosen(load_all()) for m in s.models if not wanted or any(w in s.id for w in wanted)]
    return name, values, repeats, wanted, todo


async def run(
    args: argparse.Namespace, results: Path, on_repeat: Any = None
) -> tuple[list[scoring.Outcome], dict[str, list], set[str], dict[str, dict], str]:
    """Every chosen scenario once per setting and repeat. ``on_repeat(outcomes, groups, owed, compared,
    calibration, repeats_done)`` runs after every repeat, so a run stopped part way still leaves what it finished."""
    from .driver import Harness

    # The scratch root ci_prefetch filled (HELIOX_BENCH_ROOT, whose models the workflow caches), or a fresh one.
    data_dir = Path(os.environ.get("HELIOX_BENCH_ROOT") or results / "data")
    harness = Harness(audio_root=args.audio, data_dir=data_dir, model=KINDS["cloud"], configure=configure)
    await harness.build()
    stack.enable(harness)
    name, values, repeats, _wanted, todo = plan(args)
    outcomes: list[scoring.Outcome] = []
    groups: dict[str, list[dict[str, Any]]] = {group: [] for group in GROUPS}
    # per scenario group, per setting, per repeat: the stage breakdowns, for the differences
    compared: dict[str, dict[str, dict[int, list[dict[str, Any]]]]] = {
        group: {value: {} for value in values} for group in GROUPS
    }
    owed: set[str] = set()
    total = len(todo) * len(values) * repeats
    i = 0
    try:
        calibration = calibrate(harness, args.audio)
        print(calibration, flush=True)
        for repeat in range(repeats):
            if repeat and on_repeat is not None:
                on_repeat(outcomes, groups, owed, compared if name else {}, calibration, repeat)
            for scenario, model in todo:
                for value in compare_settings.order(values, repeat):
                    i += 1
                    quiet, load = machine.wait_for_quiet(args.max_wait, sample=lambda: machine.cpu_percent(2.0),
                                                         threshold=args.threshold)  # fmt: skip
                    if not quiet:
                        owed.update(scenario.lines)
                        print(f"[{i}/{total}] skipped {scenario.id} ({model}): machine busy ({load}%)", flush=True)
                        continue
                    if name:
                        compare_settings.SETTINGS[name](harness, value)
                    stack.select(harness, KINDS[model])
                    events = await harness.run(scenario, "as-recorded")
                    outcome = scoring.score(scenario, tl.measure(events), condition="as-recorded", model=model)
                    first = value == values[0]
                    if first:
                        report.save_outcome(results, outcome, events)
                        outcomes.append(outcome)
                    for group, prefixes in GROUPS.items():
                        if scenario.id.startswith(prefixes):
                            breakdown = stages.breakdown(events)
                            if first:
                                groups[group].append(breakdown)
                            compared[group][value].setdefault(repeat, []).append(breakdown)
                    shown = f" {name}={value}" if name else ""
                    print(f"[{i}/{total}] {scenario.id} ({model}){shown} load {load}% "
                          f"reply={outcome.measures.reply_start} stop={outcome.measures.stop_time}", flush=True)  # fmt: skip
    finally:
        await harness.close(session_to=results)
    return outcomes, groups, owed, (compared if name else {}), calibration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.conversation_bench.ci_timing")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--results", default="conversation-timing")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--threshold", type=float, default=machine.BUSY_PERCENT)
    parser.add_argument("--only", default="", help="comma-separated id fragments (a quick check of the wiring)")
    parser.add_argument("--max-wait", type=float, default=120.0, help="seconds to wait for a quiet machine")
    parser.add_argument(
        "--settings", default="", help='settings to compare inside this job, e.g. "parakeet_threads=2,3,4"'
    )
    parser.add_argument("--repeats", type=int, default=0, help="repeats of every setting (default 3 when comparing)")
    parser.add_argument(
        "--limit-minutes", type=float, default=0.0, help="the step's time limit: a run that cannot fit is refused"
    )
    args = parser.parse_args(argv)
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    name, values, repeats, wanted, todo = plan(args)
    refused = compare_settings.refusal(len(todo), len(values), repeats, args.limit_minutes)
    if refused:
        print(refused)
        publish(results, args.summary, f"### Conversation Bench timing lines\n\n{refused}\n")
        return 2
    scope = ", ".join(wanted) if wanted else "every timing scenario"
    print(
        f"{len(todo)} scenario passes per setting ({scope}), {len(values)} setting(s), {repeats} repeat(s): about "
        f"{compare_settings.estimate_minutes(len(todo) * len(values) * repeats):.0f} minutes",
        flush=True,
    )

    def write(outcomes, groups, owed, compared, calibration, done: int) -> str:
        return save(results, args, outcomes, groups, owed, compared, calibration, scope, len(todo), done, repeats,
                    started)  # fmt: skip

    outcomes, groups, owed, compared, calibration = asyncio.run(run(args, results, on_repeat=write))
    text = write(outcomes, groups, owed, compared, calibration, repeats)
    publish(results, args.summary, text)
    print(text)
    return 0


def publish(results: Path, summary: str | None, text: str) -> None:
    """The summary as results/summary.md (the workflow shows it even when the step is stopped), and appended to
    ``summary`` when given."""
    (results / "summary.md").write_text(text + "\n", encoding="utf-8")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


def save(
    results: Path,
    args: argparse.Namespace,
    outcomes: list[scoring.Outcome],
    groups: dict[str, list],
    owed: set[str],
    compared: dict[str, dict],
    calibration: str,
    scope: str,
    passes: int,
    done: int,
    repeats: int,
    started: float,
) -> str:
    """Write the scorecard, stages.json, compared.json and summary.md for the repeats finished so far; the text."""
    where = "GitHub runner" if os.environ.get("GITHUB_ACTIONS") == "true" else "this machine, not a runner"
    note = stages.label(os.cpu_count(), where)
    rows = scoring.scorecard(outcomes)
    checks = check_lines(rows, note, owed)
    report.save_scorecard(results, rows, outcomes, scoring.worst(outcomes))
    (results / "stages.json").write_text(json.dumps(groups, indent=1), encoding="utf-8")
    name = compare_settings.parse(args.settings)[0]
    if compared:
        (results / "compared.json").write_text(json.dumps(compared, indent=1), encoding="utf-8")
    comparison = (
        [
            f"Settings compared on this runner, interleaved ({name}; differences are against the first setting, taken "
            "within each repeat; the range is over the repeats):",
            "",
            compare_settings.difference_table(name, compared),
            "",
        ]
        if compared
        else []
    )
    progress = (
        f"Finished {done} of {repeats} repeats so far; the run was stopped or is still going."
        if done < repeats
        else f"Finished all {repeats} repeat(s)."
    )
    text = "\n".join([
        f"### Conversation Bench timing lines: {note}",
        "",
        "These numbers are for a small CPU-only machine with nothing else running and synthetic voices. They say "
        "how long Heliox's own stages take on such a machine; they do not say how the owner's laptop or voice "
        "behaves (another CPU, other load, a GPU, a real microphone and speaker, a real voice).",
        "",
        f"Scenarios run: {scope} ({passes} scenario passes per setting and repeat). {progress}",
        "",
        "```", *checks, calibration, "```", "",
        *comparison,
        "Where the time goes (wall clock = what a person waits; audio clock = seconds of microphone audio fed; "
        "they differ only when the listener falls behind):",
        "",
        stage_table(groups),
        "",
        f"Run took {(time.perf_counter() - started) / 60:.0f} minutes. Timelines are in the artifact.",
        "",
    ])  # fmt: skip
    if done < repeats:
        (results / "summary.md").write_text(text + "\n", encoding="utf-8")
    return text


if __name__ == "__main__":
    raise SystemExit(main())
