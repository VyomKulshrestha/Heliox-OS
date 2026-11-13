"""Comparing settings inside one timing job: one runner, the settings interleaved, the differences with their spread.

Two runs of the same commit on shared runners differed by up to 0.6 s in a stage (runs 43267447050 and 43271354855,
12 November), so one run per setting cannot tell a 0.1 to 0.3 s change. Here every scenario runs once per setting,
back to back, in an order that rotates each repeat so a runner that drifts hits every setting alike; each stage's
difference against the first setting is taken within each repeat, and reported as the median with its range over
the repeats. A calibration line (the CPU's name, a fixed pure-Python loop, one Parakeet pass on a fixed clip) lets
runs on different runners be put on one scale.

Settings are given as "name=v1,v2,..." (``--settings parakeet_threads=2,3,4``); the names are in SETTINGS. A setting
with a fixed set of values (VALUES) refuses any other. ``python -m benchmarks.conversation_bench.compare_settings
--check "<spec>"`` checks a spec with nothing but the standard library, so the workflow can refuse a wrong one in
its first seconds, before the install and the model fetch.

``final_pass=whisper,parakeet`` compares the listener's final pass. Besides the timing (the "recognition" stage is
the final pass's own time after the end of the turn), it counts word errors per setting on the same sentences as
recorded and through a Bluetooth headset in call mode (CONDITIONS): the call-mode copies run once, in the first
repeat, as the words do not depend on the repeat.
"""

from __future__ import annotations

import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import stages


def _parakeet_threads(harness: Any, value: str) -> None:
    """Parakeet's thread count for the listener's final pass, loaded again (outside any scenario's timing)."""
    listener = harness.listener
    transcriber = listener._parakeet() if listener is not None else None
    if transcriber is None:
        raise RuntimeError("parakeet_threads: the listener has no Parakeet final pass")
    threads = int(value)
    if transcriber.threads != threads or transcriber._recognizer is None:
        transcriber.threads = threads
        transcriber._recognizer = None
        transcriber.load()


def _final_pass(harness: Any, value: str) -> None:
    """The listener's final pass: "whisper" or "parakeet" (Parakeet from the models fetched into the bench root),
    switched on the settings the listener reads on every utterance, and Parakeet loaded outside any timing."""
    listener = harness.listener
    if listener is None:
        raise RuntimeError("final_pass: no listener")
    voice = listener.config.voice
    if value == "parakeet" and not getattr(voice, "parakeet_model_dir", ""):
        from pilot.system.speech_models import local_path

        folder = local_path("parakeet-tdt-0.6b-v2-int8")
        if folder is not None:
            voice.parakeet_model_dir = str(folder)
    voice.final_pass = value
    if value == "parakeet":
        transcriber = listener._parakeet()
        if transcriber is None or not transcriber.load():
            raise RuntimeError("final_pass=parakeet: Parakeet is not on this machine (nothing is downloaded here)")


SETTINGS: dict[str, Callable[[Any, str], None]] = {"parakeet_threads": _parakeet_threads, "final_pass": _final_pass}
# Settings whose values are a fixed set: anything else is refused when the spec is read.
VALUES: dict[str, tuple[str, ...]] = {"final_pass": ("whisper", "parakeet")}
# Conditions run besides "as-recorded" for a setting, in the first repeat only: their word errors are counted per
# setting, their timings are not compared (conditions.py: "bt-call" is a Bluetooth headset in call mode).
CONDITIONS: dict[str, tuple[str, ...]] = {"final_pass": ("bt-call",)}

# The scenarios each setting can move, as id prefixes: a comparison runs only these unless --only names others.
# Parakeet's threads change the final pass, so the reply's start after a request; in run 43290465258 (all 42
# scenario passes, threads 2, 3 and 4) the barge-in stop times moved 0.001 s between thread counts, so they are left
# out of this comparison.
SCOPE: dict[str, tuple[str, ...]] = {
    "parakeet_threads": ("u-syn-hey_heliox", "tt-syn-reply-model"),
    "final_pass": ("u-syn-hey_heliox",),  # sentences with a known text: their words can be counted
}

# What one scenario pass costs on a 4-core GitHub runner, the load check included: run 43290465258 did 330 passes
# in 144.5 minutes (26.3 s each), after 1.7 minutes building the harness and calibrating.
SECONDS_PER_PASS = 26.3
SETUP_SECONDS = 105.0
# The share of the step's limit an estimate may use: runners differ, and the summary must still be written.
LIMIT_SHARE = 0.9


def estimate_minutes(passes: int) -> float:
    """How long ``passes`` scenario passes take on a 4-core runner, setup included (pure)."""
    return (SETUP_SECONDS + passes * SECONDS_PER_PASS) / 60


def refusal(scenarios: int, settings: int, repeats: int, limit_minutes: float) -> str:
    """ "" when scenarios x settings x repeats fits in the step's limit; else one sentence saying what would (pure)."""
    if limit_minutes <= 0:
        return ""
    passes = scenarios * settings * repeats
    budget = limit_minutes * LIMIT_SHARE
    need = estimate_minutes(passes)
    if need <= budget:
        return ""
    per_repeat = scenarios * settings * SECONDS_PER_PASS / 60
    fit_repeats = int((budget - SETUP_SECONDS / 60) // per_repeat) if per_repeat else 0
    fit_scenarios = int((budget - SETUP_SECONDS / 60) * 60 // (settings * repeats * SECONDS_PER_PASS))
    options = []
    if fit_repeats >= 1:
        options.append(f"{fit_repeats} repeat{'s' if fit_repeats != 1 else ''} of these {scenarios} scenarios")
    if fit_scenarios >= 1:
        options.append(f"at most {fit_scenarios} scenarios at {repeats} repeats (name them with --only)")
    if settings > 2:
        options.append("fewer settings")
    what = "; or ".join(options) or "fewer settings"
    return (
        f"Refused: {passes} scenario passes ({scenarios} scenarios x {settings} settings x {repeats} repeats) take "
        f"about {need:.0f} minutes at {SECONDS_PER_PASS:g} s each, more than the {budget:.0f} of this step's "
        f"{limit_minutes:g}-minute limit that may be used; what fits: {what}."
    )


def parse(spec: str) -> tuple[str, list[str]]:
    """ "parakeet_threads=2,3,4" -> ("parakeet_threads", ["2", "3", "4"]) (pure). "" -> ("", [""])."""
    if not spec.strip():
        return "", [""]
    name, _, values = spec.partition("=")
    name = name.strip()
    if name not in SETTINGS:
        raise ValueError(f"unknown setting {name!r}; known: {', '.join(SETTINGS)}")
    chosen = [v.strip() for v in values.split(",") if v.strip()]
    if len(chosen) < 1:
        raise ValueError(f"no values for {name}")
    allowed = VALUES.get(name)
    wrong = [v for v in chosen if allowed is not None and v not in allowed]
    if wrong:
        raise ValueError(f"{name} takes {', '.join(allowed or ())}; not {', '.join(wrong)}")
    return name, chosen


def order(values: list[str], repeat: int) -> list[str]:
    """The settings' order for one repeat, rotated by one each time so each comes first as often (pure)."""
    shift = repeat % len(values)
    return values[shift:] + values[:shift]


def differences(per: dict[str, dict[int, list[dict[str, Any]]]]) -> dict[str, dict[str, dict[str, float]]]:
    """For each setting after the first and each stage: its median minus the first setting's, within each repeat,
    then the median and range of those differences over the repeats (pure).

    ``per[setting][repeat]`` is that repeat's list of stage breakdowns (stages.breakdown) for one scenario group."""
    settings = list(per)
    if len(settings) < 2:
        return {}
    base = settings[0]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for setting in settings[1:]:
        by_stage: dict[str, list[float]] = {}
        for repeat in per[setting]:
            if repeat not in per[base]:
                continue
            mine = stages.summarise(per[setting][repeat])
            theirs = stages.summarise(per[base][repeat])
            for stage in mine.keys() & theirs.keys():
                by_stage.setdefault(stage, []).append(float(mine[stage]["wall"]) - float(theirs[stage]["wall"]))
        out[setting] = {
            stage: {
                "median": round(statistics.median(diffs), 3),
                "low": round(min(diffs), 3),
                "high": round(max(diffs), 3),
                "repeats": len(diffs),
            }
            for stage, diffs in sorted(by_stage.items())
        }
    return out


def difference_table(name: str, groups: dict[str, dict[str, dict[int, list[dict[str, Any]]]]]) -> str:
    """Markdown: per scenario group and stage, each setting's median, and its difference from the first setting with
    the range over repeats (pure)."""
    lines = [
        f"| scenarios | stage | {name} | median, wall clock | difference from {name}={{first}}, median (range) | repeats |",
        "|---|---|---|---|---|---|",
    ]
    for group, per in groups.items():
        settings = list(per)
        if not settings:
            continue
        lines[0] = lines[0].replace("{first}", settings[0])
        diffs = differences(per)
        for setting in settings:
            pooled = [b for runs in per[setting].values() for b in runs]
            for stage, v in stages.summarise(pooled).items():
                d = diffs.get(setting, {}).get(stage)
                shown = "-" if d is None else f"{d['median']:+.2f} s ({d['low']:+.2f} to {d['high']:+.2f})"
                count = "-" if d is None else str(d["repeats"])
                lines.append(f"| {group} | {stage} | {setting} | {v['wall']:.2f} s | {shown} | {count} |")
    return "\n".join(lines)


def cpu_name() -> str:
    """The processor's name as the system gives it ("" when it does not)."""
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or ""


def python_loop_seconds(iterations: int = 3_000_000) -> float:
    """A fixed pure-Python loop, the best of three: how fast this runner's single core is."""
    best = float("inf")
    for _ in range(3):
        started = time.perf_counter()
        total = 0
        for i in range(iterations):
            total += i & 7
        best = min(best, time.perf_counter() - started)
    return round(best, 3)


def parakeet_pass_seconds(transcriber: Any, clip: Any) -> float | None:
    """One Parakeet pass on a fixed clip after a warm one, the best of three; None without Parakeet."""
    if transcriber is None or not transcriber.load():
        return None
    transcriber.transcribe(clip)
    best = float("inf")
    for _ in range(3):
        started = time.perf_counter()
        transcriber.transcribe(clip)
        best = min(best, time.perf_counter() - started)
    return round(best, 3)


def calibration_line(cpu: str, cores: int | None, loop_s: float, pass_s: float | None, threads: int | None) -> str:
    """CALIBRATION line printed beside the scorecard (pure)."""
    parakeet = "-" if pass_s is None else f"{pass_s:.3f} s on {threads} threads"
    return f"CALIBRATION cpu={cpu or 'unknown'!r} cores={cores} python_loop={loop_s:.3f} s parakeet_pass={parakeet}"


def word_errors(said: str, heard: str) -> tuple[int, int]:
    """(word errors, words said) after the scorecard's normalising (pure)."""
    from .scoring import edit_distance, normalise

    reference = normalise(said)
    return edit_distance(reference, normalise(heard)), len(reference)


def word_error_table(name: str, tallies: dict[str, dict[str, list[int]]]) -> str:
    """Markdown: per condition and setting, n sentences, word error rate (errors over words said), and the difference
    in points from the first setting (pure). ``tallies[value][condition] = [errors, words, sentences]``."""
    values = list(tallies)
    if not values:
        return ""
    lines = [f"| condition | {name} | sentences | words | word error | difference from {name}={values[0]} |",
             "|---|---|---|---|---|---|"]  # fmt: skip
    conditions = sorted({c for per in tallies.values() for c in per}, key=lambda c: (c != "as-recorded", c))
    for condition in conditions:
        first = tallies[values[0]].get(condition)
        base = 100.0 * first[0] / first[1] if first and first[1] else None
        for value in values:
            errors, words, n = tallies[value].get(condition, [0, 0, 0])
            rate = 100.0 * errors / words if words else None
            shown = "-" if rate is None else f"{rate:.1f} %"
            diff = "-" if rate is None or base is None or value == values[0] else f"{rate - base:+.1f} points"
            lines.append(f"| {condition} | {value} | {n} | {words} | {shown} | {diff} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """--check "<spec>": exit 0 when the settings spec is one a run can compare, 2 (with the reason) otherwise."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m benchmarks.conversation_bench.compare_settings")
    parser.add_argument("--check", required=True, help='a settings spec, e.g. "final_pass=whisper,parakeet"')
    args = parser.parse_args(argv)
    try:
        name, values = parse(args.check)
    except ValueError as error:
        print(f"Refused: settings {args.check!r}: {error}")
        return 2
    print(f"Settings to compare: {name or '(none: the usual run)'}" + (f" = {', '.join(values)}" if name else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
