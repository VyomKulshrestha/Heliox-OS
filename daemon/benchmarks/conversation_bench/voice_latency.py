"""From the person's last word to Heliox's first sound, through the real listener, conversation layer and voice.

    python -m benchmarks.conversation_bench.voice_latency --root <bench root> [--audio <folder>]
        [--only u-real] [--model stub|configured] [--paid] [--wait-quiet 10] [--allow-busy] [--out <file>]

Each scenario's recording is played into the real listener on the audio clock; the server, the conversation layer and
the local voice run as in the daemon, with silent output (bench_root and driver: the bench's own folders, the
person's voice settings copied by name, the speech models read where they are). The time is taken on the audio clock
from the last word of the recording to the moment Heliox's first audio of the turn would start playing.

A turn whose first sound came after the talk model began streaming is a "model answer"; any other (an
acknowledgement, a code answer, a quick answer) is a "local answer".

Every turn prints its stages, each from the last word: the candidate end of turn (where the listener starts its
speculative final pass), the final pass started, the final words ready, handed to the server's turn (dispatch), the
first sentence handed to the voice, and the first sound. The audio clock (seconds of recording fed) is printed for
every stamp; the wall clock is printed beside it, and the stages that are computation (the final pass, the turn,
the first sentence, synthesis) carry the CPU load of that turn. "pass deferred" means the listener did not start the
pass at the candidate end (a recogniser was still busy) and ran it at the end of the turn.

``--model stub`` (the default) answers talk with the bench's stub model (fixed delays): it times everything but the
model. ``--model configured`` uses the person's own provider and model, which costs money: the run prints its bound
(calls and dollars at most) and stops unless ``--paid`` is given too. Its spend is recorded in the person's budget, the
one write the run makes outside its root.

The load decides what a number means. The run checks the CPU before it starts (waiting up to ``--wait-quiet``
minutes) and refuses on a busy machine unless ``--allow-busy``; it samples the load before every turn and prints it
beside each number. A number taken at 50 % or more is marked "under load" and is not a measurement of Heliox.

Scenario names are not printed for the person's own recordings, only a short code.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any

from . import audio as audio_refs
from . import bench_root, machine, stack
from . import timeline as tl

MODEL_FIRST = "model first text"
CANDIDATE = "candidate end"
PASS_STARTED = "final pass started"
PASS_DEFERRED = "final pass deferred"
PASS_DONE = "final pass done"
STAGES = ("candidate end", "pass started", "final words", "turn ended", "handed to the turn", "first sentence",
          "first sound")  # fmt: skip
CONVERSATION_FIELDS = tuple(
    f"conversation.{name}"
    for name in (
        "enabled",
        "max_reply_tokens",
        "claude_model",
        "openai_model",
        "openrouter_model",
        "gemini_model",
        "meta_model",
        "ollama_model",
        "provisional_answers",
    )  # fmt: skip
)
PROMPT_TOKENS_AT_MOST = 8000  # the layer's system prompt, context and turns, rounded up
PAID_CALLS_AT_MOST = 40


def stages(events: list[tl.Event]) -> dict[str, Any] | None:
    """The stamps of the turn after the recording's last word, each as {"audio": s, "wall": s} after it."""
    last = tl.all_of(events, tl.LAST_WORD)
    if not last:
        return None
    end = last[-1]
    after = [e for e in events if e.t >= end.t]

    def first(kind: str, *, where: Any = None, since: float | None = None) -> tl.Event | None:
        return next((e for e in after if e.event == kind and (since is None or e.t >= since)
                     and (where is None or where(e))), None)  # fmt: skip

    candidate = first(CANDIDATE)
    started = first(PASS_STARTED, since=candidate.t if candidate else None)
    done = first(PASS_DONE, where=lambda e: not e.data.get("cancelled"), since=started.t if started else None)
    picked = {
        "candidate end": candidate,
        "pass started": started,
        "final words": done,
        "turn ended": first("vad end"),
        "handed to the turn": first(tl.DISPATCH),
        "first sentence": first(tl.SPEAKS, where=lambda e: not e.data.get("setup")),
        "first sound": first(tl.AUDIO_STARTS, where=lambda e: not e.data.get("setup")),
    }
    base_audio = end.data.get("audio_t", end.t)
    out: dict[str, Any] = {
        name: None
        if e is None
        else {"audio": round(e.data.get("audio_t", e.t) - base_audio, 3), "wall": round(e.t - end.t, 3)}  # fmt: skip
        for name, e in picked.items()
    }
    out["deferred"] = started is not None and not started.data.get("speculative")
    sound = picked["first sound"]
    out["route"] = (
        "model answer" if sound is not None and any(e.event == MODEL_FIRST and e.t <= sound.t for e in after)
        else "local answer"
    )  # fmt: skip
    return out


def stamp_hooks(harness: Any) -> None:
    """Log the candidate end, the final pass's start (at the candidate or deferred) and its end on the timeline."""
    import pilot.system.voice as voice

    log, listener = harness.log, harness.listener
    speculation = voice._Speculation
    if not getattr(speculation, "_bench_stamped", False):
        original_start = speculation.start

        def start(self: Any, frames: list[Any]) -> None:
            log(CANDIDATE, "")
            original_start(self, frames)
            if self.task is None:
                log(PASS_DEFERRED, "")

        speculation.start = start
        speculation._bench_stamped = True
    transcribe = listener._transcribe_path

    async def stamped(path: str, *, speculative: bool = False) -> str:
        log(PASS_STARTED, "", speculative=speculative)
        try:
            text = await transcribe(path, speculative=speculative)
        except asyncio.CancelledError:
            log(PASS_DONE, "", speculative=speculative, cancelled=True)
            raise
        log(PASS_DONE, "", speculative=speculative)
        return text

    listener._transcribe_path = stamped


def stage_line(found: dict[str, Any], cpu: float | None, threshold: float) -> str:
    loaded = machine.is_busy(cpu, threshold)
    parts = []
    for name in STAGES:
        stamp = found.get(name)
        if stamp is None:
            parts.append(f"{name} -")
            continue
        compute = name in ("final words", "handed to the turn", "first sentence", "first sound")
        mark = f", CPU {cpu} %{' under load' if loaded else ''}" if compute else ""
        parts.append(f"{name} {stamp['audio']:.2f} (wall {stamp['wall']:.2f}{mark})")
    return "; ".join(parts) + ("; pass deferred" if found.get("deferred") else "")


def paid_bound(config: Any, calls: int) -> dict[str, Any]:
    """At most ``calls`` talk calls, each at most PROMPT_TOKENS_AT_MOST in and max_reply_tokens out."""
    out_tokens = int(getattr(config.conversation, "max_reply_tokens", 400) or 400)
    bound: dict[str, Any] = {"calls": calls, "input_tokens": calls * PROMPT_TOKENS_AT_MOST,
                             "output_tokens": calls * out_tokens, "usd": None}  # fmt: skip
    if getattr(config.model, "cloud_provider", "") == "claude":
        from pilot.conversation.fast_model import fast_model_name
        from pilot.models.budget_tracker import claude_price_for

        price, _known = claude_price_for(fast_model_name(config, "claude"))
        bound["usd"] = round((bound["input_tokens"] * price.input + bound["output_tokens"] * price.output) / 1e6, 4)
    return bound


def summary(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for route in ("local answer", "model answer"):
        mine = [r for r in rows if r.get("route") == route]
        timed = [r for r in mine if r.get("seconds") is not None]
        quiet = [r["seconds"] for r in timed if not machine.is_busy(r["cpu_percent"], threshold)]
        loaded = [r["seconds"] for r in timed if machine.is_busy(r["cpu_percent"], threshold)]
        out[route] = {
            "n": len(mine),
            "quiet_n": len(quiet),
            "quiet_median_s": round(statistics.median(quiet), 3) if quiet else None,
            "under_load_n": len(loaded),
            "under_load_median_s": round(statistics.median(loaded), 3) if loaded else None,
        }
    return out


def code_for(scenario_id: str, voice: str) -> str:
    return scenario_id if voice != "real" else "rec-" + hashlib.sha256(scenario_id.encode()).hexdigest()[:8]


async def _main(args: argparse.Namespace) -> int:
    from .driver import Harness
    from .models import KINDS
    from .scenarios import load_all

    quiet, load = machine.wait_for_quiet(args.wait_quiet * 60, threshold=args.threshold)
    print(f"CPU before the run: {load} % ({'quiet' if quiet else 'busy'})")
    if not quiet and not args.allow_busy:
        print("Refused: the machine is busy, and a time taken now measures the other programs. --allow-busy runs it "
              "anyway with every number marked under load.")  # fmt: skip
        return 2

    def configure(config: Any) -> None:
        from pilot.real_folders import apply_owner_settings

        print("conversation settings copied:", ", ".join(apply_owner_settings(
            config, ("conversation",), only=CONVERSATION_FIELDS)) or "none")  # fmt: skip

    harness = Harness(audio_root=args.audio, data_dir=Path(args.root), model=KINDS["cloud"], configure=configure,
                      owner_settings=True)  # fmt: skip
    scenarios = [s for s in load_all() if not s.pending and any(s.id.startswith(p) for p in args.only.split(","))]
    if args.model == "configured":
        scenarios = scenarios[: args.max_calls]  # one talk call per turn at most
        # A paid timing taken under load is money for a number nobody may quote: the stub run before it must have
        # been quiet on every turn.
        stub = json.loads(Path(args.after_stub).read_text(encoding="utf-8")) if args.after_stub else None
        loaded = (
            None if stub is None else [r for r in stub["rows"] if machine.is_busy(r["cpu_percent"], args.threshold)]
        )
        if stub is None or stub.get("model") != "stub" or loaded:
            print("Refused: a paid run needs --after-stub <the stub run's --out file>, with every turn quiet "
                  f"({'no stub run named' if stub is None else f'{len(loaded or [])} turns under load'}).")  # fmt: skip
            return 4
    await harness.build()
    print("voice settings copied:", ", ".join(harness.copied_settings) or "none")
    print("models:", ", ".join(f"{k} ({v})" for k, v in harness.models_found.items()))
    server = harness.server
    if args.model == "configured":
        bound = paid_bound(server.config, min(len(scenarios), args.max_calls))
        print(f"Paid run bound (before any call): {bound}")
        if not args.paid:
            await harness.close()
            print("Not run: add --paid to spend that.")
            return 3
        model = await _configured_model(server.config)
        if model is None:
            await harness.close()
            raise SystemExit("no talk model is set up in these settings")
        server._model_router = stack._BenchRouter()
        from pilot.conversation.adapters import layer_for

        layer_for(server)._model = lambda: model
    else:
        stack.enable(harness)
    _log_first_text(harness, server)
    stamp_hooks(harness)
    rows = []
    try:
        for scenario in scenarios:
            cpu = machine.cpu_percent(1.0)
            events = await harness.run(scenario, "as-recorded")
            during = next((e.data.get("cpu_percent") for e in events if e.event == tl.SCENARIO), None)
            worst = max([v for v in (cpu, during) if v is not None], default=None)
            found = stages(events)
            sound = (found or {}).get("first sound")
            row = {"case": code_for(scenario.id, scenario.voice), "cpu_percent": worst,
                   "seconds": None if sound is None else sound["audio"], **(found or {})}  # fmt: skip
            rows.append(row)
            mark = "" if not machine.is_busy(worst, args.threshold) else " (under load)"
            shown = "no sound" if sound is None else f"{row['route']}: {sound['audio']:.2f} s"
            print(f"{row['case']}: {shown}, CPU {worst} %{mark}", flush=True)
            if found is not None:
                print("   " + stage_line(found, worst, args.threshold), flush=True)
    finally:
        await harness.close()  # no session log: it would hold the words of the person's recordings
    result = {"model": args.model, "threshold": args.threshold, "rows": rows,
              "summary": summary(rows, args.threshold)}  # fmt: skip
    print(json.dumps(result["summary"], indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


async def _configured_model(config: Any) -> Any:
    """The person's talk model; its spend goes to their budget (pilot.real_folders.real_budget_tracker)."""
    from pilot.conversation.fast_model import build_fast_model
    from pilot.real_folders import real_budget_tracker
    from pilot.security.vault import KeyVault

    budget = real_budget_tracker(config.model)
    await budget.initialize()
    return build_fast_model(config, vault=KeyVault(config), router=None, budget=budget)


def _log_first_text(harness: Any, server: Any) -> None:
    """A MODEL_FIRST event on the timeline when the talk model's first text arrives."""
    from pilot.conversation.adapters import layer_for

    layer = layer_for(server)
    pick = layer._model

    def model() -> Any:
        chosen = pick()
        if chosen is None or getattr(chosen, "_bench_logged", False):
            return chosen
        original = chosen.stream

        async def stream(system: str, messages: list[dict[str, str]]) -> Any:
            first = True
            async for piece in original(system, messages):
                if first:
                    first = False
                    harness.log(MODEL_FIRST, "")
                yield piece

        chosen.stream = stream
        chosen._bench_logged = True
        return chosen

    layer._model = model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.conversation_bench.voice_latency")
    parser.add_argument("--root", default=None, help="the bench root (default HELIOX_BENCH_ROOT)")
    parser.add_argument("--audio", default=None)
    parser.add_argument("--only", default="u-real", help="comma-separated scenario id prefixes")
    parser.add_argument("--model", choices=("stub", "configured"), default="stub")
    parser.add_argument("--paid", action="store_true", help="with --model configured: spend the printed bound")
    parser.add_argument("--max-calls", type=int, default=PAID_CALLS_AT_MOST)
    parser.add_argument("--wait-quiet", type=float, default=0.0, help="minutes to wait for a quiet machine")
    parser.add_argument("--allow-busy", action="store_true", help="run on a busy machine, every number under load")
    parser.add_argument("--threshold", type=float, default=machine.BUSY_PERCENT)
    parser.add_argument("--out", default=None)
    parser.add_argument("--after-stub", default=None, help="with --model configured: the stub run's --out file")
    args = parser.parse_args(argv)
    args.root = args.root or os.environ.get(bench_root.ROOT_VARIABLE, "").strip()
    if not args.root:
        raise SystemExit(f"name the bench root with --root or {bench_root.ROOT_VARIABLE}")
    args.audio = audio_refs.audio_root(args.audio)
    return asyncio.run(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
