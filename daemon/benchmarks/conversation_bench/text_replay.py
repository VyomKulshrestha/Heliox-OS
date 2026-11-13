"""From the words handed over to Heliox's first sound, with the real server, conversation layer and provider, and no
audio stack: the recognisers' words are replayed as text and the voice is a model.

    python -m benchmarks.conversation_bench.text_replay --root <bench root> [--turns 12] [--model stub|configured]
        [--paid] [--max-calls 12] [--warm-lead 1.0] [--handover-s 0.60] [--out <file>]

Each turn's words (the synthetic model-turn questions, said after the wake phrase) are given to the server's own
turn entry (``_voice_turn``), as the listener gives the words its final pass settled on. Everything from there is
the daemon's own: the settle, the code handlers, the conversation layer's route and context, the request to the
person's provider (``--model configured``, paid, bounded) or the bench's stub model, the first sentence and the
hand-over to the speech coordinator. The connection is warmed as the wake phrase warms it (``warm_for_speech``),
``--warm-lead`` seconds before the words are handed over.

The voice is a MODEL, not the local voice: the speaker stand-in waits VOICE_FLOOR_S plus VOICE_PER_WORD_S per word of
the piece before its "first audio", the synthesis times measured on a 4-core runner (CI run 43404795980: 0.41 to
0.82 s for a first piece), and then the piece's spoken length. Its first-audio times are labelled "modelled". What the
run measures for real is the part between the hand-over and the piece handed to the voice: the part the conversation
layer's changes move.

The time before the hand-over is not measured here either: ``--handover-s`` (default 0.60 s, the laptop's median from
the last word to the words handed over on 13 November) is added to give an estimate from the last word, labelled as
such. The layer's own stamps (pilot.conversation.trace) are printed per turn.

Runs under the bench root (bench_root.enter: every Heliox folder inside it; the person's [model] and conversation
settings copied by name); nothing loads a speech model. The paid run's spend goes to the person's budget through
real_budget_tracker's block, the one write outside the root.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from . import bench_root, machine, stack

VOICE_FLOOR_S = 0.45
VOICE_PER_WORD_S = 0.03
SPOKEN_PER_WORD_S = 0.33  # the modelled piece's playback length after its first audio
HOLD = ("One moment.", "Let me think.", "Give me a second.")
CONVERSATION_FIELDS = ("conversation.enabled", "conversation.max_reply_tokens", "conversation.claude_model",
                       "conversation.openai_model", "conversation.openrouter_model", "conversation.gemini_model",
                       "conversation.provisional_answers")  # fmt: skip
STAMPS = ("words received", "settled", "code handlers done", "route decided", "context ready", "warm-up done",
          "request sent", "model first text", "first sentence complete", "sentence to voice")  # fmt: skip


def arms_for(compare: str) -> list[tuple[str, str, str | None]]:
    """The arms to interleave: (label, kind, value). "" one arm; "warm" warm-up before the hand-over against none;
    "section.field=a,b" a setting of the person's config set per turn (a cache rule, a shorter prompt, ...) (pure)."""
    if not compare.strip():
        return [("as configured", "none", None)]
    if compare.strip() == "warm":
        return [("warm-up before the request", "warm", "on"), ("no warm-up", "warm", "off")]
    field, _, values = compare.partition("=")
    section, _, name = field.strip().partition(".")
    chosen = [v.strip() for v in values.split(",") if v.strip()]
    if not section or not name or len(chosen) < 2:
        raise ValueError(f"--compare takes 'warm' or 'section.field=a,b'; not {compare!r}")
    return [(f"{field.strip()}={v}", "setting", v) for v in chosen]


def apply_arm(config: Any, compare: str, arm: tuple[str, str, str | None]) -> None:
    """Set the arm's setting on ``config`` (converted to the type of the field it replaces)."""
    if arm[1] != "setting":
        return
    section, _, name = compare.partition("=")[0].strip().partition(".")
    target = getattr(config, section)
    current = getattr(target, name)
    value: Any = arm[2]
    if isinstance(current, bool):
        value = str(value).casefold() in ("1", "true", "on", "yes")
    elif isinstance(current, int):
        value = int(str(value))
    elif isinstance(current, float):
        value = float(str(value))
    setattr(target, name, value)


def modelled_delay(text: str) -> float:
    """The voice model's time from a piece handed over to its first audio."""
    return round(VOICE_FLOOR_S + VOICE_PER_WORD_S * len(text.split()), 3)


class Clock:
    """Stamps on one clock (time.monotonic(), as the trace's ``at``), kept per turn."""

    def __init__(self) -> None:
        self.events: list[tuple[str, float, dict[str, Any]]] = []

    def __call__(self, name: str, at: float | None = None, **data: Any) -> None:
        self.events.append((name, time.monotonic() if at is None else at, data))


def turn_row(events: list[tuple[str, float, dict[str, Any]]], handed: float, handover_s: float) -> dict[str, Any]:
    """Each stamp's seconds after the hand-over, the first piece's kind, and the modelled first audio (pure)."""
    row: dict[str, Any] = {}
    for name, at, data in events:
        if at < handed or name in row:
            continue
        row[name] = round(at - handed, 3)
        if name == "request sent":
            row["warm"] = data.get("warm")
        if name == "piece to the voice":
            row["first_piece"] = data.get("kind")
    answer = next((at for name, at, data in events if name == "first audio (modelled)" and data.get("kind") == "content"
                   and at >= handed), None)  # fmt: skip
    row["content audio (modelled)"] = None if answer is None else round(answer - handed, 3)
    first = row.get("first audio (modelled)")
    row["estimate from the last word"] = None if first is None else round(first + handover_s, 3)
    return row


def turn_line(row: dict[str, Any]) -> str:
    """One turn: the measured stamps from the hand-over, and the provider's real first text beside the voice MODEL's
    first audio (two different kinds of figure: never to be added up into a claim about the voice)."""
    measured = ", ".join(f"{k} {row[k]:.2f}" for k in STAMPS if row.get(k) is not None and k != "model first text")
    first_text = row.get("model first text")
    modelled = row.get("first audio (modelled)")
    return (f"turn {row['turn']} [{row.get('arm')}] (CPU {row.get('cpu_percent')} %), seconds after the hand-over: "
            f"{measured}; MODEL FIRST TEXT (real provider) {'-' if first_text is None else f'{first_text:.2f}'}; "
            f"first piece {row.get('first_piece')}, its first audio MODELLED VOICE "
            f"{'-' if modelled is None else f'{modelled:.2f}'}; warm at the request {row.get('warm')}")  # fmt: skip


def summary_by_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {arm: summary([r for r in rows if r.get("arm") == arm]) for arm in dict.fromkeys(r.get("arm") for r in rows)}


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (*STAMPS, "piece to the voice", "first audio (modelled)", "content audio (modelled)",
            "estimate from the last word")  # fmt: skip
    out: dict[str, Any] = {}
    for key in keys:
        values = sorted(r[key] for r in rows if r.get(key) is not None)
        if values:
            out[key] = {"median": round(statistics.median(values), 3), "min": values[0], "max": values[-1],
                        "n": len(values)}  # fmt: skip
    out["first piece"] = {kind: sum(r.get("first_piece") == kind for r in rows) for kind in ("hold", "content")}
    out["warm at the request"] = {str(flag): sum(r.get("warm") is flag for r in rows) for flag in (True, False, None)}
    return out


async def _main(args: argparse.Namespace) -> int:
    from .voice_latency import MODEL_TURNS, CallCap, paid_bound

    root = bench_root.enter(Path(args.root))
    quiet, load = machine.wait_for_quiet(args.wait_quiet * 60, threshold=args.threshold)
    print(f"CPU before the run: {load} % ({'quiet' if quiet else 'busy'})")
    from pilot.config import PilotConfig
    from pilot.conversation import trace
    from pilot.conversation.adapters import layer_for
    from pilot.real_folders import apply_owner_settings
    from pilot.server import PilotServer

    config = PilotConfig.load()
    copied = apply_owner_settings(config, ("model", "conversation"), only=CONVERSATION_FIELDS)
    print("settings copied:", ", ".join(copied) or "none")
    config.live.enabled = False  # Live's own remarks would speak between turns
    bench_root.check_inside(root)
    server = PilotServer(config)

    async def broadcast(method: str, params: Any) -> None:
        return None

    async def planner(params: dict, ws: Any) -> dict:
        return {"status": "success", "message": "All set."}

    server._broadcast_notification = broadcast
    server._handle_execute = planner
    clock = Clock()

    async def voice(text: str, recorder: Any) -> bool:
        kind = "hold" if text.strip() in HOLD else "content"
        clock("piece to the voice", kind=kind, words=len(text.split()))
        await asyncio.sleep(modelled_delay(text))
        clock("first audio (modelled)", kind=kind)
        await asyncio.sleep(SPOKEN_PER_WORD_S * len(text.split()))
        return False

    server._speech_coordinator._speaker = voice
    server._model_router = stack._BenchRouter()
    exits = contextlib.AsyncExitStack()
    cap = None
    arms = arms_for(args.compare)
    budget: Any = None
    if args.model == "configured":
        from pilot.real_folders import real_budget_tracker

        cap = CallCap(args.max_calls)
        print(f"Paid run bound (before any call): {paid_bound(config, args.max_calls)}")
        if not args.paid:
            print("Not run: add --paid to spend that.")
            return 3
        budget = await exits.enter_async_context(real_budget_tracker(config.model, "the text replay"))

    def fresh_model() -> Any:
        """A new model for each turn (a new connection), so no arm inherits another's."""
        if args.model == "configured":
            from pilot.conversation.fast_model import build_fast_model
            from pilot.security.vault import KeyVault

            made = build_fast_model(config, vault=KeyVault(config), router=None, budget=budget)
            if made is None:
                raise SystemExit("no talk model is set up in these settings")
        else:
            from .models import KINDS

            made = stack.StubFastModel(KINDS["cloud"])
        original_stream = made.stream

        async def stream(system: str, messages: list[dict[str, str]]) -> Any:
            if cap is not None:
                if cap.calls >= cap.limit:
                    cap.refused += 1
                    return
                cap.calls += 1
            async for piece in original_stream(system, messages):
                yield piece

        made.stream = stream
        return made

    trace.listener = clock
    rows = []
    try:
        for n in range(args.turns * len(arms)):
            arm = arms[n % len(arms)]  # interleaved; each arm gets the same question in turn
            text = f"Hey Heliox, {MODEL_TURNS[(n // len(arms)) % len(MODEL_TURNS)]}"
            apply_arm(config, args.compare, arm)
            model = fresh_model()
            layer_for(server)._model = lambda m=model: m
            cpu = machine.cpu_percent(0.5)
            clock.events.clear()
            if arm[1] != "warm" or arm[2] == "on":
                server.warm_for_speech("wake")  # as the wake phrase does, --warm-lead before the words are handed over
            await asyncio.sleep(args.warm_lead)
            handed = time.monotonic()
            trace.begin()
            turn = asyncio.ensure_future(server._voice_turn(text))
            deadline = handed + 30.0
            while time.monotonic() < deadline and not any(
                name == "first audio (modelled)" and data.get("kind") == "content" for name, _at, data in clock.events
            ):
                await asyncio.sleep(0.02)
            with contextlib.suppress(asyncio.TimeoutError, Exception):
                await asyncio.wait_for(turn, 30.0)
            await asyncio.sleep(1.0)
            row = {"turn": n + 1, "arm": arm[0], "cpu_percent": cpu,
                   **turn_row(list(clock.events), handed, args.handover_s)}  # fmt: skip
            rows.append(row)
            client = getattr(model, "_client", None)
            if client is not None:
                with contextlib.suppress(Exception):
                    await client.aclose()
            print(turn_line(row), flush=True)
    finally:
        trace.listener = None
        await exits.aclose()
    result = {"model": args.model, "voice": "MODELLED (not the local voice)", "handover_s_added": args.handover_s,
              "compare": args.compare or None, "rows": rows, "summary_by_arm": summary_by_arm(rows),
              "calls": None if cap is None else {"made": cap.calls, "refused": cap.refused}}  # fmt: skip
    print(json.dumps({k: result[k] for k in ("voice", "handover_s_added", "compare", "summary_by_arm", "calls")},
                     indent=1))  # fmt: skip
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.conversation_bench.text_replay")
    parser.add_argument("--root", default=None, help="the bench root (default HELIOX_BENCH_ROOT)")
    parser.add_argument("--turns", type=int, default=12, help="turns per arm")
    parser.add_argument("--compare", default="", help='"warm", or "section.field=a,b" set per turn, interleaved')
    parser.add_argument("--model", choices=("stub", "configured"), default="stub")
    parser.add_argument("--paid", action="store_true")
    parser.add_argument("--max-calls", type=int, default=24)
    parser.add_argument("--warm-lead", type=float, default=1.0, help="seconds from the wake warm-up to the hand-over")
    parser.add_argument("--handover-s", type=float, default=0.60, help="added for the estimate from the last word")
    parser.add_argument("--wait-quiet", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=machine.BUSY_PERCENT)
    parser.add_argument("--out", default=None)
    args, _ignored = parser.parse_known_args(argv)  # voice_latency's own arguments (--audio, --only) do not apply
    args.root = args.root or os.environ.get(bench_root.ROOT_VARIABLE, "").strip()
    if not args.root:
        raise SystemExit(f"name the bench root with --root or {bench_root.ROOT_VARIABLE}")
    from pilot.real_folders import use_scratch_folders

    use_scratch_folders(Path(args.root))  # before anything imports pilot.config
    from .voice_latency import run_and_leave

    return run_and_leave(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
