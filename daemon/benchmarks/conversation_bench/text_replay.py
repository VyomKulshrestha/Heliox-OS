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


# A provider that answers this way is stopped after its first such answer (no key, no credit, no such model); the
# other arms carry on. The reason is the fast model's own classification, in plain words, never the key or account.
STOP_REASONS = frozenset({"no_key", "credit", "unknown_model", "budget"})
PROMPT_TOKENS_FOR_BOUND = 8000  # the layer's system prompt, context and turns, rounded up


def call_bound(provider: str, model: str, max_out: int) -> float:
    """The most one talk call can cost, at the product's own prices (pilot.models.prices; Claude per model), with the
    prompt rounded up to PROMPT_TOKENS_FOR_BOUND."""
    if provider == "claude":
        from pilot.models.budget_tracker import claude_price_for

        price, _known = claude_price_for(model)
        return (PROMPT_TOKENS_FOR_BOUND * price.input + max_out * price.output) / 1e6
    from pilot.models import prices

    return prices.cost_usd(provider, model, PROMPT_TOKENS_FOR_BOUND, max_out)


def call_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """One call's cost from its reported tokens, at the same prices."""
    if provider == "claude":
        from pilot.models.budget_tracker import claude_price_for

        price, _known = claude_price_for(model)
        return (input_tokens * price.input + output_tokens * price.output) / 1e6
    from pilot.models import prices

    return prices.cost_usd(provider, model, input_tokens, output_tokens)


def stock_opener(sentence: str) -> bool:
    """The filler check: a first sentence that opens with a stock phrase ("Sure", "Great question", ...)."""
    from benchmarks.conversation_eval.checks import STOCK_OPENERS

    return bool(STOCK_OPENERS.match(sentence.strip()))


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
    content = next((data for name, at, data in events if name == "piece to the voice" and data.get("kind") == "content"
                    and at >= handed), None)  # fmt: skip
    if content is not None:
        row["first_sentence_words"] = content.get("words")
        row["first_sentence_stock_opener"] = bool(content.get("stock"))
    answer = next((at for name, at, data in events if name == "first audio (modelled)" and data.get("kind") == "content"
                   and at >= handed), None)  # fmt: skip
    row["content audio (modelled)"] = None if answer is None else round(answer - handed, 3)
    first = row.get("first audio (modelled)")
    row["estimate from the last word"] = None if first is None else round(first + handover_s, 3)
    return row


def free_gb() -> float | None:
    """Free memory in GB (psutil), or None when it cannot be read."""
    try:
        import psutil

        return psutil.virtual_memory().available / 1e9
    except Exception:
        return None


def plain(reason: str) -> str:
    """The fast model's failure class in plain words."""
    return {"no_key": "the key was refused (authentication)", "credit": "no credit left on the account",
            "unknown_model": "the provider does not know that model", "budget": "Heliox's own budget gate",
            "busy": "the provider was busy or rate-limited", "offline": "the provider could not be reached",
            }.get(reason, f"the call failed ({reason})")  # fmt: skip


def call_figures(model: Any, provider: str) -> dict[str, Any]:
    """The turn's talk call: tokens, reply length and cost at the product's prices (from the model's own records)."""
    calls = list(getattr(model, "calls", []) or [])
    if not calls:
        return {}
    call = calls[-1]
    if not hasattr(call, "input_tokens"):
        return {}  # the bench's stub model keeps its own records, with no tokens
    return {"input_tokens": call.input_tokens, "output_tokens": call.output_tokens, "reply_chars": call.chars,
            "cost_usd": round(call_cost(provider, model.name, call.input_tokens, call.output_tokens), 6)}  # fmt: skip


def turn_line(row: dict[str, Any]) -> str:
    """One turn: the measured stamps from the hand-over, and the provider's real first text beside the voice MODEL's
    first audio (two different kinds of figure: never to be added up into a claim about the voice)."""
    measured = ", ".join(f"{k} {row[k]:.2f}" for k in STAMPS if row.get(k) is not None and k != "model first text")
    first_text = row.get("model first text")
    modelled = row.get("first audio (modelled)")
    return (f"turn {row['turn']} [{row.get('arm')}] (CPU {row.get('cpu_percent')} %, {row.get('free_gb')} GB free), "
            "seconds after the hand-over: "
            f"{measured}; MODEL FIRST TEXT ({row.get('source', 'real provider')}) {'-' if first_text is None else f'{first_text:.2f}'}; "
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
    for key in ("output_tokens", "reply_chars", "first_sentence_words"):
        values = sorted(r[key] for r in rows if r.get(key) is not None)
        if values:
            out[key] = {"median": statistics.median(values), "min": values[0], "max": values[-1]}
    out["cost_usd"] = round(sum(r.get("cost_usd") or 0.0 for r in rows), 6)
    out["stock openers"] = sum(bool(r.get("first_sentence_stock_opener")) for r in rows)
    out["failures"] = sorted({plain(r["failed"]) for r in rows if r.get("failed")})
    out["warm at the request"] = {str(flag): sum(r.get("warm") is flag for r in rows) for flag in (True, False, None)}
    return out


async def _main(args: argparse.Namespace) -> int:
    from .voice_latency import MODEL_TURNS, CallCap, paid_bound

    root = bench_root.enter(Path(args.root))
    free = free_gb()
    if free is not None and free < args.start_free_gb:
        print(f"Refused: {free:.2f} GB free, under the {args.start_free_gb} GB a run starts with")
        return 2
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
    # The voice is modelled: no local voice worker may start. The server prewarms the configured local voice when a
    # person talks to it (a 1.5 GB Kokoro worker on 14 November, which the memory gate stopped), so the engine is
    # the system voice, which loads nothing; the speaker stand-in below speaks every piece.
    config.voice.tts_engine = "os_native"
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
        clock("piece to the voice", kind=kind, words=len(text.split()), stock=kind == "content" and stock_opener(text))
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

        async def guarded(system: str, messages: list[dict[str, str]]) -> Any:
            try:
                async for piece in stream(system, messages):
                    yield piece
            except Exception as error:  # the fast model's own class of failure, in plain words; never key material
                clock("model failed", reason=getattr(error, "reason", type(error).__name__))
                raise

        made.stream = guarded
        return made

    if args.model == "configured":
        # The bound, per arm, before any call: the product's own price for each arm's model, the prompt rounded up.
        total = 0.0
        for arm in arms:
            apply_arm(config, args.compare, arm)
            probe = fresh_model()
            provider = str(getattr(config.model, "cloud_provider", "") or getattr(config.model, "provider", ""))
            each = call_bound(provider, probe.name, int(config.conversation.max_reply_tokens or 400))
            total += each * args.turns
            print(f"bound [{arm[0]}]: {provider} {probe.name}, at most ${each:.4f} a call, {args.turns} calls "
                  f"${each * args.turns:.4f}", flush=True)  # fmt: skip
        print(f"bound for the run: ${total:.4f}; calls stop before any one that could pass --max-usd "
              f"${args.max_usd}", flush=True)  # fmt: skip
    trace.listener = clock
    rows = []
    try:
        stopped: dict[str, str] = {}
        spent = 0.0
        for n in range(args.turns * len(arms)):
            arm = arms[n % len(arms)]  # interleaved; each arm gets the same question in turn
            if arm[0] in stopped:
                continue
            free = free_gb()
            if free is not None and free < args.min_free_gb:
                print(f"Stopped: {free:.2f} GB free, under {args.min_free_gb} GB", flush=True)
                break
            text = f"Hey Heliox, {MODEL_TURNS[(n // len(arms)) % len(MODEL_TURNS)]}"
            apply_arm(config, args.compare, arm)
            model = fresh_model()
            provider = str(getattr(config.model, "cloud_provider", "") or getattr(config.model, "provider", ""))
            if args.model == "configured":
                worst = call_bound(provider, model.name, int(config.conversation.max_reply_tokens or 400))
                if spent + worst > args.max_usd:
                    print(f"turn {n + 1} [{arm[0]}]: not run: ${spent:.4f} spent and one more call could cost up to "
                          f"${worst:.4f}, past --max-usd ${args.max_usd}", flush=True)  # fmt: skip
                    continue
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
            row = {"turn": n + 1, "arm": arm[0], "cpu_percent": cpu, "free_gb": None if free is None else round(free, 2),
                   "provider": provider, "model": model.name,
                   "source": "real provider" if args.model == "configured" else "STUB model, fixed delay",
                   **turn_row(list(clock.events), handed, args.handover_s), **call_figures(model, provider)}  # fmt: skip
            spent += row.get("cost_usd") or 0.0
            failed = next((data.get("reason") for name, _at, data in clock.events if name == "model failed"), None)
            if failed:
                row["failed"] = failed
                if failed in STOP_REASONS:
                    stopped[arm[0]] = failed
                    print(f"[{arm[0]}] stopped after its first failure: {plain(failed)}", flush=True)
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
    parser.add_argument("--start-free-gb", type=float, default=1.0, help="refuse to start under this much free")
    parser.add_argument("--min-free-gb", type=float, default=1.0, help="stop when free memory falls under this")
    parser.add_argument("--max-usd", type=float, default=0.05, help="stop before a call that could pass this spend")
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
