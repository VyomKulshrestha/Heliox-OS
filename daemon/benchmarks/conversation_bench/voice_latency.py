"""From the person's last word to Heliox's first sound, through the real listener, conversation layer and voice.

    python -m benchmarks.conversation_bench.voice_latency --root <bench root> [--audio <folder>]
        [--only u-real] [--model-turns 12] [--model stub|configured] [--paid] [--after-stub <file>]
        [--wait-quiet 10] [--allow-busy] [--out <file>]

Each scenario's recording is played into the real listener on the audio clock; the server, the conversation layer and
the local voice run as in the daemon, with silent output (bench_root and driver: the bench's own folders, the
person's voice settings copied by name, the speech models read where they are). The time is taken on the audio clock
from the last word of the recording to the moment Heliox's first audio of the turn would start playing.

A turn whose first sound came after the talk model began streaming is a "model answer"; any other (an
acknowledgement, a code answer, a quick answer) is a "local answer". The person's own recordings hold few questions
for the model, so ``--model-turns`` adds that many synthetic questions said to Heliox (rendered once with the local
voice into the bench root, labelled "synthetic"). Everything after the hand-over to the turn is the same for a
synthetic voice as for a person's; the stages before it (recognition, the end of the turn) are trusted only from
the person's own recordings.

Every turn prints its stages, each from the last word: the candidate end of turn (where the listener starts its
speculative final pass), the final pass started, the final words ready, the turn ended, handed to the server's turn
(dispatch), the conversation layer's own stamps when it exposes them (``layer.stamp``), the first sentence handed
to the voice, and the first sound. The audio clock (seconds of recording fed) is printed for every stamp, the wall
clock beside it, and the stages that are computation carry the turn's CPU load. "pass deferred": the listener did
not start the pass at the candidate end (a recogniser was still busy) and ran it at the end of the turn.

The model's connection is warmed the way the daemon warms it: when speech is known to be for Heliox, the listener
asks the server (``warm_for_speech``), which asks the layer to warm the fast model. Each model turn says whether a
warm-up had completed before the hand-over, and the first model turn after start-up is its own row ("first after
start-up"): that moment is part of a person's day too.

A turn that made no sound says why, as a code: expected (the scenario expects it ignored), no words in time, no wake
phrase in the wake pass, wake phrase heard but not taken as addressed. Each recogniser's first call is printed
against the median of its later calls.

``--model stub`` (the default) answers talk with the bench's stub model (fixed delays): it times everything but the
model. ``--model configured`` uses the person's own provider and model, which costs money: the run prints its bound
(calls and dollars at most), runs only with ``--paid`` and only after a stub run with every turn quiet
(``--after-stub``), and never makes more talk calls than ``--max-calls``. Its spend is recorded in the person's
budget, the one write the run makes outside its root.

The load decides what a number means. The run checks the CPU before it starts (waiting up to ``--wait-quiet``
minutes) and refuses on a busy machine unless ``--allow-busy``; it samples the load before every turn and prints it
beside each number. A number taken at 50 % or more is marked "under load" and is not a measurement of Heliox.

Scenario names are not printed for the person's own recordings, only a short code, and no words of theirs.

Modes (``--mode``), so each measurement others depend on is one command and its table lines up run to run:

- ``full`` (the default): the person's recordings and the synthetic model turns, every table below.
- ``first-utterance``: the first of the chosen recordings straight after start-up ("cold"), then every recogniser is
  given one call on a short clip by the bench, then the same recording again ("warmed"). Each recogniser's first
  call against its later ones is printed with it.
- ``warm-up``: the synthetic model turns only, with the model connection's own stamps per turn: when its warm-up
  started and ended against "request sent", and how many requests went out on a finished, a running or no warm-up.
- ``synthesis``: the same turns as ``full``, reported as the time from the first sentence handed to the voice to its
  first sound, by the first sentence's length in words.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

from . import audio as audio_refs
from . import bench_root, machine, stack, wavio
from . import timeline as tl

MODEL_FIRST = "model first text"
MODEL_WARM = "model warm-up done"
MODEL_WARM_STARTED = "model warm-up started"
CANDIDATE = "candidate end"
PASS_STARTED = "final pass started"
PASS_DEFERRED = "final pass deferred"
PASS_DONE = "final pass done"
RECOGNISER = "recogniser call"
TRACE = "trace: "  # the conversation layer's own stamps, when it exposes them
STAGES = ("candidate end", "pass started", "final words", "turn ended", "handed to the turn", "holding phrase",
          "first sentence", "first sound", "answer sound")  # fmt: skip
COMPUTE = ("final words", "handed to the turn", "holding phrase", "first sentence", "first sound", "answer sound")
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
PAID_CALLS_AT_MOST = 12
# Questions only a talk model answers (no tool, no data of the person's), said after the wake phrase. Rendered with
# the local voice in three voices; the conversation router sends each to the talk model (tested).
MODEL_TURNS = (
    "How does undo work?",
    "Why is the sky blue?",
    "What's a good name for a grey cat?",
    "How long should I boil an egg?",
    "What's the capital of Australia?",
    "Give me one tip for staying focused.",
    "How many days are there in a leap year?",
    "What does a VPN actually do?",
    "Suggest a quick lunch idea.",
    "How do you say thank you in Spanish?",
    "What's the difference between weather and climate?",
    "Tell me something interesting about octopuses.",
)
MODEL_TURN_VOICES = (("af_bella", 1.0), ("am_michael", 1.0), ("bm_lewis", 1.0))
MODEL_TURN_FOLDER = "model_turns"
MODES = ("full", "first-utterance", "warm-up", "synthesis")
LENGTHS = ((1, 4), (5, 8), (9, 12), (13, 10_000))  # words in the first sentence


# Stamps --------------------------------------------------------------------------------------------------------


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
        # The layer's holding phrase ("One moment.") after 2 s without a first sentence is its own stamp: the first
        # sentence is the answer's, and "first sound" is whatever the person hears first (the holding phrase too).
        "holding phrase": first(tl.SPEAKS, where=lambda e: not e.data.get("setup") and _holding(e.detail)),
        "first sentence": first(tl.SPEAKS, where=lambda e: not e.data.get("setup") and not _holding(e.detail)),
        "first sound": first(tl.AUDIO_STARTS, where=lambda e: not e.data.get("setup")),
    }
    answer = picked["first sentence"]
    picked["answer sound"] = (
        first(tl.AUDIO_STARTS, where=lambda e: not e.data.get("setup"), since=answer.t) if answer is not None else None
    )
    base_audio = end.data.get("audio_t", end.t)

    def stamp(e: tl.Event) -> dict[str, float]:
        return {"audio": round(e.data.get("audio_t", e.t) - base_audio, 3), "wall": round(e.t - end.t, 3)}

    out: dict[str, Any] = {name: None if e is None else stamp(e) for name, e in picked.items()}
    out["deferred"] = started is not None and not started.data.get("speculative")
    sound = picked["first sound"]
    out["route"] = (
        "model answer" if sound is not None and any(e.event == MODEL_FIRST and e.t <= sound.t for e in after)
        else "local answer"
    )  # fmt: skip
    limit = sound.t if sound is not None else float("inf")
    out["trace"] = [{"name": e.event[len(TRACE):], **stamp(e)} for e in after
                    if e.event.startswith(TRACE) and e.t <= limit]  # fmt: skip
    handed = picked["handed to the turn"]
    sent = next((e for e in after if e.event == TRACE + "request sent"), None)
    out["warm_before_handover"] = warm_state(events, handed) == "finished"
    out["warm_at_request"] = warm_state(events, sent or handed)
    reference = sent or handed
    if reference is not None:
        before = [e for e in events if e.t <= reference.t]
        started_warm = next((e for e in reversed(before) if e.event == MODEL_WARM_STARTED), None)
        ended = next((e for e in events if e.event == MODEL_WARM and started_warm is not None
                      and e.t >= started_warm.t), None)  # fmt: skip
        out["warm_started"] = None if started_warm is None else stamp(started_warm)
        out["warm_done"] = None if ended is None else stamp(ended)
        out["request"] = stamp(reference)
    answer = picked["first sentence"]
    if answer is not None:
        out["first_sentence_words"] = len(answer.detail.split())
        out["first_sentence_chars"] = len(answer.detail.strip())
        head = answer.detail.strip().rstrip(".!?")
        # Words up to the first comma that comes before the sentence's end (numbers only, never the words).
        out["first_comma_words"] = len(head.split(",", 1)[0].split()) if "," in head else None
        heard = out.get("answer sound")
        out["synthesis_s"] = None if heard is None else round(heard["audio"] - out["first sentence"]["audio"], 3)
    return out


def length_band(words: int) -> str:
    return next(f"{low}-{high}" if high < 10_000 else f"{low}+" for low, high in LENGTHS if low <= words <= high)


def synthesis_table(rows: list[dict[str, Any]], threshold: float) -> dict[str, dict[str, Any]]:
    """First sentence handed to the voice to its first sound, by the sentence's length in words (quiet turns)."""
    out: dict[str, dict[str, Any]] = {}
    for low, high in LENGTHS:
        band = length_band(low)
        values = [r["synthesis_s"] for r in rows if r.get("synthesis_s") is not None
                  and low <= r.get("first_sentence_words", 0) <= high
                  and not machine.is_busy(r.get("cpu_percent"), threshold)]  # fmt: skip
        out[band] = {"n": len(values), "median_s": _median(values),
                     "min_s": min(values) if values else None, "max_s": max(values) if values else None}  # fmt: skip
    return out


def warm_up_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The model turns' warm-up at the request: how many finished, running, none; and its stamps (median)."""
    turns = [r for r in rows if r.get("route") == "model answer"]
    counts = {state: sum(r.get("warm_at_request") == state for r in turns) for state in ("finished", "in progress",
                                                                                           "none")}  # fmt: skip

    def median_of(key: str) -> float | None:
        return _median([r[key]["audio"] for r in turns if r.get(key)])

    return {"turns": len(turns), **counts, "warm-up started": median_of("warm_started"),
            "warm-up done": median_of("warm_done"), "request sent": median_of("request")}  # fmt: skip


def warm_state(events: list[tl.Event], at: tl.Event | None) -> str:
    """The model connection's warm-up at a moment: "finished" (a warm-up that opened a connection had ended),
    "in progress" (one had started and not ended), or "none"."""
    if at is None:
        return "none"
    started = [e.t for e in events if e.event == MODEL_WARM_STARTED and e.t <= at.t]
    finished = [e.t for e in events if e.event == MODEL_WARM and e.data.get("opened") and e.t <= at.t]
    if started and (not finished or max(started) > max(finished)):
        return "in progress"
    return "finished" if finished else "none"


def _holding(text: str) -> bool:
    from pilot.conversation.layer import HOLD_PHRASES

    return text.strip() in HOLD_PHRASES


def stage_line(found: dict[str, Any], cpu: float | None, threshold: float) -> str:
    loaded = machine.is_busy(cpu, threshold)
    parts = []
    for name in STAGES:
        stamp = found.get(name)
        if stamp is None:
            parts.append(f"{name} -")
            continue
        mark = f", CPU {cpu} %{' under load' if loaded else ''}" if name in COMPUTE else ""
        parts.append(f"{name} {stamp['audio']:.2f} (wall {stamp['wall']:.2f}{mark})")
        if name == "handed to the turn":
            parts += [f"{t['name']} {t['audio']:.2f} (wall {t['wall']:.2f})" for t in found.get("trace") or []]
    return "; ".join(parts) + ("; pass deferred" if found.get("deferred") else "")


def drop_reason(events: list[tl.Event], expected_silent: bool) -> str | None:
    """Why a turn made no sound, as a code (never the words): None when it did."""
    from pilot.system.voice import loose_wake_prefix

    if any(e.event == tl.AUDIO_STARTS and not e.data.get("setup") for e in events):
        return None
    if expected_silent:
        return "expected (scenario expects it ignored)"
    words = tl.all_of(events, tl.TRANSCRIPT)
    if not words:
        return "no words in the waiting time"
    if any(e.event == tl.DISPATCH for e in events):
        return "handed to the turn, no sound"
    wake_text = str(words[-1].data.get("wake_text") or "")
    heard = wake_text or words[-1].detail
    if not loose_wake_prefix(heard):
        return "no wake phrase in the wake pass"
    return "wake phrase heard, not taken as addressed"


def recogniser_summary(events: list[tl.Event]) -> dict[str, dict[str, Any]]:
    """Each recogniser's first call against the median of its later calls (wall seconds)."""
    calls: dict[str, list[float]] = {}
    for e in events:
        if e.event == RECOGNISER:
            calls.setdefault(e.detail, []).append(float(e.data.get("seconds", 0.0)))
    return {
        name: {
            "calls": len(times),
            "first_s": round(times[0], 3),
            "later_median_s": round(statistics.median(times[1:]), 3) if len(times) > 1 else None,
        }  # fmt: skip
        for name, times in calls.items()
    }


def synthesis_summary(events: list[tl.Event]) -> dict[str, Any]:
    """The local voice: first sentence handed over to its first sound, the first of the run against the later ones."""
    spans, pending = [], None
    for e in events:
        if e.event == tl.SPEAKS and not e.data.get("setup"):
            pending = e
        elif e.event == tl.AUDIO_STARTS and pending is not None and not e.data.get("setup"):
            spans.append(e.t - pending.t)
            pending = None
    return {"calls": len(spans), "first_s": round(spans[0], 3) if spans else None,
            "later_median_s": round(statistics.median(spans[1:]), 3) if len(spans) > 1 else None}  # fmt: skip


def stamp_hooks(harness: Any) -> None:
    """Log the candidate end, the final pass's start (at the candidate or deferred) and end, each recogniser call,
    the layer's own stamps, and the wake check's reading of every utterance (for the drop codes) on the timeline."""
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

    whisper = listener._transcribe_whisper_text

    async def whisper_timed(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return await whisper(*args, **kwargs)
        finally:
            log(RECOGNISER, "whisper", seconds=round(time.perf_counter() - started, 3))

    listener._transcribe_whisper_text = whisper_timed
    _time_method(log, listener._parakeet(), "transcribe", "parakeet")
    # The listener's own Smart Turn object: the recorder is handed it again for every utterance.
    _time_method(log, getattr(listener, "_turn_detector", None), "probability", "smart turn")
    _time_method(log, getattr(listener._streaming, "_recognizer", None), "decode_stream", "streaming decode")

    try:
        from pilot.conversation import trace
    except ImportError:  # a tree without the layer's trace: the bench's own stamps only
        return
    trace.listener = lambda name, at=None, **data: log(TRACE + str(name), "", **data)


def _time_method(log: Any, target: Any, name: str, label: str) -> None:
    original = getattr(target, name, None) if target is not None else None
    if original is None or getattr(original, "_bench_timed", False):
        return

    def timed(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            log(RECOGNISER, label, seconds=round(time.perf_counter() - started, 4))

    timed._bench_timed = True  # type: ignore[attr-defined]
    try:
        setattr(target, name, timed)
    except (AttributeError, TypeError):  # a compiled object that does not take attributes: not timed
        return


async def warm_recognisers(harness: Any, clip: Path) -> list[str]:
    """One call on each recogniser on ``clip``, as a start-up warm-up would make; returns which were called."""
    import numpy as np

    listener = harness.listener
    samples = wavio.read(clip).astype(np.float32) / 32768.0
    called = []
    whisper = getattr(listener, "_transcribe_whisper_text", None)
    if whisper is not None:
        await whisper(str(clip), language="en")
        called.append("whisper")
    parakeet = listener._parakeet()
    if parakeet is not None:
        await asyncio.to_thread(parakeet.transcribe, samples)
        called.append("parakeet")
    detector = getattr(listener, "_turn_detector", None)
    if detector is not None and getattr(detector, "ready", False):
        await asyncio.to_thread(detector.probability, samples)
        called.append("smart turn")
    streaming = getattr(listener, "_streaming", None)
    recognizer = getattr(streaming, "_recognizer", None)
    if recognizer is not None:
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples)
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        called.append("streaming decode")
    return called


# The talk model -------------------------------------------------------------------------------------------------


class CallCap:
    """At most ``limit`` talk calls; a turn past it gets no model text (and is marked), never a paid call."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.calls = 0
        self.refused = 0


def wrap_model(harness: Any, server: Any, cap: CallCap | None) -> None:
    """Log the model's first text and every warm-up that opened a connection; enforce the call cap."""
    from pilot.conversation.adapters import layer_for

    layer = layer_for(server)
    pick = layer._model

    def model() -> Any:
        chosen = pick()
        if chosen is None or getattr(chosen, "_bench_logged", False):
            return chosen
        original_stream, original_warm = chosen.stream, getattr(chosen, "warm", None)

        async def stream(system: str, messages: list[dict[str, str]]) -> Any:
            if cap is not None and cap.calls >= cap.limit:
                cap.refused += 1
                harness.log("model call refused (over the bound)", "")
                return
            if cap is not None:
                cap.calls += 1
            first = True
            async for piece in original_stream(system, messages):
                if first:
                    first = False
                    harness.log(MODEL_FIRST, "")
                yield piece

        async def warm() -> None:
            before = getattr(chosen, "_warmed_at", None)
            started = time.perf_counter()
            harness.log(MODEL_WARM_STARTED, "")
            if original_warm is not None:
                await original_warm()
            opened = getattr(chosen, "_warmed_at", None) != before or original_warm is None
            harness.log(MODEL_WARM, "", opened=bool(opened), seconds=round(time.perf_counter() - started, 3))

        chosen.stream = stream
        chosen.warm = warm
        chosen._bench_logged = True
        return chosen

    layer._model = model


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


async def _configured_model(config: Any, exits: contextlib.AsyncExitStack) -> Any:
    """The person's talk model; its spend goes to their budget (pilot.real_folders.real_budget_tracker), a block
    entered on ``exits`` and closed with it when the run ends."""
    from pilot.conversation.fast_model import build_fast_model
    from pilot.real_folders import real_budget_tracker
    from pilot.security.vault import KeyVault

    budget = await exits.enter_async_context(real_budget_tracker(config.model, "the voice latency bench"))
    return build_fast_model(config, vault=KeyVault(config), router=None, budget=budget)


# Synthetic model turns ------------------------------------------------------------------------------------------


def model_turn_scenarios(count: int) -> list[Any]:
    """``count`` synthetic questions for the talk model, as scenarios whose clips live in the bench root."""
    from .scenarios import Scenario, Step

    out = []
    for n in range(count):
        text = MODEL_TURNS[n % len(MODEL_TURNS)]
        voice, speed = MODEL_TURN_VOICES[n % len(MODEL_TURN_VOICES)]
        ref = f"{MODEL_TURN_FOLDER}/{voice}/{n:02d}.wav"
        out.append(Scenario(id=f"m-syn-{n:02d}", set="understanding", voice="synthetic", lines=(),
                            steps=(Step(say=ref, text=f"Hey Heliox, {text}"),), expect={}, models=("cloud",),
                            note=f"{voice} {speed}"))  # fmt: skip
    return out


def render_model_turns(root: Path, scenarios: list[Any]) -> int:
    """Render the missing clips with the local voice (CPU, offline); returns how many were made."""
    missing = [s for s in scenarios if not (root / s.steps[0].say).exists()]
    if not missing:
        return 0
    from .synth import Voice

    voice = Voice()
    for scenario in missing:
        name, speed = scenario.note.split(" ")
        audio = voice.say(scenario.steps[0].text, name, float(speed))
        path = root / scenario.steps[0].say
        path.parent.mkdir(parents=True, exist_ok=True)
        wavio.write(path, audio / max(1e-6, float(abs(audio).max())) * 0.5)  # the level synth.py gives every clip
    del voice
    return len(missing)


# Report ---------------------------------------------------------------------------------------------------------


def late_words(rows: list[dict[str, Any]], session: list[tl.Event]) -> None:
    """A turn that had no words in its waiting time but whose words came before the next scenario began: those
    words arrived after the waiting time (the session log keeps what falls between scenarios)."""
    starts = [e.t for e in session if e.event == tl.SCENARIO]
    for row in rows:
        closed = row.get("closed_at")
        if row.get("dropped") != "no words in the waiting time" or closed is None:
            continue
        following = min((t for t in starts if t > closed), default=float("inf"))
        if any(e.event == tl.TRANSCRIPT and closed < e.t < following for e in session):
            row["dropped"] = "words arrived after the waiting time"


def summary(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for route in ("local answer", "model answer"):
        for voice in ("own recordings", "synthetic"):
            mine = [r for r in rows if r.get("route") == route and r.get("voice", "own recordings") == voice
                    and r.get("seconds") is not None]  # fmt: skip
            if not mine:
                continue
            quiet = [r for r in mine if not machine.is_busy(r["cpu_percent"], threshold)]
            loaded = [r for r in mine if machine.is_busy(r["cpu_percent"], threshold)]
            out[f"{route}, {voice}"] = {
                "n": len(mine),
                "quiet_n": len(quiet),
                "quiet_median_s": _median([r["seconds"] for r in quiet]),
                "under_load_n": len(loaded),
                "under_load_median_s": _median([r["seconds"] for r in loaded]),
                "stages_quiet_median": {
                    name: _median([r[name]["audio"] for r in quiet if r.get(name)]) for name in STAGES
                },  # fmt: skip
            }
    return out


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def code_for(scenario_id: str, voice: str) -> str:
    return scenario_id if voice != "real" else "rec-" + hashlib.sha256(scenario_id.encode()).hexdigest()[:8]


def _expected_silent(scenario: Any) -> bool:
    route = scenario.expect.get("route", "any")
    routes = route if isinstance(route, list) else [route]
    return bool(routes) and all(r in ("ignored", "overheard") for r in routes)


def refusal_after_stub(path: str | None, threshold: float) -> str | None:
    """Why a paid run may not start after the named stub run, or None when it may."""
    if not path:
        return "no stub run named (--after-stub)"
    stub = json.loads(Path(path).read_text(encoding="utf-8"))
    if stub.get("model") != "stub":
        return "the file named is not a stub run"
    loaded = [r for r in stub.get("rows", []) if machine.is_busy(r.get("cpu_percent"), threshold)]
    return f"{len(loaded)} turns of the stub run were under load" if loaded else None


async def _main(args: argparse.Namespace) -> int:
    from .driver import Harness
    from .models import KINDS
    from .scenarios import load_all

    if args.model == "configured":
        refused = refusal_after_stub(args.after_stub, args.threshold)
        if refused:
            print(f"Refused: a paid timing taken under load is money for a number nobody may quote ({refused}).")
            return 4
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

    root = Path(args.root)
    harness = Harness(audio_root=args.audio, data_dir=root, model=KINDS["cloud"], configure=configure,
                      owner_settings=True)  # fmt: skip
    own = [s for s in load_all() if not s.pending and any(s.id.startswith(p) for p in args.only.split(",") if p)]
    synthetic = model_turn_scenarios(args.model_turns)
    if args.mode == "first-utterance":
        own, plan = own[:1], own[:1]  # the same recording, cold and then warmed
    elif args.mode == "warm-up":
        plan = synthetic
    else:
        plan = own + synthetic
    made = render_model_turns(root, synthetic)  # into the bench root, before the server and its voice load
    if made:
        print(f"rendered {made} synthetic model-turn clips with the local voice")
    await harness.build()
    exits = contextlib.AsyncExitStack()  # the paid model's budget block, closed in the finally below
    print("voice settings copied:", ", ".join(harness.copied_settings) or "none")
    print("models:", ", ".join(f"{k} ({v})" for k, v in harness.models_found.items()))
    server = harness.server
    cap = None
    if args.model == "configured":
        cap = CallCap(args.max_calls)
        bound = paid_bound(server.config, args.max_calls)
        print(f"Paid run bound (before any call): {bound}")
        if not args.paid:
            await harness.close()
            print("Not run: add --paid to spend that.")
            return 3
        model = await _configured_model(server.config, exits)
        if model is None:
            await exits.aclose()
            await harness.close()
            raise SystemExit("no talk model is set up in these settings")
        server._model_router = stack._BenchRouter()
        from pilot.conversation.adapters import layer_for

        layer_for(server)._model = lambda: model
    else:
        stack.enable(harness)
    wrap_model(harness, server, cap)
    stamp_hooks(harness)
    rows: list[dict[str, Any]] = []
    model_turns_seen = 0
    try:
        warmed = False
        for index, scenario in enumerate(plan + (own if args.mode == "first-utterance" else [])):
            if args.mode == "first-utterance" and index == len(plan) and not warmed:
                called = await warm_recognisers(harness, root / synthetic[0].steps[0].say)
                warmed = True
                print(f"warmed by the bench: {', '.join(called)}", flush=True)
            synthetic_turn = scenario in synthetic
            harness.audio_root = str(root) if synthetic_turn else args.audio
            cpu = machine.cpu_percent(1.0)
            events = await harness.run(scenario, "as-recorded")
            during = next((e.data.get("cpu_percent") for e in events if e.event == tl.SCENARIO), None)
            worst = max([v for v in (cpu, during) if v is not None], default=None)
            found = stages(events) or {}
            sound = found.get("first sound")
            row = {"case": code_for(scenario.id, scenario.voice), "cpu_percent": worst,
                   "voice": "synthetic" if synthetic_turn else "own recordings",
                   "seconds": None if sound is None else sound["audio"], **found,
                   "dropped": drop_reason(events, _expected_silent(scenario)),
                   "closed_at": events[-1].t if events else None}  # fmt: skip
            if args.mode == "first-utterance":
                row["state"] = "warmed" if warmed else "cold (first after start-up)"
            if row.get("route") == "model answer":
                model_turns_seen += 1
                row["first after start-up"] = model_turns_seen == 1
            rows.append(row)
            mark = "" if not machine.is_busy(worst, args.threshold) else " (under load)"
            label = " [synthetic]" if synthetic_turn else ""
            if sound is None:
                shown = f"no sound: {row['dropped']}"
            else:
                extra = ""
                if row["route"] == "model answer":
                    warm = row["warm_at_request"]
                    extra = f", connection warm-up at the request: {warm}" if args.model == "configured" else ""
                    extra += ", first model turn after start-up" if row["first after start-up"] else ""
                shown = f"{row['route']}: {sound['audio']:.2f} s{extra}"
            state = f" [{row['state']}]" if row.get("state") else ""
            print(f"{row['case']}{label}{state}: {shown}, CPU {worst} %{mark}", flush=True)
            if found:
                print("   " + stage_line(found, worst, args.threshold), flush=True)
        session = harness.log.since(0.0)
    finally:
        await exits.aclose()  # the person's budget store closed: the spend lands in pilot.db, its threads end
        await harness.close()  # no session log: it would hold the words of the person's recordings
    late_words(rows, session)
    for row in rows:
        if row.get("dropped"):
            print(f"no sound, why: {row['case']}: {row['dropped']}")
    recognisers = recogniser_summary(session)
    recognisers["kokoro (first sentence to first sound)"] = synthesis_summary(session)
    result = {"mode": args.mode, "model": args.model, "threshold": args.threshold, "rows": rows,
              "summary": summary(rows, args.threshold), "recognisers": recognisers,
              "warm_up": warm_up_table(rows), "synthesis_by_length": synthesis_table(rows, args.threshold),
              "calls": None if cap is None else {"made": cap.calls, "refused": cap.refused}}  # fmt: skip
    shown = {"full": ("summary", "recognisers", "warm_up", "synthesis_by_length", "calls"),
             "first-utterance": ("recognisers",), "warm-up": ("warm_up", "calls"),
             "synthesis": ("synthesis_by_length",)}[args.mode]  # fmt: skip
    print(json.dumps({k: result[k] for k in shown}, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


def run_and_leave(main: Any, grace_s: float = 20.0) -> int:
    """Run the measurement, then leave even if a thread is stuck: a worker thread that never returns made a run hang
    on exit after it had written its results (13 November). The threads still alive are named and their stacks
    printed first, so the cause is seen; every result file is already written and closed by then."""
    import faulthandler

    code = asyncio.run(main)
    sys.stdout.flush()
    deadline = time.monotonic() + grace_s
    others = [t for t in threading.enumerate() if t is not threading.main_thread() and not t.daemon]
    while any(t.is_alive() for t in others) and time.monotonic() < deadline:
        time.sleep(0.2)
    stuck = [t.name for t in others if t.is_alive()]
    if stuck:
        print(f"Threads still running after the run, left behind: {', '.join(stuck)}", file=sys.stderr)
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        sys.stderr.flush()
        os._exit(code)
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.conversation_bench.voice_latency")
    parser.add_argument("--root", default=None, help="the bench root (default HELIOX_BENCH_ROOT)")
    parser.add_argument("--audio", default=None)
    parser.add_argument("--only", default="u-real", help="comma-separated scenario id prefixes")
    parser.add_argument("--model-turns", type=int, default=12, help="synthetic questions for the talk model")
    parser.add_argument("--mode", choices=MODES, default="full", help="which measurement (see the module text)")
    parser.add_argument("--model", choices=("stub", "configured"), default="stub")
    parser.add_argument("--paid", action="store_true", help="with --model configured: spend the printed bound")
    parser.add_argument("--max-calls", type=int, default=PAID_CALLS_AT_MOST)
    parser.add_argument("--after-stub", default=None, help="with --model configured: the stub run's --out file")
    parser.add_argument("--wait-quiet", type=float, default=0.0, help="minutes to wait for a quiet machine")
    parser.add_argument("--allow-busy", action="store_true", help="run on a busy machine, every number under load")
    parser.add_argument("--threshold", type=float, default=machine.BUSY_PERCENT)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    args.root = args.root or os.environ.get(bench_root.ROOT_VARIABLE, "").strip()
    if not args.root:
        raise SystemExit(f"name the bench root with --root or {bench_root.ROOT_VARIABLE}")
    args.audio = audio_refs.audio_root(args.audio)
    return run_and_leave(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
