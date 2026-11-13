"""Drive the real listener and dispatch with recorded audio, the way capture/voice_tests/voice_probe.py does.

Real: the daemon's continuous listener (voice activity, Smart Turn, streaming recognition, the final pass, wake
phrase, follow-ups), the server's voice dispatch, Heliox Live, the speech coordinator and the local voice
(synthesis runs; playback is silent). Replaced: the microphone (a file pump at real-time pace, never the real
device), the planner (a stub with the chosen kind of model's delays, no cloud call), the data folder (a fresh,
empty one per run: a copy of a person's data would carry undo history and scheduled runs that act on real
files), and the speaker (Heliox's audio is consumed in real time without being played).

Heavy packages (numpy, the daemon, speech models) are imported only when a run starts.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import audio as audio_refs
from . import conditions as conds
from . import timeline as tl
from . import wavio
from .models import KINDS, NO_MODEL_MESSAGE, ModelKind
from .scenarios import Scenario

STUB_SUCCESS = "All set."
VOICE_LOAD_WAIT_S = 900.0
WORDS_WAIT_S = 20.0
OUTCOME_WAIT_S = 6.0
ROOM_NOISE = 40  # int16 standard deviation of the quiet room between utterances (about -58 dBFS)


class Log:
    """Events on the wall clock; with ``audio_clock`` set, each also carries ``audio_t``, the seconds of microphone
    audio fed so far, so a stage can be read on the audio's own clock as well (the two differ when the machine
    falls behind real time)."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.events: list[tl.Event] = []
        self.lock = threading.Lock()
        self.echo = bool(os.environ.get("CONVERSATION_BENCH_VERBOSE"))
        self.audio_clock: Callable[[], float] | None = None

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def __call__(self, kind: str, detail: str = "", /, **data: Any) -> None:
        if self.audio_clock is not None and "audio_t" not in data:
            data["audio_t"] = round(self.audio_clock(), 3)
        event = tl.Event(self.now(), kind, detail, data)
        with self.lock:
            self.events.append(event)
        if self.echo:
            print(f"{event.t:8.2f}  {kind:<20} {detail[:110]}", flush=True)

    def since(self, t: float) -> list[tl.Event]:
        with self.lock:
            return [e for e in self.events if e.t >= t]


def voiced_span(x: Any) -> tuple[float, float]:
    """First and last loud sample (above 15 percent of the peak), in seconds."""
    import numpy as np

    loud = np.flatnonzero(np.abs(x) > 0.15 * float(np.abs(x).max() or 1.0))
    if not len(loud):
        return 0.0, 0.0
    return float(loud[0]) / wavio.RATE, float(loud[-1] + 1) / wavio.RATE


class FilePump:
    """Stands in for the microphone: one block per frame period at ``pace`` times real time.

    Everything is placed on the pump's own sample clock: a clip starts at a sample index (``start_at``), the gap
    before it is filled with quiet room noise inside the audio stream itself, and the person's first word, last
    word, start and stop are marked at absolute sample positions. A busy or stalled event loop therefore cannot
    stretch a pause: a 1.5 s pause is 1.5 s of audio whenever the clip was queued, as long as it was queued before
    its start. (On 12 November a pause scheduled with asyncio.sleep on a loaded loop became 12.6 s of real gap.)
    """

    def __init__(self, recorder: Any, log: Log, pace: float = 1.0) -> None:
        import numpy as np

        self.np = np
        self.recorder = recorder
        self.log = log
        self.pace = pace
        self.pending = np.zeros(0, dtype=np.int16)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.marks: list[tuple[int, str, dict[str, Any]]] = []  # absolute sample index -> event
        self.segments: list[dict[str, Any]] = []
        self.played = 0  # samples handed to the listener so far: the audio clock
        self.clip_started_at = 0
        self.rng = np.random.default_rng(1)
        self.max_lag = 0.0

    def queued_end(self) -> int:
        """The sample index where everything queued so far ends."""
        with self.lock:
            return self.played + len(self.pending)

    def play(
        self,
        samples: Any,
        label: str,
        *,
        first: float,
        last: float,
        segments: Any = None,
        start_at: int | None = None,
    ) -> int:
        """Queue a clip to start at sample ``start_at`` (default: right after what is queued); returns its end."""
        np = self.np
        pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
        with self.lock:
            end_of_queue = self.played + len(self.pending)
            gap = max(0, start_at - end_of_queue) if start_at is not None else 0
            if gap:
                noise = (self.rng.standard_normal(gap) * ROOM_NOISE).astype(np.int16)
                self.pending = np.concatenate([self.pending, noise])
            start = end_of_queue + gap
            self.pending = np.concatenate([self.pending, pcm])
            end = start + len(pcm)
            data = {"clip": label}
            self.marks += [
                (start, tl.PERSON_STARTS, {**data, "seconds": round(len(pcm) / wavio.RATE, 2)}),
                (start + int(first * wavio.RATE), tl.FIRST_WORD, data),
                (start + int(last * wavio.RATE), tl.LAST_WORD, data),
                (end, tl.PERSON_STOPS, data),
            ]
            self.marks.sort(key=lambda m: m[0])
            self.clip_started_at = start
            self.segments = list(segments or [])
        return end

    def busy(self) -> bool:
        with self.lock:
            return len(self.pending) > 0

    def current_segment(self) -> dict[str, Any]:
        """The corpus line playing now or last (for blocks of not-addressed speech)."""
        position = (self.played - self.clip_started_at) / wavio.RATE
        current: dict[str, Any] = {}
        for seg in self.segments:
            if seg["start"] <= position + 0.5:
                current = seg
        return current

    def next_block(self) -> tuple[Any, list[tuple[int, str, dict[str, Any]]]]:
        """The next frame of audio and the marks it reaches; advances the audio clock by one frame."""
        np = self.np
        frame = self.recorder.frame_size
        with self.lock:
            if len(self.pending):
                block = self.pending[:frame]
                self.pending = self.pending[frame:]
                if len(block) < frame:
                    block = np.concatenate([block, np.zeros(frame - len(block), dtype=np.int16)])
            else:
                block = (self.rng.standard_normal(frame) * ROOM_NOISE).astype(np.int16)
            reached = self.played + frame
            due = [m for m in self.marks if m[0] < reached]
            self.marks = [m for m in self.marks if m[0] >= reached]
            self.played = reached
        return block, due

    def run(self) -> None:
        frame = self.recorder.frame_size
        next_at = time.perf_counter()
        while not self.stop_event.is_set():
            block, due = self.next_block()
            for _, kind, data in due:
                self.log(kind, data.get("clip", ""), **{k: v for k, v in data.items() if k != "clip"})
            self.recorder._publish_audio_block(block)
            next_at += frame / wavio.RATE / self.pace
            delay = next_at - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                self.max_lag = max(self.max_lag, -delay)


SERVER_SOURCE = Path(__file__).resolve().parents[2] / "pilot" / "server.py"


def server_listener_wiring(source: str) -> tuple[str, dict[str, str]]:
    """How the server wires the listener it starts, read from its source (pure; nothing is imported).

    Returns the server method the listener calls for an addressed turn (``on_command=self.X``) and every attribute
    the server sets on that listener afterwards (``self._voice_listener.attr = self.method``), so the bench wires its
    own listener the same way and follows any change to the server's wiring.
    """
    import ast

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and node.name == "_handle_voice_listener_start":
            on_command = ""
            attrs: dict[str, str] = {}
            for sub_node in ast.walk(node):
                if isinstance(sub_node, ast.Call) and getattr(sub_node.func, "id", "") == "ContinuousVoiceListener":
                    for kw in sub_node.keywords:
                        if kw.arg == "on_command" and isinstance(kw.value, ast.Attribute):
                            on_command = kw.value.attr
                if isinstance(sub_node, ast.Assign) and len(sub_node.targets) == 1:
                    target, value = sub_node.targets[0], sub_node.value
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Attribute)
                        and target.value.attr == "_voice_listener"
                        and isinstance(value, ast.Attribute)
                        and isinstance(value.value, ast.Name)
                        and value.value.id == "self"
                    ):
                        attrs[target.attr] = value.attr
            return on_command, attrs
    raise ValueError("the server has no _handle_voice_listener_start")


def wire_like_server(listener: Any, server: Any, source: str | None = None) -> dict[str, str]:
    """Set on ``listener`` every attribute the server sets on its own listener; returns what was wired."""
    _, attrs = server_listener_wiring(source if source is not None else SERVER_SOURCE.read_text(encoding="utf-8"))
    for attr, method in attrs.items():
        setattr(listener, attr, getattr(server, method))
    return attrs


def table_rows(folder: Path) -> dict[str, int]:
    """Row counts of every table in every SQLite file under ``folder`` (what was stored)."""
    counts: dict[str, int] = {}
    for path in folder.rglob("*.db"):
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1) as db:
                for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                    if name.startswith("sqlite_"):
                        continue
                    counts[f"{path.name}:{name}"] = db.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        except sqlite3.Error:
            continue
    return counts


# Tables that change on their own while the daemon runs, not because of what was heard.
NOT_STORAGE = ("llm_cache.db", "permission_audit.db", "agent_gateway_audit.db", "audit")


def stored_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        k: after[k] - before.get(k, 0)
        for k in after
        if after[k] > before.get(k, 0) and not any(skip in k for skip in NOT_STORAGE)
    }


def system_cpu() -> float | None:
    """The whole machine's CPU use since the last call, in percent (None without psutil): other programs'
    load slows recognition and synthesis, so a run records it beside its timings."""
    try:
        import psutil

        return float(psutil.cpu_percent(interval=None))
    except Exception:
        return None


class Harness:
    """One listener and one server for a whole run; scenarios run one after another on it."""

    def __init__(
        self,
        *,
        audio_root: str,
        data_dir: Path,
        model: ModelKind,
        pace: float = 1.0,
        configure: Callable[[Any], None] | None = None,
        owner_settings: bool = False,
        wait_until_ready: bool = True,
    ) -> None:
        self.configure = configure  # adjusts the loaded settings before the server is built (CI: models it fetched)
        # A local run: the person's [model] and named [voice] settings copied, the speech models read in place
        # (bench_root). Off on a runner, which has neither.
        self.owner_settings = owner_settings
        # False: the first scenario may start while the listener's recognisers and the local voice are still
        # loading, as a person who speaks right after start-up would (the first-utterance mode's "during loading").
        self.wait_until_ready = wait_until_ready
        self.copied_settings: list[str] = []
        self.models_found: dict[str, str] = {}
        self.audio_root = audio_root
        self.data_dir = data_dir
        self.model = model
        self.pace = pace
        self.log = Log()
        self.babble: Any = None
        self.worker_spawns = 0
        self.voice_load_s: float | None = None
        self.server: Any = None
        self.listener: Any = None
        self.pump: FilePump | None = None
        self._setup_text = ""
        self._observed: list[Any] = [None]

    # Building -------------------------------------------------------------------------------------------

    async def build(self) -> None:
        from .bench_root import check_inside, copy_owner_settings, enter, read_models_in_place

        # Every Heliox folder (settings, data, state, runtime, ~/.cache/heliox, setup's folder) and the temporary
        # files under data_dir, before pilot.config loads: never the person's. Raises if something imported
        # pilot.config first, or if any folder resolves outside data_dir.
        root = enter(self.data_dir)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only, for this process and the voice worker
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never download: every model must already be on disk
        self.data_dir.mkdir(parents=True, exist_ok=True)
        from unittest.mock import MagicMock

        import pilot.system.local_tts_worker as tts_worker
        import pilot.system.voice as voice_module
        from pilot.config import PilotConfig
        from pilot.live.engine import LiveCompanion
        from pilot.server import PilotServer
        from pilot.system.voice import ContinuousVoiceListener

        log = self.log
        config = PilotConfig.load()
        config.save = MagicMock()
        if self.owner_settings:
            # The person's provider and voice settings, copied by name onto the bench root's own config.
            self.copied_settings = copy_owner_settings(config)
        config.live.enabled = True
        config.live.grounded_checks = False  # the local model takes minutes on a CPU
        # CPU only, said in the setting itself: the voice worker's GPU choice does not read CUDA_VISIBLE_DEVICES=""
        # as "no card" (on 12 November it put Kokoro on the failing GPU and fell back to the system voice).
        config.voice.tts_device = "cpu"
        if self.configure is not None:
            self.configure(config)
        if self.owner_settings:
            # The speech models where they already are, read-only; a missing one stops the run with its name.
            self.models_found = read_models_in_place(config)
        check_inside(root)  # once more, right before anything that can save is built
        server = PilotServer(config)
        self.server = server

        async def broadcast(method: str, params: Any) -> None:
            params = params or {}
            if method == "voice_status":
                status = str(params.get("status", ""))
                kind = {"wake_detected": tl.WAKE, "follow_up_detected": tl.FOLLOW_UP,
                        "ambient_ignored": tl.AMBIENT_IGNORED}.get(status)  # fmt: skip
                log(kind or "voice status", status if kind is None else str(params.get("transcript", "")))
            elif method == "voice_result":
                log(tl.RESULT, str(params.get("status", "")), status=params.get("status"),
                    message=str(params.get("message") or params.get("result") or "")[:300],
                    plan_id=params.get("plan_id", ""))  # fmt: skip
            elif method == "live_interjection":
                log(tl.CARD, str(params.get("title", "")), message=str(params.get("message", "")))

        server._broadcast_notification = broadcast

        async def planner_stub(params: dict, ws: Any) -> dict:
            model = self.model  # read at call time: one harness serves every kind of model
            log(tl.PLANNER, str(params.get("input", ""))[:120], model=model.name)
            try:
                await asyncio.sleep(model.plan_seconds())
            except asyncio.CancelledError:
                # The turn was stopped before its plan came back (a split sentence withdrawn, a stop): no action.
                log(tl.PLANNER_CANCELLED, str(params.get("input", ""))[:120])
                raise
            if not model.available:
                return {"status": "error", "message": NO_MODEL_MESSAGE}
            return {"status": "success", "message": STUB_SUCCESS}

        server._handle_execute = planner_stub

        async def interject_stub(params: dict, ws: Any) -> dict:
            log(tl.INTERJECT, str(params.get("input", "")), mode=params.get("mode", ""))
            return {"status": "revising", "message": "Revising the plan."}

        server._handle_interject = interject_stub

        original_speaker = server._speech_coordinator._speaker

        async def speaker(text: str, recorder: Any) -> Any:
            setup = bool(self._setup_text) and text == self._setup_text
            log(tl.SPEAKS, text[:300], setup=setup, stub=text == STUB_SUCCESS)
            self._speaking_setup = setup
            try:
                result = await original_speaker(text, recorder)
                log(tl.DONE, "", setup=setup)
                return result
            except asyncio.CancelledError:
                log(tl.CUT_OFF, "", setup=setup)
                raise

        server._speech_coordinator._speaker = speaker

        async def silent_play(path: Any) -> None:
            setup = self._speaking_setup
            log(tl.AUDIO_STARTS, "", setup=setup)
            raw, rate = wavio.decode(Path(path).read_bytes())
            await asyncio.sleep(len(raw) / rate)  # consumed in real time, never played

        voice_module._play_local_tts_file = silent_play

        def silent_system_voice(name: str) -> Any:
            # The operating system's own voice (the fallback when the local voice fails) would play aloud:
            # replaced on every system, so a run can never be heard.
            async def fake(text: str, *_args: Any, **_kwargs: Any) -> str:
                log("system voice (silent)", text[:120], fallback=name)
                log(tl.AUDIO_STARTS, "", setup=self._speaking_setup, fallback=name)
                await asyncio.sleep(0.35 * max(1, len(text.split())))
                return ""

            return fake

        voice_module.warm_output_line = lambda *_a, **_k: None  # no output device is ever opened
        for name in ("_tts_windows", "_tts_macos", "_tts_linux"):
            if hasattr(voice_module, name):
                setattr(voice_module, name, silent_system_voice(name))

        async def no_fixed_window(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("the bench never records from the real microphone")

        voice_module._record_audio = no_fixed_window

        # The product gives the voice worker 180 s to load and then falls back to the system voice. On a loaded
        # machine the load can take longer; the bench waits for it (and records how long it took), because
        # timings measured with a fallback voice say nothing about the real one.
        tts_worker.LOAD_TIMEOUT_SECONDS = max(tts_worker.LOAD_TIMEOUT_SECONDS, VOICE_LOAD_WAIT_S)
        manager = tts_worker.manager()
        original_spawn = manager._spawn

        def counted_spawn(*args: Any) -> Any:
            self.worker_spawns += 1
            log("voice worker started", str(self.worker_spawns))
            return original_spawn(*args)

        manager._spawn = counted_spawn

        observed = self._observed

        async def observe() -> Any:
            return observed[0]

        server._live = LiveCompanion(
            settings=lambda: server.config.live,
            observe=observe,
            emit=server._broadcast_notification,
            speak=server._live_speak,
            user_speaking=lambda: bool(server._voice_listener is not None and server._voice_listener.user_is_speaking),
            heliox_speaking=lambda: bool(server._speech_coordinator.status().get("active")),
            relevance_path=self.data_dir / "bench_relevance.json",
            timing_path=self.data_dir / "bench_timing.json",
        )
        await server._live.start()

        # What the server's own listener calls for an addressed turn: the turn entry point (it holds the floor,
        # lets Live see the turn first and runs the conversation layer), or the dispatch on servers without it.
        turn_entry, _ = server_listener_wiring(SERVER_SOURCE.read_text(encoding="utf-8"))
        original_dispatch = getattr(server, turn_entry)

        async def server_turn(text: str) -> None:
            # A turn the server starts itself (the joined sentence after a withdrawal): on the timeline too.
            self._busy_until = 0.0
            log(tl.DISPATCH, text, joined=True)
            await original_dispatch(text)

        setattr(server, turn_entry, server_turn)  # the listener's own turns still go through ``dispatch`` below

        async def dispatch(text: str) -> None:
            lst = self.listener
            rec = lst._recorder
            seg = self.pump.current_segment() if self.pump else {}
            self._busy_until = 0.0
            log(tl.DISPATCH, text, engine=lst.last_transcription_engine, endpoint=rec.last_endpoint_reason,
                endpoint_ms=round(rec.last_endpoint_ms), asr_ms=round(lst.last_transcription_ms),
                kind=seg.get("kind", ""), line=seg.get("text", ""))  # fmt: skip
            await original_dispatch(text)

        async def overheard(text: str) -> None:
            self._busy_until = 0.0
            log(tl.OVERHEARD, text)
            await server._voice_overheard(text)

        listener = ContinuousVoiceListener(
            wake_words=["hey heliox", "heliox", "hey pilot"],
            on_command=dispatch,
            on_status=server._voice_status_broadcast,
            workflow_control=server._voice_workflow_control_dispatch,
            config=server.config,
            on_partial=server._voice_partial,
            on_turn_hold=server._voice_turn_hold,
            on_overheard=overheard,
        )
        # Wired exactly as the server wires its own listener (read from its source), so a sentence split at a
        # pause can be withdrawn and joined; without it every bench run was the unwired case.
        wire_like_server(listener, server)
        server_withdraw = getattr(listener, "withdraw_turn", None)
        if server_withdraw is not None:

            async def withdraw(turn_id: str, joined: str, gap: float | None = None) -> Any:
                result = await server_withdraw(turn_id, joined, gap)
                log(tl.WITHDRAW, joined[:120], turn_id=turn_id, gap=gap, status=str(getattr(result, "status", result)))
                return result

            listener.withdraw_turn = withdraw  # the server's own method, with what it decided on the timeline
        self.listener = listener
        original_note = listener.note_interruption

        def note_interruption(*args: Any, **kwargs: Any) -> Any:
            log(tl.BARGE_IN, "")
            return original_note(*args, **kwargs)

        listener.note_interruption = note_interruption
        recorder = listener._recorder
        pump = FilePump(recorder, log, self.pace)
        self.pump = pump
        log.audio_clock = lambda: pump.played / wavio.RATE

        def fake_start() -> bool:
            recorder._loop = asyncio.get_event_loop()
            recorder._queue = asyncio.Queue()
            recorder.input_device_name = "conversation bench file"
            recorder.capture_backend = "file"
            recorder._soundcard_recorder = pump  # marks the stream open, so the real microphone is never opened
            threading.Thread(target=pump.run, daemon=True, name="bench-pump").start()
            return True

        recorder.start = fake_start
        recorder.stop = pump.stop_event.set
        server._voice_listener = listener
        log("listener start", str(await listener.start()))

        async def watch_voice_activity() -> None:
            state = {"speaking": False, "heard": listener.transcripts_received}
            while True:
                try:
                    poll_listener(state)
                except Exception as error:  # the watcher must outlive any surprise in what it reads
                    log("watcher error", f"{type(error).__name__}: {error}"[:200])
                await asyncio.sleep(0.01)

        def poll_listener(state: dict[str, Any]) -> None:
            if listener.transcripts_received != state["heard"]:
                # Every utterance the listener turned into words, addressed or not (read from its counters,
                # which survive changes to how it records and transcribes).
                state["heard"] = listener.transcripts_received
                self._awaiting_words_since = None
                self._busy_until = time.perf_counter() + OUTCOME_WAIT_S
                seg = self.pump.current_segment() if self.pump else {}
                log(
                    tl.TRANSCRIPT,
                    str(getattr(listener, "last_transcript", "")),
                    engine=str(getattr(listener, "last_transcription_engine", "")),
                    asr_ms=getattr(listener, "last_transcription_ms", None),
                    kind=seg.get("kind", ""),
                    wake_text=str(getattr(listener, "_last_wake_text", "") or ""),  # what the wake check read
                )
            now_speaking = bool(recorder.user_is_speaking)
            if now_speaking != state["speaking"]:
                state["speaking"] = now_speaking
                self._awaiting_words_since = None if now_speaking else time.perf_counter()
                log(
                    "vad start" if now_speaking else "vad end",
                    "" if now_speaking else f"{recorder.last_endpoint_reason}/{recorder.last_endpoint_ms:.0f}ms",
                    turn_p=recorder.last_turn_probability,
                )

        self._vad_watch = asyncio.ensure_future(watch_voice_activity())
        if not self.wait_until_ready:
            log("ready", "not waited for")
            return
        for _ in range(900):
            if listener._streaming is not None and listener._streaming.ready and listener._turn_detector is not None:
                break
            await asyncio.sleep(0.1)
        log("ready", f"streaming={listener._streaming is not None and listener._streaming.ready}")
        started = log.now()
        await server._speak_voice_response("Ready.")  # loads the voice worker, as a running daemon has
        await self._wait(lambda: bool(manager.loaded), VOICE_LOAD_WAIT_S)
        self.voice_load_s = log.now() - started
        log("voice warm", f"{log.now() - started:.1f} s", loaded=bool(manager.loaded))
        print(f"local voice ready after {log.now() - started:.0f} s (loaded={manager.loaded})", flush=True)
        babble = Path(self.audio_root) / "synthetic" / "babble.wav"
        if babble.exists():
            self.babble = wavio.read(babble)

    def save_session(self, folder: Path) -> Path:
        """Every event of the run, between scenarios included (a transcript that came after its scenario had
        closed, the drain before the next one), one per line: ``<folder>/session.jsonl``."""
        import json

        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "session.jsonl"
        with self.log.lock:
            events = list(self.log.events)
        path.write_text("".join(json.dumps(row) + "\n" for row in tl.to_json(events)), encoding="utf-8")
        return path

    async def close(self, session_to: Path | None = None) -> None:
        if session_to is not None:
            self.save_session(session_to)
        if getattr(self, "_vad_watch", None) is not None:
            self._vad_watch.cancel()
        if self.pump:
            self.pump.stop_event.set()
        from pilot.system.local_tts_worker import manager

        for closing in ([self.listener.stop()] if self.listener else []) + [manager().shutdown()]:
            try:
                await asyncio.wait_for(closing, 15)
            except Exception:
                pass

    # Audio -----------------------------------------------------------------------------------------------

    def load_clip(self, ref_text: str, condition: str) -> tuple[Any, float, float, Any]:
        ref = audio_refs.parse_ref(ref_text)
        path = audio_refs.resolve(self.audio_root, ref)
        clean = wavio.read(path, ref.start, ref.end)
        first, last = voiced_span(clean)
        heard = conds.apply(condition, clean, babble=self.babble)
        scale = len(heard) / max(1, len(clean))
        segments = None
        meta = Path(path).with_suffix(".json")
        if ref.path.startswith("synthetic/not_addressed/") and meta.exists():
            import json

            segments = json.loads(meta.read_text(encoding="utf-8"))["segments"]
            segments = [{**s, "start": s["start"] * scale, "end": s["end"] * scale} for s in segments]
        return heard, first * scale, last * scale, segments

    # Running a scenario ---------------------------------------------------------------------------------

    def _speech_idle(self) -> bool:
        status = self.server._speech_coordinator.status()
        return not status["active"] and not status["queued"]

    async def _wait(self, check: Callable[[], bool], limit: float) -> bool:
        end = time.perf_counter() + limit
        while time.perf_counter() < end:
            if check():
                return True
            await asyncio.sleep(0.05)
        return False

    _setup_task: asyncio.Task[Any] | None = None
    _speaking_setup = False
    _awaiting_words_since: float | None = None
    _busy_until = 0.0  # words were heard and the listener has not yet acted on them or let them go

    def _apply_state(self, state: dict[str, Any], scenario_id: str) -> None:
        from pilot.server import PendingConfirmation

        server = self.server
        if state.get("pending_approval"):
            plan_id = f"bench-{scenario_id}"
            server._pending_confirms[plan_id] = PendingConfirmation(plan_id=plan_id, event=asyncio.Event())
            server._active_plan_id = plan_id
        if "task_running" in state:
            server._interactive_request_active = bool(state["task_running"])
        if state.get("conversation_open"):
            self.listener.arm_follow_up_window()
        if state.get("observation"):
            from pilot.live.engine import Observation

            self._observed[0] = Observation(**state["observation"])

    async def run(self, scenario: Scenario, condition: str) -> list[tl.Event]:
        server, log, pump = self.server, self.log, self.pump
        assert pump is not None
        await self._wait(self._speech_idle, 40)
        # Let the listener finish the previous scenario's words first: on a busy CPU the recognisers fall behind
        # real time, and their late words must not be scored as this scenario's.
        drained_from = time.perf_counter()
        queue = getattr(self.listener._recorder, "_queue", None)
        await self._wait(lambda: queue is None or queue.qsize() == 0, 180)
        await self._quiet_for(2.5, 180)
        drain_s = time.perf_counter() - drained_from
        if drain_s > 6:
            log("listener behind", f"{drain_s:.0f} s spent letting the previous words finish", seconds=drain_s)
        await asyncio.sleep(0.8)
        before = table_rows(self.data_dir)
        start = log.now()
        log(tl.SCENARIO, scenario.id, condition=condition, model=self.model.name, cpu_percent=system_cpu())
        self._apply_state(scenario.setup, scenario.id)
        if scenario.setup.get("heliox_says"):
            self._setup_text = str(scenario.setup["heliox_says"])
            log(tl.SETUP_SPEECH, self._setup_text[:80], setup=True)
            self._setup_task = asyncio.ensure_future(server._speak_voice_response(self._setup_text))
        last_audio_check = start
        cursor: int | None = None  # where the previous clip ends, on the audio clock
        for step in scenario.steps:
            if step.when == "heliox_audio":
                since = last_audio_check
                await self._wait(lambda since=since: any(e.event == tl.AUDIO_STARTS for e in log.since(since)), 30)
                started = next((e for e in log.since(since) if e.event == tl.AUDIO_STARTS), None)
                anchor = int(float(started.data.get("audio_t", 0)) * wavio.RATE) if started else pump.queued_end()
            elif step.when == "heliox_done":
                since = last_audio_check
                await self._wait(
                    lambda since=since: any(e.event in (tl.DONE, tl.CUT_OFF) for e in log.since(since)), 60
                )
                ended = next((e for e in log.since(since) if e.event in (tl.DONE, tl.CUT_OFF)), None)
                anchor = int(float(ended.data.get("audio_t", 0)) * wavio.RATE) if ended else pump.queued_end()
            elif step.when == "previous_done":
                # The previous utterance has been heard, acted on and answered before the next state and words.
                await self._wait(lambda: not pump.busy(), 60)
                await self._quiet_for(1.5, 90)
                anchor = pump.queued_end()
            else:  # "start": right after the previous clip, on the audio clock
                anchor = cursor if cursor is not None else pump.queued_end()
            if step.set_state:
                self._apply_state(step.set_state, scenario.id)
            if not step.say:
                continue
            clip, first, last, segments = self.load_clip(step.say, condition)
            start_at = anchor + int((step.wait + step.delay) * wavio.RATE)
            cursor = pump.play(clip, step.say, first=first, last=last, segments=segments, start_at=start_at)
            last_audio_check = log.now()
        await self._wait(lambda: not pump.busy(), (pump.queued_end() - pump.played) / wavio.RATE / pump.pace + 60)
        await self._settle(scenario)
        pending = server._pending_confirms.get(f"bench-{scenario.id}")
        if scenario.setup.get("pending_approval") or any(s.set_state.get("pending_approval") for s in scenario.steps):
            log(
                tl.APPROVAL, "gone" if pending is None else f"confirmed={pending.confirmed} resolved={pending.resolved}"
            )
        delta = stored_delta(before, table_rows(self.data_dir))
        if delta:
            log(tl.STORED, ", ".join(f"{k}+{v}" for k, v in delta.items()), rows=sum(delta.values()))
        await self._teardown()
        return [e for e in log.since(start)]

    async def _settle(self, scenario: Scenario) -> None:
        await asyncio.sleep(2.5)
        # The recognisers may still be working on the last utterance (on a loaded CPU, for many seconds): until
        # its words arrive (or WORDS_WAIT_S passes, for an utterance heard as nothing) the scenario is not quiet.
        await self._quiet_for(min(scenario.settle, 4.0), scenario.settle + self.model.plan_seconds() + 20)

    def _quiet(self) -> bool:
        """Nothing logged for 2.5 s, no words still being recognised or acted on, no voice, no command running."""
        log, listener = self.log, self.listener
        busy = [e for e in log.since(log.now() - 2.5) if e.event not in ("voice status",)]
        since = self._awaiting_words_since
        transcribing = since is not None and time.perf_counter() - since < WORDS_WAIT_S
        transcribing = transcribing or time.perf_counter() < self._busy_until
        return (
            not busy
            and not transcribing
            and self._speech_idle()
            and not listener._command_tasks
            and not any(not t.done() for t in getattr(self.server, "_background_dispatches", ()))
            and not listener.user_is_speaking
        )

    async def _quiet_for(self, hold: float, cap_s: float) -> None:
        cap = time.perf_counter() + cap_s
        while time.perf_counter() < cap:
            if self._quiet():
                await asyncio.sleep(max(0.0, hold - 2.5))
                if self._quiet():
                    return
            await asyncio.sleep(0.2)

    async def _teardown(self) -> None:
        server, listener = self.server, self.listener
        if self._setup_task is not None and not self._setup_task.done():
            self._setup_task.cancel()
        self._setup_task = None
        self._setup_text = ""
        await server._speech_coordinator.stop_all()
        for task in [*listener._command_tasks, *getattr(server, "_background_dispatches", set())]:
            task.cancel()  # a joined turn the server started must not speak into the next scenario
        server._pending_confirms.clear()
        server._active_plan_id = ""
        server._interactive_request_active = False
        listener.clear_follow_up_window()
        listener.end_interruption()
        self._observed[0] = None
        # Every scenario starts from a clean conversation, as a person starting over would: the new-conversation
        # call drops what Live holds, clears the conversation layer and closes the follow-up windows.
        new_conversation = getattr(server, "_handle_voice_new_conversation", None)
        if new_conversation is not None:
            await new_conversation({}, None)


def model_kind(name: str) -> ModelKind:
    return KINDS[name]
