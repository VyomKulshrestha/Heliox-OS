"""Sentences cut in two at a pause: how often the listener's recorder ends a turn mid-sentence, per
``early_turn_threshold``, on recorded speech, independent of the speed of the machine.

    python -m benchmarks.conversation_bench.turn_split --audio <folder> [--sets owner,synthetic,lines]
        [--thresholds 1.0,0.99,0.97,0.95] [--out turn_split.json]

The listener's own ``_ContinuousRecorder`` decides every endpoint (VAD endpointer, Smart Turn at the 400 ms check
and, below 1.0, at the 200 ms candidate end, the hold after an unfinished sentence, the short-answer hint), but the
audio is fed one frame at a time on its own clock: the next frame goes in only when the recorder is idle again, waiting
on its queue, after any Smart Turn call on the last one. The streaming recogniser's partial words (the short-answer
hint) are decoded inline and refreshed every 0.25 s of audio, not of wall-clock time. Each clip starts from a fresh
room estimate with 2 s of the clip's own lead-in level before it and 2.5 s after it. Two runs give the same endpoints.

A real-time player is not fit for this: on a slow machine the listener drains the frames that arrived while it was
busy, which hides the second half of a split sentence.

Sets under ``--audio``: ``owner`` = real/desk and real/headset (private recordings, never in the repository),
``synthetic`` = the bench's addressed sentences (synthetic/<voice>/<speed>), ``lines`` = the not-addressed corpus
lines (synthetic/not_addressed/lines). Smart Turn and the streaming recogniser are used when they are on this
computer (nothing is downloaded); without them the endpoints are the endpointer's alone, and the output says so.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from . import audio as audio_refs
from . import wavio

RATE = 16000
SETS = {
    "owner": ("real/desk", "real/headset"),
    "synthetic": ("synthetic/af_bella", "synthetic/am_michael", "synthetic/bm_lewis"),
    "lines": ("synthetic/not_addressed/lines",),
}
ROOM_SECONDS = 2.0
TAIL_SECONDS = 2.5
PIECE_SECONDS = 0.3  # an endpoint shorter than this (endpoint silence included) is a click, not a piece


def utterances(samples: Any, gap: float = 0.9) -> list[Any]:
    """Voiced stretches of a recording (16 kHz float32) separated by at least ``gap`` seconds of quiet, each with
    0.3 s of lead-in, as int16."""
    import numpy as np

    frame = 160
    x = np.asarray(samples, dtype=np.float32)
    if len(x) <= frame:
        return []
    levels = np.sqrt(np.mean(x[: len(x) // frame * frame].reshape(-1, frame) ** 2, axis=1))
    loud = levels > max(0.003, 0.08 * float(levels.max()))
    spans, start, last = [], None, 0
    for i, on in enumerate(loud):
        if on:
            start = i if start is None else start
            last = i
        elif start is not None and (i - last) * 0.01 >= gap:
            spans.append((start, last))
            start = None
    if start is not None:
        spans.append((start, last))
    out = []
    for a, b in spans:
        if (b - a) * 0.01 < 0.3:
            continue
        clip = x[max(0, (a - 30) * frame) : (b + 1) * frame]
        out.append((np.clip(clip, -1, 1) * 32767).astype(np.int16))
    return out


class InlineSink:
    """The streaming recogniser decoded in the caller's thread, partial words refreshed on audio time."""

    def __init__(self, recognizer: Any, every: float = 0.25) -> None:
        self.recognizer = recognizer
        self.every = every
        self.stream: Any = None
        self._partial = ""
        self._since = 0.0

    def begin(self, frames: list[Any]) -> None:
        self.stream = self.recognizer.create_stream()
        self._partial, self._since = "", 0.0
        for frame in frames:
            self.feed(frame)

    def feed(self, frame: Any) -> None:
        import numpy as np

        from pilot.system.streaming_asr import tidy

        if self.stream is None:
            return
        samples = np.asarray(frame).astype(np.float32).ravel() / 32768.0
        self.stream.accept_waveform(RATE, samples)
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        self._since += len(samples) / RATE
        if self._since >= self.every:
            self._since = 0.0
            result = self.recognizer.get_result(self.stream)
            text = tidy(result if isinstance(result, str) else getattr(result, "text", ""))
            if text:
                self._partial = text

    def partial(self) -> str:
        return self._partial


async def endpoints(recorder: Any, clip: Any, seed: int = 0) -> list[dict[str, Any]]:
    """Feed one clip (int16) through ``recorder`` on the audio clock; every endpoint it decided, in order."""
    import numpy as np

    from pilot.system.vad import AdaptiveEnergyThreshold
    from pilot.system.voice import _RECORDER_STOP

    rng = np.random.default_rng(seed)
    recorder._loop = asyncio.get_running_loop()
    recorder._queue = asyncio.Queue()
    recorder._adaptive_threshold = AdaptiveEnergyThreshold(recorder.energy_threshold)
    lead = float(np.sqrt(np.mean(clip[: RATE // 5].astype(np.float64) ** 2))) or 30.0
    room = rng.normal(0, lead, int(ROOM_SECONDS * RATE)).astype(np.int16)
    tail = rng.normal(0, lead, int(TAIL_SECONDS * RATE)).astype(np.int16)
    x = np.concatenate([room, clip.astype(np.int16), tail])
    size = recorder.frame_size
    found: list[dict[str, Any]] = []

    async def consume() -> None:
        while True:
            path = await recorder.record_utterance(timeout=None, keep_backlog=True)
            if path is None:
                return
            seconds = len(wavio.read(path)) / RATE
            Path(path).unlink(missing_ok=True)
            found.append({"reason": recorder.last_endpoint_reason, "quiet_ms": round(recorder.last_endpoint_ms),
                          "seconds": round(seconds, 2)})  # fmt: skip

    task = asyncio.ensure_future(consume())
    queue = recorder._queue
    for i in range(0, len(x) - size + 1, size):
        recorder.feed(x[i : i + size])
        for _ in range(3):
            await asyncio.sleep(0)
        # Idle: the frame is taken and the recorder waits on its queue again (any Smart Turn call has returned).
        while not (queue.empty() and any(not getter.done() for getter in queue._getters)) and not task.done():
            await asyncio.sleep(0.001)
        for _ in range(3):
            await asyncio.sleep(0)
    queue.put_nowait(_RECORDER_STOP)
    await task
    return found


def pieces(found: list[dict[str, Any]]) -> int:
    return sum(1 for e in found if e["seconds"] > PIECE_SECONDS)


async def run(
    clips: list[tuple[str, Any]], thresholds: Iterable[float], make_recorder: Callable[[], Any]
) -> dict[str, list[dict[str, Any]]]:
    recorder = make_recorder()
    results: dict[str, list[dict[str, Any]]] = {}
    for threshold in thresholds:
        recorder.early_turn_threshold = threshold
        rows = []
        for name, clip in clips:
            found = await endpoints(recorder, clip)
            rows.append({"clip": name, "pieces": pieces(found), "endpoints": found})
        results[str(threshold)] = rows
    return results


def summary(results: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for threshold, rows in results.items():
        voiced = [r for r in rows if r["pieces"]]
        whole = [r for r in voiced if r["pieces"] == 1]
        quiet = sorted(r["endpoints"][-1]["quiet_ms"] for r in whole)
        out.append(
            {
                "threshold": threshold,
                "utterances": len(voiced),
                "split": [r["clip"] for r in voiced if r["pieces"] > 1],
                "ended_early": sum(r["endpoints"][-1]["reason"] == "smart_turn_early" for r in whole),
                "quiet_ms_median": quiet[len(quiet) // 2] if quiet else None,
            }
        )
    return out


def load_clips(root: str, sets: Iterable[str]) -> list[tuple[str, Any]]:
    clips = []
    for name in sets:
        for folder in SETS[name]:
            base = Path(audio_refs.resolve(root, audio_refs.parse_ref(folder)))
            for wav in sorted(base.rglob("*.wav")) if base.is_dir() else []:
                for n, clip in enumerate(utterances(wavio.read(wav))):
                    clips.append((f"{name}:{wav.relative_to(base).as_posix()}#{n}", clip))
    return clips


async def _main(args: argparse.Namespace) -> int:
    from pilot.config import PilotConfig
    from pilot.system.voice import ContinuousVoiceListener, is_short_decision

    # The recorder as the listener configures it (the person's settings), with Smart Turn and the streaming
    # recogniser when they are on this computer.
    config = PilotConfig.load()
    config.save = lambda *a, **k: None
    listener = ContinuousVoiceListener(wake_words=["hey heliox"], config=config)
    try:
        await listener._warm_turn_taking()
    except Exception as error:  # a model not on this computer: the endpointer alone
        print(f"turn-taking models unavailable ({error}); endpoints are the endpointer's alone")
    recorder = listener._recorder
    recorder.use_file_source()
    recorder._loop = asyncio.get_running_loop()
    listener._configure_turn_taking()
    streaming = listener._streaming
    sink = InlineSink(streaming._recognizer) if streaming is not None and streaming.ready else None
    recorder.stream_sink = sink
    recorder.complete_hint = (lambda: is_short_decision(sink.partial())) if sink else None
    print(f"Smart Turn: {recorder.turn_detector is not None}; streaming recogniser: {sink is not None}")
    root = audio_refs.audio_root(args.audio)
    clips = load_clips(root, [s.strip() for s in args.sets.split(",") if s.strip()])
    thresholds = [float(t) for t in args.thresholds.split(",")]
    results = await run(clips, thresholds, lambda: recorder)
    print("| threshold | utterances | split | ended at the 200 ms check | quiet before the end, median ms |")
    print("|---|---|---|---|---|")
    for row in summary(results):
        print(f"| {row['threshold']} | {row['utterances']} | {len(row['split'])} | {row['ended_early']} | "
              f"{row['quiet_ms_median']} |")  # fmt: skip
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
        print("endpoints:", args.out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.conversation_bench.turn_split")
    parser.add_argument("--audio", default=None)
    parser.add_argument("--sets", default="owner,synthetic")
    parser.add_argument("--thresholds", default="1.0,0.99,0.97,0.95")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--real-folders",
        action="store_true",
        help="required: reads your real Heliox settings and models (the wake calibration is never saved)",
    )
    args = parser.parse_args(argv)
    from pilot.real_folders import require_real_folders_flag

    require_real_folders_flag("conversation_bench.turn_split")
    return asyncio.run(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
