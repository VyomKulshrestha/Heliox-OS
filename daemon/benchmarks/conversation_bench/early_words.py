"""Are the streaming recogniser's words already right when the end of the turn first looks likely?

    python -m benchmarks.conversation_bench.early_words --root <bench root> --audio <folder> [--sets synthetic]
        [--out early_words.json]

For each recorded utterance, the listener's own recorder (VAD endpointer, Smart Turn, the hold after an unfinished
sentence, the short-answer hint) is fed one frame at a time on the audio's own clock, as in turn_split. At each
candidate end of turn (200 ms of quiet, the moment the listener starts its speculative final pass) the streaming
recogniser's words so far are taken. The candidate that the turn really ended at is compared with the final pass on
the whole turn (Parakeet when the settings choose it, else Whisper), both normalised as the scorecard normalises
them: equal, one word apart, or more.

Times are on the audio clock, from the last word (the end of the clip's last loud frame): when that candidate came,
and when the turn ended. The final pass's own running time is wall time and depends on the machine; it is printed
with the CPU load beside it and marked "under load" at 50 % or more.

Counts only for the person's own recordings: no transcript and no file name is printed or saved for them.

Runs under the bench root (``bench_root.enter``): the person's voice settings copied by name, the speech models read
where they are, nothing written outside the root.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from . import audio as audio_refs
from . import bench_root, machine, scoring, turn_split, wavio

RATE = turn_split.RATE
OWNER_SETS = ("owner",)


def word_distance(a: list[str], b: list[str]) -> int:
    """Word-level edit distance."""
    previous = list(range(len(b) + 1))
    for i, word in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (word != other)))
        previous = current
    return previous[-1]


def judge(streaming: str, final: str) -> str:
    distance = word_distance(scoring.normalise(streaming), scoring.normalise(final))
    return "equal" if distance == 0 else ("one word" if distance == 1 else "more")


def without_wake(text: str) -> str:
    """The request without its wake phrase, as the listener strips it (the two recognisers spell "Heliox" apart)."""
    from pilot.system.voice import loose_wake_prefix

    prefix = loose_wake_prefix(text)
    return text[len(prefix) :].strip(" ,.") if prefix else text


VERDICTS = ("as_is", "flushed", "flushed_request")


class StreamingNow(turn_split.InlineSink):
    """The inline streaming recogniser, with its full result readable at any frame."""

    def now(self) -> str:
        from pilot.system.streaming_asr import tidy

        if self.stream is None:
            return ""
        result = self.recognizer.get_result(self.stream)
        return tidy(result if isinstance(result, str) else getattr(result, "text", ""))

    def flushed(self, frames: list[Any]) -> tuple[str, float]:
        """The words if the stream were ended at this frame: the same audio on a fresh stream, then the tail of
        silence and the end of input the listener's own finish adds. Returns (words, seconds it took)."""
        import numpy as np

        from pilot.system.streaming_asr import TAIL_SECONDS, tidy

        started = time.perf_counter()
        stream = self.recognizer.create_stream()
        audio = np.concatenate([np.asarray(f).astype(np.float32).ravel() for f in frames]) / 32768.0
        stream.accept_waveform(RATE, audio)
        stream.accept_waveform(RATE, np.zeros(int(RATE * TAIL_SECONDS), np.float32))
        stream.input_finished()
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)
        result = self.recognizer.get_result(stream)
        return tidy(result if isinstance(result, str) else getattr(result, "text", "")), time.perf_counter() - started


async def one_clip(recorder: Any, sink: StreamingNow, clip: Any, final_pass: Any, seed: int = 0) -> dict[str, Any]:
    """Feed one clip; the decisive candidate's streaming words and times, the final pass's words and running time."""
    import numpy as np

    from pilot.system.vad import AdaptiveEnergyThreshold
    from pilot.system.voice import _RECORDER_STOP

    rng = np.random.default_rng(seed)
    recorder._loop = asyncio.get_running_loop()
    recorder._queue = asyncio.Queue()
    recorder._adaptive_threshold = AdaptiveEnergyThreshold(recorder.energy_threshold)
    lead = float(np.sqrt(np.mean(clip[: RATE // 5].astype(np.float64) ** 2))) or 30.0
    room = rng.normal(0, lead, int(turn_split.ROOM_SECONDS * RATE)).astype(np.int16)
    tail = rng.normal(0, lead, int(turn_split.TAIL_SECONDS * RATE)).astype(np.int16)
    x = np.concatenate([room, clip.astype(np.int16), tail])
    last_word = (len(room) + len(clip)) / RATE
    fed = [0.0]
    # The recorder's own last word: the end of the last frame at or above the endpointer's threshold
    # (_utterance_last_voice changes on each such frame), beside the clip's last loud frame above.
    voiced = {"stamp": getattr(recorder, "_utterance_last_voice", None), "at": None}
    candidates: list[dict[str, Any]] = []
    ends: list[dict[str, Any]] = []

    def on_candidate(frames: Any) -> None:
        candidates.append({"at": fed[0], "words": sink.now(), "frames": list(frames), "cancelled": False})

    def on_cancel() -> None:
        if candidates:
            candidates[-1]["cancelled"] = True

    recorder.on_candidate_end = on_candidate
    recorder.on_candidate_cancel = on_cancel

    async def consume() -> None:
        while True:
            path = await recorder.record_utterance(timeout=None, keep_backlog=True)
            if path is None:
                return
            at = fed[0]
            samples = wavio.read(path)
            started = time.perf_counter()
            words = final_pass(path) if len(samples) / RATE > turn_split.PIECE_SECONDS else ""
            ran = time.perf_counter() - started
            Path(path).unlink(missing_ok=True)
            decisive = next((c for c in reversed(candidates) if not c["cancelled"]), None)
            ends.append({"at": at, "last_voiced": voiced["at"], "seconds": len(samples) / RATE, "reason": recorder.last_endpoint_reason,
                         "final": words, "final_s": ran, "candidate": decisive,
                         "mid_candidates": sum(c["cancelled"] for c in candidates)})  # fmt: skip
            candidates.clear()

    task = asyncio.ensure_future(consume())
    queue = recorder._queue
    size = recorder.frame_size
    for i in range(0, len(x) - size + 1, size):
        fed[0] = (i + size) / RATE
        recorder.feed(x[i : i + size])
        for _ in range(3):
            await asyncio.sleep(0)
        while not (queue.empty() and any(not getter.done() for getter in queue._getters)) and not task.done():
            await asyncio.sleep(0.001)
        for _ in range(3):
            await asyncio.sleep(0)
        stamp = getattr(recorder, "_utterance_last_voice", None)
        if stamp is not None and stamp != voiced["stamp"]:
            voiced["stamp"], voiced["at"] = stamp, fed[0]
    queue.put_nowait(_RECORDER_STOP)
    await task
    pieces = [e for e in ends if e["seconds"] > turn_split.PIECE_SECONDS]
    if len(pieces) != 1 or pieces[0]["candidate"] is None:
        return {"usable": False, "pieces": len(pieces)}
    end = pieces[0]
    candidate = end["candidate"]
    flushed, flush_s = sink.flushed(candidate["frames"])
    return {
        "usable": True,
        "as_is": judge(candidate["words"], end["final"]),
        "flushed": judge(flushed, end["final"]),
        "flushed_request": judge(without_wake(flushed), without_wake(end["final"])),
        "flush_s": round(flush_s, 3),
        "candidate_after_last_word_s": round(candidate["at"] - last_word, 3),
        "end_after_last_word_s": round(end["at"] - last_word, 3),
        **_from_last_voiced(end["last_voiced"], candidate["at"], end["at"], last_word),
        "final_pass_s": round(end["final_s"], 3),
        "mid_candidates": end["mid_candidates"],
        "reason": end["reason"],
        "streaming": candidate["words"],
        "streaming_flushed": flushed,
        "final": end["final"],
    }


def _from_last_voiced(last_voiced: float | None, candidate: float, end: float, last_word: float) -> dict[str, Any]:
    """The same times from the recorder's own last voiced frame, and how far that frame is from the clip's."""
    if last_voiced is None:
        return {}
    return {
        "candidate_after_last_voiced_s": round(candidate - last_voiced, 3),
        "end_after_last_voiced_s": round(end - last_voiced, 3),
        "last_voiced_minus_last_word_s": round(last_voiced - last_word, 3),
    }


def summarise(rows: list[dict[str, Any]], load: list[float | None]) -> dict[str, Any]:
    used = [r for r in rows if r["usable"]]

    def spread(key: str) -> dict[str, float] | None:
        values = sorted(r[key] for r in used if r.get(key) is not None)
        if not values:
            return None
        return {"median": round(statistics.median(values), 3), "min": values[0], "max": values[-1]}

    known = [v for v in load if v is not None]
    busy = not known or max(known) >= machine.BUSY_PERCENT
    counts = {
        kind: {
            "equal": sum(r[kind] == "equal" for r in used),
            "one_word": sum(r[kind] == "one word" for r in used),
            "more": sum(r[kind] == "more" for r in used),
        }
        for kind in VERDICTS
    }
    return {
        "n": len(used),
        "not_one_turn": len(rows) - len(used),
        **counts,
        "flush_s": spread("flush_s"),
        "candidate_after_last_word_s": spread("candidate_after_last_word_s"),
        "end_after_last_word_s": spread("end_after_last_word_s"),
        "candidate_after_last_voiced_s": spread("candidate_after_last_voiced_s"),
        "end_after_last_voiced_s": spread("end_after_last_voiced_s"),
        "last_voiced_minus_last_word_s": spread("last_voiced_minus_last_word_s"),
        "final_pass_s": spread("final_pass_s"),
        "final_pass_load": "under load" if busy else "quiet",
        "cpu_percent": [round(v) for v in known],
        "mid_candidates": sum(r["mid_candidates"] for r in used),
    }


def anonymous(name: str) -> str:
    return "owner-" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]


async def _main(args: argparse.Namespace) -> int:
    bench_root.enter(Path(args.root))
    from pilot.config import PilotConfig
    from pilot.system.voice import ContinuousVoiceListener, is_short_decision

    config = PilotConfig.load()
    copied = bench_root.copy_owner_settings(config)
    config.voice.tts_device = "cpu"
    found = bench_root.read_models_in_place(config)
    print("settings copied:", ", ".join(copied) or "none")
    print("models:", ", ".join(f"{k} ({v})" for k, v in found.items()))
    bench_root.check_inside(Path(args.root))
    listener = ContinuousVoiceListener(wake_words=["hey heliox", "heliox"], config=config)
    await listener._warm_turn_taking()
    recorder = listener._recorder
    recorder.use_file_source()
    recorder._loop = asyncio.get_running_loop()
    listener._configure_turn_taking()
    streaming = listener._streaming
    if streaming is None or not streaming.ready or recorder.turn_detector is None:
        raise SystemExit("the streaming recogniser or Smart Turn did not load; nothing to compare")
    sink = StreamingNow(streaming._recognizer)
    recorder.stream_sink = sink
    recorder.complete_hint = lambda: is_short_decision(sink.partial())

    import pilot.system.voice as voice

    parakeet = listener._parakeet() if config.voice.final_pass == "parakeet" else None

    def final_pass(path: str) -> str:
        if parakeet is not None:
            return str(parakeet.transcribe(voice._normalize_voice_audio(wavio.read(path))) or "")
        return ""

    if parakeet is None:
        raise SystemExit("the final pass is not Parakeet in these settings; this count compares against Parakeet")
    print("final pass: parakeet; streaming: zipformer; Smart Turn: on")

    root = audio_refs.audio_root(args.audio)
    report: dict[str, Any] = {}
    for name in [s.strip() for s in args.sets.split(",") if s.strip()]:
        clips = turn_split.load_clips(root, [name])
        rows, load = [], []
        for index, (clip_name, clip) in enumerate(clips):
            load.append(machine.cpu_percent(0.5) if index % 10 == 0 else None)
            row = await one_clip(recorder, sink, clip, final_pass)
            if name in OWNER_SETS:
                for key in ("streaming", "streaming_flushed", "final"):
                    row.pop(key, None)
                row["clip"] = anonymous(clip_name)
            else:
                row["clip"] = clip_name
            rows.append(row)
        report[name] = {"summary": summarise(rows, load), "rows": rows}
        s = report[name]["summary"]
        print(f"\n{name}: n = {s['n']} (not one turn: {s['not_one_turn']}); mid-sentence candidates "
              f"{s['mid_candidates']}")  # fmt: skip
        for kind, label in (
            ("as_is", "streaming words as they stand"),
            ("flushed", "streaming ended there"),
            ("flushed_request", "streaming ended there, request without the wake phrase"),
        ):
            c = s[kind]
            print(f"  {label}: equal {c['equal']}, one word apart {c['one_word']}, more {c['more']}")
        print(f"  ending the stream there took (wall, {s['final_pass_load']}): {s['flush_s']}")
        print(f"  candidate after the last word (audio clock): {s['candidate_after_last_word_s']}")
        print(f"  turn ended after the last word (audio clock): {s['end_after_last_word_s']}")
        print(
            f"  from the recorder's last voiced frame instead: candidate {s['candidate_after_last_voiced_s']}, "
            f"turn ended {s['end_after_last_voiced_s']}; that frame minus the clip's last loud frame "
            f"{s['last_voiced_minus_last_word_s']}"
        )
        print(f"  final pass running time (wall, {s['final_pass_load']}, CPU {s['cpu_percent']} %): "
              f"{s['final_pass_s']}")  # fmt: skip
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("saved:", args.out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.conversation_bench.early_words")
    parser.add_argument("--root", default=None, help="the bench root (default HELIOX_BENCH_ROOT)")
    parser.add_argument("--audio", default=None)
    parser.add_argument("--sets", default="synthetic")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    args.root = args.root or os.environ.get(bench_root.ROOT_VARIABLE, "").strip()
    if not args.root:
        raise SystemExit(f"name the bench root with --root or {bench_root.ROOT_VARIABLE}")
    return asyncio.run(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
