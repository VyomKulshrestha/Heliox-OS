"""Transcript-level replay of the addressing measurement (line 2.2 and the addressed clips), in a minute or two.

Two steps:

  build   recognise every utterance once and cache it (needs the speech models, CPU, about 20 minutes):
            python -m benchmarks.conversation_bench.replay build --audio <folder> [--redo]
          writes <folder>/replay_cache.jsonl. Utterances are the corpus lines of the not-addressed blocks (their
          start and end are known from block_NN.json) and the owner's real clips, each with its Parakeet and
          Whisper text, voiced seconds and speech level in dBFS.
  run     drive a decision function over the cache (no model, no audio, stdlib only):
            python -m benchmarks.conversation_bench.replay run --audio <folder> --decide pilot.system.addressing:decide
          prints the counts per label and writes a per-utterance table (TSV) next to the cache.

The decision function is called once per utterance, in order within each stream (one stream per block or clip):

    decide(utterance: dict, context: dict) -> {"act": "dispatch" | "sorry" | "overheard" | "ignore", "why": str}

``context`` holds ``heliox_speaking`` (Heliox's voice was playing when the utterance began),
``since_heliox_ended_s`` (None before Heliox has spoken in this stream) and ``history`` (the earlier utterances of
the stream with their decisions). After a "dispatch" the replay has Heliox speak a reply of ``REPLY_S`` starting
``REPLY_AFTER_S`` after the utterance; after a "sorry", a short one. That is what opens follow-up windows, so a
cascade shows up here as it does through the real listener. The audio-level number for the same scenarios is the
ordinary run: ``python -m benchmarks.conversation_bench run --sets not_addressed,understanding,control``.

Utterance segmentation differs from the real listener's: here each corpus line is one utterance, while the
listener's endpointing sometimes joins two lines said 0.4 s apart. The audio-level run is the final number.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import audio as audio_refs

REPLY_AFTER_S = 1.0
REPLY_S = 3.0
SORRY_S = 0.7
CACHE_NAME = "replay_cache.jsonl"
ACTS = ("dispatch", "sorry", "overheard", "ignore")

# The owner's real clips: (audio reference, what was said, label, Heliox speaking when it began, mic).
REAL_CLIPS: tuple[tuple[str, str, str, bool, str], ...] = (
    ("real/headset/utt00.wav", "Hey Heliox, remind me that the launch call is Monday the twelfth of November at ten.",
     "addressed", False, "bluetooth-headset-call-mode"),
    ("real/headset/utt01.wav", "Hey Heliox, remind me that the launch call is Monday the twelfth of November at ten.",
     "addressed", False, "bluetooth-headset-call-mode"),
    ("real/headset/utt02.wav", "Hey Heliox, what's connected?", "addressed", False, "bluetooth-headset-call-mode"),
    ("real/headset/utt03.wav", "Hey Heliox, what can I say?", "addressed", False, "bluetooth-headset-call-mode"),
    ("real/headset/utt04.wav", "Hey Heliox, what can I say?", "addressed", False, "bluetooth-headset-call-mode"),
    ("real/desk/usb_cue_a.wav", "Hey Heliox, what can I say?", "addressed", False, "ptron-usb-desk"),
    ("real/desk/desk_first_try.wav", "Hey Heliox, what can I say?", "addressed", False, "ptron-usb-desk"),
    ("real/desk/take_2204.wav#3.6-6.4", "Hey Heliox, what can I say?", "addressed", False, "ptron-usb-desk"),
    ("real/desk/take_2241.wav#4.7-7.5", "Hey Heliox, what can I say?", "addressed", False, "ptron-usb-desk"),
    ("real/desk/usb_cue_b.wav", "Wait, stop.", "control", True, "ptron-usb-desk"),
    ("real/desk/desk_wait_stop.wav", "Wait, stop.", "control", True, "ptron-usb-desk"),
    ("real/desk/take_2204.wav#21.6-23.6", "Wait, stop.", "control", True, "ptron-usb-desk"),
    ("real/desk/take_2241.wav#23.9-25.8", "Wait, stop.", "control", True, "ptron-usb-desk"),
    ("real/desk/take_2204.wav#33.8-38.0", "The invoice is due on the thirty-first of November.", "not_addressed", False,
     "ptron-usb-desk"),
    ("real/desk/take_2241.wav#35.1-38.7", "The invoice is due on the thirty-first of November.", "not_addressed", False,
     "ptron-usb-desk"),
    ("real/desk/take_2241.wav#41.2-44.8", "The invoice is due on the thirty-first of November.", "not_addressed", False,
     "ptron-usb-desk"),
    ("real/desk/take_2204.wav#28.7-30.2", "", "not_addressed", False, "ptron-usb-desk"),
    ("real/desk/take_2204.wav#40.9-42.5", "", "not_addressed", False, "ptron-usb-desk"),
    ("real/headset/utt05.wav", "", "not_addressed", False, "bluetooth-headset-call-mode"),
)  # fmt: skip


# Running a decision function over the cache (pure) ----------------------------------------------------------------


def load_cache(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def streams(utterances: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for u in utterances:
        out.setdefault(u["stream"], []).append(u)
    for items in out.values():
        items.sort(key=lambda u: u["t_start"])
    return out


def replay(
    utterances: list[dict[str, Any]], decide: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
) -> list[dict[str, Any]]:
    """Every utterance with the decision made on it, and which earlier dispatch opened the window it fell in."""
    rows: list[dict[str, Any]] = []
    for name, items in streams(utterances).items():
        speaking_from, speaking_until = None, None
        opened_by = ""
        history: list[dict[str, Any]] = []
        for u in items:
            t = u["t_start"]
            speaking = (
                speaking_from is not None and speaking_until is not None and speaking_from <= t < speaking_until
            ) or bool(u.get("heliox_speaking_at_start"))
            since = None if speaking_until is None or t < speaking_until else round(t - speaking_until, 2)
            decision = dict(decide(u, {"heliox_speaking": speaking, "since_heliox_ended_s": since,
                                       "history": list(history)}))  # fmt: skip
            act = decision.get("act", "ignore")
            if act not in ACTS:
                raise ValueError(f"decide() returned an unknown act {act!r}")
            row = {**u, "act": act, "why": decision.get("why", ""), "heliox_speaking": speaking,
                   "since_heliox_ended_s": since,
                   "window_opened_by": opened_by if since is not None and act != "ignore" else ""}  # fmt: skip
            rows.append(row)
            history.append(row)
            if act in ("dispatch", "sorry"):
                speaking_from = u["t_end"] + REPLY_AFTER_S
                speaking_until = speaking_from + (REPLY_S if act == "dispatch" else SORRY_S)
                opened_by = f"{name}@{u['t_start']:.1f}: {u.get('said') or u.get('parakeet', '')}"[:120]
    return rows


def summary(rows: list[dict[str, Any]]) -> dict[str, Counter]:
    by_label: dict[str, Counter] = {}
    for r in rows:
        by_label.setdefault(r["label"], Counter())[r["act"]] += 1
    return by_label


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The numbers that matter: acting on speech for someone else (2.2), and missing what was for Heliox."""
    na = [r for r in rows if r["label"] == "not_addressed"]
    addressed = [r for r in rows if r["label"] in ("addressed", "control")]
    return {
        "not_addressed_utterances": len(na),
        "not_addressed_dispatched": sum(r["act"] == "dispatch" for r in na),
        "not_addressed_spoken": sum(r["act"] in ("dispatch", "sorry") for r in na),
        "addressed_utterances": len(addressed),
        "addressed_taken": sum(r["act"] == "dispatch" for r in addressed),
        "addressed_sorry": sum(r["act"] == "sorry" for r in addressed),
    }


def load_decide(spec: str) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name or "decide")


def write_table(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["stream", "t_start", "label", "said", "parakeet", "whisper", "act", "why", "heliox_speaking",
              "since_heliox_ended_s", "window_opened_by", "level_dbfs", "voiced_s", "mic"]  # fmt: skip
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# Building the cache (needs numpy and the recognisers) ---------------------------------------------------------------


def _level_dbfs(x: Any) -> float:
    import numpy as np

    from .conditions import speech_rms

    return round(20 * float(np.log10(max(speech_rms(x), 1e-7))), 1)


def _voiced_s(x: Any) -> float:
    import numpy as np

    frame = 320
    if len(x) < frame:
        return 0.0
    frames = x[: len(x) // frame * frame].reshape(-1, frame)
    levels = np.sqrt(np.mean(np.square(frames), axis=1))
    return round(float(np.sum(levels > 0.1 * levels.max())) * frame / 16000, 2)


class Recognisers:
    """The listener's own final-pass calls, on the CPU: Parakeet on the normalised 16 kHz samples, and the Whisper
    wake check (``_transcribe_whisper_text`` with language "en": base model, the vocabulary prompt, beam 5,
    temperature 0, the no-speech and hallucination filters)."""

    def __init__(self) -> None:
        import os

        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from unittest.mock import MagicMock

        from pilot.config import PilotConfig
        from pilot.system.voice import ContinuousVoiceListener

        config = PilotConfig.load()
        config.save = MagicMock()
        config.voice.whisper_model = "base"
        self.listener = ContinuousVoiceListener(wake_words=["hey heliox", "heliox"], config=config)
        self.parakeet = self.listener._parakeet()
        self.loop = asyncio.new_event_loop()

    def __call__(self, x: Any) -> tuple[str, str]:
        import os

        import pilot.system.voice as voice

        samples = voice._normalize_voice_audio(x)
        parakeet = self.parakeet.transcribe(samples) if self.parakeet is not None else ""
        path = voice._write_wav([voice._to_int16(x)], 16000)
        try:
            whisper = self.loop.run_until_complete(self.listener._transcribe_whisper_text(path, language="en"))
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        return parakeet, str(whisper or "").strip()


def build(root: str, redo: bool = False) -> Path:
    from . import wavio

    out = Path(root) / CACHE_NAME
    done = {} if redo or not out.exists() else {(u["stream"], u["t_start"]): u for u in load_cache(out)}
    recognise = Recognisers()
    rows: list[dict[str, Any]] = []
    for ref, said, label, speaking, mic in REAL_CLIPS:
        key = (f"real:{ref}", 0.0)
        if key not in done:
            parsed = audio_refs.parse_ref(ref)
            x = wavio.read(audio_refs.resolve(root, parsed), parsed.start, parsed.end)
            parakeet, whisper = recognise(x)
            done[key] = {"stream": key[0], "t_start": 0.0, "t_end": round(len(x) / 16000, 2), "label": label,
                         "said": said, "mic": mic, "voice": "real", "heliox_speaking_at_start": speaking,
                         "parakeet": parakeet, "whisper": whisper, "voiced_s": _voiced_s(x),
                         "level_dbfs": _level_dbfs(x)}  # fmt: skip
        rows.append(done[key])
    for meta in sorted((Path(root) / "synthetic" / "not_addressed").glob("block_*.json")):
        audio = wavio.read(meta.with_suffix(".wav"))
        stream = f"synthetic:{meta.stem}"
        for seg in json.loads(meta.read_text(encoding="utf-8"))["segments"]:
            key = (stream, seg["start"])
            if key not in done:
                x = audio[int(seg["start"] * 16000) : int(seg["end"] * 16000)]
                parakeet, whisper = recognise(x)
                done[key] = {"stream": stream, "t_start": seg["start"], "t_end": seg["end"],
                             "label": "not_addressed", "said": seg["text"], "kind": seg["kind"], "mic": "synthetic",
                             "voice": "synthetic", "heliox_speaking_at_start": False, "parakeet": parakeet,
                             "whisper": whisper, "voiced_s": _voiced_s(x), "level_dbfs": _level_dbfs(x)}  # fmt: skip
            rows.append(done[key])
        out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")  # saved per block
        print(f"{meta.stem}: cached", flush=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.conversation_bench.replay")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--audio", default=None)
    b.add_argument("--redo", action="store_true", help="recognise every utterance again")
    b.add_argument(
        "--real-folders",
        action="store_true",
        help="required: reads your real Heliox settings and models (the wake calibration is never saved)",
    )
    r = sub.add_parser("run")
    r.add_argument("--audio", default=None)
    r.add_argument("--decide", required=True, help="module:function, e.g. pilot.system.addressing:decide")
    r.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    root = audio_refs.audio_root(args.audio)
    if args.command == "build":
        from pilot.real_folders import require_real_folders_flag

        require_real_folders_flag("conversation_bench.replay build")
        print("cache:", build(root, args.redo))
        return 0
    rows = replay(load_cache(Path(root) / CACHE_NAME), load_decide(args.decide))
    out = Path(args.out) if args.out else Path(root) / "replay_table.tsv"
    write_table(rows, out)
    for label, counts in sorted(summary(rows).items()):
        print(f"{label:<14} " + ", ".join(f"{act} {counts.get(act, 0)}" for act in ACTS))
    print(json.dumps(score(rows)))
    print("table:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
