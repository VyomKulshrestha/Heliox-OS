"""Render the synthetic speech with the local Kokoro voice, on the CPU, into the audio folder (no network).

  python -m benchmarks.conversation_bench.synth --audio <folder> [--only scenarios|corpus|babble] [--blocks N]

Scenario clips: any step whose audio reference is ``synthetic/<voice>/<speed>/<name>.wav`` is rendered from the
step's ``text`` with that Kokoro voice and speed. The not-addressed corpus: ``synthetic/not_addressed/block_NN.wav``
with ``block_NN.json`` (each line's kind, voice, text, start and end), assembled from a pool of 160 unique
sentences rendered once (``lines/``) and replayed at 0.9, 1.0 or 1.1 times their speed and at different levels.
Babble for the open-office conditions: ``synthetic/babble.wav`` (two voices talking over each other). Files that exist are kept. Synthetic audio is
labelled synthetic everywhere and never scored with the owner's real voice.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path, PurePosixPath
from typing import Any

from . import audio as audio_refs
from . import conditions, corpus, wavio
from .scenarios import load_all

KOKORO_RATE = 24000
GAP_RANGE = (0.35, 1.4)  # seconds of silence between lines in a block


class Voice:
    def __init__(self) -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only: the bench never uses a GPU
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # no downloads: the voices must already be on disk
        import torch
        from kokoro import KPipeline

        torch.set_num_threads(int(os.environ.get("CONVERSATION_BENCH_SYNTH_THREADS", "4")))

        self.pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device="cpu")

    def say(self, text: str, voice: str, speed: float) -> Any:
        import numpy as np

        parts = [a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
                 for _, _, a in self.pipeline(text, voice=voice, speed=speed)]  # fmt: skip
        return wavio.to_rate(np.concatenate(parts).astype(np.float32), KOKORO_RATE)


def scenario_clips(root: str) -> list[tuple[Path, str, str, float]]:
    out = []
    for scenario in load_all():
        for step in scenario.steps:
            if (
                not step.say.startswith("synthetic/")
                or step.say.startswith("synthetic/not_addressed/")
                or not step.text
            ):
                continue
            ref = audio_refs.parse_ref(step.say)
            parts = PurePosixPath(ref.path).parts
            if len(parts) != 4:
                continue
            _, voice, speed, _name = parts
            out.append((Path(audio_refs.resolve(root, ref)), step.text, voice, float(speed)))
    return out


def render_scenarios(root: str, voice: Voice) -> int:
    made = 0
    for path, text, name, speed in scenario_clips(root):
        if path.exists():
            continue
        clip = voice.say(text, name, speed)
        wavio.write(path, clip / max(1e-6, float(abs(clip).max())) * 0.5)
        made += 1
        print("made", path.name, "<-", text, flush=True)
    return made


def pooled(folder: Path, voice: Voice, line: corpus.Line) -> Any:
    """One pool sentence, rendered once at normal speed and kept in ``lines/``."""
    import hashlib

    key = hashlib.sha1(f"{line.voice}|{line.text}".encode()).hexdigest()[:16]
    path = folder / "lines" / f"{key}.wav"
    if path.exists():
        return wavio.read(path)
    clip = voice.say(line.text, line.voice, 1.0)
    wavio.write(path, clip / max(1e-6, float(abs(clip).max())) * 0.5)
    return clip


def render_corpus(root: str, voice: Voice, limit: int | None = None) -> int:
    import numpy as np

    folder = Path(root) / "synthetic" / "not_addressed"
    rng = random.Random(5)
    made = 0
    for index, block in enumerate(corpus.blocks(corpus.lines())):
        if limit is not None and index >= limit:
            break
        wav, meta = folder / f"block_{index:02d}.wav", folder / f"block_{index:02d}.json"
        if wav.exists() and meta.exists():
            continue
        pieces, segments, cursor = [], [], 0.0
        for line in block:
            gap = np.zeros(int(rng.uniform(*GAP_RANGE) * 16000), dtype=np.float32)
            clip = conditions.time_stretch(pooled(folder, voice, line), line.speed)
            clip = clip / max(1e-6, float(np.abs(clip).max())) * rng.uniform(0.3, 0.6)
            pieces += [gap, clip]
            start = cursor + len(gap) / 16000
            cursor = start + len(clip) / 16000
            segments.append({"kind": line.kind, "voice": line.voice, "speed": line.speed, "text": line.text,
                             "start": round(start, 3), "end": round(cursor, 3)})  # fmt: skip
        wavio.write(wav, np.concatenate(pieces))
        speech = sum(s["end"] - s["start"] for s in segments)
        meta.write_text(json.dumps({"seconds": cursor, "speech_seconds": speech, "segments": segments}, indent=1))
        made += 1
        print(f"block {index:02d}: {cursor:.0f} s ({speech:.0f} s of speech)", flush=True)
    return made


def render_babble(root: str, voice: Voice) -> bool:
    import numpy as np

    path = Path(root) / "synthetic" / "babble.wav"
    if path.exists():
        return False
    rng = random.Random(9)
    tracks = []
    for speaker in ("bm_lewis", "af_bella"):
        clips = [voice.say(corpus._fill(rng.choice(corpus.MEETING), rng), speaker, 1.0) for _ in range(14)]
        tracks.append(np.concatenate([np.concatenate([c, np.zeros(2400, np.float32)]) for c in clips]))
    n = min(len(t) for t in tracks)
    wavio.write(path, (tracks[0][:n] + tracks[1][:n]) * 0.4)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audio", default=None)
    parser.add_argument("--only", choices=("scenarios", "corpus", "babble"), default=None)
    parser.add_argument("--blocks", type=int, default=None)
    args = parser.parse_args()
    root = audio_refs.audio_root(args.audio)
    voice = Voice()
    if args.only in (None, "scenarios"):
        print("scenario clips made:", render_scenarios(root, voice))
    if args.only in (None, "babble"):
        print("babble made:", render_babble(root, voice))
    if args.only in (None, "corpus"):
        print("corpus blocks made:", render_corpus(root, voice, args.blocks))


if __name__ == "__main__":
    main()
