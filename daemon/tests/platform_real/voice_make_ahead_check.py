"""The local voice on a CPU-only runner, report only: is making fixed sentences ahead harmless to the recognisers, how
fast are pieces made, what does a reload cost, and which voice speaks a reply while the voice is released.

    python tests/platform_real/voice_make_ahead_check.py --audio <bench audio> [--gap 180] [--summary <file>]

Run from daemon/ on a runner that has fetched the speech set (ci_prefetch) and rendered the bench's synthetic voices.
Every Heliox folder is moved under HELIOX_BENCH_ROOT (use_scratch_folders), where the prefetch put the models.
Nothing is played, nothing is fetched (HF_HUB_OFFLINE=1), and the exit code is 0 unless the script itself breaks:
these are numbers to read, not a gate. Lines:

  REPORT recognisers/<name>: median and worst ms for a 3 s synthetic sentence, alone and with Kokoro (4 threads, as the
         voice worker runs) making a sentence beside it all the while, and the slowdown.
  REPORT make-ahead/piece: median and worst ms to make one gathered fixed piece; pieces a minute at the store's rest
         factor, at the measured time and at twice it.
  REPORT reload/<first|after-gap>: the voice worker's cold start split (process start, imports, model load, first
         synthesis) from scripts/measure_kokoro_startup.py's stages, run twice GAP seconds apart; the second is the
         reload a person meets a few minutes after a release, with the model files in the file cache.
  REPORT released-reply: which voice speaks a reply asked for while the worker is released (the system voice when
         the local voice is still loading, the store when every piece is kept), and how long until it could start.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

DAEMON = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DAEMON))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from pilot.real_folders import use_scratch_folders  # noqa: E402

ROOT = Path(os.environ.get("HELIOX_BENCH_ROOT") or tempfile.mkdtemp(prefix="heliox-make-ahead-")).resolve()
use_scratch_folders(ROOT)

import numpy as np  # noqa: E402
import psutil  # noqa: E402

REPEATS = 5
lines: list[str] = []


def report(line: str) -> None:
    lines.append(f"REPORT {line}")
    print(f"REPORT {line}", flush=True)


def ms(values: list[float]) -> str:
    return f"median {statistics.median(values) * 1000:.0f} ms, worst {max(values) * 1000:.0f} ms"


def machine() -> str:
    memory = psutil.virtual_memory()
    return (
        f"{os.cpu_count()} logical cores, {memory.total / 1024**3:.1f} GB, {memory.available / 1024**3:.1f} GB free, "
        f"CPU {psutil.cpu_percent(1.0):.0f} %"
    )


def recognisers(audio_root: Path) -> None:
    import torch

    from benchmarks.conversation_bench import wavio
    from pilot.config import DATA_DIR
    from pilot.system import kokoro_tts, voice
    from pilot.system.parakeet_asr import ParakeetTranscriber, model_dir

    clips = sorted((audio_root / "synthetic").rglob("*.wav"))
    if not clips:
        report("recognisers: no synthetic clip under the bench audio; not measured")
        return
    clip = wavio.read(str(clips[0]))[: 3 * 16000].astype(np.float32)
    path = ROOT / "make-ahead-clip.wav"
    wavio.write(path, clip)
    voice._faster_whisper_runtime = lambda model_name="base": ("cpu", "int8")
    parakeet = ParakeetTranscriber(model_dir("", DATA_DIR), threads=4)
    tasks = {"whisper-base": lambda: asyncio.run(voice._transcribe_faster_whisper(str(path), "en", "base"))}
    if parakeet.load():
        tasks["parakeet"] = lambda: parakeet.transcribe(clip)
    else:
        report("recognisers/parakeet: not on this runner; Whisper only")
    torch.set_num_threads(4)
    kokoro_tts.configure(device="cpu")
    kokoro_tts.use_cached_weights_offline("af_heart")
    kokoro_tts._generate("Okay.", "af_heart")

    def timed(task) -> list[float]:
        task()  # its first call, not counted
        took = []
        for _ in range(REPEATS):
            started = time.perf_counter()
            task()
            took.append(time.perf_counter() - started)
        return took

    alone = {name: timed(task) for name, task in tasks.items()}
    stop = threading.Event()

    def keep_making() -> None:
        while not stop.is_set():
            kokoro_tts._generate("You have three things on your list today.", "af_heart")

    maker = threading.Thread(target=keep_making, daemon=True)
    maker.start()
    time.sleep(0.3)
    beside = {name: timed(task) for name, task in tasks.items()}
    stop.set()
    maker.join()
    for name in tasks:
        slower = statistics.median(beside[name]) / max(1e-6, statistics.median(alone[name]))
        report(f"recognisers/{name}: alone {ms(alone[name])}; beside Kokoro {ms(beside[name])}; {slower:.2f} x")


def make_ahead() -> None:
    from pilot.system import fixed_speech, kokoro_tts, local_tts_worker

    pieces = fixed_speech.gather()
    sample = list(pieces[:: max(1, len(pieces) // 20)])[:20]
    took = []
    for text in sample:
        started = time.perf_counter()
        kokoro_tts._generate(text, "af_heart")
        took.append(time.perf_counter() - started)
    t = statistics.median(took)
    rest = local_tts_worker.MAKE_AHEAD_REST_FACTOR

    def per_minute(seconds: float) -> float:
        return 60.0 / (seconds * (1 + rest))

    report(
        f"make-ahead/piece: {len(sample)} of {len(pieces)} gathered pieces, {ms(took)}; at a rest of {rest:g}: "
        f"{per_minute(t):.1f} pieces a minute, {per_minute(2 * t):.1f} at twice the time; the whole set in about "
        f"{len(pieces) / per_minute(t):.0f} minutes"
    )


def reload_split(label: str) -> float:
    """One cold start of the voice in a fresh process, split; returns when it ended (time.monotonic())."""
    spec = importlib.util.spec_from_file_location(
        "measure_kokoro_startup", DAEMON / "scripts" / "measure_kokoro_startup.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    keys = ("process_start_s", "import_torch_s", "import_kokoro_s", "model_load_s", "first_synthesis_short_s")
    with tempfile.TemporaryDirectory() as out:
        data = module._stages("cpu", out)
    parts = ", ".join(f"{key[:-2]} {data.get(key, 0.0):.2f} s" for key in keys)
    report(
        f"reload/{label}: {parts}; in all {data.get('wall_total_s', 0.0):.1f} s; resident {data.get('rss_mb', 0):.0f} MB"
    )
    return time.monotonic()


async def released_reply() -> None:
    """A reply asked for while the worker is released: within the grace it raises LocalVoiceNotReady (the system voice
    speaks it); a reply whose pieces are all kept plays from the store with no worker at all."""
    from pilot.system import local_tts_worker
    from pilot.system.local_tts_worker import LocalTTSWorkerManager, LocalVoiceNotReady, SpeechStore

    manager = LocalTTSWorkerManager()
    manager.configure(device="cpu")
    manager.head_cache = SpeechStore(ROOT / "voice_heads_check")
    played: list[float] = []

    async def play(_path: Path) -> None:
        played.append(time.perf_counter())

    started = time.perf_counter()
    try:
        await manager.speak("kokoro_tts", "It's 10:42.", "af_heart", None, play)
        report("released-reply/changing: spoken by the local voice (it loaded within the grace)")
    except LocalVoiceNotReady:
        report(
            f"released-reply/changing: the system voice speaks it; handed over after "
            f"{(time.perf_counter() - started) * 1000:.0f} ms (grace {local_tts_worker.LOAD_GRACE_SECONDS * 1000:.0f} ms)"
        )
    await manager.shutdown()
    signature = SpeechStore.signature("kokoro_tts", "af_heart")
    made = ROOT / "next-track.wav"
    from pilot.system import kokoro_tts

    audio, rate = kokoro_tts._generate("Next track.", "af_heart")
    import soundfile

    soundfile.write(made, audio, rate)
    manager.head_cache.add_fixed("Next track.", signature, made)
    played.clear()
    started = time.perf_counter()
    await manager.speak("kokoro_tts", "Next track.", "af_heart", None, play)
    report(
        f"released-reply/fixed: from the store, no worker started ({not manager.running}); playback handed the audio "
        f"after {(played[0] - started) * 1000:.0f} ms"
        if played
        else "released-reply/fixed: nothing played"
    )
    from pilot.system.voice import _tts_linux

    if sys.platform.startswith("linux"):
        out = ROOT / "system-voice.wav"
        started = time.perf_counter()
        result = await _tts_linux("It's 10:42.", None, 170, str(out))
        report(
            f"released-reply/system-voice: espeak-ng made the reply in {(time.perf_counter() - started) * 1000:.0f} ms ({result[:40]})"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--gap", type=float, default=180.0)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    report(f"machine: {machine()}")
    first_ended = None

    def first() -> None:
        # Before anything else in this run loads Kokoro; its files were just restored from the CI cache, so they may
        # already be in the file cache: read it as a start soon after the files were written, not a cold disk.
        nonlocal first_ended
        first_ended = reload_split("first")

    def after_gap() -> None:
        if first_ended is None:
            report("reload/after-gap: not measured (the first did not run)")
            return
        time.sleep(max(0.0, args.gap - (time.monotonic() - first_ended)))
        reload_split(f"after-gap ({args.gap:.0f} s after the first ended)")

    for name, step in (
        ("reload", first),
        ("recognisers", lambda: recognisers(args.audio)),
        ("make-ahead", make_ahead),
        ("released-reply", lambda: asyncio.run(released_reply())),
        ("reload", after_gap),
    ):
        try:
            step()
        except Exception as error:  # report only: one part failing leaves the others
            report(f"{name}: not measured ({type(error).__name__}: {str(error)[:200]})")
    report(f"machine at the end: {machine()}")
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as out:
            out.write("### The local voice on this runner (report only)\n\n```\n" + "\n".join(lines) + "\n```\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
