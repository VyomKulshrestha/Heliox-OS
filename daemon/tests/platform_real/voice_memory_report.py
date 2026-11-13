"""What the voice stack costs in memory, stage by stage (CI: the voice-memory workflow, Windows, macOS, Linux; report
only, nothing passes or fails on a number).

    python tests/platform_real/voice_memory_report.py --inputs   # the job's own inputs first
    python tests/platform_real/voice_memory_report.py

Stages, each printed as a MEMORY line as soon as it is measured (resident memory, MB; "peak" while the stage ran,
sampled twice a second over the whole process tree, and "settled" five seconds after it):
1. started: a real daemon (python -m pilot.server), ready, voice off: its process tree;
2. the recognisers, one stage each so each one's share shows: Whisper base (faster-whisper; the default final pass
   and the wake check), then Parakeet (the final pass once its download is chosen), then the streaming decoder and
   Smart Turn (turn-taking); each loaded through the product's own loaders and given one second of silence, in this
   process (the listener loads them in the daemon's);
3. voice loaded: the local voice (Kokoro) through the product's own worker manager (its own process, counted);
4. after one spoken reply: one sentence made by that worker;
5. after ten idle minutes: the worker is released after tts_keep_loaded_seconds (300 s by default), so this is what
   staying idle costs.
The runners have no microphone or speakers, so nothing is played or heard; what loads is what the listener and the
voice load. Runs under the speech set's scratch root (HELIOX_BENCH_ROOT, fetched by ci_prefetch, checked by
SHA-256). The system's total and free memory are printed with each line.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

from ci_only import require_runner

require_runner("starts a daemon and loads the speech models")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pilot.real_folders import use_scratch_folders  # noqa: E402

use_scratch_folders(Path(os.environ["HELIOX_BENCH_ROOT"]))

import psutil  # noqa: E402
import real_daemon  # noqa: E402

CATALOGUE_IDS = ("parakeet-tdt-0.6b-v2-int8", "whisper-base", "streaming-zipformer-en", "smart-turn-v3")
MB = 1024**2


def tree_rss(pid: int) -> int:
    try:
        root = psutil.Process(pid)
        total = root.memory_info().rss
        for child in root.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                continue
        return total
    except psutil.Error:
        return 0


def tree_private(pid: int) -> int:
    """Memory the process tree holds for itself: private bytes on Windows (its resident set, the working set, is
    trimmed when idle: 40 MB for a loaded voice in run 43404791124), the unique set elsewhere."""
    try:
        processes = [psutil.Process(pid), *psutil.Process(pid).children(recursive=True)]
    except psutil.Error:
        return 0
    total = 0
    for process in processes:
        try:
            info = process.memory_full_info()
            total += int(getattr(info, "private", 0) or getattr(info, "uss", 0) or info.rss)
        except psutil.Error:
            continue
    return total


class Sampler:
    """The peak of a process tree's resident memory, twice a second, from a thread."""

    def __init__(self, pid: int) -> None:
        self.pid, self.peak, self._stop = pid, 0, threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, tree_rss(self.pid))
            self._stop.wait(0.5)

    def reset(self) -> None:
        self.peak = tree_rss(self.pid)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def line(stage: str, peak: int, settled: int, **extra) -> None:
    memory = psutil.virtual_memory()
    record = {
        "stage": stage,
        "peak_mb": round(peak / MB),
        "settled_mb": round(settled / MB),
        "private_mb": round(tree_private(os.getpid() if extra.get("process") != "daemon" else extra.pop("pid")) / MB),
        "system_total_mb": round(memory.total / MB),
        "system_available_mb": round(memory.available / MB),
        **extra,
    }
    print("MEMORY " + json.dumps(record), flush=True)


def inputs() -> int:
    from pilot.system import speech_models

    failed = 0
    for model_id in CATALOGUE_IDS:
        ok = speech_models.installed(speech_models.BY_ID[model_id])
        failed += not ok
        print(f"CHECK voice-memory/inputs/model-{model_id}: {'PASS' if ok else 'FAIL'} - installed and checked")
    try:
        from huggingface_hub import scan_cache_dir

        from pilot.system import kokoro_tts

        ok = any(repo.repo_id == kokoro_tts.REPO_ID for repo in scan_cache_dir().repos)
    except Exception as error:  # noqa: BLE001 - said, then it fails
        ok = False
        print(f"INFO kokoro cache: {type(error).__name__}: {error}")
    failed += not ok
    print(f"CHECK voice-memory/inputs/voice-kokoro: {'PASS' if ok else 'FAIL'} - Kokoro's weights cached", flush=True)
    return 1 if failed else 0


async def started_daemon() -> None:
    real_daemon.write_config()
    daemon = real_daemon.start()
    session = None
    sampler = Sampler(daemon.pid)
    try:
        session = await real_daemon.connect(300)
        ready, _status = await session.wait_ready(300)
        await asyncio.sleep(5)
        line("started", sampler.peak, tree_rss(daemon.pid), ready=ready, process="daemon", pid=daemon.pid)
    finally:
        sampler.stop()
        await real_daemon.stop(daemon, session)


async def voice_stages(idle_minutes: float) -> None:
    import numpy as np

    from pilot.config import DATA_DIR, PilotConfig
    from pilot.system import local_tts_worker, voice
    from pilot.system.parakeet_asr import ParakeetTranscriber, model_dir

    me = os.getpid()
    sampler = Sampler(me)
    folder = Path(tempfile.mkdtemp(prefix="voice-memory-"))
    try:
        line("this process before loading", sampler.peak, tree_rss(me), process="report")

        # Each recogniser in its own stage, so its share is its own: Whisper base (the default final pass and the
        # wake check), then Parakeet (the final pass once its download is chosen), then the streaming decoder and
        # Smart Turn (turn-taking), each loaded through the product's own loaders and given one second of silence.
        silence_samples = np.zeros(16000, dtype=np.float32)
        sampler.reset()
        whisper = voice._get_faster_whisper_model("base")
        # Samples, as the listener hands them (voice._transcribe_faster_whisper): given a path, faster-whisper 1.2
        # decodes through PyAV, and PyAV 19 refuses its metadata_errors argument (run 43394164459, all three systems).
        list(whisper.transcribe(silence_samples)[0])
        await asyncio.sleep(5)
        line("Whisper base loaded and warm", sampler.peak, tree_rss(me), process="report")

        sampler.reset()
        parakeet = ParakeetTranscriber(model_dir("", DATA_DIR))
        parakeet_ok = parakeet.load()
        if parakeet_ok:
            parakeet.transcribe(silence_samples)
        await asyncio.sleep(5)
        line("Parakeet loaded and warm", sampler.peak, tree_rss(me), parakeet=parakeet_ok, process="report")

        from pilot.system.streaming_asr import StreamingTranscriber
        from pilot.system.turn_detector import SmartTurnDetector

        sampler.reset()
        streaming = StreamingTranscriber()
        streaming_ok = streaming.load() and streaming.warm()
        turn = SmartTurnDetector()
        turn_ok = turn.load()
        if turn_ok:
            turn.probability(silence_samples)
        await asyncio.sleep(5)
        line(
            "streaming decoder and Smart Turn loaded and warm",
            sampler.peak,
            tree_rss(me),
            streaming=bool(streaming_ok),
            smart_turn=bool(turn_ok),
            process="report",
        )

        config = PilotConfig()
        local_tts_worker.configure_local_voice(config.voice)
        manager = local_tts_worker.manager()
        manager.light_reason = None  # measure the local voice whatever the runner's memory
        sampler.reset()
        started = time.monotonic()
        await manager.ensure_ready("kokoro_tts", "af_heart")
        loaded_s = round(time.monotonic() - started, 1)
        await asyncio.sleep(5)
        line("voice loaded", sampler.peak, tree_rss(me), load_seconds=loaded_s, process="report + voice worker")

        sampler.reset()
        started = time.monotonic()
        await manager.synthesize(
            "kokoro_tts", "Here is the answer to your question, said once.", "af_heart", folder / "reply.wav", folder
        )
        reply_s = round(time.monotonic() - started, 2)
        await asyncio.sleep(5)
        line("after one spoken reply", sampler.peak, tree_rss(me), reply_seconds=reply_s, process="report + voice")

        sampler.reset()
        await asyncio.sleep(idle_minutes * 60)
        line(
            f"after {idle_minutes:g} idle minutes",
            sampler.peak,
            tree_rss(me),
            voice_worker_running=manager.running,
            keep_loaded_seconds=manager.keep_loaded_seconds,
            # Which rule decided (keep-by-memory, voice.tts_release): kept past keep_loaded_seconds while the computer
            # has the memory, released by the clock otherwise; and every release the manager logged.
            release_rule="clock" if manager.keep_while is None else "memory",
            kept_by_memory=bool(manager.keep_while is not None and manager._kept_by_memory()),
            releases=[reason for kind, reason, _at in getattr(manager, "reload_log", []) if kind == "released"],
            process="report + voice",
        )
        await manager.shutdown()
    finally:
        sampler.stop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", action="store_true")
    parser.add_argument("--idle-minutes", type=float, default=10.0)
    args = parser.parse_args()
    if args.inputs:
        return inputs()
    real_daemon.time_limit("voice-memory", 40 * 60 + args.idle_minutes * 60)
    try:
        asyncio.run(started_daemon())
        asyncio.run(voice_stages(args.idle_minutes))
    except BaseException as error:  # noqa: BLE001 - whatever ends the run is said
        import traceback

        traceback.print_exc(file=sys.stdout)
        print(f"CHECK voice-memory/ran: FAIL - {type(error).__name__}: {error}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
