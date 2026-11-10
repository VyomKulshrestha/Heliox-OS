"""Measure Kokoro speech-worker cold start, warm synthesis and memory (opt-in, run by hand).

Usage, from daemon/:
    .venv/Scripts/python.exe scripts/measure_kokoro_startup.py [--device cpu|cuda|auto] [--out DIR]

Writes WAV files to a temporary directory and never plays audio. Needs Kokoro and its
weights already in the Hugging Face cache; it does not download anything (offline mode).

Measurements (add --first-audio for the third):
  stages  one fresh child process that times interpreter start, imports, model load and the
          first synthesis separately, then warm syntheses, then idle RSS and GPU memory.
  worker  the real pilot.system.local_tts_worker protocol: time from spawn to "ready", then
          time from request to WAV for a short and a long sentence.
  first   time from speak() to the first audio handed to playback, cold, warm and prewarmed.
  stop    how fast playback ends after sounddevice.stop() and after a stream abort (on silence).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SHORT = "Okay."
LONG = (
    "I found three files from last week that match, and the largest one is the quarterly report. "
    "Do you want me to open it, or move all three into the archive folder?"
)


def _child(device: str, out_dir: str, started_at: float) -> None:
    timings: dict[str, float] = {"process_start_s": time.time() - started_at}
    t = time.perf_counter()
    import numpy  # noqa: F401
    import torch

    timings["import_torch_s"] = time.perf_counter() - t
    t = time.perf_counter()
    from kokoro import KModel, KPipeline  # noqa: F401

    timings["import_kokoro_s"] = time.perf_counter() - t
    t = time.perf_counter()
    kwargs = {} if device == "auto" else {"device": device}
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", **kwargs)
    timings["model_load_s"] = time.perf_counter() - t
    import soundfile

    def synth(text: str, name: str) -> float:
        start = time.perf_counter()
        chunks = [a for _g, _p, a in pipeline(text, voice="af_heart", speed=1.0, split_pattern=r"\n+")]
        audio = numpy.concatenate([c.detach().cpu().numpy() if hasattr(c, "detach") else c for c in chunks])
        soundfile.write(str(Path(out_dir) / name), audio, 24000)
        return time.perf_counter() - start

    timings["first_synthesis_short_s"] = synth(SHORT, "first.wav")
    timings["warm_short_s"] = min(synth(SHORT, f"short{i}.wav") for i in range(3))
    timings["warm_long_s"] = min(synth(LONG, f"long{i}.wav") for i in range(2))
    import psutil

    timings["rss_mb"] = psutil.Process().memory_info().rss / 2**20
    timings["private_mb"] = psutil.Process().memory_info().private / 2**20 if os.name == "nt" else 0.0
    timings["device"] = str(pipeline.model.device) if pipeline.model is not None else "none"
    if torch.cuda.is_available() and timings["device"].startswith("cuda"):
        timings["cuda_allocated_mb"] = torch.cuda.memory_allocated() / 2**20
        timings["cuda_reserved_mb"] = torch.cuda.memory_reserved() / 2**20
    print(json.dumps(timings), flush=True)


def _stages(device: str, out_dir: str, online: bool = False) -> dict[str, float]:
    env = dict(os.environ) if online else {**os.environ, "HF_HUB_OFFLINE": "1"}
    started = time.time()
    result = subprocess.run(
        [sys.executable, __file__, "--child", "--device", device, "--out", out_dir, "--started", repr(started)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(__file__).resolve().parents[1]),
        check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    if result.returncode != 0 or not lines:
        raise SystemExit(f"stage child failed: {result.stderr[-2000:]}")
    data = json.loads(lines[-1])
    data["wall_total_s"] = time.time() - started
    return data


def _tree_rss_mb(pid: int) -> float:
    import psutil

    root = psutil.Process(pid)
    return sum(p.memory_info().rss for p in [root, *root.children(recursive=True)]) / 2**20


def _spawn_worker(device: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "pilot.system.local_tts_worker",
            "--device",
            device,
            "--engine",
            "kokoro_tts",
            "--serve",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=str(Path(__file__).resolve().parents[1]),
    )


def _ask(proc: subprocess.Popen[bytes], payload: dict[str, str]) -> dict[str, object]:
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write((json.dumps(payload) + "\n").encode())
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


def _say(proc: subprocess.Popen[bytes], out_dir: str, label: str, text: str) -> float:
    text_file = Path(out_dir) / f"{label}.txt"
    text_file.write_text(text, encoding="utf-8")
    started = time.perf_counter()
    reply = _ask(
        proc, {"text_file": str(text_file), "voice": "af_heart", "output_file": str(Path(out_dir) / f"{label}.wav")}
    )
    if reply.get("status") != "ok":
        raise SystemExit(f"synthesis failed: {reply}")
    return time.perf_counter() - started


def _worker(out_dir: str, device: str) -> dict[str, float]:
    """The real worker protocol, old path (first request loads the model) and new path (load op, then speak)."""
    result: dict[str, float] = {}
    started = time.perf_counter()
    old = _spawn_worker(device)
    _say(old, out_dir, "old_first", SHORT)
    result["before_cold_spawn_to_first_wav_s"] = time.perf_counter() - started
    assert old.stdin is not None
    old.stdin.close()
    old.wait(timeout=30)

    started = time.perf_counter()
    proc = _spawn_worker(device)
    reply = _ask(proc, {"op": "load", "voice": "af_heart"})
    result["after_spawn_to_ready_s"] = time.perf_counter() - started
    result["after_ready_ok"] = float(reply.get("status") == "ok")
    result["after_first_short_s"] = _say(proc, out_dir, "short0", SHORT)
    result["after_cold_spawn_to_first_wav_s"] = time.perf_counter() - started
    result["warm_short_s"] = min(_say(proc, out_dir, f"short{i}", SHORT) for i in range(1, 4))
    result["warm_long_s"] = min(_say(proc, out_dir, f"long{i}", LONG) for i in range(2))
    result["warm_first_sentence_of_long_s"] = _say(
        proc, out_dir, "first_sentence", LONG.split("? ")[0].split(". ")[0] + "."
    )
    result["loaded_rss_mb"] = _tree_rss_mb(proc.pid)
    time.sleep(10)
    result["loaded_rss_after_10s_idle_mb"] = _tree_rss_mb(proc.pid)
    assert proc.stdin is not None
    proc.stdin.close()
    proc.wait(timeout=30)
    return result


def _stop_latency() -> dict[str, float]:
    """How fast playback ends after sounddevice.stop() (what voice.stop_speaking uses) and after a stream abort.

    Measured on 3 s of silence, so nothing is audible.
    """
    import threading

    import numpy as np
    import sounddevice as sd

    result: dict[str, float] = {}
    for label, stop in (("sd_stop", sd.stop), ("stream_abort", lambda: sd.get_stream().abort())):
        finished = threading.Event()

        def play(done: threading.Event = finished) -> None:
            sd.play(np.zeros(3 * 24000, dtype=np.float32), 24000)
            sd.wait()
            done.set()

        thread = threading.Thread(target=play)
        thread.start()
        time.sleep(0.5)
        started = time.perf_counter()
        stop()
        finished.wait(5)
        result[f"{label}_to_playback_ended_ms"] = (time.perf_counter() - started) * 1000
        thread.join(5)
    return result


async def _first_audio(device: str) -> dict[str, float]:
    """Time from "Heliox has something to say" to the first audio being handed to playback, through the manager."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pilot.system.local_tts_worker import LocalTTSWorkerManager

    manager = LocalTTSWorkerManager(device=device)
    result: dict[str, float] = {}
    first: list[float] = []

    async def play(path: Path) -> None:
        if not first:
            first.append(time.perf_counter())

    async def measure(label: str, text: str) -> None:
        first.clear()
        started = time.perf_counter()
        await manager.speak("kokoro_tts", text, "af_heart", None, play)
        result[f"{label}_to_first_audio_s"] = first[0] - started
        result[f"{label}_whole_reply_made_s"] = time.perf_counter() - started

    await measure("cold", SHORT)
    await measure("warm_short", SHORT)
    await measure("warm_long", LONG)
    await manager.shutdown()

    started = time.perf_counter()
    manager.prewarm("kokoro_tts", "af_heart")
    await asyncio.sleep(3.0)  # the person is still talking / being transcribed
    await measure("prewarmed_3s_ahead", SHORT)
    result["prewarm_3s_ahead_total_s"] = time.perf_counter() - started
    await manager.shutdown()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--out", default="")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--started", type=float, default=0.0)
    parser.add_argument("--skip-worker", action="store_true")
    parser.add_argument("--skip-stages", action="store_true")
    parser.add_argument("--first-audio", action="store_true", help="also time speak() to first audio via the manager")
    parser.add_argument("--online", action="store_true", help="let huggingface_hub check the network (old behaviour)")
    args = parser.parse_args()
    if args.child:
        _child(args.device, args.out, args.started)
        return 0
    with tempfile.TemporaryDirectory(prefix="heliox-tts-measure-") as out_dir:
        if not args.skip_stages:
            print("stages", json.dumps(_stages(args.device, out_dir, args.online), indent=1))
        if not args.skip_worker:
            print("worker", json.dumps(_worker(out_dir, args.device), indent=1))
        if args.first_audio:
            print("first_audio", json.dumps(asyncio.run(_first_audio(args.device)), indent=1))
        print("stop", json.dumps(_stop_latency(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
