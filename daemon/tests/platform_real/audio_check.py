"""Heliox's audio on a real PulseAudio server, for the Linux ``platform-readers`` job (PulseAudio already running).

    python audio_check.py playback   # a tone through Heliox's output line reaches the default sink
    python audio_check.py follow     # the default microphone changes during capture; Heliox follows it

playback: a null sink becomes the default output; ``voice._Playback`` plays one second of a 440 Hz tone on the
shared output line (PortAudio, through ALSA's ``pulse`` device), while ``parec`` records the sink's monitor. The
recording must hold the tone (its level well above silence).

follow: two null sources with different names; capture starts on the first through ``voice._ContinuousRecorder``
with the setting on "auto". The default is switched to the second while capturing; ``audio_health.CaptureWatchdog``
must decide "reopen" for "default_changed" from what it reads live, and after ``recorder.reopen()`` PulseAudio must
show Heliox's recording stream on the second source. This is the body of the listener's capture watch
(``ContinuousVoiceListener._watch_capture``), run by the check against real devices.

Each check prints one ``CHECK`` line; the exit code is the number of failed checks.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

failures = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


def pactl(*args: str) -> str:
    done = subprocess.run(["pactl", *args], capture_output=True, text=True, timeout=10, check=False)
    if done.returncode != 0:
        print(f"pactl {' '.join(args)}: exit {done.returncode}: {done.stderr.strip()}")
    return done.stdout.strip()


def load_null_source(name: str, description: str) -> None:
    """A null source with its own description. module-null-source takes ``description=`` (unlike the null sink's
    ``sink_properties=``): CI on e51aead loaded nothing with ``source_properties=``, the default stayed the earlier
    step's "Null Input", and every follow check failed for that reason."""
    if pactl("load-module", "module-null-source", f"source_name={name}", f"description={description}"):
        return
    pactl(
        "load-module",
        "module-null-source",
        f"source_name={name}",
        f"source_properties=device.description={description}",
    )


def playback() -> None:
    import array

    import numpy as np

    from pilot.system import voice

    pactl(
        "load-module",
        "module-null-sink",
        "sink_name=heliox_speaker",
        "sink_properties=device.description=HelioxSpeaker",
    )
    pactl("set-default-sink", "heliox_speaker")
    raw = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "heliox-monitor.raw"
    with raw.open("wb") as out:
        recorder = subprocess.Popen(
            ["parec", "-d", "heliox_speaker.monitor", "--format=s16le", "--rate=24000", "--channels=1"], stdout=out
        )
        time.sleep(0.5)
        rate = 24000
        tone = (0.4 * np.sin(2 * math.pi * 440 * np.arange(rate) / rate)).astype(np.float32)
        try:
            voice._Playback(tone, rate).play()
            played = "played"
        except Exception as exc:
            played = f"{type(exc).__name__}: {exc}"
        time.sleep(0.8)
        recorder.terminate()
        recorder.wait(timeout=5)
    samples = array.array("h", raw.read_bytes()[: (raw.stat().st_size // 2) * 2])
    peak = max((abs(s) for s in samples), default=0)
    rms = math.sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0.0
    check(
        "audio-linux/playback-reaches-the-default-sink",
        peak > 3000 and rms > 500,
        f"{played}; {len(samples) / 24000:.1f} s recorded from the monitor, peak {peak}, rms {rms:.0f}; "
        f"output line on {voice._output_line.device_name!r}",
    )
    voice._output_line.close()


def source_index(name: str) -> str:
    for line in pactl("list", "short", "sources").splitlines():
        parts = line.split("\t")
        if len(parts) > 1 and parts[1] == name:
            return parts[0]
    return ""


def follow() -> None:
    from pilot.system import audio_health, voice

    for name, description in (("heliox_follow_a", "HelioxMicA"), ("heliox_follow_b", "HelioxMicB")):
        load_null_source(name, description)
    print(f"sources: {pactl('list', 'short', 'sources').splitlines()}")
    pactl("set-default-source", "heliox_follow_a")
    time.sleep(0.5)
    recorder = voice._ContinuousRecorder(preferred_device="auto")
    recorder._loop = None
    opened = recorder._open_capture()
    try:
        before = audio_health.default_microphone_name()
        check(
            "audio-linux/capture-opens-on-the-default",
            opened and before == "HelioxMicA",
            f"{before!r} {recorder.last_error}",
        )
        watchdog = audio_health.CaptureWatchdog()
        watchdog.opened(before, recorder.last_error)
        pactl("set-default-source", "heliox_follow_b")
        time.sleep(1.5)
        state = audio_health.CaptureState(
            frames=int(recorder.frames_received),
            error=str(recorder.last_error or ""),
            zero_seconds=float(recorder.zero_seconds),
            active=bool(recorder.is_active),
            default_name=audio_health.default_microphone_name(),
            auto=True,
            using_fallback=bool(recorder.using_fallback_input),
        )
        decision = watchdog.check(state)
        check(
            "audio-linux/watchdog-sees-the-new-default",
            decision.action == "reopen" and decision.reason == "default_changed",
            f"{decision}; default now {state.default_name!r}",
        )
        reopened = recorder.reopen()
        time.sleep(1.0)
        outputs = pactl("list", "source-outputs")
        index = source_index("heliox_follow_b")
        check(
            "audio-linux/capture-follows-to-the-new-default",
            reopened and bool(index) and f"Source: {index}" in outputs,
            f"reopened {reopened}, source index {index!r}, device {recorder.input_device_name!r}",
        )
    finally:
        recorder._close_capture()


def main(argv: list[str]) -> int:
    if argv and argv[0] == "follow":
        follow()
    else:
        playback()
    return failures


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
