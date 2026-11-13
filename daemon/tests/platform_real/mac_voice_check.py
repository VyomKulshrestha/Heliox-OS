"""A spoken turn on a macOS runner: a synthetic voice played into a virtual audio device, heard by Heliox's own
listener through its real macOS capture path, for the mac-voice workflow.

    python mac_voice_check.py

The runner has BlackHole 2ch (a loopback driver: what is played to its output comes back on its input), installed
and loaded by the workflow (Homebrew cask, then coreaudiod restarted). The voice is the system's ``say`` written to a
WAV, or espeak-ng when this runner's ``say`` writes silence (as the macos-26 images did, speech_check.py), played to
BlackHole's output with PortAudio. Heliox listens on BlackHole's input with the settings a person would have, except
the recogniser, Faster-Whisper tiny.en on the CPU (the smallest, prefetched by the workflow), and turn ends by
silence (Smart Turn needs its own model). Before anything is judged, the runner's audio as it is after the restart
is printed (DIAG lines): Core Audio's devices, default input and output and their rates, and what PortAudio and
Heliox's own listing see, so a first failure says which of "the plug-in did not load", "it loads and gives silence"
and "Heliox does not list it" it is.

Checks, each one CHECK line (PASS, FAIL, or SKIP with its reason), in the shape of audio_check.py's Linux lines:

- ``audio-macos/device-listed``: Heliox's device listing (voice.list_audio_input_devices) lists BlackHole's input;
- ``audio-macos/microphone-permission``: what macOS says about the microphone for this process, without a prompt
  (a runner cannot click one). Not granted: the lines below SKIP with that reason;
- ``audio-macos/capture-hears-audio``: Heliox's recorder (voice._ContinuousRecorder, PortAudio on Core Audio) opens
  on BlackHole and, while a tone is played into it, returns frames at 16 kHz (within 10 %) well above silence.
  The tone and the speech clips are played at BlackHole's own rate (48 kHz), resampled; a DIAG line gives the
  samples received every half second before and during the tone, so a stall shows when it starts;
- ``voice-macos/wake-found``: the listener, started as Heliox starts it, takes "Hey Heliox, what is the weather in
  Paris tomorrow" as addressed by the wake phrase;
- ``voice-macos/request-heard``: the command it passes on is the request (similarity to the sentence at least 0.6,
  as kokoro_languages_check.py measures it);
- ``voice-macos/without-wake-not-taken``: "The weather in Paris was lovely last week" is heard and not taken.

Every wait is bounded; the exit code is the number of failed checks (a SKIP is not one). CI-only
(``ci_only.require_runner``): the workflow installs an audio driver and restarts coreaudiod, and this check plays
into and records from it.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import math
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

from ci_only import require_runner

require_runner(
    "plays into and records from an audio driver the workflow installs (BlackHole), with coreaudiod restarted"
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

failures = 0
WAKE_TURN = "Hey Heliox, what is the weather in Paris tomorrow?"
REQUEST = "what is the weather in Paris tomorrow"
NOT_ADDRESSED = "The weather in Paris was lovely last week."
RATE = 16000


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


def skip(name: str, reason: str) -> None:
    print(f"CHECK {name}: SKIP - {reason}", flush=True)


def similar(a: str, b: str) -> float:
    def keep(text: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", text.casefold()).strip()

    return difflib.SequenceMatcher(None, keep(a), keep(b)).ratio()


def diagnose(sd) -> None:
    """The runner's audio as it is now, before anything is judged."""
    try:
        profile = subprocess.run(
            ["system_profiler", "SPAudioDataType", "-json"], capture_output=True, text=True, timeout=60, check=False
        )
        items = json.loads(profile.stdout or "{}").get("SPAudioDataType", [])
        for group in items:
            for device in group.get("_items", []):
                print(
                    "DIAG Core Audio device: "
                    + json.dumps(
                        {
                            "name": device.get("_name"),
                            "input_channels": device.get("coreaudio_device_input"),
                            "output_channels": device.get("coreaudio_device_output"),
                            "rate": device.get("coreaudio_device_srate"),
                            "default_input": device.get("coreaudio_default_audio_input_device"),
                            "default_output": device.get("coreaudio_default_audio_output_device"),
                            "transport": device.get("coreaudio_device_transport"),
                        }
                    ),
                    flush=True,
                )
    except Exception as error:
        print(f"DIAG system_profiler: {type(error).__name__}: {error}", flush=True)
    try:
        for index, info in enumerate(sd.query_devices()):
            print(
                f"DIAG PortAudio device {index}: {info['name']!r}, in {info['max_input_channels']}, "
                f"out {info['max_output_channels']}, default rate {info['default_samplerate']}",
                flush=True,
            )
        print(f"DIAG PortAudio default (in, out): {tuple(sd.default.device)}", flush=True)
    except Exception as error:
        print(f"DIAG PortAudio: {type(error).__name__}: {error}", flush=True)
    plugin = Path("/Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver")
    print(f"DIAG BlackHole plug-in on disk: {plugin.exists()}", flush=True)


def blackhole_output(sd) -> int | None:
    for index, info in enumerate(sd.query_devices()):
        if "blackhole" in str(info["name"]).casefold() and int(info["max_output_channels"]) > 0:
            return index
    return None


def wav_samples(path: Path):
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        rate, width, channels = handle.getframerate(), handle.getsampwidth(), handle.getnchannels()
        raw = handle.readframes(handle.getnframes())
    if width != 2:
        return None, rate
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def make_clip(text: str, path: Path) -> tuple[Path | None, str]:
    """The sentence as speech in a WAV: ``say`` first, espeak-ng when this runner's ``say`` writes silence."""
    subprocess.run(
        ["say", "--file-format=WAVE", "--data-format=LEI16@22050", "-o", str(path), text],
        capture_output=True,
        timeout=60,
        check=False,
    )
    if path.exists():
        samples, _ = wav_samples(path)
        if samples is not None and len(samples) > 11025 and float(abs(samples).max()) > 0.05:
            return path, "say"
    espeak = shutil.which("espeak-ng")
    if espeak:
        subprocess.run([espeak, "-v", "en-us", "-s", "150", "-w", str(path), text], timeout=60, check=False)
        samples, _ = wav_samples(path) if path.exists() else (None, 0)
        if samples is not None and len(samples) > 11025 and float(abs(samples).max()) > 0.05:
            return path, "espeak-ng (this runner's say wrote silence)"
    return None, "neither say nor espeak-ng wrote speech"


def device_rate(sd, device: int) -> int:
    """The rate the device runs at. Playing at any other rate made Core Audio stop Heliox's input on the same device
    (round 1: frames stopped as the tone started, PortAudio "err='-10863' cannot do in current context")."""
    return int(round(float(sd.query_devices(device)["default_samplerate"])))


def resample(samples, rate: int, to_rate: int):
    import numpy as np

    if rate == to_rate or len(samples) == 0:
        return samples.astype(np.float32)
    count = int(round(len(samples) * to_rate / rate))
    return np.interp(np.arange(count) * rate / to_rate, np.arange(len(samples)), samples).astype(np.float32)


def play(sd, samples, rate: int, device: int) -> None:
    import numpy as np

    out_rate = device_rate(sd, device)
    samples = resample(samples, rate, out_rate)
    silence = np.zeros(int(out_rate * 1.0), dtype=np.float32)
    sd.play(np.concatenate([silence, samples, silence, silence]), out_rate, device=device, blocking=True)


async def capture(sd, voice, device_id: str, output: int) -> bool:
    import numpy as np

    recorder = voice._ContinuousRecorder(sample_rate=RATE, preferred_device=device_id)
    if not recorder.start():
        return check("audio-macos/capture-hears-audio", False, f"did not open: {recorder.last_error}")
    try:
        receiving = await recorder.wait_until_receiving(timeout=5.0)
        out_rate = device_rate(sd, output)
        print(f"DIAG the tone: 440 Hz at {out_rate} Hz, BlackHole's own rate; Heliox records at {RATE} Hz", flush=True)
        timeline: list[str] = []
        last = recorder.frames_received

        def mark(label: str) -> None:
            nonlocal last
            now = recorder.frames_received
            timeline.append(f"{label} +{(now - last) * recorder.frame_size}")
            last = now

        for step in (1, 2):
            await asyncio.sleep(0.5)
            mark(f"before {step * 0.5:.1f}s")
        start_frames, started = recorder.frames_received, time.monotonic()
        tone = (0.4 * np.sin(2 * math.pi * 440 * np.arange(out_rate * 2) / out_rate)).astype(np.float32)
        player = threading.Thread(target=lambda: sd.play(tone, out_rate, device=output, blocking=True))
        player.start()
        for step in range(1, 6):
            await asyncio.sleep(0.5)
            mark(f"tone {step * 0.5:.1f}s")
        player.join(timeout=5)
        # Samples Heliox received in each half second, before and while the tone plays (about 8000 is normal).
        print(f"DIAG samples received per 0.5 s: {', '.join(timeline)}", flush=True)
        elapsed = time.monotonic() - started
        rate = (recorder.frames_received - start_frames) * recorder.frame_size / elapsed if elapsed else 0.0
        heard = recorder.peak_frame_rms
        return check(
            "audio-macos/capture-hears-audio",
            receiving
            and "blackhole" in recorder.input_device_name.casefold()
            and heard > 0.05
            and abs(rate - RATE) <= RATE * 0.1,
            f"device {recorder.input_device_name!r}, frames arriving {receiving}, {rate:.0f} samples/s, "
            f"peak level {heard:.3f} while a tone played (silence is about 0)",
        )
    finally:
        recorder.stop()


async def spoken_turn(sd, voice, device_id: str, output: int, clips: dict) -> None:
    from pilot.config import PilotConfig

    config = PilotConfig()
    settings = config.voice
    settings.input_device = device_id
    settings.whisper_model = "tiny.en"
    settings.transcription_engine = "faster_whisper"
    settings.language = "en"
    settings.turn_detection = "silence"
    settings.final_pass = "whisper"
    commands: list[str] = []
    heard: list[tuple[str, dict]] = []

    async def on_command(text: str) -> None:
        commands.append(text)

    async def on_transcript(text, engine, endpoint_at, started, done, decision=None) -> None:
        heard.append((text, decision or {}))

    listener = voice.ContinuousVoiceListener(on_command=on_command, config=config)
    listener.on_transcript = on_transcript
    said = await listener.start()
    print(f"DIAG the listener: {said}", flush=True)
    if not listener._running:
        for name in ("wake-found", "request-heard", "without-wake-not-taken"):
            check(f"voice-macos/{name}", False, f"the listener did not start: {said}")
        return
    try:
        if listener._transcriber_warmup_task is not None:
            await asyncio.wait_for(asyncio.shield(listener._transcriber_warmup_task), 180)

        async def until(probe, seconds: float) -> bool:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if probe():
                    return True
                await asyncio.sleep(0.25)
            return bool(probe())

        samples, rate = clips["wake"]
        await asyncio.to_thread(play, sd, samples, rate, output)
        await until(lambda: bool(commands), 60)
        wake = [(text, decision) for text, decision in heard if decision.get("why") == "wake"]
        check("voice-macos/wake-found", bool(wake), f"heard {heard!r}")
        command = commands[0] if commands else ""
        score = similar(REQUEST, command)
        check("voice-macos/request-heard", score >= 0.6, f"passed on {command!r} (similarity {score:.2f})")

        before_heard, before_commands = len(heard), len(commands)
        samples, rate = clips["plain"]
        await asyncio.to_thread(play, sd, samples, rate, output)
        await until(lambda: len(heard) > before_heard, 60)
        after = heard[before_heard:]
        taken = len(commands) > before_commands or any(decision.get("act") for _, decision in after)
        check(
            "voice-macos/without-wake-not-taken",
            bool(after) and not taken,
            f"heard {after!r}, commands {commands[before_commands:]!r}",
        )
    finally:
        await listener.stop()


async def main() -> int:
    import sounddevice as sd

    from pilot.system import macos_permissions, voice

    diagnose(sd)
    devices = voice.list_audio_input_devices(sd, sample_rate=RATE)
    print(f"DIAG Heliox's listing: {devices}", flush=True)
    found = next((device for device in devices if "blackhole" in device["name"].casefold()), None)
    check("audio-macos/device-listed", found is not None, f"{found['id'] if found else 'no BlackHole input'}")
    permission = macos_permissions.microphone_state()
    granted = permission == macos_permissions.PermissionState.GRANTED
    print(f"CHECK audio-macos/microphone-permission: {'PASS' if granted else 'SKIP'} - this process: {permission}")
    output = blackhole_output(sd)
    later = (
        "audio-macos/capture-hears-audio",
        "voice-macos/wake-found",
        "voice-macos/request-heard",
        "voice-macos/without-wake-not-taken",
    )
    if found is None or output is None:
        for name in later:
            check(name, False, "not run: no BlackHole device (see the DIAG lines above)")
        return failures
    if not granted:
        for name in later:
            skip(name, f"the microphone is {permission} for this process and a runner cannot answer the prompt")
        return failures
    if not await capture(sd, voice, found["id"], output):
        for name in later[1:]:
            check(name, False, "not run: capture heard nothing")
        return failures
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp")
    clips = {}
    for key, text in (("wake", WAKE_TURN), ("plain", NOT_ADDRESSED)):
        path, maker = make_clip(text, folder / f"heliox-{key}.wav")
        print(f"DIAG the {key} clip: {maker}", flush=True)
        if path is None:
            for name in later[1:]:
                check(name, False, f"not run: {maker}")
            return failures
        clips[key] = wav_samples(path)
    await spoken_turn(sd, voice, found["id"], output, clips)
    return failures


if __name__ == "__main__":
    if sys.platform != "darwin":
        print("NOT RUN: this check is for macOS, and changed nothing here.")
        sys.exit(3)
    try:
        sys.exit(asyncio.run(main()))
    except Exception as error:  # every exit path prints a CHECK line
        check("voice-macos/check-itself", False, f"{type(error).__name__}: {error}")
        sys.exit(max(failures, 1))
