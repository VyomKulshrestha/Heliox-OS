"""A spoken turn on a macOS runner: a synthetic voice played into a virtual audio device, heard by Heliox's own
listener through its real macOS capture path, for the mac-voice workflow.

    python mac_voice_check.py <folder>
    python mac_voice_check.py <folder> --speak-through blackhole|default

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
  The tone and the speech clips are played at BlackHole's own rate (48 kHz), resampled, from a separate Python
  process: in rounds 1 and 2 an output stream opened inside this process stopped Heliox's input on the same device
  (PortAudio "err='-10863' cannot do in current context"), and a voice reaches a microphone from outside Heliox
  anyway. A DIAG line gives the samples received every half second before, during and after the tone; a last DIAG
  repeats the tone from this process, counted until 2 s after it ends, to show whether the input comes back;
- ``voice-macos/wake-found``: the listener, started as Heliox starts it, takes "Hey Heliox, what is the weather in
  Paris tomorrow" as addressed by the wake phrase;
- ``voice-macos/request-heard``: the command it passes on is the request (similarity to the sentence at least 0.6,
  as kokoro_languages_check.py measures it);
- ``voice-macos/without-wake-not-taken``: "The weather in Paris was lovely last week" is heard and not taken.

``--speak-through`` measures what a person whose microphone and speaker are one Core Audio device (a USB headset)
would meet: Heliox's listener on BlackHole's input while Heliox says one sentence through its own two voice paths,
the local voice's shared output stream inside the process (voice._play_local_tts_file) and the system voice
(voice.speak, which runs say). ``blackhole`` makes BlackHole the system default output first (SwitchAudioSource,
Homebrew switchaudio-osx), so Heliox speaks into the device it listens on; ``default`` leaves the runner's own
output, the control. The default output is read first and put back in a finally block. For each path, a DIAG line
gives the samples per half second before, during and for 5 s after the sentence, and three CHECK lines say whether
frames kept arriving while Heliox spoke (what barge-in needs), whether they arrived again within 3 s after it
stopped, and, after a stall, whether the microphone watchdog said so and the microphone worked again (its own log
lines, with the time). Round 4 adds, on the local voice: a second sentence 2 s after the first on the line as it
is, and a third at another rate (22050 Hz), which makes the output line reopen, each with its own lines and a DIAG
of the line's state; and, with Heliox silent and its line open, the tone from another process, which must keep the
input arriving and be heard (an open line must not shut other programs out). The control (``default``) plays a
quiet tone into BlackHole from another process throughout, so its input is never exact zeros and the watchdog's
silent-input rule does not fire. Since the same-device fix (a872dc6) the ``blackhole`` run exits with the number of
same-device gating lines (GATED) that failed or never printed, and prints a SUMMARY line; the experiment, the cut,
the default-output moves and the control (``default``, which always exits 0) stay measurements.

Every wait is bounded; the exit code is the number of failed checks (a SKIP is not one). CI-only
(``ci_only.require_runner``): the workflow installs an audio driver and restarts coreaudiod, and this check plays
into and records from it.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import json
import logging
import math
import re
import shutil
import subprocess
import sys
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


SAME_DEVICE = "voice-macos-speaking/blackhole/"
# The same-device lines that decide the job since the fix proved itself (a872dc6): frames while and after Heliox
# speaks in the four speaking cases, the line staying open at another rate, and frames arriving while another
# program plays with the line open (that case has no speaking, so its frames line is the one that counts).
GATED = tuple(
    SAME_DEVICE + line
    for line in (
        *(
            f"{case}/frames-{when}-speaking"
            for case in ("local-voice", "local-voice-again", "local-voice-other-rate", "system-voice")
            for when in ("while", "after")
        ),
        "local-voice-other-rate/no-new-open",
        "other-program-with-line-open/frames-arriving",
    )
)
# Lines that mean the same-device cases could not run at all; they fail the job too.
GATED_IF_FAILED = tuple(SAME_DEVICE + line for line in ("listener", "sentence", "check-itself"))
SAME_DEVICE_LINES: dict[str, bool] = {}  # every same-device CHECK printed, PASS or FAIL


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    if name.startswith(SAME_DEVICE):
        SAME_DEVICE_LINES[name] = ok
    return ok


def same_device_verdict() -> int:
    """The summary of the same-device lines and how many gating lines failed or never printed."""
    failed = [name for name in GATED if not SAME_DEVICE_LINES.get(name, False)]
    failed += [name for name in GATED_IF_FAILED if SAME_DEVICE_LINES.get(name) is False]
    passed = sum(1 for ok in SAME_DEVICE_LINES.values() if ok)
    gated_passed = sum(1 for name in GATED if SAME_DEVICE_LINES.get(name))
    print(
        f"SUMMARY same device: {passed} of {len(SAME_DEVICE_LINES)} lines passed; gating {gated_passed} of "
        f"{len(GATED)} passed{'; failed or missing: ' + ', '.join(failed) if failed else ''}",
        flush=True,
    )
    return len(failed)


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


# Plays a 16-bit WAV to the first output whose name holds argv[2], from a process of its own: a voice or another
# app reaches the microphone from outside Heliox's process, and in round 2 an output stream opened inside the same
# process stopped Heliox's input on the same device (PortAudio "err='-10863' cannot do in current context").
PLAYER = """
import sys
import wave

import numpy as np
import sounddevice as sd

path, name = sys.argv[1], sys.argv[2].casefold()
with wave.open(path, "rb") as handle:
    rate = handle.getframerate()
    data = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
devices = sd.query_devices()
device = next(
    index for index, info in enumerate(devices)
    if name in str(info["name"]).casefold() and int(info["max_output_channels"]) > 0
)
sd.play(data, rate, device=device, blocking=True)
"""


def write_wav(path: Path, samples, rate: int) -> Path:
    import numpy as np

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return path


def play_elsewhere(samples, rate: int, out_rate: int, path: Path) -> str:
    """Play ``samples`` to BlackHole at its own rate from a separate Python process, with a second of silence before
    and two after. Returns "" or what went wrong."""
    import numpy as np

    samples = resample(samples, rate, out_rate)
    silence = np.zeros(out_rate, dtype=np.float32)
    write_wav(path, np.concatenate([silence, samples, silence, silence]), out_rate)
    done = subprocess.run(
        [sys.executable, "-c", PLAYER, str(path), "blackhole"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return "" if done.returncode == 0 else f"the player exited {done.returncode}: {done.stderr.strip()[-300:]}"


async def timeline(count, start, before: int = 2, after: int = 4, limit: float = 40.0):
    """Samples received in each half second: ``before`` marks, then while the awaitable ``start()`` returns runs,
    then ``after`` marks once it has ended. Returns the marks [(phase, seconds, samples)] and its result."""
    marks: list[tuple[str, float, int]] = []
    last = count()

    def mark(phase: str, seconds: float) -> None:
        nonlocal last
        now = count()
        marks.append((phase, seconds, now - last))
        last = now

    for step in range(1, before + 1):
        await asyncio.sleep(0.5)
        mark("before", step * 0.5)
    task = asyncio.ensure_future(start())
    started, step = time.monotonic(), 0
    while not task.done() and time.monotonic() - started < limit:
        await asyncio.sleep(0.5)
        step += 1
        mark("during", step * 0.5)
    result = await asyncio.wait_for(task, 10) if not task.done() else task.result()
    for step in range(1, after + 1):
        await asyncio.sleep(0.5)
        mark("after", step * 0.5)
    return marks, result


def shown(marks) -> str:
    return ", ".join(f"{phase} {seconds:.1f}s +{samples}" for phase, seconds, samples in marks)


def stalled(marks, phase: str) -> bool:
    return any(samples == 0 for name, _, samples in marks if name == phase)


async def capture(sd, voice, device_id: str, output: int, folder: Path) -> bool:
    import numpy as np

    recorder = voice._ContinuousRecorder(sample_rate=RATE, preferred_device=device_id)
    if not recorder.start():
        return check("audio-macos/capture-hears-audio", False, f"did not open: {recorder.last_error}")
    try:
        receiving = await recorder.wait_until_receiving(timeout=5.0)
        out_rate = device_rate(sd, output)
        print(
            f"DIAG the tone: 440 Hz at {out_rate} Hz, BlackHole's own rate, played from another process; "
            f"Heliox records at {RATE} Hz",
            flush=True,
        )
        tone = (0.4 * np.sin(2 * math.pi * 440 * np.arange(out_rate * 2) / out_rate)).astype(np.float32)
        marks, problem = await timeline(
            lambda: recorder.frames_received * recorder.frame_size,
            lambda: asyncio.to_thread(play_elsewhere, tone, out_rate, out_rate, folder / "heliox-tone.wav"),
            after=2,
        )
        # Samples Heliox received in each half second (about 8000 is normal at 16 kHz).
        print(f"DIAG samples received per 0.5 s, the tone from another process: {shown(marks)}", flush=True)
        if problem:
            print(f"DIAG the player: {problem}", flush=True)
        if stalled(marks, "during"):
            print("DIAG THE TONE FROM ANOTHER PROCESS ALSO STALLED HELIOX'S INPUT", flush=True)
        during = [samples for phase, _, samples in marks if phase == "during"]
        rate = sum(during) / (0.5 * len(during)) if during else 0.0
        heard = recorder.peak_frame_rms
        return check(
            "audio-macos/capture-hears-audio",
            receiving
            and not problem
            and "blackhole" in recorder.input_device_name.casefold()
            and heard > 0.05
            and abs(rate - RATE) <= RATE * 0.1,
            f"device {recorder.input_device_name!r}, frames arriving {receiving}, {rate:.0f} samples/s, "
            f"peak level {heard:.3f} while a tone played from another process (silence is about 0)",
        )
    finally:
        recorder.stop()


async def same_process_tone(sd, voice, device_id: str, output: int) -> None:
    """DIAG only: the tone played from Heliox's own process to the device it records from, as rounds 1 and 2 did,
    with the samples counted until 2 s after the tone ends: does the input come back by itself?"""
    import numpy as np

    recorder = voice._ContinuousRecorder(sample_rate=RATE, preferred_device=device_id)
    if not recorder.start():
        print(f"DIAG the same-process tone: the recorder did not open: {recorder.last_error}", flush=True)
        return
    try:
        await recorder.wait_until_receiving(timeout=5.0)
        out_rate = device_rate(sd, output)
        tone = (0.4 * np.sin(2 * math.pi * 440 * np.arange(out_rate * 2) / out_rate)).astype(np.float32)
        marks, _ = await timeline(
            lambda: recorder.frames_received * recorder.frame_size,
            lambda: asyncio.to_thread(sd.play, tone, out_rate, device=output, blocking=True),
            after=4,
        )
        print(f"DIAG samples received per 0.5 s, the tone from this process: {shown(marks)}", flush=True)
        back = not stalled(marks[-2:], "after") if stalled(marks, "during") else True
        print(
            f"DIAG the same-process tone: input stalled while it played: {stalled(marks, 'during')}; "
            f"arriving again in the last second after it ended: {back}",
            flush=True,
        )
    finally:
        recorder.stop()


def listener_config(device_id: str):
    """The settings a person would have, listening on ``device_id``, with the smallest recogniser."""
    from pilot.config import PilotConfig

    config = PilotConfig()
    settings = config.voice
    settings.input_device = device_id
    settings.whisper_model = "tiny.en"
    settings.transcription_engine = "faster_whisper"
    settings.language = "en"
    settings.turn_detection = "silence"
    settings.final_pass = "whisper"
    return config


async def spoken_turn(sd, voice, device_id: str, output: int, clips: dict, folder: Path) -> None:
    config = listener_config(device_id)
    out_rate = device_rate(sd, output)
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
        problem = await asyncio.to_thread(play_elsewhere, samples, rate, out_rate, folder / "heliox-wake-48k.wav")
        if problem:
            print(f"DIAG the player: {problem}", flush=True)
        await until(lambda: bool(commands), 60)
        wake = [(text, decision) for text, decision in heard if decision.get("why") == "wake"]
        check("voice-macos/wake-found", bool(wake), f"heard {heard!r}")
        command = commands[0] if commands else ""
        score = similar(REQUEST, command)
        check("voice-macos/request-heard", score >= 0.6, f"passed on {command!r} (similarity {score:.2f})")

        before_heard, before_commands = len(heard), len(commands)
        samples, rate = clips["plain"]
        problem = await asyncio.to_thread(play_elsewhere, samples, rate, out_rate, folder / "heliox-plain-48k.wav")
        if problem:
            print(f"DIAG the player: {problem}", flush=True)
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
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp")
    try:
        if not await capture(sd, voice, found["id"], output, folder):
            for name in later[1:]:
                check(name, False, "not run: capture heard nothing")
            return failures
        clips = {}
        for key, text in (("wake", WAKE_TURN), ("plain", NOT_ADDRESSED)):
            path, maker = make_clip(text, folder / f"heliox-{key}.wav")
            print(f"DIAG the {key} clip: {maker}", flush=True)
            if path is None:
                for name in later[1:]:
                    check(name, False, f"not run: {maker}")
                return failures
            clips[key] = wav_samples(path)
        await spoken_turn(sd, voice, found["id"], output, clips, folder)
    finally:
        # Last, so a stall it causes cannot spoil the lines above.
        await same_process_tone(sd, voice, found["id"], output)
    return failures


# --speak-through: Heliox speaking while its listener listens (the same-device frames lines gate; the rest measures) --

SENTENCE = "Here is what I found about the weather in Paris tomorrow. It will be sunny in the afternoon."
AFTER_SECONDS = 5.0  # how long the listener is watched after Heliox stops speaking
BACK_WITHIN = 3.0  # "frames arriving again within N s"


def switch_output(name: str | None) -> str:
    """The system's default output; with ``name``, set it first. SwitchAudioSource (Homebrew switchaudio-osx)."""
    tool = shutil.which("SwitchAudioSource")
    if tool is None:
        raise RuntimeError("SwitchAudioSource is not installed (brew install switchaudio-osx)")
    if name is not None:
        subprocess.run([tool, "-t", "output", "-s", name], capture_output=True, text=True, timeout=30, check=True)
    shown_now = subprocess.run([tool, "-t", "output", "-c"], capture_output=True, text=True, timeout=30, check=True)
    return shown_now.stdout.strip()


class MicrophoneLog(logging.Handler):
    """The voice module's lines about the microphone and the output line ("Microphone: ...", "Microphone working
    again", "Output line opened on the microphone's device: reopening the microphone", "Output line: ..."), with
    the time each was logged."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.lines: list[tuple[float, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith(("Microphone", "Output line")):
            self.lines.append((time.monotonic(), message))
            EVENTS.append((time.monotonic(), f"log: {message}"))


# Every PortAudio stream opened or closed in this process, and where each sentence starts and ends, in order:
# (time, what). Printed at the end, so a failure shows which open came before which.
EVENTS: list[tuple[float, str]] = []
STATE: list = [lambda: ""]  # the line's pinning and the same-device decision, set once voice is imported


def log_streams(sd) -> None:
    """Wrap sounddevice's stream classes in this process so each open and close is recorded with its kind, device
    index and rate. Heliox's voice module reads ``sd.InputStream`` and ``sd.OutputStream`` when it opens one."""

    def recorded(base, kind: str):
        class Recorded(base):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                EVENTS.append((time.monotonic(), f"open {kind}: device {self.device}, rate {self.samplerate:g}"))

            def close(self, *args, **kwargs):
                EVENTS.append((time.monotonic(), f"close {kind}: device {self.device}"))
                return super().close(*args, **kwargs)

        Recorded.__name__ = base.__name__
        return Recorded

    sd.InputStream = recorded(sd.InputStream, "input")
    sd.OutputStream = recorded(sd.OutputStream, "output")


def pin_state(voice, sd, recorder) -> str:
    """The fix's own state: the output line pinned, and its same-device decision (absent before the fix)."""
    line = voice._output_line
    same = getattr(voice, "_same_core_audio_device", None)
    try:
        decided = same(sd, recorder.input_device) if same else "(no _same_core_audio_device here)"
    except Exception as error:
        decided = f"{type(error).__name__}: {error}"
    return f"pinned {getattr(line, 'pinned', '(no pinned here)')}, same Core Audio device {decided}"


async def measure(name: str, target: str, recorder, log: MicrophoneLog, speak) -> None:
    """Samples per half second before, while and for AFTER_SECONDS after ``speak()``; three CHECK lines."""
    first_line = len(log.lines)
    started: list[float] = []

    async def speaking():
        started.append(time.monotonic())
        EVENTS.append((started[0], f"sentence {name} starts"))
        try:
            return await speak()
        finally:
            EVENTS.append((time.monotonic(), f"sentence {name} ends"))

    marks, _ = await timeline(
        lambda: recorder.frames_received * recorder.frame_size,
        speaking,
        before=2,
        after=int(AFTER_SECONDS * 2),
        limit=40.0,
    )
    print(f"DIAG {target}, {name}: samples received per 0.5 s: {shown(marks)}", flush=True)
    first = next((samples for phase, _, samples in marks if phase == "during"), None)
    print(f"DIAG {target}, {name}: samples in the first half second of speaking: {first} (8000 is normal)", flush=True)
    since = [(moment - started[0], message) for moment, message in log.lines[first_line:]] if started else []
    print(f"DIAG {target}, {name}: the voice module's lines, seconds from the start: {since}", flush=True)
    reopened = any(message.startswith("Microphone: ") and "reopening" in message for _, message in since)
    print(f"DIAG {target}, {name}: the watchdog reopened the microphone: {reopened}", flush=True)
    print(f"DIAG {target}, {name}: after it: {STATE[0]()}", flush=True)
    during = [samples for phase, _, samples in marks if phase == "during"]
    after = [(seconds, samples) for phase, seconds, samples in marks if phase == "after"]
    prefix = f"voice-macos-speaking/{target}/{name}"
    check(
        f"{prefix}/frames-while-speaking",
        bool(during) and all(during),
        f"samples per half second while Heliox spoke: {during}",
    )
    back = next((seconds for seconds, samples in after if samples > 0), None)
    check(
        f"{prefix}/frames-after-speaking",
        back is not None and back <= BACK_WITHIN,
        f"first samples {back} s after it stopped (within {BACK_WITHIN} s expected); after: {after}",
    )
    lines = log.lines[first_line:]
    stall = not all(during) or any(samples == 0 for _, samples in after)
    noticed = [message for _, message in lines if message.startswith("Microphone: ")]
    working = [moment for moment, message in lines if message.startswith("Microphone working again")]
    if not stall:
        check(f"{prefix}/watchdog", True, "no stall, nothing for the watchdog to do")
    else:
        took = f"{working[0] - lines[0][0]:.1f} s after its first line" if working and lines else "never"
        check(
            f"{prefix}/watchdog",
            bool(working),
            f"stalled; the watchdog said {noticed or 'nothing'}; microphone working again: {took}",
        )


# --- round 4d: which ways of opening an input and an output on one Core Audio device keep the input delivering ---

QUIET = 0.02  # the output's tone level, as the control's quiet tone


class _Counter:
    """An input callback that counts the frames delivered and keeps any status PortAudio reports."""

    def __init__(self) -> None:
        self.frames = 0
        self.statuses: set[str] = set()

    def __call__(self, indata, frames, _time, status) -> None:
        self.frames += frames
        if status:
            self.statuses.add(str(status).strip())


class _Tone:
    """An output callback that plays a quiet 220 Hz tone at ``rate``."""

    def __init__(self, rate: int) -> None:
        self.rate, self.position = rate, 0

    def fill(self, outdata, frames: int) -> None:
        import numpy as np

        moments = (np.arange(frames) + self.position) / self.rate
        outdata[:, 0] = QUIET * np.sin(2 * math.pi * 220 * moments)
        self.position += frames

    def __call__(self, outdata, frames, _time, _status) -> None:
        self.fill(outdata, frames)


def _count(counter: _Counter, halves: int) -> list[int]:
    """Frames delivered in each of ``halves`` half seconds from now."""
    out, last = [], counter.frames
    for _ in range(halves):
        time.sleep(0.5)
        out.append(counter.frames - last)
        last = counter.frames
    return out


def _report(name: str, rate: int, marks: list[int], counter: _Counter | None, errors: list[str]) -> None:
    expected = rate // 2
    sustained = bool(marks) and all(samples >= 0.8 * expected for samples in marks)
    statuses = sorted(counter.statuses) if counter else []
    print(
        f"DIAG one device, {name}: input samples per 0.5 s {marks} (expected about {expected}), "
        f"{counter.frames if counter else 0} since it opened; "
        f"sustained {'yes' if sustained else 'no'}; PortAudio errors {errors or 'none'}; stream status {statuses or 'none'}",
        flush=True,
    )


def _close(*streams) -> None:
    for stream in streams:
        if stream is None:
            continue
        with contextlib.suppress(Exception):
            stream.abort()
        with contextlib.suppress(Exception):
            stream.close()


def _output(sd, device: int, rate: int, latency: str):
    stream = sd.OutputStream(
        device=device, samplerate=rate, channels=1, dtype="float32", latency=latency, callback=_Tone(rate)
    )
    stream.start()
    # How late a cut is heard depends on this: what PortAudio gives for the latency asked, on this device.
    print(
        f"DIAG one device: output at {rate} Hz asked latency {latency!r}, PortAudio reports "
        f"{float(stream.latency) * 1000:.1f} ms",
        flush=True,
    )
    return stream


def _input(sd, device: int, rate: int, counter: _Counter):
    # As Heliox's recorder opens it: mono int16, 50 ms blocks, default latency.
    stream = sd.InputStream(
        device=device, samplerate=rate, channels=1, dtype="int16", blocksize=rate // 20, callback=counter
    )
    stream.start()
    return stream


def _arrangement(name: str, rate: int, body) -> None:
    """Run one arrangement; a PortAudio error in it is reported and never stops the next. A second of rest after."""
    print(f"DIAG one device, {name}: starting (PortAudio's own error lines, if any, follow this line)", flush=True)
    streams: list = []
    errors: list[str] = []
    counter = _Counter()
    marks: list[int] = []
    try:
        marks = body(streams, counter)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        _close(*streams)
    _report(name, rate, marks, counter, errors)
    time.sleep(1.0)


def open_order_experiment(sd) -> None:
    """DIAG only, about a minute: on BlackHole as input and output, each arrangement counts the input's samples per
    half second for 3 s while the output plays a quiet tone. It answers how the product must open its two streams:
    a settle time after the output, the device's own rate, the output's latency, one duplex stream, or the input
    reopened after the output (the watchdog's path)."""
    device = blackhole_output(sd)
    if device is None or int(sd.query_devices(device)["max_input_channels"]) < 1:
        print("DIAG one device: not run, no BlackHole device with an input and an output", flush=True)
        return
    native = device_rate(sd, device)
    print(f"DIAG one device: BlackHole is PortAudio device {device} at {native} Hz", flush=True)

    # 1. The product's order with a settle: output 24 kHz, latency low; then the 16 kHz input after a pause.
    for settle in (0.1, 0.5, 1.0, 2.0, 5.0):

        def settled(streams, counter, settle=settle):
            streams.append(_output(sd, device, 24000, "low"))
            time.sleep(settle)
            streams.append(_input(sd, device, 16000, counter))
            return _count(counter, 6)

        _arrangement(f"output 24 kHz low, input 16 kHz {settle:g} s after", 16000, settled)

    # 2. Both streams at the device's own rate, 2 s apart.
    def native_rate(streams, counter):
        streams.append(_output(sd, device, native, "low"))
        time.sleep(2.0)
        streams.append(_input(sd, device, native, counter))
        return _count(counter, 6)

    _arrangement(f"output and input both at {native} Hz, input 2 s after", native, native_rate)

    # 3. The output at PortAudio's default latency instead of "low".
    def default_latency(streams, counter):
        streams.append(_output(sd, device, 24000, "high"))
        time.sleep(2.0)
        streams.append(_input(sd, device, 16000, counter))
        return _count(counter, 6)

    _arrangement("output 24 kHz default latency, input 16 kHz 2 s after", 16000, default_latency)

    # 3b. The same with almost no settle: does the default latency need one?
    def default_latency_at_once(streams, counter):
        streams.append(_output(sd, device, 24000, "high"))
        time.sleep(0.1)
        streams.append(_input(sd, device, 16000, counter))
        return _count(counter, 6)

    _arrangement("output 24 kHz default latency, input 16 kHz 0.1 s after", 16000, default_latency_at_once)

    # 3c. Today's order with the default latency: the input running, the output opened beside it, the same input
    # counted through the open and for 3 s after (no reopen).
    def default_latency_input_first(streams, counter):
        streams.append(_input(sd, device, 16000, counter))
        before = _count(counter, 2)
        streams.append(_output(sd, device, 24000, "high"))
        return before + _count(counter, 6)

    _arrangement(
        "input 16 kHz first, then output 24 kHz default latency (two half seconds before the open, six after)",
        16000,
        default_latency_input_first,
    )

    # 4. One duplex stream at the device's rate: input counted while its own output plays the tone.
    def duplex(streams, counter):
        tone = _Tone(native)

        def both(indata, outdata, frames, time_info, status):
            counter(indata, frames, time_info, status)
            tone.fill(outdata, frames)

        stream = sd.Stream(
            device=(device, device),
            samplerate=native,
            channels=(1, 1),
            dtype=("int16", "float32"),
            blocksize=native // 20,
            callback=both,
        )
        streams.append(stream)
        stream.start()
        reported = stream.latency
        print(
            f"DIAG one device: duplex at {native} Hz, PortAudio reports latency (in, out) "
            f"{tuple(round(float(value) * 1000, 1) for value in reported)} ms",
            flush=True,
        )
        return _count(counter, 6)

    _arrangement(f"one duplex stream at {native} Hz", native, duplex)

    # 5. Today's order: input running, then the output opens; the input is closed and reopened after a pause.
    for pause in (0.5, 2.0):

        def reopened(streams, counter, pause=pause):
            first = _Counter()
            streams.append(_input(sd, device, 16000, first))
            before = _count(first, 2)
            at_open = first.frames
            streams.append(_output(sd, device, 24000, "low"))
            time.sleep(pause)
            after_open = first.frames - at_open
            _close(streams.pop(0))  # the first input
            print(
                f"DIAG one device, input first: samples per 0.5 s before the output {before}; in the {pause:g} s "
                f"after the output opened {after_open} (expected about {int(16000 * pause)}); input closed, reopening",
                flush=True,
            )
            streams.append(_input(sd, device, 16000, counter))
            return _count(counter, 6)

        _arrangement(f"input first, output opened, input reopened {pause:g} s after", 16000, reopened)


async def speak_through(target: str, folder: Path) -> None:
    """Heliox's listener on BlackHole's input; Heliox speaks one sentence through its own voice paths with the
    system default output on BlackHole (``blackhole``: microphone and speaker are one Core Audio device, as with a
    USB headset) or left on the runner's own device (``default``: the control)."""
    original = switch_output(None)
    print(f"DIAG the default output before: {original}", flush=True)
    try:
        if target == "blackhole":
            print(f"DIAG the default output now: {switch_output('BlackHole 2ch')}", flush=True)
        # Imported only now: PortAudio reads the default output when it starts.
        import sounddevice as sd

        if target == "blackhole":
            # Before the listener and before the stream log, so its streams are its own.
            await asyncio.to_thread(open_order_experiment, sd)
        log_streams(sd)
        from pilot.system import voice

        devices = voice.list_audio_input_devices(sd, sample_rate=RATE)
        found = next((device for device in devices if "blackhole" in device["name"].casefold()), None)
        if found is None:
            check(f"voice-macos-speaking/{target}/listener", False, "no BlackHole input")
            return
        print(f"DIAG PortAudio default (in, out): {tuple(sd.default.device)}", flush=True)
        log = MicrophoneLog()
        voice_logger = logging.getLogger(voice.__name__)
        voice_logger.addHandler(log)
        voice_logger.setLevel(logging.INFO)

        async def ignore(*_args, **_kwargs) -> None:
            return None

        listener = voice.ContinuousVoiceListener(on_command=ignore, config=listener_config(found["id"]))
        said = await listener.start()
        print(f"DIAG the listener: {said}", flush=True)
        if not listener._running:
            check(f"voice-macos-speaking/{target}/listener", False, f"the listener did not start: {said}")
            print(f"DIAG {target}: when the listener gave up: {pin_state(voice, sd, listener._recorder)}", flush=True)
            return
        STATE[0] = lambda: pin_state(voice, sd, listener._recorder)
        print(f"DIAG {target}: at the listener's start: {STATE[0]()}", flush=True)
        background = None
        try:
            if target == "default":
                # Nothing Heliox says reaches BlackHole here, so without this its input is exact zeros and the
                # watchdog's silent-input rule (30 s) reopens it mid-measurement (round 3's control did).
                background = quiet_tone_elsewhere(folder)
            recorder = listener._recorder
            await recorder.wait_until_receiving(timeout=5.0)
            path, maker = make_clip(SENTENCE, folder / "heliox-sentence.wav")
            if path is None:
                check(f"voice-macos-speaking/{target}/sentence", False, maker)
                return
            samples, rate = wav_samples(path)
            local = write_wav(
                folder / "heliox-sentence-local.wav",
                resample(samples, rate, voice.LOCAL_VOICE_RATE),
                voice.LOCAL_VOICE_RATE,
            )
            other_rate = write_wav(folder / "heliox-sentence-22k.wav", resample(samples, rate, 22050), 22050)
            print(f"DIAG the sentence: {maker}, {len(samples) / rate:.1f} s", flush=True)
            line = voice._output_line
            # The local voice's own playback: the shared output stream inside Heliox's process (voice._output_line).
            await measure("local-voice", target, recorder, log, lambda: voice._play_local_tts_file(local))
            print(f"DIAG {target}: the output line after the first sentence: {line_state(line)}", flush=True)
            await asyncio.sleep(2.0)
            # A second sentence on the line as it is: nothing new opens if the line stayed open.
            stream_before = line.stream
            await measure("local-voice-again", target, recorder, log, lambda: voice._play_local_tts_file(local))
            print(
                f"DIAG {target}: the output line after the second sentence: {line_state(line)}; "
                f"the same stream as before: {line.stream is stream_before}",
                flush=True,
            )
            await asyncio.sleep(2.0)
            # Another rate (22050 Hz; Kokoro's is 24000, Pocket's is its model's). Before the same-device change it
            # reopened the line; with it, a same-device line keeps its rate and plays this resampled, no new open.
            stream_before, window_start = line.stream, time.monotonic()
            await measure(
                "local-voice-other-rate", target, recorder, log, lambda: voice._play_local_tts_file(other_rate)
            )
            opened = opens_between(window_start, time.monotonic(), "open output")
            print(
                f"DIAG {target}: the output line after the other rate: {line_state(line)}; "
                f"reopened: {line.stream is not stream_before}; output opens during it: {opened or 'none'}",
                flush=True,
            )
            if target == "blackhole":  # on separate devices the line reopens at another rate by design
                check(
                    f"voice-macos-speaking/{target}/local-voice-other-rate/no-new-open",
                    not opened and line.stream is stream_before,
                    f"output opens while the 22050 Hz sentence played: {opened or 'none'}",
                )
            await asyncio.sleep(2.0)
            # Heliox silent with its line still open, another program plays to BlackHole: is it heard?
            await other_program_while_line_open(target, recorder, line, folder, sd)
            await asyncio.sleep(2.0)
            # How late a cut (barge-in) is heard: only on BlackHole, where what Heliox plays comes back to the input.
            # The sentence is played at the line's current rate, so nothing reopens and the figure measures the cut.
            line_rate = int(line.rate or voice.LOCAL_VOICE_RATE)
            at_line_rate = write_wav(
                folder / f"heliox-sentence-{line_rate}.wav", resample(samples, rate, line_rate), line_rate
            )
            await cut_heard(target, recorder, voice, at_line_rate, line_rate)
            await asyncio.sleep(2.0)
            # The reopen that cannot be avoided: the default output moves away from the microphone's device and back.
            if target == "blackhole":
                await default_output_moves(recorder, log, original)
                await asyncio.sleep(2.0)
            # The system voice: speak() runs say, a process of its own.
            await measure("system-voice", target, recorder, log, lambda: voice.speak(SENTENCE))
        finally:
            if background is not None:
                background.kill()
                background.wait(timeout=10)
            await listener.stop()
    finally:
        # Every way out prints the order, the listener's own refusal to start included (round 4b on 676d6ed did not).
        begin = EVENTS[0][0] if EVENTS else 0.0
        for moment, what in EVENTS:
            print(f"DIAG {target}, in order: {moment - begin:7.2f} s  {what}", flush=True)
        if target == "blackhole" and original:
            print(f"DIAG the default output put back: {switch_output(original)}", flush=True)


def line_state(line) -> str:
    stream = line.stream
    return f"open {stream is not None}, rate {line.rate}, device {line.device_name!r}"


def quiet_tone_elsewhere(folder: Path) -> subprocess.Popen:
    """A quiet 220 Hz tone (level 0.02) played to BlackHole for two minutes from another process: the control's
    input is then never exact zeros, and the level stays far below the tone the other-program case listens for."""
    import numpy as np

    out_rate = 48000
    tone = (0.02 * np.sin(2 * math.pi * 220 * np.arange(out_rate * 120) / out_rate)).astype(np.float32)
    path = write_wav(folder / "heliox-quiet-tone.wav", tone, out_rate)
    return subprocess.Popen(
        [sys.executable, "-c", PLAYER, str(path), "blackhole"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


QUIET_LEVEL = 0.005  # a 50 ms frame below this RMS counts as quiet (the sentence speaks at about 0.05 or more)


async def cut_heard(target: str, recorder, voice, local: Path, line_rate: int = 0) -> None:
    """DIAG only: the time from a cut being asked (the local voice's playback task cancelled, as barge-in does) to
    the output going quiet, read from outside voice.py. On BlackHole what Heliox plays comes back to its input, so
    "quiet" is the recorder's frame level falling below QUIET_LEVEL for three frames in a row. The time includes the
    input's own buffering and has the frame's resolution (50 ms). On the runner's own output (the control) nothing
    played comes back to the input, so it cannot be measured from outside and is not run."""
    if target != "blackhole":
        print(f"DIAG {target}, cut: not measured; the output does not come back to the input here", flush=True)
        return
    window_start = time.monotonic()
    task = asyncio.ensure_future(voice._play_local_tts_file(local))
    loud_by = time.monotonic() + 3.0
    while recorder.last_frame_rms < QUIET_LEVEL * 4 and time.monotonic() < loud_by and not task.done():
        await asyncio.sleep(0.01)
    playing = recorder.last_frame_rms >= QUIET_LEVEL * 4
    await asyncio.sleep(1.0)  # well into the sentence
    asked = time.monotonic()
    task.cancel()
    quiet_frames, quiet_at, deadline = 0, None, asked + 3.0
    last_seen = recorder.frames_received
    while time.monotonic() < deadline:
        await asyncio.sleep(0.005)
        if recorder.frames_received == last_seen:
            continue  # judge each frame once
        last_seen = recorder.frames_received
        if recorder.last_frame_rms < QUIET_LEVEL:
            quiet_frames += 1
            if quiet_frames == 1:
                quiet_at = time.monotonic()
            if quiet_frames >= 3:
                break
        else:
            quiet_frames, quiet_at = 0, None
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await asyncio.wait_for(task, 5)
    heard = (
        f"{(quiet_at - asked) * 1000:.0f} ms" if quiet_at is not None and quiet_frames >= 3 else "not quiet within 3 s"
    )
    opened = opens_between(window_start, time.monotonic(), "open output")
    print(
        f"DIAG {target}, cut: the sentence at {line_rate or 'its own'} Hz (the line's rate); output opens during it: "
        f"{opened or 'none'}",
        flush=True,
    )
    print(
        f"DIAG {target}, cut: the sentence was heard playing before the cut: {playing}; quiet at the microphone "
        f"{heard} after the cut was asked (50 ms frames, input buffering included)",
        flush=True,
    )


def opens_between(start: float, end: float, kind: str) -> list[str]:
    """The stream log's entries of ``kind`` ("open output", "close input", ...) between two moments."""
    return [what for moment, what in EVENTS if start <= moment <= end and what.startswith(kind)]


HOOK_REOPEN = "Output line opened on the microphone's device: reopening the microphone"


def default_speaker_seen() -> str:
    """The default output as the voice module reads it (audio_health.default_speaker_name, through soundcard); ""
    when it cannot, and then Heliox cannot follow a move of the default output."""
    try:
        from pilot.system.audio_health import default_speaker_name

        return default_speaker_name()
    except Exception as error:
        return f"{type(error).__name__}: {error}"


async def default_output_moves(recorder, log: MicrophoneLog, original: str) -> None:
    """The one reopen that cannot be avoided: the system default output moves from BlackHole (the microphone's
    device) to the runner's own device, then back, while the listener runs. Non-gating: the order of opens and
    closes, the samples per half second across it, and whether exactly one reopen sequence ran for each move (the
    output line's hook or the watchdog, never both)."""
    away = original or "Apple Virtual Sound Device"
    print(
        f"DIAG blackhole, default output moves: what the voice module reads now: {default_speaker_seen()!r}", flush=True
    )
    first_line, window_start = len(log.lines), time.monotonic()
    moments: dict[str, float] = {}

    async def moves() -> None:
        await asyncio.sleep(1.0)
        moments["away"] = time.monotonic()
        print(f"DIAG blackhole, default output moved away: {await asyncio.to_thread(switch_output, away)}", flush=True)
        await asyncio.sleep(5.0)
        moments["back"] = time.monotonic()
        back = await asyncio.to_thread(switch_output, "BlackHole 2ch")
        print(f"DIAG blackhole, default output moved back: {back}", flush=True)
        await asyncio.sleep(5.0)

    marks, _ = await timeline(lambda: recorder.frames_received * recorder.frame_size, moves, before=2, after=2)
    print(f"DIAG blackhole, default output moves: samples received per 0.5 s: {shown(marks)}", flush=True)
    window = [(moment, what) for moment, what in EVENTS if window_start <= moment]
    for moment, what in window:
        print(f"DIAG blackhole, default output moves, in order: {moment - window_start:6.2f} s  {what}", flush=True)
    lines = log.lines[first_line:]
    for name, start, end in (
        ("away", moments.get("away", 0.0), moments.get("back", float("inf"))),
        ("back", moments.get("back", float("inf")), float("inf")),
    ):
        hook = sum(1 for moment, message in lines if start <= moment < end and message.startswith(HOOK_REOPEN))
        watchdog = sum(
            1
            for moment, message in lines
            if start <= moment < end and message.startswith("Microphone: ") and "reopening" in message
        )
        outputs = [what for moment, what in window if start <= moment < end and what.startswith("open output")]
        if not (hook or watchdog or outputs):
            # Nothing reopened: the move was not seen (on 09e886c the voice module could not read the default
            # output, soundcard not installed), which proves nothing either way.
            skip(
                f"voice-macos-speaking/blackhole/default-output-moves-{name}/one-reopen-sequence",
                f"not exercised: no output opened and nothing reopened after the move "
                f"(the output line's device: {default_speaker_seen()!r})",
            )
            continue
        check(
            f"voice-macos-speaking/blackhole/default-output-moves-{name}/one-reopen-sequence",
            not (hook and watchdog),
            f"the line's hook reopened the microphone {hook} time(s), the watchdog {watchdog} time(s); "
            f"output opens: {outputs or 'none'}",
        )
    during = [samples for phase, _, samples in marks if phase == "during"]
    print(
        f"DIAG blackhole, default output moves: half seconds with no samples: {during.count(0)} of {len(during)}",
        flush=True,
    )


async def other_program_while_line_open(target: str, recorder, line, folder: Path, sd) -> None:
    """With Heliox silent and its output line open, the tone from another process (as in the capture check) must
    keep the input arriving and be heard: an open line must not shut other programs out."""
    import numpy as np

    blackhole = blackhole_output(sd)
    out_rate = device_rate(sd, blackhole) if blackhole is not None else 48000
    tone = (0.4 * np.sin(2 * math.pi * 440 * np.arange(out_rate * 2) / out_rate)).astype(np.float32)
    print(f"DIAG {target}: the output line while another program plays: {line_state(line)}", flush=True)
    recorder.peak_frame_rms = 0.0  # the check's own reading of the level, from here on
    marks, problem = await timeline(
        lambda: recorder.frames_received * recorder.frame_size,
        lambda: asyncio.to_thread(play_elsewhere, tone, out_rate, out_rate, folder / "heliox-other-program.wav"),
        after=2,
    )
    heard = recorder.peak_frame_rms
    print(f"DIAG {target}, another program: samples received per 0.5 s: {shown(marks)}", flush=True)
    if problem:
        print(f"DIAG the player: {problem}", flush=True)
    during = [samples for phase, _, samples in marks if phase == "during"]
    prefix = f"voice-macos-speaking/{target}/other-program-with-line-open"
    check(f"{prefix}/frames-arriving", bool(during) and all(during), f"samples per half second: {during}")
    check(
        f"{prefix}/tone-heard",
        heard > 0.1 and not problem,
        f"peak level {heard:.3f} (the tone is about 0.28; the control's quiet tone about 0.014)",
    )


if __name__ == "__main__":
    if sys.platform != "darwin":
        print("NOT RUN: this check is for macOS, and changed nothing here.")
        sys.exit(3)
    folder_arg = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else Path("/tmp")
    if "--speak-through" in sys.argv:
        where = sys.argv[sys.argv.index("--speak-through") + 1]
        try:
            asyncio.run(speak_through(where, folder_arg))
        except Exception as error:  # every exit path prints a CHECK line
            check(f"voice-macos-speaking/{where}/check-itself", False, f"{type(error).__name__}: {error}")
        if where != "blackhole":
            print("DIAG the control: a measurement, these lines do not fail the job", flush=True)
            sys.exit(0)
        # Same device: the gating lines decide; the experiment, the cut and the move stay measurements.
        sys.exit(min(same_device_verdict(), 100))
    try:
        sys.exit(asyncio.run(main()))
    except Exception as error:  # every exit path prints a CHECK line
        check("voice-macos/check-itself", False, f"{type(error).__name__}: {error}")
        sys.exit(max(failures, 1))
