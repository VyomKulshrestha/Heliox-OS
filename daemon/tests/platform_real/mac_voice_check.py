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
silent-input rule does not fire. A measurement: it always exits 0.

Every wait is bounded; the exit code is the number of failed checks (a SKIP is not one). CI-only
(``ci_only.require_runner``): the workflow installs an audio driver and restarts coreaudiod, and this check plays
into and records from it.
"""

from __future__ import annotations

import asyncio
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


# --speak-through: Heliox speaking while its listener listens (a measurement; it never fails the job) ----------------

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
    """The listener's "Microphone: ..." and "Microphone working again" lines, with the time each was logged."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.lines: list[tuple[float, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith("Microphone"):
            self.lines.append((time.monotonic(), message))


async def measure(name: str, target: str, recorder, log: MicrophoneLog, speak) -> None:
    """Samples per half second before, while and for AFTER_SECONDS after ``speak()``; three CHECK lines."""
    first_line = len(log.lines)
    marks, _ = await timeline(
        lambda: recorder.frames_received * recorder.frame_size,
        speak,
        before=2,
        after=int(AFTER_SECONDS * 2),
        limit=40.0,
    )
    print(f"DIAG {target}, {name}: samples received per 0.5 s: {shown(marks)}", flush=True)
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
            return
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
            # Another rate (22050 Hz; Kokoro's is 24000, Pocket's is its model's) makes _OutputLine.ensure reopen.
            stream_before = line.stream
            await measure(
                "local-voice-other-rate", target, recorder, log, lambda: voice._play_local_tts_file(other_rate)
            )
            print(
                f"DIAG {target}: the output line after the other rate: {line_state(line)}; "
                f"reopened: {line.stream is not stream_before}",
                flush=True,
            )
            await asyncio.sleep(2.0)
            # Heliox silent with its line still open, another program plays to BlackHole: is it heard?
            await other_program_while_line_open(target, recorder, line, folder, sd)
            await asyncio.sleep(2.0)
            # The system voice: speak() runs say, a process of its own.
            await measure("system-voice", target, recorder, log, lambda: voice.speak(SENTENCE))
        finally:
            if background is not None:
                background.kill()
                background.wait(timeout=10)
            await listener.stop()
    finally:
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
        print("DIAG a measurement (round 4): these lines do not fail the job", flush=True)
        sys.exit(0)
    try:
        sys.exit(asyncio.run(main()))
    except Exception as error:  # every exit path prints a CHECK line
        check("voice-macos/check-itself", False, f"{type(error).__name__}: {error}")
        sys.exit(max(failures, 1))
