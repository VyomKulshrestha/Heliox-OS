"""Voice Input/Output — talk to Heliox OS like JARVIS.

Speech-to-text via Whisper (local or API), text-to-speech via
system TTS or edge-tts, and optional wake word detection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import warnings
import wave
from pathlib import Path
from typing import Any

from pilot.config import PilotConfig
from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_powershell
from pilot.system.vad import AdaptiveEnergyThreshold, EndpointEvent, UtteranceEndpointer, frame_rms
from pilot.system.voice_calibration import WakeWordCalibrator

logger = logging.getLogger("pilot.system.voice")

_VOICE_TRANSCRIPTION_PROMPT = (
    "Hey Heliox. Desktop command vocabulary: open, close, click, type, search, scroll, launch, switch, "
    "select, save, send, stop, continue, approve, deny. Proper names: Heliox, GitHub, Google, YouTube, "
    "Gmail, Spotify, Chrome, Visual Studio Code, PowerShell, Docker."
)

_AUTONOMOUS_COMMAND_START = re.compile(
    r"^(?:"
    r"open|click|press|choose|select|show|find|search|look|tell|explain|summari[sz]e|"
    r"read|write|create|launch|close|stop|cancel|continue|go|navigate|visit|browse|"
    r"play|pause|send|type|scroll|switch|check|inspect|research|remind|schedule|turn|"
    r"set|run|execute|use|approve|deny|save|download|upload|install|uninstall|"
    r"what|who|where|when|why|how|which|can\s+you|could\s+you|would\s+you|will\s+you|"
    r"do\s+you|is\s+(?:this|that|there|it)|are\s+(?:you|there|these|those)"
    r")\b",
    re.IGNORECASE,
)
_AUTONOMOUS_SHORT_DECISIONS = frozenset(
    {
        "yes",
        "no",
        "do it",
        "go ahead",
        "continue",
        "cancel",
        "stop",
        "approve",
        "deny",
        "not now",
        "sounds good",
    }
)

_HELIOX_PRONUNCIATION = "Hee-lee-ox"


def _prepare_spoken_text(text: str) -> str:
    """Apply product-name pronunciations consistently across every TTS engine."""
    return re.sub(r"\bheliox\b", _HELIOX_PRONUNCIATION, text, flags=re.IGNORECASE)


def _looks_like_autonomous_voice_intent(text: str) -> bool:
    """Reject ambient fragments while preserving natural hands-free intent."""
    normalized = " ".join(text.casefold().strip(" \t,.:;!?-").split())
    return normalized in _AUTONOMOUS_SHORT_DECISIONS or bool(_AUTONOMOUS_COMMAND_START.match(normalized))


def _remove_phrase_case_insensitive(text: str, phrase: str) -> str:
    """Remove one recognized phrase without lowercasing the remaining command."""
    start = text.casefold().find(phrase.casefold())
    if start < 0:
        return text.strip(" \t,.:;!?-")
    end = start + len(phrase)
    return f"{text[:start]}{text[end:]}".strip(" \t,.:;!?-")


# The currently in-flight speak() call, if any -- module-level so that ANY
# two callers anywhere in the daemon (executor.py's cognitive-stress-gate
# phrase, AutonomousExecutor's end-of-job announcement, server.py's voice
# response path, ...) automatically supersede each other, mirroring the
# frontend's tts.ts calling speechSynthesis.cancel() before every speak.
# Keyed by nothing (there's only ever one daemon-side voice output device),
# same pop/cancel idiom VoiceGestureWorkflowEngine._active_tasks uses per
# workflow_id, just with a single slot instead of a dict.
_current_speech_task: asyncio.Task[str] | None = None
_local_tts_process: subprocess.Popen[bytes] | None = None
_local_tts_engine: str | None = None
_local_tts_idle_task: asyncio.Task[None] | None = None
_LOCAL_TTS_IDLE_SECONDS = 10.0


async def _play_local_tts_file(path: Path) -> None:
    """Play a synthesized WAV without importing the model runtime."""

    def _play() -> None:
        import sounddevice as sd
        import soundfile

        audio, sample_rate = soundfile.read(path, dtype="float32")
        sd.play(audio, sample_rate)
        sd.wait()

    try:
        await asyncio.to_thread(_play)
    except asyncio.CancelledError:
        import sounddevice as sd

        sd.stop()
        raise


async def _shutdown_local_tts_worker(process: subprocess.Popen[bytes] | None = None) -> None:
    global _local_tts_process, _local_tts_engine, _local_tts_idle_task
    worker = process or _local_tts_process
    idle_task = _local_tts_idle_task
    _local_tts_idle_task = None
    if idle_task is not None and idle_task is not asyncio.current_task() and not idle_task.done():
        idle_task.cancel()
    if worker is None:
        return
    if worker is _local_tts_process:
        _local_tts_process = None
        _local_tts_engine = None

    def _close() -> None:
        try:
            if worker.poll() is None:
                if worker.stdin is not None:
                    try:
                        worker.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass
                try:
                    worker.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    worker.terminate()
                    try:
                        worker.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        worker.kill()
                        worker.wait()
        finally:
            if worker.stdin is not None and not worker.stdin.closed:
                try:
                    worker.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            if worker.stdout is not None:
                worker.stdout.close()

    await asyncio.to_thread(_close)


async def _expire_local_tts_worker(process: subprocess.Popen[bytes]) -> None:
    await asyncio.sleep(_LOCAL_TTS_IDLE_SECONDS)
    if process is _local_tts_process:
        await _shutdown_local_tts_worker(process)


def _schedule_local_tts_worker_expiry(process: subprocess.Popen[bytes]) -> None:
    global _local_tts_idle_task
    if _local_tts_idle_task is not None and not _local_tts_idle_task.done():
        _local_tts_idle_task.cancel()
    _local_tts_idle_task = asyncio.create_task(_expire_local_tts_worker(process))


async def _get_local_tts_worker(engine: str) -> subprocess.Popen[bytes]:
    global _local_tts_process, _local_tts_engine, _local_tts_idle_task
    if _local_tts_idle_task is not None and not _local_tts_idle_task.done():
        _local_tts_idle_task.cancel()
    _local_tts_idle_task = None
    if _local_tts_process is not None and _local_tts_process.poll() is None and _local_tts_engine == engine:
        return _local_tts_process
    await _shutdown_local_tts_worker()

    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    command = [sys.executable, "-m", "pilot.system.local_tts_worker", "--engine", engine, "--serve"]
    _local_tts_process = await asyncio.to_thread(
        subprocess.Popen,
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    _local_tts_engine = engine
    return _local_tts_process


def _exchange_local_tts_request(process: subprocess.Popen[bytes], request: dict[str, str]) -> bytes:
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("Local TTS worker streams are unavailable")
    process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
    process.stdin.flush()
    return process.stdout.readline()


async def _run_local_tts_worker(
    engine: str,
    text: str,
    voice: str,
    output_file: str | None,
) -> None:
    """Synthesize heavy local voices outside the long-lived daemon process."""
    if engine not in {"kokoro_tts", "pocket_tts"}:
        raise ValueError(f"Unsupported local TTS engine: {engine}")
    if len(text) > 20_000:
        raise ValueError("Speech text exceeds the 20,000 character limit")

    with tempfile.TemporaryDirectory(prefix="heliox-tts-") as temp_dir:
        request_path = Path(temp_dir) / "request.txt"
        request_path.write_text(text, encoding="utf-8")
        target_path = Path(output_file) if output_file else Path(temp_dir) / "speech.wav"
        process = await _get_local_tts_worker(engine)
        request = {
            "text_file": str(request_path),
            "voice": voice,
            "output_file": str(target_path),
        }
        try:
            response_line = await asyncio.to_thread(_exchange_local_tts_request, process, request)
        except asyncio.CancelledError:
            await _shutdown_local_tts_worker(process)
            raise
        if not response_line:
            await _shutdown_local_tts_worker(process)
            raise RuntimeError(f"{engine} worker exited before returning audio")
        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as exc:
            await _shutdown_local_tts_worker(process)
            raise RuntimeError(f"{engine} worker returned an invalid response") from exc
        if response.get("status") != "ok":
            await _shutdown_local_tts_worker(process)
            reason = str(response.get("error") or "unknown synthesis failure")[-500:]
            raise RuntimeError(f"{engine} worker failed: {reason}")
        if not target_path.is_file() or target_path.stat().st_size == 0:
            await _shutdown_local_tts_worker(process)
            raise RuntimeError(f"{engine} worker produced no audio")
        try:
            if output_file is None:
                await _play_local_tts_file(target_path)
        finally:
            _schedule_local_tts_worker_expiry(process)


# ── Text-to-Speech ───────────────────────────────────────────────────


async def _supersede_current_speech() -> None:
    """Cancels whatever speak() call is currently in flight, if any, and
    waits for it to actually stop (including killing its OS TTS subprocess)
    before returning."""
    global _current_speech_task
    task = _current_speech_task
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("Superseded speak() task raised on cancellation", exc_info=True)
    _current_speech_task = None


async def stop_speaking() -> str:
    """Stop any daemon-side speech that is currently playing."""
    await _supersede_current_speech()
    return "Speech stopped"


async def speak(
    text: str,
    voice: str | None = None,
    rate: int = 170,
    volume: float = 1.0,
    output_file: str | None = None,
) -> str:
    """Speak text aloud using system TTS.

    Supersedes any speak() call still in progress -- the new text always
    wins immediately rather than overlapping with or queuing behind the
    old one, the same "cancel-and-replace" semantics tts.ts's speakText()
    already gives the frontend via speechSynthesis.cancel().
    """
    global _current_speech_task

    await _supersede_current_speech()

    task = asyncio.ensure_future(_speak_impl(text, voice, rate, volume, output_file))
    _current_speech_task = task
    try:
        return await task
    finally:
        if _current_speech_task is task:
            _current_speech_task = None


async def _speak_impl(
    text: str,
    voice: str | None = None,
    rate: int = 170,
    volume: float = 1.0,
    output_file: str | None = None,
) -> str:
    """The actual OS-dispatch work for a single speak() call -- split out
    from speak() so it can be tracked as its own cancellable task (see
    _current_speech_task/_supersede_current_speech above) while keeping
    speak()'s own cancellation-propagates-to-proc.kill() behavior intact:
    cancelling a task that's awaiting another task cancels that inner task
    too, so nothing about the existing barge-in kill path changes."""
    spoken_text = _prepare_spoken_text(text)
    try:
        from pilot.cognitive.cognitive_engine import CognitiveEngine

        engine = CognitiveEngine.get_instance()
        if engine.is_loaded:
            cog_load = (await engine.predict_cognitive_state()).cognitive_load
            if cog_load > 0.6:
                reduction = int((cog_load - 0.5) * 80)
                rate = max(100, rate - reduction)
                logger.info(f"Modulating voice rate to {rate} due to cognitive load {cog_load:.2f}")
    except Exception as e:
        logger.debug(f"Failed to modulate TTS rate by cognitive load: {e}")

    config = PilotConfig.load()
    if config.voice.tts_engine == "kokoro_tts":
        try:
            await _run_local_tts_worker("kokoro_tts", spoken_text, config.voice.tts_voice, output_file)
            if output_file:
                return f"Speech saved to {output_file}"
            return f"Spoken: {text[:80]}..."
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("Kokoro TTS failed (%s), falling back to OS-native TTS", e, exc_info=True)

    if config.voice.tts_engine == "pocket_tts":
        try:
            # Pocket TTS has no SAPI-style integer rate/volume knob to carry
            # the cognitive-load rate modulation above onto -- out of scope
            # for this pass, see SECURITY.md.
            await _run_local_tts_worker("pocket_tts", spoken_text, config.voice.tts_voice, output_file)
            if output_file:
                return f"Speech saved to {output_file}"
            return f"Spoken: {text[:80]}..."
        except asyncio.CancelledError:
            # Barge-in -- propagate exactly like the OS-native paths do,
            # do not fall back to them mid-interruption.
            raise
        except Exception as e:
            logger.warning("Pocket TTS failed (%s), falling back to OS-native TTS", e, exc_info=True)
            # falls through to the platform dispatch below

    if CURRENT_PLATFORM == Platform.WINDOWS:
        return await _tts_windows(spoken_text, voice, rate, volume, output_file)
    elif CURRENT_PLATFORM == Platform.MACOS:
        return await _tts_macos(spoken_text, voice, rate, output_file)
    else:
        return await _tts_linux(spoken_text, voice, rate, output_file)


async def speak_interruptible(
    text: str,
    recorder: _ContinuousRecorder | None = None,
    **speak_kwargs: Any,
) -> bool:
    """Speaks `text` aloud via speak(), but stops immediately if the user
    starts talking mid-playback instead of finishing the sentence over
    them — the barge-in half of "listen while speaking." Detection reuses
    `recorder`'s continuous VAD stream (see pilot.system.vad), so no
    separate audio capture is needed just to watch for an interruption.

    Returns True if playback was interrupted, False if it completed
    normally. If `recorder` is None or its input stream isn't active
    (sounddevice missing, or barge-in disabled by the caller), this is
    equivalent to plain speak() — always returns False.
    """
    if recorder is None or not recorder.is_active:
        await speak(text, **speak_kwargs)
        return False

    speak_task = asyncio.create_task(speak(text, **speak_kwargs))
    watch_task = asyncio.create_task(recorder.wait_for_speech_start(timeout=None))

    done, _pending = await asyncio.wait({speak_task, watch_task}, return_when=asyncio.FIRST_COMPLETED)

    if watch_task in done:
        speak_task.cancel()
        try:
            await speak_task
        except asyncio.CancelledError:
            pass
        return True

    watch_task.cancel()
    try:
        await watch_task
    except asyncio.CancelledError:
        pass
    return False


async def _tts_windows(
    text: str,
    voice: str | None,
    rate: int,
    volume: float,
    output_file: str | None,
) -> str:
    safe_text = text.replace("'", "''").replace('"', '""')

    if output_file:
        script = (
            f"Add-Type -AssemblyName System.Speech; "
            f"$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$synth.Rate = {(rate - 170) // 20}; "
            f"$synth.Volume = {int(volume * 100)}; "
        )

        if voice:
            script += f"$synth.SelectVoice('{voice}'); "

        script += (
            f"$synth.SetOutputToWaveFile('{output_file}'); "
            f"$synth.Speak('{safe_text}'); "
            "$synth.SetOutputToDefaultAudioDevice(); "
            "$synth.Dispose()"
        )

        code, out, err = await run_powershell(script)

        if code != 0:
            return f"TTS save failed: {err}"

        return f"Speech saved to {output_file}"

    script = (
        f"Add-Type -AssemblyName System.Speech; "
        f"$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$synth.Rate = {(rate - 170) // 20}; "
        f"$synth.Volume = {int(volume * 100)}; "
    )

    if voice:
        script += f"$synth.SelectVoice('{voice}'); "

    script += f"$synth.Speak('{safe_text}'); $synth.Dispose()"

    code, out, err = await run_powershell(script)

    if code != 0:
        return f"TTS failed: {err}"

    return f"Spoken: {text[:80]}..."


async def _tts_linux(
    text: str,
    voice: str | None,
    rate: int,
    output_file: str | None,
) -> str:
    cmd = ["espeak"]

    if voice:
        cmd.extend(["-v", voice])

    cmd.extend(["-s", str(rate)])

    if output_file:
        cmd.extend(["-w", output_file])

    cmd.append(text)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        await proc.communicate()
    except asyncio.CancelledError:
        # Barge-in interrupting speak() mid-playback — kill espeak rather
        # than leaving it running after this await appears cancelled.
        proc.kill()
        raise

    if output_file:
        return f"Speech saved to {output_file}"

    return f"Spoken: {text[:80]}..."


async def _tts_macos(
    text: str,
    voice: str | None,
    rate: int,
    output_file: str | None,
) -> str:
    cmd = ["say"]

    if voice:
        cmd.extend(["-v", voice])

    cmd.extend(["-r", str(rate)])

    if output_file:
        cmd.extend(["-o", output_file])

    cmd.append(text)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        await proc.communicate()
    except asyncio.CancelledError:
        # Barge-in interrupting speak() mid-playback — kill `say` rather
        # than leaving it running after this await appears cancelled.
        proc.kill()
        raise

    if output_file:
        return f"Speech saved to {output_file}"

    return f"Spoken: {text[:80]}..."


async def list_voices() -> str:
    """List available TTS voices."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_powershell(
            "Add-Type -AssemblyName System.Speech; "
            "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$synth.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }; "
            "$synth.Dispose()"
        )
        return out.strip() if code == 0 else f"Error: {err}"

    elif CURRENT_PLATFORM == Platform.MACOS:
        proc = await asyncio.create_subprocess_exec(
            "say",
            "-v",
            "?",
            stdout=asyncio.subprocess.PIPE,
        )
        out, _ = await proc.communicate()
        return out.decode("utf-8", errors="replace")

    else:
        proc = await asyncio.create_subprocess_exec(
            "espeak",
            "--voices",
            stdout=asyncio.subprocess.PIPE,
        )
        out, _ = await proc.communicate()
        return out.decode("utf-8", errors="replace")


# ── Speech-to-Text ───────────────────────────────────────────────────


async def listen(
    duration: int = 5,
    language: str = "auto",
    model: str = "base",
) -> str:
    """Listen to the microphone and transcribe speech."""
    audio_path = await _record_audio(duration)

    try:
        result = await _transcribe_speech(audio_path, language, model)
        return result["text"]

    except ImportError:
        pass

    if CURRENT_PLATFORM == Platform.WINDOWS:
        return await _transcribe_windows(audio_path)

    return "ERROR: Install whisper for speech-to-text: pip install openai-whisper"


def _resolve_input_device(
    sd: Any,
    *,
    sample_rate: int,
    channels: int,
    dtype: str,
    preferred_device: str = "auto",
    allow_preferred_fallback: bool = False,
) -> int:
    """Return a usable PortAudio input device for the requested format.

    Windows can expose real microphones while leaving PortAudio's default
    input at ``-1``. Prefer a valid OS default, then a microphone/microphone
    array, and only then generic capture devices. Loopback/output-like inputs
    are kept as a last resort.
    """

    def _usable(index: int) -> bool:
        try:
            info = sd.query_devices(index)
            if int(info.get("max_input_channels", 0)) < channels:
                return False
            sd.check_input_settings(
                device=index,
                channels=channels,
                dtype=dtype,
                samplerate=sample_rate,
            )
            return True
        except Exception:
            return False

    devices = list(sd.query_devices())

    def _hostapi_name(info: Any) -> str:
        try:
            return str(sd.query_hostapis(int(info.get("hostapi", -1))).get("name", "Unknown"))
        except Exception:
            return "Unknown"

    def _device_id(info: Any) -> str:
        return f"{_hostapi_name(info)}::{str(info.get('name', '')).strip()}"

    preferred = preferred_device.strip()
    if preferred and preferred.casefold() != "auto":
        matching = [index for index, info in enumerate(devices) if _device_id(info).casefold() == preferred.casefold()]
        for index in matching:
            if _usable(index):
                return index
        if not allow_preferred_fallback:
            if matching:
                raise RuntimeError(
                    f"Configured microphone '{preferred}' does not support {sample_rate} Hz, {channels} channel, {dtype}"
                )
            raise RuntimeError(
                f"Configured microphone '{preferred}' is no longer available. Choose another input in Settings."
            )

    try:
        default_device = sd.default.device
        try:
            # sounddevice exposes this as _InputOutputPair, which is
            # indexable but is not a list/tuple.
            default_input = int(default_device[0])
        except (IndexError, TypeError):
            default_input = int(default_device)
    except (AttributeError, IndexError, TypeError, ValueError):
        default_input = -1

    if default_input >= 0 and _usable(default_input):
        return default_input

    def _priority(item: tuple[int, Any]) -> tuple[int, int, int]:
        index, info = item
        name = str(info.get("name", "")).lower()
        loopback_like = any(
            marker in name for marker in ("stereo mix", "loopback", "what u hear", "pc speaker", "output")
        )
        if "microphone array" in name:
            kind = 0
        elif "microphone" in name or "mic input" in name:
            kind = 1
        elif "input" in name:
            kind = 2
        else:
            kind = 3
        return (1 if loopback_like else 0, kind, index)

    candidates = [
        (index, info) for index, info in enumerate(devices) if int(info.get("max_input_channels", 0)) >= channels
    ]
    for index, _info in sorted(candidates, key=_priority):
        if _usable(index):
            return index

    raise RuntimeError(f"No usable microphone supports {sample_rate} Hz, {channels} channel, {dtype}")


def list_audio_input_devices(
    sd: Any,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    dtype: str = "int16",
) -> list[dict[str, Any]]:
    """Return usable input devices with stable, user-visible identifiers."""
    devices: list[dict[str, Any]] = []
    try:
        default_device = sd.default.device
        try:
            default_input = int(default_device[0])
        except (IndexError, TypeError):
            default_input = int(default_device)
    except (AttributeError, IndexError, TypeError, ValueError):
        default_input = -1

    for index, info in enumerate(list(sd.query_devices())):
        if int(info.get("max_input_channels", 0)) < channels:
            continue
        try:
            sd.check_input_settings(
                device=index,
                channels=channels,
                dtype=dtype,
                samplerate=sample_rate,
            )
        except Exception:
            continue
        try:
            hostapi = str(sd.query_hostapis(int(info.get("hostapi", -1))).get("name", "Unknown"))
        except Exception:
            hostapi = "Unknown"
        name = str(info.get("name", index)).strip()
        devices.append(
            {
                "id": f"{hostapi}::{name}",
                "name": name,
                "hostapi": hostapi,
                "index": index,
                "is_default": index == default_input,
            }
        )
    return devices


async def _record_audio(duration: int) -> str:
    """Record audio from the microphone."""
    output_path = os.path.join(
        tempfile.gettempdir(),
        f"pilot_audio_{os.getpid()}.wav",
    )

    try:
        import sounddevice as sd

        sample_rate = 16000
        input_device = _resolve_input_device(
            sd,
            sample_rate=sample_rate,
            channels=1,
            dtype="int16",
        )

        data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            device=input_device,
        )

        sd.wait()

        with wave.open(output_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(data.tobytes())

        return output_path

    except ImportError:
        pass

    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_powershell(
            f"Add-Type -AssemblyName System.Speech; "
            f"$recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine; "
            f"$recognizer.SetInputToDefaultAudioDevice(); "
            f"$grammar = New-Object System.Speech.Recognition.DictationGrammar; "
            f"$recognizer.LoadGrammar($grammar); "
            f"$result = $recognizer.Recognize([TimeSpan]::FromSeconds({duration})); "
            f"if ($result) {{ $result.Text }} else {{ 'No speech detected' }}; "
            f"$recognizer.Dispose()"
        )

        Path(output_path).write_text(
            out.strip() if code == 0 else "",
        )

        return output_path

    raise RuntimeError("Install sounddevice for audio recording: pip install sounddevice")


_whisper_model_cache: dict[str, Any] = {}
_faster_whisper_model_cache: dict[tuple[str, str, str], Any] = {}


def _load_pcm_wav_for_whisper(audio_path: str) -> Any | None:
    """Load an uncompressed PCM WAV without requiring the ffmpeg executable.

    Heliox records 16 kHz mono int16 WAV files itself, so routing those files
    back through Whisper's ffmpeg subprocess adds an avoidable external
    dependency. Return ``None`` for formats we do not decode locally so
    Whisper can retain its normal path-based fallback.
    """
    if Path(audio_path).suffix.casefold() != ".wav":
        return None

    import numpy as np

    try:
        with wave.open(audio_path, "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            compression = wav_file.getcomptype()
            frames = wav_file.readframes(wav_file.getnframes())
    except (EOFError, OSError, wave.Error):
        return None

    if compression != "NONE" or channels < 1 or sample_rate <= 0:
        return None

    if sample_width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        return None

    if channels > 1:
        if samples.size % channels:
            return None
        samples = samples.reshape(-1, channels).mean(axis=1)

    if sample_rate != 16000 and samples.size:
        output_size = max(1, round(samples.size * 16000 / sample_rate))
        source_positions = np.arange(samples.size, dtype=np.float64)
        target_positions = np.linspace(0, samples.size - 1, output_size)
        samples = np.interp(target_positions, source_positions, samples).astype(np.float32)

    return samples.astype(np.float32, copy=False)


def _normalize_voice_audio(samples: Any) -> Any:
    """Raise quiet microphone speech to a stable level before Whisper.

    Bluetooth hands-free capture on Windows can be valid but extremely quiet.
    VAD has already isolated an utterance by the time this runs, so bounded
    peak normalization improves recognition without changing stored audio.
    """
    import numpy as np

    audio = np.asarray(samples, dtype=np.float32)
    if not audio.size:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak < 0.002 or peak >= 0.5:
        return audio
    gain = min(12.0, 0.5 / peak)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32, copy=False)


def _band_limit_voice_audio(samples: Any, sample_rate: int = 16000) -> Any:
    """Remove frequencies outside the useful speech band before Whisper.

    Windows Bluetooth microphone capture can contain low-frequency rumble
    and near-Nyquist noise that browser capture normally suppresses.  A
    tapered FFT mask keeps the intelligible speech band without adding a
    SciPy dependency or changing the audio duration.
    """
    import numpy as np

    audio = np.asarray(samples, dtype=np.float32)
    if audio.size < 256 or sample_rate <= 0:
        return audio

    frequencies = np.fft.rfftfreq(audio.size, d=1.0 / sample_rate)
    mask = np.ones(frequencies.shape, dtype=np.float64)

    low_stop_hz = 60.0
    low_pass_hz = 120.0
    high_pass_hz = min(7200.0, sample_rate * 0.45)
    high_stop_hz = min(7800.0, sample_rate * 0.49)

    mask[frequencies < low_stop_hz] = 0.0
    low_transition = (frequencies >= low_stop_hz) & (frequencies < low_pass_hz)
    mask[low_transition] = (frequencies[low_transition] - low_stop_hz) / (low_pass_hz - low_stop_hz)

    if high_stop_hz > high_pass_hz:
        high_transition = (frequencies > high_pass_hz) & (frequencies <= high_stop_hz)
        mask[high_transition] = (high_stop_hz - frequencies[high_transition]) / (high_stop_hz - high_pass_hz)
    mask[frequencies > high_stop_hz] = 0.0

    spectrum = np.fft.rfft(audio)
    filtered = np.fft.irfft(spectrum * mask, n=audio.size)
    return filtered.astype(np.float32, copy=False)


def _get_whisper_model(model_name: str) -> Any:
    """Loads (and caches) a Whisper model by name. Previously reloaded from
    disk on every single call — including every ~3s wake-word poll cycle in
    ContinuousVoiceListener's loop — which is the dominant latency cost in
    the whole listen/transcribe cycle. Cached in-process for the life of
    the daemon; different model names (e.g. switching config.voice.
    whisper_model at runtime) just add another cache entry."""
    if model_name not in _whisper_model_cache:
        import whisper

        _whisper_model_cache[model_name] = whisper.load_model(model_name)
    return _whisper_model_cache[model_name]


def _faster_whisper_runtime() -> tuple[str, str]:
    """Choose a conservative local compute mode for Faster-Whisper."""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except (ImportError, RuntimeError):
        pass
    return "cpu", "int8"


def _get_faster_whisper_model(model_name: str) -> Any:
    from faster_whisper import WhisperModel

    device, compute_type = _faster_whisper_runtime()
    cache_key = (model_name, device, compute_type)
    if cache_key not in _faster_whisper_model_cache:
        _faster_whisper_model_cache[cache_key] = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )
    return _faster_whisper_model_cache[cache_key]


async def _transcribe_faster_whisper(
    audio_path: str,
    language: str,
    model_name: str,
) -> dict[str, str]:
    """Transcribe locally through CTranslate2 with command-aware decoding."""

    def _do() -> dict[str, str]:
        model = _get_faster_whisper_model(model_name)
        kwargs: dict[str, Any] = {
            "beam_size": 5,
            "temperature": 0,
            "condition_on_previous_text": False,
            "initial_prompt": _VOICE_TRANSCRIPTION_PROMPT,
            "hotwords": _VOICE_TRANSCRIPTION_PROMPT,
            # Endpointing already isolated the utterance. A second VAD can
            # remove short commands such as "stop" or "close".
            "vad_filter": False,
        }
        if language != "auto":
            kwargs["language"] = language
        segments, info = model.transcribe(audio_path, **kwargs)
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        detected_language = getattr(info, "language", None)
        return {
            "text": text.strip(),
            "language": detected_language or (language if language != "auto" else "en"),
        }

    return await asyncio.to_thread(_do)


async def _transcribe_whisper(
    audio_path: str,
    language: str,
    model_name: str,
) -> dict:
    """Transcribe audio using OpenAI Whisper (local) with multilingual support."""

    def _do():
        mdl = _get_whisper_model(model_name)

        kwargs = {}

        if language != "auto":
            kwargs["language"] = language
        kwargs["temperature"] = 0
        kwargs["beam_size"] = 5
        kwargs["condition_on_previous_text"] = False
        kwargs["initial_prompt"] = _VOICE_TRANSCRIPTION_PROMPT

        local_audio = _load_pcm_wav_for_whisper(audio_path)
        if local_audio is not None:
            local_audio = _band_limit_voice_audio(local_audio)
            local_audio = _normalize_voice_audio(local_audio)
        result = mdl.transcribe(local_audio if local_audio is not None else audio_path, **kwargs)

        return {
            "text": result["text"].strip(),
            "language": result.get(
                "language",
                language if language != "auto" else "en",
            ),
        }

    return await asyncio.to_thread(_do)


async def _transcribe_speech(
    audio_path: str,
    language: str,
    model_name: str,
    engine: str = "auto",
) -> dict[str, str]:
    """Use the requested local ASR engine with a fail-soft auto fallback."""
    normalized_engine = engine.strip().casefold()
    if normalized_engine not in {"auto", "faster_whisper", "openai_whisper"}:
        raise ValueError(f"Unsupported transcription engine: {engine}")

    if normalized_engine in {"auto", "faster_whisper"}:
        try:
            return await _transcribe_faster_whisper(audio_path, language, model_name)
        except ImportError:
            if normalized_engine == "faster_whisper":
                raise
        except Exception as error:
            if normalized_engine == "faster_whisper":
                raise
            logger.warning("Faster-Whisper unavailable; using OpenAI Whisper: %s", error)

    return await _transcribe_whisper(audio_path, language, model_name)


async def _transcribe_windows(audio_path: str) -> str:
    """Transcribe using Windows Speech Recognition."""
    if Path(audio_path).suffix == ".wav":
        txt_content = Path(audio_path).read_text(errors="replace")

        if txt_content.strip() and not txt_content.startswith("RIFF"):
            return txt_content.strip()

    code, out, err = await run_powershell(
        "Add-Type -AssemblyName System.Speech; "
        "$recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine; "
        f"$recognizer.SetInputToWaveFile('{audio_path}'); "
        "$grammar = New-Object System.Speech.Recognition.DictationGrammar; "
        "$recognizer.LoadGrammar($grammar); "
        "$result = $recognizer.Recognize(); "
        "if ($result) { $result.Text } else { 'No speech detected' }; "
        "$recognizer.Dispose()"
    )

    return out.strip() if code == 0 else f"Transcription failed: {err}"


# ── Continuous VAD-based recording ───────────────────────────────────
#
# Replaces the old approach of recording a BLIND fixed-duration window
# (sd.rec(duration) + sd.wait()) per poll cycle: opens one continuous
# sounddevice.InputStream and endpoints utterances on actual silence (see
# pilot.system.vad) instead of guessing a duration. This is also the
# mechanical basis for barge-in: the same per-frame energy check that
# detects "utterance started" while listening also detects "user started
# talking" while Heliox is mid-speech (see speak_interruptible() below).

_RECORDER_STOP = object()


class _ContinuousRecorder:
    """Wraps a single continuous `sounddevice.InputStream` and exposes
    higher-level async waits for "an utterance happened" or "speech
    started" (for barge-in), instead of callers managing raw audio
    themselves. One instance per listener; `start()`/`stop()` open and
    close the actual audio stream."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 50,
        energy_threshold: float = 0.02,
        start_frames: int = 2,
        silence_frames: int = 12,
        max_utterance_seconds: float = 20.0,
        preferred_device: str = "auto",
        adaptive_threshold_enabled: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = max(1, int(sample_rate * frame_ms / 1000))
        self.frame_ms = frame_ms
        self.energy_threshold = energy_threshold
        self.adaptive_threshold_enabled = adaptive_threshold_enabled
        self._adaptive_threshold = AdaptiveEnergyThreshold(energy_threshold)
        self.start_frames = start_frames
        self.silence_frames = silence_frames
        self.max_frames = max(1, int(max_utterance_seconds * 1000 / frame_ms))
        self.preferred_device = preferred_device

        self._stream: Any = None
        self._soundcard_recorder: Any = None
        self._soundcard_thread: threading.Thread | None = None
        self._soundcard_stop: threading.Event | None = None
        self._queue: asyncio.Queue[Any] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.input_device: int | None = None
        self.input_device_name = ""
        self.using_fallback_input = False
        self.input_device_notice = ""
        self.last_error = ""
        self.frames_received = 0
        self.last_frame_rms = 0.0
        self.peak_frame_rms = 0.0
        self.last_above_threshold_at = 0.0
        self.utterances_captured = 0
        self.capture_backend = ""

    def _make_endpointer(self) -> UtteranceEndpointer:
        return UtteranceEndpointer(
            energy_threshold=self.effective_energy_threshold,
            start_frames=self.start_frames,
            silence_frames=self.silence_frames,
            max_frames=self.max_frames,
        )

    def start(self) -> bool:
        """Opens the continuous input stream. Returns False (rather than
        raising) if sounddevice isn't installed or the stream can't open —
        callers fall back to the legacy fixed-duration recording path."""
        if self._stream is not None:
            return True

        self.using_fallback_input = False
        self.input_device_notice = ""
        self._loop = asyncio.get_event_loop()
        self._queue = asyncio.Queue()

        if CURRENT_PLATFORM == Platform.WINDOWS:
            try:
                if self._start_windows_wasapi():
                    return True
            except Exception as error:
                logger.info(
                    "Native Windows microphone capture unavailable; trying PortAudio (%s)",
                    error,
                )

        try:
            import sounddevice as sd
        except ImportError:
            return False

        def _callback(indata, frames, time_info, status):
            # Runs on PortAudio's own thread, not the asyncio loop -- must
            # hand off via call_soon_threadsafe rather than touching the
            # queue directly.
            block = indata[:, 0].copy()
            self._publish_audio_block(block)

        try:
            self.using_fallback_input = False
            self.input_device_notice = ""
            self.input_device = _resolve_input_device(
                sd,
                sample_rate=self.sample_rate,
                channels=1,
                dtype="int16",
                preferred_device=self.preferred_device,
                allow_preferred_fallback=True,
            )
            device_info = sd.query_devices(self.input_device)
            self.input_device_name = str(device_info.get("name", self.input_device))
            configured = self.preferred_device.strip()
            if configured and configured.casefold() != "auto":
                try:
                    hostapi = str(sd.query_hostapis(int(device_info.get("hostapi", -1))).get("name", "Unknown"))
                except Exception:
                    hostapi = "Unknown"
                resolved_id = f"{hostapi}::{self.input_device_name}"
                self.using_fallback_input = resolved_id.casefold() != configured.casefold()
                if self.using_fallback_input:
                    self.input_device_notice = (
                        f"Configured microphone '{configured}' is unavailable; using '{resolved_id}' until it returns."
                    )
                    logger.warning(self.input_device_notice)
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.frame_size,
                callback=_callback,
                device=self.input_device,
            )
            self._stream.start()
            self.capture_backend = "portaudio"
            self.last_error = ""
            return True
        except Exception as exc:
            self.last_error = str(exc)
            logger.warning("Failed to open continuous audio input stream", exc_info=True)
            self._stream = None
            return False

    def _publish_audio_block(self, block: Any) -> None:
        signal_rms = frame_rms(block)
        self.frames_received += 1
        self.last_frame_rms = signal_rms
        self.peak_frame_rms = max(self.peak_frame_rms, signal_rms)
        if self.adaptive_threshold_enabled:
            self._adaptive_threshold.observe(signal_rms)
        if signal_rms >= self.effective_energy_threshold:
            self.last_above_threshold_at = time.time()
        loop = self._loop
        queue = self._queue
        if loop is not None and queue is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(queue.put_nowait, block)
            except RuntimeError:
                # The native capture thread can finish its final blocking
                # read while the asyncio loop is already shutting down.
                # Dropping that late frame is the only valid outcome.
                return

    def _start_windows_wasapi(self) -> bool:
        """Capture through Windows Audio Session API instead of PortAudio."""
        import numpy as np
        import soundcard as sc

        preferred_name = ""
        if self.preferred_device.strip().casefold() != "auto":
            preferred_name = self.preferred_device.rsplit("::", 1)[-1].strip()

        microphones = list(sc.all_microphones(include_loopback=False))
        microphone = next(
            (mic for mic in microphones if preferred_name and mic.name.strip().casefold() == preferred_name.casefold()),
            None,
        )
        if microphone is None:
            microphone = sc.default_microphone()
        if microphone is None:
            return False

        self.using_fallback_input = bool(
            preferred_name and microphone.name.strip().casefold() != preferred_name.casefold()
        )
        if self.using_fallback_input:
            self.input_device_notice = (
                f"Configured microphone '{self.preferred_device.strip()}' is unavailable; "
                f"using '{microphone.name}' until it returns."
            )
            logger.warning(self.input_device_notice)

        recorder_context = microphone.recorder(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.frame_size,
        )
        recorder = recorder_context.__enter__()
        stop_event = threading.Event()

        def _pump() -> None:
            # SoundCard warns on recoverable WASAPI buffer discontinuities. A
            # single warning is useful evidence; hundreds of identical lines
            # hide actual capture failures in the daemon log.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "once",
                    message=r"data discontinuity in recording",
                    category=sc.SoundcardRuntimeWarning,
                )
                try:
                    while not stop_event.is_set():
                        data = recorder.record(numframes=self.frame_size)
                        mono = np.asarray(data[:, 0], dtype=np.float32)
                        block = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
                        self._publish_audio_block(block)
                except Exception as error:
                    self.last_error = str(error)
                    logger.warning("Native Windows microphone stream failed", exc_info=True)

        self._soundcard_recorder = recorder_context
        self._soundcard_stop = stop_event
        self._soundcard_thread = threading.Thread(
            target=_pump,
            name="heliox-wasapi-microphone",
            daemon=True,
        )
        self.input_device = None
        self.input_device_name = microphone.name
        self.capture_backend = "windows_wasapi"
        self.last_error = ""
        # Publish initialized state before the native thread can report an
        # immediate failure; otherwise this assignment can erase its error.
        self._soundcard_thread.start()
        return True

    def stop(self) -> None:
        if self._soundcard_stop is not None:
            self._soundcard_stop.set()
        if self._soundcard_thread is not None:
            self._soundcard_thread.join(timeout=2.0)
        if self._soundcard_recorder is not None:
            try:
                self._soundcard_recorder.__exit__(None, None, None)
            except Exception:
                pass
        self._soundcard_recorder = None
        self._soundcard_stop = None
        self._soundcard_thread = None
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._queue is not None:
            self._queue.put_nowait(_RECORDER_STOP)
        self._queue = None

    @property
    def is_active(self) -> bool:
        return self._stream is not None or self._soundcard_recorder is not None

    @property
    def noise_floor_rms(self) -> float | None:
        if not self.adaptive_threshold_enabled:
            return None
        return self._adaptive_threshold.noise_floor

    @property
    def effective_energy_threshold(self) -> float:
        if not self.adaptive_threshold_enabled:
            return self.energy_threshold
        return self._adaptive_threshold.threshold

    async def wait_until_receiving(self, timeout: float = 1.0) -> bool:
        """Confirm an opened backend is actually delivering audio frames.

        Windows can expose disconnected WDM-KS endpoints that accept an input
        stream but never invoke its callback. Treating those endpoints as live
        leaves the companion listening forever without hearing anything.
        """
        starting_frames = self.frames_received
        deadline = asyncio.get_running_loop().time() + max(0.0, timeout)
        while asyncio.get_running_loop().time() < deadline:
            if self.frames_received > starting_frames:
                return True
            await asyncio.sleep(0.05)
        return self.frames_received > starting_frames

    async def _drain_stale_frames(self, queue: asyncio.Queue[Any]) -> None:
        """Discards any frames queued while nobody was waiting, so a fresh
        wait starts listening from "now" instead of replaying a backlog."""
        while not queue.empty():
            queue.get_nowait()

    async def wait_for_speech_start(self, timeout: float | None = None) -> bool:
        """Waits until sustained speech is detected (used for barge-in:
        the caller cares only that the user started talking, not the full
        utterance). Returns False on timeout or if the stream isn't
        active."""
        queue = self._queue
        if queue is None:
            return False

        await self._drain_stale_frames(queue)
        endpointer = self._make_endpointer()
        deadline = None if timeout is None else asyncio.get_event_loop().time() + timeout

        while True:
            remaining = None if deadline is None else deadline - asyncio.get_event_loop().time()
            if remaining is not None and remaining <= 0:
                return False
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return False

            if frame is _RECORDER_STOP:
                return False

            if endpointer.push(frame_rms(frame)) == EndpointEvent.STARTED:
                return True

    async def record_utterance(self, timeout: float | None = None) -> str | None:
        """Waits for a full utterance (speech start through endpoint) and
        returns the path to a WAV file containing it, or None if nothing
        was captured before `timeout`."""
        queue = self._queue
        if queue is None:
            return None

        await self._drain_stale_frames(queue)
        endpointer = self._make_endpointer()
        deadline = None if timeout is None else asyncio.get_event_loop().time() + timeout
        # Small pre-roll so the moment speech is CONFIRMED (after
        # start_frames) doesn't clip the first syllable that triggered it.
        preroll: list[Any] = []
        captured: list[Any] = []

        while True:
            remaining = None if deadline is None else deadline - asyncio.get_event_loop().time()
            if remaining is not None and remaining <= 0:
                return None
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return None

            if frame is _RECORDER_STOP:
                return None

            event = endpointer.push(frame_rms(frame))

            # Check event states before is_speaking: push() already called
            # reset() internally on ENDED/MAX_DURATION, so is_speaking is
            # back to False by the time we get here — checking is_speaking
            # first would misroute the utterance's final frame into the
            # preroll buffer instead of appending it and returning.
            if event == EndpointEvent.STARTED:
                captured = preroll + [frame]
                preroll = []
                continue

            if event in (EndpointEvent.ENDED, EndpointEvent.MAX_DURATION):
                captured.append(frame)
                self.utterances_captured += 1
                return _write_wav(captured, self.sample_rate)

            if not endpointer.is_speaking:
                preroll.append(frame)
                preroll_frames = max(self.start_frames + 1, int(300 / self.frame_ms))
                if len(preroll) > preroll_frames:
                    preroll.pop(0)
                continue

            captured.append(frame)


def _write_wav(frames: list[Any], sample_rate: int) -> str:
    import numpy as np

    output_path = os.path.join(tempfile.gettempdir(), f"pilot_utterance_{os.getpid()}_{id(frames)}.wav")
    audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.int16)

    with wave.open(output_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    return output_path


# ── Continuous Voice Listener (JARVIS Mode) ──────────────────────────


class ContinuousVoiceListener:
    """Always-on microphone listener with wake word detection."""

    def __init__(
        self,
        wake_words: list[str] | None = None,
        on_command: Any | None = None,
        on_status: Any | None = None,
        workflow_control: Any | None = None,
        config: PilotConfig | None = None,
    ) -> None:
        self.wake_words = wake_words or [
            "hey heliox",
            "heliox",
            "hey pilot",
        ]

        self._on_command = on_command
        self._on_status = on_status
        # Optional async callable(command_text) -> bool, checked before normal
        # command dispatch — lets a paused/waiting VoiceGestureWorkflow claim
        # a "continue"/"cancel" utterance instead of it being planned as a
        # brand-new command. See VoiceGestureWorkflowEngine.handle_control_phrase.
        self._workflow_control = workflow_control
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._transcriber_warmup_task: asyncio.Task[None] | None = None
        self._command_tasks: set[asyncio.Task[None]] = set()
        self._diagnostic_waiter: asyncio.Future[dict[str, Any]] | None = None
        self._listening_for_command = False
        self._follow_up_until = 0.0
        self._wake_free_suppression_depth = 0
        self._wake_free_suppression_generation = 0
        self._sample_rate = 16000

        # The daemon passes its live config instance so Settings changes take
        # effect immediately. Standalone callers retain the disk-load fallback.
        self.config = config or PilotConfig.load()
        self.last_detected_language = "en"
        self.last_transcript = ""
        self.transcripts_received = 0
        self.last_transcription_ms = 0.0
        self.last_command_pipeline_ms = 0.0
        # On-device wake-word calibration (continual-learning loop) — see
        # voice_calibration.py. Only ever a fallback tried after the fixed
        # exact-match loop below misses; the common case is untouched.
        self._wake_calibrator = WakeWordCalibrator(self.wake_words)

        # Continuous VAD-based recorder (see pilot.system.vad) — replaces
        # blind fixed-duration recording windows with natural start/stop
        # endpointing. Falls back to the legacy _record_and_transcribe()
        # fixed-duration path if sounddevice isn't installed or the input
        # stream fails to open (see start()).
        self._recorder = _ContinuousRecorder(
            sample_rate=self._sample_rate,
            energy_threshold=self.config.voice.vad_energy_threshold,
            silence_frames=max(1, int(self.config.voice.vad_silence_ms / 50)),
            max_utterance_seconds=self.config.voice.vad_max_utterance_seconds,
            preferred_device=self.config.voice.input_device,
            adaptive_threshold_enabled=self.config.voice.vad_adaptive_threshold_enabled,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def follow_up_remaining_seconds(self) -> float:
        """Seconds left in the bounded wake-free conversation turn."""
        return max(0.0, self._follow_up_until - time.monotonic())

    def arm_follow_up_window(self, seconds: float | None = None) -> None:
        """Allow exactly the next utterance to omit the wake word briefly.

        The server calls this only after Heliox has finished speaking, which
        avoids treating TTS echo as user intent. Accepting an utterance consumes
        the window; a completed response arms a fresh turn.
        """
        configured = self.config.voice.follow_up_window_seconds if seconds is None else seconds
        duration = max(0.0, min(float(configured), 120.0))
        self._follow_up_until = time.monotonic() + duration if duration else 0.0

    def clear_follow_up_window(self) -> None:
        self._follow_up_until = 0.0

    def suppress_wake_free_commands(self) -> None:
        """Block wake-free routing while Heliox is speaking.

        Wake-word barge-in remains available. Incrementing the generation also
        invalidates an utterance whose capture overlapped TTS, even if speech
        finished before transcription returned.
        """
        self._wake_free_suppression_depth += 1
        self._wake_free_suppression_generation += 1

    def resume_wake_free_commands(self) -> None:
        self._wake_free_suppression_depth = max(0, self._wake_free_suppression_depth - 1)

    async def start(self) -> str:
        if self._running:
            return "Voice listener is already running."

        self._running = True
        recorder_active = self._recorder.start()
        if not recorder_active:
            self._running = False
            detail = self._recorder.last_error or "No usable microphone input is available."
            logger.error("Continuous voice listener could not start: %s", detail)
            return f"Voice listener could not start: {detail}"

        if not await self._recorder.wait_until_receiving():
            self._running = False
            self._recorder.last_error = (
                f"Microphone '{self._recorder.input_device_name or 'selected input'}' opened "
                "but delivered no audio. Connect or choose an active input in Settings."
            )
            detail = self._recorder.last_error
            self._recorder.stop()
            logger.warning("Continuous voice listener could not start: %s", detail)
            return f"Voice listener could not start: {detail}"

        self._task = asyncio.create_task(self._listen_loop())
        self._transcriber_warmup_task = asyncio.create_task(self._warm_transcriber())

        logger.info(
            "Continuous voice listener started (wake words: %s, vad_recorder=%s)",
            self.wake_words,
            recorder_active,
        )

        if self.config.voice.continuous_conversation_enabled:
            return "Continuous voice listener started. Speak naturally; the wake phrase is optional."
        return f"Voice listener started. Say '{self.wake_words[0]}' to activate."

    async def _warm_transcriber(self) -> None:
        """Load the configured local recognizer before the first utterance."""

        started = time.perf_counter()
        try:
            if self.config.voice.transcription_engine in {"auto", "faster_whisper"}:
                await asyncio.to_thread(_get_faster_whisper_model, self.config.voice.whisper_model)
            else:
                await asyncio.to_thread(_get_whisper_model, self.config.voice.whisper_model)
            logger.info(
                "Voice recognizer warmup completed (model=%s, duration_ms=%.0f)",
                self.config.voice.whisper_model,
                (time.perf_counter() - started) * 1000,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # Normal transcription retains its Windows fallback path.
            logger.warning("Voice recognizer warmup unavailable: %s", error)

    def adopt_command_tasks(self, tasks: set[asyncio.Task[None]]) -> None:
        """Take over commands still running from a listener this one replaced."""
        for task in tasks:
            if not task.done():
                self._command_tasks.add(task)
                task.add_done_callback(self._command_tasks.discard)

    async def stop(self, *, cancel_commands: bool = True) -> str:
        """Stop listening; ``cancel_commands=False`` leaves commands it already started running.

        Replacing the listener after a microphone setting change must not kill a
        voice-started task in the middle of its work.
        """
        self._running = False
        self._recorder.stop()

        if self._diagnostic_waiter and not self._diagnostic_waiter.done():
            self._diagnostic_waiter.cancel()
        self._diagnostic_waiter = None

        if self._task:
            self._task.cancel()

            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if self._transcriber_warmup_task and not self._transcriber_warmup_task.done():
            self._transcriber_warmup_task.cancel()
            await asyncio.gather(self._transcriber_warmup_task, return_exceptions=True)
        self._transcriber_warmup_task = None

        current = asyncio.current_task()
        command_tasks = tuple(task for task in self._command_tasks if task is not current)
        if cancel_commands:
            for task in command_tasks:
                task.cancel()
            if command_tasks:
                await asyncio.gather(*command_tasks, return_exceptions=True)

        logger.info("Continuous voice listener stopped")

        return "Voice listener stopped."

    async def capture_diagnostic(self, timeout_seconds: float = 12.0) -> dict[str, Any]:
        """Return one recognized utterance without routing it as a command."""
        if not self._running:
            raise RuntimeError("Voice listener is not running.")
        if self._diagnostic_waiter and not self._diagnostic_waiter.done():
            raise RuntimeError("A microphone recognition test is already running.")

        waiter: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._diagnostic_waiter = waiter
        try:
            return await asyncio.wait_for(waiter, timeout=max(1.0, float(timeout_seconds)))
        finally:
            if self._diagnostic_waiter is waiter:
                self._diagnostic_waiter = None

    async def _listen_loop(self) -> None:
        while self._running:
            try:
                # Ambient wake-word listening: no timeout on the VAD path —
                # it waits for the next actual utterance instead of polling
                # a blind fixed window every cycle.
                capture_generation = self._wake_free_suppression_generation
                transcript = await self._record_and_transcribe(duration=3, timeout=None)

                if not transcript or transcript.strip() == "No speech detected":
                    await asyncio.sleep(0.2)
                    continue

                self.last_transcript = transcript.strip()
                self.transcripts_received += 1
                transcript_lower = transcript.lower().strip()

                if self._diagnostic_waiter and not self._diagnostic_waiter.done():
                    self._diagnostic_waiter.set_result(
                        {
                            "transcript": self.last_transcript,
                            "language": self.last_detected_language,
                            "latency_ms": round(self.last_transcription_ms),
                        }
                    )
                    logger.info("Recognition test captured an utterance without dispatching it")
                    continue

                logger.debug("Heard: %s", transcript_lower)

                wake_detected = False
                command_text = transcript.strip()
                wake_free_allowed = (
                    self._wake_free_suppression_depth == 0
                    and capture_generation == self._wake_free_suppression_generation
                )
                conversational_follow_up = wake_free_allowed and (
                    self.config.voice.continuous_conversation_enabled or self.follow_up_remaining_seconds > 0
                )

                for wake in self.wake_words:
                    if wake in transcript_lower:
                        wake_detected = True
                        command_text = _remove_phrase_case_insensitive(transcript, wake)
                        break

                if wake_detected:
                    # A real hit confirms any near-miss variant that was
                    # pending from a recent failed wake attempt — see
                    # voice_calibration.py.
                    self._wake_calibrator.confirm_pending_if_followed_by_hit()
                elif self.config.adaptive_calibration.voice_wake_word_enabled:
                    variant_hit = self._wake_calibrator.match_promoted_variant(transcript_lower)
                    if isinstance(variant_hit, str) and variant_hit:
                        wake_detected = True
                        command_text = _remove_phrase_case_insensitive(transcript, variant_hit)
                        self._wake_calibrator.confirm_pending_if_followed_by_hit()
                    else:
                        near_miss = self._wake_calibrator.check_near_miss(transcript_lower)
                        if isinstance(near_miss, str) and near_miss:
                            # Whisper commonly renders a product wake word
                            # phonetically (for example "Heliocs"). When a
                            # close leading phrase is immediately followed by
                            # a real command, accept it for this utterance.
                            # The command still passes through the normal
                            # voice policy/permission gates. A wake-only
                            # near-miss keeps the conservative repeated-
                            # confirmation calibration behavior.
                            trailing_command = transcript[len(near_miss) :].strip(" \t,.:;-")
                            if len(trailing_command) >= 3:
                                wake_detected = True
                                command_text = trailing_command
                                logger.info(
                                    "Accepted close wake-word transcription: '%s'",
                                    near_miss,
                                )
                            else:
                                self._wake_calibrator.record_pending(near_miss)

                self._wake_calibrator.tick()

                if not wake_detected and not conversational_follow_up:
                    await asyncio.sleep(0.1)
                    continue

                if conversational_follow_up and not wake_detected:
                    if self.config.voice.continuous_conversation_enabled and not _looks_like_autonomous_voice_intent(
                        transcript_lower
                    ):
                        logger.debug("Ignored ambient non-command speech: '%s'", transcript_lower)
                        if self._on_status:
                            try:
                                await self._on_status(
                                    "ambient_ignored",
                                    {"transcript": transcript, "reason": "no command intent"},
                                )
                            except Exception:
                                pass
                        continue
                    # Consume before dispatch so room speech cannot enqueue
                    # multiple wake-free commands while this one is running.
                    self.clear_follow_up_window()
                    command_text = transcript.strip(" \t,.:;!?-")
                    logger.info("Conversational voice follow-up detected: '%s'", command_text)

                logger.info(
                    "%s Command: '%s'",
                    "Conversational follow-up detected."
                    if conversational_follow_up and not wake_detected
                    else "Wake word detected!",
                    command_text,
                )

                if self._on_status:
                    try:
                        await self._on_status(
                            (
                                "follow_up_detected"
                                if conversational_follow_up and not wake_detected
                                else "wake_detected"
                            ),
                            {
                                "transcript": transcript,
                                "wake_free": conversational_follow_up and not wake_detected,
                            },
                        )
                    except Exception:
                        pass

                if not command_text or len(command_text) < 3:
                    if self._on_status:
                        try:
                            await self._on_status(
                                "listening",
                                {"message": "I'm listening..."},
                            )
                        except Exception:
                            pass

                    command_text = await self._record_and_transcribe(duration=8, timeout=8.0)

                    if not command_text or command_text.strip() == "No speech detected":
                        if self._on_status:
                            try:
                                await self._on_status(
                                    "timeout",
                                    {"message": ("Didn't catch that.")},
                                )
                            except Exception:
                                pass

                        continue

                if self._workflow_control:
                    try:
                        consumed = await self._workflow_control(command_text)
                    except Exception:
                        logger.warning("Workflow control-phrase check failed (non-fatal)", exc_info=True)
                        consumed = False
                    if consumed:
                        continue

                logger.info(
                    "Dispatching voice command: '%s'",
                    command_text,
                )

                if self._on_command:
                    task = asyncio.create_task(self._dispatch_command(command_text))
                    self._command_tasks.add(task)
                    task.add_done_callback(self._command_tasks.discard)
                    # Give the newly scheduled command a chance to publish its
                    # acknowledgement/state before the next capture cycle.
                    # Real transcription already yields, but this also avoids
                    # starving dispatch when a local recognizer returns
                    # immediately.
                    await asyncio.sleep(0)

            except asyncio.CancelledError:
                break

            except Exception:
                logger.debug(
                    "Voice listener error",
                    exc_info=True,
                )
                await asyncio.sleep(1.0)

    async def _dispatch_command(self, command_text: str) -> None:
        """Run a recognized command without pausing ambient listening."""
        started = time.perf_counter()
        try:
            await self._on_command(command_text)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.error("Voice command dispatch failed: %s", error)
        finally:
            self.last_command_pipeline_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "Voice command pipeline completed (duration_ms=%.0f, command=%r)",
                self.last_command_pipeline_ms,
                command_text[:80],
            )

    async def _transcribe_path(self, audio_path: str) -> str:
        """Transcribes an already-recorded WAV file with multilingual support."""
        try:
            result = await _transcribe_speech(
                audio_path,
                self.config.voice.language,
                self.config.voice.whisper_model,
                self.config.voice.transcription_engine,
            )

            self.last_detected_language = result["language"]

            return result["text"]

        except ImportError:
            pass

        if CURRENT_PLATFORM == Platform.WINDOWS:
            return await _transcribe_windows(audio_path)

        return ""

    async def _record_and_transcribe(
        self,
        duration: int = 3,
        timeout: float | None = None,
    ) -> str:
        """Records one utterance and transcribes it with multilingual support.

        Prefers the continuous VAD-based recorder (natural start/stop
        endpointing on actual silence — see pilot.system.vad) when its
        input stream is active; falls back to the legacy fixed-`duration`
        recording window otherwise (e.g. sounddevice missing or the stream
        failed to open). `timeout` only applies to the VAD path — `None`
        waits indefinitely for the next utterance (the ambient
        wake-word-listening case); a finite value bounds how long to wait
        for the command that follows a detected wake word.
        """
        try:
            if self._recorder.is_active:
                audio_path = await self._recorder.record_utterance(timeout=timeout)
                if not audio_path:
                    return ""
            else:
                audio_path = await _record_audio(duration)

            started = time.perf_counter()
            transcript = await self._transcribe_path(audio_path)
            self.last_transcription_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "Voice transcription completed (duration_ms=%.0f, characters=%d)",
                self.last_transcription_ms,
                len(transcript.strip()),
            )
            return transcript

        except Exception as e:
            logger.debug(
                "Record/transcribe failed: %s",
                e,
            )
            return ""

    def get_stats(self) -> dict:
        """Return listener statistics."""
        return {
            "running": self._running,
            "wake_words": self.wake_words,
            "listening_for_command": self._listening_for_command,
            "language": self.last_detected_language,
            "configured_language": self.config.voice.language,
            "transcription_engine": self.config.voice.transcription_engine,
            "whisper_model": self.config.voice.whisper_model,
            "vad_recorder_active": self._recorder.is_active,
            "input_device": self._recorder.input_device,
            "input_device_name": self._recorder.input_device_name,
            "configured_input_device": self._recorder.preferred_device,
            "using_fallback_input": self._recorder.using_fallback_input,
            "input_device_notice": self._recorder.input_device_notice,
            "input_error": self._recorder.last_error,
            "capture_backend": self._recorder.capture_backend,
            "energy_threshold": self._recorder.energy_threshold,
            "adaptive_threshold_enabled": self._recorder.adaptive_threshold_enabled,
            "effective_energy_threshold": self._recorder.effective_energy_threshold,
            "noise_floor_rms": self._recorder.noise_floor_rms,
            "frames_received": self._recorder.frames_received,
            "last_frame_rms": self._recorder.last_frame_rms,
            "peak_frame_rms": self._recorder.peak_frame_rms,
            "last_above_threshold_at": self._recorder.last_above_threshold_at,
            "utterances_captured": self._recorder.utterances_captured,
            "transcripts_received": self.transcripts_received,
            "last_transcript": self.last_transcript,
            "last_transcription_ms": round(self.last_transcription_ms),
            "last_command_pipeline_ms": round(self.last_command_pipeline_ms),
            "follow_up_active": self.follow_up_remaining_seconds > 0,
            "follow_up_remaining_seconds": round(self.follow_up_remaining_seconds, 1),
            "continuous_conversation_enabled": self.config.voice.continuous_conversation_enabled,
            "wake_free_suppressed": self._wake_free_suppression_depth > 0,
        }


# Legacy function — now delegates to ContinuousVoiceListener
async def start_wake_word_listener(
    wake_word: str = "hey heliox",
    callback_command: str = "",
) -> str:
    """Start listening for a wake word in the background."""
    return (
        f"Wake word listener configured for '{wake_word}'. "
        "Use the voice_listener_start endpoint for continuous JARVIS-mode listening."
    )
