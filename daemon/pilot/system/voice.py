"""Voice Input/Output — talk to Heliox OS like JARVIS.

Speech-to-text via Whisper (local or API), text-to-speech via
system TTS or edge-tts, and optional wake word detection.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import wave
from pathlib import Path
from typing import Any

from pilot.config import PilotConfig
from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_powershell

logger = logging.getLogger("pilot.system.voice")


# ── Text-to-Speech ───────────────────────────────────────────────────


async def speak(
    text: str,
    voice: str | None = None,
    rate: int = 170,
    volume: float = 1.0,
    output_file: str | None = None,
) -> str:
    """Speak text aloud using system TTS."""

    try:
        from pilot.cognitive.tribe_engine import TribeEngine

        tribe = TribeEngine.get_instance()
        if tribe.is_loaded and hasattr(tribe, "_last_cognitive_load"):
            cog_load = tribe._last_cognitive_load
            if cog_load > 0.6:
                reduction = int((cog_load - 0.5) * 80)
                rate = max(100, rate - reduction)
                logger.info(
                    f"Modulating voice rate to {rate} due to cognitive load {cog_load:.2f}"
                )
    except Exception as e:
        logger.debug(f"Failed to modulate TTS rate by cognitive load: {e}")

    if CURRENT_PLATFORM == Platform.WINDOWS:
        return await _tts_windows(text, voice, rate, volume, output_file)
    elif CURRENT_PLATFORM == Platform.MACOS:
        return await _tts_macos(text, voice, rate, output_file)
    else:
        return await _tts_linux(text, voice, rate, output_file)


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

    await proc.communicate()

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

    await proc.communicate()

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
        result = await _transcribe_whisper(audio_path, language, model)
        return result["text"]

    except ImportError:
        pass

    if CURRENT_PLATFORM == Platform.WINDOWS:
        return await _transcribe_windows(audio_path)

    return "ERROR: Install whisper for speech-to-text: pip install openai-whisper"


async def _record_audio(duration: int) -> str:
    """Record audio from the microphone."""
    output_path = os.path.join(
        tempfile.gettempdir(),
        f"pilot_audio_{os.getpid()}.wav",
    )

    try:
        import sounddevice as sd

        sample_rate = 16000

        data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
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

    raise RuntimeError(
        "Install sounddevice for audio recording: pip install sounddevice"
    )


async def _transcribe_whisper(
    audio_path: str,
    language: str,
    model_name: str,
) -> dict:
    """Transcribe audio using OpenAI Whisper (local) with multilingual support."""
    import whisper

    def _do():
        mdl = whisper.load_model(model_name)

        kwargs = {}

        if language != "auto":
            kwargs["language"] = language

        result = mdl.transcribe(audio_path, **kwargs)

        return {
            "text": result["text"].strip(),
            "language": result.get(
                "language",
                language if language != "auto" else "en",
            ),
        }

    return await asyncio.to_thread(_do)


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


# ── Continuous Voice Listener (JARVIS Mode) ──────────────────────────


class ContinuousVoiceListener:
    """Always-on microphone listener with wake word detection."""

    def __init__(
        self,
        wake_words: list[str] | None = None,
        on_command: Any | None = None,
        on_status: Any | None = None,
    ) -> None:
        self.wake_words = wake_words or [
            "hey heliox",
            "heliox",
            "hey pilot",
        ]

        self._on_command = on_command
        self._on_status = on_status
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._listening_for_command = False
        self._sample_rate = 16000
        self._vad_enabled = False

        self.config = PilotConfig.load()
        self.last_detected_language = "en"

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> str:
        if self._running:
            return "Voice listener is already running."

        self._running = True
        self._task = asyncio.create_task(self._listen_loop())

        logger.info(
            "Continuous voice listener started (wake words: %s)",
            self.wake_words,
        )

        return (
            f"Voice listener started. "
            f"Say '{self.wake_words[0]}' to activate."
        )

    async def stop(self) -> str:
        self._running = False

        if self._task:
            self._task.cancel()

            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("Continuous voice listener stopped")

        return "Voice listener stopped."

    async def _listen_loop(self) -> None:
        while self._running:
            try:
                transcript = await self._record_and_transcribe(duration=3)

                if not transcript or transcript.strip() == "No speech detected":
                    await asyncio.sleep(0.2)
                    continue

                transcript_lower = transcript.lower().strip()

                logger.debug("Heard: %s", transcript_lower)

                wake_detected = False
                command_text = transcript_lower

                for wake in self.wake_words:
                    if wake in transcript_lower:
                        wake_detected = True
                        command_text = transcript_lower.replace(
                            wake,
                            "",
                        ).strip()
                        break

                if not wake_detected:
                    await asyncio.sleep(0.1)
                    continue

                logger.info(
                    "Wake word detected! Command: '%s'",
                    command_text,
                )

                if self._on_status:
                    try:
                        await self._on_status(
                            "wake_detected",
                            {"transcript": transcript},
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

                    command_text = await self._record_and_transcribe(
                        duration=8
                    )

                    if (
                        not command_text
                        or command_text.strip() == "No speech detected"
                    ):
                        if self._on_status:
                            try:
                                await self._on_status(
                                    "timeout",
                                    {
                                        "message": (
                                            "Didn't catch that."
                                        )
                                    },
                                )
                            except Exception:
                                pass

                        continue

                logger.info(
                    "Dispatching voice command: '%s'",
                    command_text,
                )

                if self._on_command:
                    try:
                        await self._on_command(command_text)
                    except Exception as e:
                        logger.error(
                            "Voice command dispatch failed: %s",
                            e,
                        )

            except asyncio.CancelledError:
                break

            except Exception:
                logger.debug(
                    "Voice listener error",
                    exc_info=True,
                )
                await asyncio.sleep(1.0)

    async def _record_and_transcribe(
        self,
        duration: int = 3,
    ) -> str:
        """Record audio and transcribe it with multilingual support."""
        try:
            audio_path = await _record_audio(duration)

            try:
                result = await _transcribe_whisper(
                    audio_path,
                    self.config.voice.language,
                    self.config.voice.whisper_model,
                )

                self.last_detected_language = result["language"]

                return result["text"]

            except ImportError:
                pass

            if CURRENT_PLATFORM == Platform.WINDOWS:
                return await _transcribe_windows(audio_path)

            return ""

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
            "whisper_model": self.config.voice.whisper_model,
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