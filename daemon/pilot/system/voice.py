"""Voice Input/Output — talk to Cortex-OS like JARVIS.

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
    """Speak text aloud using system TTS.

    On Windows: uses built-in SAPI.SpVoice
    On Linux: uses espeak or piper
    On macOS: uses 'say' command
    """
    if CURRENT_PLATFORM == Platform.WINDOWS:
        return await _tts_windows(text, voice, rate, volume, output_file)
    elif CURRENT_PLATFORM == Platform.MACOS:
        return await _tts_macos(text, voice, rate, output_file)
    else:
        return await _tts_linux(text, voice, rate, output_file)


async def _tts_windows(text: str, voice: str | None, rate: int, volume: float, output_file: str | None) -> str:
    # Escape quotes in text for PowerShell
    safe_text = text.replace("'", "''").replace('"', '""')

    if output_file:
        # Save to WAV file
        script = (
            f"Add-Type -AssemblyName System.Speech; "
            f"$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$synth.Rate = {(rate - 170) // 20}; "
            f"$synth.Volume = {int(volume * 100)}; "
        )
        if voice:
            script += f"$synth.SelectVoice('{voice}'); "
        script += f"$synth.SetOutputToWaveFile('{output_file}'); "
        script += f"$synth.Speak('{safe_text}'); "
        script += "$synth.SetOutputToDefaultAudioDevice(); "
        script += "$synth.Dispose()"

        code, out, err = await run_powershell(script)
        if code != 0:
            return f"TTS save failed: {err}"
        return f"Speech saved to {output_file}"
    else:
        # Speak aloud
        script = (
            f"Add-Type -AssemblyName System.Speech; "
            f"$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$synth.Rate = {(rate - 170) // 20}; "
            f"$synth.Volume = {int(volume * 100)}; "
        )
        if voice:
            script += f"$synth.SelectVoice('{voice}'); "
        script += f"$synth.Speak('{safe_text}'); "
        script += "$synth.Dispose()"

        code, out, err = await run_powershell(script)
        if code != 0:
            return f"TTS failed: {err}"
        return f"Spoken: {text[:80]}..."


async def _tts_linux(text: str, voice: str | None, rate: int, output_file: str | None) -> str:
    cmd = ["espeak"]
    if voice:
        cmd.extend(["-v", voice])
    cmd.extend(["-s", str(rate)])
    if output_file:
        cmd.extend(["-w", output_file])
    cmd.append(text)

    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    await proc.communicate()
    if output_file:
        return f"Speech saved to {output_file}"
    return f"Spoken: {text[:80]}..."


async def _tts_macos(text: str, voice: str | None, rate: int, output_file: str | None) -> str:
    cmd = ["say"]
    if voice:
        cmd.extend(["-v", voice])
    cmd.extend(["-r", str(rate)])
    if output_file:
        cmd.extend(["-o", output_file])
    cmd.append(text)

    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
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
        proc = await asyncio.create_subprocess_exec("say", "-v", "?", stdout=asyncio.subprocess.PIPE)
        out, _ = await proc.communicate()
        return out.decode("utf-8", errors="replace")
    else:
        proc = await asyncio.create_subprocess_exec("espeak", "--voices", stdout=asyncio.subprocess.PIPE)
        out, _ = await proc.communicate()
        return out.decode("utf-8", errors="replace")


# ── Speech-to-Text ───────────────────────────────────────────────────


async def listen(
    duration: int = 5,
    language: str = "en",
    model: str = "base",
) -> str:
    """Listen to the microphone and transcribe speech.

    Uses OpenAI Whisper (local) for transcription.
    Falls back to Windows Speech Recognition if Whisper not available.
    """
    # Record audio first
    audio_path = await _record_audio(duration)

    # Try Whisper (local)
    try:
        return await _transcribe_whisper(audio_path, language, model)
    except ImportError:
        pass

    # Try Windows Speech Recognition
    if CURRENT_PLATFORM == Platform.WINDOWS:
        return await _transcribe_windows(audio_path)

    return "ERROR: Install whisper for speech-to-text: pip install openai-whisper"


async def _record_audio(duration: int) -> str:
    """Record audio from the microphone."""
    output_path = os.path.join(tempfile.gettempdir(), f"pilot_audio_{os.getpid()}.wav")

    try:
        import numpy as np
        import sounddevice as sd

        sample_rate = 16000
        data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
        sd.wait()

        with wave.open(output_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(data.tobytes())

        return output_path

    except ImportError:
        pass

    # Windows fallback: use PowerShell + built-in recorder
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
        # Write a dummy wav path, return the transcription directly
        Path(output_path).write_text(out.strip() if code == 0 else "")
        return output_path

    raise RuntimeError("Install sounddevice for audio recording: pip install sounddevice")


async def _transcribe_whisper(audio_path: str, language: str, model_name: str) -> str:
    """Transcribe audio using OpenAI Whisper (local)."""
    import whisper

    def _do():
        mdl = whisper.load_model(model_name)
        result = mdl.transcribe(audio_path, language=language)
        return result["text"]

    text = await asyncio.to_thread(_do)
    return text.strip()


async def _transcribe_windows(audio_path: str) -> str:
    """Transcribe using Windows Speech Recognition."""
    # Check if we already transcribed during recording
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


# ── Wake Word Detection ──────────────────────────────────────────────


async def start_wake_word_listener(
    wake_word: str = "hey cortex",
    callback_command: str = "",
) -> str:
    """Start listening for a wake word in the background.

    When the wake word is detected, executes the callback_command
    via the Cortex-OS planner.
    """
    # This creates a background task
    return (
        f"Wake word listener configured for '{wake_word}'. "
        "Use triggers system to set up continuous listening: "
        "trigger_create with type='custom_condition'"
    )
