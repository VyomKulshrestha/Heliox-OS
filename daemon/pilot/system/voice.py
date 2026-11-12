"""Voice Input/Output — talk to Heliox OS like JARVIS.

Speech-to-text via Whisper (local or API), text-to-speech via
system TTS or edge-tts, and optional wake word detection.
"""

from __future__ import annotations

import asyncio
import contextlib
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
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from pilot.config import PilotConfig
from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_powershell
from pilot.system.vad import AdaptiveEnergyThreshold, EndpointEvent, UtteranceEndpointer, frame_rms
from pilot.system.voice_calibration import WakeWordCalibrator

logger = logging.getLogger("pilot.system.voice")

# When the person's last word of the utterance being handled ended (time.monotonic(), from the capture thread's
# clock), set for the whole of each voice dispatch. Latency to Heliox's first sound is measured from here:
#     from pilot.system.voice import voice_turn_ended_at
#     waited = time.monotonic() - (voice_turn_ended_at.get() or time.monotonic())
voice_turn_ended_at: ContextVar[float | None] = ContextVar("voice_turn_ended_at", default=None)

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


# What a person says back to Heliox in an open conversation: corrections, decisions and short replies. Anything
# else said while the conversation is open is overheard (Live may check it) but never becomes a command.
_CONVERSATION_TURN_START = re.compile(
    r"^(?:skip|don'?t|do\s+not|wait|hold\s+on|no|nope|not\s+(?:that|now|this|yet)|instead|actually|undo|stop|"
    r"cancel|never\s*mind|what\s+about)\b",
    re.IGNORECASE,
)
# Acknowledgements count only as a whole short reply: "Okay, thanks" is to Heliox, "okay so anyway…" is not.
_SHORT_REPLY_START = re.compile(r"^(?:yes|yeah|yep|sure|okay|ok|thanks|thank\s+you|great|perfect)\b", re.IGNORECASE)
_SHORT_REPLY_WORDS = 3


def looks_like_conversation_turn(text: str) -> bool:
    """A command, or a reply a person gives Heliox mid-conversation ("Skip the backup.", "Wait, stop", "No")."""
    normalized = " ".join(text.casefold().replace(",", " ").strip(" \t.:;!?-").split())
    if not normalized:
        return False
    if (
        _looks_like_autonomous_voice_intent(normalized)
        or _CONVERSATION_TURN_START.match(normalized)
        or normalized in STOP_TALKING_PHRASES
    ):
        return True
    return len(normalized.split()) <= _SHORT_REPLY_WORDS and bool(_SHORT_REPLY_START.match(normalized))


# Words that are safe to act on from the fast, less accurate recogniser: they can only stop Heliox or deny.
_SAFE_FAST_PHRASES = frozenset(
    {"no", "nope", "stop", "stop it", "cancel", "wait", "wait stop", "wait wait", "hold on", "no stop", "never mind"}
)
# Ordinary ways to tell Heliox to stop talking, as whole utterances only ("that's enough for the backup folder,
# now zip it" is a request, not a stop).
STOP_TALKING_PHRASES = frozenset(
    {
        "that's enough",
        "okay that's enough",
        "ok that's enough",
        "enough",
        "got it",
        "okay okay",
        "ok ok",
        "okay got it",
        "stop there",
        "hold on",
        "alright that's enough",
    }
)


def is_stop_talking(text: str) -> bool:
    return " ".join(text.casefold().replace(",", " ").strip(" \t.:;!?-").split()) in STOP_TALKING_PHRASES


def is_safe_fast_phrase(text: str) -> bool:
    """A whole utterance that only stops or denies, so a mishearing costs nothing ("No.", "Wait, stop")."""
    return " ".join(text.casefold().replace(",", " ").strip(" \t.:;!?-").split()) in _SAFE_FAST_PHRASES


# Whisper fills silence and noise with phrases from its training data, or with the vocabulary prompt itself.
_HALLUCINATIONS = frozenset(
    {"thank you", "thanks for watching", "thank you for watching", "you", "bye", "thank you so much"}
)
NO_SPEECH_LIMIT = 0.5  # real speech in the rehearsal scored 0.03 to 0.14; noise read as words scored 0.63 to 0.77


def looks_hallucinated(text: str) -> bool:
    """Text Whisper produces from noise: a stock sign-off, or one word over and over ("Google, Google, Google")."""
    words = re.findall(r"[a-z']+", text.casefold())
    if not words or " ".join(words) in _HALLUCINATIONS:
        return bool(words)
    if any(leak in " ".join(words) for leak in ("command vocabulary", "proper names")):
        return True  # the vocabulary prompt read back
    # The same word or short phrase four times in a row ("Google, Google…", "Good morning. Good morning…").
    for size in (1, 2, 3):
        for start in range(len(words) - 4 * size + 1):
            unit = words[start : start + size]
            if all(words[start + k * size : start + (k + 1) * size] == unit for k in range(1, 4)):
                return True
    return False


REPLY_TO_QUESTION_WORDS = 12  # an answer to "Which file?" is a name or a phrase, not a story


def should_act_on_follow_up(text: str, *, question_open: bool = False) -> bool:
    """Whether a sentence said without the wake word, in an open conversation or over Heliox's voice, is for
    Heliox to act on.

    It is when it stops or denies ("Wait, stop.", "No"), corrects or undoes ("Skip the backup.", "Undo that."), is
    itself a clear request ("Open my mail"), or answers the question Heliox just asked ("Which file?" "The blue
    one."). Anything else ("PowerShell" on its own, "The invoice is due on the thirty-first of November.") is
    overheard: Live may check it, nothing runs, and it never enters the conversation the planner reads.
    """
    if looks_hallucinated(text):  # "Thanks for watching", "Open, open, open, open": noise read as words
        return False
    if looks_like_conversation_turn(text):
        return True
    words = re.findall(r"[A-Za-z0-9']+", text)
    return question_open and 0 < len(words) <= REPLY_TO_QUESTION_WORDS


SHORT_CUT_OFF_SECONDS = 1.6  # "Wait, stop." is about a second of voice; a sentence said over Heliox is longer


def is_cut_off_stop(voiced_seconds: float | None) -> bool:
    """Speech that began while Heliox was talking and holds at most SHORT_CUT_OFF_SECONDS of voice is "stop".

    Whatever the recognisers make of it. Said over Heliox's voice, "Wait, stop." came out as "PowerShell",
    "Brutes gold", "Bruins go home", "Very discovered.", "Good score.", "Visual Studio Code." and "Where we store
    home" in four rehearsals: at this length the words cannot be trusted, and nothing a person means as a request
    fits in a second and a half said over someone else. Heliox has already stopped talking; it acknowledges, and the
    words go nowhere (with an approval waiting, "Stop" denies it, as a spoken "No" would). A longer utterance said
    over Heliox is heard as what it is.
    """
    return voiced_seconds is not None and voiced_seconds <= SHORT_CUT_OFF_SECONDS


def early_wake_agrees(opening_text: str, final_text: str) -> bool:
    """The wake phrase read from the opening counts only when the first word Whisper heard after it is among
    Parakeet's first five words: both recognisers then heard the same sentence start, and the opening is not a
    guess from a fragment. "Hey Heliox, what can" / "Hey Lokes, what can I say?" agrees; "Hey Heliox." alone, or
    a first word Parakeet did not hear, does not, and the whole utterance is checked as before."""
    prefix = loose_wake_prefix(opening_text)
    if not prefix:
        return False
    after = re.findall(r"[a-z']+", opening_text[len(prefix) :].casefold())
    if not after:
        return False

    def stem(word: str) -> str:
        return re.sub(r"[^a-z']", "", word.casefold()).split("'")[0]  # "what's" and "what" are the same start

    return stem(after[0]) in [stem(word) for word in final_text.split()[:5]]


def _cancel(task: asyncio.Future[Any] | None) -> None:
    if task is not None and not task.done():
        task.cancel()


def is_short_decision(text: str) -> bool:
    """ "No", "Yes", "Stop": complete answers, however short."""
    return " ".join(text.casefold().strip(" \t,.:;!?-").split()) in _AUTONOMOUS_SHORT_DECISIONS


# How "Hey Heliox" comes out of a recogniser that has never heard the name: "Hay heliog", "Hey helioch's",
# "Heliocs". Matched on sound at the very start of an utterance: a greeting-like word, then a word that starts
# like "heli" and is short. "Hello", "Hillary", "Healy" and "helicopter" do not match.
# "Here Heliox", "Dear Heliox" and "Hello Heliox" are how Whisper wrote the owner's "Hey Heliox" through a headset.
_WAKE_LEAD = frozenset(
    {"hey", "hay", "hi", "hei", "heh", "eh", "ay", "a", "oh", "okay", "ok", "here", "hear", "dear", "hello"}
)
_WAKE_NAME = re.compile(r"^h[ei]l{1,2}[iy][a-z]{0,5}$")
_WAKE_NAME_ALONE = re.compile(r"^hel{1,2}io[a-z]{0,3}$")
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def loose_wake_prefix(transcript: str) -> str:
    """The leading words that sound like the wake phrase, as they appear in ``transcript``; "" when none."""
    words = list(_WORD.finditer(transcript[:60]))
    if not words:
        return ""

    def name_like(match: re.Match[str], pattern: re.Pattern[str]) -> bool:
        token = match.group(0).casefold()
        token = token[:-2] if token.endswith("'s") else token
        return bool(pattern.match(token))

    first = words[0]
    if first.group(0).casefold() in _WAKE_LEAD and len(words) > 1 and name_like(words[1], _WAKE_NAME):
        return transcript[: words[1].end()]
    if name_like(first, _WAKE_NAME_ALONE):
        return transcript[: first.end()]
    return ""


_ECHO_FREE_OUTPUTS = ("headphone", "headset", "earphone", "earbud", "airpods", "buds", "hands-free", "handsfree")


def echo_free_output(device_name: str) -> bool:
    """Whether Heliox's voice goes to the person's ears only (a headset), so the microphone cannot hear it."""
    name = device_name.casefold()
    return any(word in name for word in _ECHO_FREE_OUTPUTS)


def default_output_name() -> str:
    """The name of the device Heliox's voice plays on, or "" when it cannot be read."""
    try:
        import soundcard as sc

        speaker = sc.default_speaker()
        return str(speaker.name) if speaker is not None else ""
    except Exception:
        return ""


class BargeInGate:
    """Decides, from the listener's voice-activity state, that the person has started talking over Heliox.

    Pure and clock-driven so it can be tested. With a headset Heliox cannot hear itself, so voice activity that
    lasts ``hold_seconds`` is the person (a cough or a click is shorter). With open speakers the microphone also
    hears Heliox, so it needs voice activity that lasts longer and is clearly louder than what the microphone
    picked up while Heliox was talking before it (``echo_factor`` times that level).
    """

    def __init__(
        self,
        *,
        echo_free: bool,
        hold_seconds: float = 0.25,
        open_hold_seconds: float = 0.4,
        echo_factor: float = 2.5,
        baseline_seconds: float = 0.5,
    ) -> None:
        self.echo_free = echo_free
        self.hold = hold_seconds if echo_free else open_hold_seconds
        self.echo_factor = echo_factor
        self.baseline_seconds = baseline_seconds
        self._started: float | None = None
        self._first: float | None = None
        self._samples: list[tuple[float, float]] = []  # (when, microphone level) while Heliox talks

    def push(self, now: float, speaking: bool, level: float) -> bool:
        if self._first is None:
            self._first = now
        self._samples.append((now, level))
        del self._samples[:-400]
        if not speaking:
            self._started = None
            return False
        if self._started is None:
            self._started = now
        if now - self._started < self.hold:
            return False
        if self.echo_free:
            return True
        before = [lvl for when, lvl in self._samples if when < self._started]
        if not before or self._started - self._first < self.baseline_seconds:
            return False  # no idea yet how loud Heliox itself is in the microphone
        during = sorted(lvl for when, lvl in self._samples if when >= self._started)
        echo = sorted(before)[len(before) // 2]
        return during[len(during) // 2] >= self.echo_factor * max(echo, 1e-4)


class BargeInWatcher:
    """Stops Heliox when the person talks over it, from the listener's own voice-activity state.

    It reads flags the listener keeps up to date and never takes audio frames, so it cannot race the listener for
    the utterance (which is then heard and transcribed as usual). Shaped like a recorder for speak_interruptible.
    """

    POLL_SECONDS = 0.02

    def __init__(self, listener: Any, *, echo_free: bool, clock: Any = time.monotonic) -> None:
        self.listener = listener
        self.gate = BargeInGate(echo_free=echo_free)
        self.clock = clock
        self.detected_at: float | None = None

    @property
    def is_active(self) -> bool:
        return bool(self.listener.is_running and self.listener._recorder.is_active)

    async def wait_for_speech_start(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else self.clock() + timeout
        recorder = self.listener._recorder
        while deadline is None or self.clock() < deadline:
            if self.gate.push(self.clock(), self.listener.user_is_speaking, float(recorder.last_frame_rms)):
                self.detected_at = self.clock()
                self.listener.note_interruption()
                logger.info("The person started talking over Heliox; stopping speech")
                return True
            await asyncio.sleep(self.POLL_SECONDS)
        return False


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


# The local voice (Kokoro/Pocket) runs in a worker process whose lifecycle lives in local_tts_worker: it stays
# loaded while a conversation is plausible, starts when the person is heard, and speaks sentence by sentence.
FADE_SECONDS = 0.008  # a cut-off voice ramps down this fast: no click, and nothing left to drain


OUTPUT_LINE_IDLE_SECONDS = 300.0  # as long as the local voice stays loaded (tts_keep_loaded_seconds' default)


class _OutputLine:
    """One output stream kept open between Heliox's sentences, playing silence when nothing is being said.

    Opening a stream on the owner's machine took 1.0 to 1.1 s before the first sample reached the device (3.2 s for
    the first open after a while; MME, WASAPI and DirectSound alike, Realtek speakers). Opened per sentence, that
    put a 3 s pause before "Okay, stopped." after the person cut Heliox off, although the words were ready. The line
    opens when listening starts or Heliox is about to speak, stays open while a conversation is plausible, and
    closes after OUTPUT_LINE_IDLE_SECONDS of silence.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()  # opening and closing
        self._current_lock = threading.Lock()
        self.stream: Any = None
        self.rate = 0
        self.channels = 0
        self.current: _Playback | None = None
        self.last_used = time.monotonic()
        self._idle_timer: threading.Timer | None = None
        # The system default output when the line opened ("" when it cannot be read), and whether the next open
        # must refresh PortAudio's device list first (it is read once, so a new default is otherwise not seen).
        self.device_name = ""
        self._stale_devices = False
        self.notice = ""

    def callback(self, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        with self._current_lock:
            current = self.current
        if current is None:
            outdata.fill(0)
            return
        current.callback(outdata, frames, time_info, status)
        if current.fade_left == 0:
            # The cut-off has faded to nothing: silence from the next block.
            with self._current_lock:
                if self.current is current:
                    self.current = None

    def ensure(self, rate: int, channels: int) -> Any:
        """The open stream for this format, opening (or reopening for another format) when needed."""
        with self._lock:
            if self.stream is not None and self.rate == rate and self.channels == channels:
                return self.stream
            self._close_locked()
            import sounddevice as sd

            if self._stale_devices:
                self._stale_devices = False
                _refresh_portaudio_devices(sd)
            from pilot.system.audio_health import default_speaker_name

            self.device_name = default_speaker_name()
            stream = sd.OutputStream(
                samplerate=rate, channels=channels, dtype="float32", callback=self.callback, latency="low"
            )
            stream.start()
            self.stream, self.rate, self.channels = stream, rate, channels
            self._touch_locked()
            return stream

    def attach(self, playback: _Playback) -> None:
        with self._current_lock:
            self.current = playback

    def detach(self, playback: _Playback) -> None:
        with self._current_lock:
            if self.current is playback:
                self.current = None
        with self._lock:
            self._touch_locked()

    def _touch_locked(self) -> None:
        self.last_used = time.monotonic()
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        self._idle_timer = threading.Timer(OUTPUT_LINE_IDLE_SECONDS, self._close_if_idle)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _close_if_idle(self) -> None:
        with self._lock:
            if self.current is None and time.monotonic() - self.last_used >= OUTPUT_LINE_IDLE_SECONDS - 0.5:
                self._close_locked()

    def _close_locked(self) -> None:
        stream, self.stream, self.rate, self.channels = self.stream, None, 0, 0
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.abort()
            with contextlib.suppress(Exception):
                stream.close()

    def close(self) -> None:
        with self._lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
            self._close_locked()

    def device_lost(self, reason: str) -> None:
        """The stream stopped playing (device removed): close it; the next sentence reopens on the current default."""
        logger.warning("Speaker output lost (%s); reopening on the current default device", reason)
        self.notice = f"The speaker output stopped ({reason}); switched to the current default device."
        with self._current_lock:
            self.current = None
        with self._lock:
            self._stale_devices = True
            self._close_locked()

    def check_device(self, default_name: str) -> bool:
        """Follow the system default output (headphones unplugged, another device chosen) while idle. Returns True
        when the line was closed to be reopened on the new device."""
        with self._lock:
            if self.stream is None or self.current is not None:
                return False
            moved = bool(default_name and self.device_name and default_name != self.device_name)
            stopped = getattr(self.stream, "active", True) is False
            if not (moved or stopped):
                return False
            self.notice = f"Speaking on {default_name} now." if moved else "The speaker output stopped; reopening it."
            logger.info("Output line: %s", self.notice)
            self._stale_devices = True
            self._close_locked()
        rate = self.rate or LOCAL_VOICE_RATE
        warm_output_line(rate)
        return True


def _refresh_audio_devices() -> None:
    """Re-read PortAudio's device list between streams; the output line is closed first when it is idle."""
    if _output_line.current is not None:
        return
    try:
        import sounddevice as sd
    except Exception:
        return
    _output_line.close()
    _refresh_portaudio_devices(sd)


def _refresh_portaudio_devices(sd: Any) -> None:
    """Re-read the device list, so a new default output is used. Only when no PortAudio input is open: the
    microphone on Windows is captured through WASAPI (soundcard), not PortAudio."""
    if _portaudio_inputs_open:
        return  # re-initialising would close the microphone stream
    try:
        sd._terminate()
        sd._initialize()
    except Exception:
        logger.debug("Could not refresh the audio device list", exc_info=True)


_portaudio_inputs_open = 0  # microphone streams opened through PortAudio (the non-Windows path)
_output_line = _OutputLine()
LOCAL_VOICE_RATE = 24000  # Kokoro's output; another rate reopens the line


def warm_output_line(rate: int = LOCAL_VOICE_RATE) -> None:
    """Open the output line in the background, so Heliox's next words do not wait for the device. Never blocks."""

    def open_line() -> None:
        try:
            _output_line.ensure(rate, 1)
        except Exception:
            logger.debug("Could not open the output line ahead of speech", exc_info=True)

    threading.Thread(target=open_line, name="heliox-output-line", daemon=True).start()


class _Playback:
    """Plays one buffer on the shared output line, so a cut-off can fade out at once instead of draining.

    The callback ramps the voice down over FADE_SECONDS; after that the line plays silence, and only what the
    device had already buffered (its "low" latency, about 90 ms here) is still heard.
    """

    def __init__(self, audio: Any, sample_rate: int) -> None:
        import numpy as np

        self.audio = np.asarray(audio, dtype=np.float32)
        if self.audio.ndim == 1:
            self.audio = self.audio[:, None]
        self.sample_rate = sample_rate
        self.position = 0
        self.fade_left: int | None = None
        self.finished = threading.Event()
        self.faded = threading.Event()
        self.stream: Any = None

    def callback(self, outdata: Any, frames: int, _time: Any, _status: Any) -> None:
        chunk = self.audio[self.position : self.position + frames]
        self.position += len(chunk)
        outdata[: len(chunk)] = chunk
        outdata[len(chunk) :] = 0
        if self.fade_left is not None:
            outdata[:] = fade_block(outdata, self.fade_left, int(self.sample_rate * FADE_SECONDS))
            self.fade_left = max(0, self.fade_left - frames)
            if self.fade_left == 0:
                self.faded.set()
        if self.position >= len(self.audio) and not self.finished.is_set():
            self.finished.set()

    def play(self) -> None:
        self.stream = _output_line.ensure(self.sample_rate, self.audio.shape[1])
        _output_line.attach(self)
        # The device can disappear mid-sentence (headphones unplugged): its callbacks stop, and without a limit
        # Heliox would wait for the end of a sentence that never plays.
        if not self.finished.wait(len(self.audio) / max(1, self.sample_rate) + 2.0):
            _output_line.device_lost("no audio played for longer than the sentence")
            return
        if self.fade_left is not None:
            return  # cut off: the line's callback finishes the fade and then plays silence
        # Let the last block reach the speaker before the next sentence takes the line.
        time.sleep(max(0.0, float(getattr(self.stream, "latency", 0.0) or 0.0)))
        _output_line.detach(self)

    def cut(self) -> None:
        """Fade out over the next block; the line plays silence after it. Returns at once: nothing here waits on
        the audio device, so a slow machine cannot turn the fade into a click."""
        self.fade_left = int(self.sample_rate * FADE_SECONDS)
        self.finished.set()


def fade_block(block: Any, fade_left: int, fade_total: int) -> Any:
    """``block`` with the remaining ``fade_left`` samples of a linear fade-out applied, silence after it."""
    import numpy as np

    out = np.array(block, dtype=np.float32, copy=True)
    frames = len(out)
    if fade_total <= 0 or fade_left <= 0:
        out[:] = 0
        return out
    start = fade_left / fade_total
    ramp = np.clip(start - np.arange(frames, dtype=np.float32) / fade_total, 0.0, 1.0)
    return out * ramp[:, None] if out.ndim == 2 else out * ramp


def _read_audio_file(path: Path) -> tuple[Any, int]:
    """A WAV as float32 samples and its rate (soundfile, an optional audio package, is imported only here)."""
    import soundfile

    audio, sample_rate = soundfile.read(path, dtype="float32")
    return audio, int(sample_rate)


async def _play_local_tts_file(path: Path) -> None:
    """Play a synthesized WAV without importing the model runtime. Cancelling it (barge-in) cuts it at once."""
    audio, sample_rate = await asyncio.to_thread(_read_audio_file, path)
    playback = _Playback(audio, sample_rate)
    try:
        await asyncio.to_thread(playback.play)
    except asyncio.CancelledError:
        await asyncio.to_thread(playback.cut)
        raise


async def _shutdown_local_tts_worker() -> None:
    from pilot.system.local_tts_worker import manager

    await manager().shutdown()


async def _run_local_tts_worker(
    engine: str,
    text: str,
    voice: str,
    output_file: str | None,
) -> None:
    """Speak (or write) with a heavy local voice, outside the long-lived daemon process."""
    from pilot.system.local_tts_worker import manager

    await manager().speak(engine, text, voice, output_file, play=_play_local_tts_file)


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
            if not getattr(e, "already_logged", False):
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
            if not getattr(e, "already_logged", False):
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
    try:
        done, _pending = await asyncio.wait({speak_task, watch_task}, return_when=asyncio.FIRST_COMPLETED)
        return watch_task in done
    finally:
        # Whatever ended this (the line finished, the person talked over it, or a more urgent line replaced
        # it), neither the voice nor the watcher may outlive it: a stray watcher would treat the person's next
        # sentence as an interruption.
        for task in (speak_task, watch_task):
            if not task.done():
                task.cancel()
        for task in (speak_task, watch_task):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


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
    from pilot.system.linux_voice import espeak_program, no_voice_message

    program = espeak_program()
    if program is None:
        # Neither espeak-ng nor espeak is installed: say so once in the log and carry on in text.
        return no_voice_message()
    cmd = [program]

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

    if proc.returncode:
        return f"TTS failed: {program} exited with code {proc.returncode}"

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


# Set when CUDA is visible but its libraries are not (a GPU without the CUDA 12 runtime): CPU from then on.
_faster_whisper_cuda_unusable = False


# Why Whisper runs where it does, for the voice stats and the model health panel.
_faster_whisper_device_reason = ""
# Graphics memory Whisper base/small need on the card, with room for a short utterance's activations.
WHISPER_GPU_NEEDS_MB = {"tiny": 300, "base": 400, "small": 900, "medium": 2200}


def _faster_whisper_runtime(model_name: str = "base") -> tuple[str, str]:
    """Where Faster-Whisper runs: the CPU unless an NVIDIA card is present, healthy and free.

    Whisper runs inside the daemon, so a CUDA fault there would end the daemon (the reason Gnomon has its own
    worker). The GPU arbiter decides: a card that is hot, held by another Heliox process (Gnomon), short of memory,
    unreadable or switched off for the session after faults means the CPU. A CUDA error at run time counts as a fault
    and keeps Whisper on the CPU for the rest of the session. The wake check and the short final passes cost about
    a second on the CPU (measured on the owner's sentences), so the CPU is never a bad place to be.
    """
    global _faster_whisper_device_reason
    if _faster_whisper_cuda_unusable:
        _faster_whisper_device_reason = "The graphics card failed for Whisper earlier this session."
        return "cpu", "int8"
    try:
        import ctranslate2

        cards = ctranslate2.get_cuda_device_count()
    except (ImportError, RuntimeError, OSError):
        cards = 0
    if cards <= 0:
        _faster_whisper_device_reason = "No CUDA graphics card."
        return "cpu", "int8"
    from pilot.system.gpu_arbiter import MIB, Needs, arbiter

    needs = Needs(gpu_bytes=WHISPER_GPU_NEEDS_MB.get(model_name.split(".")[0], 1500) * MIB, default_device_only=True)
    decision = arbiter().check("whisper", needs)
    if not decision.use_gpu or decision.device != "cuda":
        _faster_whisper_device_reason = decision.reason or "The graphics card is not available."
        return "cpu", "int8"
    _faster_whisper_device_reason = "The graphics card is healthy and free."
    return "cuda", "float16"


def _whisper_gpu_failed(error: BaseException) -> None:
    """A CUDA error in Whisper: the CPU for the rest of the session, and a fault for the arbiter."""
    global _faster_whisper_cuda_unusable
    _faster_whisper_cuda_unusable = True
    try:
        from pilot.system.gpu_arbiter import arbiter

        arbiter().record_fault("whisper", f"Faster-Whisper: {str(error)[:200]}")
    except Exception:
        logger.debug("Could not record the GPU fault", exc_info=True)
    for key in [key for key in _faster_whisper_model_cache if key[1] == "cuda"]:
        _faster_whisper_model_cache.pop(key, None)  # free the card
    logger.warning("Faster-Whisper cannot use the GPU (%s); using the CPU", str(error)[:120])


def _get_faster_whisper_model(model_name: str) -> Any:
    from faster_whisper import WhisperModel

    device, compute_type = _faster_whisper_runtime(model_name)
    cache_key = (model_name, device, compute_type)
    if cache_key not in _faster_whisper_model_cache:
        options: dict[str, Any] = {}
        if device == "cpu":
            # Measured on the rehearsal recording: 8 threads take base from 2.2 s to 1.1 s for a 6 s sentence.
            options["cpu_threads"] = max(1, min(8, os.cpu_count() or 4))
        from pilot.system.speech_models import local_path

        # Whisper base downloaded into Heliox's data folder (the Voice choice) is used from there.
        local = local_path("whisper-base") if model_name == "base" else None
        try:
            _faster_whisper_model_cache[cache_key] = WhisperModel(
                str(local) if local is not None else model_name,
                device=device,
                compute_type=compute_type,
                **options,
            )
        except (RuntimeError, ValueError) as error:
            if device != "cuda":
                raise
            _whisper_gpu_failed(error)
            return _get_faster_whisper_model(model_name)
    return _faster_whisper_model_cache[cache_key]


def _speech_text(segments: Any) -> str:
    """The text of the segments Whisper heard as speech; noise read as words is dropped."""
    kept = [
        segment.text.strip()
        for segment in segments
        if segment.text.strip() and float(getattr(segment, "no_speech_prob", 0.0) or 0.0) <= NO_SPEECH_LIMIT
    ]
    text = " ".join(kept)
    return "" if looks_hallucinated(text) else text


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
            # Measured on a real voice through a Bluetooth headset: without timestamps and without the prompt
            # repeated as hotwords, "Monday the 12th of November" was heard right and the pass took 0.7 to 0.8 s
            # instead of 1.0 s; with hotwords it heard "mildly the 12th".
            "without_timestamps": True,
            # Endpointing already isolated the utterance. A second VAD can
            # remove short commands such as "stop" or "close".
            "vad_filter": False,
        }
        if language != "auto":
            kwargs["language"] = language
        # Decode the WAV here and hand faster-whisper the samples. Given a path it decodes through PyAV, whose
        # 19.x release dropped an argument faster-whisper 1.2 still passes, so every local transcription failed
        # and fell back to the much slower OpenAI Whisper.
        audio = _load_pcm_wav_for_whisper(audio_path)
        if audio is not None:
            audio = _normalize_voice_audio(_band_limit_voice_audio(audio))
        try:
            segments, info = model.transcribe(audio if audio is not None else audio_path, **kwargs)
            text = _speech_text(segments)
        except RuntimeError as error:
            # A visible GPU without cuBLAS/cuDNN fails only when the first segment encodes: switch to the CPU
            # build for the rest of the session and transcribe this utterance again.
            if not any(name in str(error).lower() for name in ("cublas", "cudnn", "cuda")):
                raise
            _whisper_gpu_failed(error)
            model = _get_faster_whisper_model(model_name)
            segments, info = model.transcribe(audio if audio is not None else audio_path, **kwargs)
            text = _speech_text(segments)
        detected_language = getattr(info, "language", None)
        return {
            "text": text.strip(),
            "language": detected_language or (language if language != "auto" else "en"),
        }

    return await asyncio.to_thread(_counted, _do)


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
        # {headset name: input id} chosen by the person for a headset in call mode (VoiceConfig).
        self.headset_input_choices: dict[str, str] = {}

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
        # time.monotonic() when a frame above the speech threshold was last captured (the capture thread's clock,
        # not when the frame was processed): the end of the person's last word, for latency measured from there.
        self.last_voice_monotonic = 0.0
        self.last_turn_ended_at: float | None = None  # last_voice_monotonic when the last utterance ended
        self.utterances_captured = 0
        # True between the endpointer confirming speech and the utterance ending (read by the live layer).
        self.user_is_speaking = False
        # When the current or last utterance's speech was confirmed (time.monotonic()); None before any.
        self.speech_started_at: float | None = None
        self.capture_backend = ""
        # When the current run of exact digital zeros began (time.monotonic()); None while there is any signal.
        self._zeros_since: float | None = None
        # Semantic turn-taking (pilot.system.turn_detector): asked after a short silence whether the person is
        # done. Done ends the turn at once; not done keeps listening for up to turn_hold_frames of silence.
        self.turn_detector: Any = None
        # 400 ms: on a Bluetooth hands-free microphone the owner paused 250 to 350 ms between words (the link sends
        # digital silence), and Smart Turn, asked after 200 ms, ended the turn at every one of those pauses.
        self.turn_check_frames = 8
        self.turn_recheck_frames = 12  # ask again every 600 ms of continued silence
        # A candidate end of turn after 200 ms of quiet: the listener starts the final pass on the audio so far, so
        # the words are ready when the turn does end there; voice resuming cancels it (on_candidate_cancel).
        self.speculate_frames = 4
        self.on_candidate_end: Any = None  # callable(frames) -> None
        self.on_candidate_cancel: Any = None  # callable() -> None
        # The opening of the utterance (pre-roll included), handed over once while the person is still talking,
        # so the wake phrase can be read before they finish (see EARLY_WAKE_SECONDS).
        self.opening_frames = max(1, int(EARLY_WAKE_SECONDS * 1000 / frame_ms))
        self.on_opening: Any = None  # callable(frames) -> None
        # Smart Turn can also be asked at the candidate end; at or above this it ends the turn 200 ms sooner.
        # Off (1.0) until its effect on split sentences is measured on recorded speech: pauses between words on a
        # noise-gated headset last 250 to 350 ms, and asking after 200 ms once ended the turn at every such pause.
        # 0.95 is the value to measure: the owner's mid-sentence pauses scored 0.71 to 0.83, real ends 0.94 to 0.98.
        self.early_turn_threshold = 1.0
        self.turn_hold_frames = 50  # 2.5 s
        # 0.85, not 0.5: on the owner's headset Smart Turn gave 0.71 to 0.83 at mid-sentence pauses ("...the launch call
        # is Monday | the twelfth of November") and 0.94 to 0.98 at the real ends of their sentences.
        self.turn_threshold = 0.85
        self.on_hold: Any = None  # called once per utterance when the person pauses mid-thought
        # Callable() -> bool: the words so far are a complete short answer, so a short pause ends the turn.
        self.complete_hint: Any = None
        self.short_answer_frames = 8  # 400 ms
        # Called when speech starts: the listener picks up a recogniser that finished loading while it waited.
        self.on_speech_start: Any = None
        # Streaming recognition (pilot.system.streaming_asr) sees each frame as it is captured.
        self.stream_sink: Any = None
        self.last_turn_probability: float | None = None
        self.last_endpoint_reason = ""
        self.last_endpoint_ms = 0.0
        # Seconds of voice in the last utterance: from the confirmed start to the end, without the pre-roll or the
        # trailing silence that ended it. None until an utterance has been captured.
        self.last_voiced_seconds: float | None = None

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
        return self._open_capture()

    def effective_preferred_device(self) -> str:
        """The configured microphone, or, with "auto" and the system default being a headset the person chose
        another microphone for, that microphone. The headset can stay the output; only the input moves."""
        configured = (self.preferred_device or "auto").strip() or "auto"
        if configured.casefold() != "auto" or not self.headset_input_choices:
            return configured
        return self.headset_input_choices.get(self.device_key(), configured)

    def device_key(self) -> str:
        """The stable key of the microphone in use with "auto": the system default's (pulse source name on Linux,
        device name elsewhere), else the name of the device the listener opened."""
        from pilot.system.audio_health import default_microphone_key

        return default_microphone_key() or self.input_device_name

    def _open_capture(self) -> bool:
        """Open the capture backend; frames go to the existing queue. Safe to call from a worker thread."""
        self._zeros_since = None
        self._opening_preferred = self.effective_preferred_device()
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

        preferred = self._opening_preferred
        if preferred.startswith("pulse::"):
            # A PulseAudio/PipeWire source chosen by its stable name: PortAudio cannot name Pulse sources, but its
            # ALSA "pulse" device opens whatever PULSE_SOURCE names when the stream starts.
            os.environ["PULSE_SOURCE"] = preferred.split("::", 1)[1]
            preferred = "auto"
        else:
            os.environ.pop("PULSE_SOURCE", None)
        try:
            self.using_fallback_input = False
            self.input_device_notice = ""
            self.input_device = _resolve_input_device(
                sd,
                sample_rate=self.sample_rate,
                channels=1,
                dtype="int16",
                preferred_device=preferred,
                allow_preferred_fallback=True,
            )
            device_info = sd.query_devices(self.input_device)
            self.input_device_name = str(device_info.get("name", self.input_device))
            configured = preferred.strip()
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
            global _portaudio_inputs_open
            _portaudio_inputs_open += 1
            self.capture_backend = "portaudio"
            self.last_error = ""
            return True
        except Exception as exc:
            self.last_error = str(exc)
            logger.warning("Failed to open continuous audio input stream", exc_info=True)
            self._stream = None
            return False

    @property
    def zero_seconds(self) -> float:
        """How long every frame has been exact digital zeros (a muted or captured device); 0 while there is signal."""
        since = self._zeros_since
        return 0.0 if since is None else max(0.0, time.monotonic() - since)

    def _publish_audio_block(self, block: Any) -> None:
        signal_rms = frame_rms(block)
        if signal_rms == 0.0:
            if self._zeros_since is None:
                self._zeros_since = time.monotonic()
        else:
            self._zeros_since = None
        self.frames_received += 1
        self.last_frame_rms = signal_rms
        self.peak_frame_rms = max(self.peak_frame_rms, signal_rms)
        if self.adaptive_threshold_enabled and not self.user_is_speaking:
            # Learn the room from the gaps, not from the person: a quiet headset voice that fills most of the
            # window otherwise lifts the "noise floor" above its own softer words, and the sentence is cut into
            # pieces (measured on a Bluetooth hands-free microphone).
            self._adaptive_threshold.observe(signal_rms)
        if signal_rms >= self.effective_energy_threshold:
            self.last_above_threshold_at = time.time()
            self.last_voice_monotonic = time.monotonic()
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
        preferred = getattr(self, "_opening_preferred", self.preferred_device)
        if preferred.strip().casefold() != "auto":
            preferred_name = preferred.rsplit("::", 1)[-1].strip()

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
                f"Configured microphone '{preferred.strip()}' is unavailable; "
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

    def reopen(self) -> bool:
        """Close the capture backend and open it again on the current device, keeping the frame queue, so an
        utterance being waited for carries on with the new stream. Blocking: call from a worker thread."""
        portaudio = self.capture_backend == "portaudio"
        self._close_capture()
        self.last_error = ""
        if portaudio:
            # PortAudio (macOS, Linux, and Windows when WASAPI capture is unavailable) reads the device list once:
            # without a refresh a new default microphone, or one plugged back in, is not seen.
            _refresh_audio_devices()
        return self._open_capture()

    def stop(self) -> None:
        self._close_capture()
        if self._queue is not None:
            self._queue.put_nowait(_RECORDER_STOP)
        self._queue = None

    def _close_capture(self) -> None:
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
            global _portaudio_inputs_open
            _portaudio_inputs_open = max(0, _portaudio_inputs_open - 1)

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

    async def record_utterance(self, timeout: float | None = None, keep_backlog: bool = False) -> str | None:
        """Waits for a full utterance (speech start through endpoint) and
        returns the path to a WAV file containing it, or None if nothing
        was captured before `timeout`. ``keep_backlog`` keeps frames that
        arrived before the call instead of starting from now."""
        queue = self._queue
        if queue is None:
            return None

        self.speech_started_at = None
        self.last_voiced_seconds = None
        if not keep_backlog:
            await self._drain_stale_frames(queue)
        endpointer = self._make_endpointer()
        deadline = None if timeout is None else asyncio.get_event_loop().time() + timeout
        # Small pre-roll so the moment speech is CONFIRMED (after
        # start_frames) doesn't clip the first syllable that triggered it.
        preroll: list[Any] = []
        captured: list[Any] = []
        lead = 0
        held = False
        candidate = False
        opening_sent = False
        silence_frames_used = endpointer.silence_frames

        while True:
            remaining = None if deadline is None else deadline - asyncio.get_event_loop().time()
            if remaining is not None and remaining <= 0:
                return None
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                self.user_is_speaking = False
                return None

            if frame is _RECORDER_STOP:
                self.user_is_speaking = False
                return None

            if not endpointer.is_speaking:
                # Follow the room while waiting. A wait that began before the threshold had learned the room kept
                # the 0.02 default, and a quiet headset voice (0.01) waited unheard behind it.
                endpointer.energy_threshold = self.effective_energy_threshold
            event = endpointer.push(frame_rms(frame))

            # Check event states before is_speaking: push() already called
            # reset() internally on ENDED/MAX_DURATION, so is_speaking is
            # back to False by the time we get here — checking is_speaking
            # first would misroute the utterance's final frame into the
            # preroll buffer instead of appending it and returning.
            if event == EndpointEvent.STARTED:
                captured = preroll + [frame]
                lead = len(preroll)
                preroll = []
                candidate = False
                opening_sent = False
                self.user_is_speaking = True
                self.speech_started_at = time.monotonic()
                if self.on_speech_start is not None and self.stream_sink is None:
                    try:
                        self.on_speech_start()
                    except Exception:
                        logger.debug("Speech-start setup failed", exc_info=True)
                held = False
                silence_frames_used = endpointer.silence_frames
                if self.stream_sink is not None:
                    self.stream_sink.begin(captured)
                continue

            if event in (EndpointEvent.ENDED, EndpointEvent.MAX_DURATION):
                captured.append(frame)
                if self.stream_sink is not None:
                    self.stream_sink.feed(frame)
                self.utterances_captured += 1
                self.user_is_speaking = False
                self.last_endpoint_reason = "silence" if event == EndpointEvent.ENDED else "max_duration"
                self.last_endpoint_ms = silence_frames_used * self.frame_ms
                self._note_voiced(len(captured) - lead)
                return _write_wav(captured, self.sample_rate)

            if not endpointer.is_speaking:
                preroll.append(frame)
                preroll_frames = max(self.start_frames + 1, int(300 / self.frame_ms))
                if len(preroll) > preroll_frames:
                    preroll.pop(0)
                continue

            captured.append(frame)
            if self.stream_sink is not None:
                self.stream_sink.feed(frame)
            if not opening_sent and len(captured) >= self.opening_frames:
                opening_sent = True
                self._notify(self.on_opening, list(captured))

            silence_now = endpointer.trailing_silence
            if candidate and silence_now == 0:
                candidate = False  # they carried on: the early pass was for half a sentence
                self._notify(self.on_candidate_cancel)
            elif not candidate and silence_now >= self.speculate_frames:
                candidate = True
                self._notify(self.on_candidate_end, list(captured))
                if self.turn_detector is not None and self.early_turn_threshold < 1.0:
                    import numpy as np

                    probability = await asyncio.to_thread(self.turn_detector.probability, np.concatenate(captured))
                    if probability is not None and probability >= self.early_turn_threshold:
                        self.last_turn_probability = probability
                        self.last_endpoint_ms = endpointer.trailing_silence * self.frame_ms
                        endpointer.reset()
                        self.utterances_captured += 1
                        self.user_is_speaking = False
                        self.last_endpoint_reason = "smart_turn_early"
                        self._note_voiced(len(captured) - lead)
                        return _write_wav(captured, self.sample_rate)

            if (
                self.complete_hint is not None
                # From 400 ms of quiet, checked each frame for another 400 ms: the recogniser's words lag the audio.
                and self.short_answer_frames <= endpointer.trailing_silence <= 2 * self.short_answer_frames
                and self.complete_hint()
            ):
                # A one-word answer ("No.", "Stop.") sounds unfinished to Smart Turn, which would hold the turn for
                # 2 s. The words say it is complete; 400 ms of quiet shows nothing more is coming ("No problem,
                # go ahead" carries on well before that, and its full text is then judged as a whole).
                self.last_endpoint_ms = endpointer.trailing_silence * self.frame_ms
                endpointer.reset()
                self.utterances_captured += 1
                self.user_is_speaking = False
                self.last_endpoint_reason = "short_answer"
                self._note_voiced(len(captured) - lead)
                return _write_wav(captured, self.sample_rate)

            if self.turn_detector is not None and self._turn_check_due(endpointer.trailing_silence):
                import numpy as np

                silence = endpointer.trailing_silence
                probability = await asyncio.to_thread(self.turn_detector.probability, np.concatenate(captured))
                if probability is not None:
                    self.last_turn_probability = probability
                    if probability >= self.turn_threshold:
                        # The sentence sounds finished: answer now instead of waiting out the silence.
                        endpointer.reset()
                        self.utterances_captured += 1
                        self.user_is_speaking = False
                        self.last_endpoint_reason = "smart_turn"
                        self.last_endpoint_ms = silence * self.frame_ms
                        self._note_voiced(len(captured) - lead)
                        return _write_wav(captured, self.sample_rate)
                    # Mid-thought: give them time instead of cutting them off.
                    endpointer.silence_frames = max(endpointer.silence_frames, self.turn_hold_frames)
                    silence_frames_used = endpointer.silence_frames
                    if not held and self.on_hold is not None:
                        held = True
                        try:
                            self.on_hold()
                        except Exception:
                            logger.debug("Turn hold callback failed", exc_info=True)

    @staticmethod
    def _notify(callback: Any, *args: Any) -> None:
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:
            logger.debug("Candidate end-of-turn callback failed", exc_info=True)

    def _note_voiced(self, frames: int) -> None:
        self.last_turn_ended_at = self.last_voice_monotonic or None
        self.last_voiced_seconds = max(0.0, (frames * self.frame_ms - self.last_endpoint_ms) / 1000.0)

    def _turn_check_due(self, silence: int) -> bool:
        if silence < self.turn_check_frames:
            return False
        return (silence - self.turn_check_frames) % max(1, self.turn_recheck_frames) == 0


# How much of an utterance Whisper reads for the wake decision. The whole of it: on the owner's headset, reading
# only the first 2 s found "Heliox" in 20 of 30 tries against 30 of 30 for the whole sentence (fixed boundaries).
WAKE_OPENING_SECONDS = 30.0
# The wake phrase read from the first seconds of an utterance while the person is still speaking the rest. Used
# only when it agrees with Parakeet on the first word after the phrase; otherwise the whole utterance is checked
# as before, so this can only make the answer sooner, never accept what the whole-utterance check would not see.
EARLY_WAKE_SECONDS = 1.6


def _to_int16(samples: Any) -> Any:
    import numpy as np

    return (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767).astype(np.int16)


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


_recogniser_threads = 0
_recogniser_lock = threading.Lock()


def _recognisers_running() -> int:
    return _recogniser_threads


def _counted(function: Any, *args: Any) -> Any:
    """Run a recogniser in the calling worker thread, counted, so speculation knows when the CPU is taken."""
    global _recogniser_threads
    with _recogniser_lock:
        _recogniser_threads += 1
    try:
        return function(*args)
    finally:
        with _recogniser_lock:
            _recogniser_threads -= 1


class _Speculation:
    """The final pass (Parakeet and the Whisper wake check, or Whisper alone) started at a candidate end of turn.

    200 ms into a pause the recorder hands over the audio so far; the pass runs while Smart Turn and the remaining
    quiet decide whether the turn really ended. If the person carries on, the pass is cancelled and a new one starts
    at the next pause. If the turn ends at that pause, the same words are used: nothing voiced came after them.
    """

    def __init__(self, listener: ContinuousVoiceListener) -> None:
        self.listener = listener
        self.task: asyncio.Task[str] | None = None

    def start(self, frames: list[Any]) -> None:
        self.cancel()
        if _recognisers_running():
            # Recognisers run in threads that cannot be stopped: a pass dropped because the person carried on
            # keeps the CPU busy until it finishes. No second one starts beside it (on 4 cores two at once made
            # both several times slower); the pass then runs at the end of the turn, as without speculation.
            return
        path = _write_wav(frames, self.listener._recorder.sample_rate)
        self.task = asyncio.ensure_future(self.listener._transcribe_path(path, speculative=True))
        self.task.add_done_callback(lambda task: task.cancelled() or task.exception())

    def cancel(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()
        self.task = None

    def take(self) -> asyncio.Task[str] | None:
        task, self.task = self.task, None
        return task


class ContinuousVoiceListener:
    """Always-on microphone listener with wake word detection."""

    def __init__(
        self,
        wake_words: list[str] | None = None,
        on_command: Any | None = None,
        on_status: Any | None = None,
        workflow_control: Any | None = None,
        config: PilotConfig | None = None,
        on_partial: Any | None = None,
        on_turn_hold: Any | None = None,
        on_overheard: Any | None = None,
    ) -> None:
        self.wake_words = wake_words or [
            "hey heliox",
            "heliox",
            "hey pilot",
        ]

        self._on_command = on_command
        self._on_status = on_status
        # Async callables: a live caption of the words so far (streaming engine), and "still listening" when the
        # person pauses mid-thought (Smart Turn says the sentence isn't finished).
        self._on_partial = on_partial
        self._on_turn_hold = on_turn_hold
        # Async callable(text): something said while a conversation is open that is not addressed to Heliox as a
        # command (a promise, a date). The server lets Live check it; it never becomes a task.
        self._on_overheard = on_overheard
        # Callable() -> bool, set by the server: a task the person started is running or an approval is waiting.
        self.task_open: Any = None
        self._interrupted_until = 0.0
        # The microphone button addresses the next sentence whatever it says; talking over Heliox does not.
        self._button_until = 0.0
        # Heliox's last words to the person asked them something ("Which file?"): a short answer is a reply.
        self._question_until = 0.0
        # The call app the person is in a call with ("" when none), kept up to date while listening.
        self.in_call = ""
        self._call_watch_task: asyncio.Task[None] | None = None
        # Keeps the microphone open through device changes and failures (pilot.system.audio_health).
        self._capture_watch_task: asyncio.Task[None] | None = None
        self._capture_watchdog: Any = None
        self.capture_notice = ""
        self._early_wake: asyncio.Task[str] | None = None
        # Off until the false-accept comparison has run on speech not addressed to Heliox (a wake phrase accepted
        # wrongly would send overheard speech to a model): the opening is then never read early and the whole
        # utterance is checked as before.
        self.early_wake_enabled = False
        # What the sound says about the microphone (pilot.system.mic_quality), judged once per device.
        from pilot.system.mic_quality import MicQuality

        self._mic_quality = MicQuality()
        self.last_wake_source = ""  # "opening" when the early reading of the opening decided, else "whole"
        self._turn_detector: Any = None
        self._streaming: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.last_transcription_engine = "whisper"
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
        self._heliox_quiet_since = 0.0
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
        self._recorder.headset_input_choices = dict(getattr(self.config.voice, "headset_input_choices", {}) or {})

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def follow_up_remaining_seconds(self) -> float:
        """Seconds left in the bounded wake-free conversation turn."""
        return max(0.0, self._follow_up_until - time.monotonic())

    def arm_follow_up_window(self, seconds: float | None = None, *, asked: bool = False) -> None:
        """Allow exactly the next utterance to omit the wake word briefly.

        The server calls this only after Heliox has finished speaking, which
        avoids treating TTS echo as user intent. Accepting an utterance consumes
        the window; a completed response arms a fresh turn.
        """
        configured = self.config.voice.follow_up_window_seconds if seconds is None else seconds
        duration = max(0.0, min(float(configured), 120.0))
        self._follow_up_until = time.monotonic() + duration if duration else 0.0
        self._question_until = self._follow_up_until if asked else 0.0

    def clear_follow_up_window(self) -> None:
        self._follow_up_until = 0.0
        self._question_until = 0.0

    @property
    def question_open(self) -> bool:
        """Heliox asked the person something and is still waiting for the answer."""
        return time.monotonic() < self._question_until

    def conversation_open(self) -> bool:
        """The person is talking with Heliox: it spoke to them in the last follow-up window (30 s by default), or a
        task they started is running, or an approval is waiting. Only then do replies without the wake word count;
        the rest of the time other speech in the room is not addressed to Heliox."""
        if self.follow_up_remaining_seconds > 0:
            return True
        try:
            return bool(self.task_open and self.task_open())
        except Exception:
            return False

    def note_interruption(self, seconds: float = 12.0, *, by_button: bool = False) -> None:
        """The person talked over Heliox, or pressed the microphone button. After the button the next sentence
        is for Heliox whatever it says; talking over Heliox stops its voice, and what was said counts only when
        it is a reply (see should_act_on_follow_up)."""
        self._interrupted_until = time.monotonic() + seconds
        self._button_until = self._interrupted_until if by_button else 0.0

    def end_interruption(self) -> None:
        self._interrupted_until = 0.0
        self._button_until = 0.0

    def _final_pass_status(self) -> dict[str, Any]:
        """Which recogniser's words are acted on, and why Parakeet is not in use when it was chosen."""
        chosen = getattr(self.config.voice, "final_pass", "whisper")
        if chosen != "parakeet":
            return {"final_pass": chosen, "final_pass_notice": ""}
        from pilot.config import DATA_DIR
        from pilot.system.parakeet_asr import missing_files, model_dir

        folder = model_dir(getattr(self.config.voice, "parakeet_model_dir", ""), DATA_DIR)
        missing = missing_files(folder)
        if missing:
            return {
                "final_pass": "whisper",
                "final_pass_notice": f"Parakeet model not found in {folder} (missing {', '.join(missing)}); "
                "using Whisper. Download it in Settings, Voice.",
            }
        loaded = getattr(self, "_parakeet_transcriber", None)
        error = getattr(loaded, "error", "") if loaded is not None else ""
        return {"final_pass": "whisper" if error else "parakeet", "final_pass_notice": error}

    def _final_pass_available(self) -> bool:
        """Whether a local Whisper is installed to check the fast recogniser's words."""
        import importlib.util

        return any(importlib.util.find_spec(name) for name in ("faster_whisper", "whisper"))

    def _addressed(self) -> bool:
        """Whether this utterance was addressed by an interruption or the microphone button: it is still inside
        the window, or it started inside it (a sentence begun at 9 s of a 10 s window ends after it)."""
        until = self._interrupted_until
        started = getattr(self._recorder, "speech_started_at", None)
        return time.monotonic() < until or (isinstance(started, float) and until > 0 and started <= until)

    def _cutting_off(self, final: bool = False) -> bool:
        """The utterance began while Heliox was talking (not after the button) and is still, or ended, short enough
        to be a cut-off: its words are not shown as what the person said."""
        if not self._addressed() or self._button_until:
            return False
        if final:
            return self._is_short_cut_off()
        started = getattr(self._recorder, "speech_started_at", None)
        return not isinstance(started, float) or time.monotonic() - started <= SHORT_CUT_OFF_SECONDS

    def _caption(self, text: str) -> None:
        if self._on_partial and not self._cutting_off():
            self._schedule(self._on_partial, text)

    def _is_short_cut_off(self) -> bool:
        voiced = getattr(self._recorder, "last_voiced_seconds", None)
        return isinstance(voiced, float) and voiced <= SHORT_CUT_OFF_SECONDS

    def _take_interruption(self) -> str:
        """ "button", "barge_in" or "" for the utterance just heard; the window closes either way."""
        kind = ("button" if self._button_until else "barge_in") if self._addressed() else ""
        self.end_interruption()
        return kind

    async def wait_for_floor(self, max_seconds: float = 8.0, quiet_seconds: float = 0.35) -> float:
        """Wait while the person is still talking, so Heliox does not start over them. Returns seconds waited.

        Talking means mid-utterance, or voice in the microphone in the last ``quiet_seconds``: the end of a turn
        can come at a short pause ("...the twelfth of November | at ten"), and the tail is still theirs."""
        started = time.monotonic()
        while time.monotonic() - started < max_seconds:
            recent = time.time() - float(getattr(self._recorder, "last_above_threshold_at", 0.0) or 0.0)
            if not self.user_is_speaking and recent >= quiet_seconds:
                break
            await asyncio.sleep(0.05)
        return time.monotonic() - started

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
        if self._wake_free_suppression_depth == 0:
            self._heliox_quiet_since = time.monotonic()

    def _began_after_heliox_spoke(self) -> bool:
        """The utterance's speech started once Heliox had stopped talking, so it is the person, not its echo."""
        started = getattr(self._recorder, "speech_started_at", None)
        return isinstance(started, float) and started >= self._heliox_quiet_since

    @property
    def user_is_speaking(self) -> bool:
        """Whether the person is mid-utterance right now, from the continuous VAD."""
        return bool(self._running and getattr(self._recorder, "user_is_speaking", False))

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
        self._call_watch_task = asyncio.create_task(self._watch_calls())
        self._capture_watch_task = asyncio.create_task(self._watch_capture())

        logger.info(
            "Continuous voice listener started (wake words: %s, vad_recorder=%s)",
            self.wake_words,
            recorder_active,
        )

        if self.config.voice.continuous_conversation_enabled:
            return "Continuous voice listener started. Speak naturally; the wake phrase is optional."
        return f"Voice listener started. Say '{self.wake_words[0]}' to activate."

    def _schedule(self, callback: Any, *args: Any) -> None:
        """Run an async callback on the listener's loop, from any thread."""
        loop = self._loop
        if callback is None or loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(callback(*args)))

    def _configure_turn_taking(self) -> Any:
        """Point the recorder at the turn detector and streaming recognizer the settings ask for.

        Returns the streaming transcriber for this utterance, or None to use Whisper."""
        voice = self.config.voice
        recorder = self._recorder
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        smart = getattr(voice, "turn_detection", "smart_turn") == "smart_turn"
        wants_streaming = voice.transcription_engine == "streaming"
        if (smart and self._turn_detector is None) or (wants_streaming and self._streaming is None):
            # Turned on while listening: load in the background and use the plain path until it is ready.
            asyncio.ensure_future(self._warm_turn_taking())
        recorder.turn_detector = (
            self._turn_detector if smart and self._turn_detector is not None and self._turn_detector.ready else None
        )
        recorder.turn_hold_frames = max(1, int(getattr(voice, "turn_hold_ms", 2500.0) / recorder.frame_ms))
        recorder.on_hold = (lambda: self._schedule(self._on_turn_hold)) if self._on_turn_hold else None
        streaming = self._streaming if voice.transcription_engine == "streaming" else None
        if streaming is not None and streaming.ready:
            streaming.on_partial = self._caption if self._on_partial else None
            recorder.stream_sink = streaming
            recorder.complete_hint = lambda: is_short_decision(streaming.partial())
            return streaming
        recorder.stream_sink = None
        recorder.complete_hint = None
        return None

    async def _warm_turn_taking(self) -> None:
        """Load Smart Turn and, when chosen, the streaming recognizer before the first utterance."""
        from pilot.system.streaming_asr import StreamingTranscriber
        from pilot.system.turn_detector import SmartTurnDetector

        self._loop = asyncio.get_running_loop()
        voice = self.config.voice
        if getattr(voice, "turn_detection", "smart_turn") == "smart_turn":
            if self._turn_detector is None:
                self._turn_detector = SmartTurnDetector()
            if not self._turn_detector.ready:
                await asyncio.to_thread(self._turn_detector.load)
        if voice.transcription_engine == "streaming":
            if self._streaming is None:
                self._streaming = StreamingTranscriber()
            if not self._streaming.ready:
                await asyncio.to_thread(self._streaming.load)

    async def _watch_calls(self, every_seconds: float = 2.0) -> None:
        """Keep ``in_call`` current and say so in the voice bar when it changes."""
        from pilot.system.call_detection import current_call

        while self._running:
            app = ""
            if getattr(self.config.voice, "quiet_in_calls", True):
                try:
                    app = await asyncio.to_thread(current_call)
                except Exception:
                    logger.debug("Call detection failed", exc_info=True)
            if app != self.in_call:
                self.in_call = app
                if app:
                    logger.info("In a call (%s): Heliox stays quiet and does not take speech as addressed to it", app)
                if self._on_status:
                    try:
                        if app:
                            await self._on_status("in_call", {"app": app, "message": "Quiet while you're in a call"})
                        else:
                            await self._on_status("call_ended", {})
                    except Exception:
                        logger.debug("Call status callback failed", exc_info=True)
            await asyncio.sleep(every_seconds)

    async def _watch_capture(self, every_seconds: float = 1.0) -> None:
        """Reopen the microphone when it fails, stalls, goes silent or the default device changes, and say so."""
        from pilot.system.audio_health import (
            CaptureState,
            CaptureWatchdog,
            default_microphone_name,
            default_speaker_name,
            microphone_present,
        )

        recorder = self._recorder
        configured = str(getattr(recorder, "preferred_device", "auto") or "auto").strip()
        auto = configured.casefold() in {"", "auto"}
        configured_name = "" if auto else configured.rsplit("::", 1)[-1].strip()
        # The device probes read the system default live on all three systems (WASAPI, CoreAudio, PulseAudio or
        # PipeWire through soundcard) and answer "" when they cannot tell, which turns the default checks off.
        live_devices = True

        async def default_name() -> str:
            return await asyncio.to_thread(default_microphone_name) if live_devices else ""

        watchdog = self._capture_watchdog = CaptureWatchdog()
        watchdog.opened(await default_name(), recorder.last_error)
        permission_checked_at = -1e9
        await self._check_microphone_permission()  # macOS: a refused permission is said at once, not after 30 s
        while self._running:
            await asyncio.sleep(every_seconds)
            if not self._running:
                return
            fallback = bool(recorder.using_fallback_input)
            state = CaptureState(
                frames=int(recorder.frames_received),
                error=str(recorder.last_error or ""),
                zero_seconds=float(recorder.zero_seconds),
                active=bool(recorder.is_active),
                default_name=await default_name(),
                auto=auto,
                configured_available=bool(
                    fallback and live_devices and await asyncio.to_thread(microphone_present, configured_name)
                ),
                using_fallback=fallback,
            )
            if (
                state.zero_seconds >= 2.0
                and not self.capture_notice
                and time.monotonic() - permission_checked_at >= 10.0
            ):
                # Pure zeros from the start of listening is also what a refused macOS permission looks like.
                permission_checked_at = time.monotonic()
                await self._check_microphone_permission()
            if watchdog.recovered(state):
                self.capture_notice = ""
                logger.info("Microphone working again (%s)", recorder.input_device_name)
                await self._capture_status("microphone_recovered", "", f"Listening on {recorder.input_device_name}.")
            decision = watchdog.check(state)
            if decision.action == "notice":
                self.capture_notice = decision.message
                logger.warning("Microphone: %s", decision.message)
                await self._capture_status("microphone_problem", decision.reason, decision.message)
            elif decision.action == "reopen":
                self.capture_notice = decision.message
                logger.warning("Microphone: %s", decision.message)
                await self._capture_status("microphone_problem", decision.reason, decision.message)
                try:
                    opened = await asyncio.to_thread(recorder.reopen)
                except Exception as error:
                    recorder.last_error = str(error)
                    opened = False
                if opened:
                    watchdog.reopen_succeeded(await default_name(), recorder.last_error)
                    self._mic_quality.reset()  # possibly another microphone: judge it afresh
                else:
                    watchdog.reopen_failed()
                    logger.warning("Microphone could not be reopened: %s", recorder.last_error or "no device")
            if live_devices:
                # The speakers too: follow a new default output (headphones unplugged) while Heliox is not talking.
                speaker = await asyncio.to_thread(default_speaker_name)
                if await asyncio.to_thread(_output_line.check_device, speaker):
                    await self._capture_status("speaker_changed", "", _output_line.notice)

    async def _check_microphone_permission(self) -> bool:
        """On macOS, say at once when the microphone permission was refused or not yet granted (it never prompts).
        Returns True when it reported a problem; elsewhere the check answers "cannot tell" and nothing is said."""
        try:
            from pilot.system.macos_permissions import PermissionState, microphone_state, report

            state = await asyncio.to_thread(microphone_state)
        except Exception:
            logger.debug("Microphone permission check failed", exc_info=True)
            return False
        if state not in (PermissionState.DENIED, PermissionState.NOT_ASKED):
            return False
        message = report("microphone", state).message
        if message and message != self.capture_notice:
            self.capture_notice = message
            if self._capture_watchdog is not None:
                self._capture_watchdog.report_problem("permission")
            logger.warning("Microphone: %s", message)
            await self._capture_status("microphone_problem", "permission", message)
        return True

    async def _capture_status(self, status: str, reason: str, message: str) -> None:
        if not self._on_status:
            return
        try:
            await self._on_status(status, {"reason": reason, "message": message})
        except Exception:
            logger.debug("Microphone status callback failed", exc_info=True)

    async def _warm_transcriber(self) -> None:
        """Load the configured local recognizer before the first utterance."""

        started = time.perf_counter()
        try:
            await self._warm_turn_taking()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.info("Turn-taking models unavailable: %s", error)
        try:
            if self.config.voice.transcription_engine in {"auto", "faster_whisper", "streaming"}:
                await asyncio.to_thread(_get_faster_whisper_model, self.config.voice.whisper_model)
            else:
                await asyncio.to_thread(_get_whisper_model, self.config.voice.whisper_model)
            parakeet = self._parakeet()
            if parakeet is not None:
                await asyncio.to_thread(parakeet.load)  # about 4 s; otherwise the first sentence pays for it
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

        if self._call_watch_task and not self._call_watch_task.done():
            self._call_watch_task.cancel()
            await asyncio.gather(self._call_watch_task, return_exceptions=True)
        self._call_watch_task = None
        if self._capture_watch_task and not self._capture_watch_task.done():
            self._capture_watch_task.cancel()
            await asyncio.gather(self._capture_watch_task, return_exceptions=True)
        self._capture_watch_task = None

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
                # Keep what was said while the last utterance was being transcribed: the accurate pass takes
                # about a second, and a turn that ended at a pause ("Hey Heliox, ...") is usually still going.
                transcript = await self._record_and_transcribe(duration=3, timeout=None, keep_backlog=True)

                if not transcript or transcript.strip() == "No speech detected":
                    await asyncio.sleep(0.2)
                    continue

                self.last_transcript = transcript.strip()
                self.transcripts_received += 1
                transcript_lower = transcript.lower().strip()

                if self.in_call and not (self._diagnostic_waiter and not self._diagnostic_waiter.done()):
                    # In a call, what is said is for the people in the call: not a command, not for Live, and
                    # not even a wake phrase, which nothing shows to be safe from meeting speech.
                    logger.debug("Ignored speech during a call (%s)", self.in_call)
                    self.end_interruption()
                    continue

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
                wake_free_allowed = self._wake_free_suppression_depth == 0 and (
                    capture_generation == self._wake_free_suppression_generation
                    # The wait for this utterance began before Heliox spoke, but the person only started
                    # talking after it finished: a reply, not an echo.
                    or self._began_after_heliox_spoke()
                )
                conversational_follow_up = wake_free_allowed and (
                    self.config.voice.continuous_conversation_enabled or self.follow_up_remaining_seconds > 0
                )

                interruption = self._take_interruption()
                by_button = interruption == "button"
                interrupted = bool(interruption)
                # The whole leading phrase first ("Here Heliox, remind me…" leaves "remind me…", not "Here , remind").
                heard_as = loose_wake_prefix(transcript)
                if heard_as:
                    wake_detected = True
                    command_text = transcript[len(heard_as) :].strip(" \t,.:;!?-")
                    logger.info("Wake phrase recognised: '%s'", heard_as)
                for wake in () if wake_detected else self.wake_words:
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

                conversation = interrupted or (wake_free_allowed and self.conversation_open())
                conversational_follow_up = conversational_follow_up or conversation

                if not wake_detected and not conversational_follow_up:
                    await asyncio.sleep(0.1)
                    continue

                voiced = getattr(self._recorder, "last_voiced_seconds", None)
                cut_off = interruption == "barge_in" and is_cut_off_stop(voiced if isinstance(voiced, float) else None)
                if cut_off:
                    # Not shown or kept as the person's words: they are not what was said.
                    logger.info("A short cut-off over Heliox (heard as %r) is taken as stop", transcript.strip())
                    command_text = "Stop"
                    wake_detected = False
                    conversational_follow_up = True
                elif conversational_follow_up and not wake_detected and conversation and not by_button:
                    if not should_act_on_follow_up(transcript, question_open=self.question_open):
                        # Said while talking with Heliox, but not to it as a request: Live may check it.
                        logger.debug("Overheard in an open conversation: '%s'", transcript_lower)
                        if self._on_overheard:
                            try:
                                await self._on_overheard(transcript.strip())
                            except Exception:
                                logger.debug("Overheard-speech callback failed", exc_info=True)
                        continue
                    command_text = transcript.strip(" \t,.:;!?-")
                    logger.info("Conversational reply detected: '%s'", command_text)
                elif conversational_follow_up and not wake_detected and not interrupted:
                    # With no conversation open, only a request counts: a bare "No" or "Yes" is someone else's.
                    if self.config.voice.continuous_conversation_enabled and (
                        not _looks_like_autonomous_voice_intent(transcript_lower) or is_short_decision(transcript_lower)
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
                elif conversational_follow_up and not wake_detected:
                    # After the microphone button: whatever the person says is for Heliox. The window stays open
                    # for the rest of the conversation, so "No" after "Skip the backup" still counts.
                    command_text = transcript.strip(" \t,.:;!?-")
                    logger.info("Addressed with the microphone button: '%s'", command_text)

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
                                "transcript": "" if cut_off else transcript,
                                "wake_free": conversational_follow_up and not wake_detected,
                                "cut_off": cut_off,
                            },
                        )
                    except Exception:
                        pass

                if not command_text or (len(command_text) < 3 and not is_short_decision(command_text)):
                    if self._on_status:
                        try:
                            await self._on_status(
                                "listening",
                                {"message": "I'm listening..."},
                            )
                        except Exception:
                            pass

                    command_text = await self._record_and_transcribe(duration=8, timeout=8.0, keep_backlog=True)

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
                    ended = getattr(self._recorder, "last_turn_ended_at", None)
                    ended = ended if isinstance(ended, float) else None
                    task = asyncio.create_task(self._dispatch_command(command_text, ended))
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

    async def _dispatch_command(self, command_text: str, turn_ended_at: float | None = None) -> None:
        """Run a recognized command without pausing ambient listening."""
        started = time.perf_counter()
        if turn_ended_at is not None:
            voice_turn_ended_at.set(turn_ended_at)  # this task's own context: one value per dispatch
            logger.info("Last word to dispatch: %.0f ms", 1000 * (time.monotonic() - turn_ended_at))
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

    def _parakeet(self) -> Any:
        """The Parakeet final pass when it is chosen and its files are present, else None (Whisper is used)."""
        if getattr(self.config.voice, "final_pass", "whisper") != "parakeet":
            return None
        if str(getattr(self.config.voice, "language", "auto")).casefold() not in {"auto", "en", "english"}:
            return None  # Parakeet is English-only: another language set in Settings keeps Whisper
        from pilot.config import DATA_DIR
        from pilot.system.parakeet_asr import ParakeetTranscriber, model_dir

        folder = model_dir(getattr(self.config.voice, "parakeet_model_dir", ""), DATA_DIR)
        current = getattr(self, "_parakeet_transcriber", None)
        if current is None or current.folder != folder:
            current = ParakeetTranscriber(folder)
            self._parakeet_transcriber = current
            if not current.available:
                logger.warning("voice.final_pass is parakeet but its model is missing; using Whisper (%s)", folder)
        return current if current.available else None

    async def _transcribe_path(self, audio_path: str, *, speculative: bool = False) -> str:
        """Transcribes an already-recorded WAV file with multilingual support. ``speculative``: started before the
        turn was known to have ended, so nothing is offered to Live yet (the person may still be talking)."""
        parakeet = self._parakeet()
        if parakeet is not None:
            samples = _load_pcm_wav_for_whisper(audio_path)
            if samples is not None:
                from pilot.system.parakeet_asr import merge_wake

                samples = _normalize_voice_audio(samples)
                self.last_transcription_engine = "parakeet"
                # Whisper reads the utterance for the wake phrase at the same moment Parakeet reads it for the
                # words: the two run side by side instead of one after the other.
                opening = _write_wav([_to_int16(samples[: int(WAKE_OPENING_SECONDS * 16000)])], 16000)

                def whole_utterance_check() -> asyncio.Task[str]:
                    # English, not detected: on the owner's desk microphone automatic detection read "Hey Heliox,
                    # what can I say?" as Dutch ("Heliox.com, nice video.") in 2 of 5 cuts, and English hit 9 of 9,
                    # twice as fast. Parakeet, whose words are used, is English-only anyway.
                    task = asyncio.ensure_future(self._transcribe_whisper_text(opening, language="en"))
                    task.add_done_callback(lambda done: done.cancelled() or done.exception())
                    return task

                early_wake = getattr(self, "_early_wake", None)
                # With an early reading of the opening under way, the whole-utterance check waits to see whether
                # it is needed: two Whisper passes at once would slow both on a small machine.
                wake_check = whole_utterance_check() if early_wake is None else None
                said = await asyncio.to_thread(_counted, parakeet.transcribe, samples)
                addressed = self._addressed()  # the microphone button, or talking over Heliox
                in_conversation = addressed or self.conversation_open()
                if addressed and self._button_until and len(said.split()) > 2:
                    # The button was pressed for this sentence: no wake phrase to look for, nothing to wait for.
                    _cancel(wake_check)
                    return said
                if addressed and self._is_short_cut_off():
                    # A word or two said over Heliox: what it does next does not depend on a wake phrase.
                    _cancel(wake_check)
                    return said
                if in_conversation and len(said.split()) > 2 and looks_like_conversation_turn(said):
                    # Mid-conversation a reply needs no wake phrase: answer from Parakeet without waiting.
                    _cancel(wake_check)
                    return said
                if early_wake is not None:
                    try:
                        heard_early = await asyncio.shield(early_wake)  # shared by every pass of this utterance
                    except Exception:
                        heard_early = ""
                    if early_wake_agrees(heard_early, said):
                        self.last_wake_source = "opening"
                        return merge_wake(heard_early, said, loose_wake_prefix)
                    wake_check = whole_utterance_check()
                self.last_wake_source = "whole"
                if in_conversation and len(said.split()) > 2 and self._on_overheard and not speculative:
                    # While Whisper decides whether this began with the wake phrase, Live can already check the
                    # words ("…Monday, the twelfth of November"): a correction does not depend on the wake phrase,
                    # and Live never acts, it only speaks up. The same sentence offered again later is not repeated.
                    self._schedule(self._on_overheard, said)
                heard = await wake_check
                if not said and not heard:
                    return ""
                return merge_wake(heard, said, loose_wake_prefix)
        return await self._transcribe_whisper_text(audio_path)

    async def _transcribe_whisper_text(self, audio_path: str, language: str | None = None) -> str:
        try:
            result = await _transcribe_speech(
                audio_path,
                language or self.config.voice.language,
                self.config.voice.whisper_model,
                # The streaming engine falls back to the automatic Whisper path when it has no text.
                "auto"
                if self.config.voice.transcription_engine == "streaming"
                else self.config.voice.transcription_engine,
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
        keep_backlog: bool = False,
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
            streaming = None
            if self._recorder.is_active:
                streaming = self._configure_turn_taking()
                # The first utterance after start is often waited for before the recognisers finish loading;
                # they are picked up when the person starts talking.
                self._recorder.on_speech_start = self._configure_turn_taking
                speculation = self._speculate_final_pass()
                self._early_wake = None
                self._recorder.on_opening = self._start_early_wake if self.early_wake_enabled else None
                try:
                    audio_path = await self._recorder.record_utterance(timeout=timeout, keep_backlog=keep_backlog)
                finally:
                    self._recorder.on_candidate_end = self._recorder.on_candidate_cancel = None
                    self._recorder.on_opening = None
                if not audio_path:
                    speculation.cancel()
                    return ""
                if streaming is None and self._recorder.stream_sink is not None:
                    streaming = self._recorder.stream_sink
            else:
                audio_path = await _record_audio(duration)

            started = time.perf_counter()
            early = speculation.take() if self._recorder.is_active else None
            heard = await streaming.finish() if streaming is not None else ""
            if self._cutting_off(final=True):
                # A short cut-off over Heliox is a stop whatever it says (is_cut_off_stop): no recogniser is waited
                # for, and the streaming guess is kept only for the log.
                transcript = heard or "(cut off)"
                self.last_transcription_engine = "none"
            elif heard and is_safe_fast_phrase(heard):
                # "No", "Stop", "Wait": safe to act on from the fast recogniser (they only stop or deny).
                transcript = heard
                self.last_transcription_engine = "streaming"
            else:
                # Everything else is acted on only in the accurate pass's words. The fast recogniser is for the
                # live caption and turn-taking: on a real voice through a Bluetooth headset it heard "Hey Heliox,
                # remind me that the launch call is Monday..." as "Do you know my way down the lawn spot...".
                self.last_transcription_engine = "whisper"  # _transcribe_path says "parakeet" when it used it
                # Started at the candidate end, 200 ms into the quiet, when the turn ended there: most of the pass
                # has already run by the time the end of turn is decided.
                transcript = await early if early is not None else await self._transcribe_path(audio_path)
                if not transcript and heard and not self._final_pass_available():
                    transcript = heard
                    self.last_transcription_engine = "streaming"
                if transcript and heard and self._on_partial and not self._cutting_off(final=True):
                    self._schedule(self._on_partial, transcript)  # the caption shows the words acted on
            if early is not None and not early.done():
                early.cancel()
            self._early_wake = None
            if self._recorder.is_active and not self._mic_quality.verdict:
                self._schedule_mic_quality(audio_path)
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

    def _schedule_mic_quality(self, audio_path: str) -> None:
        """Judge the microphone from this utterance in a worker thread, off the path to dispatch."""

        def observe() -> str:
            samples = _load_pcm_wav_for_whisper(audio_path)
            return self._mic_quality.observe(samples) if samples is not None else ""

        async def judge() -> None:
            try:
                verdict = await asyncio.to_thread(observe)
            except Exception:
                logger.debug("Microphone quality check failed", exc_info=True)
                return
            if verdict and verdict != "ok":
                from pilot.system.mic_quality import MESSAGES

                headset = await asyncio.to_thread(self._recorder.device_key)
                logger.warning("Microphone %s: %s", headset, MESSAGES[verdict])
                offer = await asyncio.to_thread(self._other_input_for, headset) if verdict == "narrowband" else None
                if self._on_status:
                    try:
                        await self._on_status(
                            "microphone_quality",
                            {
                                "reason": verdict,
                                "message": MESSAGES[verdict],
                                "headset": headset,
                                "suggested_input": offer,
                            },
                        )
                    except Exception:
                        logger.debug("Microphone status callback failed", exc_info=True)

        asyncio.ensure_future(judge())

    def _other_input_for(self, headset: str) -> dict[str, str] | None:
        """A microphone to use instead of a headset in call mode: {"id", "name"}, or None when there is none."""
        try:
            import sounddevice as sd

            from pilot.system.audio_health import pulse_sources
            from pilot.system.mic_quality import suggest_input

            # Linux: the PulseAudio/PipeWire sources by their stable names (PortAudio only sees "pulse" there).
            devices = pulse_sources() or list_audio_input_devices(sd)
            return suggest_input(devices, headset, sys.platform)
        except Exception:
            logger.debug("Could not list other microphones", exc_info=True)
            return None

    def _start_early_wake(self, frames: list[Any]) -> None:
        """Read the opening for the wake phrase now, while the person is still speaking (Parakeet path only)."""
        if self._parakeet() is None or _recognisers_running():
            return
        path = _write_wav(frames, self._recorder.sample_rate)
        task = asyncio.ensure_future(self._transcribe_whisper_text(path, language="en"))
        task.add_done_callback(lambda done: done.cancelled() or done.exception())
        self._early_wake = task

    def _speculate_final_pass(self) -> _Speculation:
        """Arm the recorder to start the final pass at each candidate end of turn (see _Speculation)."""
        speculation = _Speculation(self)
        self._recorder.on_candidate_end = speculation.start
        self._recorder.on_candidate_cancel = speculation.cancel
        return speculation

    def get_stats(self) -> dict:
        """Return listener statistics."""
        return {
            "running": self._running,
            "wake_words": self.wake_words,
            "listening_for_command": self._listening_for_command,
            "language": self.last_detected_language,
            "configured_language": self.config.voice.language,
            "transcription_engine": self.config.voice.transcription_engine,
            **self._final_pass_status(),
            "whisper_model": self.config.voice.whisper_model,
            # Where the loaded Whisper runs ("cpu"/"cuda"; "" before it loads) and why (the GPU arbiter's reason).
            "whisper_device": next(iter(sorted({key[1] for key in _faster_whisper_model_cache})), ""),
            "whisper_device_reason": _faster_whisper_device_reason,
            "turn_detection": getattr(self.config.voice, "turn_detection", "smart_turn"),
            "turn_detector_ready": bool(self._turn_detector is not None and self._turn_detector.ready),
            "turn_detector_error": self._turn_detector.error if self._turn_detector is not None else "",
            "streaming_ready": bool(self._streaming is not None and self._streaming.ready),
            "streaming_error": self._streaming.error if self._streaming is not None else "",
            "last_transcription_engine": self.last_transcription_engine,
            "last_turn_probability": self._recorder.last_turn_probability,
            "last_endpoint_reason": self._recorder.last_endpoint_reason,
            "last_endpoint_ms": self._recorder.last_endpoint_ms,
            "last_voiced_seconds": self._recorder.last_voiced_seconds,
            "vad_recorder_active": self._recorder.is_active,
            "input_device": self._recorder.input_device,
            "input_device_name": self._recorder.input_device_name,
            "configured_input_device": self._recorder.preferred_device,
            "using_fallback_input": self._recorder.using_fallback_input,
            "input_device_notice": self._recorder.input_device_notice,
            "input_error": self._recorder.last_error,
            # What is wrong with the microphone now ("" when it works), and what Heliox told the person.
            "capture_problem": getattr(self._capture_watchdog, "problem", "") if self._capture_watchdog else "",
            "capture_notice": self.capture_notice,
            # "ok", "narrowband" (Bluetooth call mode), "quiet" or "clipping"; "" until enough speech was heard.
            "mic_quality": self._mic_quality.verdict,
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
