"""Pocket TTS as a stream: audio chunks as the model makes them, for the voice on a computer without a usable card.

Measured on one development laptop (16-core laptop CPU, 19 November 2026): Pocket's first chunk of 80 ms came
0.074 to 0.136 s after the request (median 0.089 s), against 0.40 to 1.68 s for Kokoro on the same CPU making a
whole piece; it runs at 0.35 x real time on one thread and holds 1.1 GB. Its audio starts with quiet (the first sound
at speech level a median 0.15 s in, 0.45 s at worst), which ``stream`` drops inside the stream.

``PocketStream.stream(text, stop)`` is a generator of float32 mono chunks at ``sample_rate``:

* the text goes through ``spoken_digits.words_for_voice`` first: Pocket misreads digits ("3:15" came back as "three
  feet", "2026" as "2020"), so it never sees one;
* the quiet lead is held back until the first chunk with sound at SOUND_LEVEL, then KEEP_LEAD_SECONDS of it go out
  before that sound (the same lead the Kokoro pieces keep); a stream with no sound at all gives nothing;
* ``stop`` (a threading.Event) is read before each chunk is asked for and before each is handed out, and the
  model's generator is closed at once: nothing more is made once it is set, so a stop takes effect within one
  chunk (about 80 ms of audio, made in about 30 ms here).

The model is loaded once per PocketStream (``load``), offline when the files are on disk. Nothing here plays sound
or touches the output line: the playback path that starts on the first chunk is a separate piece of work.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from typing import Any

from pilot.system.spoken_digits import words_for_voice

DEFAULT_VOICE = "alba"
SOUND_LEVEL = 0.02  # the first sample at this level is where the voice starts to be heard
KEEP_LEAD_SECONDS = 0.03


def _load_pocket() -> Any:
    from pocket_tts import TTSModel

    return TTSModel.load_model()


def _to_array(chunk: Any) -> Any:
    import numpy as np

    value = chunk.detach().cpu().numpy() if hasattr(chunk, "detach") else chunk
    return np.asarray(value, dtype=np.float32).reshape(-1)


class PocketStream:
    """One Pocket model and its voices, made into streams of chunks."""

    def __init__(self, loader: Callable[[], Any] = _load_pocket, *, voice: str = DEFAULT_VOICE) -> None:
        self._loader = loader
        self.voice = voice or DEFAULT_VOICE
        self._model: Any = None
        self._states: dict[str, Any] = {}
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def sample_rate(self) -> int:
        return int(self.load().sample_rate)

    def load(self) -> Any:
        with self._lock:
            if self._model is None:
                self._model = self._loader()
            return self._model

    def _state(self, voice: str) -> Any:
        model = self.load()
        with self._lock:
            if voice not in self._states:
                self._states[voice] = model.get_state_for_audio_prompt(voice)
            return self._states[voice]

    def prepare(self, voice: str = "") -> None:
        """Load the model and the voice, and make one throwaway stream so the first real one is warm."""
        for _chunk in self.stream("Okay.", voice=voice):
            pass

    def stream(
        self, text: str, stop: threading.Event | None = None, *, voice: str = "", today: Any = None
    ) -> Iterator[Any]:
        """Chunks of ``text`` spoken, as they are made (see the module's notes)."""
        import numpy as np

        spoken = words_for_voice(text, today).strip()
        if not spoken:
            return
        model = self.load()
        state = self._state(voice or self.voice)
        rate = int(model.sample_rate)
        keep = int(KEEP_LEAD_SECONDS * rate)
        held: list[Any] = []  # quiet chunks before the first sound
        heard = False
        made = model.generate_audio_stream(state, spoken)
        try:
            while True:
                if stop is not None and stop.is_set():
                    return
                try:
                    chunk = _to_array(next(made))
                except StopIteration:
                    return
                if stop is not None and stop.is_set():
                    return
                if heard:
                    yield chunk
                    continue
                loud = np.flatnonzero(np.abs(chunk) >= SOUND_LEVEL)
                if loud.size == 0:
                    held.append(chunk)
                    continue
                heard = True
                lead = np.concatenate([*held, chunk]) if held else chunk
                start = max(0, sum(len(c) for c in held) + int(loud[0]) - keep)
                held.clear()
                yield lead[start:]
        finally:
            close = getattr(made, "close", None)
            if close is not None:
                close()
