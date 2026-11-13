"""Heliox's voice in other languages, for real, on a CI runner (the ``voice-languages`` workflow).

For each language Kokoro speaks, the voice ``voice_by_language`` offers is fetched from Hugging Face by Kokoro
itself, reads one sentence through Heliox's own synthesiser (``kokoro_tts._generate``, so the reader follows the
voice's first letter), and Whisper small, told the language, writes back what it heard. A check passes when the
audio is speech-length and Whisper heard most of the sentence's characters; it also shows the same sentence read
the old way (the American English reader), so the difference is in the log. German has no Kokoro voice: its line
says what the German sentence sounds like through the English voice, which is what a German speaker hears without
a German system voice.

It fetches the voices and Whisper small into the runner's caches, so it runs only on a runner (ci_only).
Each check prints one ``CHECK`` line; the exit code is the number of failed checks.
"""

from __future__ import annotations

import difflib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ci_only import require_runner  # noqa: E402

require_runner("fetches Kokoro's voices and Whisper small into this computer's Hugging Face cache")

import numpy as np  # noqa: E402

from pilot.system import kokoro_tts  # noqa: E402
from pilot.system.voice_by_language import KOKORO_VOICES, kokoro_lang_code  # noqa: E402

SENTENCES = {
    "es": "Mañana a las nueve tienes la reunión con Priya.",
    "fr": "Demain à neuf heures, tu as la réunion avec Priya.",
    "hi": "कल सुबह नौ बजे आपकी मीटिंग है।",
    "it": "Domani alle nove hai la riunione con Priya.",
    "ja": "明日の朝九時に会議があります。",
    "pt": "Amanhã às nove você tem a reunião com a Priya.",
    "zh": "你明天早上九点开会。",
}
GERMAN = "Morgen um neun hast du das Meeting mit Priya."
failures = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


def heard(model, audio: np.ndarray, language: str) -> str:
    x = np.interp(np.arange(0, len(audio), 24000 / 16000), np.arange(len(audio)), audio).astype(np.float32)
    segments, _info = model.transcribe(x, language=language, beam_size=5, temperature=0)
    return " ".join(s.text.strip() for s in segments)


def similar(a: str, b: str) -> float:
    keep = lambda t: "".join(ch for ch in t.casefold() if ch.isalnum())  # noqa: E731
    return difflib.SequenceMatcher(None, keep(a), keep(b)).ratio()


def main() -> int:
    from faster_whisper import WhisperModel

    kokoro_tts.configure(device="cpu")
    whisper = WhisperModel("small", device="cpu", compute_type="int8")
    for language, sentence in SENTENCES.items():
        voice = KOKORO_VOICES[language]
        try:
            started = time.perf_counter()
            audio, rate = kokoro_tts._generate(sentence, voice)
            seconds = len(audio) / rate
            text = heard(whisper, audio, language)
            score = similar(sentence, text)
            check(
                f"voice-{language}/{voice}-is-heard-as-the-sentence",
                seconds > 1.0 and score >= 0.6,
                f"reader {kokoro_lang_code(voice)!r}, {seconds:.1f} s of audio in {time.perf_counter() - started:.1f} s, "
                f"Whisper heard {text!r} (similarity {score:.2f})",
            )
        except Exception as error:  # a reader's package missing on the runner is a failure to see, not a crash
            check(f"voice-{language}/{voice}-is-heard-as-the-sentence", False, f"{type(error).__name__}: {error}")
        try:
            old, rate = kokoro_tts._generate(sentence, "af_heart")
            text = heard(whisper, old, language)
            print(f"INFO voice-{language}/through-the-english-reader: {len(old) / rate:.1f} s, Whisper heard {text!r} "
                  f"(similarity {similar(sentence, text):.2f})", flush=True)  # fmt: skip
        except Exception as error:
            print(f"INFO voice-{language}/through-the-english-reader: {type(error).__name__}: {error}", flush=True)
    audio, rate = kokoro_tts._generate(GERMAN, "af_heart")
    text = heard(whisper, audio, "de")
    print(f"INFO voice-de/no-kokoro-voice: through the English voice, Whisper heard {text!r} "
          f"(similarity {similar(GERMAN, text):.2f})", flush=True)  # fmt: skip
    return failures


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    raise SystemExit(main())
