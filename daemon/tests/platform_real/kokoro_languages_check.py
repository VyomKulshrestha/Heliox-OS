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
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ci_only import require_runner  # noqa: E402

require_runner("fetches Kokoro's voices and Whisper small into this computer's Hugging Face cache")

import numpy as np  # noqa: E402

from pilot.system import kokoro_tts, voice_by_language  # noqa: E402
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


def failure(error: BaseException) -> str:
    """The exception's type chain and its traceback, since a reader can raise with an empty message (run
    43341873370: the Japanese one raised RuntimeError("") from MeCab)."""
    chain, seen = [], error
    while seen is not None:
        chain.append(f"{type(seen).__name__}: {str(seen).strip() or '(no message)'}")
        seen = seen.__cause__ or seen.__context__
    return " <- ".join(chain) + "\n" + "".join(traceback.format_exception(error)).strip()


def resident_mb() -> float:
    """This process's resident memory, in MB."""
    import psutil

    return psutil.Process().memory_info().rss / 1_048_576


def timed(what: str, call):
    started = time.perf_counter()
    result = call()
    return result, time.perf_counter() - started


def japanese_refused() -> None:
    """Without the dictionary: plan_voice must not offer Kokoro's Japanese voice, must name the download and its size,
    and nothing may have been fetched for it."""
    fetched_before = _unidic_files()
    plan = voice_by_language.plan_voice("ja", "kokoro_tts", "af_heart", platform="linux")
    check(
        "voice-ja/refused-without-the-dictionary",
        plan.voice != "jf_alpha"
        and not (plan.change and plan.engine == "kokoro_tts")
        and "770 MB" in plan.say
        and "python -m unidic download" in plan.say
        and _unidic_files() == fetched_before,
        f"missing: {voice_by_language.reader_missing('ja')}; offered {plan.engine}/{plan.voice}, change={plan.change}; "
        f"said {plan.say!r}",
    )


def _unidic_files() -> int:
    """How many files unidic's dictionary folder holds (0 when there is none)."""
    try:
        import unidic

        folder = Path(unidic.DICDIR)
        return sum(1 for _ in folder.rglob("*")) if folder.is_dir() else 0
    except Exception:
        return 0


def main() -> int:
    from faster_whisper import WhisperModel
    from huggingface_hub import hf_hub_download

    kokoro_tts.configure(device="cpu")
    whisper = WhisperModel("small", device="cpu", compute_type="int8")
    # What a person waits the first time they hear their language, apart from the model itself: the model's weights
    # and the English reader are loaded first (a person who speaks English already has them), then for each
    # language the voice file is fetched, its reader started, and the sentence made, each timed on its own.
    before_model = resident_mb()
    _, model_seconds = timed("model", lambda: kokoro_tts._generate("Okay.", "af_heart"))
    after_english = resident_mb()
    print(
        f"INFO memory/model-and-english-reader: {after_english - before_model:.0f} MB (resident {after_english:.0f} MB)",
        flush=True,
    )
    print(f"INFO model-and-english-reader: {model_seconds:.1f} s (fetching and loading the weights, once)", flush=True)
    for language, sentence in SENTENCES.items():
        voice = KOKORO_VOICES[language]
        if language == "ja" and voice_by_language.reader_missing("ja"):
            # The product does not offer the Japanese voice without unidic's dictionary (about 770 MB, a download
            # only the person starts): the check follows it, and the reader itself is not exercised here.
            japanese_refused()
            continue
        try:
            _, fetch = timed("voice", lambda voice=voice: hf_hub_download(kokoro_tts.REPO_ID, f"voices/{voice}.pt"))
            _, reader = timed("reader", lambda voice=voice: kokoro_tts._get_pipeline(kokoro_lang_code(voice)))
            (audio, rate), made = timed(
                "synthesis", lambda voice=voice, sentence=sentence: kokoro_tts._generate(sentence, voice)
            )
            seconds = len(audio) / rate
            text = heard(whisper, audio, language)
            score = similar(sentence, text)
            check(
                f"voice-{language}/{voice}-is-heard-as-the-sentence",
                seconds > 1.0 and score >= 0.6,
                f"reader {kokoro_lang_code(voice)!r}, {seconds:.1f} s of audio; voice file {fetch:.1f} s, reader "
                f"{reader:.1f} s, synthesis {made:.1f} s; Whisper heard {text!r} (similarity {score:.2f})",
            )
        except Exception as error:  # a reader's package missing on the runner is a failure to see, not a crash
            check(f"voice-{language}/{voice}-is-heard-as-the-sentence", False, failure(error))
        try:
            old, rate = kokoro_tts._generate(sentence, "af_heart")
            text = heard(whisper, old, language)
            print(f"INFO voice-{language}/through-the-english-reader: {len(old) / rate:.1f} s, Whisper heard {text!r} "
                  f"(similarity {similar(sentence, text):.2f})", flush=True)  # fmt: skip
        except Exception as error:
            print(f"INFO voice-{language}/through-the-english-reader: {failure(error)}", flush=True)
    # Every language read with the one model the English pipeline loaded; what each language's reader added on top,
    # against what a pipeline with its own model costs (how it was before: one copy per language).
    after_languages = resident_mb()
    models = {id(pipeline.model) for pipeline in kokoro_tts._pipeline_cache.values() if pipeline.model is not None}
    check(
        "voice/one-model-for-every-language",
        len(models) == 1,
        f"{len(kokoro_tts._pipeline_cache)} readers, {len(models)} model(s); {after_languages - after_english:.0f} MB "
        f"for all the other languages' readers together (resident {after_languages:.0f} MB)",
    )
    try:
        from kokoro import KPipeline

        own = resident_mb()
        extra = KPipeline(lang_code="e", repo_id=kokoro_tts.REPO_ID, device="cpu")  # its own copy, as before
        print(f"INFO memory/a-second-copy-of-the-model: {resident_mb() - own:.0f} MB, what each language cost before",
              flush=True)  # fmt: skip
        del extra
    except Exception as error:
        print(f"INFO memory/a-second-copy-of-the-model: {failure(error)}", flush=True)
    audio, rate = kokoro_tts._generate(GERMAN, "af_heart")
    text = heard(whisper, audio, "de")
    print(f"INFO voice-de/no-kokoro-voice: through the English voice, Whisper heard {text!r} "
          f"(similarity {similar(GERMAN, text):.2f})", flush=True)  # fmt: skip
    return failures


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    raise SystemExit(main())
