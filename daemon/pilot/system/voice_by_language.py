"""Heliox's voice in the person's language: which voice speaks it, and what the person hears when there is none.

Kokoro reads text with the rules of one language, chosen by the first letter of its voice ("af_heart": American
English, "ef_dora": Spanish). Before this module every reply went through the American English reader whatever the
voice language: Spanish and German came out with English sounds, Hindi was partly read, and Mandarin was read as
the words "Chinese letter" nine times over (13 November, the language audit). Kokoro has voices for Spanish,
French, Hindi, Italian, Japanese, Brazilian Portuguese and Mandarin; it has none for German or most other languages.

``plan_voice`` picks, for the voice language:
1. the voice the person already has, when it speaks that language;
2. else Kokoro's voice for it, when Kokoro has one and the packages its reader needs are installed (Mandarin needs
   jieba, pypinyin and cn2an; Japanese needs pyopenjtalk). The voice file (about 0.5 MB) is fetched from Hugging
   Face by Kokoro the first time it speaks, the way the English voice is;
3. else a system voice of that language (Windows' speech voices, macOS's ``say`` voices, espeak-ng's languages);
4. else nothing changes, and the plan says plainly what the person hears: the replies read with English sounds,
   and where to add a voice for their language.
Nothing is changed or downloaded until the person accepts the offer (``voice_for_language_apply``).
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

# Kokoro's reader for each first letter of a voice name (kokoro.KPipeline's lang_code).
KOKORO_LANG_CODES = {"a": "en", "b": "en", "e": "es", "f": "fr", "h": "hi", "i": "it", "j": "ja", "p": "pt", "z": "zh"}
# The voice offered for each language Kokoro speaks.
KOKORO_VOICES = {
    "en": "af_heart",
    "es": "ef_dora",
    "fr": "ff_siwis",
    "hi": "hf_alpha",
    "it": "if_sara",
    "ja": "jf_alpha",
    "pt": "pf_dora",
    "zh": "zf_xiaobei",
}
# What Kokoro's reader needs beyond its English one (misaki's extras), by language.
KOKORO_NEEDS = {"zh": ("jieba", "pypinyin", "cn2an"), "ja": ("pyopenjtalk", "fugashi", "unidic")}
# Japanese also needs unidic's dictionary on disk, which ``pip install unidic`` does not bring: it is fetched apart
# (``python -m unidic download``, about 770 MB). Without it Kokoro's Japanese reader raises as it starts ("Failed
# initializing MeCab ... no such file: .../unidic/dicdir/mecabrc", voice-languages run 43341873370).
# Scripts Kokoro's English reader cannot read at all: it says "Chinese letter" or "Japanese letter" for each
# character instead (the same run). Latin-script languages come out with English sounds.
NOT_LATIN = frozenset({"zh", "ja", "ko", "ru", "ar", "hi"})
KOKORO_VOICE_BYTES = 523_000  # a voice file is about half a megabyte

LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
    "pt": "Portuguese", "zh": "Mandarin", "de": "German", "nl": "Dutch", "pl": "Polish", "ru": "Russian",
    "tr": "Turkish", "ar": "Arabic", "ko": "Korean", "sv": "Swedish",
}  # fmt: skip

WHERE_TO_ADD = {
    "windows": "Windows Settings, Time & language, Speech, Add voices",
    "macos": "System Settings, Accessibility, Spoken Content, System voice, Manage voices",
    "linux": "espeak-ng (it speaks most languages; install the espeak-ng package)",
}


def kokoro_language(voice: str) -> str:
    """The language a Kokoro voice speaks, from its first letter; "" when it is not a Kokoro voice name."""
    return KOKORO_LANG_CODES.get(voice[:1], "") if re.match(r"^[a-z][fm]_[a-z]+$", voice or "") else ""


def kokoro_lang_code(voice: str) -> str:
    """kokoro.KPipeline's lang_code for a voice: its first letter, or "a" (American English) for anything else."""
    return voice[:1] if kokoro_language(voice) else "a"


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _japanese_dictionary() -> bool:
    """Whether unidic's dictionary is on disk (its mecabrc), which Kokoro's Japanese reader opens as it starts."""
    try:
        import unidic

        return (Path(unidic.DICDIR) / "mecabrc").is_file()
    except Exception:
        return False


def reader_missing(
    language: str, installed: Callable[[str], bool] = _installed, dictionary: Callable[[], bool] = _japanese_dictionary
) -> str:
    """What Kokoro's reader for ``language`` lacks on this computer, in words; "" when it can start."""
    missing = [module for module in KOKORO_NEEDS.get(language, ()) if not installed(module)]
    if missing:
        return ", ".join(missing) + (" are" if len(missing) > 1 else " is") + " not installed"
    if language == "ja" and not dictionary():
        return "the Japanese dictionary (unidic, about 770 MB, fetched with python -m unidic download) is not here"
    return ""


def without_a_voice(name: str, language: str) -> str:
    """What a person hears from the English voice in their language."""
    if language in NOT_LATIN:
        return f'so Heliox cannot read {name} replies aloud: the English voice says "letter" for each character'
    return f"so it reads {name} replies with English sounds"


@dataclass(frozen=True)
class VoicePlan:
    language: str  # the voice language, as a code ("de")
    engine: str  # the engine to use: "kokoro_tts", "os_native", or the current one when nothing changes
    voice: str  # the Kokoro voice, or the system voice's name
    change: bool  # True when this differs from what the person has
    download_bytes: int  # what the first use fetches (a Kokoro voice file), 0 for none
    say: str  # one plain sentence for the person: what they will hear

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def plan_voice(
    language: str,
    engine: str,
    voice: str,
    system_voice: str = "",
    *,
    system_voices: Iterable[tuple[str, str]] = (),
    platform: str = "windows",
    installed: Callable[[str], bool] = _installed,
    dictionary: Callable[[], bool] = _japanese_dictionary,
) -> VoicePlan:
    """The voice for ``language`` given the person's engine and voice, and the system voices (name, language code)."""
    name = LANGUAGE_NAMES.get(language, language)
    if not language or language == "en":
        return VoicePlan("en", engine, voice, False, 0, "")
    lacking = reader_missing(language, installed, dictionary)
    if engine == "kokoro_tts" and kokoro_language(voice) == language and not lacking:
        return VoicePlan(language, engine, voice, False, 0, f"Heliox speaks {name} with the voice {voice}.")
    system = [(n, lang) for n, lang in system_voices if lang == language]
    if engine == "os_native" and any(n == system_voice for n, _ in system):
        return VoicePlan(language, engine, system_voice, False, 0, f"Heliox speaks {name} with {system_voice}.")
    kokoro = KOKORO_VOICES.get(language, "")
    if kokoro and not lacking:
        return VoicePlan(
            language, "kokoro_tts", kokoro, True, KOKORO_VOICE_BYTES,
            f"Heliox can speak {name} with Kokoro's voice {kokoro}; about half a megabyte is fetched the first time.",
        )  # fmt: skip
    if system:
        chosen = system[0][0]
        return VoicePlan(language, "os_native", chosen, True, 0, f"Heliox can speak {name} with {chosen}, a voice already on this computer.")  # fmt: skip
    where = WHERE_TO_ADD.get(platform, WHERE_TO_ADD["linux"])
    missing = f" Kokoro's {name} voice cannot start here: {lacking}." if kokoro else ""
    if engine == "kokoro_tts" and kokoro_language(voice) == language:
        # Chosen already, but its reader raises as it starts: Heliox falls back to the system voice, which read a
        # Japanese sentence as "yen, pi, sigma" on Windows (13 November). Say so rather than call the voice working.
        return VoicePlan(
            language, engine, voice, False, 0,
            f"Heliox's {name} voice {voice} cannot start on this computer: {lacking}. Until it can, replies are read "
            f"by the system voice, which does not read {name}. A {name} voice can be added in {where}.",
        )  # fmt: skip
    return VoicePlan(
        language, engine, voice, False, 0,
        f"Heliox has no {name} voice on this computer, {without_a_voice(name, language)}.{missing} "
        f"A {name} voice can be added in {where}.",
    )  # fmt: skip


# ── The system's voices with their languages ──


def windows_voices(output: str) -> list[tuple[str, str]]:
    """``Name|Culture`` lines (from System.Speech) -> [(name, language code)]."""
    out = []
    for line in output.splitlines():
        name, _, culture = line.strip().partition("|")
        if name and culture:
            out.append((name.strip(), culture.strip().split("-")[0].casefold()))
    return out


def macos_voices(output: str) -> list[tuple[str, str]]:
    """``say -v ?`` lines ("Anna               de_DE    # Hallo! Ich heiße Anna.") -> [(name, language code)]."""
    out = []
    for line in output.splitlines():
        match = re.match(r"^(.+?)\s+([a-z]{2,3})[_-][A-Za-z0-9]+\s+#", line)
        if match:
            out.append((match.group(1).strip(), match.group(2)))
    return out


def espeak_voices(output: str) -> list[tuple[str, str]]:
    """``espeak-ng --voices`` table ("Pty Language Age/Gender VoiceName File Other") -> [(language, language code)].
    The voice is named by its language code, which is what espeak-ng's -v takes."""
    out = []
    for line in output.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            code = parts[1]
            out.append((code, code.split("-")[0].casefold()))
    return out
