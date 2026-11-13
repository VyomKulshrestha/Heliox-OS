"""The words that stop Heliox or deny an approval, in the person's voice language, and what Whisper is told.

Voice can always stop and deny, never approve. A "no" or "stop" that Heliox does not understand in the person's
language leaves a task running, so these words are a safety matter: one table, keyed by the voice language, with
English always included (people everywhere say "stop" and "no"). Each language lists the words as its speakers say
them and as Whisper writes them, the loanword "stop" among them, and for Hindi both scripts, since Whisper writes
either. Adding a language is a few lines here and a case in tests/test_voice_words.py.

The words are matched as a whole utterance after ``normalize`` (case and punctuation gone, combining marks kept, so
"नहीं" stays itself), never inside a sentence: "para" alone stops, "para mañana" does not.

``transcription_prompt`` is what Whisper is told before it listens. In English that is a command vocabulary, which
helps Whisper with "stop", "approve" and app names. In any other language the vocabulary pulled short unclear words
towards English commands (13 November, synthetic speech: Mandarin "停", stop, written "Send."; Hindi "नहीं"
written "Lucky."), so another language gets the names only.
"""

from __future__ import annotations

import re
import unicodedata

# Greetings said before the name ("Oye Heliox", "Hallo Heliox"), in Latin letters as Whisper writes them; the name
# itself must still follow (pilot.system.addressing.vocative_wake), so a greeting alone wakes nothing. All of them
# count whatever the voice language: people greet in one language and speak in another. A greeting in another
# script needs no entry ("嘿，Heliox", "हे Heliox"): the wake rule reads Latin words only, so the name comes first.
# On 13 November, with Whisper small, "Oye Heliox" and "Hallo Heliox" were not wake phrases (synthetic speech).
GREETINGS: dict[str, tuple[str, ...]] = {
    "es": ("oye", "hoy", "hola", "ey", "eh"),  # "hoy": how Whisper small wrote "oye" once
    "de": ("hallo", "hey", "hi"),
    "hi": ("namaste", "suno", "arre", "hello"),
    "fr": ("salut", "dis", "allo", "allô", "coucou", "bonjour"),
    "it": ("ciao", "ehi", "senti", "ascolta"),
    "pt": ("oi", "ola", "olá", "ei", "escuta"),
}
GREETING_WORDS = frozenset(word for words in GREETINGS.values() for word in words if word.isascii())

# English: unchanged from the lists the server used before the table (pilot.server re-exports them).
ENGLISH_DENIALS = frozenset(
    {
        "no",
        "nope",
        "deny",
        "deny it",
        "do not",
        "dont",
        "don t",
        "do not do that",
        "don t do that",
        "dont do that",
        "cancel that",
        "reject",
        "reject it",
        "stop",
        "stop it",
        "cancel",
        "cancel it",
        "wait stop",
        "no stop",
        "no don t",
        "no dont",
        "not that",
    }
)
ENGLISH_STOPS = frozenset(
    {
        "stop",
        "cancel",
        "stop it",
        "stop that",
        "stop everything",
        "cancel everything",
        "stop all tasks",
        "cancel all tasks",
        "never mind",
        "nevermind",
        # Said over Heliox to make it stop talking: it stops and listens, and nothing is planned.
        "wait",
        "wait wait",
        "wait stop",
        "hold on",
        "wait a second",
        "one second",
        "one sec",
        "that s enough",
        "okay that s enough",
        "ok that s enough",
        "alright that s enough",
        "enough",
        "stop there",
    }
)

# Per language: "deny" denies a waiting approval and stops a running task; "stop" stops a task or Heliox's voice
# and plans nothing. The loanword "stop" is in every language's lists.
WORDS: dict[str, dict[str, tuple[str, ...]]] = {
    "es": {
        "deny": (
            "no",
            "no no",
            "no lo hagas",
            "eso no",
            "cancela",
            "cancélalo",
            "cancelar",
            "niégalo",
            "rechaza",
            "para",
            "stop",
            "alto",
            "detente",
            "no para",
        ),  # fmt: skip
        "stop": (
            "para",
            "para ya",
            "basta",
            "ya basta",
            "basta ya",
            "detente",
            "alto",
            "stop",
            "espera",
            "un momento",
            "cancela",
            "cancelar",
            "olvídalo",
            "déjalo",
            "ya está",
            "suficiente",
        ),  # fmt: skip
    },
    "de": {
        "deny": (
            "nein",
            "nein nein",
            "nein danke",
            "nicht",
            "nicht das",
            "lass das",
            "ablehnen",
            "abbrechen",
            "stopp",
            "stop",
            "halt",
            "nein stopp",
        ),  # fmt: skip
        "stop": (
            "stopp",
            "stop",
            "halt",
            "hör auf",
            "aufhören",
            "abbrechen",
            "warte",
            "warte mal",
            "moment",
            "einen moment",
            "genug",
            "das reicht",
            "es reicht",
            "vergiss es",
            "lass es",
        ),  # fmt: skip
    },
    "hi": {
        "deny": (
            "नहीं",
            "नही",
            "ना",
            "मत करो",
            "मत",
            "रद्द करो",
            "कैंसल करो",
            "रुको",
            "स्टॉप",
            "nahin",
            "nahi",
            "nahi nahi",
            "na",
            "mat karo",
            "radd karo",
            "cancel karo",
            "ruko",
            "stop",
        ),  # fmt: skip
        "stop": (
            "रुको",
            "रुक जाओ",
            "रुकिए",
            "बस",
            "बस करो",
            "बंद करो",
            "स्टॉप",
            "ठहरो",
            "एक मिनट",
            "छोड़ो",
            "ruko",
            "ruk jao",
            "rukiye",
            "bas",
            "bas karo",
            "band karo",
            "stop",
            "thehro",
            "ek minute",
            "chhodo",
            "rehne do",
        ),  # fmt: skip
    },
    "zh": {
        "deny": ("不", "不要", "不用", "不行", "别", "別", "取消", "拒绝", "拒絕", "停", "停止", "stop"),
        "stop": (
            "停",
            "停下",
            "停止",
            "停一下",
            "暂停",
            "暫停",
            "别说了",
            "別說了",
            "等一下",
            "等等",
            "够了",
            "夠了",
            "算了",
            "取消",
            "stop",
        ),  # fmt: skip
    },
    "fr": {
        "deny": ("non", "non non", "annule", "annuler", "refuse", "ne fais pas ça", "pas ça", "stop", "arrête"),
        "stop": (
            "stop",
            "arrête",
            "arrête toi",
            "arrêtez",
            "attends",
            "attendez",
            "ça suffit",
            "assez",
            "annule",
            "laisse tomber",
            "une seconde",
        ),  # fmt: skip
    },
    "it": {
        "deny": ("no", "no no", "annulla", "rifiuta", "non farlo", "stop", "fermati"),
        "stop": ("stop", "basta", "fermati", "ferma", "fermo", "aspetta", "annulla", "lascia perdere", "un attimo"),
    },
    "pt": {
        "deny": ("não", "nao", "não não", "cancela", "cancelar", "recusa", "não faça isso", "para", "pare", "stop"),
        "stop": (
            "para",
            "pare",
            "chega",
            "basta",
            "espera",
            "espere",
            "stop",
            "cancela",
            "deixa pra lá",
            "um momento",
        ),  # fmt: skip
    },
    "ja": {
        "deny": ("いいえ", "いや", "だめ", "ダメ", "キャンセル", "やめて", "やめ", "ストップ", "止めて"),
        "stop": (
            "止めて",
            "とめて",
            "止まって",
            "やめて",
            "やめ",
            "ストップ",
            "待って",
            "まって",
            "ちょっと待って",
            "もういい",
            "キャンセル",
        ),  # fmt: skip
    },
}

# Language names as settings or a system locale write them ("German_Germany" on Windows), to their codes.
_NAMES = {
    "english": "en", "spanish": "es", "español": "es", "german": "de", "deutsch": "de", "hindi": "hi",
    "chinese": "zh", "mandarin": "zh", "french": "fr", "français": "fr", "italian": "it", "italiano": "it",
    "portuguese": "pt", "português": "pt", "japanese": "ja",
}  # fmt: skip


def normalize(text: str) -> str:
    """Casefold, punctuation to spaces, whitespace collapsed; letters, digits and combining marks kept."""
    kept = "".join(
        ch if ch.isalnum() or ch.isspace() or unicodedata.category(ch).startswith("M") else " "
        for ch in text.casefold()
    )
    return " ".join(kept.split())


def system_language(system_locale: str | None = None) -> str:
    """The system's language code from its locale, "" when it cannot be told."""
    if system_locale is None:
        import locale
        import os

        try:
            system_locale = locale.getlocale()[0] or os.environ.get("LANG", "")
        except (ValueError, TypeError):
            system_locale = ""
    return language_code(system_locale or "")


def language_code(value: str) -> str:
    """ "de", "de-DE", "de_DE.UTF-8", "German", "German_Germany" -> "de"; "" for C/POSIX or nothing."""
    folded = (value or "").strip().casefold()
    if not folded or folded in {"c", "posix"} or folded.startswith(("c.", "posix")):
        return ""
    head = re.split(r"[_\-.@ ]", folded, maxsplit=1)[0]
    if head in _NAMES:
        return _NAMES[head]
    return head[:2] if len(head) in {2, 3} and head.isalpha() else head


# The language Whisper last detected on a pass in "auto" (a person who never opened Settings): "" before the
# first. One value for the process; the listener is the only one that sets it.
_detected = ""


def note_detected(language: str) -> None:
    global _detected
    _detected = language_code(language)


def last_detected() -> str:
    return _detected


def voice_language(setting: str, system_locale: str | None = None, detected: str | None = None) -> str:
    """The language the person speaks to Heliox: the voice language setting; for "auto", the language Whisper last
    detected when that is not English, else the system's (an English system and nothing detected: English)."""
    setting = (setting or "auto").strip()
    if setting.casefold() == "auto":
        heard = _detected if detected is None else language_code(detected)
        if heard and heard != "en":
            return heard
        return system_language(system_locale) or "en"
    return language_code(setting) or "en"


def has_words_for(language: str) -> bool:
    return language == "en" or language in WORDS


# The loanword "stop" as other scripts write it: Whisper writes it so whatever the voice language is set to (on
# 13 November a German "Stopp" came out as "ストップ!" in "auto"). It stops and denies in every language.
STOP_LOANWORDS = frozenset({"stop", "stopp", "ストップ", "स्टॉप", "стоп", "스톱", "סטופ"})


def _table(kind: str, language: str) -> frozenset[str]:
    english = ENGLISH_DENIALS if kind == "deny" else ENGLISH_STOPS
    words = WORDS.get(language, {}).get(kind, ())
    return english | STOP_LOANWORDS | frozenset(normalize(w) for w in words)


def is_denial(text: str, language: str = "en") -> bool:
    """The whole utterance denies (and stops) in English or the person's language."""
    return normalize(text) in _table("deny", language)


def is_stop(text: str, language: str = "en") -> bool:
    """The whole utterance stops a task or Heliox's voice in English or the person's language."""
    return normalize(text) in _table("stop", language)


def is_control_word(text: str, language: str = "en") -> bool:
    return is_denial(text, language) or is_stop(text, language)


_HAN = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")  # Chinese characters (and Japanese kanji)
_KANA = re.compile("[\u3040-\u30ff]+")  # Japanese kana


def spoken_word_count(text: str) -> int:
    """Words in an utterance. Chinese and Japanese have no spaces: each Chinese character counts as a word, and a
    run of kana as one word for every three characters ("ストップ" is one)."""
    count = 0
    for token in normalize(text).split():
        count += sum(len(run) for run in _HAN.findall(token))
        count += sum(max(1, round(len(run) / 3)) for run in _KANA.findall(token))
        count += len(_KANA.sub(" ", _HAN.sub(" ", token)).split())
    return count


# What Whisper reads before it listens (initial_prompt).
ENGLISH_PROMPT = (
    "Hey Heliox. Desktop command vocabulary: open, close, click, type, search, scroll, launch, switch, "
    "select, save, send, stop, continue, approve, deny. Proper names: Heliox, GitHub, Google, YouTube, "
    "Gmail, Spotify, Chrome, Visual Studio Code, PowerShell, Docker."
)
# The wake phrase and the names, nothing else. "Hey Heliox." first: without it, Whisper small wrote a Mandarin
# "嘿，黑利奥克斯" as "欸,欸裡奧克斯" and the wake phrase was lost; with it, "Hey, Heliox" (synthetic speech,
# 13 November). PowerShell is left out: Whisper base wrote a Spanish "Para." as "PowerShell." with it.
NAMES_PROMPT = "Hey Heliox. Heliox, GitHub, Google, YouTube, Gmail, Spotify, Chrome, Visual Studio Code, Docker."


def transcription_prompt(language: str, system_locale: str | None = None, detected: str | None = None) -> str:
    """The English vocabulary for English; the names only for any other language. ``language`` is the language a
    pass is run in ("en" for the wake read, the setting otherwise). For "auto": English when Whisper last detected
    English, the names when it last detected another language, and before any detection the system's language."""
    from pilot.system.speech_models import speaks_english

    if (language or "auto").strip().casefold() == "auto":
        heard = _detected if detected is None else language_code(detected)
        if heard:
            return ENGLISH_PROMPT if heard == "en" else NAMES_PROMPT
    return ENGLISH_PROMPT if speaks_english(language, system_locale) else NAMES_PROMPT
