"""The voice a face brings: what Heliox does with the voice a face pack names when a person chooses that face.

A face pack's manifest may name the Kokoro voice that suits the face, with a plain hint ("feminine" or "masculine")
for when that exact voice cannot speak here (docs/FACE_PACKS.md, "voice"). Before this, the man's face spoke with
the product's default voice, a woman's, because nothing paired a face with a voice.

``plan_face_voice`` decides, from the face's voice and the person's voice settings:
- the face's voice is already Heliox's: nothing to do ("same");
- the person has never chosen a voice (``voice.tts_voice_chosen`` is false) and the face's voice can speak here:
  it is switched with the face ("offer", switched by ``face_voice`` with apply "auto");
- the person chose a voice: it is kept, and the face's voice is offered with one control ("offer");
- the face's voice file is not on disk: nothing is switched or fetched; accepting the offer is the consent, and
  Kokoro fetches the file (about half a megabyte) the first time it speaks, the way ``voice_for_language_apply``
  adds a voice ("fetch");
- Heliox speaks with another engine, or in a language the face's voice does not speak: nothing changes. In another
  language, a Kokoro voice of that language matching the hint is offered when there is one ("unavailable").
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

from pilot.system.voice_by_language import KOKORO_VOICE_BYTES, LANGUAGE_NAMES, TTS_VOICES, kokoro_language

HINTS = {"feminine": "f", "masculine": "m"}


@dataclass(frozen=True)
class FaceVoicePlan:
    voice: str  # the voice offered or in use; "" when the face names none Heliox knows
    action: str  # "none" | "same" | "offer" | "fetch" | "unavailable"
    download_bytes: int  # what using ``voice`` fetches the first time it speaks
    say: str  # one plain sentence for the Face row; "" for none

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def face_voice_valid(voice: object) -> bool:
    """A Kokoro voice Heliox can be set to (voice.tts_voice accepts it)."""
    return isinstance(voice, str) and voice in TTS_VOICES and bool(kokoro_language(voice))


def _matching(language: str, hint: str, usable: Callable[[str], bool]) -> str:
    """A Kokoro voice for ``language`` of the hinted kind (the second letter of its name), or ""."""
    letter = HINTS.get(hint, "")
    if not letter:
        return ""
    for voice in sorted(TTS_VOICES):
        if kokoro_language(voice) == language and voice[1] == letter and usable(voice):
            return voice
    return ""


def plan_face_voice(
    voice: str,
    hint: str,
    *,
    engine: str,
    current: str,
    chosen: bool,
    language: str,
    on_disk: Callable[[str], bool],
    reader_ok: Callable[[str], bool] = lambda _language: True,
) -> FaceVoicePlan:
    """What the Face row says and offers for a face whose pack names ``voice`` (and ``hint``)."""
    if not face_voice_valid(voice):
        return FaceVoicePlan("", "none", 0, "")
    if engine == "kokoro_tts" and current == voice:
        return FaceVoicePlan(voice, "same", 0, "")
    if engine != "kokoro_tts":
        now = "Pocket TTS" if engine == "pocket_tts" else "the system voice"
        return FaceVoicePlan(
            voice, "unavailable", 0,
            f"This face's voice is {voice}, a Kokoro voice. Heliox speaks with {now}, so the voice stays as it is; "
            "to hear this face's voice, choose Kokoro TTS under Voice.",
        )  # fmt: skip
    spoken = language or "en"
    if kokoro_language(voice) != spoken:
        name = LANGUAGE_NAMES.get(spoken, spoken)
        other = _matching(spoken, hint, lambda _voice: reader_ok(spoken))
        if other and other != current:
            fetch = not on_disk(other)
            return FaceVoicePlan(
                other, "fetch" if fetch else "offer", KOKORO_VOICE_BYTES if fetch else 0,
                f"This face's voice, {voice}, speaks {LANGUAGE_NAMES.get(kokoro_language(voice), 'English')}. "
                f"{other} is a {hint} {name} voice that suits it"
                + ("; about half a megabyte is fetched the first time it speaks." if fetch else "."),
            )  # fmt: skip
        return FaceVoicePlan(
            voice, "unavailable", 0,
            f"This face's voice, {voice}, speaks {LANGUAGE_NAMES.get(kokoro_language(voice), 'English')}, and "
            f"Heliox speaks {name}, so the voice stays as it is.",
        )  # fmt: skip
    kept = f"Your voice, {current}, is kept. " if chosen else ""
    if not on_disk(voice):
        return FaceVoicePlan(
            voice, "fetch", KOKORO_VOICE_BYTES,
            f"{kept}This face's voice, {voice}, is not on this computer yet; using it fetches about half a megabyte "
            "the first time it speaks.",
        )  # fmt: skip
    return FaceVoicePlan(voice, "offer", 0, f"{kept}This face's voice is {voice}.")


def switched_say(voice: str, *, by_face: bool) -> str:
    """The sentence after Heliox's voice changed to ``voice``."""
    if by_face:
        return f"Heliox now speaks with {voice}, this face's voice. Any other voice can be picked under Voice."
    return f"Heliox now speaks with {voice}."
