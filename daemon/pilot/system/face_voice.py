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
- Heliox speaks with Pocket TTS: the face's approved Pocket voice is used the same way (``PACK_VOICES``' Pocket
  column, ``POCKET_BY_KIND``: alba for a feminine face, michael for a masculine one). Where Pocket has no approved
  voice for a face (a kind with none), the face brings its Kokoro voice instead of keeping a voice of the wrong kind: switched with the face while the person never
  chose a voice and Kokoro's model and the voice are on disk, otherwise offered with the download it takes;
- Heliox speaks with the system voice, or in a language the face's voice does not speak: nothing changes. In another
  language, a Kokoro voice of that language matching the hint is offered when there is one ("unavailable").

The daemon applies that rule itself whenever ``face.pack`` changes, by any route: Settings' Face row, the general
settings update from any client, a spoken setting, or a config file edited by hand (read at start and when the file is
reloaded). It knows each pack's voice from ``PACK_VOICES``, a copy of the ``voice`` field of every pack the window
ships; a test checks the two agree, so a pack that names a voice is added there too.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from pilot.system.voice_by_language import KOKORO_VOICE_BYTES, LANGUAGE_NAMES, TTS_VOICES, kokoro_language
from pilot.system.voice_default import VOICES

HINTS = {"feminine": "f", "masculine": "m"}

# The Pocket TTS voice approved for each kind of face, from voices whose terms allow any use. alba was listened to and
# read back (numbers: 0 read wrong in 65 pieces once spoken as words). michael (VCTK, CC BY 4.0) was chosen from nine
# masculine candidates by read-back and measurement on 24 November 2026: 5 of 5 sentences read back, the steadiest
# loudness (1.1 dB apart), no clipping, pitch about 91 Hz; it is marked "checked", not yet heard by a person.
POCKET_BY_KIND: dict[str, str] = {"feminine": "alba", "masculine": "michael"}
# A Pocket voice is 3.6 to 7.1 MB, fetched by Pocket TTS the first time it speaks (speech_models.json: each voice's
# "bytes", else "voice_bytes").
POCKET_VOICE_BYTES = int(VOICES.get("pocket_tts", {}).get("voice_bytes", 6_000_000))


def pocket_voice_bytes(voice: str) -> int:
    """What fetching the Pocket voice ``voice`` downloads: its own size when known, else the usual one."""
    term = VOICES.get("pocket_tts", {}).get("voice_terms", {}).get(voice, {})
    return int(term.get("bytes") or POCKET_VOICE_BYTES)


@dataclass(frozen=True)
class PackVoices:
    """One face pack's voice per engine. ``kokoro`` and ``hint`` copy the pack's manifest ("voice"); ``pocket`` is
    the approved Pocket TTS voice for the face, "" while none is approved."""

    kokoro: str
    hint: str
    pocket: str = ""

    def by_engine(self) -> dict[str, str]:
        return {"kokoro_tts": self.kokoro, "pocket_tts": self.pocket}


# The voices of each pack in tauri-app/ui/public/faces that names one (its manifest's "voice"), one column per engine;
# packs naming none are left out. tests/test_face_voice.py reads every manifest and fails when the Kokoro column or the
# hint differs.
PACK_VOICES: dict[str, PackVoices] = {
    "portrait-man": PackVoices("am_michael", "masculine", pocket=POCKET_BY_KIND["masculine"]),
    "portrait-woman": PackVoices("af_heart", "feminine", pocket=POCKET_BY_KIND["feminine"]),
}


def pack_voice(pack: str) -> tuple[str, str] | None:
    """(Kokoro voice, hint) the face pack ``pack`` names, or None for "off" and a pack without a voice."""
    found = PACK_VOICES.get(pack)
    return (found.kokoro, found.hint) if found else None


def pocket_voice(pack: str, voice: str, hint: str) -> str:
    """The approved Pocket voice for a face naming Kokoro ``voice`` with ``hint``: the pack's own column when ``pack``
    is that face, else the one approved for its kind; "" when none is approved."""
    from pilot.system.voice_default import pocket_voice_allowed

    found = PACK_VOICES.get(pack)
    chosen = found.pocket if found is not None and found.kokoro == voice else POCKET_BY_KIND.get(hint, "")
    return chosen if chosen and pocket_voice_allowed(chosen) else ""  # only voices whose terms allow any use


def face_voices(pack: str) -> dict[str, str] | None:
    """The chosen face's voice per engine (for the voice a new install starts with), None for no face or no voice."""
    found = PACK_VOICES.get(pack)
    return found.by_engine() if found else None


@dataclass(frozen=True)
class FaceVoicePlan:
    voice: str  # the voice offered or in use; "" when the face names none Heliox knows
    action: str  # "none" | "same" | "offer" | "fetch" | "unavailable" | "kept" | "downloading"
    download_bytes: int  # what using ``voice`` fetches the first time it speaks
    say: str  # one plain sentence for the Face row; "" for none
    engine: str = "kokoro_tts"  # the engine ``voice`` belongs to: using it switches Heliox to that engine
    # For "fetch": whether the download can start now, and why not in a few words (the daemon fills these).
    download_ok: bool = True
    download_reason: str = ""

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


def _megabytes(size: int) -> str:
    return "about half a megabyte" if size < 1_000_000 else f"about {round(size / 1_000_000)} MB"


def plan_face_voice(voice: str, hint: str, *, kept: bool = False, **given: Any) -> FaceVoicePlan:
    """What the Face row says and offers for a face whose pack names Kokoro ``voice`` (and ``hint``); see _plan.
    ``kept``: the person chose to keep their own voice with this face ("Use this face with the voice I have"), so
    nothing is offered or switched for it again ("kept")."""
    plan = _plan(voice, hint, **given)
    if kept and plan.action in {"offer", "fetch"}:
        current = given.get("current", "")
        return FaceVoicePlan(plan.voice, "kept", 0, f"You chose to keep your voice, {current}, with this face.",
                             plan.engine)  # fmt: skip
    return plan


def _plan(
    voice: str,
    hint: str,
    *,
    engine: str,
    current: str,
    chosen: bool,
    language: str,
    on_disk: Callable[[str], bool],
    reader_ok: Callable[[str], bool] = lambda _language: True,
    pocket: str = "",
    pocket_on_disk: Callable[[str], bool] = lambda _voice: True,
    kokoro_model_on_disk: Callable[[], bool] = lambda: True,
    kokoro_model_bytes: int = 0,
) -> FaceVoicePlan:
    """What the Face row says and offers for a face whose pack names Kokoro ``voice`` (and ``hint``). ``pocket``:
    the face's approved Pocket TTS voice ("" for none yet). With Pocket TTS speaking, that voice is used the way the
    Kokoro one is with Kokoro; without one, the face brings its Kokoro voice, the Kokoro model's download
    (``kokoro_model_bytes``) included when it is not on disk."""
    if not face_voice_valid(voice):
        return FaceVoicePlan("", "none", 0, "")
    spoken = language or "en"
    if engine == "pocket_tts" and pocket and spoken == "en":
        if current == pocket:
            return FaceVoicePlan(pocket, "same", 0, "", "pocket_tts")
        kept = f"Your voice, {current}, is kept. " if chosen else ""
        if not pocket_on_disk(pocket):
            return FaceVoicePlan(
                pocket, "fetch", pocket_voice_bytes(pocket),
                f"{kept}This face's Pocket TTS voice, {pocket}, is not on this computer yet; using it fetches "
                f"{_megabytes(pocket_voice_bytes(pocket))} the first time it speaks.", "pocket_tts",
            )  # fmt: skip
        return FaceVoicePlan(pocket, "offer", 0, f"{kept}This face's Pocket TTS voice is {pocket}.", "pocket_tts")
    if engine == "pocket_tts":
        # No approved Pocket voice for this face: the face brings its Kokoro voice rather than keep one of the wrong kind.
        brought = _kokoro_plan(voice, hint, current=current, chosen=chosen, spoken=spoken, on_disk=on_disk,
                               reader_ok=reader_ok)  # fmt: skip
        if brought.action not in {"offer", "fetch"}:
            return brought
        why = "Pocket TTS has no approved voice for this face yet, so the face brings its Kokoro voice. "
        if kokoro_model_on_disk():
            return FaceVoicePlan(brought.voice, brought.action, brought.download_bytes, why + brought.say)
        size = brought.download_bytes + kokoro_model_bytes
        kept = f"Your voice, {current}, is kept. " if chosen else ""
        return FaceVoicePlan(
            brought.voice, "fetch", size,
            f"{kept}{why}Kokoro is not on this computer yet; using {brought.voice} fetches {_megabytes(size)} the "
            "first time it speaks.",
        )  # fmt: skip
    if engine != "kokoro_tts":
        return FaceVoicePlan(
            voice, "unavailable", 0,
            f"This face's voice is {voice}, a Kokoro voice. Heliox speaks with the system voice, so the voice stays as "
            "it is; to hear this face's voice, choose Kokoro TTS under Voice.",
        )  # fmt: skip
    if current == voice:
        return FaceVoicePlan(voice, "same", 0, "")
    return _kokoro_plan(voice, hint, current=current, chosen=chosen, spoken=spoken, on_disk=on_disk,
                        reader_ok=reader_ok)  # fmt: skip


def _kokoro_plan(
    voice: str,
    hint: str,
    *,
    current: str,
    chosen: bool,
    spoken: str,
    on_disk: Callable[[str], bool],
    reader_ok: Callable[[str], bool],
) -> FaceVoicePlan:
    """The face's Kokoro voice (or one of the language and kind) offered in place of ``current``."""
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
