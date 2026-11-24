"""The first-run pick, the face's voice and the Pocket allow-list agree: for every row of the voice table
(speech_models.json, "voice_rules") and every language it names, with no face and with each face that brings a voice,
the voice a new install starts with is on the list for its engine and is of the face's kind. The first-run screen
reads the same rows (tauri-app/ui/src/lib/firstRun.ts, voicePick: the first row whose machine matches and whose
languages hold "*" or the language); the rule is restated here from the file so the two cannot drift apart."""

from __future__ import annotations

import json

import pytest

from pilot.system import face_voice, voice_default
from pilot.system.face_voice import PACK_VOICES, face_voice_valid, face_voices, pocket_voice
from pilot.system.voice_default import (
    MANIFEST,
    default_voice,
    first_voice,
    pick,
    pocket_voice_allowed,
    pocket_voices_offered,
)

TABLE = json.loads(MANIFEST.read_text(encoding="utf-8"))
ROWS = TABLE["voice_rules"]["rows"]
LANGUAGES = sorted({lang for row in ROWS for lang in row["languages"] if lang != "*"} | {"en", "de", "ko"})
FACES = [None, *sorted(PACK_VOICES)]


def _screen_pick(machine: str, language: str) -> str:
    """firstRun.ts voicePick: the first row for the machine whose languages hold "*" or the language."""
    for row in ROWS:
        if row["machine"] == machine and ("*" in row["languages"] or language in row["languages"]):
            return row["engine"]
    return voice_default.DEFAULT_ENGINE


def _kind_letter(hint: str) -> str:
    return face_voice.HINTS[hint]


@pytest.mark.parametrize("machine", voice_default.MACHINES)
@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("face", FACES)
@pytest.mark.parametrize("starting", ["kokoro_tts", "pocket_tts"])
def test_the_first_voice_is_allowed_and_of_the_face_s_kind(machine, language, face, starting):
    rule = pick(machine, language)
    assert rule.engine == _screen_pick(machine, language)  # the daemon and the first-run screen pick the same
    voices = face_voices(face) if face else None
    # The voice set before the table applies: the face's own voice for that engine (choosing a face sets it), else
    # the engine's default.
    before = (voices or {}).get(starting) or default_voice(starting)
    found = first_voice(
        engine=starting,
        voice=before,
        chosen=False,
        machine=machine,
        language=language,
        models_on_disk=frozenset(),
        ever_loaded=False,
        face_voices=voices,
    )
    engine, voice = (found.engine, found.voice) if found else (starting, before)
    if found is None:
        # Nothing to change only when the install already starts on the engine the table (or the face) settles on.
        settled = rule.engine if voices is None or voices.get(rule.engine) else "kokoro_tts"
        assert starting == settled
    if engine == "pocket_tts":
        assert pocket_voice_allowed(voice) and voice in pocket_voices_offered()
    else:
        assert engine == "kokoro_tts" and face_voice_valid(voice)
    if face is not None:
        pack = PACK_VOICES[face]
        if engine == "kokoro_tts":
            assert voice == pack.kokoro and voice[1] == _kind_letter(pack.hint)
        else:
            assert voice == pack.pocket and voice == face_voice.POCKET_BY_KIND[pack.hint]
    elif found is not None:
        assert voice == default_voice(engine)


@pytest.mark.parametrize("face", sorted(PACK_VOICES))
def test_a_face_never_brings_a_voice_off_the_list_or_of_the_other_kind(face):
    pack = PACK_VOICES[face]
    assert face_voice_valid(pack.kokoro) and pack.kokoro[1] == _kind_letter(pack.hint)
    brought = pocket_voice(face, pack.kokoro, pack.hint)
    assert brought == pack.pocket  # "" while no Pocket voice of that kind is approved
    if brought:
        assert pocket_voice_allowed(brought) and brought == face_voice.POCKET_BY_KIND[pack.hint]
        terms = voice_default.pocket_voice_terms(brought)
        assert terms.get("listened") is True


def test_every_default_on_the_screen_is_on_the_list():
    for entry in TABLE["voices"]:
        if entry["engine"] == "pocket_tts":
            assert pocket_voice_allowed(entry["default_voice"])
        else:
            assert face_voice_valid(entry["default_voice"])
        assert default_voice(entry["engine"]) == entry["default_voice"]
