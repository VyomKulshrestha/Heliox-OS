"""Heliox's voice in the person's language, with fakes only: no voice is downloaded, no model loads, no audio plays.
The real voices are fetched and heard on a CI runner (tests/platform_real/kokoro_languages_check.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from pilot.config import PilotConfig
from pilot.system import kokoro_tts, voice
from pilot.system.voice_by_language import (
    espeak_voices,
    kokoro_lang_code,
    kokoro_language,
    macos_voices,
    plan_voice,
    windows_voices,
)

ALL = lambda _module: True  # noqa: E731
NONE = lambda _module: False  # noqa: E731


def test_kokoro_reads_each_voice_with_its_own_language():
    assert [kokoro_lang_code(v) for v in ("af_heart", "bm_lewis", "ef_dora", "hf_alpha", "zf_xiaobei")] == list("abehz")
    assert kokoro_lang_code("alba") == "a" and kokoro_lang_code("") == "a"  # not a Kokoro voice: English as before
    assert kokoro_language("ef_dora") == "es" and kokoro_language("giovanni") == ""


def test_the_synthesiser_takes_the_reader_of_the_voice_s_language(monkeypatch):
    asked = []

    def pipeline(language="a"):
        asked.append(language)

        def run(text, voice, speed, split_pattern):
            return iter([("g", "p", [0.0] * 240)])

        return run

    monkeypatch.setattr(kokoro_tts, "_get_pipeline", pipeline)
    monkeypatch.setattr(kokoro_tts, "_allow_fetching_missing_voice", lambda voice: None)
    kokoro_tts._generate("你明天早上九点开会。", "zf_xiaobei")
    kokoro_tts._generate("Okay.", "af_heart")
    assert asked == ["z", "a"]


@pytest.mark.parametrize(
    ("language", "engine", "voice", "installed", "system", "expected"),
    [
        ("es", "kokoro_tts", "af_heart", ALL, [], ("kokoro_tts", "ef_dora", True)),
        ("hi", "kokoro_tts", "af_heart", ALL, [], ("kokoro_tts", "hf_alpha", True)),
        ("zh", "kokoro_tts", "af_heart", ALL, [], ("kokoro_tts", "zf_xiaobei", True)),
        # Mandarin without the packages its reader needs: a Mandarin system voice, else nothing changes
        ("zh", "kokoro_tts", "af_heart", NONE, [("Microsoft Huihui", "zh")], ("os_native", "Microsoft Huihui", True)),
        ("zh", "kokoro_tts", "af_heart", NONE, [], ("kokoro_tts", "af_heart", False)),
        # German: Kokoro has no voice; a German system voice, else nothing changes
        ("de", "kokoro_tts", "af_heart", ALL, [("Microsoft Hedda", "de"), ("Microsoft Zira", "en")],
         ("os_native", "Microsoft Hedda", True)),
        ("de", "kokoro_tts", "af_heart", ALL, [("Microsoft Zira", "en")], ("kokoro_tts", "af_heart", False)),
        ("es", "kokoro_tts", "ef_dora", ALL, [], ("kokoro_tts", "ef_dora", False)),  # already speaks it
        ("en", "kokoro_tts", "af_heart", ALL, [], ("kokoro_tts", "af_heart", False)),
    ],
)  # fmt: skip
def test_the_voice_offered_for_each_language(language, engine, voice, installed, system, expected):
    plan = plan_voice(language, engine, voice, system_voices=system, installed=installed)
    assert (plan.engine, plan.voice, plan.change) == expected
    if plan.change and plan.engine == "kokoro_tts":
        assert 0 < plan.download_bytes < 1_000_000 and "first time" in plan.say


def test_a_german_speaker_is_told_plainly_what_they_hear_and_where_to_add_a_voice():
    plan = plan_voice("de", "kokoro_tts", "af_heart", platform="windows")
    assert not plan.change
    assert "reads German replies with English sounds" in plan.say and "Time & language, Speech" in plan.say
    mac = plan_voice("de", "kokoro_tts", "af_heart", platform="macos")
    assert "Spoken Content" in mac.say
    zh = plan_voice("zh", "kokoro_tts", "af_heart", installed=NONE, platform="linux")
    assert "jieba" in zh.say and "espeak-ng" in zh.say


def test_the_system_voices_are_read_with_their_languages():
    assert windows_voices("Microsoft David Desktop|en-US\r\nMicrosoft Helena Desktop|es-ES\n") == [
        ("Microsoft David Desktop", "en"),
        ("Microsoft Helena Desktop", "es"),
    ]
    assert macos_voices("Anna                de_DE    # Hallo! Ich heiße Anna.\nTingting  zh_CN  # 你好\n") == [
        ("Anna", "de"),
        ("Tingting", "zh"),
    ]
    table = "Pty Language       Age/Gender VoiceName          File                 Other Languages\n 5  de              --/M      German             gmw/de\n 5  cmn             --/M      Chinese_(Mandarin) sit/cmn\n"
    assert espeak_voices(table) == [("de", "de"), ("cmn", "cmn")]


@pytest.mark.asyncio
async def test_the_system_voice_chosen_for_the_language_speaks(monkeypatch):
    config = PilotConfig()
    config.voice.tts_engine = "os_native"
    config.voice.system_voice = "Microsoft Hedda Desktop"
    monkeypatch.setattr(voice.PilotConfig, "load", classmethod(lambda cls, *a, **k: config))
    spoken = AsyncMock(return_value="Spoken")
    for name in ("_tts_windows", "_tts_macos", "_tts_linux"):
        monkeypatch.setattr(voice, name, spoken)
    await voice._speak_impl("Morgen um neun.")
    assert spoken.await_args.args[1] == "Microsoft Hedda Desktop"
    await voice._speak_impl("Morgen um neun.", voice="Anna")  # a caller's own choice still wins
    assert spoken.await_args.args[1] == "Anna"


@pytest.mark.asyncio
async def test_the_offer_is_listed_and_accepting_it_fetches_nothing_itself(monkeypatch):
    from pilot.server import PilotServer
    from pilot.system import local_tts_worker

    config = PilotConfig()
    config.voice.language = "es"
    config.save = lambda *a, **k: None
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._system_voices = []
    configured = []
    monkeypatch.setattr(local_tts_worker, "configure_local_voice", lambda voice_config: configured.append(1))
    listed = await server._handle_speech_models_list({}, None)
    assert listed["voice"]["voice"] == "ef_dora" and listed["voice"]["change"] is True
    result = await server._handle_voice_for_language_apply({}, None)
    assert config.voice.tts_engine == "kokoro_tts" and config.voice.tts_voice == "ef_dora"
    assert result["change"] is False and configured == [1]  # Kokoro fetches the voice when it first speaks


def test_japanese_is_offered_only_with_its_dictionary_on_disk():
    # run 43341873370: pyopenjtalk, fugashi and unidic installed, the dictionary not fetched: the reader raised
    without = plan_voice("ja", "kokoro_tts", "af_heart", installed=ALL, dictionary=lambda: False)
    assert not without.change and without.voice == "af_heart"
    assert "Japanese dictionary" in without.say and "cannot read Japanese replies aloud" in without.say
    with_it = plan_voice("ja", "kokoro_tts", "af_heart", installed=ALL, dictionary=lambda: True)
    assert with_it.change and with_it.voice == "jf_alpha"


def test_a_chosen_voice_whose_reader_cannot_start_is_not_called_working():
    plan = plan_voice("ja", "kokoro_tts", "jf_alpha", installed=ALL, dictionary=lambda: False)
    assert not plan.change and "cannot start on this computer" in plan.say and "system voice" in plan.say
    working = plan_voice("ja", "kokoro_tts", "jf_alpha", installed=ALL, dictionary=lambda: True)
    assert working.say == "Heliox speaks Japanese with the voice jf_alpha."


def test_a_script_the_english_voice_cannot_read_is_said_so():
    plan = plan_voice("zh", "kokoro_tts", "af_heart", installed=NONE)
    assert "cannot read Mandarin replies aloud" in plan.say
    assert "with English sounds" in plan_voice("de", "kokoro_tts", "af_heart").say
