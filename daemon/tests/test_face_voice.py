"""A face brings its voice (pilot.system.face_voice, the face_voice RPC), with fakes only: no voice file is read from
or fetched into a real cache, no model loads, nothing is saved to disk."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig, _merge_config, _validate_config_types
from pilot.server import PilotServer
from pilot.system import kokoro_tts, local_tts_worker
from pilot.system.face_voice import plan_face_voice

HERE = lambda _voice: True  # noqa: E731
NOT_HERE = lambda _voice: False  # noqa: E731


def plan(voice="am_michael", hint="masculine", **given):
    settings = {"engine": "kokoro_tts", "current": "af_heart", "chosen": False, "language": "en", "on_disk": HERE}
    settings.update(given)
    return plan_face_voice(voice, hint, **settings)


def test_a_face_without_a_known_voice_offers_nothing():
    assert plan(voice="").action == "none" and plan(voice="").say == ""
    assert plan(voice="not_a_voice").action == "none"
    assert plan(voice="alba").action == "none"  # a Pocket voice is not a Kokoro voice a pack can name


def test_the_face_s_own_voice_already_speaking_needs_nothing():
    same = plan(current="am_michael")
    assert same.action == "same" and same.say == ""


def test_a_voice_on_disk_is_offered_and_a_chosen_voice_is_named_as_kept():
    never = plan()
    assert never.action == "offer" and never.voice == "am_michael" and never.download_bytes == 0
    assert "Your voice" not in never.say
    kept = plan(chosen=True, current="bf_emma")
    assert kept.action == "offer" and kept.say.startswith("Your voice, bf_emma, is kept.")


def test_a_voice_not_on_disk_is_never_fetched_without_the_person():
    missing = plan(on_disk=NOT_HERE)
    assert missing.action == "fetch" and missing.download_bytes > 0
    assert "not on this computer yet" in missing.say and "half a megabyte" in missing.say


@pytest.mark.parametrize(("engine", "named"), [("pocket_tts", "Pocket TTS"), ("os_native", "the system voice")])
def test_another_engine_keeps_its_voice_and_says_how_to_hear_the_face_s(engine, named):
    other = plan(engine=engine, current="alba")
    assert other.action == "unavailable" and named in other.say and "choose Kokoro TTS under Voice" in other.say


def test_another_language_keeps_its_voice_or_offers_one_of_its_own_by_the_hint():
    spanish_man = plan(language="es", current="ef_dora")
    assert spanish_man.action == "unavailable" and "speaks English" in spanish_man.say and "Spanish" in spanish_man.say
    spanish_woman = plan(voice="af_heart", hint="feminine", language="es", current="af_sky", on_disk=NOT_HERE)
    assert spanish_woman.action == "fetch" and spanish_woman.voice == "ef_dora"
    assert "feminine Spanish voice" in spanish_woman.say
    already = plan(voice="af_heart", hint="feminine", language="es", current="ef_dora")
    assert already.action == "unavailable"  # the voice of the hint is the one speaking
    no_reader = plan(
        voice="af_heart", hint="feminine", language="zh", current="af_sky", reader_ok=lambda _language: False
    )
    assert no_reader.action == "unavailable"


def test_never_chosen_is_known_and_an_old_file_keeps_its_voice():
    assert PilotConfig().voice.tts_voice_chosen is False
    _validate_config_types({"voice": {"tts_voice_chosen": True}})
    with pytest.raises(ValueError):
        _validate_config_types({"voice": {"tts_voice_chosen": "yes"}})
    # a file from before the field: every save wrote tts_voice, so the default may have been chosen on purpose
    assert _merge_config(PilotConfig(), {"voice": {"tts_voice": "af_heart"}}).voice.tts_voice_chosen is True
    assert _merge_config(PilotConfig(), {"voice": {"language": "en"}}).voice.tts_voice_chosen is False
    saved = {"voice": {"tts_voice": "am_michael", "tts_voice_chosen": False}}
    assert _merge_config(PilotConfig(), saved).voice.tts_voice_chosen is False


def _server(monkeypatch, *, on_disk=HERE, chosen=False, engine="kokoro_tts", voice="af_heart"):
    config = PilotConfig()
    config.voice.language = "en"
    config.voice.tts_engine, config.voice.tts_voice, config.voice.tts_voice_chosen = engine, voice, chosen
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    monkeypatch.setattr(kokoro_tts, "voice_on_disk", on_disk)
    configured: list[str] = []
    monkeypatch.setattr(local_tts_worker, "configure_local_voice", lambda v, **_: configured.append(v.tts_voice))
    return server, config, configured


@pytest.mark.asyncio
async def test_choosing_the_face_switches_a_voice_never_chosen_and_says_so(monkeypatch):
    server, config, configured = _server(monkeypatch)
    result = await server._handle_face_voice({"voice": "am_michael", "hint": "masculine", "apply": "auto"}, None)
    assert result["switched"] is True and result["voice"] == "am_michael" and result["action"] == "same"
    assert result["say"].startswith("Heliox now speaks with am_michael, this face's voice.")
    assert config.voice.tts_voice == "am_michael" and config.voice.tts_voice_chosen is False
    assert configured == ["am_michael"] and config.save.call_count == 1
    server._broadcast_notification.assert_awaited_once_with(
        "config_updated", {"method": "face_voice", "section": "voice"}
    )
    # the woman's face next: the voice was the man's face's, never the person's, so it follows the face again
    back = await server._handle_face_voice({"voice": "af_heart", "hint": "feminine", "apply": "auto"}, None)
    assert back["switched"] is True and config.voice.tts_voice == "af_heart"


@pytest.mark.asyncio
async def test_a_chosen_voice_is_kept_and_the_face_s_voice_offered(monkeypatch):
    server, config, configured = _server(monkeypatch, chosen=True, voice="bf_emma")
    result = await server._handle_face_voice({"voice": "am_michael", "apply": "auto"}, None)
    assert result["switched"] is False and result["action"] == "offer"
    assert config.voice.tts_voice == "bf_emma" and not configured and config.save.call_count == 0
    accepted = await server._handle_face_voice({"voice": "am_michael", "apply": "accept"}, None)
    assert accepted["switched"] is True and accepted["say"] == "Heliox now speaks with am_michael."
    assert config.voice.tts_voice == "am_michael" and config.voice.tts_voice_chosen is True


@pytest.mark.asyncio
async def test_a_voice_not_on_disk_is_not_switched_by_choosing_the_face(monkeypatch):
    server, config, _configured = _server(monkeypatch, on_disk=NOT_HERE)
    result = await server._handle_face_voice({"voice": "am_michael", "apply": "auto"}, None)
    assert result["switched"] is False and result["action"] == "fetch" and config.voice.tts_voice == "af_heart"
    accepted = await server._handle_face_voice({"voice": "am_michael", "apply": "accept"}, None)
    assert accepted["switched"] is True and config.voice.tts_voice == "am_michael"
    assert "half a megabyte is fetched the first time it speaks" in accepted["say"]


@pytest.mark.asyncio
async def test_another_engine_is_never_switched(monkeypatch):
    server, config, _configured = _server(monkeypatch, engine="os_native")
    for apply in ("auto", "accept"):
        result = await server._handle_face_voice({"voice": "am_michael", "apply": apply}, None)
        assert result["switched"] is False and result["action"] == "unavailable"
    assert config.voice.tts_engine == "os_native" and config.save.call_count == 0


@pytest.mark.asyncio
async def test_bad_requests_and_a_failed_save_change_nothing(monkeypatch):
    server, config, _configured = _server(monkeypatch)
    assert (await server._handle_face_voice({"voice": 3}, None))["status"] == "error"
    assert (await server._handle_face_voice({"voice": "am_michael", "apply": "now"}, None))["status"] == "error"
    config.save.side_effect = OSError("disk full")
    failed = await server._handle_face_voice({"voice": "am_michael", "apply": "auto"}, None)
    assert failed["status"] == "error" and config.voice.tts_voice == "af_heart"
    assert config.voice.tts_voice_chosen is False


@pytest.mark.asyncio
async def test_a_voice_set_by_hand_sticks(monkeypatch):
    server, config, _configured = _server(monkeypatch)
    picked = await server._handle_update_config({"section": "voice", "values": {"tts_voice": "bm_lewis"}}, MagicMock())
    assert picked["status"] == "ok" and config.voice.tts_voice_chosen is True
    result = await server._handle_face_voice({"voice": "am_michael", "apply": "auto"}, None)
    assert result["switched"] is False and config.voice.tts_voice == "bm_lewis"
    config.save.side_effect = OSError("disk full")
    config.voice.tts_voice_chosen = False
    refused = await server._handle_update_config({"section": "voice", "values": {"tts_voice": "af_sky"}}, MagicMock())
    assert refused["status"] == "error" and config.voice.tts_voice_chosen is False


@pytest.mark.asyncio
async def test_accepting_the_voice_for_the_language_counts_as_a_choice(monkeypatch):
    server, config, _configured = _server(monkeypatch)
    config.voice.language = "es"
    server._system_voices = []
    await server._handle_voice_for_language_apply({}, None)
    assert config.voice.tts_voice == "ef_dora" and config.voice.tts_voice_chosen is True
