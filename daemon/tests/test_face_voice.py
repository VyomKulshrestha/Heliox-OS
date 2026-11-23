"""A face brings its voice (pilot.system.face_voice, the face_voice RPC), with fakes only: no voice file is read from
or fetched into a real cache, no model loads, nothing is saved to disk."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig, _merge_config, _validate_config_types
from pilot.server import PilotServer
from pilot.system import kokoro_tts, local_tts_worker
from pilot.system.face_voice import PACK_VOICES, pack_voice, plan_face_voice

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


# The daemon applies the rule wherever face.pack changes: no client has to ask (face_voice with apply "auto").

FACES = Path(__file__).resolve().parents[2] / "tauri-app" / "ui" / "public" / "faces"


def test_the_daemon_knows_the_voice_every_shipped_pack_names():
    listed = json.loads((FACES / "packs.json").read_text(encoding="utf-8"))["packs"]
    named = {}
    for pack in listed:
        manifest = json.loads((FACES / pack / "manifest.json").read_text(encoding="utf-8"))
        voice = manifest.get("voice") or {}
        if voice.get("kokoro"):
            named[pack] = (voice["kokoro"], voice.get("hint", ""))
    assert named == PACK_VOICES
    assert pack_voice("off") is None and pack_voice("placeholder") is None


async def _set_face(server, pack):
    return await server._handle_update_config({"section": "face", "values": {"pack": pack}}, MagicMock())


@pytest.mark.asyncio
async def test_setting_the_face_by_the_settings_update_brings_its_voice(monkeypatch):
    server, config, configured = _server(monkeypatch)
    result = await _set_face(server, "portrait-man")
    assert result["status"] == "ok" and result["face_voice"]["switched"] is True
    assert config.face.pack == "portrait-man" and config.voice.tts_voice == "am_michael"
    assert config.voice.tts_voice_chosen is False and configured == ["am_michael"]
    # The Face row asks only what fits ("") and is told what happened, whoever set the face.
    told = await server._handle_face_voice({"voice": "am_michael", "hint": "masculine", "apply": ""}, None)
    assert told["switched"] is True and told["say"].startswith("Heliox now speaks with am_michael, this face's voice.")
    # The woman's face next: the voice was never the person's, so it follows the face again.
    back = await _set_face(server, "portrait-woman")
    assert back["face_voice"]["switched"] is True and config.voice.tts_voice == "af_heart"
    # Off changes no voice; a face without a voice of its own neither.
    for pack in ("off", "placeholder"):
        same = await _set_face(server, pack)
        assert "face_voice" not in same and config.voice.tts_voice == "af_heart"


@pytest.mark.asyncio
async def test_a_chosen_voice_is_kept_whatever_route_sets_the_face(monkeypatch):
    server, config, configured = _server(monkeypatch, chosen=True, voice="bf_emma")
    result = await _set_face(server, "portrait-man")
    assert result["face_voice"] == {**result["face_voice"], "switched": False, "action": "offer", "voice": "am_michael"}
    assert result["face_voice"]["say"].startswith("Your voice, bf_emma, is kept.")
    assert config.voice.tts_voice == "bf_emma" and not configured and config.save.call_count == 1  # the face only
    told = await server._handle_face_voice({"voice": "am_michael", "hint": "masculine", "apply": ""}, None)
    assert told["switched"] is False and told["action"] == "offer"


@pytest.mark.asyncio
async def test_a_switch_said_by_the_face_ends_when_the_person_picks_a_voice(monkeypatch):
    server, config, _configured = _server(monkeypatch)
    await _set_face(server, "portrait-man")
    await server._handle_update_config({"section": "voice", "values": {"tts_voice": "bm_lewis"}}, MagicMock())
    told = await server._handle_face_voice({"voice": "am_michael", "hint": "masculine", "apply": ""}, None)
    assert told["switched"] is False and told["action"] == "offer" and "Your voice, bm_lewis, is kept." in told["say"]
    again = await _set_face(server, "portrait-woman")
    assert again["face_voice"]["switched"] is False and config.voice.tts_voice == "bm_lewis"


@pytest.mark.asyncio
async def test_the_face_s_voice_is_not_switched_when_it_cannot_speak_or_cannot_be_saved(monkeypatch):
    server, config, _configured = _server(monkeypatch, on_disk=NOT_HERE)
    missing = await _set_face(server, "portrait-man")
    assert missing["face_voice"]["switched"] is False and missing["face_voice"]["action"] == "fetch"
    assert config.voice.tts_voice == "af_heart"
    server, config, configured = _server(monkeypatch, engine="pocket_tts", voice="alba")
    other = await _set_face(server, "portrait-man")
    assert other["face_voice"]["action"] == "unavailable" and config.voice.tts_voice == "alba"
    server, config, configured = _server(monkeypatch)
    saves = iter([None, OSError("disk full")])

    def save():
        failure = next(saves)
        if failure:
            raise failure

    config.save.side_effect = save
    kept = await _set_face(server, "portrait-man")
    assert kept["status"] == "ok" and kept["face_voice"]["switched"] is False
    assert config.face.pack == "portrait-man" and config.voice.tts_voice == "af_heart" and not configured


@pytest.mark.asyncio
async def test_a_face_set_by_a_spoken_setting_brings_its_voice(monkeypatch):
    server, config, _configured = _server(monkeypatch)
    from pilot import server as server_module
    from pilot.personal.spoken_settings import SpokenSetting

    # No phrase in the spoken-settings table changes the face today; any that does goes through the same update.
    monkeypatch.setattr(
        server_module,
        "spoken_setting",
        lambda _said, _config: SpokenSetting("face", {"pack": "portrait-man"}, "Okay, the man's face."),
    )
    assert await server._settings_request("show the man's face") == "Okay, the man's face."
    assert config.face.pack == "portrait-man" and config.voice.tts_voice == "am_michael"


def test_a_face_in_the_config_file_brings_its_voice_at_start(monkeypatch):
    server, config, configured = _server(monkeypatch)
    edited = _merge_config(PilotConfig(), {"face": {"pack": "portrait-man"}, "voice": {"tts_voice_chosen": False}})
    config.face, config.voice.tts_voice_chosen = edited.face, edited.voice.tts_voice_chosen
    brought = server._face_brings_its_voice(configure=False)
    assert brought["switched"] is True and config.voice.tts_voice == "am_michael" and not configured
    # start() checks it before the voice is set up, so the first reply is in the face's voice.
    source = inspect.getsource(PilotServer.start)
    assert 0 < source.index("_face_brings_its_voice(configure=False)") < source.index("self._set_up_voice()")
    # A file whose voice was chosen keeps it.
    server, config, _configured = _server(monkeypatch, chosen=True, voice="bf_emma")
    config.face.pack = "portrait-man"
    assert server._face_brings_its_voice(configure=False)["switched"] is False and config.voice.tts_voice == "bf_emma"


def test_a_face_chosen_by_editing_the_file_while_heliox_runs_brings_its_voice(monkeypatch):
    server, config, configured = _server(monkeypatch)
    edits = iter([True, True, False])

    def reload_if_edited():
        edited = next(edits)
        if edited:
            config.face.pack = "portrait-man"
        return edited

    config.reload_if_edited = reload_if_edited
    assert server._adopt_edited_config() is True
    assert config.voice.tts_voice == "am_michael" and configured == ["am_michael"]
    config.voice.tts_voice = "af_heart"  # the same face reloaded again: the face did not change, nothing is applied
    assert server._adopt_edited_config() is True and config.voice.tts_voice == "af_heart"
    assert server._adopt_edited_config() is False
