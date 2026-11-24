"""The face picker's two choices for a face whose voice is not on the computer, the fetch before the switch, the
suggestion's "Not now", and the Pocket voices Heliox may use. Fakes only: nothing is fetched, no model loads, no
network is reached, nothing is saved to disk."""

from __future__ import annotations

import importlib.metadata
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig, _merge_config, _validate_config_types
from pilot.server import PilotServer
from pilot.system import face_voice, kokoro_tts, local_tts_worker, voice_default, voice_fetch
from pilot.system.face_voice import plan_face_voice


def _server(monkeypatch, *, engine="pocket_tts", voice="alba", models=("pocket_tts",), chosen=False, pack="off"):
    monkeypatch.setattr(voice_default, "voice_models_on_disk", lambda cache=None: frozenset(models))
    monkeypatch.setattr(voice_default, "pocket_voice_on_disk", lambda _voice, cache=None: True)
    monkeypatch.setattr(kokoro_tts, "voice_on_disk", lambda _voice: "kokoro_tts" in models)
    configured: list[str] = []
    monkeypatch.setattr(local_tts_worker, "configure_local_voice", lambda v, **_: configured.append(v.tts_voice))
    config = PilotConfig()
    config.voice.language = "en"
    config.voice.tts_engine, config.voice.tts_voice, config.voice.tts_voice_chosen = engine, voice, chosen
    config.face.pack = pack
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._voice_download_check = lambda engine: (True, "")
    server._reply_boundary_poll = 0
    return server, config, configured


MAN = {"voice": "am_michael", "hint": "masculine", "pack": "portrait-man"}


def _notes(server, method):
    return [call.args[1] for call in server._broadcast_notification.await_args_list if call.args[0] == method]


# ── Before the face is chosen ──


@pytest.mark.asyncio
async def test_a_face_whose_voice_is_not_here_says_so_and_its_size_before_it_is_chosen(monkeypatch):
    server, config, _configured = _server(monkeypatch)
    ahead = await server._handle_face_voice({**MAN, "apply": ""}, None)
    assert ahead["action"] == "fetch" and ahead["engine"] == "kokoro_tts" and ahead["download_ok"] is True
    assert 330_000_000 <= ahead["download_bytes"] <= 331_000_000
    assert config.face.pack == "off" and config.save.call_count == 0  # asking changes nothing


@pytest.mark.asyncio
async def test_without_a_network_the_download_cannot_start_and_says_why(monkeypatch):
    server, config, _configured = _server(monkeypatch)
    server._voice_download_check = lambda engine: (False, "No connection to huggingface.co right now.")
    ahead = await server._handle_face_voice({**MAN, "apply": ""}, None)
    assert ahead["download_ok"] is False and ahead["download_reason"] == "No connection to huggingface.co right now."
    fetched: list[object] = []
    server._voice_fetcher = fetched.append
    refused = await server._handle_face_voice({**MAN, "apply": "download"}, None)
    assert refused["status"] == "error" and "huggingface.co" in refused["message"] and fetched == []
    # The other choice still works without a network.
    config.face.pack = "portrait-man"
    kept = await server._handle_face_voice({**MAN, "apply": "keep"}, None)
    assert kept["status"] == "ok" and kept["action"] == "kept"


# ── "Use this face with the voice I have" ──


@pytest.mark.asyncio
async def test_keeping_the_voice_is_recorded_and_nothing_asks_or_swaps_again(monkeypatch):
    server, config, configured = _server(monkeypatch, pack="portrait-man")
    kept = await server._handle_face_voice({**MAN, "apply": "keep"}, None)
    assert kept["action"] == "kept" and kept["say"] == "You chose to keep your voice, alba, with this face."
    assert config.face.kept_voice == ["portrait-man"] and config.voice.tts_voice_chosen is True
    assert config.save.call_count == 1 and not configured
    # The face set again by any route: no offer, no switch, even once Kokoro is on the computer.
    monkeypatch.setattr(
        voice_default, "voice_models_on_disk", lambda cache=None: frozenset({"kokoro_tts", "pocket_tts"})
    )
    brought = server._face_brings_its_voice(configure=False)
    assert brought["switched"] is False and brought["action"] == "kept" and config.voice.tts_voice == "alba"
    again = await server._handle_face_voice({**MAN, "apply": "keep"}, None)
    assert config.face.kept_voice == ["portrait-man"] and again["action"] == "kept"


def test_the_kept_faces_are_read_back_from_the_file_and_checked():
    loaded = _merge_config(PilotConfig(), {"face": {"pack": "portrait-man", "kept_voice": ["portrait-man", "../x", 3]}})
    assert loaded.face.kept_voice == ["portrait-man"]
    _validate_config_types({"face": {"kept_voice": ["portrait-man"]}})
    with pytest.raises(ValueError):
        _validate_config_types({"face": {"kept_voice": "portrait-man"}})


# ── "Use this face and download its voice" ──


@pytest.mark.asyncio
async def test_the_download_runs_while_the_current_voice_speaks_and_the_switch_waits_for_a_reply_to_end(monkeypatch):
    server, config, configured = _server(monkeypatch, pack="portrait-man")
    fetched: list[list[voice_fetch.HubFile]] = []
    server._voice_fetcher = fetched.append
    boundary = iter([False, False, True])  # a reply is being spoken twice when asked, then it has ended
    seen: list[tuple[str, str]] = []

    def between() -> bool:
        seen.append((config.voice.tts_engine, config.voice.tts_voice))  # nothing switched while a reply runs
        return next(boundary)

    server._between_replies = between
    started = await server._handle_face_voice({**MAN, "apply": "download"}, None)
    assert started["status"] == "ok" and started["action"] == "downloading"
    assert "alba keeps speaking until it is ready" in started["say"]
    assert config.voice.tts_engine == "pocket_tts"  # the current voice speaks meanwhile
    await server._face_voice_fetch_task
    paths = [item.path for item in fetched[0]]
    assert paths == ["config.json", "kokoro-v1_0.pth", "voices/am_michael.pt"]
    assert seen == [("pocket_tts", "alba")] * 3
    assert (config.voice.tts_engine, config.voice.tts_voice, config.voice.tts_voice_chosen) == (
        "kokoro_tts",
        "am_michael",
        True,
    )
    assert configured == ["am_michael"]
    ready = _notes(server, "face_voice_ready")
    assert ready == [{"status": "ok", "pack": "portrait-man", "voice": "am_michael",
                      "say": "Heliox now speaks with am_michael, this face's voice."}]  # fmt: skip


@pytest.mark.asyncio
async def test_a_failed_download_switches_nothing_and_the_picker_is_told(monkeypatch):
    server, config, configured = _server(monkeypatch, pack="portrait-man")

    def fail(_files):
        raise OSError("connection reset")

    server._voice_fetcher = fail
    server._between_replies = lambda: True
    await server._handle_face_voice({**MAN, "apply": "download"}, None)
    await server._face_voice_fetch_task
    assert (config.voice.tts_engine, config.voice.tts_voice) == ("pocket_tts", "alba") and not configured
    assert _notes(server, "face_voice_ready")[0]["status"] == "failed"
    after = await server._handle_face_voice({**MAN, "apply": ""}, None)
    assert after["action"] == "fetch" and after["say"].startswith("The last download stopped: OSError")


@pytest.mark.asyncio
async def test_a_face_changed_during_the_download_is_not_switched_to(monkeypatch):
    server, config, _configured = _server(monkeypatch, pack="portrait-man")
    server._voice_fetcher = lambda _files: None

    def between() -> bool:
        config.face.pack = "portrait-woman"  # the person picked another face meanwhile
        return True

    server._between_replies = between
    await server._handle_face_voice({**MAN, "apply": "download"}, None)
    await server._face_voice_fetch_task
    assert config.voice.tts_engine == "pocket_tts" and _notes(server, "face_voice_ready") == []


@pytest.mark.asyncio
async def test_downloading_a_voice_already_here_uses_it_at_once(monkeypatch):
    server, config, _configured = _server(
        monkeypatch, models=("pocket_tts", "kokoro_tts"), chosen=True, pack="portrait-man"
    )
    used = await server._handle_face_voice({**MAN, "apply": "download"}, None)
    assert (
        used["switched"] is True and config.voice.tts_voice == "am_michael" and config.voice.tts_engine == "kokoro_tts"
    )


def test_the_manager_says_when_no_reply_is_being_spoken():
    now = [100.0]
    manager = local_tts_worker.LocalTTSWorkerManager(clock=lambda: now[0])
    manager.note_spoken()
    assert manager.between_replies() is False  # Heliox has just spoken
    now[0] += local_tts_worker.REPLY_GAP_SECONDS + 0.1
    assert manager.between_replies() is True
    manager._busy = 1
    assert manager.between_replies() is False  # a piece is being made or played


# ── What is fetched, and whether it can be ──


def test_the_files_fetched_come_from_the_manifest():
    kokoro = voice_fetch.files_for("kokoro_tts", "am_michael", model_on_disk=True)
    assert [(f.repo, f.path) for f in kokoro] == [("hexgrad/Kokoro-82M", "voices/am_michael.pt")]
    pocket = voice_fetch.files_for("pocket_tts", "alba", model_on_disk=False)
    assert [f.path for f in pocket] == ["languages/english_2026-10/embeddings/alba.safetensors"]
    assert voice_fetch.download_bytes("pocket_tts", model_on_disk=True) == 6_244_152
    asked: list[dict] = []
    voice_fetch.fetch(kokoro, download=lambda **given: asked.append(given))
    assert asked == [{"repo_id": "hexgrad/Kokoro-82M", "filename": "voices/am_michael.pt", "revision": None}]


def test_whether_a_download_can_start():
    assert voice_fetch.network_ok({"HF_HUB_OFFLINE": "1"}, reach=lambda: None) == (
        False,
        "Heliox is set to work offline.",
    )

    def unreachable() -> None:
        raise OSError("no route")

    assert voice_fetch.network_ok({}, reach=unreachable) == (False, "No connection to huggingface.co right now.")
    assert voice_fetch.network_ok({}, reach=lambda: None) == (True, "")


# ── "Not now" on the suggestion ──


def test_not_now_hides_the_suggestion_for_good(monkeypatch):
    server, config, _configured = _server(monkeypatch, engine="kokoro_tts", voice="af_heart")

    def must_not_probe(_device="auto"):
        raise AssertionError("the card was read for a suggestion that was turned off")

    monkeypatch.setattr(voice_default, "machine_now", must_not_probe)
    config.voice.engine_suggestion_dismissed = True
    assert server._voice_suggestion_now() is None
    _validate_config_types({"voice": {"engine_suggestion_dismissed": True}})
    assert _merge_config(
        PilotConfig(), {"voice": {"engine_suggestion_dismissed": True}}
    ).voice.engine_suggestion_dismissed


@pytest.mark.asyncio
async def test_not_now_is_a_plain_setting_and_chooses_no_voice(monkeypatch):
    server, config, _configured = _server(monkeypatch, engine="kokoro_tts", voice="af_heart")
    result = await server._handle_update_config(
        {"section": "voice", "values": {"engine_suggestion_dismissed": True}}, MagicMock()
    )
    assert result["status"] == "ok" and config.voice.engine_suggestion_dismissed is True
    assert config.voice.tts_voice_chosen is False


# ── Pocket voices: only those whose terms allow any use ──


def test_the_terms_table_reads_as_the_published_terms():
    terms = voice_default.VOICES["pocket_tts"]["voice_terms"]
    allowed = {name for name, term in terms.items() if term["use"] == "allowed"}
    assert terms["cosette"]["use"] == terms["jean"]["use"] == "never"  # CC BY-NC 4.0
    assert {name for name, term in terms.items() if term["use"] == "unknown"} == {
        "giovanni", "lola", "juergen", "rafael", "daan"}  # fmt: skip
    assert {"alba", "michael", "george", "javert", "bill_boerst", "estelle"} <= allowed
    assert all(terms[name]["licence"] in {"CC0-1.0", "CC-BY-4.0"} for name in allowed)
    assert all(terms[name]["attribution"] for name in allowed if terms[name]["licence"] == "CC-BY-4.0")
    assert voice_default.pocket_voices_offered() == ["alba"]  # the only one listened to and read back


def test_every_built_in_voice_of_the_installed_pocket_tts_is_in_the_table():
    """A new upstream voice is a decision, never offered by default: the table must name it first."""
    try:
        source = importlib.metadata.distribution("pocket-tts").locate_file("pocket_tts/utils/utils.py")
        text = open(source, encoding="utf-8").read()  # noqa: PTH123 - a path from the package's own metadata
    except (importlib.metadata.PackageNotFoundError, OSError):
        pytest.skip("pocket-tts is not installed here")
    block = text[text.index("_ORIGINS_OF_PREDEFINED_VOICES") :].split("}", 1)[0]
    names = set(re.findall(r'^\s*"([a-z_]+)"\s*:', block, re.MULTILINE))
    assert names and names <= set(voice_default.VOICES["pocket_tts"]["voice_terms"])


def test_a_voice_not_on_the_list_is_never_picked_by_the_rule_or_a_face(monkeypatch):
    nc_face = {"kokoro_tts": "af_heart", "pocket_tts": "cosette"}
    assert voice_default.first_voice(engine="kokoro_tts", voice="af_heart", chosen=False, machine="processor",
                                     language="en", models_on_disk=frozenset(), ever_loaded=False,
                                     face_voices=nc_face) is None  # fmt: skip
    monkeypatch.setitem(face_voice.POCKET_BY_KIND, "feminine", "lola")
    assert face_voice.pocket_voice("someone-elses-pack", "bf_emma", "feminine") == ""
    plan = plan_face_voice("bf_emma", "feminine", engine="pocket_tts", current="alba", chosen=False, language="en",
                           on_disk=lambda _voice: True, pocket="")  # fmt: skip
    assert plan.engine == "kokoro_tts"  # no allowed Pocket voice: the face's Kokoro voice comes instead


@pytest.mark.asyncio
async def test_the_settings_call_refuses_a_voice_not_on_the_list_but_keeps_one_already_chosen(monkeypatch):
    server, config, _configured = _server(monkeypatch, engine="kokoro_tts", voice="af_heart")
    update = server._handle_update_config
    for voice in ("cosette", "jean", "no_such_voice"):  # not even a name Heliox knows
        refused = await update(
            {"section": "voice", "values": {"tts_engine": "pocket_tts", "tts_voice": voice}}, MagicMock()
        )
        assert refused["status"] == "error" and config.voice.tts_engine == "kokoro_tts"
    for voice in ("giovanni", "lola"):  # known names whose terms are not established
        refused = await update(
            {"section": "voice", "values": {"tts_engine": "pocket_tts", "tts_voice": voice}}, MagicMock()
        )
        assert refused["status"] == "error" and voice in refused["message"] and "not established" in refused["message"]
        assert config.voice.tts_engine == "kokoro_tts"
    allowed = await update(
        {"section": "voice", "values": {"tts_engine": "pocket_tts", "tts_voice": "alba"}}, MagicMock()
    )
    assert allowed["status"] == "ok" and config.voice.tts_voice == "alba"
    # Someone already speaking with lola (saved before this list) keeps it working.
    server, config, _configured = _server(monkeypatch, engine="pocket_tts", voice="lola", chosen=True)
    update = server._handle_update_config
    same = await update({"section": "voice", "values": {"tts_engine": "pocket_tts", "tts_voice": "lola"}}, MagicMock())
    other = await update({"section": "voice", "values": {"tts_keep_loaded_seconds": 600}}, MagicMock())
    assert same["status"] == "ok" and other["status"] == "ok" and config.voice.tts_voice == "lola"
    moved = await update({"section": "voice", "values": {"tts_voice": "giovanni"}}, MagicMock())
    assert moved["status"] == "error" and config.voice.tts_voice == "lola"
