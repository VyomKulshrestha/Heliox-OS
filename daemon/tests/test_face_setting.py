"""face.pack: which face the conversation view shows, "off" by default. The daemon makes mouth data for the window only
while one is chosen."""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock

import pytest

from pilot.config import PilotConfig, _merge_config, _validate_config_types, face_pack_valid
from pilot.server import PilotServer
from pilot.system import speech_notices


def test_the_face_is_off_until_a_person_picks_one():
    assert PilotConfig().face.pack == "off"


def test_a_saved_pack_id_is_kept_and_a_malformed_one_is_ignored():
    _validate_config_types({"face": {"pack": "placeholder"}})
    assert _merge_config(PilotConfig(), {"face": {"pack": "placeholder"}}).face.pack == "placeholder"
    assert _merge_config(PilotConfig(), {"face": {"pack": "../../etc"}}).face.pack == "off"
    with pytest.raises(ValueError):
        _validate_config_types({"face": {"pack": 3}})
    assert face_pack_valid("off") and face_pack_valid("woman-1") and not face_pack_valid("Woman")


@pytest.mark.asyncio
async def test_the_window_can_choose_a_pack_and_cannot_send_a_path():
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    chosen = await server._handle_update_config({"section": "face", "values": {"pack": "placeholder"}}, MagicMock())
    assert chosen["status"] == "ok"
    assert config.face.pack == "placeholder"
    refused = await server._handle_update_config({"section": "face", "values": {"pack": "a/b"}}, MagicMock())
    assert refused["status"] == "error" and config.face.pack == "placeholder"


@pytest.mark.asyncio
async def test_mouth_data_follows_the_live_setting():
    config = PilotConfig()
    server = PilotServer(config)
    server._install_speech_notices()
    try:
        assert not speech_notices.face_on()
        config.face.pack = "placeholder"
        assert speech_notices.face_on()
        config.face.pack = "off"
        assert not speech_notices.face_on()
    finally:
        task = speech_notices._hub.task
        speech_notices.uninstall()
        if task is not None:
            with contextlib.suppress(BaseException):
                await task
