from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.system.voice import ContinuousVoiceListener, _resolve_input_device


class _FakeSoundDevice:
    def __init__(self, devices, *, default_input=-1, usable=()):
        self._devices = devices
        self._usable = set(usable)
        self.default = SimpleNamespace(device=[default_input, 0])

    def query_devices(self, index=None):
        return self._devices if index is None else self._devices[index]

    def check_input_settings(self, *, device, **_kwargs):
        if device not in self._usable:
            raise RuntimeError("unsupported format")


def test_input_device_uses_valid_default_first():
    sounddevice = _FakeSoundDevice(
        [
            {"name": "Default Mic", "max_input_channels": 2},
            {"name": "Microphone Array", "max_input_channels": 4},
        ],
        default_input=0,
        usable={0, 1},
    )

    assert (
        _resolve_input_device(
            sounddevice,
            sample_rate=16000,
            channels=1,
            dtype="int16",
        )
        == 0
    )


def test_input_device_recovers_from_missing_windows_default():
    sounddevice = _FakeSoundDevice(
        [
            {"name": "Stereo Mix", "max_input_channels": 2},
            {"name": "Microphone Array 1", "max_input_channels": 2},
            {"name": "Microphone Array 3", "max_input_channels": 4},
        ],
        default_input=-1,
        usable={0, 2},
    )

    assert (
        _resolve_input_device(
            sounddevice,
            sample_rate=16000,
            channels=1,
            dtype="int16",
        )
        == 2
    )


def test_input_device_reports_when_no_format_is_usable():
    sounddevice = _FakeSoundDevice(
        [{"name": "Microphone", "max_input_channels": 2}],
        default_input=-1,
        usable=set(),
    )

    with pytest.raises(RuntimeError, match="No usable microphone"):
        _resolve_input_device(
            sounddevice,
            sample_rate=16000,
            channels=1,
            dtype="int16",
        )


@pytest.mark.asyncio
async def test_listener_does_not_report_running_when_microphone_open_fails():
    listener = ContinuousVoiceListener(config=PilotConfig())
    listener._recorder.start = MagicMock(return_value=False)
    listener._recorder.last_error = "device unavailable"

    result = await listener.start()

    assert listener.is_running is False
    assert listener._task is None
    assert "device unavailable" in result


@pytest.mark.asyncio
async def test_server_surfaces_microphone_start_failure(monkeypatch):
    class _FailedListener:
        is_running = False

        def __init__(self, **_kwargs):
            pass

        async def start(self):
            return "Voice listener could not start: no microphone"

    monkeypatch.setattr("pilot.system.voice.ContinuousVoiceListener", _FailedListener)
    server = PilotServer(PilotConfig())

    result = await server._handle_voice_listener_start({}, MagicMock())

    assert result["status"] == "error"
    assert "no microphone" in result["message"]
    assert server._voice_listener is None
