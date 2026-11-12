"""Quiet in calls on macOS and Linux: which recording processes count, with fakes everywhere, plus a run of the
real reader on the system the tests run on (it must answer without raising, whatever it finds)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from pilot.system import call_detection
from pilot.system.call_detection import (
    MicUser,
    mac_call_from_busy_input,
    mac_mic_users,
    parse_pactl_monitor_sources,
    parse_pactl_source_outputs,
)

PACTL_OUTPUTS = """Source Output #41
\tDriver: protocol-native.c
\tOwner Module: 9
\tClient: 88
\tSource: 2
\tCorked: no
\tProperties:
\t\tapplication.name = "ZOOM VoiceEngine"
\t\tapplication.process.id = "4242"
\t\tapplication.process.binary = "zoom"

Source Output #42
\tSource: 1
\tCorked: no
\tProperties:
\t\tapplication.name = "OBS"
\t\tapplication.process.id = "5151"

Source Output #43
\tSource: 2
\tCorked: no
\tProperties:
\t\tapplication.name = "python3"
\t\tapplication.process.id = "1000"

Source Output #44
\tSource: 2
\tCorked: yes
\tProperties:
\t\tapplication.name = "Firefox"
\t\tapplication.process.id = "6262"
"""
PACTL_SOURCES = "1\talsa_output.pci-0000_00_1f.3.analog-stereo.monitor\tPipeWire\ts32le 2ch 48000Hz\tIDLE\n" \
    "2\talsa_input.pci-0000_00_1f.3.analog-stereo\tPipeWire\ts32le 2ch 48000Hz\tRUNNING\n"  # fmt: skip


def test_linux_recording_streams_name_the_app_and_skip_monitors():
    users = parse_pactl_source_outputs(PACTL_OUTPUTS, parse_pactl_monitor_sources(PACTL_SOURCES))
    assert users == [
        MicUser("ZOOM VoiceEngine", True, 4242),
        MicUser("python3", True, 1000),
        MicUser("Firefox", False, 6262),  # corked: a page that holds the microphone but is not using it
    ]  # OBS recording the speakers' monitor is not a microphone user


def test_linux_heliox_itself_is_never_a_call(monkeypatch):
    def run(args):
        text = PACTL_SOURCES if "short" in args else PACTL_OUTPUTS
        return SimpleNamespace(returncode=0, stdout=text)

    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/pactl")
    users = call_detection._mic_users_linux({1000}, run=run)
    assert call_detection.decide(users, [], []) == "ZOOM VoiceEngine"
    only_heliox = [user for user in users if user.pid in {1000, 6262}]
    assert call_detection.decide(only_heliox, [], []) == ""


def test_linux_without_pactl_cannot_tell(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert call_detection._mic_users_linux({1}) is None


def test_mac_recording_processes_other_than_heliox():
    names = {7: "zoom.us", 8: "Python"}.get
    assert mac_mic_users([7, 8], {8}, names) == [MicUser("zoom.us", True, 7)]
    assert mac_mic_users(None, {8}, names) is None  # macOS before 14: cannot tell this way


def test_older_macos_counts_a_busy_input_only_with_a_call_app_open_and_heliox_not_listening():
    assert mac_call_from_busy_input(True, False, ["Finder", "zoom.us"]) == "Zoom"
    assert mac_call_from_busy_input(True, True, ["zoom.us"]) == ""  # it may be Heliox itself
    assert mac_call_from_busy_input(True, False, ["Finder"]) == ""  # a voice memo is not a call
    assert mac_call_from_busy_input(None, False, ["zoom.us"]) == ""


def test_current_call_on_macos_uses_coreaudio_then_falls_back(monkeypatch):
    monkeypatch.setattr(call_detection.sys, "platform", "darwin")
    monkeypatch.setattr(call_detection, "_processes", lambda: ["zoom.us"])
    monkeypatch.setattr(call_detection, "_own_pids", lambda: {os.getpid()})
    monkeypatch.setattr(call_detection, "_process_name", lambda pid: "FaceTime")
    monkeypatch.setattr(call_detection, "_recording_pids_macos", lambda: [os.getpid(), 77])
    assert call_detection.current_call() == "FaceTime"
    monkeypatch.setattr(call_detection, "_recording_pids_macos", lambda: [os.getpid()])
    assert call_detection.current_call() == ""  # only Heliox records
    monkeypatch.setattr(call_detection, "_recording_pids_macos", lambda: None)  # macOS 13
    monkeypatch.setattr(call_detection, "_default_input_busy_macos", lambda: True)
    assert call_detection.current_call(heliox_listening=False) == "Zoom"
    assert call_detection.current_call(heliox_listening=True) == ""


def test_current_call_on_linux(monkeypatch):
    monkeypatch.setattr(call_detection.sys, "platform", "linux")
    monkeypatch.setattr(call_detection, "_processes", lambda: [])
    monkeypatch.setattr(call_detection, "_mic_users_linux", lambda _own: [MicUser("Discord", True, 3)])
    assert call_detection.current_call() == "Discord"


def test_a_reader_that_raises_is_left_out(monkeypatch):
    def broken():
        raise OSError("CoreAudio is not there")

    monkeypatch.setattr(call_detection.sys, "platform", "darwin")
    monkeypatch.setattr(call_detection, "_recording_pids_macos", broken)
    monkeypatch.setattr(call_detection, "_default_input_busy_macos", broken)
    monkeypatch.setattr(call_detection, "_processes", lambda: [])
    assert call_detection.current_call() == ""


@pytest.mark.skipif(sys.platform not in {"darwin", "linux"}, reason="the real macOS or Linux reader")
def test_the_real_reader_on_this_system_answers():
    assert isinstance(call_detection.current_call(), str)  # CoreAudio or pactl, or "cannot tell": never raises


@pytest.mark.skipif(not os.environ.get("HELIOX_REAL_PULSE"), reason="needs a running PulseAudio/PipeWire; opt-in")
def test_real_pulseaudio_sees_another_app_recording():
    """CI (platform readers, Linux): a null source stands in for a microphone and parecord for a call app."""
    import time

    recorder = subprocess.Popen(["parecord", "--device=heliox_mic", "--raw", "/dev/null"])
    try:
        for _ in range(50):
            users = call_detection._mic_users_linux(call_detection._own_pids())
            if users:
                break
            time.sleep(0.1)
        assert users is not None, "pactl did not answer"
        assert any(user.pid == recorder.pid for user in users), users
        assert call_detection.current_call() != ""
    finally:
        recorder.terminate()
