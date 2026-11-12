"""The recording helper with a fake input stream: one sentence at a time, redo, skip, quit and resume, and the
manifest and scenarios it leaves for the bench. Nothing is recorded and no audio package is needed beyond numpy."""

from __future__ import annotations

import json

import pytest

from benchmarks.conversation_bench import record
from benchmarks.conversation_bench.scenarios import load_local

np = pytest.importorskip("numpy")


class FakeCapture:
    def __init__(self) -> None:
        self.takes = 0

    def start(self) -> None:
        self.takes += 1

    def stop(self):
        return np.full(16000, 0.1 * self.takes, dtype=np.float32)


def answers(*replies):
    queue = list(replies)
    return lambda _prompt: queue.pop(0) if queue else "q"


def test_the_sets_are_the_sizes_the_scorecard_needs():
    sentences = record.items("sentences")
    assert len(sentences) == 41 and len({i.text for i in sentences}) == 41
    assert sum(len(i.text.split()) for i in sentences) >= 400
    wake = record.items("wake")
    assert len(wake) == 20 and all(i.text.lower().startswith(("hey heliox", "heliox")) for i in wake)
    assert [i.label for i in record.items("room")] == ["not_addressed"]
    assert {i.label for i in sentences} == {"addressed", "control"}
    with pytest.raises(ValueError):
        record.items("everything")


def test_redo_keeps_the_second_take_and_skip_and_quit_resume_later(tmp_path):
    cap = FakeCapture()
    # first sentence: start, stop, redo; start, stop, keep. Second: start, stop, skip. Third: quit.
    counts = record.record_set(str(tmp_path), "Desk Mic", "sentences", cap,
                               ask=answers("", "", "r", "", "", "", "", "", "s", "q"), say=lambda _t: None)  # fmt: skip
    assert counts == {"kept": 1, "skipped": 1, "already": 0}
    folder = tmp_path / "real" / "desk-mic"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert list(manifest["clips"]) == ["s00"]
    from benchmarks.conversation_bench import wavio

    kept = wavio.read(folder / "sentences" / "s00.wav")
    assert abs(float(kept.mean()) - 0.2) < 0.01  # the redo, not the first take
    again = record.record_set(str(tmp_path), "desk-mic", "sentences", FakeCapture(), ask=answers("q"),
                              say=lambda _t: None)  # fmt: skip
    assert again["already"] == 1


def test_the_bench_reads_the_recordings_as_real_scenarios(tmp_path):
    record.record_set(str(tmp_path), "headset", "wake", FakeCapture(), ask=answers("", "", "", "q"),
                      say=lambda _t: None)  # fmt: skip
    record.record_set(str(tmp_path), "headset", "room", FakeCapture(), ask=answers("", ""), say=lambda _t: None,
                      sleep=lambda _s: None)  # fmt: skip
    scenarios = {s.id: s for s in load_local(tmp_path)}
    wake = scenarios["own-headset-w00"]
    assert wake.voice == "real" and wake.mic == "headset" and wake.expect == {"woken": True}
    assert wake.steps[0].say == "real/headset/wake/w00.wav" and wake.said == record.WAKE[0]
    room = scenarios["own-headset-room"]
    assert room.set == "not_addressed" and room.steps[0].text == ""
    assert "headset: room 1, wake 1" in record.summary(str(tmp_path))


def test_microphone_names_are_folder_safe():
    assert record.safe_mic("Desk USB (pTron)") == "desk-usb-ptron"
    with pytest.raises(ValueError):
        record.safe_mic("///")


class NarrowbandCapture(FakeCapture):
    """Speech-band noise with nothing above 3.4 kHz, as a Bluetooth headset in call mode delivers it."""

    def stop(self):
        from scipy.signal import butter, sosfilt

        rng = np.random.default_rng(0)
        x = sosfilt(butter(10, 3000, "low", fs=16000, output="sos"), rng.standard_normal(3 * 16000))
        return (0.2 * x / np.abs(x).max()).astype(np.float32)


class WidebandCapture(FakeCapture):
    def stop(self):
        rng = np.random.default_rng(0)
        return (0.1 * rng.standard_normal(3 * 16000)).astype(np.float32)


def test_a_call_mode_headset_recorded_as_desk_stops_with_a_plain_sentence(tmp_path):
    pytest.importorskip("scipy")
    said = []
    counts = record.record_set(str(tmp_path), "desk", "sentences", NarrowbandCapture(), ask=answers("", "", ""),
                               say=said.append)  # fmt: skip
    assert counts["stopped"] == 1 and counts["kept"] == 0
    assert "Bluetooth headset in call mode" in said[-1] and "--mic desk" in said[-1]
    assert not (tmp_path / "real" / "desk" / "sentences" / "s00.wav").exists()
    ok = record.record_set(str(tmp_path), "headset", "sentences", NarrowbandCapture(), ask=answers("", "", "", "q"),
                           say=lambda _t: None)  # fmt: skip
    assert ok["kept"] == 1  # named a headset: kept, and the label is right
    wide = record.record_set(str(tmp_path), "desk", "sentences", WidebandCapture(), ask=answers("", "", "", "q"),
                             say=lambda _t: None)  # fmt: skip
    assert wide["kept"] == 1


def test_headset_labels_and_device_names():
    assert record.label_says_headset("BT headset") and not record.label_says_headset("desk")
    assert record.hands_free("Headset (WH-1000XM4 Hands-Free AG Audio)")
    assert not record.hands_free("Microphone (pTron USB)")
    assert record.narrowband(np.zeros(100, dtype=np.float32)) is None


class SpeakingLate(FakeCapture):
    """A reply that starts 0.8 s after the cue."""

    def stop(self):
        x = np.zeros(32000, dtype=np.float32)
        x[12800:] = 0.3 * np.random.default_rng(1).standard_normal(19200)  # wideband, as a desk microphone gives
        return x


class FakePlayer:
    def __init__(self):
        self.played = []
        self.stopped = 0

    def play(self, audio, rate):
        self.played.append((len(audio), rate))

    def stop(self):
        self.stopped += 1


def test_followups_store_the_time_from_the_cue_and_become_window_scenarios(tmp_path):
    said = []
    counts = record.record_set(str(tmp_path), "desk", "followup", SpeakingLate(), ask=answers("", "", "", "q"),
                               say=said.append)  # fmt: skip
    assert counts["kept"] == 1
    assert any("Heliox has finished" in line for line in said)
    manifest = json.loads((tmp_path / "real" / "desk" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["clips"]["f00"]["cue_to_voice_s"] == pytest.approx(0.8, abs=0.03)
    s = {x.id: x for x in load_local(tmp_path)}["own-desk-f00"]
    assert s.setup["heliox_says"] == record.FOLLOWUP[0][0] and s.steps[0].when == "heliox_done"
    assert s.expect == {"follow_up": True} and "0.8 s" in s.note
    kinds = {label for _, _, label, _ in record.FOLLOWUP}
    assert kinds == {"addressed", "not_addressed"} and len(record.FOLLOWUP) == 15


def test_the_echo_set_says_it_plays_sound_and_plays_only_when_asked(tmp_path):
    said, player = [], FakePlayer()
    voice = lambda text: (np.zeros(24000, dtype=np.float32), 24000)  # noqa: E731
    refused = record.record_set(str(tmp_path), "desk", "echo", FakeCapture(), ask=answers("no"), say=said.append,
                                player=player, heliox_voice=voice)  # fmt: skip
    assert refused["kept"] == 0 and player.played == [] and "PLAYS SOUND" in said[0]
    counts = record.record_set(str(tmp_path), "desk", "echo", FakeCapture(), ask=answers("yes", "", "", "", "q"),
                               say=lambda _t: None, player=player, heliox_voice=voice)  # fmt: skip
    assert counts["kept"] == 1 and player.played == [(24000, 24000)] and player.stopped == 1
    s = {x.id: x for x in load_local(tmp_path)}["own-desk-e00-stop"]
    assert s.set == "control" and s.expect == {"stopped": True} and s.steps[0].when == "heliox_audio"
    unavailable = record.record_set(str(tmp_path), "laptop", "echo", FakeCapture(), ask=answers("yes"),
                                    say=said.append)  # fmt: skip
    assert unavailable["kept"] == 0 and "cannot run" in said[-1]
    actions = {i.action for i in record.items("echo")}
    assert actions == {"stop", "cough", "typing", "silent"} and len(record.items("echo")) == 16


def test_echo_noise_must_not_stop_heliox():
    s = record.scenario_for(record.items("echo")[1], "desk", {"playback_offset_s": 0.4, "seconds": 6.0})
    assert s["expect"] == {"stopped": False} and s["lines"] == ["1.2d"]
    assert s["steps"][0]["say"].endswith("#0.40-6.00")
