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
