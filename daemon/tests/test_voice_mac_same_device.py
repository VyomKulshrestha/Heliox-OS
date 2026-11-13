"""macOS, with the microphone on the default output's Core Audio device: the output line opens before the microphone
and stays open, and every later open of the line (another rate, a new default output) reopens the microphone at once.
Windows, Linux and a Mac with separate devices run today's order. Stand-in PortAudio streams; the checks count events
in order, never the clock."""

from __future__ import annotations

import sys
import threading
import time
import types

import numpy as np
import pytest

from pilot.system import voice
from pilot.system.platform_detect import Platform

IN_AND_OUT = 3  # one Core Audio device, inputs and outputs together (BlackHole 2ch, a USB headset)
SPEAKERS = 1


class FakeSD:
    """Stand-in PortAudio. An output stream's callback runs once when it starts (the device asked for audio). An
    input stream keeps delivering blocks from its own thread until it is stopped, unless ``input_plan`` says the next
    attempt "fails" (raises, like PaMacCore err -50), is "silent" (opens and delivers nothing) or "bursts" (one block,
    then nothing: what run 43419901063 saw)."""

    def __init__(self, default_in: int, default_out: int) -> None:
        self.events: list[str] = []
        self.default = types.SimpleNamespace(device=(default_in, default_out))
        self.input_plan: list[str] = []
        self.output_latencies: list[str] = []  # the latency each output stream was opened with, in order
        # PortAudio's device list as of its last read; _initialize() (a refresh) adds ``plugged_in``.
        self.devices = [
            {"index": 0, "name": "MacBook Pro Microphone", "max_input_channels": 1, "max_output_channels": 0},
            {"index": SPEAKERS, "name": "MacBook Pro Speakers", "max_input_channels": 0, "max_output_channels": 2},
            {"index": 2, "name": "Apple Virtual Sound Device", "max_input_channels": 0, "max_output_channels": 2},
            {"index": IN_AND_OUT, "name": "BlackHole 2ch", "max_input_channels": 2, "max_output_channels": 2},
        ]
        self.plugged_in: list[dict] = []
        self.refreshes = 0
        # "hangs": a start that blocks on a stalled Core Audio unit until ``release`` is set; its close then waits for
        # ``close_gate`` when one is given.
        self.release = threading.Event()
        self.close_gate: threading.Event | None = None
        sd = self

        class Stream:
            kind = ""

            def __init__(self, **kwargs) -> None:
                self.device = kwargs.get("device")
                self.samplerate = kwargs.get("samplerate")
                self.callback = kwargs.get("callback")
                self.active = False
                self.latency_setting = kwargs.get("latency")
                self.latency = 0.01 if self.latency_setting == "low" else 0.1  # seconds, as PortAudio reports it

            def start(self) -> None:
                where = self.device if self.device is not None else sd.default.device[1]
                plan = sd.input_plan.pop(0) if self.kind == "input" and sd.input_plan else "works"
                if plan == "fails":
                    sd.events.append(f"failed {self.kind} {where}")
                    raise RuntimeError("PaMacCore (AUHAL) err='-50'")
                if plan.startswith("hangs"):
                    self.hung = True
                    sd.events.append(f"hung {self.kind} {where}")
                    sd.release.wait(10)
                    self.active = True
                    if plan == "hangs-then-delivers":
                        for _ in range(5):
                            self.callback(np.zeros((800, 1), dtype=np.int16) + 100, 800, None, None)
                    return
                self.active = True
                sd.events.append(f"open {self.kind} {where} {self.samplerate}")
                if self.kind == "output":
                    sd.output_latencies.append(self.latency_setting)
                    self.callback(np.zeros((240, 1), dtype=np.float32), 240, None, None)
                elif plan == "bursts":
                    self.callback(np.zeros((800, 1), dtype=np.int16) + 100, 800, None, None)
                elif plan == "works":

                    def deliver() -> None:
                        while self.active:
                            self.callback(np.zeros((800, 1), dtype=np.int16) + 100, 800, None, None)
                            time.sleep(0.005)

                    threading.Thread(target=deliver, daemon=True).start()

            def stop(self) -> None:
                self.active = False

            def abort(self) -> None:
                self.active = False

            def close(self) -> None:
                if getattr(self, "hung", False) and sd.close_gate is not None:
                    sd.close_gate.wait(10)
                where = self.device if self.device is not None else sd.default.device[1]
                sd.events.append(f"close {self.kind} {where}")

        class InputStream(Stream):
            kind = "input"

        class OutputStream(Stream):
            kind = "output"

        self.InputStream = InputStream
        self.OutputStream = OutputStream

    def query_devices(self, index=None, kind=None):
        if index is None and kind is None:
            return list(self.devices)  # PortAudio's list as of its last read
        if kind == "output":
            return {"index": self.default.device[1], "name": "out"}
        if kind == "input":
            return {"index": self.default.device[0], "name": "in"}
        return {"index": index, "name": f"device {index}", "hostapi": 0, "max_input_channels": 2}

    def query_hostapis(self, index=None):
        return {"name": "Core Audio"}

    def _terminate(self):
        pass

    def _initialize(self):
        self.refreshes += 1
        self.devices = self.devices + self.plugged_in
        self.plugged_in = []


def setup(
    monkeypatch, platform: Platform, default_in: int, default_out: int
) -> tuple[FakeSD, voice._ContinuousRecorder]:
    sd = FakeSD(default_in, default_out)
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.setattr(voice, "CURRENT_PLATFORM", platform)
    monkeypatch.setattr(voice, "_output_line", voice._OutputLine())
    monkeypatch.setattr(voice, "_resolve_input_device", lambda sd_, **kwargs: sd_.default.device[0])
    monkeypatch.setattr("pilot.system.audio_health.default_speaker_name", lambda: "")
    monkeypatch.setattr(voice, "MAC_SAME_DEVICE_ORDER", "output_first")  # variant A; the variant B test sets its own
    monkeypatch.setattr(voice, "MIC_RETRY_PAUSES", (0.0, 0.0, 0.0))
    monkeypatch.setattr(voice, "MIC_FRAMES_WITHIN_SECONDS", 0.3)  # the open runs in its own thread now
    monkeypatch.setattr(voice, "MIC_CLOSE_WAIT_SECONDS", 5.0)  # stand-in closes return at once: keep the order
    monkeypatch.setattr(voice, "MIC_SUSTAIN_SECONDS", 0.1)
    monkeypatch.setattr(voice, "_portaudio_inputs_open", 0)
    recorder = voice._ContinuousRecorder()
    monkeypatch.setattr(recorder, "_start_windows_wasapi", lambda: False)  # the PortAudio path on every system
    return sd, recorder


def wait_for(condition, limit: int = 300) -> None:
    for _ in range(limit):
        if condition():
            return
        time.sleep(0.01)


def finish(recorder: voice._ContinuousRecorder) -> None:
    # The hook's reopen first: closing while it is still under its sustained test would leave its stream behind.
    for thread in threading.enumerate():
        if thread.name == "heliox-input-after-output":
            thread.join(5)
    recorder._close_capture()
    voice._output_line.close()


def test_macos_with_one_device_opens_the_output_line_before_the_microphone_and_keeps_it(monkeypatch):
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    assert recorder._open_capture()
    assert sd.events == [f"open output {IN_AND_OUT} 24000", f"open input {IN_AND_OUT} 16000"]
    line = voice._output_line
    assert line.pinned
    line.last_used -= voice.OUTPUT_LINE_IDLE_SECONDS + 1
    line._close_if_idle()
    assert line.stream is not None  # no idle close while the microphone shares the device
    finish(recorder)


@pytest.mark.parametrize("platform", [Platform.WINDOWS, Platform.LINUX])
def test_windows_and_linux_keep_today_s_order_even_on_one_device_index(monkeypatch, platform):
    sd, recorder = setup(monkeypatch, platform, IN_AND_OUT, IN_AND_OUT)
    assert recorder._open_capture()
    assert sd.events == [f"open input {IN_AND_OUT} 16000"]
    assert not voice._output_line.pinned and voice._output_line.on_opened is None
    voice._output_line.ensure(24000, 1)
    time.sleep(0.05)
    assert sd.events == [f"open input {IN_AND_OUT} 16000", f"open output {IN_AND_OUT} 24000"]  # no reopen
    finish(recorder)


def test_a_mac_with_separate_devices_keeps_today_s_order(monkeypatch):
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, SPEAKERS)
    assert recorder._open_capture()
    assert sd.events == [f"open input {IN_AND_OUT} 16000"]
    assert not voice._output_line.pinned
    voice._output_line.ensure(24000, 1)
    time.sleep(0.05)
    assert sd.events == [f"open input {IN_AND_OUT} 16000", f"open output {SPEAKERS} 24000"]  # the input untouched
    finish(recorder)


def test_on_one_device_a_new_open_of_the_line_reopens_the_microphone_straight_after(monkeypatch):
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    recorder._open_capture()
    sd.events.clear()
    voice._output_line.ensure(48000, 1)  # another rate (Pocket against Kokoro): the line reopens
    wait_for(lambda: sd.events.count(f"open input {IN_AND_OUT} 16000") == 1)
    assert sd.events == [
        f"close output {IN_AND_OUT}",
        f"open output {IN_AND_OUT} 48000",
        f"close input {IN_AND_OUT}",
        f"open input {IN_AND_OUT} 16000",
    ]
    finish(recorder)


def test_a_default_output_moved_off_the_microphone_s_device_unpins_the_line_and_leaves_the_microphone(monkeypatch):
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    recorder._open_capture()
    sd.events.clear()
    sd.default.device = (IN_AND_OUT, SPEAKERS)  # the person picks the speakers as the output
    voice._output_line.close()
    voice._output_line.ensure(24000, 1)
    wait_for(lambda: not voice._output_line.pinned)
    time.sleep(0.05)
    assert f"close input {IN_AND_OUT}" not in sd.events and not voice._output_line.pinned
    finish(recorder)


def test_a_default_output_moved_onto_the_microphone_s_device_reopens_the_microphone(monkeypatch):
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, SPEAKERS)
    recorder._open_capture()
    sd.events.clear()
    sd.default.device = (IN_AND_OUT, IN_AND_OUT)  # a headset plugged in: it becomes input and output
    voice._output_line.close()
    voice._output_line.ensure(24000, 1)
    wait_for(lambda: f"open input {IN_AND_OUT} 16000" in sd.events)
    assert sd.events[-2:] == [f"close input {IN_AND_OUT}", f"open input {IN_AND_OUT} 16000"]
    assert voice._output_line.pinned
    finish(recorder)


def test_stopping_the_listener_unpins_the_line(monkeypatch):
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    recorder._open_capture()
    recorder.stop()
    assert not voice._output_line.pinned and voice._output_line.on_opened is None
    voice._output_line.close()


def test_same_device_is_decided_by_device_index_and_only_on_macos(monkeypatch):
    sd = FakeSD(IN_AND_OUT, IN_AND_OUT)
    monkeypatch.setattr(voice, "CURRENT_PLATFORM", Platform.MACOS)
    assert voice._same_core_audio_device(sd, None) is True
    assert voice._same_core_audio_device(sd, SPEAKERS) is False
    sd.default.device = (-1, -1)  # PortAudio without a default: asks for the default devices
    assert voice._same_core_audio_device(sd, IN_AND_OUT) is False
    monkeypatch.setattr(voice, "CURRENT_PLATFORM", Platform.LINUX)
    assert voice._same_core_audio_device(FakeSD(IN_AND_OUT, IN_AND_OUT), None) is False


def test_no_output_stream_is_opened_outside_the_shared_line():
    """Every in-process sound goes through _OutputLine: no other sd.play or sd.OutputStream in the daemon."""
    from pathlib import Path

    pilot = Path(voice.__file__).resolve().parents[1]
    found = []
    for path in pilot.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in ("sd.play(", "sd.OutputStream(", "sounddevice.play(", "sounddevice.OutputStream("):
            if pattern in text:
                found.append(f"{path.relative_to(pilot).as_posix()}: {pattern}")
    assert found == ["system/voice.py: sd.OutputStream("]  # _OutputLine.ensure, and nothing else


@pytest.mark.asyncio
async def test_playing_through_the_line_cuts_on_cancel(monkeypatch):
    import asyncio

    cut: list[bool] = []

    class Playback:
        def __init__(self, audio, rate) -> None:
            self.started = threading.Event()
            self.stop = threading.Event()

        def play(self) -> None:
            self.stop.wait(2)

        def cut(self) -> None:
            cut.append(True)
            self.stop.set()

    monkeypatch.setattr(voice, "_Playback", Playback)
    task = asyncio.ensure_future(voice.play_on_output_line([0.0], 24000))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cut == [True]


def test_an_input_that_fails_or_stays_silent_beside_the_line_is_retried_until_it_delivers(monkeypatch):
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    sd.input_plan = ["fails", "silent", "works"]  # err -50, then no frames, then frames
    assert recorder._open_capture()
    assert sd.events == [
        f"open output {IN_AND_OUT} 24000",
        f"failed input {IN_AND_OUT}",
        f"close input {IN_AND_OUT}",  # opened but not started: closed, not left behind
        f"open input {IN_AND_OUT} 16000",
        f"close input {IN_AND_OUT}",
        f"open input {IN_AND_OUT} 16000",
    ]
    assert voice._output_line.pinned and voice._output_line.stream is not None
    finish(recorder)


def test_when_every_attempt_fails_at_start_the_listener_falls_back_to_today_s_order(monkeypatch):
    """No worse than before the fix: the line is closed and the microphone opened first, so the listener starts."""
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    sd.input_plan = ["fails", "silent", "fails", "silent"]
    assert recorder._open_capture()
    assert sd.events[-2:] == [f"close output {IN_AND_OUT}", f"open input {IN_AND_OUT} 16000"]
    assert sd.events.count(f"failed input {IN_AND_OUT}") == 2
    assert voice._output_line.stream is None and not voice._output_line.pinned
    finish(recorder)


def test_after_a_later_open_of_the_line_failed_attempts_end_in_a_plain_open_and_the_line_stays(monkeypatch):
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    recorder._open_capture()
    sd.events.clear()
    sd.input_plan = ["fails", "fails", "fails", "fails", "silent"]
    voice._output_line.ensure(22050, 1)  # Heliox is about to speak at another rate
    wait_for(lambda: f"open input {IN_AND_OUT} 16000" in sd.events)
    assert sd.events[:2] == [f"close output {IN_AND_OUT}", f"open output {IN_AND_OUT} 22050"]
    assert sd.events.count(f"failed input {IN_AND_OUT}") == 4
    assert voice._output_line.stream is not None  # never closed under a sentence being spoken
    finish(recorder)


def test_variant_b_keeps_today_s_order_at_start_and_retries_after_a_later_open(monkeypatch):
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    monkeypatch.setattr(voice, "MAC_SAME_DEVICE_ORDER", "input_first")
    assert recorder._open_capture()
    assert sd.events == [f"open input {IN_AND_OUT} 16000"]  # input first, as before
    assert voice._output_line.pinned
    sd.events.clear()
    sd.input_plan = ["fails", "works"]
    voice._output_line.ensure(24000, 1)
    wait_for(lambda: f"open input {IN_AND_OUT} 16000" in sd.events)
    assert sd.events == [
        f"open output {IN_AND_OUT} 24000",
        f"close input {IN_AND_OUT}",
        f"failed input {IN_AND_OUT}",
        f"close input {IN_AND_OUT}",
        f"open input {IN_AND_OUT} 16000",
    ]
    finish(recorder)


def test_a_first_burst_is_not_a_working_microphone_and_the_fallback_closes_the_line_it_opened(monkeypatch):
    """Run 43419901063: frames 0.09 s after the open, then nothing. Every attempt bursting ends in today's order, with
    the output line this recorder opened closed, so the listener starts as before the fix."""
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    sd.input_plan = ["bursts", "bursts", "bursts", "bursts"]
    assert recorder._open_capture()
    assert sd.events.count(f"close input {IN_AND_OUT}") == 4  # each burst was tried and closed
    assert sd.events[-2:] == [f"close output {IN_AND_OUT}", f"open input {IN_AND_OUT} 16000"]
    assert voice._output_line.stream is None and not voice._output_line.pinned
    finish(recorder)


def test_a_burst_then_a_sustained_attempt_is_kept(monkeypatch):
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    sd.input_plan = ["bursts", "works"]
    assert recorder._open_capture()
    assert sd.events == [
        f"open output {IN_AND_OUT} 24000",
        f"open input {IN_AND_OUT} 16000",
        f"close input {IN_AND_OUT}",
        f"open input {IN_AND_OUT} 16000",
    ]
    assert voice._output_line.stream is not None and voice._output_line.pinned
    finish(recorder)


def test_a_listener_that_gives_up_leaves_no_output_line_it_opened(monkeypatch):
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    assert recorder._open_capture()
    assert voice._output_line.stream is not None
    recorder.stop()  # what the listener's start does when the microphone delivers no audio
    assert voice._output_line.stream is None and not voice._output_line.pinned


def test_on_one_device_the_line_opens_at_the_default_latency_and_logs_it_once(monkeypatch, caplog):
    """Round 4d: a "low" output stopped the 16 kHz input beside it; at the default latency the input kept going."""
    caplog.set_level("INFO", logger=voice.logger.name)
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    assert recorder._open_capture()
    voice._output_line.ensure(48000, 1)  # another rate: the line reopens, still at the default latency
    wait_for(lambda: sd.events.count(f"open input {IN_AND_OUT} 16000") == 2)
    assert sd.output_latencies == ["high", "high"]
    logged = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Output line opened at latency")]
    assert logged == ["Output line opened at latency 'high': latency 100 ms at 24000 Hz"]
    finish(recorder)


@pytest.mark.parametrize(
    ("platform", "default_out"),
    [(Platform.WINDOWS, IN_AND_OUT), (Platform.LINUX, IN_AND_OUT), (Platform.MACOS, SPEAKERS)],
)
def test_everywhere_else_the_line_keeps_low_latency(monkeypatch, platform, default_out):
    sd, recorder = setup(monkeypatch, platform, IN_AND_OUT, default_out)
    assert recorder._open_capture()
    voice._output_line.ensure(24000, 1)
    assert sd.output_latencies == ["low"]
    finish(recorder)


def test_a_low_latency_line_already_open_is_reopened_at_the_default_before_the_microphone(monkeypatch):
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    voice._output_line.ensure(24000, 1)  # warmed for speech before the listener started
    assert recorder._open_capture()
    assert sd.events == [
        f"open output {IN_AND_OUT} 24000",
        f"close output {IN_AND_OUT}",
        f"open output {IN_AND_OUT} 24000",
        f"open input {IN_AND_OUT} 16000",
    ]
    assert sd.output_latencies == ["low", "high"]
    finish(recorder)


def test_the_fallback_and_a_stopped_listener_return_the_line_to_low_latency(monkeypatch):
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    sd.input_plan = ["fails", "fails", "fails", "fails"]
    assert recorder._open_capture()  # every attempt failed: microphone first, as before the fix
    assert voice._output_line.wanted_latency == "low"
    voice._output_line.ensure(24000, 1)
    assert sd.output_latencies == ["high", "low"]
    finish(recorder)

    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    assert recorder._open_capture()
    recorder.stop()
    assert voice._output_line.wanted_latency == "low"


def test_a_default_output_moved_off_the_microphone_s_device_opens_at_low_latency_again(monkeypatch):
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    recorder._open_capture()
    sd.default.device = (IN_AND_OUT, SPEAKERS)
    voice._output_line.close()
    voice._output_line.ensure(24000, 1)  # the open itself sees it lands away from the microphone
    wait_for(lambda: not voice._output_line.pinned)
    voice._output_line.ensure(22050, 1)
    assert sd.output_latencies == ["high", "low", "low"]  # the open away from the microphone is already "low"
    finish(recorder)


def sine(rate: int, seconds: float, hertz: float = 220.0) -> np.ndarray:
    t = np.arange(int(rate * seconds), dtype=np.float32) / rate
    return (0.5 * np.sin(2 * np.pi * hertz * t)).astype(np.float32)


def test_on_a_pinned_line_a_sentence_at_another_rate_is_resampled_and_the_line_is_not_reopened(monkeypatch):
    """Run 43426052142: a 22050 Hz sentence reopened the line and the microphone lost 3.2 s."""
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    assert recorder._open_capture()
    sd.events.clear()
    for rate, channels in ((22050, 1), (48000, 2), (24000, 2)):
        audio = np.repeat(sine(rate, 0.5)[:, None], channels, axis=1)
        playback = voice._Playback(audio, rate)
        playback.fit_to_line()
        assert playback.sample_rate == 24000 and playback.audio.shape[1] == 1
        assert abs(len(playback.audio) - 12000) <= 2
        voice._output_line.ensure(playback.sample_rate, playback.audio.shape[1])
    assert sd.events == []  # no output open, so no microphone reopen
    finish(recorder)


def test_a_pinned_line_that_closed_reopens_at_the_local_voice_s_rate(monkeypatch):
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    assert recorder._open_capture()
    voice._output_line.close()  # the default output changed: the line reopens on it
    assert voice._output_line.rate_for(22050) == 24000
    finish(recorder)


@pytest.mark.parametrize(
    ("platform", "default_out"),
    [(Platform.WINDOWS, IN_AND_OUT), (Platform.LINUX, IN_AND_OUT), (Platform.MACOS, SPEAKERS)],
)
def test_everywhere_else_the_line_reopens_for_another_rate_as_before(monkeypatch, platform, default_out):
    sd, recorder = setup(monkeypatch, platform, IN_AND_OUT, default_out)
    assert recorder._open_capture()
    voice._output_line.ensure(24000, 1)
    playback = voice._Playback(sine(22050, 0.2), 22050)
    playback.fit_to_line()
    assert playback.sample_rate == 22050 and len(playback.audio) == len(sine(22050, 0.2))
    voice._output_line.ensure(playback.sample_rate, 1)
    assert sd.events[-1] == f"open output {default_out} 22050"
    finish(recorder)


def test_the_resampler_keeps_the_ends_without_a_click():
    for rate in (22050, 16000, 48000):
        audio = sine(rate, 0.3)[:, None]
        out = voice.resample_for_line(audio, rate, 24000)[:, 0]
        assert abs(len(out) - 7200) <= 2
        assert abs(float(out[0])) < 0.02  # the sine starts at zero, and so does the resampled one
        steps = np.abs(np.diff(out))
        assert float(steps.max()) < 2 * np.pi * 220 * 0.5 / 24000 * 1.2  # no jump larger than the sine's own slope
        flat = voice.resample_for_line(np.full((rate // 10, 1), 0.4, dtype=np.float32), rate, 24000)[:, 0]
        assert float(np.abs(flat - 0.4).max()) < 0.01  # a held level is not pulled to zero at either end


def test_the_hook_s_reopen_and_the_watchdog_s_make_one_sequence_of_opens_and_closes(monkeypatch):
    """Run 43426052142: the watchdog read the hook's own close as "stream closed" and ran a second retry loop."""
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    monkeypatch.setattr(voice, "MIC_SUSTAIN_SECONDS", 0.3)
    assert recorder._open_capture()
    sd.events.clear()
    generation = recorder.reopen_generation
    hook = threading.Thread(target=recorder._input_after_output)
    hook.start()
    wait_for(lambda: recorder.reopening)
    assert recorder.reopened_since(generation)  # the watchdog's state from now is not read
    results: list[bool] = []
    watchdog = threading.Thread(target=lambda: results.append(recorder.reopen()))
    watchdog.start()
    hook.join(5)
    watchdog.join(5)
    assert sd.events == [f"close input {IN_AND_OUT}", f"open input {IN_AND_OUT} 16000"]
    assert results == [True] and recorder.is_active
    assert recorder.reopen_generation == generation + 2 and not recorder.reopening
    finish(recorder)


def test_a_line_open_during_the_watchdog_s_reopen_leaves_that_reopen_to_finish(monkeypatch):
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    assert recorder._open_capture()
    sd.events.clear()
    with recorder._reopen_lock:  # the watchdog's reopen holds it
        recorder._input_after_output()
    assert sd.events == []
    finish(recorder)


def test_a_listener_stopped_during_a_reopen_leaves_no_input_open(monkeypatch):
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    monkeypatch.setattr(voice, "MIC_SUSTAIN_SECONDS", 0.3)
    assert recorder._open_capture()
    sd.events.clear()
    hook = threading.Thread(target=recorder._input_after_output)
    hook.start()
    wait_for(lambda: f"open input {IN_AND_OUT} 16000" in sd.events)  # the attempt is under its sustained test
    recorder.stop()
    hook.join(5)
    # the stream before the reopen, and the one the reopen opened after stop()
    assert sd.events.count(f"close input {IN_AND_OUT}") == sd.events.count(f"open input {IN_AND_OUT} 16000") + 1
    assert not recorder.is_active
    voice._output_line.close()


APPLE_VIRTUAL = 2


def speaker(monkeypatch, name: str) -> None:
    """The system default output, as soundcard reads it live."""
    monkeypatch.setattr("pilot.system.audio_health.default_speaker_name", lambda: name)


def test_a_default_output_moved_away_and_back_on_a_same_device_mac(monkeypatch):
    """Run 43439172983: "Speaking on Apple Virtual Sound Device now", then the line reopened on BlackHole, because
    PortAudio's default is only re-read with no microphone open. PortAudio's default stays BlackHole here throughout."""
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    speaker(monkeypatch, "BlackHole 2ch")
    assert recorder._open_capture()
    line = voice._output_line
    assert line.pinned and line.device_index == IN_AND_OUT
    sd.events.clear()
    sd.output_latencies.clear()

    speaker(monkeypatch, "Apple Virtual Sound Device")
    assert line.check_device("Apple Virtual Sound Device")
    time.sleep(0.05)  # room for the hook's thread, had it reopened the microphone
    assert sd.events == [f"close output {IN_AND_OUT}", f"open output {APPLE_VIRTUAL} 24000"]  # the microphone kept
    assert sd.output_latencies == ["low"]
    assert not line.pinned and line.wanted_latency == "low"
    assert line.notice == "Speaking on Apple Virtual Sound Device now."
    sd.events.clear()

    speaker(monkeypatch, "BlackHole 2ch")
    assert line.check_device("BlackHole 2ch")
    wait_for(lambda: f"open input {IN_AND_OUT} 16000" in sd.events)
    assert sd.events == [
        f"close output {APPLE_VIRTUAL}",
        f"open output {IN_AND_OUT} 24000",
        f"close input {IN_AND_OUT}",
        f"open input {IN_AND_OUT} 16000",
    ]  # output first at the default latency, then the sustained reopen
    assert sd.output_latencies[-1] == "high" and line.pinned
    assert line.notice == "Speaking on BlackHole 2ch now."
    finish(recorder)


def test_a_new_default_missing_from_the_device_list_while_listening_keeps_the_line_and_says_so(monkeypatch):
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    speaker(monkeypatch, "BlackHole 2ch")
    assert recorder._open_capture()
    sd.events.clear()
    speaker(monkeypatch, "USB Headset")  # plugged in after PortAudio read its list
    sd.plugged_in = [{"index": 4, "name": "USB Headset", "max_input_channels": 1, "max_output_channels": 2}]
    line = voice._output_line
    assert line.check_device("USB Headset")
    assert line.notice == (
        "USB Headset is not in the device list Heliox can open while it listens; still speaking on BlackHole 2ch."
    )
    assert not line.check_device("USB Headset")  # said once
    assert sd.events == [] and sd.refreshes == 0  # nothing closed, the microphone untouched
    assert line.device_name == "BlackHole 2ch" and line.pinned
    finish(recorder)


def test_with_no_microphone_open_a_new_device_is_found_after_a_refresh(monkeypatch):
    sd, _recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    speaker(monkeypatch, "MacBook Pro Speakers")
    line = voice._output_line
    line.ensure(24000, 1)
    speaker(monkeypatch, "USB Headset")
    sd.plugged_in = [{"index": 4, "name": "USB Headset", "max_input_channels": 1, "max_output_channels": 2}]
    assert line.check_device("USB Headset")
    assert sd.refreshes == 1 and sd.events[-1] == "open output 4 24000"
    assert line.notice == "Speaking on USB Headset now."
    line.close()


def test_a_default_missing_at_the_first_open_is_not_named(monkeypatch):
    sd, _recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, SPEAKERS)
    speaker(monkeypatch, "Gone Headphones")
    line = voice._output_line
    line.ensure(24000, 1)
    assert sd.events == [f"open output {SPEAKERS} 24000"]
    assert line.device_name == f"device {SPEAKERS}"  # PortAudio's own default, named as PortAudio names it
    line.close()


@pytest.mark.parametrize("platform", [Platform.WINDOWS, Platform.LINUX])
def test_windows_and_linux_open_the_host_default_which_follows_the_system(monkeypatch, platform):
    """No device index is passed: MME's Sound Mapper and ALSA's default (PulseAudio, PipeWire) follow the system
    default themselves, and their device names do not match soundcard's, so there is no lookup by name."""
    sd, recorder = setup(monkeypatch, platform, IN_AND_OUT, IN_AND_OUT)
    speaker(monkeypatch, "Speakers (Realtek(R) Audio)")
    assert recorder._open_capture()
    line = voice._output_line
    line.ensure(24000, 1)
    sd.events.clear()
    speaker(monkeypatch, "Headphones (Rockerz)")
    assert line.check_device("Headphones (Rockerz)")
    assert sd.events == [f"close output {IN_AND_OUT}", f"open output {IN_AND_OUT} 24000"]  # device=None: the default
    assert line.device_index is None and line.notice == "Speaking on Headphones (Rockerz) now."
    finish(recorder)


def test_an_open_that_hangs_is_given_up_at_the_first_frame_time_and_closed_when_it_returns(monkeypatch, caplog):
    """Run 43442841532: "attempt 1 ... no audio" came 3.1 s after the open, because the start itself hung."""
    caplog.set_level("INFO", logger=voice.logger.name)
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    sd.input_plan = ["hangs", "works"]
    assert recorder._open_capture()  # attempt 2 kept while attempt 1 is still inside its start
    assert sd.events == [
        f"open output {IN_AND_OUT} 24000",
        f"hung input {IN_AND_OUT}",
        f"open input {IN_AND_OUT} 16000",
    ]
    assert voice._portaudio_inputs_open == 1
    said = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Microphone attempt 1 beside")]
    assert len(said) == 1 and said[0].startswith(
        "Microphone attempt 1 beside the output line: the open did not return after"
    )
    sd.release.set()
    wait_for(lambda: f"close input {IN_AND_OUT}" in sd.events)
    wait_for(lambda: voice._portaudio_inputs_open == 1)
    assert sd.events[-1] == f"close input {IN_AND_OUT}" and voice._portaudio_inputs_open == 1
    finish(recorder)


def test_an_attempt_given_up_stays_counted_until_its_close_returns(monkeypatch):
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    monkeypatch.setattr(voice, "MIC_CLOSE_WAIT_SECONDS", 0.0)  # the decision does not wait for this close
    sd.input_plan = ["hangs", "works"]
    sd.close_gate = threading.Event()
    assert recorder._open_capture()
    sd.release.set()  # its start returns; the close of the stalled unit blocks
    wait_for(lambda: voice._portaudio_inputs_open == 2)
    assert voice._portaudio_inputs_open == 2  # the kept microphone and the one still closing
    sd.close_gate.set()
    wait_for(lambda: voice._portaudio_inputs_open == 1)
    assert voice._portaudio_inputs_open == 1 and sd.events[-1] == f"close input {IN_AND_OUT}"
    finish(recorder)


def test_frames_from_an_attempt_given_up_never_reach_the_recorder(monkeypatch):
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    sd.input_plan = ["hangs-then-delivers", "silent", "silent", "silent", "silent"]
    assert recorder._open_capture()  # every attempt failed: the fallback's open is silent too
    frames = recorder.frames_received
    sd.release.set()  # the first attempt's start returns and its callback delivers five blocks
    wait_for(lambda: f"close input {IN_AND_OUT}" in sd.events[-1:])
    assert recorder.frames_received == frames
    finish(recorder)


def test_a_failed_start_closes_the_stream_it_opened(monkeypatch):
    sd, recorder = setup(monkeypatch, Platform.MACOS, IN_AND_OUT, IN_AND_OUT)
    sd.input_plan = ["fails", "works"]
    assert recorder._open_capture()
    assert sd.events[1:3] == [f"failed input {IN_AND_OUT}", f"close input {IN_AND_OUT}"]
    finish(recorder)
