"""Heliox's audio on a real PulseAudio server, for the Linux ``platform-readers`` job (PulseAudio already running).

    python audio_check.py playback   # a tone through Heliox's output line reaches the default sink
    python audio_check.py follow     # the default microphone changes during capture; Heliox follows it
    python audio_check.py taken      # the chosen microphone is taken, then given back, during capture

playback: a null sink becomes the default output; ``voice._Playback`` plays one second of a 440 Hz tone on the
shared output line (PortAudio, through ALSA's ``pulse`` device), while ``parec`` records the sink's monitor. The
recording must hold the tone (its level well above silence).

follow: two null sources with different names; capture starts on the first through ``voice._ContinuousRecorder``
with the setting on "auto". The default is switched to the second while capturing; ``audio_health.CaptureWatchdog``
must decide "reopen" for "default_changed" from what it reads live, and after ``recorder.reopen()`` PulseAudio must
show Heliox's recording stream on the second source. This is the body of the listener's capture watch
(``ContinuousVoiceListener._watch_capture``), run by the check against real devices.

taken: the person chose a microphone (not "auto"). While Heliox captures from it, another program takes it, the
way a call app can: first the source is suspended (it stays listed but delivers nothing), then it is removed (the
device is gone), and each time it is given back. The body of the listener's capture watch runs once a second against
the real devices, as in ``ContinuousVoiceListener._watch_capture``: it must see each loss within a few seconds, say
so in the sentence the window shows (``microphone_problem``), keep or regain a working stream, and once the
microphone is back be on it again and say so (``microphone_recovered``: "Listening on ..."). Every decision and its
sentence is printed. What it cannot show: Windows' exclusive mode and macOS, which the runners have no microphone
for; and that the window shows the sentence (the session tests do).

Each check prints one ``CHECK`` line; the exit code is the number of failed checks.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path

from ci_only import require_runner

require_runner("adds PulseAudio sinks and sources and changes the default speaker and microphone")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

failures = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


def pactl(*args: str) -> str:
    done = subprocess.run(["pactl", *args], capture_output=True, text=True, timeout=10, check=False)
    if done.returncode != 0:
        print(f"pactl {' '.join(args)}: exit {done.returncode}: {done.stderr.strip()}")
    return done.stdout.strip()


def load_null_source(name: str, description: str) -> None:
    """A null source with its own description. module-null-source takes ``description=`` (unlike the null sink's
    ``sink_properties=``): CI on e51aead loaded nothing with ``source_properties=``, the default stayed the earlier
    step's "Null Input", and every follow check failed for that reason."""
    if pactl("load-module", "module-null-source", f"source_name={name}", f"description={description}"):
        return
    pactl(
        "load-module",
        "module-null-source",
        f"source_name={name}",
        f"source_properties=device.description={description}",
    )


def playback() -> None:
    import array

    import numpy as np

    from pilot.system import voice

    pactl(
        "load-module",
        "module-null-sink",
        "sink_name=heliox_speaker",
        "sink_properties=device.description=HelioxSpeaker",
    )
    pactl("set-default-sink", "heliox_speaker")
    raw = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "heliox-monitor.raw"
    with raw.open("wb") as out:
        recorder = subprocess.Popen(
            ["parec", "-d", "heliox_speaker.monitor", "--format=s16le", "--rate=24000", "--channels=1"], stdout=out
        )
        time.sleep(0.5)
        rate = 24000
        tone = (0.4 * np.sin(2 * math.pi * 440 * np.arange(rate) / rate)).astype(np.float32)
        try:
            voice._Playback(tone, rate).play()
            played = "played"
        except Exception as exc:
            played = f"{type(exc).__name__}: {exc}"
        time.sleep(0.8)
        recorder.terminate()
        recorder.wait(timeout=5)
    samples = array.array("h", raw.read_bytes()[: (raw.stat().st_size // 2) * 2])
    peak = max((abs(s) for s in samples), default=0)
    rms = math.sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0.0
    check(
        "audio-linux/playback-reaches-the-default-sink",
        peak > 3000 and rms > 500,
        f"{played}; {len(samples) / 24000:.1f} s recorded from the monitor, peak {peak}, rms {rms:.0f}; "
        f"output line on {voice._output_line.device_name!r}",
    )
    voice._output_line.close()


def source_index(name: str) -> str:
    for line in pactl("list", "short", "sources").splitlines():
        parts = line.split("\t")
        if len(parts) > 1 and parts[1] == name:
            return parts[0]
    return ""


def follow() -> None:
    from pilot.system import audio_health, voice

    for name, description in (("heliox_follow_a", "HelioxMicA"), ("heliox_follow_b", "HelioxMicB")):
        load_null_source(name, description)
    print(f"sources: {pactl('list', 'short', 'sources').splitlines()}")
    pactl("set-default-source", "heliox_follow_a")
    time.sleep(0.5)
    recorder = voice._ContinuousRecorder(preferred_device="auto")
    recorder._loop = None
    opened = recorder._open_capture()
    try:
        before = audio_health.default_microphone_name()
        check(
            "audio-linux/capture-opens-on-the-default",
            opened and before == "HelioxMicA",
            f"{before!r} {recorder.last_error}",
        )
        watchdog = audio_health.CaptureWatchdog()
        watchdog.opened(before, recorder.last_error)
        pactl("set-default-source", "heliox_follow_b")
        time.sleep(1.5)
        state = audio_health.CaptureState(
            frames=int(recorder.frames_received),
            error=str(recorder.last_error or ""),
            zero_seconds=float(recorder.zero_seconds),
            active=bool(recorder.is_active),
            default_name=audio_health.default_microphone_name(),
            auto=True,
            using_fallback=bool(recorder.using_fallback_input),
        )
        decision = watchdog.check(state)
        check(
            "audio-linux/watchdog-sees-the-new-default",
            decision.action == "reopen" and decision.reason == "default_changed",
            f"{decision}; default now {state.default_name!r}",
        )
        reopened = recorder.reopen()
        time.sleep(1.0)
        outputs = pactl("list", "source-outputs")
        index = source_index("heliox_follow_b")
        check(
            "audio-linux/capture-follows-to-the-new-default",
            reopened and bool(index) and f"Source: {index}" in outputs,
            f"reopened {reopened}, source index {index!r}, device {recorder.input_device_name!r}",
        )
    finally:
        recorder._close_capture()


class Watch:
    """The body of ``ContinuousVoiceListener._watch_capture``, run by the check: one call per second."""

    def __init__(self, recorder, configured_name: str) -> None:
        from pilot.system import audio_health

        self.health = audio_health
        self.recorder = recorder
        self.configured_name = configured_name
        self.watchdog = audio_health.CaptureWatchdog()
        key, name = audio_health.default_microphone_identity()
        self.watchdog.opened(name, recorder.last_error, key)
        self.said: list[tuple[float, str, str, str]] = []  # (seconds, status, reason, sentence)
        self.started = time.monotonic()

    def say(self, status: str, reason: str, sentence: str) -> None:
        moment = time.monotonic() - self.started
        self.said.append((moment, status, reason, sentence))
        print(f"INFO {moment:5.1f} s {status} {reason or '-'}: {sentence} (on {self.recorder.input_device_name!r})")

    def tick(self) -> None:
        recorder, health = self.recorder, self.health
        fallback = bool(recorder.using_fallback_input)
        key, name = health.default_microphone_identity()
        state = health.CaptureState(
            frames=int(recorder.frames_received),
            error=str(recorder.last_error or ""),
            zero_seconds=float(recorder.zero_seconds),
            active=bool(recorder.is_active),
            default_name=name,
            default_key=key,
            auto=False,
            configured_available=bool(fallback and health.microphone_present(self.configured_name)),
            using_fallback=fallback,
        )
        if self.watchdog.recovered(state):
            self.say("microphone_recovered", "", f"Listening on {recorder.input_device_name}.")
        decision = self.watchdog.check(state)
        if decision.action == "notice":
            self.say("microphone_problem", decision.reason, decision.message)
        elif decision.action == "reopen":
            self.say("microphone_problem", decision.reason, decision.message)
            try:
                opened = recorder.reopen()
            except Exception as error:
                recorder.last_error = str(error)
                opened = False
            if opened:
                key, name = health.default_microphone_identity()
                self.watchdog.reopen_succeeded(name, recorder.last_error, key)
            else:
                self.watchdog.reopen_failed()
                print(f"INFO reopen failed: {recorder.last_error or 'no device'}")

    def run(self, seconds: float, until=None) -> bool:
        """Tick once a second for up to ``seconds``; True as soon as ``until()`` holds."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(1.0)
            self.tick()
            if until is not None and until():
                return True
        return until is None

    def since(self, mark: int, status: str) -> list[tuple[float, str, str, str]]:
        return [entry for entry in self.said[mark:] if entry[1] == status]


def capturing_from(name: str) -> bool:
    """PulseAudio shows a recording stream on the source ``name`` (Heliox's is the only one on the runner)."""
    index = source_index(name)
    return bool(index) and f"Source: {index}" in pactl("list", "source-outputs")


def taken() -> None:
    from pilot.system import voice

    load_null_source("heliox_taken_a", "HelioxChosenMic")
    load_null_source("heliox_taken_b", "HelioxOtherMic")
    pactl("set-default-source", "heliox_taken_b")
    time.sleep(0.5)
    recorder = voice._ContinuousRecorder(preferred_device="HelioxChosenMic")
    recorder._loop = None
    opened = recorder._open_capture()
    try:
        check(
            "audio-linux/taken/opens-the-chosen-microphone",
            opened and capturing_from("heliox_taken_a"),
            f"device {recorder.input_device_name!r}, error {recorder.last_error!r}",
        )
        watch = Watch(recorder, "HelioxChosenMic")
        watch.run(3.0)

        # 1. Taken: the chosen microphone stays listed but delivers nothing.
        mark = len(watch.said)
        taken_at = time.monotonic() - watch.started
        pactl("suspend-source", "heliox_taken_a", "1")
        seen = watch.run(10.0, lambda: bool(watch.since(mark, "microphone_problem")))
        problems = watch.since(mark, "microphone_problem")
        check(
            "audio-linux/taken/a-silenced-microphone-is-said-within-seconds",
            seen and bool(problems) and problems[0][0] - taken_at < 10.0,
            f"said {problems[0][0] - taken_at:.1f} s after it was taken: "
            f"{[(round(t, 1), r, s) for t, _status, r, s in problems[:3]]}"
            if problems
            else "nothing said in 10 s",
        )
        # 2. Given back: the same microphone delivers again, and that is said.
        mark = len(watch.said)
        pactl("suspend-source", "heliox_taken_a", "0")
        back = watch.run(
            15.0, lambda: bool(watch.since(mark, "microphone_recovered")) and capturing_from("heliox_taken_a")
        )
        check(
            "audio-linux/taken/given-back-it-is-heard-again-and-said",
            back,
            f"said: {[(round(t, 1), s) for t, _status, _r, s in watch.said[mark:]][:4]}; on the chosen microphone: "
            f"{capturing_from('heliox_taken_a')}",
        )

        # 3. Gone: the chosen microphone is removed while captured.
        mark = len(watch.said)
        index = ""
        for line in pactl("list", "short", "modules").splitlines():
            if "module-null-source" in line and "source_name=heliox_taken_a" in line:
                index = line.split("\t", 1)[0]
        pactl("unload-module", index) if index else print("INFO the chosen microphone's module was not found")
        watch.run(10.0, lambda: bool(watch.since(mark, "microphone_problem")))
        frames_then = int(recorder.frames_received)
        time.sleep(2.0)
        hearing = int(recorder.frames_received) > frames_then
        check(
            "audio-linux/taken/a-removed-microphone-leaves-heliox-hearing",
            hearing and (capturing_from("heliox_taken_b") or bool(watch.since(mark, "microphone_problem"))),
            f"frames still arriving: {hearing}; on the other microphone: {capturing_from('heliox_taken_b')}; said: "
            f"{[(round(t, 1), r, s) for t, _status, r, s in watch.since(mark, 'microphone_problem')][:3]}",
        )
        # 4. Plugged back: Heliox goes back to the chosen microphone by itself and says so.
        mark = len(watch.said)
        load_null_source("heliox_taken_a", "HelioxChosenMic")
        returned = watch.run(15.0, lambda: capturing_from("heliox_taken_a"))
        check(
            "audio-linux/taken/plugged-back-it-is-used-again",
            returned,
            f"said: {[(round(t, 1), r, s) for t, _status, r, s in watch.said[mark:]][:4]}; device "
            f"{recorder.input_device_name!r}",
        )
    finally:
        recorder._close_capture()


def main(argv: list[str]) -> int:
    if argv and argv[0] == "follow":
        follow()
    elif argv and argv[0] == "taken":
        taken()
    else:
        playback()
    return failures


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
