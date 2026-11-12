"""Record your own voice for the Conversation Bench, one sentence at a time (nothing leaves this computer).

  python -m benchmarks.conversation_bench.record --mic desk [--set sentences|wake|room] [--device NAME] [--audio DIR]

A full pass takes about fifteen minutes per microphone (61 short sentences, then five minutes of room sound).
``--mic`` is a short name for the microphone you are using (desk, laptop, headset, ...): record the same sets on
each microphone you have, under a different name. The input device is the one Heliox is set to use, or the one
named with ``--device``; no system setting is changed and nothing is played. For each sentence: press Enter, say
it, press Enter. Then keep it, redo it, skip it or quit. Stopping and starting again carries on where you left off.

Sets:
  sentences  41 different things people say to Heliox (about 420 words; with the wake set, about 540):
             requests, questions about the day, follow-ups, control words, and long sentences with a pause at
             the comma.
  wake       20 sentences that start with "Hey Heliox".
  room       5 minutes of a television, a video or another person talking near the microphone, not to Heliox.

Files go to ``<audio folder>/real/<mic>/<set>/<id>.wav`` (16 kHz mono) with ``manifest.json`` and
``scenarios.json`` beside them in ``real/<mic>/``; ``run`` picks those scenarios up automatically. They are
private: the audio folder is never committed.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from . import audio as audio_refs
from . import wavio

RATE = 16000
ROOM_SECONDS = 300.0

SENTENCES: tuple[str, ...] = (
    "Remind me in twenty minutes to stand up and stretch for a bit.",
    "What's on my calendar today, and is anything moved since this morning?",
    "And tomorrow?",
    "Open Notepad and start a new note called ideas for the launch.",
    "Turn the volume up a little, it's too quiet to hear the video.",
    "What's the battery level, and how long will it last at this rate?",
    "How much disk space do I have left on the main drive?",
    "Set a timer for ten minutes and tell me when it's halfway.",
    "Add milk, eggs and coffee beans to my shopping list.",
    "What did I work on this morning before the stand-up meeting?",
    "Find the invoice from last week and show it to me on screen.",
    "Delete the old installers in my Downloads folder, but keep anything from this month.",
    "Actually, only the ones from last year.",
    "Wait, stop.",
    "No, don't do that.",
    "Never mind.",
    "Undo that.",
    "Say that again, a bit more slowly please.",
    "Shorter, please.",
    "Why is my laptop fan so loud when nothing much is open?",
    "How do I see which program is using most of the processor right now?",
    "What time is my first meeting on Thursday, and who is it with?",
    "Remind me that the launch call is on Monday the sixteenth at ten in the morning.",
    "Move the design review to three in the afternoon and let everyone know.",
    "Email Priya the deck and tell her it's the final version for Friday.",
    "My flight is on Friday morning, so remind me on Thursday evening to pack.",
    "Start a focus session for forty-five minutes and keep notifications quiet until it ends.",
    "What's the weather like this weekend, will it rain on Saturday?",
    "Explain how undo works, in one or two sentences.",
    "Open the budget sheet, then scroll down to the totals for October.",
    "When you have a moment, can you tell me what's still on my to-do list?",
    "Close Chrome, it's using all the memory and the laptop is slow.",
    "Search my documents for the contract with Acme from last spring.",
    "Pause the music for a minute.",
    "Thanks, that's all for now.",
    "Read me the last message from Sam, the one about the budget.",
    "Rename the file to final report, with today's date at the end.",
    "Show me the photos I took yesterday afternoon at the office.",
    "What's two hundred and forty times twelve, and what's that divided by three?",
    "Turn on do not disturb until five o'clock this afternoon.",
    "Okay, go back to what you were doing before I interrupted you.",
)
WAKE: tuple[str, ...] = (
    "Hey Heliox, what can I say?",
    "Hey Heliox, open Notepad.",
    "Hey Heliox, what time is it?",
    "Hey Heliox, remind me to call the bank at four.",
    "Hey Heliox, what's on today?",
    "Hey Heliox, stop.",
    "Hey Heliox, undo that.",
    "Hey Heliox, turn the volume down.",
    "Hey Heliox, how much battery do I have?",
    "Hey Heliox, set a timer for five minutes.",
    "Hey Heliox, what's the date today?",
    "Hey Heliox, find my notes from the launch meeting.",
    "Hey Heliox, mute.",
    "Hey Heliox, what did I tell you about my flight?",
    "Heliox, what's next on my list?",
    "Heliox, close the browser.",
    "Hey Heliox, good morning.",
    "Hey Heliox, start a pomodoro.",
    "Hey Heliox, explain how DNS works, briefly.",
    "Hey Heliox, add bread to the shopping list.",
)
CONTROL = {"Wait, stop.", "No, don't do that.", "Never mind.", "Undo that."}


class Capture(Protocol):
    """An input stream that records between start() and stop(); stop() returns 16 kHz mono float32 samples."""

    def start(self) -> None: ...

    def stop(self) -> Any: ...


class DeviceCapture:
    """The chosen input device through sounddevice (imported only here); records, never plays, changes nothing."""

    def __init__(self, device: str | int | None) -> None:
        import sounddevice as sd

        self.sd = sd
        self.device = device
        self.blocks: list[Any] = []
        self.stream: Any = None

    def describe(self) -> tuple[str, float]:
        """The input device's name and its own sample rate (recording is resampled to 16 kHz by the driver)."""
        info = self.sd.query_devices(self.device, "input")
        return str(info["name"]), float(info["default_samplerate"])

    def start(self) -> None:
        self.blocks = []

        def callback(data: Any, _frames: int, _time: Any, _status: Any) -> None:
            self.blocks.append(data[:, 0].copy())

        self.stream = self.sd.InputStream(samplerate=RATE, channels=1, dtype="float32", device=self.device,
                                          callback=callback)  # fmt: skip
        self.stream.start()

    def stop(self) -> Any:
        import numpy as np

        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        return np.concatenate(self.blocks) if self.blocks else np.zeros(0, dtype=np.float32)


HEADSET_WORDS = ("headset", "bluetooth", "bt", "hands-free", "handsfree", "airpods", "buds")


def label_says_headset(mic: str) -> bool:
    return any(word in safe_mic(mic) for word in HEADSET_WORDS)


def narrowband(samples: Any) -> bool | None:
    """Bluetooth call mode from the sound: almost nothing above 4.5 kHz against the speech band (the product's own
    test, pilot.system.mic_quality). None when the take is too short to tell."""
    import numpy as np

    from pilot.system.mic_quality import NARROWBAND_RATIO

    x = np.asarray(samples, dtype=np.float32)
    if len(x) < 2 * RATE or float(np.abs(x).max()) < 1e-3:
        return None
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / RATE)
    speech = float(spectrum[(freqs >= 300) & (freqs < 3400)].sum())
    high = float(spectrum[(freqs >= 4500) & (freqs < 7500)].sum())
    return speech > 0 and high / speech < NARROWBAND_RATIO


CALL_MODE_STOP = (
    "This microphone is a Bluetooth headset in call mode, but you started with --mic {mic}. Recordings from it would "
    "be filed under the wrong microphone and give wrong numbers. Run again with --mic headset, or choose another "
    "microphone with --device."
)


def hands_free(device_name: str) -> bool:
    """The device name says call mode (Windows and Linux name it; on macOS the first take's sound decides)."""
    from pilot.system.mic_quality import hands_free_name

    return hands_free_name(device_name)


def heliox_input_device() -> str | None:
    """The input device Heliox is set to use (None for the system default)."""
    try:
        from pilot.config import PilotConfig

        name = str(PilotConfig.load().voice.input_device or "auto")
    except Exception:
        return None
    return None if name in ("", "auto", "default") else name


@dataclass(frozen=True)
class Item:
    id: str
    set: str
    text: str  # "" for the room recording
    label: str  # addressed | control | not_addressed


def items(which: str) -> list[Item]:
    if which == "sentences":
        return [Item(f"s{i:02d}", "sentences", t, "control" if t in CONTROL else "addressed")
                for i, t in enumerate(SENTENCES)]  # fmt: skip
    if which == "wake":
        return [Item(f"w{i:02d}", "wake", t, "addressed") for i, t in enumerate(WAKE)]
    if which == "room":
        return [Item("room00", "room", "", "not_addressed")]
    raise ValueError(f"unknown set {which!r}")


def safe_mic(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", name.casefold()).strip("-")
    if not cleaned:
        raise ValueError("give the microphone a short name, like desk or headset")
    return cleaned


def mic_folder(root: str, mic: str) -> Path:
    return Path(root) / "real" / safe_mic(mic)


def load_manifest(folder: Path) -> dict[str, Any]:
    path = folder / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"mic": folder.name, "clips": {}}


def scenario_for(item: Item, mic: str) -> dict[str, Any]:
    """The bench scenario for one recorded clip (written to the private scenarios.json, never to the repo)."""
    ref = f"real/{mic}/{item.set}/{item.id}.wav"
    if item.set == "room":
        return {"id": f"own-{mic}-room", "set": "not_addressed", "voice": "real", "mic": mic, "lines": ["2.2"],
                "models": ["cloud"], "note": "television or another person near the microphone",
                "steps": [{"say": ref}], "expect": {}}  # fmt: skip
    woken = item.text.lower().startswith(("hey heliox", "heliox"))
    expect: dict[str, Any] = {"woken": True} if woken else {}
    return {"id": f"own-{mic}-{item.id}", "set": "understanding", "voice": "real", "mic": mic,
            "lines": ["2.1a", "2.1b", "1.3"] if woken else ["2.1a", "1.3"], "models": ["cloud"],
            "steps": [{"say": ref, "text": item.text}], "expect": expect}  # fmt: skip


def save(folder: Path, manifest: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    mic = folder.name
    clips = [Item(c["id"], c["set"], c["text"], c["label"]) for c in manifest["clips"].values()]
    scenarios = [scenario_for(item, mic) for item in sorted(clips, key=lambda i: i.id)]
    (folder / "scenarios.json").write_text(json.dumps({"scenarios": scenarios}, indent=1), encoding="utf-8")


def record_set(
    root: str,
    mic: str,
    which: str,
    capture: Capture,
    *,
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, int]:
    """Walk through one set; returns counts of kept, skipped and already-done clips. Resumes from the manifest."""
    folder = mic_folder(root, mic)
    manifest = load_manifest(folder)
    counts = {"kept": 0, "skipped": 0, "already": 0}
    check_sound = which != "room"  # the first kept take is checked for call-mode sound (a room may be anything)
    todo = items(which)
    for n, item in enumerate(todo, 1):
        if item.id in manifest["clips"]:
            counts["already"] += 1
            continue
        while True:
            if item.set == "room":
                say(f"\nRoom recording: let a television, a video or another person talk near the microphone for "
                    f"{ROOM_SECONDS / 60:.0f} minutes. Do not talk to Heliox.")  # fmt: skip
                if ask("Press Enter to start (q to quit): ").strip().lower() == "q":
                    save(folder, manifest)
                    return counts
                capture.start()
                sleep(ROOM_SECONDS)
            else:
                say(f"\n[{which} {n}/{len(todo)}]  Say:  {item.text}")
                if ask("Press Enter, then say it (q to quit): ").strip().lower() == "q":
                    save(folder, manifest)
                    return counts
                capture.start()
                ask("Recording... press Enter when you have finished. ")
            samples = capture.stop()
            seconds = len(samples) / RATE
            choice = ask(f"Got {seconds:.1f} s. Keep it (Enter), redo (r), skip (s), quit (q)? ").strip().lower()
            if choice == "r":
                continue
            if choice == "q":
                save(folder, manifest)
                return counts
            if choice == "s":
                counts["skipped"] += 1
                break
            if check_sound and not label_says_headset(mic) and narrowband(samples):
                say(CALL_MODE_STOP.format(mic=mic))
                save(folder, manifest)
                counts["stopped"] = 1
                return counts
            check_sound = False
            wavio.write(folder / item.set / f"{item.id}.wav", samples)
            manifest["clips"][item.id] = {"id": item.id, "set": item.set, "text": item.text, "label": item.label,
                                          "seconds": round(seconds, 2), "recorded": time.strftime("%Y-%m-%d %H:%M")}  # fmt: skip
            save(folder, manifest)
            counts["kept"] += 1
            break
    save(folder, manifest)
    return counts


def summary(root: str) -> str:
    lines = []
    for folder in sorted((Path(root) / "real").glob("*/manifest.json")):
        clips = json.loads(folder.read_text(encoding="utf-8"))["clips"].values()
        by_set: dict[str, int] = {}
        words = 0
        for c in clips:
            by_set[c["set"]] = by_set.get(c["set"], 0) + 1
            words += len(c["text"].split())
        lines.append(f"  {folder.parent.name}: " + ", ".join(f"{k} {v}" for k, v in sorted(by_set.items()))
                     + f" ({words} words)")  # fmt: skip
    return "\n".join(lines) or "  nothing recorded yet"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.conversation_bench.record", description=__doc__)
    parser.add_argument("--mic", required=True, help="a short name for this microphone: desk, laptop, headset, ...")
    parser.add_argument("--set", choices=("sentences", "wake", "room"), default=None, help="default: all three")
    parser.add_argument("--device", default=None, help="input device name or number (default: Heliox's)")
    parser.add_argument("--audio", default=None)
    args = parser.parse_args(argv)
    root = audio_refs.audio_root(args.audio)
    device = args.device if args.device is not None else heliox_input_device()
    capture = DeviceCapture(int(device) if isinstance(device, str) and device.isdigit() else device)
    name, rate = capture.describe()
    print(f"Recording from: {name} (its own rate {rate:.0f} Hz; clips are saved at 16 kHz)")
    print(f"Saving into: {mic_folder(root, args.mic)}")
    if hands_free(name) and not label_says_headset(args.mic):
        print(CALL_MODE_STOP.format(mic=args.mic))
        return 1
    for which in [args.set] if args.set else ["sentences", "wake", "room"]:
        counts = record_set(root, args.mic, which, capture)
        if counts.get("stopped"):
            return 1
        print(f"{which}: {counts['kept']} kept, {counts['skipped']} skipped, {counts['already']} already recorded")
    print("\nRecorded so far:\n" + summary(root))
    print(f"\nRun the bench on it:\n  python -m benchmarks.conversation_bench run --audio {root} "
          f"--sets understanding,not_addressed --only own-{safe_mic(args.mic)}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
