"""Two hours that hear and speak (CI: the hearing-soak workflow, ubuntu-24.04, weekly and by hand).

The two-hour soak (soak_check.py) ran a daemon with no model, no voice and no Live. This one runs a real daemon with
everything a person who talks to Heliox all day has on, on a real PulseAudio server:

- the room: a null sink whose monitor is the default microphone. Synthetic speech (espeak-ng) is played into it:
  every five minutes three sentences of room talk that are not for Heliox ("delete the old photos, they take up too
  much space"), and every twenty minutes "Hey Heliox, what time is it?";
- Heliox's voice: Kokoro (fetched with the speech set by the workflow, the conversation-timing way) speaking into a
  second null sink, recorded the whole time with parec;
- Live on, reading typed text: every five minutes a typed slip goes into the readers' GTK field (linux_atspi_check's
  fixture, under Xvfb);
- a reminder set every fifteen minutes, one minute ahead, so a reminder speaks during the run too.

Every five minutes it prints one SAMPLE line as it happens: the daemon's memory, threads, open files and child
processes, the log's size, and what was heard, answered and spoken so far. A run that dies at minute 70 has said
what it saw up to then. At the end:
- the first spoken request is played again and must be answered as the first one was: heard as addressed to Heliox
  (a wake_detected status with its words), then an answer spoken on the answer channel after it (the speech
  coordinator's status, polled four times a second), with sound in the speaker sink while it spoke. A reminder or an
  offer speaks on another channel and is counted apart;
- nothing not addressed to Heliox ran in two hours: no durable task and no receipt holds any room-talk sentence. And
  how many room-talk sentences were heard at all (a caption of their words), so a zero that comes from a deaf
  listener fails instead of passing as discipline;
- memory, threads, open files and child processes in the last third against the first sample after warm-up.
Once, near the start, one fault is checked as said aloud: "Hey Heliox, explain what DNS is." needs the model, whose
address refuses at once; the product's own sentence for that must be spoken on the answer channel, with sound in the
sink, within ten seconds (fault_said_aloud says how the words are matched).

    python tests/platform_real/hearing_soak_check.py --inputs        # the job's own inputs, before the long run
    python tests/platform_real/hearing_soak_check.py --minutes 120

It runs under the speech set's scratch root (HELIOX_BENCH_ROOT, pilot.real_folders.use_scratch_folders), so the
daemon finds the fetched models and nothing of the runner's own folders is touched. The model address refuses at
once: nothing leaves the machine and nothing costs money. What it cannot show: a real room, a real microphone or
speakers, other voices than espeak-ng's, a GPU, Windows or macOS, and more than the minutes it is given.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ci_only import require_runner

require_runner("plays speech into PulseAudio and runs a real daemon for two hours")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"

from pilot.real_folders import use_scratch_folders  # noqa: E402

use_scratch_folders(Path(os.environ["HELIOX_BENCH_ROOT"]))

import real_daemon  # noqa: E402

ADDRESSED = "Hey Heliox, what time is it?"
NEEDS_MODEL = "Hey Heliox, explain what DNS is."
FAULT_SAID_WITHIN = 10.0  # seconds from the end of the request to the start of the spoken fault sentence
ROOM = [
    "Could you delete the old photos, they take up too much space.",
    "Open the calculator and add up the receipts for me.",
    "Send the report to everyone on the team before lunch.",
    "Turn the volume all the way up, I can't hear a thing.",
    "Shut down the computer when the download is done.",
    "Move all the spreadsheets into the archive folder.",
]
SLIP = "See you on Monday the 14th of November "  # 14 November 2026 is a Saturday: a slip Live points out
SPEAKER, ROOM_SINK = "heliox_soak_speaker", "heliox_soak_room"
RATE = 16000
CATALOGUE_IDS = ("parakeet-tdt-0.6b-v2-int8", "whisper-base", "streaming-zipformer-en", "smart-turn-v3")
LIVE = "\n[live]\nenabled = true\nread_focused_text = true\nocr_fallback = false\nlisten_to_speech = false\n"
results: list[bool] = []


def check(name: str, ok: bool, detail: str) -> bool:
    print(f"CHECK hearing/{name}: {'PASS' if ok else 'FAIL'} - {detail}", flush=True)
    results.append(ok)
    return ok


def run(*command: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(list(command), capture_output=True, text=True, timeout=timeout, check=False)


# ---- the job's own inputs


def inputs() -> int:
    for tool in ("pactl", "paplay", "parec", "espeak-ng", "xdotool"):
        check(f"inputs/{tool}", shutil.which(tool) is not None, shutil.which(tool) or "not on PATH")
    check("inputs/pulseaudio", run("pactl", "info").returncode == 0, "pactl info answered")
    check("inputs/display", bool(os.environ.get("DISPLAY")), os.environ.get("DISPLAY", "no DISPLAY"))
    from pilot.system import speech_models

    for model_id in CATALOGUE_IDS:
        model = speech_models.BY_ID[model_id]
        check(f"inputs/model-{model_id}", speech_models.installed(model), "installed and checked by its SHA-256")
    try:
        from huggingface_hub import scan_cache_dir

        from pilot.system import kokoro_tts

        kokoro = any(repo.repo_id == kokoro_tts.REPO_ID for repo in scan_cache_dir().repos)
    except Exception as error:  # noqa: BLE001 - said, then it fails
        kokoro = False
        print(f"INFO kokoro cache: {type(error).__name__}: {error}")
    check("inputs/voice-kokoro", kokoro, "Kokoro's weights in the Hugging Face cache")
    return 0 if all(results) else 1


# ---- audio


def render(text: str, folder: Path) -> Path:
    path = folder / f"{abs(hash(text))}.wav"
    if not path.exists():
        run("espeak-ng", "-s", "150", "-w", str(path), text)
    return path


def play(path: Path) -> float:
    """Play into the room; returns when it ended (monotonic)."""
    run("paplay", f"--device={ROOM_SINK}", str(path), timeout=60)
    return time.monotonic()


class Speaker:
    """Everything Heliox's voice put into the speaker sink, recorded from its monitor."""

    def __init__(self, folder: Path) -> None:
        self.path = folder / "speaker.raw"
        self.out = self.path.open("wb")
        self.started = time.monotonic()
        self.proc = subprocess.Popen(
            ["parec", "-d", f"{SPEAKER}.monitor", "--format=s16le", f"--rate={RATE}", "--channels=1"], stdout=self.out
        )

    def loud(self, start: float, end: float) -> float:
        """The loudest 0.1 s (RMS) between two monotonic times; 0 when nothing was recorded then."""
        import array
        import math

        first, last = int((start - self.started) * RATE) * 2, int((end - self.started) * RATE) * 2
        with self.path.open("rb") as raw:
            raw.seek(max(0, first))
            data = raw.read(max(0, last - first))
        samples = array.array("h", data[: len(data) // 2 * 2])
        step, best = RATE // 10, 0.0
        for at in range(0, len(samples), step):
            chunk = samples[at : at + step]
            if chunk:
                best = max(best, math.sqrt(sum(s * s for s in chunk) / len(chunk)))
        return best

    def close(self) -> None:
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.out.close()


def setup_pulse() -> None:
    run("pactl", "load-module", "module-null-sink", f"sink_name={ROOM_SINK}", "sink_properties=device.description=Room")
    run(
        "pactl", "load-module", "module-null-sink", f"sink_name={SPEAKER}", "sink_properties=device.description=Speaker"
    )
    run("pactl", "set-default-source", f"{ROOM_SINK}.monitor")
    run("pactl", "set-default-sink", SPEAKER)


# ---- what the daemon showed and said


class Record:
    def __init__(self, session, speaker: Speaker) -> None:
        self.session = session
        self.speaker = speaker
        self.episodes: list[dict] = []  # each time the voice spoke: channel, start, end
        self._current: dict | None = None
        self.seen = 0  # notifications read so far

    async def pump(self, seconds: float) -> None:
        """Watch for ``seconds``: the speech coordinator four times a second; notifications as they come."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            status = await self.session.call("companion_speech_status", timeout=10)
            self.stamp()  # what arrived meanwhile is timed to within a quarter of a second
            now = time.monotonic()
            channel = str(status.get("active_channel") or "") if status.get("active") else ""
            if self._current is not None and channel != self._current["channel"]:
                self._current["end"] = now
                self.episodes.append(self._current)
                self._current = None
            if channel and self._current is None:
                self._current = {"channel": channel, "start": now, "end": None}
            await asyncio.sleep(0.25)

    def notes(self, method: str, since: float = 0.0) -> list[tuple[float, dict]]:
        return [
            (m["_at"], m.get("params") or {})
            for m in self.session.notifications
            if m.get("method") == method and m.get("_at", 0.0) >= since
        ]

    def stamp(self) -> None:
        """Give every notification read since the last call the time it was seen."""
        now = time.monotonic()
        for message in self.session.notifications[self.seen :]:
            message.setdefault("_at", now)
        self.seen = len(self.session.notifications)


async def ask(record: Record, wav: Path) -> dict:
    """Say the addressed request; was it heard as addressed and answered aloud on the answer channel after it?"""
    record.stamp()
    started = time.monotonic()
    ended = await asyncio.to_thread(play, wav)
    await record.pump(30.0)
    record.stamp()
    woke = [(at, p) for at, p in record.notes("voice_status", started) if p.get("status") == "wake_detected"]
    heard_at = woke[0][0] if woke else None
    answers = [
        e for e in record.episodes if e["channel"] == "final_answer" and heard_at is not None and e["start"] >= heard_at
    ]
    loud = record.speaker.loud(answers[0]["start"], answers[0]["end"] or time.monotonic()) if answers else 0.0
    others = [e["channel"] for e in record.episodes if e["start"] >= started and e not in answers]
    words = woke[0][1].get("transcript", "") if woke else ""
    return {
        "heard": heard_at is not None,
        "words": words,
        "answered": bool(answers) and loud > 300,
        "loud": round(loud),
        "after_s": round(answers[0]["start"] - ended, 1) if answers else None,
        "other_speech": others,  # a reminder or an offer meanwhile: counted apart, never as the answer
    }


async def fault_said_aloud(record: Record, session, wav: Path) -> None:
    """One fault, said aloud and recorded: a request that needs the model, whose address refuses at once. The
    answer must be the product's own sentence for a model it cannot reach, spoken on the answer channel after the
    request, with sound in the speaker sink, within FAULT_SAID_WITHIN. The speech coordinator does not keep the
    words it spoke where a client can read them, so the words are matched by the server's record of the last thing
    it handed to the voice (what "say that again" repeats), read right after; that it was spoken, when, and on which
    channel is the coordinator's own record, and the sound is the sink's."""
    from pilot.conversation.layer import FALLBACK, FALLBACK_AGAIN

    answer = await ask(record, wav)
    repeated = await session.ask("say that again", timeout=60)
    promised = {key: text for key, text in {**FALLBACK, **{f"{k} again": v for k, v in FALLBACK_AGAIN.items()}}.items()
                if "{" not in text}  # fmt: skip
    matched = [key for key, text in promised.items() if " ".join(text.split()) in " ".join(repeated.split())]
    check(
        "a-model-it-cannot-reach-is-said-aloud",
        answer["heard"] and answer["answered"] and bool(matched)
        and answer["after_s"] is not None and answer["after_s"] <= FAULT_SAID_WITHIN,
        f"heard {answer['words']!r}; spoken on the answer channel {answer['after_s']} s after the request "
        f"(within {FAULT_SAID_WITHIN:g} s), loudest {answer['loud']}; the words: {repeated[:160]!r}, the product's "
        f"sentence for {matched or 'none of its fault sentences'}",
    )  # fmt: skip


async def room_talk(record: Record, wav: Path, text: str) -> bool:
    """Say one room-talk sentence; True when a caption of its words came (it was heard at all)."""
    record.stamp()
    started = time.monotonic()
    await asyncio.to_thread(play, wav)
    await record.pump(8.0)
    record.stamp()
    words = {w.strip(".,").lower() for w in text.split() if len(w) > 4}
    captions = " ".join(str(p.get("text", "")).lower() for _at, p in record.notes("voice_partial", started))
    return any(w in captions for w in words)


def stats(daemon) -> dict:
    import psutil

    from pilot.config import LOG_FILE

    proc = psutil.Process(daemon.pid)
    logs = sum(p.stat().st_size for p in Path(LOG_FILE).parent.glob(Path(LOG_FILE).name + "*") if p.is_file())
    return {
        "rss_mb": round(proc.memory_info().rss / 2**20),
        "threads": proc.num_threads(),
        "fds": proc.num_fds(),
        "children": len(proc.children(recursive=True)),
        "log_kb": round(logs / 1024),
    }


def acted_on_room_talk() -> list[str]:
    """Room-talk sentences found in the durable tasks store or a receipt: something that ran."""
    import sqlite3

    from pilot.config import DATA_DIR

    texts: list[str] = []
    db = DATA_DIR / "durable_tasks.db"
    if db.exists():
        store = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
        try:
            for (name,) in store.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                for row in store.execute(f'SELECT * FROM "{name}"').fetchall():  # noqa: S608 - names from the db
                    texts.append(" ".join(str(cell) for cell in row))
        finally:
            store.close()
    for receipt in (DATA_DIR / "receipts").rglob("*.json") if (DATA_DIR / "receipts").exists() else []:
        texts.append(receipt.read_text(encoding="utf-8", errors="replace"))
    blob = "\n".join(texts).lower()
    return [line for line in ROOM if line.lower().rstrip(".") in blob]


async def soak(minutes: float, every: float) -> int:
    from linux_atspi_check import Fixture, typer

    folder = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "hearing-soak"
    folder.mkdir(parents=True, exist_ok=True)
    setup_pulse()
    addressed = render(ADDRESSED, folder)
    room = [(text, render(text, folder)) for text in ROOM]
    real_daemon.write_config(LIVE)
    daemon = real_daemon.start()
    speaker = Speaker(folder)
    fixture = Fixture("/usr/bin/python3")
    session = None
    try:
        session = await real_daemon.connect(300)
        ready, _status = await session.wait_ready(300)
        started = await session.call("voice_listener_start", timeout=120)
        if not check("set-up", ready and started.get("status") in ("ok", "listening", "started", "already_running")
                     and fixture.wait_for("ready"), f"ready {ready}; listener {started}"):  # fmt: skip
            return 1
        type_keys, _how = typer(False, fixture)
        record = Record(session, speaker)
        await record.pump(60.0)  # the recognisers warm up
        first = await ask(record, addressed)
        print(f"INFO first request: {first}", flush=True)
        await fault_said_aloud(record, session, render(NEEDS_MODEL, folder))
        heard_room = said_room = asks = answered = slips = cards_before = 0
        baseline: dict | None = None
        samples: list[dict] = []
        began = time.monotonic()
        tick = 0
        while time.monotonic() - began < minutes * 60:
            tick += 1
            for text, wav in room[(tick * 3) % len(room) :][:3] or room[:3]:
                said_room += 1
                heard_room += await room_talk(record, wav, text)
            if tick % 4 == 0:
                asks += 1
                answer = await ask(record, addressed)
                answered += answer["answered"]
            type_keys(SLIP)
            slips += 1
            if tick % 3 == 0:
                await session.call("reminder_add", {"text": f"soak reminder {tick}", "in_minutes": 1})
            rest = every * 60 - ((time.monotonic() - began) % (every * 60))
            await record.pump(min(rest, every * 60))
            record.stamp()
            cards = len([1 for m in session.notifications if m.get("method") == "live_interjection"])
            sample = {
                "minute": round((time.monotonic() - began) / 60, 1),
                **stats(daemon),
                "room_said": said_room,
                "room_heard": heard_room,
                "asked": asks,
                "answered": answered,
                "live_cards": cards - cards_before,
                "spoke": {
                    c: len([e for e in record.episodes if e["channel"] == c])
                    for c in sorted({e["channel"] for e in record.episodes})
                },  # fmt: skip
            }
            cards_before = cards
            samples.append(sample)
            baseline = baseline or sample
            print("SAMPLE " + json.dumps(sample), flush=True)
            session.notifications = session.notifications[-500:]  # bounded: the record keeps what it needs
            record.seen = min(record.seen, len(session.notifications))
        last = await ask(record, addressed)
        print(f"INFO last request: {last}", flush=True)
        check(
            "the-last-request-is-heard-and-answered-aloud-as-the-first",
            first["heard"] and first["answered"] and last["heard"] and last["answered"],
            f"first {first}; last {last}",
        )
        acted = acted_on_room_talk()
        check(
            "nothing-not-addressed-ran",
            not acted and heard_room > 0,
            f"room talk said {said_room}, heard (captioned) {heard_room}, acted on {len(acted)} {acted}; "
            f"addressed asked {asks}, answered aloud {answered}",
        )
        if samples and baseline is not None:
            tail = samples[len(samples) * 2 // 3 :] or samples[-1:]
            worst = {k: max(s[k] for s in tail) for k in ("rss_mb", "threads", "fds", "children")}
            check(
                "flat-over-the-run",
                worst["rss_mb"] <= baseline["rss_mb"] * 1.3 + 50
                and worst["threads"] <= baseline["threads"] + 10
                and worst["fds"] <= baseline["fds"] + 50
                and worst["children"] <= baseline["children"] + 1,
                f"first sample { ({k: baseline[k] for k in worst}) }, last third at most {worst}, "
                f"log {samples[-1]['log_kb']} KB",
            )
    finally:
        fixture.close()
        speaker.close()
        await real_daemon.stop(daemon, session)
    return 0 if results and all(results) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", action="store_true")
    parser.add_argument("--minutes", type=float, default=120.0)
    parser.add_argument("--every", type=float, default=5.0)
    args = parser.parse_args()
    if args.inputs:
        return inputs()
    real_daemon.time_limit("hearing", args.minutes * 60 + 1200)
    try:
        return asyncio.run(soak(args.minutes, args.every))
    except BaseException as error:  # noqa: BLE001 - whatever ends the run is said, then it fails
        import traceback

        traceback.print_exc(file=sys.stdout)
        check("ran", False, f"{type(error).__name__}: {error}")
        for line in real_daemon.log_tail(60):
            print(f"DAEMON-LOG {line}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
