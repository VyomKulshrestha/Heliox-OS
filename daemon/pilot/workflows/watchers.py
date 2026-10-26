"""Watchers: "tell me when Chrome closes", "let me know when my download finishes".

A watcher waits for one thing to happen on this computer and then tells the person once:
an app closing, the downloads in the Downloads folder finishing, a new file arriving
there, the battery reaching a level, or the processor settling down. Watchers only
look; they never close, open or change anything. They live in memory, so a restart
forgets them, and each one gives up after a day.

Watchers answer something the person just asked for, so a focus session does not hold
them back.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot.workflows.watchers")

CHECK_EVERY_SECONDS = 5.0
GIVE_UP_AFTER_SECONDS = 24 * 3600
MAX_WATCHERS = 10
CPU_CALM_CHECKS = 3  # consecutive readings under the threshold
PARTIAL_SUFFIXES = (".crdownload", ".part", ".partial", ".download", ".opdownload")
PRONOUNS = {"it", "it s", "that", "that s", "this", "they", "they re", "everything", "you", "you re"}

SET_REQUEST = re.compile(
    r"^(?:please )?(?:tell me|let me know|notify me|alert me|ping me|say something) (?:when|once|as soon as) (?P<what>.+)$"
)
LIST_REQUEST = re.compile(
    r"^what are you (?:watching|waiting) for$|^(?:what|which) watch(?:ers|es) do i have$|^(?:list|show) my watch(?:ers|es)$"
)
CANCEL_REQUEST = re.compile(
    r"^(?:please )?(?:stop watching|stop waiting for|cancel the watch(?:er)? (?:for|on)|never mind about)"
    r"(?: (?:for|on))? (?P<what>.+)$"
)

_DOWNLOAD = re.compile(r"^(?:my |the )?downloads? (?:finish(?:es)?|completes?|(?:is|are) (?:done|finished|complete))$")
_ARRIVES = re.compile(
    r"^(?:a |the |my )?(?P<name>[\w .-]+?) (?:appears|shows up|arrives|lands|is downloaded|gets downloaded)"
    r"(?: in (?:my |the )?downloads(?: folder)?)?$"
)
_BATTERY = re.compile(
    r"^(?:my |the )?battery (?:is |gets |goes |drops |falls |reaches |hits |charges )*"
    r"(?P<dir>above |over |below |under |to |at )?(?P<pct>\d{1,3})(?: percent)?$"
)
_BATTERY_WORDS = re.compile(
    r"^(?:my |the )?battery (?:is |gets )?(?P<state>full|fully charged|charged|low|running low)$"
)
_CPU = re.compile(
    r"^(?:my |the )?(?:cpu|processor|computer|pc|laptop)(?: usage| load)? (?:drops |falls |goes |is |gets )*"
    r"(?:below|under) (?P<pct>\d{1,2})(?: percent)?$"
)
_CALM = re.compile(r"^(?:my |the )?(?:computer|pc|laptop|cpu|processor) (?:calms down|settles down|is idle|is quiet)$")
_CLOSES = re.compile(
    r"^(?:my |the )?(?P<name>[\w .+-]+?) (?:closes|quits|exits|stops running|is closed|has closed|"
    r"finishes|is finished|is done|ends)$"
)


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s%.+-]", " ", text.casefold()).replace("%", " percent").split()).strip(" .")


def _stem(process_name: str) -> str:
    name = process_name.casefold()
    return name[:-4] if name.endswith(".exe") else name


def matches_process(query: str, process_name: str) -> bool:
    """Whether a spoken app name ("google chrome", "obs") fits a process name ("chrome.exe")."""
    stem = _stem(process_name)
    squashed = query.replace(" ", "")
    if not squashed or not stem:
        return False
    if squashed == stem or (len(squashed) >= 4 and squashed in stem) or (len(stem) >= 4 and stem in squashed):
        return True
    words = [word for word in query.split() if len(word) >= 3 and word not in ("app", "the", "my")]
    return bool(words) and words[-1] == stem


@dataclass
class Probes:
    """What watchers read, so tests can stand in for the real computer."""

    processes: Callable[[], list[tuple[int, str, float]]]
    alive: Callable[[int, float], bool]
    battery: Callable[[], tuple[float, bool] | None]
    cpu: Callable[[], float]
    downloads: Path = field(default_factory=lambda: Path.home() / "Downloads")


def system_probes() -> Probes:
    import psutil

    def processes() -> list[tuple[int, str, float]]:
        rows = []
        for proc in psutil.process_iter(["pid", "name", "create_time"]):
            info = proc.info
            if info.get("name"):
                rows.append((int(info["pid"]), str(info["name"]), float(info.get("create_time") or 0)))
        return rows

    def alive(pid: int, created: float) -> bool:
        try:
            return abs(psutil.Process(pid).create_time() - created) < 1.0
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
            return False

    def battery() -> tuple[float, bool] | None:
        reading = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
        return None if reading is None else (float(reading.percent), bool(reading.power_plugged))

    psutil.cpu_percent(interval=None)  # the first reading only starts the measurement
    return Probes(processes=processes, alive=alive, battery=battery, cpu=lambda: psutil.cpu_percent(interval=None))


class WatcherStore:
    """Active watchers in memory, checked every few seconds by a background loop."""

    def __init__(
        self,
        *,
        probes: Probes | Callable[[], Probes] = system_probes,
        deliver: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._probes_source = probes
        self._probes: Probes | None = probes if isinstance(probes, Probes) else None
        self._deliver = deliver
        self._clock = clock
        self._items: list[dict[str, Any]] = []
        self._task: asyncio.Task[None] | None = None

    @property
    def probes(self) -> Probes:
        if self._probes is None:
            self._probes = self._probes_source()  # type: ignore[operator]
        return self._probes

    def list(self) -> list[dict[str, Any]]:
        return [{key: item[key] for key in ("id", "kind", "label", "created_at")} for item in self._items]

    def _partials(self) -> list[Path]:
        folder = self.probes.downloads
        try:
            return [path for path in folder.iterdir() if path.suffix.casefold() in PARTIAL_SUFFIXES]
        except OSError:
            return []

    def _finished_files(self, since: float) -> list[Path]:
        folder = self.probes.downloads
        try:
            files = [
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.casefold() not in PARTIAL_SUFFIXES and path.stat().st_mtime >= since
            ]
        except OSError:
            return []
        return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)

    def plan(self, sentence: str) -> dict[str, Any] | str | None:
        """Turn "tell me when …" into a watcher, an explanation of why not, or None if it is not one."""
        asked = SET_REQUEST.match(_normalize(sentence))
        if not asked:
            return None
        what = asked.group("what").strip()
        if _DOWNLOAD.match(what):
            partials = self._partials()
            if not partials:
                return "I don't see a download in progress in your Downloads folder."
            return {
                "kind": "download",
                "label": "your download finishes" if len(partials) == 1 else f"your {len(partials)} downloads finish",
                "files": sorted(str(path) for path in partials),
            }
        battery = _BATTERY.match(what) or _BATTERY_WORDS.match(what)
        if battery:
            reading = self.probes.battery()
            if reading is None:
                return "I can't read a battery on this computer."
            if "state" in battery.groupdict():
                state = battery.group("state")
                below = state in ("low", "running low")
                level = 20 if below else 99
            else:
                level = int(battery.group("pct"))
                direction = (battery.group("dir") or "").strip()
                if direction in ("below", "under") or "drops" in what or "falls" in what:
                    below = True
                elif direction in ("above", "over") or "charges" in what:
                    below = False
                else:  # "reaches 50": downwards if it is above that now
                    below = reading[0] > level
            if not 1 <= level <= 100:
                return "Give me a battery level from 1 to 100 percent."
            label = f"the battery {'drops to' if below else 'reaches'} {level}%"
            if level == 99 and not below:
                label = "the battery is full"
            return {"kind": "battery", "label": label, "level": level, "below": below}
        cpu = _CPU.match(what)
        if cpu or _CALM.match(what):
            level = int(cpu.group("pct")) if cpu else 20
            if not 1 <= level <= 99:
                return "Give me a processor level from 1 to 99 percent."
            return {"kind": "cpu", "label": f"the processor stays under {level}%", "level": level, "calm": 0}
        arrives = _ARRIVES.match(what)
        if arrives and not _CLOSES.match(what):
            name = arrives.group("name").strip()
            try:
                existing = sorted(path.name for path in self.probes.downloads.iterdir())
            except OSError:
                return "I can't read your Downloads folder."
            return {"kind": "file", "label": f"{name} arrives in Downloads", "name": name, "existing": existing}
        closes = _CLOSES.match(what)
        if closes:
            name = re.sub(r"^(?:the |my )", "", closes.group("name")).strip()
            if name.endswith(" app"):
                name = name[: -len(" app")]
            if name in PRONOUNS:
                return None  # "let me know when it's done" is about the current task, not an app
            running = [(pid, created) for pid, proc, created in self.probes.processes() if matches_process(name, proc)]
            if not running:
                return f"I can only watch apps that are running, and I can't find {name} running right now."
            return {"kind": "process", "label": f"{name} closes", "name": name, "pids": running}
        return None

    def add(self, sentence: str) -> dict[str, Any]:
        planned = self.plan(sentence)
        if planned is None:
            raise ValueError('Say what to watch for, for example "tell me when Chrome closes".')
        if isinstance(planned, str):
            raise ValueError(planned)
        return self._arm(planned)

    def _arm(self, planned: dict[str, Any]) -> dict[str, Any]:
        if len(self._items) >= MAX_WATCHERS:
            raise ValueError(f"I can watch up to {MAX_WATCHERS} things at once.")
        entry = {"id": uuid.uuid4().hex[:10], "created_at": self._clock(), **planned}
        self._items.append(entry)
        return entry

    def cancel(self, watcher_id: str | None = None, *, label: str | None = None) -> int:
        if watcher_id:
            kept = [item for item in self._items if item["id"] != watcher_id]
        else:
            words = _normalize(label or "").split()
            kept = [item for item in self._items if not all(word in item["label"].casefold() for word in words)]
        removed = len(self._items) - len(kept)
        self._items = kept
        return removed

    def _happened(self, item: dict[str, Any]) -> str | None:
        """The sentence to say when ``item`` has happened, else None."""
        probes = self.probes
        kind = item["kind"]
        if kind == "process":
            if any(probes.alive(pid, created) for pid, created in item["pids"]):
                return None
            return f"{item['name'].capitalize()} has closed."
        if kind == "download":
            if any(Path(path).exists() for path in item["files"]):
                return None
            finished = self._finished_files(item["created_at"] - 1)
            if finished:
                return f"Your download finished: {finished[0].name}."
            return "Your download is no longer in progress; it may have finished or been cancelled."
        if kind == "file":
            needle = item["name"].casefold().replace(" ", "")
            for path in self._finished_files(item["created_at"] - 1):
                if path.name not in item["existing"] and needle in path.name.casefold().replace(" ", ""):
                    return f"{path.name} is in your Downloads folder."
            return None
        if kind == "battery":
            reading = probes.battery()
            if reading is None:
                return None
            percent, plugged = reading
            if item["below"] and percent <= item["level"]:
                return f"Your battery is at {round(percent)}%{'' if plugged else ' and not charging'}."
            if not item["below"] and percent >= item["level"]:
                return "Your battery is full." if item["level"] >= 99 else f"Your battery is at {round(percent)}%."
            return None
        if kind == "cpu":
            if probes.cpu() < item["level"]:
                item["calm"] += 1
            else:
                item["calm"] = 0
            if item["calm"] >= CPU_CALM_CHECKS:
                return f"Your computer has settled down; the processor is under {item['level']}%."
        return None

    async def tick(self) -> int:
        """Tell the person about every watcher whose event happened. Returns how many fired."""
        fired = 0
        now = self._clock()
        for item in list(self._items):
            if now - item["created_at"] > GIVE_UP_AFTER_SECONDS:
                text = f"I stopped waiting for {item['label']}; it didn't happen within a day."
            else:
                try:
                    text = await asyncio.to_thread(self._happened, item)
                except Exception:
                    logger.warning("Watcher check failed for %s", item["label"], exc_info=True)
                    continue
                if text is None:
                    continue
            self._items = [other for other in self._items if other["id"] != item["id"]]
            fired += 1
            if self._deliver is not None:
                await self._deliver({"id": item["id"], "kind": item["kind"], "label": item["label"], "text": text})
        return fired

    async def _loop(self) -> None:
        while True:
            try:
                if self._items:
                    await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Watcher check failed", exc_info=True)
            await asyncio.sleep(CHECK_EVERY_SECONDS)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def answer(self, text: str) -> str | None:
        """Start, list or cancel watchers from a sentence; None when it is not about watching."""
        normalized = _normalize(text)
        if LIST_REQUEST.match(normalized):
            if not self._items:
                return 'I\'m not watching for anything. Say "tell me when Chrome closes" to start.'
            return "I'm waiting until " + "; ".join(item["label"] for item in self._items) + "."
        cancel = CANCEL_REQUEST.match(normalized)
        if cancel:
            what = cancel.group("what").strip()
            if what in ("everything", "all", "anything"):
                removed = len(self._items)
                self._items = []
            else:
                removed = self.cancel(label=what)
            if not removed:
                return f"I wasn't watching for {what}."
            return "Okay, I stopped watching." if removed == 1 else f"Okay, I stopped {removed} watchers."
        planned = self.plan(text)
        if planned is None:
            return None
        if isinstance(planned, str):
            return planned
        try:
            entry = self._arm(planned)
        except ValueError as exc:
            return str(exc)
        return f"Okay. I'll tell you when {entry['label']}."

    async def answer_async(self, text: str) -> str | None:
        return await asyncio.to_thread(self.answer, text)
