"""Health check: "why is my computer slow?", "what's using my memory?", "how much disk space do I have?".

The answers come from one short reading of the processor, memory, disks, battery and the
running apps (processes of the same app are added together, so Chrome counts once), all
in code and without a model. Heliox only looks: it suggests what could be closed, and
closing anything is an ordinary request ("close Chrome") that goes through approval.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from typing import Any

SLOW_QUESTION = re.compile(
    r"^(?:hey heliox |heliox )?(?:why is my (?:computer|pc|laptop|machine|system) (?:so |running |being )*(?:slow|laggy|sluggish)"
    r"|what(?: s| is) (?:slowing|making) (?:down )?my (?:computer|pc|laptop|machine|system)(?: (?:down|slow))?"
    r"|how is my (?:computer|pc|laptop|machine|system) (?:doing|running)"
    r"|(?:check|run a) (?:my |the )?(?:computer|pc|laptop|system)(?: s)?(?: health)?(?: check)?"
    r"|is my (?:computer|pc|laptop) (?:ok|okay|alright|healthy))$"
)
MEMORY_QUESTION = re.compile(
    r"^what(?: s| is) (?:using|eating|taking up|hogging) (?:all )?(?:of )?my (?:memory|ram)$"
    r"|^how much (?:memory|ram) (?:am i using|is (?:being )?used|is free|do i have(?: left| free)?)$"
)
CPU_QUESTION = re.compile(r"^what(?: s| is) (?:using|eating|hogging) (?:all )?(?:of )?my (?:cpu|processor)$")
DISK_QUESTION = re.compile(
    r"^how much (?:disk |hard drive |drive |storage )?(?:space|storage) (?:do i have|is) (?:left|free|there)$"
    r"|^how much (?:disk |hard drive |drive )?(?:space|storage) do i have$"
    r"|^(?:is my (?:disk|drive|hard drive|storage) (?:full|almost full|nearly full)"
    r"|how full is my (?:disk|drive|hard drive|storage))$"
)
HIDDEN_PROCESSES = {
    "system idle process",
    "idle",
    "system",
    "registry",
    "memory compression",
    "memcompression",
    "secure system",
}


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def _gb(value: float) -> str:
    gb = value / 1024**3
    return f"{gb:.1f} GB" if gb < 10 else f"{gb:.0f} GB"


def _drive(mountpoint: str) -> str:
    """A drive as people say it: "C:" for "C:\\", and "/" for the root."""
    return mountpoint.rstrip("\\/") or "/"


def _app_name(process_name: str) -> str:
    name = process_name[:-4] if process_name.casefold().endswith(".exe") else process_name
    return name[:1].upper() + name[1:]


def read_snapshot(sample_seconds: float = 0.5) -> dict[str, Any]:
    """One reading of this computer. Blocks for about ``sample_seconds``; call it in a thread."""
    import psutil

    procs = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            proc.cpu_percent(None)
            procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    cpu_total = psutil.cpu_percent(interval=sample_seconds)
    cores = psutil.cpu_count() or 1
    apps: dict[str, dict[str, Any]] = {}
    for proc in procs:
        try:
            name = proc.info.get("name") or ""
            if proc.info.get("pid") in (0, 4) or name.casefold() in HIDDEN_PROCESSES or not name:
                continue
            with proc.oneshot():
                cpu = proc.cpu_percent(None) / cores
                memory = proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        entry = apps.setdefault(_app_name(name), {"app": _app_name(name), "cpu": 0.0, "memory": 0, "processes": 0})
        entry["cpu"] += cpu
        entry["memory"] += memory
        entry["processes"] += 1
    disks = []
    for part in psutil.disk_partitions(all=False):
        if "cdrom" in part.opts or not part.fstype:
            continue
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except (PermissionError, OSError):
            continue
        disks.append({"drive": part.mountpoint, "total": usage.total, "free": usage.free})
    memory = psutil.virtual_memory()
    battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    return {
        "at": time.time(),
        "cpu": cpu_total,
        "memory": {"total": memory.total, "used": memory.total - memory.available, "percent": memory.percent},
        "disks": disks,
        "battery": None if battery is None else {"percent": battery.percent, "plugged": bool(battery.power_plugged)},
        "apps": sorted(apps.values(), key=lambda entry: entry["memory"], reverse=True)[:30],
    }


def _app_memory(entry: dict[str, Any]) -> str:
    count = entry["processes"]
    return f"{entry['app']} {_gb(entry['memory'])}" + (f" across {count} processes" if count > 2 else "")


def describe_disks(snapshot: dict[str, Any]) -> str:
    disks = snapshot["disks"]
    if not disks:
        return "I couldn't read any drives."
    parts = []
    for disk in disks:
        share = disk["free"] / disk["total"] * 100 if disk["total"] else 0
        parts.append(f"{_drive(disk['drive'])} has {_gb(disk['free'])} free of {_gb(disk['total'])} ({share:.0f}%)")
    low = [
        disk for disk in disks if disk["total"] and (disk["free"] / disk["total"] < 0.1 or disk["free"] < 10 * 1024**3)
    ]
    note = " That is getting full; clearing Downloads or old installers would help." if low else ""
    return "; ".join(parts) + "." + note


def describe_memory(snapshot: dict[str, Any]) -> str:
    memory = snapshot["memory"]
    top = ", ".join(_app_memory(entry) for entry in snapshot["apps"][:4])
    return (
        f"Memory is {memory['percent']:.0f}% used ({_gb(memory['used'])} of {_gb(memory['total'])}). Most of it: {top}."
    )


def describe_cpu(snapshot: dict[str, Any]) -> str:
    busiest = sorted(snapshot["apps"], key=lambda entry: entry["cpu"], reverse=True)[:4]
    busy = [f"{entry['app']} {entry['cpu']:.0f}%" for entry in busiest if entry["cpu"] >= 1]
    if not busy:
        return f"The processor is at {snapshot['cpu']:.0f}%, and no app is using much of it."
    return f"The processor is at {snapshot['cpu']:.0f}%. Busiest: {', '.join(busy)}."


def describe_health(snapshot: dict[str, Any]) -> str:
    """Why the computer might be slow, in a few sentences."""
    memory = snapshot["memory"]
    findings = []
    culprit = None
    if memory["percent"] >= 85:
        culprit = snapshot["apps"][0] if snapshot["apps"] else None
        findings.append(
            f"Memory is nearly full ({memory['percent']:.0f}%, {_gb(memory['used'])} of {_gb(memory['total'])}), "
            "which is the most likely reason things feel slow"
        )
    if snapshot["cpu"] >= 80:
        busiest = max(snapshot["apps"], key=lambda entry: entry["cpu"], default=None)
        culprit = culprit or busiest
        findings.append(
            f"the processor is busy at {snapshot['cpu']:.0f}%"
            + (f", mostly {busiest['app']} ({busiest['cpu']:.0f}%)" if busiest and busiest["cpu"] >= 5 else "")
        )
    low = [disk for disk in snapshot["disks"] if disk["total"] and disk["free"] / disk["total"] < 0.05]
    if low:
        findings.append(f"{_drive(low[0]['drive'])} is almost full ({_gb(low[0]['free'])} free)")
    battery = snapshot.get("battery")
    if battery and not battery["plugged"] and battery["percent"] <= 20:
        findings.append(f"the battery is at {battery['percent']:.0f}% and may be saving power")
    top = ", ".join(_app_memory(entry) for entry in snapshot["apps"][:3])
    if not findings:
        return (
            f"Nothing looks overloaded: the processor is at {snapshot['cpu']:.0f}% and memory at "
            f"{memory['percent']:.0f}%. The biggest apps are {top}."
        )
    text = findings[0][0].upper() + findings[0][1:]
    if len(findings) > 1:
        text += "; " + "; ".join(findings[1:])
    text += f". The biggest apps are {top}."
    if culprit is not None:
        text += f' If you want, say "close {culprit["app"]}" and I\'ll ask before closing it.'
    return text


class SystemHealth:
    """Answers questions about how this computer is doing, from a fresh reading each time."""

    def __init__(self, read: Callable[[], dict[str, Any]] = read_snapshot) -> None:
        self._read = read

    async def snapshot(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._read)

    async def answer_async(self, text: str) -> str | None:
        normalized = _normalize(text)
        describe = None
        if SLOW_QUESTION.match(normalized):
            describe = describe_health
        elif MEMORY_QUESTION.match(normalized):
            describe = describe_memory
        elif CPU_QUESTION.match(normalized):
            describe = describe_cpu
        elif DISK_QUESTION.match(normalized):
            describe = describe_disks
        if describe is None:
            return None
        try:
            snapshot = await self.snapshot()
        except Exception as exc:
            return f"I couldn't read how this computer is doing: {exc}"
        return describe(snapshot)
