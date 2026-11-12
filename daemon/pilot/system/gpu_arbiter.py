"""One GPU user at a time across Heliox's own processes, and a look at the machine before a model loads.

Why: on the owner's laptop the GPU's cooler fails under load. Past about 86 C CUDA kernels fault ("illegal memory
access", wrong results) and Windows resets the GPU, which kills every CUDA context on it at once. On 10 November
2026 the speech worker and Gnomon were on the card together and one reset took down both; a bitsandbytes fault
inside the daemon process took down the whole daemon.

Heliox runs on Windows, macOS and Linux, on machines with an NVIDIA card, Apple Silicon, an integrated GPU or only a
CPU. Three small probes, each of which can answer "cannot tell", describe the machine:

* NVIDIA: ``nvidia-smi`` (a separate program: reading it never creates a CUDA context in the caller) gives every
  card's free memory and, where the driver exposes it, its temperature.
* Apple Silicon: a Mac on arm64 has unified memory, so "is there room on the GPU" is "is there free system memory".
  There is no CUDA on a Mac.
* System memory: ``psutil`` (a declared dependency) on all three systems.

``decide`` is pure: given a reading, a component's needs, the limits and who holds the GPU, it says where the work
may run, or why not. ``GpuLease`` makes it one process at a time: an exclusive lock on a file (``msvcrt.locking`` on
Windows, ``fcntl.flock`` on macOS and Linux), taken by the process that holds the GPU context and released by the
operating system when that process ends, however it ends.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot.system.gpu_arbiter")

MIB = 1024**2
GIB = 1024**3
DEFAULT_MAX_TEMPERATURE_C = 83.0
DEFAULT_HEADROOM_MB = 512
DEFAULT_FAULT_LIMIT = 3
DEFAULT_FAULT_WINDOW_SECONDS = 600.0
READING_CACHE_SECONDS = 2.0
NVIDIA_SMI_TIMEOUT_SECONDS = 3.0
LOCK_FILE_NAME = "heliox-gpu.lock"
HOLDER_FILE_NAME = "heliox-gpu.holder.json"

COMPONENT_NAMES = {"gnomon": "Gnomon", "voice": "the voice", "ocr": "screen text reading"}

# "Do not start GPU work now". A component with a CPU path uses it; one without reports why it is unavailable.
BLOCKING_CODES = frozenset({"hot", "busy", "low_memory", "low_host_memory", "disabled", "unreadable", "unsupported"})

NO_CUDA_ON_MAC = "Not supported on this system: Gnomon's 4-bit model needs an NVIDIA graphics card with CUDA."
NO_NVIDIA_CARD = (
    "Not supported on this system: Gnomon's 4-bit model needs an NVIDIA graphics card with CUDA, and none was found."
)


# ── Readings ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GpuInfo:
    """One NVIDIA card as nvidia-smi reports it. None means the driver did not say."""

    index: int
    name: str = ""
    temperature_c: float | None = None
    free_bytes: int | None = None
    total_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "temperature_c": self.temperature_c,
            "free_mb": None if self.free_bytes is None else round(self.free_bytes / MIB),
            "total_mb": None if self.total_bytes is None else round(self.total_bytes / MIB),
        }


@dataclass(frozen=True)
class MachineReading:
    """What Heliox can tell about the machine. ``kind``: "nvidia", "apple" (Apple Silicon) or "cpu" (no NVIDIA card
    found: an integrated GPU or none). ``error`` says what could not be read ("cannot tell")."""

    kind: str
    gpus: tuple[GpuInfo, ...] = ()
    host_free_bytes: int | None = None
    host_total_bytes: int | None = None
    error: str = ""

    @property
    def unified_memory(self) -> bool:
        return self.kind == "apple"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "unified_memory": self.unified_memory,
            "host_free_mb": None if self.host_free_bytes is None else round(self.host_free_bytes / MIB),
            "host_total_mb": None if self.host_total_bytes is None else round(self.host_total_bytes / MIB),
            "error": self.error,
        }


def _number(text: str) -> float | None:
    try:
        return float(text.strip())
    except ValueError:
        return None  # "[N/A]", "[Not Supported]", ""


def parse_nvidia_smi(output: str) -> tuple[tuple[GpuInfo, ...], str]:
    """Read ``nvidia-smi --query-gpu=index,name,temperature.gpu,memory.free,memory.total
    --format=csv,noheader,nounits``: one line per card. Returns (cards, error)."""
    cards: list[GpuInfo] = []
    for row in output.splitlines():
        if not row.strip():
            continue
        parts = [part.strip() for part in row.split(",")]
        if len(parts) < 5:
            return tuple(cards), f"unexpected nvidia-smi output: {row[:120]}"
        index = _number(parts[0])
        free, total = _number(parts[3]), _number(parts[4])
        cards.append(
            GpuInfo(
                index=int(index) if index is not None else len(cards),
                name=parts[1],
                temperature_c=_number(parts[2]),
                free_bytes=None if free is None else int(free * MIB),
                total_bytes=None if total is None else int(total * MIB),
            )
        )
    return tuple(cards), "" if cards else "nvidia-smi listed no cards"


def probe_nvidia(
    *,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., Any] = subprocess.run,
    timeout: float = NVIDIA_SMI_TIMEOUT_SECONDS,
) -> tuple[tuple[GpuInfo, ...], str] | None:
    """None when there is no nvidia-smi (no NVIDIA driver); otherwise (cards, error)."""
    executable = which("nvidia-smi")
    if not executable:
        return None
    command = [
        executable,
        "--query-gpu=index,name,temperature.gpu,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    options: dict[str, Any] = {"capture_output": True, "text": True, "timeout": timeout}
    if os.name == "nt":
        options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = run(command, **options)
    except subprocess.TimeoutExpired:
        # During a GPU reset nvidia-smi can hang for seconds: the card is not usable right now.
        return (), f"nvidia-smi did not answer within {timeout:.0f} s"
    except OSError as error:
        return (), f"nvidia-smi could not run: {error}"
    if getattr(completed, "returncode", 0) != 0:
        detail = (getattr(completed, "stderr", "") or getattr(completed, "stdout", "") or "").strip()
        return (), f"nvidia-smi failed: {detail[:160] or completed.returncode}"
    return parse_nvidia_smi(getattr(completed, "stdout", "") or "")


def is_apple_silicon(system: str | None = None, machine: str | None = None) -> bool:
    system = system if system is not None else platform.system()
    machine = machine if machine is not None else platform.machine()
    return system == "Darwin" and machine.lower() in {"arm64", "aarch64"}


def probe_host_memory() -> tuple[int | None, int | None]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return int(memory.available), int(memory.total)
    except Exception:
        return None, None


def read_machine(
    *,
    nvidia: Callable[[], tuple[tuple[GpuInfo, ...], str] | None] = probe_nvidia,
    apple: Callable[[], bool] = is_apple_silicon,
    host: Callable[[], tuple[int | None, int | None]] = probe_host_memory,
) -> MachineReading:
    free, total = host()
    if apple():
        return MachineReading("apple", host_free_bytes=free, host_total_bytes=total)
    cards = nvidia()
    if cards is None:
        return MachineReading("cpu", host_free_bytes=free, host_total_bytes=total)
    gpus, error = cards
    return MachineReading("nvidia", gpus, host_free_bytes=free, host_total_bytes=total, error=error)


# ── The decision (pure) ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Needs:
    """What a component needs to start. ``gpu_bytes``: graphics memory for the model. ``host_bytes``: free system
    memory the process needs on top (Gnomon's loader asks for 2.5 GiB). ``requires_cuda``: no CPU or Metal path.
    ``default_device_only``: the component always uses the first visible card (it cannot be pointed at another)."""

    gpu_bytes: int = 0
    host_bytes: int = 0
    requires_cuda: bool = False
    default_device_only: bool = False


@dataclass(frozen=True)
class Limits:
    max_temperature_c: float = DEFAULT_MAX_TEMPERATURE_C
    headroom_bytes: int = DEFAULT_HEADROOM_MB * MIB
    visible: tuple[int, ...] | None = None  # CUDA_VISIBLE_DEVICES as indexes; None: every card


@dataclass(frozen=True)
class GpuDecision:
    code: str  # "ok" | "no_gpu" | one of BLOCKING_CODES
    reason: str = ""
    device: str = ""  # "cuda" | "mps" | "" (the component's own choice)
    gpu_index: int | None = None  # NVIDIA: the card to use (set CUDA_VISIBLE_DEVICES to it)
    kind: str = ""

    @property
    def use_gpu(self) -> bool:
        return self.code == "ok"

    @property
    def blocked(self) -> bool:
        return self.code in BLOCKING_CODES

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "reason": self.reason,
            "device": self.device,
            "gpu_index": self.gpu_index,
            "kind": self.kind,
        }


def _gb(value: int) -> str:
    return f"{value / GIB:.1f}"


def decide(
    component: str,
    needs: Needs,
    reading: MachineReading,
    limits: Limits = Limits(),
    *,
    holder: dict[str, Any] | None = None,
    disabled_reason: str = "",
) -> GpuDecision:
    """Where may ``component`` start its model now? Pure: every input is passed in."""
    kind = reading.kind
    if needs.requires_cuda and kind == "apple":
        return GpuDecision("unsupported", NO_CUDA_ON_MAC, kind=kind)
    if needs.requires_cuda and kind == "cpu" and not reading.error:
        # No nvidia-smi: the NVIDIA driver installs it with CUDA on Windows and Linux, so there is no CUDA here.
        # Said at once, instead of starting a worker that imports torch (about 1 GB on an 8 GB machine) to find out.
        return GpuDecision("unsupported", NO_NVIDIA_CARD, kind=kind)
    if disabled_reason:
        return GpuDecision("disabled", disabled_reason, kind=kind)
    host_need = needs.host_bytes + (needs.gpu_bytes if reading.unified_memory else 0)
    if host_need and reading.host_free_bytes is not None and reading.host_free_bytes < host_need:
        return GpuDecision(
            "low_host_memory",
            f"Needs {_gb(host_need)} GB of free memory, {_gb(reading.host_free_bytes)} GB free now.",
            kind=kind,
        )
    if holder is not None and holder.get("component") != component and kind in {"nvidia", "apple"}:
        who = COMPONENT_NAMES.get(str(holder.get("component")), "another Heliox process")
        return GpuDecision("busy", f"The graphics card is in use by {who}.", kind=kind)
    if kind == "apple":
        return GpuDecision("ok", device="mps", kind=kind)
    if kind != "nvidia":
        # No NVIDIA card found (integrated GPU or none, or no driver tools): nothing to protect or arbitrate.
        return GpuDecision("no_gpu", "No NVIDIA graphics card found.", kind=kind)
    if reading.error:
        return GpuDecision("unreadable", f"The graphics card did not report its state ({reading.error}).", kind=kind)
    cards = [gpu for gpu in reading.gpus if limits.visible is None or gpu.index in limits.visible]
    if limits.visible is not None:
        cards.sort(key=lambda gpu: limits.visible.index(gpu.index))  # type: ignore[union-attr]
    if needs.default_device_only:
        cards = cards[:1]
    if not cards:
        return GpuDecision("no_gpu", "No visible NVIDIA graphics card.", kind=kind)
    cool = [gpu for gpu in cards if gpu.temperature_c is None or gpu.temperature_c < limits.max_temperature_c]
    if not cool:
        coolest = min(cards, key=lambda gpu: gpu.temperature_c or 0.0)
        return GpuDecision(
            "hot",
            f"The graphics card is at {coolest.temperature_c:.0f} °C (limit {limits.max_temperature_c:.0f} °C).",
            kind=kind,
        )
    need = needs.gpu_bytes + limits.headroom_bytes if needs.gpu_bytes else 0
    roomy = [gpu for gpu in cool if not need or gpu.free_bytes is None or gpu.free_bytes >= need]
    if not roomy:
        best = max(cool, key=lambda gpu: gpu.free_bytes or 0)
        return GpuDecision(
            "low_memory",
            f"Needs {_gb(need)} GB of free graphics memory, {_gb(best.free_bytes or 0)} GB free now.",
            kind=kind,
        )
    # Several cards: the one with the most free memory (a card that did not say counts as least).
    chosen = max(roomy, key=lambda gpu: gpu.free_bytes if gpu.free_bytes is not None else -1)
    return GpuDecision("ok", device="cuda", gpu_index=chosen.index, kind=kind)


def visible_from_env(value: str | None) -> tuple[int, ...] | None:
    """CUDA_VISIBLE_DEVICES as card indexes; None when unset or not plain numbers (UUIDs: cannot tell)."""
    if value is None or not value.strip():
        return None
    try:
        return tuple(int(part) for part in value.split(",") if part.strip())
    except ValueError:
        return None


# ── The lease: one process at a time ─────────────────────────────────────────


def default_lock_dir() -> Path:
    override = os.environ.get("HELIOX_GPU_LOCK_DIR", "").strip()
    if override:
        return Path(override)
    from pilot.config import DATA_DIR

    return DATA_DIR / "runtime"


def _pid_alive(pid: int) -> bool:
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except Exception:
        return True  # cannot tell; the lock itself is the truth


class GpuLease:
    """An exclusive, non-blocking lock on a file, held by the process that owns the GPU context.

    Both locks belong to the open file, so two leases in one process exclude each other as two processes do, and
    both are released by the operating system when the process ends.
    """

    def __init__(self, lock_dir: Path | str | None = None) -> None:
        self.lock_dir = Path(lock_dir) if lock_dir is not None else default_lock_dir()
        self._fd: int | None = None

    @property
    def lock_path(self) -> Path:
        return self.lock_dir / LOCK_FILE_NAME

    @property
    def holder_path(self) -> Path:
        return self.lock_dir / HOLDER_FILE_NAME

    @property
    def held(self) -> bool:
        return self._fd is not None

    def try_acquire(self, component: str) -> bool:
        if self._fd is not None:
            return True
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.lock_path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            _lock_fd(fd)
        except OSError:
            os.close(fd)
            return False
        self._fd = fd
        with contextlib.suppress(OSError):
            self.holder_path.write_text(
                json.dumps({"component": component, "pid": os.getpid(), "since": time.time()}), encoding="utf-8"
            )
        return True

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        with contextlib.suppress(OSError):
            self.holder_path.unlink()
        try:
            _unlock_fd(fd)
        except OSError:
            pass
        finally:
            os.close(fd)


def _lock_fd(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_fd(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


def lease_holder(lock_dir: Path | str | None = None) -> dict[str, Any] | None:
    """Who holds the GPU now ({"component", "pid"}), or None. Takes and drops the lock to see."""
    probe = GpuLease(lock_dir)
    try:
        if probe.try_acquire("probe"):
            probe.release()
            return None
    except OSError:
        return None  # the lock file cannot be created: nothing else can hold it either
    try:
        holder = json.loads(probe.holder_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"component": "", "pid": None}
    if not isinstance(holder, dict):
        return {"component": "", "pid": None}
    pid = holder.get("pid")
    if isinstance(pid, int) and not _pid_alive(pid):
        return {"component": "", "pid": None}
    return {"component": str(holder.get("component") or ""), "pid": pid if isinstance(pid, int) else None}


# ── The arbiter: the daemon's view, with the session's fault count ───────────


@dataclass
class GpuFault:
    component: str
    message: str
    at: float  # wall clock, for the person
    monotonic: float  # for the window

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "message": self.message,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.at)),
        }


@dataclass
class GpuArbiter:
    """One per daemon (``arbiter()``); workers take the lease themselves."""

    read: Callable[[], MachineReading] = read_machine
    holder: Callable[[], dict[str, Any] | None] = lease_holder
    clock: Callable[[], float] = time.monotonic
    wall: Callable[[], float] = time.time
    environ: Any = field(default_factory=lambda: os.environ)
    max_temperature_c: float = DEFAULT_MAX_TEMPERATURE_C
    headroom_bytes: int = DEFAULT_HEADROOM_MB * MIB
    fault_limit: int = DEFAULT_FAULT_LIMIT
    fault_window_seconds: float = DEFAULT_FAULT_WINDOW_SECONDS
    cache_seconds: float = READING_CACHE_SECONDS
    _faults: deque[GpuFault] = field(default_factory=lambda: deque(maxlen=50))
    _disabled_reason: str = ""
    _cached: tuple[float, MachineReading] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def configure(self, settings: Any) -> None:
        """Apply ``[gpu]`` settings; anything missing or out of range keeps its default."""

        def number(name: str, default: float, low: float, high: float) -> float:
            try:
                value = float(getattr(settings, name, default))
            except (TypeError, ValueError):
                return default
            return value if low <= value <= high else default

        self.max_temperature_c = number("max_temperature_c", DEFAULT_MAX_TEMPERATURE_C, 40.0, 105.0)
        self.headroom_bytes = int(number("vram_headroom_mb", DEFAULT_HEADROOM_MB, 0, 16384) * MIB)
        self.fault_limit = int(number("fault_limit", DEFAULT_FAULT_LIMIT, 1, 100))
        self.fault_window_seconds = number("fault_window_seconds", DEFAULT_FAULT_WINDOW_SECONDS, 10.0, 86400.0)

    def reading(self, *, fresh: bool = False) -> MachineReading:
        now = self.clock()
        with self._lock:
            cached = self._cached
        if not fresh and cached is not None and now - cached[0] < self.cache_seconds:
            return cached[1]
        try:
            value = self.read()
        except Exception as error:  # a probe must never take the caller down
            value = MachineReading("cpu", error=f"{type(error).__name__}: {error}"[:200])
        with self._lock:
            self._cached = (now, value)
        return value

    def limits(self) -> Limits:
        return Limits(
            self.max_temperature_c,
            self.headroom_bytes,
            visible_from_env(self.environ.get("CUDA_VISIBLE_DEVICES")),
        )

    def check(self, component: str, needs: Needs = Needs(), *, fresh: bool = True) -> GpuDecision:
        """May ``component`` start now, and where? Takes no lock."""
        reading = self.reading(fresh=fresh)
        holder = self.holder() if reading.kind in {"nvidia", "apple"} else None
        return decide(component, needs, reading, self.limits(), holder=holder, disabled_reason=self._disabled_reason)

    def record_fault(self, component: str, message: str) -> bool:
        """A GPU process faulted (crash, CUDA error, hang). Returns True when the GPU is now off for the session."""
        now = self.clock()
        with self._lock:
            self._faults.append(GpuFault(component, message[:300], self.wall(), now))
            recent = [fault for fault in self._faults if now - fault.monotonic <= self.fault_window_seconds]
            if len(recent) >= self.fault_limit and not self._disabled_reason:
                minutes = self.fault_window_seconds / 60
                self._disabled_reason = (
                    f"Stopped using the graphics card for this session after {len(recent)} faults in "
                    f"{minutes:.0f} minutes. Restart Heliox to use it again."
                )
                logger.warning("GPU switched off for this session: %d faults in %.0f s", len(recent), minutes * 60)
            return bool(self._disabled_reason)

    @property
    def disabled(self) -> bool:
        return bool(self._disabled_reason)

    @property
    def disabled_reason(self) -> str:
        return self._disabled_reason

    def faults(self) -> list[GpuFault]:
        with self._lock:
            return list(self._faults)

    def last_fault(self, component: str | None = None) -> GpuFault | None:
        with self._lock:
            for fault in reversed(self._faults):
                if component is None or fault.component == component:
                    return fault
        return None

    def status(self) -> dict[str, Any]:
        reading = self.reading(fresh=False)
        last = self.last_fault()
        return {
            "machine": reading.to_dict(),
            "max_temperature_c": self.max_temperature_c,
            "holder": self.holder() if reading.kind in {"nvidia", "apple"} else None,
            "disabled": self.disabled,
            "disabled_reason": self._disabled_reason,
            "faults_this_session": len(self.faults()),
            "last_fault": last.to_dict() if last else None,
        }


_shared = GpuArbiter()


def arbiter() -> GpuArbiter:
    return _shared


VOICE_NEEDS = Needs(gpu_bytes=900 * MIB, default_device_only=True)


def voice_device(configured: str) -> tuple[str, bool]:
    """Where the local voice should start: (device, take_gpu_lease).

    "cpu" stays "cpu". On an NVIDIA machine "auto" and "cuda" move to the CPU when the card is hot, busy, short of
    memory, unreadable or switched off for the session. On Apple Silicon and machines without an NVIDIA card the
    setting is left as it is (on a Mac the CPU and Metal share the same memory, so moving saves nothing).
    """
    if configured == "cpu":
        return "cpu", False
    decision = arbiter().check("voice", VOICE_NEEDS)
    if decision.kind != "nvidia":
        return configured, False
    if decision.blocked:
        logger.info("Local voice starts on the CPU: %s", decision.reason)
        return "cpu", False
    return configured, True
