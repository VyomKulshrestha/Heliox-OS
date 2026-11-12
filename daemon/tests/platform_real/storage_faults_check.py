"""A real full disk and a real read-only data folder under a running daemon (Linux CI runner, passwordless sudo).

Heliox's data folder (XDG_DATA_HOME) is put on a file system made for the check, three times, each with a fresh
daemon:

1. read-only at start: a tmpfs mounted read-only before the daemon starts. The daemon must listen and answer, say
   why it cannot start fully (``autonomy_review_status`` storage, ``startup_status`` waiting_for_storage), and start
   fully by itself within its 30 s recheck once the tmpfs is remounted read-write; then a note is saved.
2. full while running: a small tmpfs, the daemon ready, then a file fills it to the last byte. "note that ..." must
   be answered with the full-disk sentence (not passed on as if unheard), the daemon must keep answering, and once
   the file is removed the next note must be saved and the problem cleared from the status.
3. remounted read-only while running: a tmpfs cannot be remounted read-only while the daemon holds its databases
   open for writing (the kernel refuses with "busy"), so this case uses a small ext4 image mounted with
   errors=remount-ro, and ext4's own ``trigger_fs_error`` makes the kernel remount it read-only, as it does when a
   drive fails. The note must be answered with the read-only sentence and the daemon must keep answering.

Run with the environment's Python and a fresh HOME, from the repository root:

    python daemon/tests/platform_real/storage_faults_check.py

What this cannot show: a disk filled slowly by other programs over days, real hardware failing, network or removable
drives, recovery after a kernel remount (that needs the drive unmounted and checked, which a person does), the
window drawing the sentence, and Windows or macOS (no tmpfs; their disk-full errors are covered by fakes).
Prints CHECK lines; exit code 1 if one fails.
Off a CI runner it changes nothing and exits with 3 (ci_only.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import subprocess
import sys
import time
from pathlib import Path

from ci_only import require_runner

require_runner("mounts file systems with sudo and makes the kernel remount one read-only")

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "heliox-storage-check"
MOUNT = BASE / "data"
IMAGE = BASE / "data.ext4"
RECHECK = 30.0  # STORAGE_RECHECK_SECONDS in pilot.server
MARGIN = 15.0
FULL_WORDS = "The disk is full"
READ_ONLY_WORDS = "cannot be written to"

os.environ["XDG_DATA_HOME"] = str(MOUNT)  # before pilot.config is imported, here and in the daemon
sys.path.insert(0, str(HERE))

import real_daemon  # noqa: E402

results: list[tuple[bool, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    line = f"CHECK storage/{name}: {'PASS' if ok else 'FAIL'} - {detail}"
    print(line, flush=True)
    results.append((ok, line))
    return ok


def sudo(*args: str) -> str:
    done = subprocess.run(["sudo", *args], capture_output=True, text=True, timeout=60, check=False)
    output = (done.stdout + done.stderr).strip()
    if done.returncode != 0:
        raise RuntimeError(f"sudo {' '.join(args)} failed ({done.returncode}): {output}")
    return output


def ids() -> str:
    return f"uid={os.getuid()},gid={os.getgid()}"


def mount_tmpfs(size: str, read_only: bool = False) -> None:
    MOUNT.mkdir(parents=True, exist_ok=True)
    sudo("mount", "-t", "tmpfs", "-o", f"size={size},mode=0700,{ids()}", "tmpfs", str(MOUNT))
    if read_only:
        sudo("mount", "-o", "remount,ro", str(MOUNT))


def unmount() -> None:
    with contextlib.suppress(RuntimeError):
        sudo("umount", "-l", str(MOUNT))


def options() -> str:
    done = subprocess.run(["findmnt", "-no", "OPTIONS", str(MOUNT)], capture_output=True, text=True, check=False)
    return done.stdout.strip()


def fill() -> tuple[Path, int]:
    """Write a file until the file system refuses (ENOSPC), down to single bytes."""
    filler = MOUNT / "filler.bin"
    written = 0
    with open(filler, "wb", buffering=0) as handle:
        for chunk in (1 << 20, 1 << 12, 1):
            block = b"\0" * chunk
            while True:
                try:
                    handle.write(block)
                    written += chunk
                except OSError as error:
                    if error.errno != errno.ENOSPC:
                        raise
                    break
    return filler, written


async def fresh_daemon(limit: float = 240.0):
    real_daemon.write_config()
    daemon = real_daemon.start()
    session = await real_daemon.connect(limit)
    return daemon, session


async def storage_status(session) -> dict | None:
    status = await session.call("autonomy_review_status", timeout=60)
    return status.get("storage")


async def read_only_at_start() -> None:
    mount_tmpfs("16m", read_only=True)
    daemon = session = None
    try:
        print(f"INFO data folder on a read-only tmpfs: {options()}")
        daemon, session = await fresh_daemon()
        pong, took = await session.timed("ping")
        check("read-only-at-start/answers", "rpc_error" not in pong, f"ping answered in {took:.2f} s: {pong}")
        await asyncio.sleep(2)
        # Only ping, health, ready, startup_status and daemon_stop answer before start-up ends; the reason is here.
        startup = await session.call("startup_status")
        check(
            "read-only-at-start/says-why",
            startup.get("state") == "waiting_for_storage" and READ_ONLY_WORDS in str(startup.get("reason")),
            f"startup state={startup.get('state')!r}, reason={startup.get('reason')!r}",
        )
        sudo("mount", "-o", "remount,rw", str(MOUNT))
        began = time.monotonic()
        ready, status = await session.wait_ready(RECHECK + MARGIN + 60)
        took = time.monotonic() - began
        check(
            "read-only-at-start/starts-when-writable",
            ready and took <= RECHECK + MARGIN,
            f"{'ready' if ready else 'not ready'} {took:.1f} s after the folder became writable (rechecks every "
            f"{RECHECK:g} s): {status.get('state')!r} {status.get('stage')!r}",
        )
        said = await session.ask("note that the storage check can save")
        check("read-only-at-start/saves-after", said.strip().startswith("Noted"), f"a note afterwards: {said!r}")
    finally:
        if daemon is not None:
            await real_daemon.stop(daemon, session)
        unmount()


async def full_while_running() -> None:
    mount_tmpfs("48m")
    daemon = session = None
    filler = None
    try:
        daemon, session = await fresh_daemon()
        ready, status = await session.wait_ready(240)
        if not check("full/ready-first", ready, f"start-up on a writable tmpfs: {status.get('state')!r}"):
            return
        said = await session.ask("note that the disk has room")
        print(f"INFO a note before the disk is full: {said!r}")
        filler, written = fill()
        print(f"INFO filled the tmpfs with {written / 2**20:.1f} MB; {options()}")
        began = time.monotonic()
        said = await session.ask("note that the disk is full now")
        took = time.monotonic() - began
        problem = await storage_status(session)
        check(
            "full/says-so",
            FULL_WORDS in said and (problem or {}).get("kind") == "full",
            f"answered in {took:.1f} s: {said!r}; status storage={problem!r}",
        )
        pong, took = await session.timed("ping")
        sizes, sizes_took = await session.timed("storage_sizes")
        check(
            "full/keeps-answering",
            "rpc_error" not in pong and daemon.poll() is None,
            f"ping {took:.2f} s, storage_sizes {sizes_took:.2f} s ({str(sizes)[:160]}); daemon running: "
            f"{daemon.poll() is None}",
        )
        filler.unlink()
        filler = None
        said = await session.ask("note that the disk has room again")
        notes = (await session.call("notes_list")).get("notes") or []
        texts = [note.get("text") for note in notes]
        check(
            "full/saves-again",
            said.strip().startswith("Noted")
            and "the disk has room again" in texts
            and "the disk is full now" not in texts,
            f"after freeing space: {said!r}; notes kept: {texts}",
        )
        cleared = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            cleared = await storage_status(session)
            if cleared is None:
                break
            await asyncio.sleep(2)
        check(
            "full/problem-cleared",
            cleared is None,
            "the status no longer reports a storage problem"
            if cleared is None
            else f"30 s after a successful save the status still says {cleared!r}",
        )
    finally:
        if filler is not None:
            with contextlib.suppress(OSError):
                filler.unlink()
        if daemon is not None:
            await real_daemon.stop(daemon, session)
        unmount()


async def remounted_read_only() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    MOUNT.mkdir(parents=True, exist_ok=True)
    subprocess.run(["truncate", "-s", "96M", str(IMAGE)], check=True)
    subprocess.run(["mkfs.ext4", "-q", "-F", str(IMAGE)], check=True)
    sudo("mount", "-o", "loop,errors=remount-ro", str(IMAGE), str(MOUNT))
    sudo("chown", f"{os.getuid()}:{os.getgid()}", str(MOUNT))
    daemon = session = None
    try:
        daemon, session = await fresh_daemon()
        ready, status = await session.wait_ready(240)
        if not check("remounted-read-only/ready-first", ready, f"start-up on ext4: {status.get('state')!r}"):
            return
        print(f"INFO a note before: {await session.ask('note that the drive works')!r}")
        device = subprocess.run(
            ["findmnt", "-no", "SOURCE", str(MOUNT)], capture_output=True, text=True, check=True
        ).stdout.strip()
        trigger = Path("/sys/fs/ext4") / Path(device).name / "trigger_fs_error"
        subprocess.run(
            ["sudo", "tee", str(trigger)], input="heliox storage check\n", text=True, capture_output=True, check=True
        )
        await asyncio.sleep(1)
        now = options()
        if not check(
            "remounted-read-only/kernel-remounted", now.split(",")[0] == "ro", f"{device} after the error: {now}"
        ):
            return
        began = time.monotonic()
        said = await session.ask("note that the drive failed")
        took = time.monotonic() - began
        problem = await storage_status(session)
        check(
            "remounted-read-only/says-so",
            READ_ONLY_WORDS in said and (problem or {}).get("kind") == "read_only",
            f"answered in {took:.1f} s: {said!r}; status storage={problem!r}",
        )
        pong, took = await session.timed("ping")
        check(
            "remounted-read-only/keeps-answering",
            "rpc_error" not in pong and daemon.poll() is None,
            f"ping {took:.2f} s; daemon running: {daemon.poll() is None}",
        )
    finally:
        if daemon is not None:
            await real_daemon.stop(daemon, session)
        unmount()
        with contextlib.suppress(OSError):
            IMAGE.unlink()


async def run() -> int:
    if not sys.platform.startswith("linux"):
        print("CHECK storage/linux-only: FAIL - this check mounts file systems and runs on Linux only")
        return 1
    for case in (read_only_at_start, full_while_running, remounted_read_only):
        print(f"INFO --- {case.__name__.replace('_', ' ')} ---", flush=True)
        try:
            await case()
        except Exception as error:
            check(f"{case.__name__.replace('_', '-')}/ran", False, f"{type(error).__name__}: {error}")
            for line in real_daemon.log_lines(("Storage problem", "cannot write", "Traceback", "Error")):
                print(f"DAEMON-LOG {line}")
    return 0 if results and all(ok for ok, _line in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
