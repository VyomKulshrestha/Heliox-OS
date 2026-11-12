"""The learner's saved state survives a reboot, and an earlier version's key is moved (CI runners: ubuntu, macOS,
Windows). A real daemon, three starts:

1. a ledger of SEEDED events; the first start replays it in full and saves the learner's state; a clean stop;
2. the runtime folder is emptied, as Linux does at every reboot (and macOS may do to Library/Caches); the second
   start must load the saved state ("from_snapshot": True) instead of replaying, with the key still in the data
   folder's keys folder (owner-only on Linux and macOS);
3. the key is put back where an earlier version kept it (the runtime folder) and removed from the new place; the third
   start must move it back, remove the old file, and still load the saved state.

Run with the environment's Python. Heliox's folders go to a fresh folder through the XDG_* variables, the runtime
folder included, so emptying it touches nothing else. Prints CHECK lines; exit code 1 if one fails.
Off a CI runner it changes nothing and exits with 3 (ci_only.py).

What this cannot show: a real reboot, and a person's own data folder after months of use.
"""

from __future__ import annotations

import ast
import asyncio
import os
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path

from ci_only import require_runner

require_runner("starts real daemons and writes a ledger, a saved state and its key in a folder it makes")

HERE = Path(__file__).resolve().parent
FOLDERS = Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()) / "heliox-learner-key-check"
shutil.rmtree(FOLDERS, ignore_errors=True)
for variable in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_RUNTIME_DIR"):
    os.environ[variable] = str(FOLDERS / variable.lower())  # before pilot.config is imported, here and in the daemon
os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"  # the key file is the key
sys.path.insert(0, str(HERE))

import real_daemon  # noqa: E402

SEEDED = 5_000
results: list[tuple[bool, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    line = f"CHECK learner-key/{name}: {'PASS' if ok else 'FAIL'} - {detail}"
    print(line, flush=True)
    results.append((ok, line))
    return ok


def last_load() -> dict:
    """The daemon's own line: "Learner loaded after ready: {...}"."""
    for line in reversed(real_daemon.log_lines(("Learner loaded after ready:",), last=50)):
        try:
            return ast.literal_eval(line.split("Learner loaded after ready:", 1)[1].strip())
        except (ValueError, SyntaxError):
            continue
    return {}


async def one_start(label: str) -> dict:
    """Start, wait for the learner, stop cleanly; the learner's load summary."""
    from pilot.config import LOG_FILE

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")  # this start's lines only
    daemon = real_daemon.start()
    session = None
    try:
        session = await real_daemon.connect(240)
        ready, status = await session.wait_ready(240)
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            status = await session.call("startup_status", timeout=30)
            if (status.get("late_parts") or {}).get("online_learning") in ("ready", "failed"):
                break
            await asyncio.sleep(1)
        state = (status.get("late_parts") or {}).get("online_learning")
        print(f"INFO {label}: ready={ready}, learner {state}")
    finally:
        await real_daemon.stop(daemon, session)
    return last_load()


def owner_only(path: Path) -> str:
    if os.name != "posix":
        return "Windows: the profile folder's own permissions"
    return f"file {stat.filemode(path.stat().st_mode)}, folder {stat.filemode(path.parent.stat().st_mode)}"


async def run() -> int:
    import seed_ledger

    from pilot.config import DATA_DIR, RUNTIME_DIR
    from pilot.server import KEYS_FOLDER, LEARNER_SNAPSHOT_FILE, LEARNER_SNAPSHOT_KEY_FILE

    real_daemon.write_config()
    await seed_ledger.main(SEEDED, False)
    key, old_key, snapshot = (
        DATA_DIR / KEYS_FOLDER / LEARNER_SNAPSHOT_KEY_FILE,
        RUNTIME_DIR / LEARNER_SNAPSHOT_KEY_FILE,
        DATA_DIR / LEARNER_SNAPSHOT_FILE,
    )

    first = await one_start("first start")
    check(
        "first-start-saves",
        first.get("replayed", 0) >= SEEDED and snapshot.exists() and key.exists(),
        f"replayed {first.get('replayed')} events in full; saved state {'written' if snapshot.exists() else 'missing'}; "
        f"key at {key} ({owner_only(key) if key.exists() else 'missing'})",
    )
    if os.name == "posix" and key.exists():
        modes = (stat.S_IMODE(key.stat().st_mode), stat.S_IMODE(key.parent.stat().st_mode))
        check("owner-only", modes == (0o600, 0o700), f"key {oct(modes[0])}, its folder {oct(modes[1])}")

    shutil.rmtree(RUNTIME_DIR, ignore_errors=True)  # what a reboot does to /run/user/<uid>
    second = await one_start("after the runtime folder was emptied")
    check(
        "survives-a-reboot",
        second.get("from_snapshot") is True and second.get("replayed", SEEDED) < SEEDED,
        f"from the saved state: {second.get('from_snapshot')}, events read after it: {second.get('replayed')} "
        f"(refused: {second.get('refused') or 'nothing'})",
    )

    old_key.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(key), str(old_key))  # where an earlier version kept it
    third = await one_start("with the key where an earlier version kept it")
    moved = real_daemon.log_lines(("Learner key:",), last=5)
    check(
        "earlier-key-moved",
        key.exists() and not old_key.exists() and third.get("from_snapshot") is True,
        f"key in the data folder: {key.exists()}, old file left: {old_key.exists()}, from the saved state: "
        f"{third.get('from_snapshot')}; log: {moved[-1] if moved else 'no line'}",
    )
    return 0 if results and all(ok for ok, _line in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
