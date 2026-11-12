"""A locked keychain under a real daemon (macOS CI runner). The keychain is one this check makes; the runner's login
keychain is never locked, read or written.

The check makes ``heliox-check.keychain-db`` in RUNNER_TEMP, makes it the user's default keychain and the only one
in the user's search list (so the daemon's keyring calls go to it and nowhere else), locks it, then starts the
daemon with the macOS keyring backend. Heliox's folders go to a fresh folder through the XDG_* variables; HOME stays
the runner's, because the keychain preferences belong to the real user. Checks:

- the daemon becomes ready and answers ping in under a second the whole time (no call into the keychain holds the
  event loop; each is bounded at 3 s in a thread of its own);
- saving an API key (``store_api_key``, a made-up test value) answers within 10 s and does not claim it was saved;
- the status (``autonomy_review_status`` credential_store) gives a plain sentence when the keychain did not answer;
- after the keychain is unlocked, a restarted daemon saves and lists the test key (the store works once it is open).

Before anything changes, the previous default and search list, and the commands that put them back, are written to
``heliox-check-keychain-restore.json`` in RUNNER_TEMP and printed, so the log of a killed run still says what to
restore. At the end they are put back and the test keychain deleted, whatever happened. It runs only on a CI runner
(``ci_only.require_runner``): anywhere else it changes nothing and exits with 3.

What this cannot show: the dialog a person sees for a locked login keychain on their own Mac, and what happens when
they answer it; a keychain locked by sleep or a screen lock while Heliox runs.
Prints CHECK lines; exit code 1 if one fails.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ci_only import require_runner

require_runner("makes a keychain and sets it as this user's default and only searched keychain")

HERE = Path(__file__).resolve().parent
KEYCHAIN = Path(os.environ.get("RUNNER_TEMP", tempfile.gettempdir())) / "heliox-check.keychain-db"
RESTORE = KEYCHAIN.with_name("heliox-check-keychain-restore.json")
PASSWORD = "heliox-check-only"  # a throwaway for a keychain this check makes and deletes
TEST_KEY = "sk-heliox-check-not-a-real-key"
FOLDERS = Path(tempfile.mkdtemp(prefix="heliox-keychain-check-"))
for variable in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_RUNTIME_DIR"):
    os.environ[variable] = str(FOLDERS / variable.lower())  # before pilot.config is imported, here and in the daemon
os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.macOS.Keyring"
sys.path.insert(0, str(HERE))

import real_daemon  # noqa: E402

results: list[tuple[bool, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    line = f"CHECK keychain/{name}: {'PASS' if ok else 'FAIL'} - {detail}"
    print(line, flush=True)
    results.append((ok, line))
    return ok


def security(*args: str) -> str:
    done = subprocess.run(["security", *args], capture_output=True, text=True, timeout=30, check=False)
    output = (done.stdout + done.stderr).strip()
    if done.returncode != 0:
        raise RuntimeError(f"security {' '.join(args[:2])} failed ({done.returncode}): {output}")
    return output


def keychain_list(output: str) -> list[str]:
    return [shlex.split(line.strip())[0] for line in output.splitlines() if line.strip()]


async def watch_ping(session, until: float, slowest: list[float]) -> None:
    while time.monotonic() < until:
        _reply, took = await session.timed("ping", timeout=30)
        slowest.append(took)
        await asyncio.sleep(0.25)


async def locked() -> None:
    real_daemon.write_config()
    daemon = real_daemon.start()
    session = None
    try:
        session = await real_daemon.connect(240)
        watcher = await real_daemon.connect(30)  # a second window: its pings must keep answering throughout
        times: list[float] = []
        ping_task = asyncio.create_task(watch_ping(watcher, time.monotonic() + 45, times))
        ready, status = await session.wait_ready(240)
        check("ready", ready, f"start-up with a locked keychain: {status.get('state')!r} {status.get('reason')!r}")
        began = time.monotonic()
        stored = await session.call("store_api_key", {"provider": "claude", "api_key": TEST_KEY}, timeout=60)
        took = time.monotonic() - began
        check(
            "save-answers",
            took <= 10 and stored.get("status") != "ok",
            f"store_api_key answered in {took:.1f} s: {str(stored)[:300]}",
        )
        await ping_task
        slowest = max(times) if times else None
        check(
            "loop-free",
            bool(times) and slowest is not None and slowest < 1.0,
            f"{len(times)} pings from a second connection over 45 s, slowest {slowest if slowest is None else round(slowest, 2)} s",
        )
        status = await session.call("autonomy_review_status", timeout=60)
        store = status.get("credential_store") or {}
        listed = await session.call("list_api_keys", timeout=30)
        # A keychain that did not answer: the status carries the sentence. One that refused at once: the reply does.
        error = stored.get("rpc_error") if isinstance(stored.get("rpc_error"), dict) else {}
        said = str(store.get("message") or stored.get("message") or error.get("message") or "")
        check(
            "says-so",
            bool(said) and "Traceback" not in said,
            f"what the person is told: {said[:300]!r}; credential_store state={store.get('state')!r}; "
            f"list_api_keys: {str(listed)[:200]}",
        )
        await watcher.close()
    finally:
        await real_daemon.stop(daemon, session)


async def unlocked() -> None:
    daemon = real_daemon.start()
    session = None
    try:
        session = await real_daemon.connect(240)
        ready, _status = await session.wait_ready(240)
        stored = await session.call("store_api_key", {"provider": "claude", "api_key": TEST_KEY}, timeout=60)
        listed = await session.call("list_api_keys", timeout=30)
        found = (
            subprocess.run(
                ["security", "find-generic-password", "-s", "pilot-ai-command-center", "-a", "claude", str(KEYCHAIN)],
                capture_output=True,
                timeout=30,
                check=False,
            ).returncode
            == 0
        )
        check(
            "works-when-unlocked",
            ready and stored.get("status") == "ok" and "claude" in (listed.get("providers") or []),
            f"store_api_key: {str(stored)[:160]}; listed: {listed.get('providers')!r}; in the test keychain: "
            f"{'yes' if found else 'no'}",
        )
        await session.call("delete_api_key", {"provider": "claude"}, timeout=30)
    finally:
        await real_daemon.stop(daemon, session)


async def run() -> int:
    if sys.platform != "darwin":
        print("CHECK keychain/macos-only: FAIL - this check makes a macOS keychain and runs on macOS only")
        return 1
    default_before = keychain_list(security("default-keychain", "-d", "user"))
    search_before = keychain_list(security("list-keychains", "-d", "user"))
    print(f"INFO before: default {default_before}, search list {search_before}")
    # Written before anything changes, so even the log of a killed run says what to put back, and how.
    RESTORE.write_text(
        json.dumps(
            {
                "default_keychain": default_before,
                "search_list": search_before,
                "restore": [
                    ["security", "list-keychains", "-d", "user", "-s", *search_before],
                    ["security", "default-keychain", "-d", "user", "-s", *default_before[:1]],
                    ["security", "delete-keychain", str(KEYCHAIN)],
                ],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"INFO the settings to restore are in {RESTORE}; if this run is killed, run:")
    for command in json.loads(RESTORE.read_text(encoding="utf-8"))["restore"]:
        print(f"INFO   {shlex.join(command)}", flush=True)
    try:
        if KEYCHAIN.exists():
            security("delete-keychain", str(KEYCHAIN))
        security("create-keychain", "-p", PASSWORD, str(KEYCHAIN))
        security("set-keychain-settings", str(KEYCHAIN))  # no automatic lock: only the check locks it
        security("list-keychains", "-d", "user", "-s", str(KEYCHAIN))
        security("default-keychain", "-d", "user", "-s", str(KEYCHAIN))
        security("lock-keychain", str(KEYCHAIN))
        print(f"INFO test keychain {KEYCHAIN} made, the only one searched, and locked")
        await locked()
        security("unlock-keychain", "-p", PASSWORD, str(KEYCHAIN))
        await unlocked()
    except Exception as error:
        check("ran", False, f"{type(error).__name__}: {error}")
        for line in real_daemon.log_lines(("credential store", "keyring", "Keychain", "Traceback")):
            print(f"DAEMON-LOG {line}")
    finally:
        try:
            if search_before:
                security("list-keychains", "-d", "user", "-s", *search_before)
            if default_before:
                security("default-keychain", "-d", "user", "-s", default_before[0])
            if KEYCHAIN.exists():
                security("delete-keychain", str(KEYCHAIN))
            print(
                f"INFO restored: default {keychain_list(security('default-keychain', '-d', 'user'))}, search list "
                f"{keychain_list(security('list-keychains', '-d', 'user'))}"
            )
        except Exception as error:
            check("restored", False, f"could not put the runner's keychain settings back: {error}")
    return 0 if results and all(ok for ok, _line in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
