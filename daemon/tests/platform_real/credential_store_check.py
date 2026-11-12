"""Heliox's credential cleanup against the real credential store of a CI runner, checked with the system's own tool.

Run with the daemon environment's Python (the one ``keyring`` is installed in):

    python credential_store_check.py full          # write, remove with the daemon's cleanup, check none is left
    python credential_store_check.py write STATE   # write one test entry per Heliox service; STATE lists them
    python credential_store_check.py gone STATE    # check with the system's tool that none of them is left
    python credential_store_check.py kept STATE    # check that all of them are still there ("keep my data")
    python credential_store_check.py drop STATE    # remove this check's own entries with keyring, then "gone"

``gone``, ``kept`` and ``drop`` need only ``keyring``, so they run with the runner's Python once the environment
has been removed.

The entries are written with ``keyring`` itself, one under each of Heliox's four services, with an account name
made for this run. Before writing, the store is searched for any entry under those services (by service where the
system allows it, and by every account name Heliox knows); if one exists, nothing is written and the check fails,
so the cleanup can only ever meet entries this check created. The system tools: ``cmdkey /list`` on Windows,
``security find-generic-password`` on macOS (the login keychain), ``secret-tool lookup`` on Linux (the Secret
Service, a GNOME keyring unlocked for the session by ``with_keyring.sh``).

Each check prints one ``CHECK`` line; the exit code is the number of failed checks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

failures = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


def run(args: list[str]) -> tuple[int, str]:
    done = subprocess.run(args, capture_output=True, text=True, timeout=30, check=False)
    return done.returncode, done.stdout + done.stderr


def windows_targets(output: str) -> set[str]:
    """Target names in ``cmdkey /list`` output ("Target: LegacyGeneric:target=name" or "Target: name"). Pure."""
    targets = set()
    for line in output.splitlines():
        text = line.strip()
        if not text.casefold().startswith("target:"):
            continue
        target = text.split(":", 1)[1].strip()
        if "target=" in target:
            target = target.split("target=", 1)[1]
        targets.add(target.strip())
    return targets


def present(service: str, account: str) -> bool:
    """Whether the system's own tool sees the entry."""
    if sys.platform == "win32":
        _code, out = run(["cmdkey", "/list"])
        targets = windows_targets(out)
        return service in targets or f"{account}@{service}" in targets
    if sys.platform == "darwin":
        code, _out = run(["/usr/bin/security", "find-generic-password", "-s", service, "-a", account])
        return code == 0
    code, _out = run(["secret-tool", "lookup", "service", service, "username", account])
    return code == 0


def existing_entries() -> list[str]:
    """Every Heliox entry the store shows before this check writes anything."""
    import keyring

    from pilot.system import credential_cleanup as cleanup

    finder = {"win32": cleanup.windows_entries, "darwin": cleanup.macos_entries}.get(
        sys.platform, cleanup.linux_entries
    )
    found = []
    known = cleanup.known_accounts()
    for service in cleanup.services():
        try:
            found += [f"{service} / {account}" for account in finder(service)]
        except Exception as error:
            print(f"search of {service} by service failed: {type(error).__name__}: {error}")
        for account in known.get(service, []):
            if keyring.get_password(service, account) is not None:
                found.append(f"{service} / {account}")
    return sorted(set(found))


def write(state: Path) -> list[tuple[str, str]]:
    import keyring

    from pilot.system import credential_cleanup as cleanup

    print(f"keyring backend: {keyring.get_keyring()!r}")
    before = existing_entries()
    if not check("credentials/store-has-no-heliox-entries-before", not before, repr(before)):
        print("RESULT: entries this check did not create exist; nothing was written and nothing will be removed.")
        return []
    account = f"heliox-ci-{os.getpid()}"
    written = []
    for service in cleanup.services():
        try:
            keyring.set_password(service, account, f"test secret for {service}")
            written.append((service, account))
        except Exception as error:
            check(f"credentials/write {service}", False, f"{type(error).__name__}: {error}")
    state.write_text(json.dumps(written), encoding="utf-8")
    seen = [f"{s} / {a}" for s, a in written if present(s, a)]
    check(
        "credentials/system-tool-sees-the-test-entries",
        len(written) == len(cleanup.services()) and len(seen) == len(written),
        f"{len(seen)} of {len(written)} written: {seen}",
    )
    return written


def written_in(state: Path) -> list[tuple[str, str]]:
    return [(str(pair[0]), str(pair[1])) for pair in json.loads(state.read_text(encoding="utf-8"))]


def kept(state: Path) -> None:
    written = written_in(state)
    there = [f"{s} / {a}" for s, a in written if present(s, a)]
    check("credentials/kept-with-the-data", bool(written) and len(there) == len(written), f"{there}")


def drop(state: Path) -> None:
    """Remove only the entries this check wrote, then check none is left."""
    import keyring

    for service, account in written_in(state):
        try:
            keyring.delete_password(service, account)
        except Exception as error:
            print(f"drop {service} / {account}: {type(error).__name__}: {error}")
    gone(state)


def gone(state: Path) -> None:
    written = written_in(state)
    left = [f"{s} / {a}" for s, a in written if present(s, a)]
    check(
        "credentials/system-tool-finds-none-left",
        bool(written) and not left,
        f"{len(written)} written, left: {left}",
    )


def main(argv: list[str]) -> int:
    mode = argv[0] if argv else "full"
    if mode == "write":
        write(Path(argv[1]))
    elif mode == "gone":
        gone(Path(argv[1]))
    elif mode == "kept":
        kept(Path(argv[1]))
    elif mode == "drop":
        drop(Path(argv[1]))
    else:
        state = Path(os.environ.get("RUNNER_TEMP", ".")) / "heliox-credential-check.json"
        written = write(state)
        if written:
            code, out = run([sys.executable, "-m", "pilot.system.credential_cleanup", "remove"])
            report = {}
            for line in reversed(out.splitlines()):
                try:
                    report = json.loads(line)
                    break
                except ValueError:
                    continue
            print(f"cleanup exit {code}: {json.dumps(report)}")
            removed = set(report.get("removed", []))
            check(
                "credentials/cleanup-reports-each-test-entry-removed",
                all(f"{s} / {a}" in removed for s, a in written) and not report.get("remaining"),
                f"removed {sorted(removed)}, remaining {report.get('remaining')}",
            )
            gone(state)
    return failures


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
