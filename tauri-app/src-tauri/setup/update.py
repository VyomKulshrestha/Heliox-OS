"""Update Heliox's environment to the desktop app's version, keeping the person's groups, all or nothing.

The desktop app runs this with the *base* Python of Heliox's environment (the one named in ``env/pyvenv.cfg``),
never the environment's own interpreter, so the environment can be put back if the update fails. The daemon must
be stopped first (on Windows its files are locked while it runs).

    python update.py plan  --env DIR --version V [--source PATH] [--groups-file FILE] [--extras id=extra;...]
    python update.py apply --env DIR --version V [--source PATH] [--groups-file FILE] [--extras id=extra;...]

``plan`` changes nothing: it prints the installed and target versions, the groups that will be kept and each
package that will change, with its download size where the index says, as ``@@`` lines like the setup scripts.

``apply``:

1. takes a snapshot of the environment as hard links (``env.previous``): no data is copied, and because pip
   replaces a file by removing it and writing a new one, the snapshot keeps the old files intact;
2. upgrades in place with ``--upgrade-strategy only-if-needed``, so packages that did not change are not touched;
3. checks that the daemon imports and reports the new version;
4. on success deletes the snapshot and records the version and groups in the groups file; on any failure puts the
   snapshot back, so the environment is exactly as it was, and says why.

Groups come from the groups file the first-run setup writes; for an install made before it existed (the old
desktop installed ``pilot-daemon[all]``), they are inferred from what is installed. The decisions are pure
functions tested by ``daemon/tests/test_update_path.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

DAEMON = "pilot-daemon"

# Distributions that show a group is installed, for installs made before groups were recorded.
GROUP_MARKERS: dict[str, tuple[str, ...]] = {
    "voice": ("faster-whisper", "kokoro", "openai-whisper", "pocket-tts"),
    "screen": ("easyocr",),
    "browser": ("playwright",),
    "local": ("llama-cpp-python",),
    "platform": ("jeepney", "pyobjc-framework-applicationservices"),
    "more": ("chromadb", "paramiko", "wasmtime", "python-docx"),
}

DEFAULT_EXTRAS = {
    "voice": "voice",
    "screen": "vision",
    "browser": "browser",
    "local": "llamacpp",
    "more": "full,automation,documents,supervision,network,ssh,wasm",
}


def say(line: str) -> None:
    print(line, flush=True)


def normalise(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def infer_groups(installed: dict[str, str]) -> list[str]:
    """The groups whose marker packages are installed (``installed`` maps normalised names to versions)."""
    return [
        group
        for group, markers in GROUP_MARKERS.items()
        if any(normalise(m) in installed for m in markers)
    ]


def parse_extras(text: str, platform: str = sys.platform) -> dict[str, str]:
    extras = dict(DEFAULT_EXTRAS)
    extras["platform"] = (
        "macos" if platform == "darwin" else "linux" if platform.startswith("linux") else ""
    )
    for pair in filter(None, (text or "").split(";")):
        key, _, value = pair.partition("=")
        extras[key.strip()] = value.strip()
    return extras


def requirement(
    version: str, groups: list[str], extras: dict[str, str], source: str | None = None
) -> str:
    """The one requirement that brings the daemon and every kept group to ``version``."""
    wanted: list[str] = []
    for group in groups:
        for extra in (extras.get(group) or "").split(","):
            if extra and extra not in wanted:
                wanted.append(extra)
    suffix = f"[{','.join(wanted)}]" if wanted else ""
    return f"{source}{suffix}" if source else f"{DAEMON}{suffix}=={version}"


def changes_from_report(report: dict, installed: dict[str, str]) -> list[dict]:
    """Packages pip would install or upgrade, from ``pip install --dry-run --report``, against what is there."""
    changes = []
    for item in report.get("install", []):
        meta = item.get("metadata", {})
        name = normalise(str(meta.get("name", "")))
        new = str(meta.get("version", ""))
        old = installed.get(name, "")
        if old == new and not item.get("is_direct"):
            continue
        changes.append(
            {
                "name": name,
                "from": old,
                "to": new,
                "url": (item.get("download_info") or {}).get("url", ""),
            }
        )
    return changes


def version_tuple(text: str) -> tuple:
    parts = []
    for piece in text.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def needs_update(installed_version: str, app_version: str) -> bool:
    return not installed_version or version_tuple(installed_version) < version_tuple(app_version)


def env_python(env: Path) -> Path:
    return env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(args: list[str], echo: bool = True) -> tuple[int, str]:
    process = subprocess.run(args, capture_output=True, text=True, check=False)
    output = (process.stdout or "") + (process.stderr or "")
    if echo:
        for line in output.splitlines()[-200:]:
            say(line)
    return process.returncode, output


def installed_packages(env: Path) -> dict[str, str]:
    code, output = run(
        [
            str(env_python(env)),
            "-m",
            "pip",
            "list",
            "--format",
            "json",
            "--disable-pip-version-check",
        ],
        False,
    )
    if code != 0:
        return {}
    try:
        return {normalise(p["name"]): p["version"] for p in json.loads(output[output.index("[") :])}
    except (ValueError, KeyError):
        return {}


# Run with the environment's own Python (it has websockets and pilot.config) before the update touches it: asks the
# daemon on the port to stop with this installation's window token and prints one line of JSON with the outcome
# ("stopping", "not_heliox", "cannot_stop" or "no_answer"). It never ends a process itself.
STOP_HELPER = r'''
import asyncio
import json
import sys

PORT = 8785


async def ask(port: int = PORT, timeout: float = 10.0) -> dict[str, str]:
    import websockets

    from pilot.config import RUNTIME_DIR

    try:
        token = (RUNTIME_DIR / "auth_token").read_text(encoding="utf-8").strip()
    except OSError as error:
        return {"outcome": "not_heliox", "detail": f"this installation has no window token ({error})"}

    async def call(ws, method: str, params: dict, request_id: int) -> dict:
        await ws.send(json.dumps({"jsonrpc": "2.0", "method": method, "params": params, "id": request_id}))
        while True:
            message = json.loads(await ws.recv())
            if message.get("id") == request_id:
                return message

    try:
        async with asyncio.timeout(timeout):
            async with websockets.connect(f"ws://127.0.0.1:{port}", open_timeout=timeout) as ws:
                auth = await call(ws, "auth", {"token": token}, 1)
                if "error" in auth:
                    return {"outcome": "not_heliox", "detail": "it did not accept this installation's token"}
                reply = await call(ws, "daemon_stop", {"reason": "update"}, 2)
                if "error" in reply:
                    return {"outcome": "cannot_stop", "detail": str(reply["error"].get("message", ""))[:200]}
                return {"outcome": "stopping"}
    except TimeoutError:
        return {"outcome": "no_answer", "detail": f"no answer within {timeout:g} s"}
    except (OSError, ValueError, websockets.exceptions.WebSocketException) as error:
        return {"outcome": "not_heliox", "detail": f"{type(error).__name__}: {str(error)[:160]}"}


if __name__ == "__main__":
    print(json.dumps(asyncio.run(ask(int(sys.argv[1]) if len(sys.argv) > 1 else PORT))))
'''
STOP_WAIT_SECONDS = 30.0


def ask_daemon_to_stop(python: Path, port: int = 8785) -> dict[str, str]:
    code, output = run([str(python), "-c", STOP_HELPER, str(port)], echo=False)
    for line in reversed(output.splitlines()):
        try:
            found = json.loads(line)
        except ValueError:
            continue
        if isinstance(found, dict) and "outcome" in found:
            return {"outcome": str(found["outcome"]), "detail": str(found.get("detail", ""))}
    return {"outcome": "no_answer", "detail": f"the check did not finish (exit {code})"}


def wait_until_free(seconds: float = STOP_WAIT_SECONDS, port: int = 8785, sleep=None, clock=None) -> bool:
    import time

    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    deadline = clock() + seconds
    while daemon_running(port):
        if clock() >= deadline:
            return False
        sleep(0.5)
    return True


def parse_port_holder(text: str, port: int, platform: str = sys.platform) -> tuple[str, str] | None:
    """(program, process id) listening on ``port``, from ``netstat -ano`` and ``tasklist /FO CSV`` output joined on
    Windows, ``lsof -nP -iTCP:<port> -sTCP:LISTEN`` on macOS, or ``ss -ltnpH sport = :<port>`` on Linux."""
    import re

    if platform == "win32":
        pid = ""
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0] == "TCP" and parts[1].endswith(f":{port}") and parts[3] == "LISTENING":
                pid = parts[4]
                break
        if not pid:
            return None
        for line in text.splitlines():
            match = re.match(r'^"([^"]+)","(\d+)"', line.strip())
            if match and match.group(2) == pid:
                return match.group(1), pid
        return "an unknown program", pid
    if platform == "darwin":
        for line in text.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return parts[0], parts[1]
        return None
    match = re.search(r'users:\(\("([^"]+)",pid=(\d+)', text)
    return (match.group(1), match.group(2)) if match else None


def port_holder(port: int = 8785) -> tuple[str, str] | None:
    """Which program listens on the port, read with the system's own tools; None when that cannot be found."""
    try:
        if sys.platform == "win32":
            netstat = run(["netstat", "-ano", "-p", "TCP"], echo=False)[1]
            found = parse_port_holder(netstat, port, "win32")
            if found is None:
                return None
            tasks = run(["tasklist", "/FI", f"PID eq {found[1]}", "/FO", "CSV", "/NH"], echo=False)[1]
            return parse_port_holder(netstat + "\n" + tasks, port, "win32")
        if sys.platform == "darwin":
            listing = run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], echo=False)[1]
            return parse_port_holder(listing, port, "darwin")
        return parse_port_holder(run(["ss", "-ltnpH", f"sport = :{port}"], echo=False)[1], port, "linux")
    except OSError:
        return None


def how_to_stop(holder: tuple[str, str] | None, platform: str = sys.platform, port: int = 8785) -> str:
    """What holds the port and how to stop it on this system; the update never ends a process it did not start."""
    if holder is None:
        what = f"Something on this computer is using port {port}, and the update could not tell what."
        which = "it"
    else:
        what = f"Port {port} is held by {holder[0]} (process {holder[1]})."
        which = f"process {holder[1]}"
    if platform == "win32":
        how = f"Open Task Manager (Ctrl+Shift+Esc), go to Details, select {which} and choose End task."
    elif platform == "darwin":
        how = f"Open Activity Monitor, find {which} and choose Quit."
    elif holder is not None:
        how = f"Stop it from your system monitor, or run: kill {holder[1]}"
    else:
        how = "Stop it from your system monitor."
    return f"{what} {how} Then start the update again."


def make_way(python: Path, port: int = 8785, ask=None, wait=None, holder=None, running=None) -> tuple[bool, str]:
    """Before updating: if Heliox's own daemon still runs (for example one the app only borrowed, so quitting the
    app would not stop it), ask it to stop through its RPC and wait, bounded. Otherwise say what holds the port and
    how to stop it. Never ends a process. Returns (go ahead, the hint when not)."""
    ask = ask or ask_daemon_to_stop
    wait = wait or wait_until_free
    holder = holder or port_holder
    running = running or daemon_running
    if not running(port):
        return True, ""
    asked = ask(python, port)
    if asked["outcome"] == "stopping":
        say("@@step update running Asking Heliox's background service to stop for the update")
        if wait(port=port):
            return True, ""
        return False, (
            f"Heliox's background service said it would stop but was still running after {STOP_WAIT_SECONDS:.0f} s. "
            + how_to_stop(holder(port), port=port)
        )
    if asked["outcome"] == "cannot_stop":
        return False, (
            "Heliox's background service is from an older version that cannot be asked to stop. "
            + how_to_stop(holder(port), port=port)
        )
    return False, how_to_stop(holder(port), port=port)


def daemon_running(port: int = 8785) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def snapshot(env: Path, backup: Path) -> None:
    """A copy of ``env`` made of hard links (no data copied); a full copy where links are not possible."""
    if backup.exists():
        shutil.rmtree(backup)

    def link(src: str, dst: str) -> None:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    shutil.copytree(env, backup, symlinks=True, copy_function=link)


def restore(env: Path, backup: Path) -> None:
    if env.exists():
        shutil.rmtree(env, ignore_errors=True)
    os.replace(backup, env)


def head_size(url: str) -> int:
    if not url.startswith("https://"):
        return 0
    try:
        request = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": "heliox-update"}
        )
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - https only, index URLs
            return int(response.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def read_groups(path: Path | None) -> list[str] | None:
    if not path or not path.exists():
        return None
    try:
        groups = json.loads(path.read_text(encoding="utf-8")).get("groups")
        return [str(g) for g in groups] if isinstance(groups, list) else None
    except (OSError, ValueError):
        return None


def write_groups(path: Path | None, version: str, groups: list[str]) -> None:
    if not path:
        return
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
    data.update({"version": version, "groups": groups})
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["plan", "apply"])
    parser.add_argument("--env", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source")
    parser.add_argument("--groups-file")
    parser.add_argument("--extras", default="")
    args = parser.parse_args(argv)
    env = Path(args.env)
    groups_file = Path(args.groups_file) if args.groups_file else None
    python = env_python(env)

    if not python.exists() or run([str(python), "-c", "import sys"], False)[0] != 0:
        say(
            "@@step update failed Heliox's environment does not start: its Python is missing or broken."
        )
        say(
            "@@hint update Use Repair in Settings: it builds a fresh environment with the same parts."
        )
        return 1
    installed = installed_packages(env)
    current = installed.get(DAEMON, "")
    recorded = read_groups(groups_file)
    groups = recorded if recorded is not None else infer_groups(installed)
    extras = parse_extras(args.extras)
    target = requirement(args.version, groups, extras, args.source)
    say(
        f"@@update installed {current or 'unknown'} target {args.version} groups {','.join(groups) or '-'}"
    )
    if not needs_update(current, args.version) and not args.source:
        say("@@step update skipped Heliox is already up to date.")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "report.json"
        code, _ = run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--dry-run",
                "--quiet",
                "--disable-pip-version-check",
                "--upgrade-strategy",
                "only-if-needed",
                "--report",
                str(report_path),
                target,
            ],
            echo=args.mode == "plan",
        )
        if code != 0 or not report_path.exists():
            say("@@step update failed Could not work out what the update changes.")
            say("@@hint update Check the internet connection and try again.")
            return 1
        changes = changes_from_report(
            json.loads(report_path.read_text(encoding="utf-8")), installed
        )
    total = 0
    for change in changes:
        size = head_size(change["url"]) if args.mode == "plan" else 0
        total += size
        say(f"@@change {change['name']} {change['from'] or '-'} {change['to']} {size}")
    say(f"@@update changes {len(changes)} bytes {total}")
    if args.mode == "plan":
        return 0

    go_ahead, hint = make_way(python)
    if not go_ahead:
        say("@@step update failed Heliox's background service is still running.")
        say(f"@@hint update {hint}")
        return 1
    backup = env.with_name(env.name + ".previous")
    say("@@step update running Keeping a copy of the current environment")
    try:
        snapshot(env, backup)
    except OSError as error:
        say(
            f"@@step update failed Could not keep a copy of the environment ({error}); nothing was changed."
        )
        say("@@hint update Free some disk space and try again.")
        return 1
    say(f"@@step update running Updating {len(changes)} packages")
    code, _ = run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--progress-bar",
            "off",
            "--upgrade-strategy",
            "only-if-needed",
            target,
        ]
    )
    check = [
        str(python),
        "-c",
        "import importlib.metadata as m, pilot.server; print(m.version('pilot-daemon'))",
    ]
    verified = code == 0 and run(check, False)[0] == 0
    if not verified:
        restore(env, backup)
        say(
            "@@step update failed The update did not install cleanly, so Heliox put the previous version back."
        )
        say(
            "@@hint update Nothing changed. Check the internet connection and try again; the details show pip's output."
        )
        return 1
    shutil.rmtree(backup, ignore_errors=True)
    write_groups(groups_file, args.version, groups)
    say(f"@@step update done Updated to {args.version}, keeping {', '.join(groups) or 'the core'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
