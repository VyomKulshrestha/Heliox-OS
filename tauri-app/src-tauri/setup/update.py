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

    if daemon_running():
        say("@@step update failed Heliox's background service is still running.")
        say("@@hint update Quit Heliox completely and start the update again.")
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
