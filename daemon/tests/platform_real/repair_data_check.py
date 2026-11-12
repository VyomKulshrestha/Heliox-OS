"""Settings and data survive a repair unchanged, for the CI jobs that repair a real install.

    python repair_data_check.py before STATE   # after setup, before the environment is broken
    python repair_data_check.py after STATE    # after the repair

``before`` adds a marker line to ``~/.config/heliox-os/config.toml`` (a TOML comment, so the daemon reads the
file as before; created if setup made none), writes a marker file in ``~/.local/share/heliox-os``, and records the
SHA-256 of every file under the settings folder (outside the parts setup rebuilds: env, python, tools, uv-cache,
setup, env.broken) and under the data folder. ``after`` checks every recorded file is still there, byte for byte.
The job stops the daemon before ``before`` and starts nothing until ``after`` has run, so nothing but the repair
could have changed them.

Each check prints one ``CHECK`` line; the exit code is the number of failed checks.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sys
from pathlib import Path

REBUILT = {"env", "python", "tools", "uv-cache", "setup", "env.broken"}
failures = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


def folders() -> tuple[Path, Path]:
    home = Path.home()
    return home / ".config" / "heliox-os", home / ".local" / "share" / "heliox-os"


def hashes() -> dict[str, str]:
    settings, data = folders()
    found: dict[str, str] = {}
    for root in (settings, data):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if root == settings and relative.parts and relative.parts[0] in REBUILT:
                continue
            if path.is_file() and not path.is_symlink():
                found[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


def before(state: Path) -> None:
    settings, data = folders()
    settings.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    marker = secrets.token_hex(8)
    config = settings / "config.toml"
    with config.open("a", encoding="utf-8") as file:
        file.write(f"\n# heliox-ci repair marker {marker}\n")
    (data / "ci-repair-marker.txt").write_text(f"{marker}\n", encoding="utf-8")
    recorded = hashes()
    state.write_text(json.dumps({"marker": marker, "files": recorded}), encoding="utf-8")
    print(f"recorded {len(recorded)} files under the settings and data folders")


def after(state: Path) -> None:
    saved = json.loads(state.read_text(encoding="utf-8"))
    settings, data = folders()
    marker = saved["marker"]
    config = settings / "config.toml"
    check(
        "repair/settings-kept",
        config.exists() and f"# heliox-ci repair marker {marker}" in config.read_text(encoding="utf-8"),
        str(config),
    )
    check(
        "repair/data-kept",
        (data / "ci-repair-marker.txt").read_text(encoding="utf-8").strip() == marker
        if (data / "ci-repair-marker.txt").exists()
        else False,
    )
    now = hashes()
    changed = [path for path, digest in saved["files"].items() if now.get(path) != digest]
    check(
        "repair/every-settings-and-data-file-unchanged", not changed, f"{len(saved['files'])} files; changed {changed}"
    )


def main(argv: list[str]) -> int:
    mode, state = argv[0], Path(argv[1])
    if mode == "before":
        before(state)
    else:
        after(state)
    return failures


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
