"""The Tauri CLI that builds every installer is the one tauri-app's manifest and lock file name, and it belongs with
the tauri crate the app is built against.

Until 13 November 2026 the release ran "npx @tauri-apps/cli" with nothing installed, so every release was built with
whichever CLI was newest that day, while tauri-app/package-lock.json, last touched in May, named no CLI at all. These
checks read the files only (no npm): the manifest names one exact version, the lock pins that same version, the
"tauri" script runs the installed binary, and the CLI's major.minor is the tauri crate's in Cargo.lock (Tauri
releases the tauri crate and the CLI together: tauri 2.11.2 and tauri-cli 2.11.2 both came out on 16 May 2026 and
both require tauri-utils ^2.9.2, the version Cargo.lock holds; the 2.12 CLI requires tauri-utils ~2.10).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "tauri-app"


def manifest() -> dict:
    return json.loads((APP / "package.json").read_text(encoding="utf-8"))


def lock() -> dict:
    return json.loads((APP / "package-lock.json").read_text(encoding="utf-8"))


def cargo_lock_version(name: str) -> str:
    text = (APP / "src-tauri" / "Cargo.lock").read_text(encoding="utf-8")
    found = re.search(rf'\[\[package\]\]\nname = "{re.escape(name)}"\nversion = "([^"]+)"', text)
    assert found, f"{name} is not in Cargo.lock"
    return found.group(1)


def test_the_manifest_names_one_exact_cli_version():
    declared = manifest()["devDependencies"]["@tauri-apps/cli"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", declared), f"a range lets the CLI move on its own: {declared!r}"


def test_the_lock_pins_the_version_the_manifest_names():
    declared = manifest()["devDependencies"]["@tauri-apps/cli"]
    packages = lock()["packages"]
    assert packages[""]["devDependencies"]["@tauri-apps/cli"] == declared
    assert packages["node_modules/@tauri-apps/cli"]["version"] == declared
    for path, entry in packages.items():
        if path.startswith("node_modules/@tauri-apps/cli-"):  # the per-system binaries move with it
            assert entry["version"] == declared, path


def test_the_tauri_script_runs_the_installed_cli():
    assert manifest()["scripts"]["tauri"] == "tauri"


def test_a_contributors_setup_installs_the_same_cli():
    """setup.ps1 (Windows contributors) installs the Rust tauri-cli at the version package.json names, exactly."""
    script = (APP.parent / "setup.ps1").read_text(encoding="utf-8")
    assert "devDependencies.'@tauri-apps/cli'" in script
    assert 'cargo install tauri-cli --version "=$pinned"' in script
    assert '--version "^2"' not in script


def test_the_cli_belongs_with_the_tauri_crate():
    cli = manifest()["devDependencies"]["@tauri-apps/cli"].split(".")
    crate = cargo_lock_version("tauri").split(".")
    assert cli[:2] == crate[:2], f"CLI {'.'.join(cli)} with the tauri crate {'.'.join(crate)}"
