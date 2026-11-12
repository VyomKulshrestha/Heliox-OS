"""The update path from an older daemon to the app's version: groups kept, only changed packages, all or nothing."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "heliox_update", ROOT / "tauri-app" / "src-tauri" / "setup" / "update.py"
)
update = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(update)


def test_groups_are_inferred_for_installs_made_before_they_were_recorded():
    old_all = {
        "pilot-daemon": "0.15.0",
        "faster-whisper": "1.1",
        "easyocr": "1.7",
        "playwright": "1.4",
        "chromadb": "0.5",
    }
    assert update.infer_groups(old_all) == ["voice", "screen", "browser", "more"]
    assert update.infer_groups({"pilot-daemon": "0.15.0", "jeepney": "0.9"}) == ["platform"]
    assert update.infer_groups({"pilot-daemon": "0.15.0"}) == []


def test_the_requirement_keeps_every_group_and_pins_the_version():
    extras = update.parse_extras("", platform="linux")
    assert update.requirement("0.17.0", [], extras) == "pilot-daemon==0.17.0"
    assert (
        update.requirement("0.17.0", ["voice", "platform"], extras)
        == "pilot-daemon[voice,linux]==0.17.0"
    )
    assert (
        update.requirement("0.17.0", ["platform"], update.parse_extras("", platform="win32"))
        == "pilot-daemon==0.17.0"
    )
    assert (
        update.requirement("0.17.0", ["voice"], extras, source="/src/daemon")
        == "/src/daemon[voice]"
    )
    assert update.parse_extras("voice=voice,extra2", platform="darwin")["voice"] == "voice,extra2"


def test_only_changed_packages_are_listed():
    report = {
        "install": [
            {
                "metadata": {"name": "pilot-daemon", "version": "0.17.0"},
                "download_info": {"url": "https://x/p.whl"},
            },
            {
                "metadata": {"name": "NumPy", "version": "2.2.6"},
                "download_info": {"url": "https://x/n.whl"},
            },
            {
                "metadata": {"name": "mcp", "version": "2.1"},
                "download_info": {"url": "https://x/m.whl"},
            },
        ]
    }
    installed = {"pilot-daemon": "0.16.0", "numpy": "2.2.6", "mcp": "2.0"}
    changes = update.changes_from_report(report, installed)
    assert [(c["name"], c["from"], c["to"]) for c in changes] == [
        ("pilot-daemon", "0.16.0", "0.17.0"),
        ("mcp", "2.0", "2.1"),
    ]


def test_versions_compare_as_numbers():
    assert update.needs_update("0.9.0", "0.16.0")
    assert update.needs_update("", "0.16.0")
    assert not update.needs_update("0.16.0", "0.16.0")
    assert not update.needs_update("0.17.0", "0.16.0")  # a newer daemon is never downgraded


def test_snapshot_keeps_the_old_files_when_pip_replaces_them_and_restore_puts_them_back(tmp_path):
    env = tmp_path / "env"
    (env / "lib" / "pkg").mkdir(parents=True)
    module = env / "lib" / "pkg" / "__init__.py"
    module.write_text("VERSION = 1\n", encoding="utf-8")
    (env / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    backup = tmp_path / "env.previous"
    update.snapshot(env, backup)
    # pip replaces a file by removing it and writing a new one; the hard-linked snapshot keeps the old content.
    module.unlink()
    module.write_text("VERSION = 2\n", encoding="utf-8")
    (env / "lib" / "new.py").write_text("x = 1\n", encoding="utf-8")
    assert (backup / "lib" / "pkg" / "__init__.py").read_text(encoding="utf-8") == "VERSION = 1\n"
    update.restore(env, backup)
    assert module.read_text(encoding="utf-8") == "VERSION = 1\n"
    assert not (env / "lib" / "new.py").exists() and not backup.exists()


def test_groups_are_recorded_after_an_update(tmp_path):
    groups = tmp_path / "setup.json"
    assert update.read_groups(groups) is None
    update.write_groups(groups, "0.17.0", ["platform", "voice"])
    assert json.loads(groups.read_text(encoding="utf-8")) == {
        "version": "0.17.0",
        "groups": ["platform", "voice"],
    }
    assert update.read_groups(groups) == ["platform", "voice"]


def test_a_broken_environment_is_sent_to_repair(tmp_path, capsys):
    assert update.main(["plan", "--env", str(tmp_path / "missing"), "--version", "0.17.0"]) == 1
    out = capsys.readouterr().out
    assert "@@step update failed" in out and "Repair" in out
