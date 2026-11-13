"""The Conversation Bench stays inside its own root: every Heliox folder, the wake calibration, the temporary files.

The bench refuses to build a listener or a server when any folder would resolve outside its root, copies the
person's settings only by name, and reads the speech models where they are without writing there.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from benchmarks.conversation_bench import bench_root, writes
from pilot import config
from pilot.real_folders import RealFoldersRefused

DAEMON = Path(__file__).resolve().parents[1]


def _fake_home_env(home: Path) -> dict[str, str]:
    """A process started by the person with a home of its own: no test root, no XDG folders, home and app folders
    in ``home``."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("XDG_") and k != "HELIOX_TEST_ROOT"}
    for name in ("HOME", "USERPROFILE"):
        env[name] = str(home)
    env["LOCALAPPDATA"] = str(home / "AppData" / "Local")
    env["APPDATA"] = str(home / "AppData" / "Roaming")
    env["PYTHONPATH"] = str(DAEMON)
    return env


ENTER_AND_LEARN = """
import sys
from pathlib import Path
from benchmarks.conversation_bench import bench_root
root = bench_root.enter(Path(sys.argv[1]))
from pilot import config
from pilot.system.voice_calibration import WakeWordCalibrator
calibrator = WakeWordCalibrator(["hey heliox"])
calibrator.record_pending(calibrator.check_near_miss("hey helix open the deck"))
calibrator.confirm_pending_if_followed_by_hit()
print(root)
for name, path in sorted(bench_root.check_inside(root).items()):
    print(name, path)
"""


def test_a_bench_root_holds_every_folder_and_a_near_miss_save(tmp_path):
    home, root = tmp_path / "home", tmp_path / "bench-root"
    home.mkdir()
    shown = subprocess.run(
        [sys.executable, "-c", ENTER_AND_LEARN, str(root)],
        cwd=DAEMON,
        env=_fake_home_env(home),
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    ).stdout.splitlines()

    assert Path(shown[0]) == root.resolve()
    folders = dict(line.split(" ", 1) for line in shown[1:])
    assert set(folders) == {"config", "data", "state", "runtime", "cache", "setup"}
    assert all(writes.inside(path, root) for path in folders.values())
    saved = list(root.rglob("wake_variants.json"))
    assert len(saved) == 1 and writes.inside(str(saved[0]), Path(folders["cache"]))
    assert "hey helix" in saved[0].read_text(encoding="utf-8")
    assert [p for p in home.rglob("*") if p.is_file()] == []  # nothing in the home folder


def test_the_bench_refuses_when_pilot_config_was_imported_first(tmp_path):
    home, root = tmp_path / "home", tmp_path / "bench-root"
    home.mkdir()
    code = (
        "import sys; from pathlib import Path\n"
        "import pilot.config\n"
        "from benchmarks.conversation_bench import bench_root\n"
        "bench_root.enter(Path(sys.argv[1]))\n"
    )
    ran = subprocess.run(
        [sys.executable, "-c", code, str(root)],
        cwd=DAEMON,
        env=_fake_home_env(home),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert ran.returncode != 0
    assert "before pilot.config is imported" in ran.stderr
    assert not root.exists()


@pytest.mark.parametrize("name", ["CONFIG_DIR", "DATA_DIR", "STATE_DIR", "RUNTIME_DIR", "CACHE_DIR", "SETUP_DIR"])
def test_any_folder_outside_the_root_is_refused(monkeypatch, tmp_path, name):
    monkeypatch.setattr(config, name, tmp_path / "home" / "heliox-os")
    with pytest.raises(RealFoldersRefused, match="outside its own root"):
        bench_root.check_inside(config._TEST_ROOT)


def test_real_folders_are_refused(monkeypatch):
    monkeypatch.setattr(config, "_TEST_ROOT", None)
    with pytest.raises(RealFoldersRefused, match="not a scratch root|outside its own root"):
        bench_root.check_inside(Path.home())


def test_the_harness_builds_nothing_when_a_folder_is_outside(monkeypatch, tmp_path):
    from benchmarks.conversation_bench.driver import Harness
    from benchmarks.conversation_bench.models import KINDS

    for variable in ("TMP", "TEMP", "TMPDIR", "PYTHONDONTWRITEBYTECODE"):
        monkeypatch.setenv(variable, os.environ.get(variable, ""))
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "home" / ".cache" / "heliox")
    built = []
    monkeypatch.setattr("pilot.server.PilotServer", lambda *a, **k: built.append(a))

    harness = Harness(audio_root=str(tmp_path), data_dir=tmp_path / "root", model=KINDS["cloud"])
    with pytest.raises(RealFoldersRefused, match="cache"):
        asyncio.run(harness.build())
    assert built == []


def test_the_settings_come_across_by_name_only(monkeypatch):
    from pilot import real_folders

    owner = {
        "model": {"provider": "ollama", "not_a_field": 1},
        "voice": {"final_pass": "parakeet", "input_device": "Desk mic", "whisper_model": "base", "wake_word": "x"},
    }
    monkeypatch.setattr(real_folders, "owner_settings", lambda section: dict(owner.get(section, {})))
    target = config.PilotConfig()

    copied = bench_root.copy_owner_settings(target)

    assert copied == ["model.provider", "voice.final_pass", "voice.whisper_model"]
    assert target.voice.final_pass == "parakeet"
    assert target.voice.input_device != "Desk mic"  # a device is never copied


def test_the_models_are_read_in_place_and_a_missing_one_stops_the_run(monkeypatch, tmp_path):
    from pilot import real_folders
    from pilot.system import model_presence, parakeet_asr, speech_models

    monkeypatch.setattr(speech_models, "local_path", speech_models.local_path)
    monkeypatch.setattr(real_folders, "owner_settings", lambda section: {})
    monkeypatch.setattr(model_presence, "hf_cached", lambda repo, name: None)
    owner = tmp_path / "owner-data"
    target = config.PilotConfig()
    target.voice.final_pass = "parakeet"
    target.voice.whisper_model = ""
    target.voice.tts_engine = "system"

    with pytest.raises(bench_root.MissingModels) as missing:
        bench_root.read_models_in_place(target, owner_folder=owner)
    for name in ("streaming-zipformer-en", "smart-turn-v3", bench_root.PARAKEET_ID):
        assert name in str(missing.value)

    parakeet = owner / "models" / parakeet_asr.MODEL_NAME
    parakeet.mkdir(parents=True)
    for name in parakeet_asr.FILES:
        (parakeet / name).write_bytes(b"")
    monkeypatch.setattr(model_presence, "hf_cached", lambda repo, name: str(tmp_path / name))
    before = sorted((p.name, p.stat().st_mtime) for p in parakeet.iterdir())

    found = bench_root.read_models_in_place(target, owner_folder=owner)

    assert found[bench_root.PARAKEET_ID] == "in place"
    assert Path(target.voice.parakeet_model_dir) == parakeet
    assert found["smart-turn-v3"] == "Hugging Face cache"
    assert sorted((p.name, p.stat().st_mtime) for p in parakeet.iterdir()) == before  # read, never written


def test_writes_sees_files_opened_for_writing_only(tmp_path):
    seen = writes.Writes().install()
    target = tmp_path / "made.txt"
    target.write_text("x", encoding="utf-8")
    target.read_text(encoding="utf-8")
    (tmp_path / "folder").mkdir()
    seen.stop()
    (tmp_path / "after.txt").write_text("y", encoding="utf-8")

    assert str(target) in seen.paths
    assert str(tmp_path / "folder") in seen.paths
    assert str(tmp_path / "after.txt") not in seen.paths
    assert writes.outside(seen.paths, tmp_path) == []
    assert writes.newer_files([tmp_path], 0.0) == sorted(str(p) for p in tmp_path.rglob("*") if p.is_file())
