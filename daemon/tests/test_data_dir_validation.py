from pathlib import Path

import pytest

from pilot import config
from pilot.config import ensure_dirs


def test_ensure_dirs_creates_and_probes_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "pilot-data"
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(config, "PLUGINS_DIR", tmp_path / "plugins")
    monkeypatch.setattr(config, "SCREENSHOTS_DIR", tmp_path / "screenshots")

    ensure_dirs()

    assert data_dir.is_dir()
    assert not list(data_dir.glob(".pilot-write-test-*"))


def test_ensure_dirs_rejects_file_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "pilot-data"
    data_dir.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(config, "PLUGINS_DIR", tmp_path / "plugins")
    monkeypatch.setattr(config, "SCREENSHOTS_DIR", tmp_path / "screenshots")

    with pytest.raises(RuntimeError, match="DATA_DIR is not writable"):
        ensure_dirs()


def test_ensure_dirs_validates_configured_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    runtime_dir = tmp_path / "runtime"
    plugins_dir = tmp_path / "plugins"
    screenshots_dir = tmp_path / "screenshots"

    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "STATE_DIR", state_dir)
    monkeypatch.setattr(config, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(config, "PLUGINS_DIR", plugins_dir)
    monkeypatch.setattr(config, "SCREENSHOTS_DIR", screenshots_dir)

    ensure_dirs()

    assert config_dir.is_dir()
    assert data_dir.is_dir()
    assert state_dir.is_dir()
    assert runtime_dir.is_dir()
    assert plugins_dir.is_dir()
    assert screenshots_dir.is_dir()
