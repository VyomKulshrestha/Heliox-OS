from pathlib import Path

import pytest

from pilot import config
from pilot.config import DataDirNotWritableError, ensure_dirs, validate_data_dir_writable


def test_validate_data_dir_writable_creates_and_probes_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "pilot-data"

    validate_data_dir_writable(data_dir)

    assert data_dir.is_dir()
    assert not list(data_dir.glob(".pilot-write-test-*"))


def test_validate_data_dir_writable_rejects_file_path(tmp_path: Path) -> None:
    data_dir = tmp_path / "pilot-data"
    data_dir.write_text("not a directory", encoding="utf-8")

    with pytest.raises(DataDirNotWritableError, match="not a directory"):
        validate_data_dir_writable(data_dir)


def test_ensure_dirs_validates_configured_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    runtime_dir = tmp_path / "runtime"

    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "STATE_DIR", state_dir)
    monkeypatch.setattr(config, "RUNTIME_DIR", runtime_dir)

    ensure_dirs()

    assert config_dir.is_dir()
    assert data_dir.is_dir()
    assert state_dir.is_dir()
    assert runtime_dir.is_dir()
