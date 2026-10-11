from __future__ import annotations

import tomllib

import pytest

import pilot.config as config_module
from pilot.config import PilotConfig


def _redirect_config_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", tmp_path / "config.toml")
    monkeypatch.setattr(config_module, "RESTRICTIONS_FILE", tmp_path / "restrictions.toml")


def test_save_replaces_complete_config_files_without_leaving_temporary_files(tmp_path, monkeypatch):
    _redirect_config_files(tmp_path, monkeypatch)
    config = PilotConfig()
    config.voice.tts_engine = "os_native"

    config.save()

    assert tomllib.loads(config_module.CONFIG_FILE.read_text(encoding="utf-8"))["voice"]["tts_engine"] == "os_native"
    assert tomllib.loads(config_module.RESTRICTIONS_FILE.read_text(encoding="utf-8"))
    assert list(tmp_path.glob(".*.tmp")) == []


def test_failed_atomic_replace_preserves_previous_config(tmp_path, monkeypatch):
    _redirect_config_files(tmp_path, monkeypatch)
    config = PilotConfig()
    config.save()
    original = config_module.CONFIG_FILE.read_bytes()
    real_replace = config_module.os.replace

    def fail_config_replace(source, destination):
        if destination == config_module.CONFIG_FILE:
            raise OSError("simulated replace failure")
        real_replace(source, destination)

    config.voice.tts_engine = "os_native"
    monkeypatch.setattr(config_module.os, "replace", fail_config_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        config.save()

    assert config_module.CONFIG_FILE.read_bytes() == original
    assert list(tmp_path.glob(".*.tmp")) == []


def test_malformed_restrictions_fall_back_to_built_in_safety_defaults(tmp_path, monkeypatch, caplog):
    _redirect_config_files(tmp_path, monkeypatch)
    config_module.RESTRICTIONS_FILE.write_text("not = [valid", encoding="utf-8")

    loaded = PilotConfig.load()

    assert loaded.restrictions == config_module.Restrictions()
    assert "Failed to load restrictions.toml" in caplog.text


def test_saving_keeps_the_persons_comments(tmp_path, monkeypatch):
    _redirect_config_files(tmp_path, monkeypatch)
    config = PilotConfig()
    config.save()
    text = config_module.CONFIG_FILE.read_text(encoding="utf-8")
    text = text.replace("[voice]\n", "# Voice tweaks I made by hand\n[voice]\n", 1)
    text = text.replace("tts_engine =", "# keep the local engine\ntts_engine =", 1)
    config_module.CONFIG_FILE.write_text(text + "# end of my notes\n", encoding="utf-8")

    config.voice.tts_engine = "os_native"
    config.save()

    saved = config_module.CONFIG_FILE.read_text(encoding="utf-8")
    assert "# Voice tweaks I made by hand\n[voice]\n" in saved
    assert '# keep the local engine\ntts_engine = "os_native"' in saved
    assert saved.rstrip().endswith("# end of my notes")
    assert tomllib.loads(saved)["voice"]["tts_engine"] == "os_native"


def test_a_briefly_locked_config_file_is_retried(tmp_path, monkeypatch):
    _redirect_config_files(tmp_path, monkeypatch)
    real_replace = config_module.os.replace
    attempts = {"count": 0}

    def locked_once(source, destination):
        if destination == config_module.CONFIG_FILE and attempts["count"] == 0:
            attempts["count"] += 1
            raise PermissionError("held by another program")
        real_replace(source, destination)

    monkeypatch.setattr(config_module.os, "replace", locked_once)
    PilotConfig().save()

    assert attempts["count"] == 1
    assert tomllib.loads(config_module.CONFIG_FILE.read_text(encoding="utf-8"))
