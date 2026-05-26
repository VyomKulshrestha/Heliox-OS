"""Tests for consistent config/data directory path consolidation.

Verifies that all modules use the centralized constants from ``pilot.config``
instead of hardcoded path strings.
"""

from pathlib import Path

import pytest

from pilot.config import (
    CONFIG_DIR,
    CONFIG_FILE,
    DATA_DIR,
    DB_FILE,
    LOG_FILE,
    PERSONA_FILE,
    PLUGINS_DIR,
    SCREENSHOTS_DIR,
    STATE_DIR,
    ensure_dirs,
    migrate_old_paths,
)


class TestConfigConstants:
    """All path constants must be consistent and predictable."""

    def test_config_dir_is_heliox_os(self):
        assert CONFIG_DIR.name == "heliox-os"
        assert ".config" in str(CONFIG_DIR)

    def test_data_dir_is_heliox_os(self):
        assert DATA_DIR.name == "heliox-os"
        assert ".local" in str(DATA_DIR)

    def test_state_dir_is_heliox_os(self):
        assert STATE_DIR.name == "heliox-os"

    def test_plugins_dir_under_config_dir(self):
        assert PLUGINS_DIR.parent == CONFIG_DIR
        assert PLUGINS_DIR.name == "plugins"

    def test_screenshots_dir_under_data_dir(self):
        assert SCREENSHOTS_DIR.parent == DATA_DIR
        assert SCREENSHOTS_DIR.name == "screenshots"

    def test_persona_file_under_data_dir(self):
        assert PERSONA_FILE.parent == DATA_DIR
        assert PERSONA_FILE.name == "persona.md"

    def test_config_file_under_config_dir(self):
        assert CONFIG_FILE.parent == CONFIG_DIR
        assert CONFIG_FILE.name == "config.toml"

    def test_db_file_under_data_dir(self):
        assert DB_FILE.parent == DATA_DIR
        assert DB_FILE.name == "pilot.db"

    def test_log_file_under_state_dir(self):
        assert LOG_FILE.parent == STATE_DIR
        assert LOG_FILE.name == "pilot.log"

    def test_all_dirs_are_under_home(self):
        home = Path.home()
        assert str(CONFIG_DIR).startswith(str(home))
        assert str(DATA_DIR).startswith(str(home))
        assert str(STATE_DIR).startswith(str(home))
        assert str(PLUGINS_DIR).startswith(str(home))
        assert str(SCREENSHOTS_DIR).startswith(str(home))


class TestEnsureDirs:
    """ensure_dirs() must create all required directories."""

    def test_creates_all_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pilot.config.CONFIG_DIR", tmp_path / "config")
        monkeypatch.setattr("pilot.config.DATA_DIR", tmp_path / "data")
        monkeypatch.setattr("pilot.config.STATE_DIR", tmp_path / "state")
        monkeypatch.setattr("pilot.config.RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr("pilot.config.PLUGINS_DIR", tmp_path / "config" / "plugins")
        monkeypatch.setattr("pilot.config.SCREENSHOTS_DIR", tmp_path / "data" / "screenshots")

        ensure_dirs()

        assert (tmp_path / "config").is_dir()
        assert (tmp_path / "data").is_dir()
        assert (tmp_path / "state").is_dir()
        assert (tmp_path / "runtime").is_dir()
        assert (tmp_path / "config" / "plugins").is_dir()
        assert (tmp_path / "data" / "screenshots").is_dir()


class TestMigrateOldPaths:
    """Old hardcoded paths must be migrated to the new layout."""

    def test_migrates_old_heliox_plugins(self, tmp_path, monkeypatch):
        plugins_dir = tmp_path / "config" / "plugins"
        monkeypatch.setattr("pilot.config.PLUGINS_DIR", plugins_dir)

        old_plugins = tmp_path / ".heliox" / "plugins"
        old_plugins.mkdir(parents=True)
        (old_plugins / "my_plugin.py").write_text("# test")

        with monkeypatch.context() as m:
            m.setattr(Path, "home", lambda: tmp_path)
            migrate_old_paths()

        assert (plugins_dir / "my_plugin.py").exists()
        assert not old_plugins.exists()

    def test_migrates_old_persona(self, tmp_path, monkeypatch):
        persona_file = tmp_path / "data" / "persona.md"
        monkeypatch.setattr("pilot.config.PERSONA_FILE", persona_file)

        old_persona = tmp_path / ".heliox" / "persona.md"
        old_persona.parent.mkdir(parents=True)
        old_persona.write_text("# Persona")

        with monkeypatch.context() as m:
            m.setattr(Path, "home", lambda: tmp_path)
            migrate_old_paths()

        assert persona_file.exists()
        assert persona_file.read_text() == "# Persona"
        assert not old_persona.exists()

    def test_migrates_old_screenshots(self, tmp_path, monkeypatch):
        screenshots_dir = tmp_path / "data" / "screenshots"
        monkeypatch.setattr("pilot.config.SCREENSHOTS_DIR", screenshots_dir)

        old_screenshots = tmp_path / ".heliox" / "screenshots"
        old_screenshots.mkdir(parents=True)
        (old_screenshots / "shot1.png").write_text("data")

        with monkeypatch.context() as m:
            m.setattr(Path, "home", lambda: tmp_path)
            migrate_old_paths()

        assert (screenshots_dir / "shot1.png").exists()
        assert not old_screenshots.exists()

    def test_migrates_old_config_pilot(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "config"
        monkeypatch.setattr("pilot.config.CONFIG_DIR", config_dir)

        old_config = tmp_path / ".config" / "pilot"
        old_config.mkdir(parents=True)
        (old_config / "config.toml").write_text("key = 'val'")

        with monkeypatch.context() as m:
            m.setattr(Path, "home", lambda: tmp_path)
            migrate_old_paths()

        assert (config_dir / "config.toml").exists()
        assert not old_config.exists()

    def test_skips_migration_when_destination_exists(self, tmp_path, monkeypatch):
        plugins_dir = tmp_path / "config" / "plugins"
        plugins_dir.mkdir(parents=True)
        (plugins_dir / "existing.py").write_text("# existing")
        monkeypatch.setattr("pilot.config.PLUGINS_DIR", plugins_dir)

        old_plugins = tmp_path / ".heliox" / "plugins"
        old_plugins.mkdir(parents=True)
        (old_plugins / "old.py").write_text("# old")

        with monkeypatch.context() as m:
            m.setattr(Path, "home", lambda: tmp_path)
            migrate_old_paths()

        assert (plugins_dir / "existing.py").exists()
        assert not (plugins_dir / "old.py").exists(), "should not overwrite"
        assert old_plugins.exists(), "old dir should remain unmoved"
