"""The suite-wide folder guard (tests/folder_guard.py): Heliox's folders point at the session folder, and anything a
test tries under the real ones is refused, by its path as written, differently cased, or through a link.

The refusal checks use a stand-in "real" folder in tmp_path, so nothing here goes near the real folders, and each
removes its own record so the session is not failed by the check of the check.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import pilot.config as config_module

folder_guard = sys.modules["folder_guard"]  # loaded by conftest before pilot


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    monkeypatch.setitem(folder_guard._state, "roots", (folder_guard._key(real), folder_guard._resolved(real)))
    before = {test: set(hits) for test, hits in folder_guard._state["refused"].items()}
    yield real
    folder_guard._state["refused"].clear()
    folder_guard._state["refused"].update(before)


def test_heliox_folders_point_at_the_session_folder():
    session = folder_guard._state["session"]
    for folder in (
        config_module.CONFIG_DIR,
        config_module.DATA_DIR,
        config_module.STATE_DIR,
        config_module.RUNTIME_DIR,
    ):
        assert session in Path(folder).parents, folder
    for real in folder_guard.real_folders({key: value for key, value in os.environ.items() if "HELIOX_" not in key}):
        assert session not in real.parents


def test_an_open_under_a_real_folder_is_refused_and_names_the_test(stand_in):
    target = stand_in / "experience.db"
    with pytest.raises(PermissionError) as refused:
        open(target, "w").close()
    assert "test_an_open_under_a_real_folder_is_refused_and_names_the_test" in str(refused.value)
    assert str(target) in str(refused.value)
    assert not target.exists()


def test_sqlite_and_folder_creation_are_refused_too(stand_in):
    import sqlite3

    with pytest.raises(PermissionError):
        sqlite3.connect(stand_in / "pilot.db")
    with pytest.raises(PermissionError):
        (stand_in / "keys").mkdir()
    assert list(stand_in.iterdir()) == []


@pytest.mark.skipif(sys.platform not in ("win32", "darwin"), reason="case-insensitive file systems only")
def test_a_differently_cased_path_is_refused(stand_in):
    with pytest.raises(PermissionError):
        open(str(stand_in / "notes.json").upper(), "w").close()


def test_a_path_through_a_link_is_refused(stand_in, tmp_path):
    link = tmp_path / "link"
    folder_guard._state["active"] = False
    try:
        try:
            os.symlink(stand_in, link, target_is_directory=True)
        except OSError:
            if sys.platform != "win32":
                raise
            import _winapi

            _winapi.CreateJunction(str(stand_in), str(link))  # a junction needs no privilege on Windows
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"no symbolic links or junctions here: {error}")
    finally:
        folder_guard._state["active"] = True
    with pytest.raises(PermissionError):
        open(link / "todos.json", "w").close()
    assert not (stand_in / "todos.json").exists()


def test_paths_elsewhere_are_untouched(stand_in, tmp_path):
    (tmp_path / "elsewhere.txt").write_text("fine", encoding="utf-8")
    assert (tmp_path / "elsewhere.txt").read_text(encoding="utf-8") == "fine"


def test_the_cache_and_setup_folders_point_at_the_session_folder_too():
    session = folder_guard._state["session"]
    assert session / ".cache" / "heliox" == config_module.CACHE_DIR
    assert session / ".config" / "heliox-os" == config_module.SETUP_DIR
    assert config_module.TEST_ROOT_VARIABLE == "HELIOX_TEST_ROOT"
    assert (session / config_module.TEST_ROOT_MARKER).is_file()  # the name the guard writes


def test_a_test_root_without_its_marker_is_ignored(tmp_path, monkeypatch):
    """Setting HELIOX_TEST_ROOT by hand, or a leftover in a shell, never moves a real install's folders."""
    monkeypatch.setenv("HELIOX_TEST_ROOT", str(tmp_path))
    assert config_module._test_root() is None
    (tmp_path / config_module.TEST_ROOT_MARKER).write_text("", encoding="utf-8")
    assert config_module._test_root() == tmp_path
