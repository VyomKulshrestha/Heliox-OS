"""The learner's snapshot key lives in the data folder, survives a reboot, and an earlier version's key is moved there.

Earlier versions kept the key in the runtime folder: Linux empties it at every reboot (/run/user/<uid>) and macOS
keeps it under Library/Caches, which the system may clear, so the saved state stopped matching its key and every
first start after a reboot replayed the whole ledger.
"""

from __future__ import annotations

import os
import shutil
import stat

import pytest

from pilot.config import PilotConfig
from pilot.intelligence import learner_snapshot
from pilot.system.key_file import KeyFileRefused, load_or_create, move_key

KEY = "a" * 64
OTHER = "b" * 64


@pytest.fixture
def folders(tmp_path, monkeypatch):
    runtime, data = tmp_path / "runtime", tmp_path / "data"
    monkeypatch.setattr("pilot.config.RUNTIME_DIR", runtime)
    monkeypatch.setattr("pilot.server.DATA_DIR", data)
    return runtime, data


def _server():
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server._vault = None  # no system keyring: the key file is the key
    return server


def _paths(runtime, data):
    from pilot.server import KEYS_FOLDER, LEARNER_SNAPSHOT_FILE, LEARNER_SNAPSHOT_KEY_FILE

    return (
        runtime / LEARNER_SNAPSHOT_KEY_FILE,
        data / KEYS_FOLDER / LEARNER_SNAPSHOT_KEY_FILE,
        data / LEARNER_SNAPSHOT_FILE,
    )


def _old_key(path, secret=KEY, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(secret, encoding="utf-8")
    if os.name == "posix":
        os.chmod(path, mode)


def _snapshot(path, secret):
    learner_snapshot.write(path, secret.encode(), b"state", model_version="v", reset_before=0, sequence=7)


async def test_the_key_survives_a_reboot_that_empties_the_runtime_folder(folders):
    runtime, data = folders
    first = await _server()._learner_key()
    shutil.rmtree(runtime, ignore_errors=True)  # what Linux does to /run/user/<uid> at every reboot
    assert await _server()._learner_key() == first
    _old, new, _snap = _paths(runtime, data)
    assert new.exists() and not runtime.exists()


async def test_an_earlier_versions_key_that_signs_the_saved_state_is_moved_and_the_old_file_removed(folders):
    runtime, data = folders
    old, new, snap = _paths(runtime, data)
    _old_key(old)
    _snapshot(snap, KEY)
    assert await _server()._learner_key() == KEY.encode()  # the saved state stays valid after the update
    assert new.read_text(encoding="utf-8") == KEY
    assert not old.exists()
    if os.name == "posix":
        assert stat.S_IMODE(new.stat().st_mode) == 0o600
        assert stat.S_IMODE(new.parent.stat().st_mode) == 0o700


async def test_a_key_that_does_not_sign_this_folders_saved_state_is_left_alone(folders):
    """Two data folders can share a runtime folder (XDG_DATA_HOME, a test): the other one's key is not taken."""
    runtime, data = folders
    old, new, snap = _paths(runtime, data)
    _old_key(old)
    _snapshot(snap, OTHER)
    key = await _server()._learner_key()
    assert key != KEY.encode() and new.exists()
    assert old.read_text(encoding="utf-8") == KEY


def test_a_move_cut_short_leaves_one_key(tmp_path):
    old, new = tmp_path / "runtime" / "k", tmp_path / "data" / "keys" / "k"
    _old_key(old)
    _old_key(new)
    assert "removed" in move_key(old, new, lambda _key: True)
    assert not old.exists() and new.read_text(encoding="utf-8") == KEY


def test_a_different_key_already_in_place_is_kept_and_the_old_one_not_touched(tmp_path):
    old, new = tmp_path / "runtime" / "k", tmp_path / "data" / "keys" / "k"
    _old_key(old, OTHER)
    _old_key(new)
    assert move_key(old, new, lambda _key: True) == ""
    assert new.read_text(encoding="utf-8") == KEY and old.read_text(encoding="utf-8") == OTHER


@pytest.mark.skipif(os.name != "posix", reason="Windows keeps other users out of the profile folder by its ACLs")
def test_an_old_key_others_can_read_is_not_moved(tmp_path):
    old, new = tmp_path / "runtime" / "k", tmp_path / "data" / "keys" / "k"
    _old_key(old, mode=0o644)
    said = move_key(old, new, lambda _key: True)
    assert "left" in said and "chmod 600" in said
    assert not new.exists() and old.exists()


@pytest.mark.skipif(os.name != "posix", reason="folder modes are POSIX")
def test_the_keys_folder_is_narrowed_to_the_owner(tmp_path):
    folder = tmp_path / "keys"
    folder.mkdir(mode=0o755)
    os.chmod(folder, 0o755)
    load_or_create(folder / "k")
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    assert stat.S_IMODE((folder / "k").stat().st_mode) == 0o600


def test_a_damaged_old_key_is_not_moved(tmp_path):
    old, new = tmp_path / "runtime" / "k", tmp_path / "data" / "keys" / "k"
    _old_key(old, "short")
    assert "damaged" in move_key(old, new, lambda _key: True)
    assert not new.exists()
    with pytest.raises(KeyFileRefused):
        load_or_create(old)


def test_signed_by_reads_the_signature_without_unpickling(tmp_path):
    snap = tmp_path / "snapshot.bin"
    _snapshot(snap, KEY)
    assert learner_snapshot.signed_by(snap, KEY.encode())
    assert not learner_snapshot.signed_by(snap, OTHER.encode())
    snap.write_bytes(snap.read_bytes()[:-2])  # truncated
    assert not learner_snapshot.signed_by(snap, KEY.encode())
    assert not learner_snapshot.signed_by(tmp_path / "missing.bin", KEY.encode())
