"""Under scratch folders and the test suite, secrets never reach this computer's credential store
(pilot.security.credential_store).

The processes started here use a made-up "system store" (a keyring backend in a module of the test's own folder,
named by PYTHON_KEYRING_BACKEND) that records every call to a file, so what a scratch run or a real install would
read or write is seen without touching Credential Manager, the keychain or the Secret Service. Tests that provoke the
guard on purpose clear its record, so the suite's own CREDENTIAL-GUARD does not fail them; nothing they call reaches
a store, because the refusal comes before the backend's own code.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pilot.security import credential_store
from pilot.security.credential_store import VAULT_SERVICE, CredentialStoreRefused

DAEMON = Path(__file__).resolve().parents[1]

FAKE_SYSTEM_STORE = """
import json, os
from keyring.backend import KeyringBackend
from keyring.errors import PasswordDeleteError


class Recording(KeyringBackend):
    priority = 1

    def _log(self, *call):
        with open(os.environ["FAKE_STORE_LOG"], "a", encoding="utf-8") as log:
            log.write(json.dumps(call) + "\\n")

    def get_password(self, service, username):
        self._log("get", service, username)
        return {"claude": "real-claude-key", "openai": "real-openai-key", "plan_secret:index": "[]"}.get(username)

    def set_password(self, service, username, password):
        self._log("set", service, username)

    def delete_password(self, service, username):
        self._log("delete", service, username)
        raise PasswordDeleteError("fake")
"""


@pytest.fixture
def expected_refusals():
    yield
    credential_store.clear()


def _run(tmp_path: Path, code: str, *, scratch: bool, backend: str = "fakereal.Recording", root: Path | None = None):
    """Run ``code`` in a new Python process: a scratch run (HELIOX_TEST_ROOT left for use_scratch_folders to set) or a
    real install (no test root; every home and XDG folder in tmp_path, so even a stray write lands there)."""
    (tmp_path / "fakereal.py").write_text(FAKE_SYSTEM_STORE, encoding="utf-8")
    log = tmp_path / "system_store_calls.jsonl"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HELIOX_", "XDG_", "PYTEST_"))}
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env.update(
        PYTHONPATH=os.pathsep.join([str(tmp_path), str(DAEMON)]),
        PYTHON_KEYRING_BACKEND=backend,
        FAKE_STORE_LOG=str(log),
        HOME=str(home),
        USERPROFILE=str(home),
        LOCALAPPDATA=str(home / "AppData" / "Local"),
        APPDATA=str(home / "AppData" / "Roaming"),
        XDG_CONFIG_HOME=str(home / ".config"),
        XDG_DATA_HOME=str(home / ".local" / "share"),
        XDG_STATE_HOME=str(home / ".local" / "state"),
        XDG_RUNTIME_DIR=str(home / "runtime"),
    )
    if root is not None:
        env["HELIOX_TEST_ROOT"] = str(root)
    done = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "scratch")],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=str(tmp_path),
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    return json.loads(done.stdout.strip().splitlines()[-1]), calls


SCRATCH_RUN = """
import asyncio, json, sys
from pathlib import Path
from pilot.real_folders import use_scratch_folders
root = use_scratch_folders(Path(sys.argv[1]){extra})
import keyring
from pilot.config import PilotConfig
from pilot.security import credential_store
from pilot.security.vault import KeyVault, VAULT_SERVICE

async def main():
    vault = KeyVault(PilotConfig())
    seen = {{"mode": credential_store.mode(), "available": vault.available, "backend": vault.backend_name}}
    for name in ("claude", "openai", "plan_secret:index", "heliox_task_receipt_signing_key"):
        seen[name] = await vault.get_key(name)
    await vault.store_key("plan_secret:abc", "header-value")
    await vault.store_key("claude", "scratch-claude")
    vault.clear_cache()
    seen["claude_after_write"] = await vault.get_key("claude")
    await vault.delete_key("claude")
    vault.clear_cache()
    seen["claude_after_delete"] = await vault.get_key("claude")
    seen["listed"] = credential_store.scratch_accounts(VAULT_SERVICE)
    file = root / credential_store.SCRATCH_FILE
    seen["file_text"] = file.read_text(encoding="utf-8") if file.exists() else ""
    print(json.dumps(seen))

asyncio.run(main())
"""


def test_a_scratch_run_keeps_every_secret_in_its_root_and_reads_nothing_real(tmp_path):
    seen, calls = _run(tmp_path, SCRATCH_RUN.format(extra=""), scratch=True)

    assert calls == []  # the system store was never asked, though PYTHON_KEYRING_BACKEND named it
    assert seen["mode"] == "file" and seen["available"] is True and seen["backend"] == "ScratchCredentialStore"
    assert seen["claude"] is None and seen["openai"] is None and seen["plan_secret:index"] is None
    assert seen["heliox_task_receipt_signing_key"] is None
    assert seen["claude_after_write"] == "scratch-claude" and seen["claude_after_delete"] is None
    assert seen["listed"] == ["plan_secret:abc"]
    assert "header-value" in seen["file_text"] and "real-" not in seen["file_text"]


def test_a_named_model_key_is_read_from_the_system_store_and_nothing_else(tmp_path):
    seen, calls = _run(tmp_path, SCRATCH_RUN.format(extra=', real_credentials=("claude",)'), scratch=True)

    assert seen["claude"] == "real-claude-key"
    assert seen["openai"] is None and seen["plan_secret:index"] is None  # present there, not named: absent
    assert seen["claude_after_write"] == "scratch-claude"  # its own write is kept in the scratch store
    assert seen["claude_after_delete"] == "real-claude-key"  # the delete removed the scratch copy only
    # Only reads of the named entry reached the system store: no write, no delete, no other name.
    assert {tuple(call) for call in calls} == {("get", VAULT_SERVICE, "claude")}
    assert "real-claude-key" not in seen["file_text"]  # never copied into the scratch store
    assert seen["listed"] == ["plan_secret:abc"]


def test_the_fail_backend_under_a_scratch_root_means_no_store_and_no_real_reads(tmp_path):
    seen, calls = _run(
        tmp_path,
        SCRATCH_RUN.replace('    await vault.store_key("plan_secret:abc", "header-value")\n', "")
        .replace('    await vault.store_key("claude", "scratch-claude")\n', "")
        .replace('    await vault.delete_key("claude")\n', "")
        .format(extra=', real_credentials=("claude",)'),
        scratch=True,
        backend=credential_store.FAIL_BACKEND,
    )

    assert seen["mode"] == "none" and seen["available"] is False and seen["claude"] is None
    assert calls == [] and seen["file_text"] == ""


REAL_INSTALL = """
import asyncio, json
import keyring
from pilot.config import PilotConfig, _TEST_ROOT
from pilot.security import credential_store
from pilot.security.vault import KeyVault, VAULT_SERVICE

async def main():
    vault = KeyVault(PilotConfig())
    print(json.dumps({
        "test_root": str(_TEST_ROOT) if _TEST_ROOT else None,
        "isolated": credential_store.isolated() is not None,
        "service": VAULT_SERVICE,
        "backend": type(keyring.get_keyring()).__name__,
        "available": vault.available,
        "claude": await vault.get_key("claude"),
    }))

asyncio.run(main())
"""


@pytest.mark.parametrize("unmarked_root", [False, True], ids=["no_test_root", "unmarked_test_root"])
def test_a_real_install_uses_the_system_store_unchanged(tmp_path, unmarked_root):
    root = None
    if unmarked_root:
        root = tmp_path / "not-a-scratch-root"  # HELIOX_TEST_ROOT left in a shell, without the marker
        root.mkdir()
    seen, calls = _run(tmp_path, REAL_INSTALL, scratch=False, root=root)

    assert seen == {
        "test_root": None,
        "isolated": False,
        "service": "pilot-ai-command-center",
        "backend": "Recording",
        "available": True,
        "claude": "real-claude-key",
    }
    assert calls == [["get", "pilot-ai-command-center", "claude"]]


def test_a_real_install_without_a_store_still_fails_closed(tmp_path):
    """A Linux machine with no Secret Service: keyring falls back to its fail backend, and the vault refuses."""
    seen, calls = _run(tmp_path, REAL_INSTALL, scratch=False, backend=credential_store.FAIL_BACKEND)

    assert seen["isolated"] is False and seen["backend"] == "Keyring"
    assert seen["available"] is False and seen["claude"] is None and calls == []


# ── in this process: the suite's own store, the guard, the names that may be read ──


def test_the_suite_has_no_store_at_all_and_reads_nothing_real():
    import keyring
    import keyring.backends.fail

    from pilot import config

    assert credential_store.isolated() is not None and credential_store.mode() == "none"
    assert Path(credential_store.isolated()) == Path(config._TEST_ROOT)
    assert isinstance(keyring.get_keyring(), keyring.backends.fail.Keyring)
    assert credential_store.allow_real_reads(["claude"]) == ()  # the explicit way through is closed here too
    assert credential_store.readable() == frozenset()


def test_every_system_backend_refuses_and_is_counted(expected_refusals):
    from keyring.backend import KeyringBackend

    systems = [cls for cls in KeyringBackend._classes if credential_store._is_system_backend(cls)]
    assert systems  # at least the Secret Service and KWallet classes load on every platform
    for cls in systems:
        for method in ("get_password", "set_password", "delete_password"):
            assert getattr(cls.__dict__.get(method), "_heliox_refuses", True)
    before = len(credential_store.attempts())
    store = systems[0].__new__(systems[0])
    with pytest.raises(CredentialStoreRefused, match="credential store"):
        store.get_password(VAULT_SERVICE, "claude")
    with pytest.raises(CredentialStoreRefused):
        store.set_password(VAULT_SERVICE, "plan_secret:x", "v")
    reached = credential_store.attempts()[before:]
    assert len(reached) == 2 and all(where.startswith("tests/test_credential_store.py") for where, _ in reached)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows' credential functions")
def test_windows_credential_functions_refuse_listing_and_writing(expected_refusals):
    from win32ctypes.pywin32 import win32cred

    from pilot.system import credential_cleanup

    for name in ("CredRead", "CredWrite", "CredDelete", "CredEnumerate"):
        assert getattr(getattr(win32cred, name), "_heliox_refuses", False), name
    with pytest.raises(CredentialStoreRefused):
        credential_cleanup.windows_entries(VAULT_SERVICE)  # the real listing, called directly: refused
    with pytest.raises(CredentialStoreRefused):
        win32cred.CredWrite({}, 0)


def test_the_cleanup_lists_the_scratch_store_never_the_systems():
    from pilot.system import credential_cleanup

    assert credential_cleanup._finder(sys.platform) is credential_store.scratch_accounts
    assert credential_store.scratch_accounts(VAULT_SERVICE) == []


@pytest.mark.parametrize(
    "names",
    [("plan_secret:index",), ("heliox_task_receipt_signing_key",), ("github_token",), ("claude", "heliox_mesh")],
)
def test_only_model_provider_keys_can_be_named(names):
    with pytest.raises(ValueError, match="model provider"):
        credential_store.check_readable(names)


def test_a_bare_string_is_not_a_list_of_names():
    with pytest.raises(ValueError, match="list"):
        credential_store.check_readable("claude")
    assert credential_store.check_readable(("claude", "claude_backup_1")) == ("claude", "claude_backup_1")


def test_the_scratch_store_keeps_a_named_real_key_out_of_its_file(tmp_path, monkeypatch):
    store = credential_store._scratch_store_class()(tmp_path / "store.json")
    reads: list[tuple[str, str]] = []

    class System:
        def get_password(self, service, account):
            reads.append((service, account))
            return "real-value"

    monkeypatch.setattr(credential_store, "_system_store", lambda: System())
    monkeypatch.setattr(credential_store, "_readable", frozenset({"claude"}))
    assert store.get_password(VAULT_SERVICE, "claude") == "real-value"
    assert store.get_password(VAULT_SERVICE, "openai") is None
    assert store.get_password("another-service", "claude") is None  # the vault's service only
    assert reads == [(VAULT_SERVICE, "claude")]
    assert not (tmp_path / "store.json").exists()
    store.set_password(VAULT_SERVICE, "openai", "scratch")
    assert "real-value" not in (tmp_path / "store.json").read_text(encoding="utf-8")
    assert store.accounts(VAULT_SERVICE) == ["openai"]
