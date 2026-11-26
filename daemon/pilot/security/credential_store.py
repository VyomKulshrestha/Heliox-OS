"""Where Heliox keeps secrets when it runs under scratch folders or the test suite: never this computer's own store.

In a real install nothing here runs. Heliox reads and writes the system credential store through ``keyring``
(Windows Credential Manager, the macOS keychain, the Secret Service on Linux) under the services its modules name,
and fails closed where there is none (pilot.security.vault).

``isolate(root)`` runs when Heliox's folders are a marked scratch root: from ``pilot.config`` when it finds one (so a
daemon or tool started under a scratch root is covered without asking), from ``pilot.real_folders.use_scratch_folders``
and from the test suite's folder guard. A real install never has that mark (pilot.config ignores HELIOX_TEST_ROOT
without it), so this cannot switch on there by accident. From then on, for the life of the process:

- keyring's backend is the scratch root's own store, ``ScratchCredentialStore``: a file in the root (SCRATCH_FILE),
  owner-only where the system has modes. Nothing real is ever written there, and it goes when the root goes. When
  PYTHON_KEYRING_BACKEND names keyring's fail backend (the test suite and the platform checks set it), there is no
  store at all instead, as on a machine without one. No other value of that variable is honoured: under a scratch
  root it can choose between those two, never the system store.
- every keyring backend class that reaches a system store, and on Windows the credential functions keyring calls,
  refuse with ``CredentialStoreRefused`` (a KeyringError, so Heliox handles it as a failing store) and record the
  attempt (``attempts()``). The test suite fails a test that made one and counts them (CREDENTIAL-GUARD).
- ``allow_real_reads(names)`` is the one way through, for a trial run that needs the person's model provider key:
  exactly those entries of the vault's service are read from the system store when the scratch store has none of its
  own. Only model provider keys (MODEL_KEY_ENTRIES) can be named. They are read only: never written, deleted, listed
  or copied into the scratch store, and their values are never logged. Without a call, nothing is read from the
  system store. With no store at all (the fail backend), nothing is read either.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import json
import logging
import os
import sys
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot.security.credential_store")

# The vault's service in the system store (pilot.security.vault imports it from here).
VAULT_SERVICE = "pilot-ai-command-center"
SCRATCH_FILE = "scratch-credentials.json"
FAIL_BACKEND = "keyring.backends.fail.Keyring"
# The model providers whose key a trial run may read from the system store, and their backup keys (pilot.models.cloud
# tries "<provider>_backup_1" to "_backup_5" after the main one).
MODEL_PROVIDERS = ("openai", "openrouter", "anthropic", "claude", "gemini", "meta")
MODEL_KEY_ENTRIES = frozenset(
    [*MODEL_PROVIDERS, *(f"{provider}_backup_{n}" for provider in MODEL_PROVIDERS for n in range(1, 6))]
)
# keyring's backends that reach a system store; refused under a scratch root. A third-party backend (keyrings.*) too.
_SYSTEM_BACKEND_MODULES = ("Windows", "macOS", "SecretService", "libsecret", "kwallet")
_NOT_SYSTEM = ("keyring.backends.fail", "keyring.backends.null", "keyring.backends.chainer")
# Each method or function refused, and whether it only reads (an allowed read may pass through it).
_BACKEND_METHODS = {"get_password": True, "get_credential": True, "set_password": False, "delete_password": False}
_WINDOWS_CREDENTIAL_FUNCTIONS = {"CredRead": True, "CredEnumerate": True, "CredWrite": False, "CredDelete": False}

_lock = threading.RLock()
_root: Path | None = None
_mode = ""  # "file" (the scratch store) or "none" (keyring's fail backend)
_store: Any = None
_readable: frozenset[str] = frozenset()
_real: Any = None
_real_looked = False
_attempts: list[tuple[str, str]] = []
_permit = threading.local()
_store_class: type | None = None


def _keyring_error() -> type[Exception]:
    try:
        from keyring.errors import KeyringError

        return KeyringError
    except Exception:  # keyring missing: a plain error will do
        return RuntimeError


class CredentialStoreRefused(_keyring_error()):  # type: ignore[misc]
    """Code running under a scratch root or the test suite reached this computer's credential store."""


def isolated() -> Path | None:
    """The scratch root whose store this process uses, or None (a real install)."""
    return _root


def mode() -> str:
    """ "file" (the scratch root's store), "none" (no store at all) or "" (not isolated: a real install)."""
    return _mode


def attempts() -> list[tuple[str, str]]:
    """(test or "outside the test suite", what) for each refused reach for the system store."""
    with _lock:
        return list(_attempts)


def clear() -> None:
    with _lock:
        _attempts.clear()


def isolate(root: Path | str) -> str:
    """Make the scratch root's store this process's credential store, and refuse the system's. Returns the mode.

    Idempotent: the first root of the process stays (pilot.config fixes the folders once, too)."""
    global _root, _mode, _store
    with _lock:
        if _root is not None:
            return _mode
        try:
            import keyring
            import keyring.backends.fail
        except ImportError:  # no keyring at all: nothing can reach a store
            _root, _mode = Path(root), "none"
            return _mode
        none = os.environ.get("PYTHON_KEYRING_BACKEND", "").strip() == FAIL_BACKEND
        store = keyring.backends.fail.Keyring() if none else _scratch_store_class()(Path(root) / SCRATCH_FILE)
        _refuse_system_stores()
        keyring.set_keyring(store)
        _root, _mode, _store = Path(root), "none" if none else "file", store
    logger.info(
        "Scratch folders: secrets go to %s, never to this computer's credential store",
        "no store at all" if none else Path(root) / SCRATCH_FILE,
    )
    return _mode


def check_readable(names: Iterable[str]) -> tuple[str, ...]:
    """The names, when each is a model provider key a trial run may read; ValueError otherwise."""
    if isinstance(names, str):
        raise ValueError("name the entries as a list, such as ('claude',)")
    wanted = tuple(dict.fromkeys(str(name) for name in names))
    refused = [name for name in wanted if name not in MODEL_KEY_ENTRIES]
    if refused:
        raise ValueError(
            f"only a model provider's key can be read from this computer's credential store under scratch folders, "
            f"not {', '.join(map(repr, refused))} (one of {', '.join(MODEL_PROVIDERS)}, or '<provider>_backup_1'..5)"
        )
    return wanted


def allow_real_reads(names: Iterable[str]) -> tuple[str, ...]:
    """Let exactly these vault entries be read, read only, from the system store. Returns the names now readable:
    () when there is no store at all (the test suite), so nothing real is read there either."""
    global _readable
    wanted = check_readable(names)
    with _lock:
        if _root is None:
            raise RuntimeError("allow_real_reads is for scratch folders; a real install reads its own store already")
        if _mode != "file" or not wanted:
            return ()
        _readable = _readable | frozenset(wanted)
    logger.info("Scratch folders: %s read from this computer's credential store, read only", ", ".join(wanted))
    return wanted


def readable() -> frozenset[str]:
    return _readable


def scratch_accounts(service: str) -> list[str]:
    """The accounts the scratch store holds under ``service`` (never the system store's): for credential_cleanup."""
    store = _store
    if _mode != "file" or store is None:
        return []
    return store.accounts(service)


# ── refusing the system store ──


def _where() -> str:
    return os.environ.get("PYTEST_CURRENT_TEST", "").split(" (")[0] or "outside the test suite"


def _refuse(what: str) -> None:
    where = _where()
    with _lock:
        _attempts.append((where, what))
    logger.warning("Refused under scratch folders: %s would reach this computer's credential store", what)
    raise CredentialStoreRefused(
        f"Refused under scratch folders or the test suite: {what} would reach this computer's credential store. "
        "Use a fake store; a trial run that needs a model key names it with use_scratch_folders(real_credentials=...)."
    )


def _refusing(function: Callable[..., Any], what: str, reads: bool) -> Callable[..., Any]:
    @functools.wraps(function)
    def refusing(*args: Any, **kwargs: Any) -> Any:
        if reads and getattr(_permit, "on", False):
            return function(*args, **kwargs)  # an allowed read (allow_real_reads), in this thread, now
        _refuse(what)

    refusing._heliox_refuses = True  # type: ignore[attr-defined]
    return refusing


def _is_system_backend(cls: type) -> bool:
    module = getattr(cls, "__module__", "") or ""
    if not (module.startswith("keyring.backends.") or module.startswith("keyrings.")):
        return False
    return not any(module == name or module.startswith(name + ".") for name in _NOT_SYSTEM)


def _refuse_system_stores() -> None:
    for name in _SYSTEM_BACKEND_MODULES:
        with contextlib.suppress(Exception):  # a backend that cannot load here cannot reach a store either
            importlib.import_module(f"keyring.backends.{name}")
    from keyring.backend import KeyringBackend

    for cls in list(getattr(KeyringBackend, "_classes", ())):
        if not _is_system_backend(cls):
            continue
        for method, reads in _BACKEND_METHODS.items():
            _patch(cls, method, reads)
    for module_name in ("win32ctypes.pywin32.win32cred", "win32cred"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for function, reads in _WINDOWS_CREDENTIAL_FUNCTIONS.items():
            original = getattr(module, function, None)
            if callable(original) and not getattr(original, "_heliox_refuses", False):
                setattr(module, function, _refusing(original, f"{module_name}.{function}", reads))


def _patch(cls: type, method: str, reads: bool) -> None:
    original = cls.__dict__.get(method)
    if original is None or getattr(original, "_heliox_refuses", False):
        return
    setattr(cls, method, _refusing(original, f"{cls.__module__}.{cls.__qualname__}.{method}", reads))


def _system_store() -> Any:
    """The store this computer would use (PYTHON_KEYRING_BACKEND, keyring's own config file, else the best viable
    backend), found once; None when there is none (reads are then absent, as KeyVault fails closed)."""
    global _real, _real_looked
    with _lock:
        if _real_looked:
            return _real
        _real_looked = True
        try:
            import keyring.backends.fail
            import keyring.core
            from keyring import backend

            found = keyring.core.load_env() or keyring.core.load_config()
            if found is None:
                viable = [ring for ring in backend.get_all_keyring() if not isinstance(ring, _scratch_store_class())]
                found = max(viable, key=backend.by_priority, default=None)
            if found is None or isinstance(found, keyring.backends.fail.Keyring):
                return None
            if float(getattr(found, "priority", 0)) <= 0:
                return None
            _real = found
        except Exception:
            logger.warning("Scratch folders: this computer's credential store could not be found")
            _real = None
        return _real


def _read_real(service: str, account: str) -> str | None:
    store = _system_store()
    if store is None:
        return None
    _permit.on = True
    try:
        return store.get_password(service, account)
    finally:
        _permit.on = False


# ── the scratch store ──


def _scratch_store_class() -> type:
    """Defined on first use, under a scratch root only: in a real install keyring never sees the class."""
    global _store_class
    if _store_class is not None:
        return _store_class
    from keyring.backend import KeyringBackend
    from keyring.errors import PasswordDeleteError

    class ScratchCredentialStore(KeyringBackend):
        """Secrets in one owner-only JSON file in the scratch root: {service: {account: secret}}."""

        priority = 1  # a working store for KeyVault; never chosen by keyring itself (it needs a path)

        def __init__(self, path: Path) -> None:
            super().__init__()
            self._path = Path(path)
            self._file_lock = threading.Lock()

        def _load(self) -> dict[str, dict[str, str]]:
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
            return data if isinstance(data, dict) else {}

        def _save(self, data: dict[str, dict[str, str]]) -> None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_name(self._path.name + ".tmp")
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
            os.replace(temporary, self._path)

        def get_password(self, service: str, username: str) -> str | None:
            with self._file_lock:
                value = self._load().get(service, {}).get(username)
            if value is None and service == VAULT_SERVICE and username in _readable:
                return _read_real(service, username)  # read only; never kept here
            return value

        def set_password(self, service: str, username: str, password: str) -> None:
            with self._file_lock:
                data = self._load()
                data.setdefault(service, {})[username] = password
                self._save(data)

        def delete_password(self, service: str, username: str) -> None:
            with self._file_lock:
                data = self._load()
                if username not in data.get(service, {}):
                    raise PasswordDeleteError("not in the scratch store")
                del data[service][username]
                self._save(data)

        def accounts(self, service: str) -> list[str]:
            with self._file_lock:
                return sorted(self._load().get(service, {}))

    _store_class = ScratchCredentialStore
    return _store_class
