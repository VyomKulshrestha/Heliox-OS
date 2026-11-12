"""API-key storage backed exclusively by the operating system keyring.

Heliox intentionally fails closed when no secure credential backend is
available. Older releases used a machine-identifier-derived encrypted file;
that file is detected and left untouched, but is never decrypted or reused.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from pilot.config import DATA_DIR

if TYPE_CHECKING:
    from pilot.config import PilotConfig

logger = logging.getLogger("pilot.security.vault")

VAULT_SERVICE = "pilot-ai-command-center"
LEGACY_VAULT_FILE = DATA_DIR / "vault.enc"
KNOWN_PROVIDERS = (
    "openai",
    "openrouter",
    "anthropic",
    "claude",
    "gemini",
    "meta",
    "caldav",
    "email",
    "heliox_mesh",
)


class VaultUnavailableError(RuntimeError):
    """Raised when persistent secret storage has no secure OS backend."""


class KeyVault:
    """Store API keys in Windows Credential Manager, Keychain, or Secret Service."""

    def __init__(self, config: PilotConfig) -> None:
        self._config = config
        self._keyring_available = False
        self._backend_name = ""
        self._cache: dict[str, str] = {}
        self._detect_backend()

    @property
    def available(self) -> bool:
        """Whether a secure operating-system credential backend is usable."""
        return self._keyring_available

    @property
    def backend_name(self) -> str:
        """Human-readable keyring backend name for diagnostics."""
        return self._backend_name

    def _detect_backend(self) -> None:
        try:
            import keyring
            import keyring.backends

            backend = keyring.get_keyring()
            priority = float(getattr(backend, "priority", 0))
            self._keyring_available = priority > 0 and not isinstance(
                backend,
                keyring.backends.fail.Keyring,  # type: ignore[attr-defined]
            )
            self._backend_name = type(backend).__name__
        except Exception:
            self._keyring_available = False
            self._backend_name = ""
            logger.debug("Unable to initialize the system keyring", exc_info=True)

        if self._keyring_available:
            logger.info("Using secure OS credential backend: %s", self._backend_name)
        else:
            logger.error("No secure OS credential backend is available; API-key persistence is disabled")

        if LEGACY_VAULT_FILE.exists():
            logger.warning(
                "Legacy vault.enc detected and ignored. Re-enter API keys in Settings so they "
                "are stored in the operating system keyring."
            )

    def _require_backend(self) -> None:
        if not self._keyring_available:
            raise VaultUnavailableError(
                "Secure credential storage is unavailable. Enable Windows Credential Manager, "
                "macOS Keychain, or a Secret Service-compatible keyring, then restart Heliox OS."
            )

    async def get_key(self, provider: str) -> str | None:
        """Retrieve a key without falling back to insecure file storage."""
        if provider in self._cache:
            return self._cache[provider]
        if not self._keyring_available:
            return None

        # In a worker thread, bounded (pilot.security.keyring_guard): a keychain that waits on a dialog must never
        # hold the event loop.
        key = await asyncio.to_thread(self._read_from_keyring, provider)
        if key:
            self._cache[provider] = key
        return key

    async def store_key(self, provider: str, api_key: str) -> None:
        """Persist an API key in the operating system keyring."""
        self._require_backend()
        await asyncio.to_thread(self._write_to_keyring, provider, api_key)
        self._cache[provider] = api_key
        logger.info("API key stored for provider: %s", provider)

    async def delete_key(self, provider: str) -> None:
        """Remove an API key from the operating system keyring."""
        self._require_backend()
        await asyncio.to_thread(self._remove_from_keyring, provider)
        self._cache.pop(provider, None)
        logger.info("API key removed for provider: %s", provider)

    async def list_providers(self) -> list[str]:
        """List known providers that currently have stored credentials."""
        if not self._keyring_available:
            return []
        return await asyncio.to_thread(self._list_keyring_providers)

    def clear_cache(self) -> None:
        """Clear decrypted keys held in process memory."""
        self._cache.clear()

    def _read_from_keyring(self, provider: str) -> str | None:
        from keyring.errors import KeyringError

        from pilot.security import keyring_guard

        try:
            return keyring_guard.get_password(VAULT_SERVICE, provider)
        except KeyringError as exc:
            raise VaultUnavailableError(
                f"The operating system credential store could not read the {provider} key."
            ) from exc

    def _write_to_keyring(self, provider: str, api_key: str) -> None:
        from keyring.errors import KeyringError

        from pilot.security import keyring_guard

        try:
            keyring_guard.set_password(VAULT_SERVICE, provider, api_key)
        except KeyringError as exc:
            raise VaultUnavailableError(
                f"The operating system credential store could not save the {provider} key."
            ) from exc

    def _remove_from_keyring(self, provider: str) -> None:
        from keyring.errors import KeyringError, PasswordDeleteError

        from pilot.security import keyring_guard

        try:
            keyring_guard.delete_password(VAULT_SERVICE, provider)
        except PasswordDeleteError:
            return
        except KeyringError as exc:
            raise VaultUnavailableError(
                f"The operating system credential store could not delete the {provider} key."
            ) from exc

    def _list_keyring_providers(self) -> list[str]:
        return [provider for provider in KNOWN_PROVIDERS if self._read_from_keyring(provider)]
