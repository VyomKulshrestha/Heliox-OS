"""Encrypted key vault for API key storage.

Primary: GNOME Keyring via libsecret (SecretStorage)
Fallback: AES-256-GCM encrypted file with Argon2id key derivation

API keys are NEVER logged, included in action plans, or sent to local LLMs.
Keys are decrypted only at the moment of an API call.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from pilot.config import DATA_DIR

if TYPE_CHECKING:
    from pilot.config import PilotConfig

logger = logging.getLogger("pilot.security.vault")

VAULT_FILE = DATA_DIR / "vault.enc"
VAULT_SERVICE = "pilot-ai-command-center"


class KeyVault:
    """Secure storage for API keys."""

    def __init__(self, config: PilotConfig) -> None:
        self._config = config
        self._keyring_available = False
        self._cache: dict[str, str] = {}
        self._detect_backend()

    def _detect_backend(self) -> None:
        try:
            import keyring
            import keyring.backends

            kr = keyring.get_keyring()
            self._keyring_available = not isinstance(
                kr,
                keyring.backends.fail.Keyring,  # type: ignore[attr-defined]
            )
        except Exception:
            self._keyring_available = False

        if self._keyring_available:
            logger.info("Using system keyring (libsecret) for API key storage")
        else:
            logger.info("System keyring not available; using encrypted file vault")

    async def get_key(self, provider: str) -> str | None:
        """Retrieve a decrypted API key for the given provider."""
        if provider in self._cache:
            return self._cache[provider]

        key = self._read_key(provider)
        if key:
            self._cache[provider] = key
        return key

    async def store_key(self, provider: str, api_key: str) -> None:
        """Store an API key securely."""
        self._write_key(provider, api_key)
        self._cache[provider] = api_key
        logger.info("API key stored for provider: %s", provider)

    async def delete_key(self, provider: str) -> None:
        """Remove a stored API key."""
        self._remove_key(provider)
        self._cache.pop(provider, None)
        logger.info("API key removed for provider: %s", provider)

    async def list_providers(self) -> list[str]:
        """List providers that have stored keys."""
        if self._keyring_available:
            return self._list_keyring_providers()
        return self._list_file_providers()

    def clear_cache(self) -> None:
        """Clear in-memory key cache."""
        self._cache.clear()

    def _read_key(self, provider: str) -> str | None:
        if self._keyring_available:
            return self._read_from_keyring(provider)
        return self._read_from_file(provider)

    def _write_key(self, provider: str, api_key: str) -> None:
        if self._keyring_available:
            self._write_to_keyring(provider, api_key)
        else:
            self._write_to_file(provider, api_key)

    def _remove_key(self, provider: str) -> None:
        if self._keyring_available:
            self._remove_from_keyring(provider)
        else:
            self._remove_from_file(provider)

    # -- Keyring backend --

    def _read_from_keyring(self, provider: str) -> str | None:
        import keyring

        try:
            return keyring.get_password(VAULT_SERVICE, provider)
        except Exception:
            logger.exception("Failed to read from keyring")
            return None

    def _write_to_keyring(self, provider: str, api_key: str) -> None:
        import keyring

        keyring.set_password(VAULT_SERVICE, provider, api_key)

    def _remove_from_keyring(self, provider: str) -> None:
        import keyring

        with contextlib.suppress(Exception):
            keyring.delete_password(VAULT_SERVICE, provider)

    def _list_keyring_providers(self) -> list[str]:
        known = ["openai", "claude", "gemini"]
        import keyring

        return [p for p in known if keyring.get_password(VAULT_SERVICE, p)]

    # -- Encrypted file backend --

    def _get_vault_data(self) -> dict[str, str]:
        if not VAULT_FILE.exists():
            return {}
        try:
            encrypted = VAULT_FILE.read_bytes()
            return self._decrypt_vault(encrypted)
        except Exception:
            logger.exception("Failed to decrypt vault file")
            return {}

    def _save_vault_data(self, data: dict[str, str]) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        encrypted = self._encrypt_vault(data)
        VAULT_FILE.write_bytes(encrypted)
        os.chmod(VAULT_FILE, 0o600)

    def _read_from_file(self, provider: str) -> str | None:
        data = self._get_vault_data()
        return data.get(provider)

    def _write_to_file(self, provider: str, api_key: str) -> None:
        data = self._get_vault_data()
        data[provider] = api_key
        self._save_vault_data(data)

    def _remove_from_file(self, provider: str) -> None:
        data = self._get_vault_data()
        data.pop(provider, None)
        self._save_vault_data(data)

    def _list_file_providers(self) -> list[str]:
        return list(self._get_vault_data().keys())

    def _derive_key(self, salt: bytes) -> bytes:
        """Derive encryption key using PBKDF2-HMAC-SHA256.

        Uses machine-id as the passphrase for unattended operation.
        For maximum security, users should configure keyring instead.
        """
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        machine_id = self._get_machine_id()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=600_000,
        )
        return kdf.derive(machine_id.encode("utf-8"))

    def _encrypt_vault(self, data: dict[str, str]) -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        plaintext = json.dumps(data).encode("utf-8")
        salt = os.urandom(16)
        key = self._derive_key(salt)
        nonce = os.urandom(12)
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        envelope = {
            "v": 1,
            "salt": base64.b64encode(salt).decode(),
            "nonce": base64.b64encode(nonce).decode(),
            "data": base64.b64encode(ciphertext).decode(),
        }
        return json.dumps(envelope).encode("utf-8")

    def _decrypt_vault(self, raw: bytes) -> dict[str, str]:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        envelope = json.loads(raw)
        salt = base64.b64decode(envelope["salt"])
        nonce = base64.b64decode(envelope["nonce"])
        ciphertext = base64.b64decode(envelope["data"])

        key = self._derive_key(salt)
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return json.loads(plaintext)

    @staticmethod
    def _get_machine_id() -> str:
        """Read a stable per-machine identifier used as the KDF passphrase.

        Tries platform-specific sources in order of preference:
          - Linux: /etc/machine-id or /var/lib/dbus/machine-id
          - macOS: IOPlatformUUID from ioreg
          - Windows: MachineGuid from the Cryptography registry key

        Falls back to a hardcoded constant only when no platform source
        is reachable. The fallback weakens machine-binding (any attacker
        who obtains the vault file can attempt decryption with the known
        constant), so users on platforms where the fallback is hit should
        prefer the system-keyring backend instead.
        """
        import platform
        import subprocess

        # Linux: standard machine-id files written by systemd / D-Bus.
        for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            try:
                value = Path(path).read_text().strip()
                if value:
                    return value
            except OSError:
                continue

        system = platform.system()

        # macOS: IOPlatformUUID is stable across reboots and unique per device.
        if system == "Darwin":
            try:
                result = subprocess.run(
                    ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                for line in result.stdout.splitlines():
                    if "IOPlatformUUID" in line:
                        parts = line.split('"')
                        if len(parts) >= 2:
                            uuid = parts[-2].strip()
                            if uuid:
                                return uuid
            except Exception:
                pass

        # Windows: MachineGuid written by Windows Setup; unique per installation.
        if system == "Windows":
            try:
                import winreg  # available only on Windows

                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\Microsoft\Cryptography",
                ) as key:
                    guid, _ = winreg.QueryValueEx(key, "MachineGuid")
                    if guid:
                        return str(guid)
            except Exception:
                pass

        return "pilot-fallback-id"
