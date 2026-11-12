"""Per-participant encryption keys for the co-adaptation store (spec section 10.6).

``get`` never creates a key. Only ``get_or_create`` does, and only ``CoAdaptationStore.add_participant`` calls it
(for a new participant at consent grant), so a write for an erased participant fails instead of minting a fresh
key. Keyring calls are synchronous; the store runs them with ``asyncio.to_thread``.
"""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from pilot.coadaptation.core.consent import PARTICIPANT_KEY_PATTERN

_KEY_BYTES = 32
_PARTICIPANT_KEY = re.compile(PARTICIPANT_KEY_PATTERN)
_UNAVAILABLE = "secure OS key storage is unavailable; co-adaptation fails closed"


class CoAdaptationKeyUnavailable(RuntimeError):
    """The OS keyring is missing, insecure or failing. Nothing is stored or read without it."""


class CoAdaptationKeyMissing(LookupError):
    """No key exists for this participant (never created, or erased)."""


@runtime_checkable
class KeyProvider(Protocol):
    def get(self, participant_key: str) -> bytes: ...

    def get_or_create(self, participant_key: str) -> bytes: ...

    def delete(self, participant_key: str) -> bool: ...

    def available(self) -> bool: ...


def _check_participant_key(participant_key: str) -> None:
    # Keyring account names hold the pseudonymous key only, never a display label.
    if not isinstance(participant_key, str) or not _PARTICIPANT_KEY.fullmatch(participant_key):
        raise ValueError("participant keys must match ^p-[0-9a-f]{8}$")


class OsKeyringKeyProvider:
    """Keys in the operating-system keyring, base64 of 32 random bytes per participant.

    Fails closed exactly like the neural recording key store: a fail backend or a backend with priority <= 0
    raises ``CoAdaptationKeyUnavailable``.
    """

    SERVICE = "heliox-co-adaptation"

    def _keyring(self):
        """Return the ``keyring`` module once its backend is known to be secure."""
        try:
            import keyring
            import keyring.backends.fail

            backend = keyring.get_keyring()
            if float(getattr(backend, "priority", 0)) <= 0 or isinstance(backend, keyring.backends.fail.Keyring):
                raise CoAdaptationKeyUnavailable(_UNAVAILABLE)
            return keyring
        except CoAdaptationKeyUnavailable:
            raise
        except Exception as exc:
            raise CoAdaptationKeyUnavailable(_UNAVAILABLE) from exc

    def available(self) -> bool:
        try:
            self._keyring()
        except CoAdaptationKeyUnavailable:
            return False
        return True

    def get(self, participant_key: str) -> bytes:
        _check_participant_key(participant_key)
        keyring = self._keyring()
        try:
            encoded = keyring.get_password(self.SERVICE, participant_key)
        except Exception as exc:
            raise CoAdaptationKeyUnavailable(_UNAVAILABLE) from exc
        if not encoded:
            raise CoAdaptationKeyMissing("no co-adaptation key exists for this participant")
        try:
            key = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise CoAdaptationKeyUnavailable("the stored co-adaptation key is invalid") from exc
        if len(key) != _KEY_BYTES:
            raise CoAdaptationKeyUnavailable("the stored co-adaptation key is invalid")
        return key

    def get_or_create(self, participant_key: str) -> bytes:
        try:
            return self.get(participant_key)
        except CoAdaptationKeyMissing:
            pass
        keyring = self._keyring()
        key = AESGCM.generate_key(bit_length=256)
        try:
            keyring.set_password(self.SERVICE, participant_key, base64.b64encode(key).decode("ascii"))
        except Exception as exc:
            raise CoAdaptationKeyUnavailable(_UNAVAILABLE) from exc
        return key

    def delete(self, participant_key: str) -> bool:
        _check_participant_key(participant_key)
        keyring = self._keyring()
        try:
            if not keyring.get_password(self.SERVICE, participant_key):
                return False
            keyring.delete_password(self.SERVICE, participant_key)
        except Exception as exc:
            raise CoAdaptationKeyUnavailable(_UNAVAILABLE) from exc
        return True


class InMemoryKeyProvider:
    """Keys held in process memory, for tests and simulations. ``available=False`` behaves like a failing keyring."""

    def __init__(self, *, available: bool = True, keys: Mapping[str, bytes] | None = None) -> None:
        self._available = available
        self._keys: dict[str, bytes] = {}
        for participant_key, key in (keys or {}).items():
            _check_participant_key(participant_key)
            if len(key) != _KEY_BYTES:
                raise ValueError("co-adaptation keys are 32 bytes")
            self._keys[participant_key] = bytes(key)

    def set_available(self, available: bool) -> None:
        self._available = available

    def available(self) -> bool:
        return self._available

    def _require_available(self) -> None:
        if not self._available:
            raise CoAdaptationKeyUnavailable(_UNAVAILABLE)

    def get(self, participant_key: str) -> bytes:
        _check_participant_key(participant_key)
        self._require_available()
        try:
            return self._keys[participant_key]
        except KeyError:
            raise CoAdaptationKeyMissing("no co-adaptation key exists for this participant")

    def get_or_create(self, participant_key: str) -> bytes:
        _check_participant_key(participant_key)
        self._require_available()
        if participant_key not in self._keys:
            self._keys[participant_key] = AESGCM.generate_key(bit_length=256)
        return self._keys[participant_key]

    def delete(self, participant_key: str) -> bool:
        _check_participant_key(participant_key)
        self._require_available()
        return self._keys.pop(participant_key, None) is not None

    def __contains__(self, participant_key: object) -> bool:
        return participant_key in self._keys

    def __len__(self) -> int:
        return len(self._keys)
