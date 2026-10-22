"""Secure, least-privileged cross-device content handoff.

Air Handoff deliberately runs on a separate HTTP port from the authenticated
Heliox control daemon.  Mobile receivers can only pair, poll for content that
was explicitly dropped to them, download that encrypted content, and
acknowledge receipt.  They cannot invoke actions, inspect plans, or reach the
desktop JSON-RPC surface.

When the person turns on phone requests, a paired phone may also send a short,
encrypted task request.  The request only appears on the computer: it runs when
the person presses Run there, or, if they allowed it, right away as a background
job limited to a fixed list of read-only actions.  The phone never approves
anything and only ever receives the job's short result.

The HTTP transport is protected at the application layer:

* pairing uses a high-entropy secret carried in the QR URL fragment (fragments
  are never sent in HTTP requests), authenticated ephemeral X25519, HKDF,
  and AES-GCM;
* paired-device requests use timestamped, nonce-bound HMAC signatures;
* transfer metadata and payloads are AES-GCM encrypted for the target device;
* device secrets are persisted only through the operating-system keyring.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import mimetypes
import os
import secrets
import socket
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from pilot.config import DATA_DIR
from pilot.security.vault import KeyVault, VaultUnavailableError

logger = logging.getLogger("pilot.air_handoff")

AIR_HANDOFF_DIR = DATA_DIR / "air_handoff"
DEVICE_METADATA_FILE = AIR_HANDOFF_DIR / "devices.json"
TRANSFER_DIR = AIR_HANDOFF_DIR / "transfers"

PAIRING_TTL_SECONDS = 5 * 60
TRANSFER_TTL_SECONDS = 10 * 60
REQUEST_CLOCK_SKEW_SECONDS = 90
REQUEST_NONCE_TTL_SECONDS = 5 * 60
MAX_PAIR_ATTEMPTS = 12
DEFAULT_MAX_TRANSFER_BYTES = 25 * 1024 * 1024
MAX_REQUEST_CHARS = 500
MAX_OPEN_REQUESTS_PER_DEVICE = 5
MAX_REQUESTS_PER_MINUTE = 6
MAX_KEPT_REQUESTS = 50


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64u(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _canonical_request(method: str, path: str, timestamp: str, nonce: str, body: bytes) -> bytes:
    return "\n".join((method.upper(), path, timestamp, nonce, hashlib.sha256(body).hexdigest())).encode("utf-8")


def _local_ipv4() -> str:
    """Return a LAN-reachable IPv4 address without transmitting data."""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 9))
            address = str(sock.getsockname()[0])
            if address and not address.startswith("127."):
                return address
    except OSError:
        pass
    try:
        address = socket.gethostbyname(socket.gethostname())
        if address and not address.startswith("127."):
            return address
    except OSError:
        pass
    return "127.0.0.1"


@dataclass(slots=True)
class PairedDevice:
    device_id: str
    name: str
    created_at: float
    last_seen_at: float
    revoked: bool = False

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PairingSession:
    session_id: str
    secret: bytes
    created_at: float
    expires_at: float
    base_url: str
    attempts: int = 0

    @property
    def active(self) -> bool:
        return time.time() < self.expires_at and self.attempts < MAX_PAIR_ATTEMPTS

    def public_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "expires_at": self.expires_at,
            "pairing_url": f"{self.base_url}/#pair={_b64u(self.secret)}",
        }


@dataclass(slots=True)
class HandoffDraft:
    draft_id: str
    kind: str
    filename: str
    mime_type: str
    path: str
    size: int
    created_at: float
    expires_at: float

    def public_dict(self) -> dict[str, Any]:
        return {
            "draft_id": self.draft_id,
            "kind": self.kind,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size": self.size,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


@dataclass(slots=True)
class HandoffTransfer:
    transfer_id: str
    target_device_id: str
    kind: str
    filename: str
    mime_type: str
    path: str
    size: int
    created_at: float
    expires_at: float
    status: str = "ready"
    acknowledged_at: float | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "transfer_id": self.transfer_id,
            "kind": self.kind,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size": self.size,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "status": self.status,
        }


@dataclass(slots=True)
class PhoneRequest:
    """A task a paired phone asked for; it runs only as the desktop settings allow."""

    request_id: str
    device_id: str
    device_name: str
    text: str
    created_at: float
    status: str = "waiting"  # waiting | running | done | failed | dismissed
    job_id: str = ""
    result: str = ""
    read_only: bool = False

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)

    def phone_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "text": self.text,
            "status": self.status,
            "created_at": self.created_at,
            "result": self.result,
        }


class AirHandoffError(RuntimeError):
    """A user-facing Air Handoff contract failure."""


class AirHandoffManager:
    """Pair devices and stage explicit one-target content transfers."""

    def __init__(
        self,
        vault: KeyVault,
        *,
        data_dir: Path = AIR_HANDOFF_DIR,
        max_transfer_bytes: int = DEFAULT_MAX_TRANSFER_BYTES,
        screenshot_capture: Callable[[str], Awaitable[Any]] | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._vault = vault
        self._data_dir = data_dir
        self._metadata_file = data_dir / "devices.json"
        self._transfer_dir = data_dir / "transfers"
        self._max_transfer_bytes = max(1, int(max_transfer_bytes))
        self._screenshot_capture = screenshot_capture
        self._now = now
        self._devices: dict[str, PairedDevice] = {}
        self._pairing: PairingSession | None = None
        self._draft: HandoffDraft | None = None
        self._transfers: dict[str, HandoffTransfer] = {}
        self._recent: deque[dict[str, Any]] = deque(maxlen=20)
        self._used_nonces: dict[str, dict[str, float]] = {}
        self._requests: dict[str, PhoneRequest] = {}
        self._request_times: dict[str, deque[float]] = {}
        self._accepting_requests: Callable[[], bool] = lambda: False
        self._on_request: Callable[[PhoneRequest], Awaitable[None]] | None = None
        self._lock = asyncio.Lock()
        self._purge_orphaned_payloads()
        self._load_metadata()

    @staticmethod
    def _vault_key(device_id: str) -> str:
        return f"air_handoff:{device_id}"

    def _load_metadata(self) -> None:
        if not self._metadata_file.exists():
            return
        try:
            payload = json.loads(self._metadata_file.read_text(encoding="utf-8"))
            for raw in payload.get("devices", []):
                device = PairedDevice(
                    device_id=str(raw["device_id"]),
                    name=str(raw["name"]),
                    created_at=float(raw["created_at"]),
                    last_seen_at=float(raw.get("last_seen_at", raw["created_at"])),
                    revoked=bool(raw.get("revoked", False)),
                )
                self._devices[device.device_id] = device
        except Exception:
            logger.warning("Ignoring invalid Air Handoff device metadata", exc_info=True)

    def _save_metadata(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        payload = {"devices": [device.public_dict() for device in self._devices.values()]}
        temporary = self._metadata_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, self._metadata_file)
        try:
            os.chmod(self._metadata_file, 0o600)
        except OSError:
            pass

    async def _device_secret(self, device_id: str) -> bytes:
        device = self._devices.get(device_id)
        if device is None or device.revoked:
            raise AirHandoffError("Unknown or revoked Air Handoff device")
        encoded = await self._vault.get_key(self._vault_key(device_id))
        if not encoded:
            raise AirHandoffError("The device credential is unavailable; pair the device again")
        try:
            secret = _unb64u(encoded)
        except Exception as exc:
            raise AirHandoffError("The stored device credential is invalid") from exc
        if len(secret) != 32:
            raise AirHandoffError("The stored device credential is invalid")
        return secret

    def start_pairing(self, base_url: str) -> dict[str, Any]:
        if not self._vault.available:
            raise AirHandoffError("Secure OS credential storage is required before pairing a phone")
        now = self._now()
        self._pairing = PairingSession(
            session_id=str(uuid.uuid4()),
            secret=secrets.token_bytes(32),
            created_at=now,
            expires_at=now + PAIRING_TTL_SECONDS,
            base_url=base_url.rstrip("/"),
        )
        return self._pairing.public_dict()

    def cancel_pairing(self) -> None:
        self._pairing = None

    def pairing_status(self) -> dict[str, Any] | None:
        if self._pairing is None or not self._pairing.active:
            self._pairing = None
            return None
        return self._pairing.public_dict()

    async def complete_pairing(
        self,
        *,
        device_name: str,
        client_public_key: bytes,
        client_proof: bytes,
    ) -> dict[str, str]:
        async with self._lock:
            pairing = self._pairing
            if pairing is None or not pairing.active:
                self._pairing = None
                raise AirHandoffError("Pairing is not active or has expired")
            pairing.attempts += 1
            if not 1 <= len(device_name.strip()) <= 80:
                raise AirHandoffError("Device name must contain 1 to 80 characters")

            expected = hmac.new(
                pairing.secret,
                b"pair-v1:" + client_public_key,
                hashlib.sha256,
            ).digest()
            if not hmac.compare_digest(expected, client_proof):
                raise AirHandoffError("Pairing proof was rejected")

            try:
                client_key = x25519.X25519PublicKey.from_public_bytes(client_public_key)
            except ValueError as exc:
                raise AirHandoffError("The phone supplied an invalid pairing key") from exc

            server_private = x25519.X25519PrivateKey.generate()
            server_public = server_private.public_key().public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw,
            )
            try:
                shared = server_private.exchange(client_key)
            except ValueError as exc:
                raise AirHandoffError("The phone supplied an invalid pairing key") from exc
            wrapping_key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=pairing.secret,
                info=b"heliox-air-handoff-pair-v1",
            ).derive(shared)

            device_id = secrets.token_urlsafe(12)
            device_secret = secrets.token_bytes(32)
            now = self._now()
            device = PairedDevice(
                device_id=device_id,
                name=device_name.strip(),
                created_at=now,
                last_seen_at=now,
            )
            try:
                await self._vault.store_key(self._vault_key(device_id), _b64u(device_secret))
            except VaultUnavailableError as exc:
                raise AirHandoffError(str(exc)) from exc
            self._devices[device_id] = device
            self._save_metadata()

            credential = json.dumps(
                {"device_id": device_id, "device_secret": _b64u(device_secret)},
                separators=(",", ":"),
            ).encode("utf-8")
            nonce = secrets.token_bytes(12)
            encrypted = AESGCM(wrapping_key).encrypt(
                nonce,
                credential,
                b"heliox-air-handoff-credential-v1",
            )
            server_proof = hmac.new(
                pairing.secret,
                b"server-v1:" + server_public + client_public_key,
                hashlib.sha256,
            ).digest()
            self._pairing = None
            self._recent.appendleft({"event": "paired", "device_id": device_id, "name": device.name, "at": now})
            return {
                "server_public_key": _b64u(server_public),
                "server_proof": _b64u(server_proof),
                "nonce": _b64u(nonce),
                "credential": _b64u(encrypted),
            }

    async def list_devices(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for device in self._devices.values():
            if device.revoked:
                continue
            item = device.public_dict()
            item["credential_available"] = bool(await self._vault.get_key(self._vault_key(device.device_id)))
            result.append(item)
        return sorted(result, key=lambda item: item["last_seen_at"], reverse=True)

    async def revoke_device(self, device_id: str) -> None:
        async with self._lock:
            device = self._devices.get(device_id)
            if device is None:
                raise AirHandoffError("Unknown Air Handoff device")
            try:
                await self._vault.delete_key(self._vault_key(device_id))
            except VaultUnavailableError as exc:
                raise AirHandoffError(str(exc)) from exc
            device.revoked = True
            self._save_metadata()
            self._used_nonces.pop(device_id, None)
            for transfer_id, transfer in list(self._transfers.items()):
                if transfer.target_device_id != device_id:
                    continue
                self._delete_path(Path(transfer.path))
                self._transfers.pop(transfer_id, None)
            self._recent.appendleft(
                {"event": "revoked", "device_id": device_id, "name": device.name, "at": self._now()}
            )

    async def authenticate_request(
        self,
        *,
        device_id: str,
        method: str,
        path: str,
        timestamp: str,
        nonce: str,
        signature: str,
        body: bytes,
    ) -> PairedDevice:
        try:
            request_time = float(timestamp)
        except ValueError as exc:
            raise AirHandoffError("Invalid request timestamp") from exc
        now = self._now()
        if abs(now - request_time) > REQUEST_CLOCK_SKEW_SECONDS:
            raise AirHandoffError("The phone clock is too far out of sync")
        if not nonce or len(nonce) > 200:
            raise AirHandoffError("Invalid request nonce")

        secret = await self._device_secret(device_id)
        try:
            supplied = _unb64u(signature)
        except Exception as exc:
            raise AirHandoffError("Invalid request signature") from exc
        expected = hmac.new(
            secret,
            _canonical_request(method, path, timestamp, nonce, body),
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(expected, supplied):
            raise AirHandoffError("Request signature was rejected")

        device_nonces = self._used_nonces.setdefault(device_id, {})
        cutoff = now - REQUEST_NONCE_TTL_SECONDS
        for used_nonce, used_at in list(device_nonces.items()):
            if used_at < cutoff:
                device_nonces.pop(used_nonce, None)
        if nonce in device_nonces:
            raise AirHandoffError("Replay request rejected")
        device_nonces[nonce] = now

        device = self._devices[device_id]
        previous_last_seen = device.last_seen_at
        device.last_seen_at = now
        if now - previous_last_seen >= 30:
            self._save_metadata()
        return device

    async def encrypt_json_for(self, device_id: str, payload: Any, *, aad: bytes) -> dict[str, str]:
        secret = await self._device_secret(device_id)
        nonce = secrets.token_bytes(12)
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ciphertext = AESGCM(secret).encrypt(nonce, encoded, aad)
        return {"nonce": _b64u(nonce), "ciphertext": _b64u(ciphertext)}

    async def encrypt_bytes_for(self, device_id: str, payload: bytes, *, aad: bytes) -> bytes:
        secret = await self._device_secret(device_id)
        nonce = secrets.token_bytes(12)
        return nonce + AESGCM(secret).encrypt(nonce, payload, aad)

    def _clear_expired(self) -> None:
        now = self._now()
        if self._draft and self._draft.expires_at <= now:
            self._delete_path(Path(self._draft.path))
            self._draft = None
        for transfer_id, transfer in list(self._transfers.items()):
            if transfer.expires_at <= now:
                self._recent.appendleft(
                    {
                        "event": "expired",
                        "transfer_id": transfer_id,
                        "device_id": transfer.target_device_id,
                        "at": now,
                    }
                )
                self._delete_path(Path(transfer.path))
                self._transfers.pop(transfer_id, None)
            elif transfer.status == "acknowledged":
                self._delete_path(Path(transfer.path))
                self._transfers.pop(transfer_id, None)

    @staticmethod
    def _delete_path(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove expired Air Handoff payload %s", path)

    def _purge_orphaned_payloads(self) -> None:
        """Remove transfer bytes that cannot be recovered after a restart."""

        if not self._transfer_dir.is_dir():
            return
        try:
            entries = list(self._transfer_dir.iterdir())
        except OSError:
            logger.warning("Could not inspect stale Air Handoff payloads", exc_info=True)
            return
        for entry in entries:
            if entry.is_file() or entry.is_symlink():
                self._delete_path(entry)

    async def clear_ephemeral(self) -> None:
        """Discard non-persistent pairing, draft, transfer, and replay state."""

        async with self._lock:
            self._pairing = None
            self._draft = None
            self._transfers.clear()
            self._used_nonces.clear()
            self._purge_orphaned_payloads()

    async def grab_screenshot(self) -> dict[str, Any]:
        if self._screenshot_capture is None:
            from pilot.system.screen import screenshot

            capture = screenshot
        else:
            capture = self._screenshot_capture
        self._transfer_dir.mkdir(parents=True, exist_ok=True)
        draft_id = secrets.token_urlsafe(10)
        path = self._transfer_dir / f"{draft_id}.png"
        await capture(str(path))
        return await self._create_draft(path=path, kind="screenshot", mime_type="image/png")

    async def grab_file(self, path_value: str) -> dict[str, Any]:
        source = Path(path_value).expanduser().resolve()
        if not source.is_file():
            raise AirHandoffError("The selected file does not exist")
        with source.open("rb") as handle:
            payload = handle.read(self._max_transfer_bytes + 1)
        if len(payload) > self._max_transfer_bytes:
            raise AirHandoffError(
                f"Air Handoff payload exceeds the {self._max_transfer_bytes // (1024 * 1024)} MB limit"
            )
        self._transfer_dir.mkdir(parents=True, exist_ok=True)
        snapshot = self._transfer_dir / f"{secrets.token_urlsafe(10)}{source.suffix[:16]}"
        snapshot.write_bytes(payload)
        result = await self._create_draft(path=snapshot, kind="file")
        if self._draft is not None:
            self._draft.filename = source.name[:180]
            result = self._draft.public_dict()
        return result

    async def grab_text(self, text: str, *, filename: str = "heliox-note.txt") -> dict[str, Any]:
        encoded = text.encode("utf-8")
        if not encoded:
            raise AirHandoffError("There is no text to hand off")
        self._transfer_dir.mkdir(parents=True, exist_ok=True)
        path = self._transfer_dir / f"{secrets.token_urlsafe(10)}.txt"
        path.write_bytes(encoded)
        result = await self._create_draft(path=path, kind="text", mime_type="text/plain")
        if self._draft is not None:
            self._draft.filename = Path(filename).name[:180] or "heliox-note.txt"
            result = self._draft.public_dict()
        return result

    async def _create_draft(
        self,
        *,
        path: Path,
        kind: str,
        mime_type: str | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            self._clear_expired()
            size = path.stat().st_size
            if size > self._max_transfer_bytes:
                if path.parent == self._transfer_dir:
                    self._delete_path(path)
                raise AirHandoffError(
                    f"Air Handoff payload exceeds the {self._max_transfer_bytes // (1024 * 1024)} MB limit"
                )
            if self._draft is not None and Path(self._draft.path).parent == self._transfer_dir:
                self._delete_path(Path(self._draft.path))
            now = self._now()
            self._draft = HandoffDraft(
                draft_id=secrets.token_urlsafe(10),
                kind=kind,
                filename=path.name[:180],
                mime_type=mime_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                path=str(path),
                size=size,
                created_at=now,
                expires_at=now + TRANSFER_TTL_SECONDS,
            )
            try:
                if path.parent == self._transfer_dir:
                    os.chmod(path, 0o600)
            except OSError:
                pass
            self._recent.appendleft({"event": "grabbed", "draft_id": self._draft.draft_id, "at": now})
            return self._draft.public_dict()

    async def cancel_draft(self) -> None:
        async with self._lock:
            if self._draft and Path(self._draft.path).parent == self._transfer_dir:
                self._delete_path(Path(self._draft.path))
            self._draft = None

    async def drop(self, target_device_id: str) -> dict[str, Any]:
        async with self._lock:
            self._clear_expired()
            await self._device_secret(target_device_id)
            if self._draft is None:
                raise AirHandoffError("Grab content before dropping it to a phone")
            draft = self._draft
            transfer_id = secrets.token_urlsafe(12)
            now = self._now()
            transfer = HandoffTransfer(
                transfer_id=transfer_id,
                target_device_id=target_device_id,
                kind=draft.kind,
                filename=draft.filename,
                mime_type=draft.mime_type,
                path=draft.path,
                size=draft.size,
                created_at=now,
                expires_at=now + TRANSFER_TTL_SECONDS,
            )
            self._transfers[transfer_id] = transfer
            self._draft = None
            device = self._devices[target_device_id]
            self._recent.appendleft(
                {
                    "event": "dropped",
                    "transfer_id": transfer_id,
                    "device_id": target_device_id,
                    "name": device.name,
                    "at": now,
                }
            )
            result = transfer.public_dict()
            result["target_device_id"] = target_device_id
            result["target_device_name"] = device.name
            return result

    async def pending_for(self, device_id: str) -> list[dict[str, Any]]:
        async with self._lock:
            self._clear_expired()
            await self._device_secret(device_id)
            return [
                transfer.public_dict()
                for transfer in self._transfers.values()
                if transfer.target_device_id == device_id and transfer.status == "ready"
            ]

    async def transfer_bytes(self, device_id: str, transfer_id: str) -> tuple[HandoffTransfer, bytes]:
        async with self._lock:
            self._clear_expired()
            await self._device_secret(device_id)
            transfer = self._transfers.get(transfer_id)
            if transfer is None or transfer.target_device_id != device_id or transfer.status != "ready":
                raise AirHandoffError("Transfer is unavailable")
            path = Path(transfer.path)
            if not path.is_file() or path.stat().st_size != transfer.size:
                raise AirHandoffError("Transfer payload is unavailable")
            return transfer, path.read_bytes()

    async def acknowledge(self, device_id: str, transfer_id: str) -> None:
        async with self._lock:
            transfer = self._transfers.get(transfer_id)
            if transfer is None or transfer.target_device_id != device_id:
                raise AirHandoffError("Transfer is unavailable")
            transfer.status = "acknowledged"
            transfer.acknowledged_at = self._now()
            self._recent.appendleft(
                {
                    "event": "acknowledged",
                    "transfer_id": transfer_id,
                    "device_id": device_id,
                    "at": transfer.acknowledged_at,
                }
            )

    # ------------------------------------------------------------------ phone requests

    def set_request_handler(
        self,
        *,
        accepting: Callable[[], bool],
        on_request: Callable[[PhoneRequest], Awaitable[None]] | None,
    ) -> None:
        """Decide whether phones may send requests, and who is told when one arrives."""
        self._accepting_requests = accepting
        self._on_request = on_request

    @property
    def accepting_requests(self) -> bool:
        try:
            return bool(self._accepting_requests())
        except Exception:
            return False

    async def submit_request(self, device_id: str, envelope: Any) -> PhoneRequest:
        """Decrypt and queue a phone's request. Raises AirHandoffError when it is refused."""
        if not self.accepting_requests:
            raise AirHandoffError("This computer is not accepting requests from phones")
        if not isinstance(envelope, dict):
            raise AirHandoffError("Malformed request")
        secret = await self._device_secret(device_id)
        try:
            clear = AESGCM(secret).decrypt(
                _unb64u(str(envelope.get("nonce", ""))), _unb64u(str(envelope.get("ciphertext", ""))), b"request-v1"
            )
            payload = json.loads(clear.decode("utf-8"))
        except Exception as exc:
            raise AirHandoffError("The request could not be decrypted") from exc
        text = " ".join(str(payload.get("text", "")).split()) if isinstance(payload, dict) else ""
        if not text:
            raise AirHandoffError("The request is empty")
        if len(text) > MAX_REQUEST_CHARS:
            raise AirHandoffError(f"Requests are limited to {MAX_REQUEST_CHARS} characters")
        async with self._lock:
            now = self._now()
            times = self._request_times.setdefault(device_id, deque(maxlen=MAX_REQUESTS_PER_MINUTE))
            if len(times) == MAX_REQUESTS_PER_MINUTE and now - times[0] < 60:
                raise AirHandoffError("Too many requests; wait a minute")
            open_count = sum(
                1 for item in self._requests.values() if item.device_id == device_id and item.status == "waiting"
            )
            if open_count >= MAX_OPEN_REQUESTS_PER_DEVICE:
                raise AirHandoffError("This phone already has requests waiting on the computer")
            times.append(now)
            device = self._devices[device_id]
            request = PhoneRequest(
                request_id=secrets.token_urlsafe(9),
                device_id=device_id,
                device_name=device.name,
                text=text,
                created_at=now,
            )
            self._requests[request.request_id] = request
            while len(self._requests) > MAX_KEPT_REQUESTS:
                self._requests.pop(next(iter(self._requests)))
            self._recent.appendleft(
                {"event": "request", "request_id": request.request_id, "device_id": device_id, "at": now}
            )
        if self._on_request is not None:
            await self._on_request(request)
        return request

    def requests_for(self, device_id: str) -> list[dict[str, Any]]:
        mine = [item for item in self._requests.values() if item.device_id == device_id]
        return [item.phone_dict() for item in reversed(mine[-10:])]

    def list_requests(self) -> list[dict[str, Any]]:
        return [item.public_dict() for item in reversed(list(self._requests.values()))]

    def get_request(self, request_id: str) -> PhoneRequest | None:
        return self._requests.get(request_id)

    def request_for_job(self, job_id: str) -> PhoneRequest | None:
        return next((item for item in self._requests.values() if job_id and item.job_id == job_id), None)

    async def send_text(self, device_id: str, text: str, *, filename: str = "heliox-answer.txt") -> dict[str, Any]:
        """Queue a text answer for one paired phone without touching the person's held draft."""
        async with self._lock:
            self._clear_expired()
            await self._device_secret(device_id)
            encoded = text.encode("utf-8")[: self._max_transfer_bytes]
            self._transfer_dir.mkdir(parents=True, exist_ok=True)
            path = self._transfer_dir / f"{secrets.token_urlsafe(10)}.txt"
            path.write_bytes(encoded)
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
            now = self._now()
            transfer = HandoffTransfer(
                transfer_id=secrets.token_urlsafe(12),
                target_device_id=device_id,
                kind="text",
                filename=Path(filename).name[:180],
                mime_type="text/plain",
                path=str(path),
                size=len(encoded),
                created_at=now,
                expires_at=now + TRANSFER_TTL_SECONDS,
            )
            self._transfers[transfer.transfer_id] = transfer
            return transfer.public_dict()

    async def status(self) -> dict[str, Any]:
        self._clear_expired()
        return {
            "paired_devices": await self.list_devices(),
            "pairing": self.pairing_status(),
            "draft": self._draft.public_dict() if self._draft else None,
            "ready_transfers": sum(transfer.status == "ready" for transfer in self._transfers.values()),
            "recent": list(self._recent),
            "max_transfer_bytes": self._max_transfer_bytes,
            "secure_storage_available": self._vault.available,
            "accepting_requests": self.accepting_requests,
            "requests": self.list_requests()[:10],
        }


class AirHandoffServer:
    """Small aiohttp receiver service with no access to Heliox execution APIs."""

    def __init__(self, manager: AirHandoffManager, *, host: str, port: int) -> None:
        self.manager = manager
        self.host = host
        self.port = int(port)
        self._runner: Any = None
        self._site: Any = None
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    @property
    def base_url(self) -> str:
        return f"http://{_local_ipv4()}:{self.port}"

    async def start(self) -> None:
        if self._running:
            return
        from aiohttp import web

        app = web.Application(client_max_size=512 * 1024)
        app["air_handoff"] = self
        app.add_routes(
            [
                web.get("/", self._index),
                web.get("/app.js", self._javascript),
                web.get("/styles.css", self._styles),
                web.get("/api/status", self._public_status),
                web.post("/api/pair", self._pair),
                web.get("/api/pending", self._pending),
                web.get("/api/transfers/{transfer_id}", self._transfer),
                web.post("/api/transfers/{transfer_id}/ack", self._ack),
                web.get("/api/requests", self._requests_list),
                web.post("/api/requests", self._request_submit),
            ]
        )
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        if self.port == 0 and self._site._server and self._site._server.sockets:
            self.port = int(self._site._server.sockets[0].getsockname()[1])
        self._running = True
        logger.info("Air Handoff receiver ready at %s", self.base_url)

    async def stop(self) -> None:
        self.manager.cancel_pairing()
        if self._runner is not None:
            await self._runner.cleanup()
        self._runner = None
        self._site = None
        self._running = False

    @staticmethod
    def _asset_path(name: str) -> Path:
        return Path(__file__).with_name("air_handoff_web") / name

    @staticmethod
    def _security_headers(response: Any) -> Any:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' blob: data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    async def _asset_response(self, name: str, content_type: str) -> Any:
        from aiohttp import web

        path = self._asset_path(name)
        if not path.is_file():
            raise web.HTTPNotFound()
        response = web.Response(body=path.read_bytes(), content_type=content_type)
        return self._security_headers(response)

    async def _index(self, _request: Any) -> Any:
        return await self._asset_response("index.html", "text/html")

    async def _javascript(self, _request: Any) -> Any:
        return await self._asset_response("app.js", "text/javascript")

    async def _styles(self, _request: Any) -> Any:
        return await self._asset_response("styles.css", "text/css")

    async def _public_status(self, _request: Any) -> Any:
        from aiohttp import web

        response = web.json_response(
            {
                "service": "Heliox Air Handoff",
                "pairing_active": self.manager.pairing_status() is not None,
            }
        )
        return self._security_headers(response)

    async def _pair(self, request: Any) -> Any:
        from aiohttp import web

        try:
            raw = await request.json()
            result = await self.manager.complete_pairing(
                device_name=str(raw.get("device_name", "")),
                client_public_key=_unb64u(str(raw.get("client_public_key", ""))),
                client_proof=_unb64u(str(raw.get("client_proof", ""))),
            )
            response = web.json_response(result)
        except (AirHandoffError, ValueError, json.JSONDecodeError) as exc:
            response = web.json_response({"error": str(exc)}, status=403)
        return self._security_headers(response)

    async def _authenticate(self, request: Any, body: bytes = b"") -> PairedDevice:
        return await self.manager.authenticate_request(
            device_id=request.headers.get("X-Heliox-Device", ""),
            method=request.method,
            path=request.path,
            timestamp=request.headers.get("X-Heliox-Time", ""),
            nonce=request.headers.get("X-Heliox-Nonce", ""),
            signature=request.headers.get("X-Heliox-Signature", ""),
            body=body,
        )

    async def _pending(self, request: Any) -> Any:
        from aiohttp import web

        try:
            device = await self._authenticate(request)
            pending = await self.manager.pending_for(device.device_id)
            encrypted = await self.manager.encrypt_json_for(device.device_id, pending, aad=b"pending-v1")
            response = web.json_response(encrypted)
        except AirHandoffError as exc:
            response = web.json_response({"error": str(exc)}, status=403)
        return self._security_headers(response)

    async def _transfer(self, request: Any) -> Any:
        from aiohttp import web

        try:
            device = await self._authenticate(request)
            transfer_id = str(request.match_info["transfer_id"])
            transfer, payload = await self.manager.transfer_bytes(device.device_id, transfer_id)
            aad = f"transfer-v1:{transfer_id}".encode()
            encrypted = await self.manager.encrypt_bytes_for(device.device_id, payload, aad=aad)
            response = web.Response(
                body=encrypted,
                content_type="application/vnd.heliox.encrypted",
            )
        except AirHandoffError as exc:
            response = web.json_response({"error": str(exc)}, status=403)
        return self._security_headers(response)

    async def _ack(self, request: Any) -> Any:
        from aiohttp import web

        body = await request.read()
        try:
            device = await self._authenticate(request, body)
            transfer_id = str(request.match_info["transfer_id"])
            await self.manager.acknowledge(device.device_id, transfer_id)
            response = web.json_response({"status": "ok"})
        except AirHandoffError as exc:
            response = web.json_response({"error": str(exc)}, status=403)
        return self._security_headers(response)

    async def _requests_list(self, request: Any) -> Any:
        from aiohttp import web

        try:
            device = await self._authenticate(request)
            payload = {
                "accepting": self.manager.accepting_requests,
                "requests": self.manager.requests_for(device.device_id),
            }
            encrypted = await self.manager.encrypt_json_for(device.device_id, payload, aad=b"requests-v1")
            response = web.json_response(encrypted)
        except AirHandoffError as exc:
            response = web.json_response({"error": str(exc)}, status=403)
        return self._security_headers(response)

    async def _request_submit(self, request: Any) -> Any:
        from aiohttp import web

        body = await request.read()
        try:
            device = await self._authenticate(request, body)
            try:
                envelope = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise AirHandoffError("Malformed request") from exc
            submitted = await self.manager.submit_request(device.device_id, envelope)
            response = web.json_response(
                {"status": "received", "request_id": submitted.request_id, "state": submitted.status}
            )
        except AirHandoffError as exc:
            response = web.json_response({"error": str(exc)}, status=403)
        return self._security_headers(response)


__all__ = [
    "AirHandoffError",
    "AirHandoffManager",
    "AirHandoffServer",
    "HandoffDraft",
    "HandoffTransfer",
    "PairedDevice",
    "PhoneRequest",
    "_b64u",
    "_canonical_request",
    "_sha256",
    "_unb64u",
]
