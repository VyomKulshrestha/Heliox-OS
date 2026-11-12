"""Explicit-consent encrypted neural recording and BrainVision BIDS export."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Self
from uuid import UUID, uuid4

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from pilot.neural.acquisition import NeuralSampleWindow
from pilot.neural.protocol import Identifier, NeuralStimulusMarkerV1, NeuralStreamDescriptorV1

Purpose = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=256)]


class NeuralRecordingError(RuntimeError):
    pass


class NeuralRecordingConsentV1(BaseModel):
    """Purpose-bound, expiring local consent. ``authorized`` must be literal true."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=1, strict=True, ge=1, le=1)
    consent_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    subject_key: Identifier
    purpose: Purpose
    granted_at: datetime
    expires_at: datetime
    retention_days: int = Field(strict=True, ge=1, le=365)
    allow_bids_export: bool = False
    authorized: bool

    @model_validator(mode="after")
    def validate_consent(self) -> Self:
        if self.authorized is not True:
            raise ValueError("raw neural recording requires explicit authorization")
        if self.granted_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("consent timestamps must include a timezone")
        if self.expires_at <= self.granted_at:
            raise ValueError("recording consent must expire after it is granted")
        if self.expires_at > self.granted_at + timedelta(days=self.retention_days):
            raise ValueError("consent expiry cannot exceed the retention period")
        return self

    def require_active(self, *, now: datetime | None = None) -> None:
        current = now or datetime.now(UTC)
        if current >= self.expires_at:
            raise NeuralRecordingError("raw neural recording consent has expired")


class NeuralRecordingKeyStore:
    """Store the per-subject AES key only in the operating-system keyring."""

    SERVICE = "heliox-neural-recordings"

    def get_or_create(self, subject_key: str) -> bytes:
        try:
            import keyring
            import keyring.backends

            backend = keyring.get_keyring()
            if float(getattr(backend, "priority", 0)) <= 0 or isinstance(
                backend,
                keyring.backends.fail.Keyring,  # type: ignore[attr-defined]
            ):
                raise NeuralRecordingError("secure OS key storage is unavailable; recording fails closed")
            from pilot.security import keyring_guard  # bounded: a waiting keychain dialog cannot hold Heliox

            encoded = keyring_guard.get_password(self.SERVICE, subject_key)
            if encoded:
                key = base64.b64decode(encoded, validate=True)
                if len(key) != 32:
                    raise NeuralRecordingError("stored neural recording key is invalid")
                return key
            key = AESGCM.generate_key(bit_length=256)
            keyring_guard.set_password(self.SERVICE, subject_key, base64.b64encode(key).decode("ascii"))
            return key
        except NeuralRecordingError:
            raise
        except Exception as exc:
            raise NeuralRecordingError("secure OS key storage is unavailable; recording fails closed") from exc


class EncryptedNeuralRecorder:
    """Append independently authenticated sample chunks to a local-only container."""

    def __init__(
        self,
        *,
        destination: Path,
        descriptor: NeuralStreamDescriptorV1,
        consent: NeuralRecordingConsentV1,
        key: bytes | None = None,
        key_store: NeuralRecordingKeyStore | None = None,
    ) -> None:
        consent.require_active()
        if consent.session_id != descriptor.session_id:
            raise NeuralRecordingError("recording consent does not match the acquisition session")
        self._destination = destination.expanduser().resolve()
        if self._destination.exists():
            raise NeuralRecordingError("recording destination already exists")
        self._descriptor = descriptor
        self._consent = consent
        self._key = key or (key_store or NeuralRecordingKeyStore()).get_or_create(consent.subject_key)
        if len(self._key) != 32:
            raise NeuralRecordingError("recording encryption key must be 256 bits")
        self._last_sequence = -1
        self._last_marker_sequence = -1
        self._destination.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "type": "header",
            "schema_version": 1,
            "cipher": "AES-256-GCM",
            "descriptor": descriptor.model_dump(mode="json"),
            "consent": consent.model_dump(mode="json"),
        }
        self._destination.write_text(json.dumps(header, sort_keys=True) + "\n", encoding="utf-8")
        try:
            os.chmod(self._destination, 0o600)
        except OSError:
            pass

    @property
    def destination(self) -> Path:
        return self._destination

    @classmethod
    def open_existing(
        cls,
        source: Path,
        *,
        key: bytes | None = None,
        key_store: NeuralRecordingKeyStore | None = None,
    ) -> EncryptedNeuralRecorder:
        path = source.expanduser().resolve()
        try:
            header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            if header.get("type") != "header" or header.get("cipher") != "AES-256-GCM":
                raise ValueError("unsupported recording header")
            descriptor = NeuralStreamDescriptorV1.model_validate(header["descriptor"])
            consent = NeuralRecordingConsentV1.model_validate(header["consent"])
        except (OSError, IndexError, KeyError, ValueError) as exc:
            raise NeuralRecordingError("invalid encrypted neural recording") from exc
        consent.require_active()
        instance = cls.__new__(cls)
        instance._destination = path
        instance._descriptor = descriptor
        instance._consent = consent
        instance._key = key or (key_store or NeuralRecordingKeyStore()).get_or_create(consent.subject_key)
        instance._last_sequence = -1
        instance._last_marker_sequence = -1
        return instance

    def append(self, window: NeuralSampleWindow) -> None:
        self._consent.require_active()
        if self._last_sequence >= 0 and window.sequence_start <= self._last_sequence:
            raise NeuralRecordingError("recording rejected replayed or reordered samples")
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            samples_uv=window.samples_uv,
            timestamps_ns=window.timestamps_ns,
            dropped_before=np.asarray(window.dropped_before, dtype=np.int64),
        )
        plaintext = buffer.getvalue()
        aad = json.dumps(
            {
                "session_id": str(self._descriptor.session_id),
                "sequence_start": window.sequence_start,
                "sequence_end": window.sequence_end,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        nonce = secrets.token_bytes(12)
        ciphertext = AESGCM(self._key).encrypt(nonce, plaintext, aad)
        record = {
            "type": "chunk",
            "sequence_start": window.sequence_start,
            "sequence_end": window.sequence_end,
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            "plaintext_sha256": hashlib.sha256(plaintext).hexdigest(),
        }
        with self._destination.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._last_sequence = window.sequence_end

    def append_marker(self, marker: NeuralStimulusMarkerV1) -> None:
        self._consent.require_active()
        if marker.session_id != self._descriptor.session_id:
            raise NeuralRecordingError("stimulus marker does not match the recording session")
        if marker.sequence <= self._last_marker_sequence:
            raise NeuralRecordingError("recording rejected a replayed stimulus marker")
        plaintext = marker.model_dump_json().encode("utf-8")
        aad = json.dumps(
            {
                "session_id": str(self._descriptor.session_id),
                "marker_sequence": marker.sequence,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        nonce = secrets.token_bytes(12)
        ciphertext = AESGCM(self._key).encrypt(nonce, plaintext, aad)
        record = {
            "type": "marker",
            "marker_sequence": marker.sequence,
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            "plaintext_sha256": hashlib.sha256(plaintext).hexdigest(),
        }
        with self._destination.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._last_marker_sequence = marker.sequence

    def export_bids_brainvision(self, destination: Path) -> Path:
        if not self._consent.allow_bids_export:
            raise NeuralRecordingError("BIDS export was not included in this recording consent")
        self._consent.require_active()
        windows, markers = self._decrypt_records()
        if not windows:
            raise NeuralRecordingError("recording has no sample chunks to export")
        samples = np.concatenate([window.samples_uv for window in windows], axis=1)
        timestamps = np.concatenate([window.timestamps_ns for window in windows])
        subject = "".join(character for character in self._consent.subject_key if character.isalnum())
        if not subject:
            raise NeuralRecordingError("subject key cannot form a BIDS participant label")
        root = destination.expanduser().resolve()
        if root.exists():
            raise NeuralRecordingError("BIDS export destination already exists; export never overwrites data")
        eeg_dir = root / f"sub-{subject}" / "eeg"
        eeg_dir.mkdir(parents=True, exist_ok=True)
        stem = f"sub-{subject}_task-heliox_eeg"
        data_name = f"{stem}.eeg"
        marker_name = f"{stem}.vmrk"
        (eeg_dir / data_name).write_bytes(samples.T.astype("<f4", copy=False).tobytes())
        channel_lines = "\n".join(
            f"Ch{index + 1}={name},,1,uV" for index, name in enumerate(self._descriptor.channel_names)
        )
        (eeg_dir / f"{stem}.vhdr").write_text(
            "Brain Vision Data Exchange Header File Version 1.0\n"
            "[Common Infos]\n"
            f"DataFile={data_name}\nMarkerFile={marker_name}\nDataFormat=BINARY\n"
            "DataOrientation=MULTIPLEXED\n"
            f"NumberOfChannels={self._descriptor.channel_count}\n"
            f"SamplingInterval={1_000_000 / self._descriptor.sample_rate_hz:.6f}\n"
            "[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n"
            f"[Channel Infos]\n{channel_lines}\n",
            encoding="utf-8",
            newline="\n",
        )
        marker_lines = ["Mk1=New Segment,,1,1,0"]
        event_lines = ["onset\tduration\ttrial_type\tvalue"]
        first_timestamp = int(timestamps[0])
        for index, marker in enumerate(markers, start=2):
            label = f"{marker.target_id or 'grid'}:{marker.event.value}"
            sample_position = int(np.searchsorted(timestamps, marker.received_monotonic_ns, side="left")) + 1
            sample_position = min(max(1, sample_position), len(timestamps))
            onset = max(0.0, (marker.received_monotonic_ns - first_timestamp) / 1_000_000_000)
            marker_lines.append(f"Mk{index}=Stimulus,{label},{sample_position},1,0")
            event_lines.append(f"{onset:.9f}\t0\t{marker.event.value}\t{marker.target_id or 'grid'}")
        marker_text = "\n".join(marker_lines)
        (eeg_dir / marker_name).write_text(
            "Brain Vision Data Exchange Marker File, Version 1.0\n"
            f"[Common Infos]\nDataFile={data_name}\n"
            f"[Marker Infos]\n{marker_text}\n",
            encoding="utf-8",
            newline="\n",
        )
        (root / "dataset_description.json").write_text(
            json.dumps(
                {"Name": "Heliox consented neural recording", "BIDSVersion": "1.9.0", "DatasetType": "raw"},
                indent=2,
            ),
            encoding="utf-8",
        )
        (root / "participants.tsv").write_text(
            f"participant_id\nsub-{subject}\n",
            encoding="utf-8",
            newline="\n",
        )
        (eeg_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "TaskName": "heliox",
                    "SamplingFrequency": self._descriptor.sample_rate_hz,
                    "PowerLineFrequency": "n/a",
                    "EEGReference": self._descriptor.reference,
                    "RecordingType": "continuous",
                    "SoftwareFilters": "n/a",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        channels = "name\ttype\tunits\tsampling_frequency\treference\tstatus\n" + "".join(
            f"{name}\tEEG\tuV\t{self._descriptor.sample_rate_hz}\t{self._descriptor.reference}\tgood\n"
            for name in self._descriptor.channel_names
        )
        (eeg_dir / f"{stem}_channels.tsv").write_text(channels, encoding="utf-8", newline="\n")
        (eeg_dir / f"{stem}_events.tsv").write_text(
            "\n".join(event_lines) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return root

    def _decrypt_records(self) -> tuple[list[NeuralSampleWindow], list[NeuralStimulusMarkerV1]]:
        lines = self._destination.read_text(encoding="utf-8").splitlines()
        windows: list[NeuralSampleWindow] = []
        markers: list[NeuralStimulusMarkerV1] = []
        for line in lines[1:]:
            record = json.loads(line)
            if record.get("type") == "marker":
                aad = json.dumps(
                    {
                        "session_id": str(self._descriptor.session_id),
                        "marker_sequence": record["marker_sequence"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                plaintext = self._decrypt_payload(record, aad)
                try:
                    markers.append(NeuralStimulusMarkerV1.model_validate_json(plaintext))
                except ValueError as exc:
                    raise NeuralRecordingError("encrypted stimulus marker is invalid") from exc
                continue
            if record.get("type") != "chunk":
                raise NeuralRecordingError("unknown encrypted recording record")
            aad = json.dumps(
                {
                    "session_id": str(self._descriptor.session_id),
                    "sequence_start": record["sequence_start"],
                    "sequence_end": record["sequence_end"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            plaintext = self._decrypt_payload(record, aad)
            with np.load(io.BytesIO(plaintext), allow_pickle=False) as chunk:
                windows.append(
                    NeuralSampleWindow(
                        chunk["samples_uv"],
                        chunk["timestamps_ns"],
                        int(record["sequence_start"]),
                        int(chunk["dropped_before"].item()),
                    )
                )
        return windows, markers

    def _decrypt_payload(self, record: dict[str, object], aad: bytes) -> bytes:
        try:
            plaintext = AESGCM(self._key).decrypt(
                base64.b64decode(str(record["nonce"]), validate=True),
                base64.b64decode(str(record["ciphertext"]), validate=True),
                aad,
            )
        except Exception as exc:
            raise NeuralRecordingError("encrypted neural record failed authentication") from exc
        if hashlib.sha256(plaintext).hexdigest() != record.get("plaintext_sha256"):
            raise NeuralRecordingError("encrypted neural record hash mismatch")
        return plaintext


def prune_expired_neural_recordings(
    directory: Path,
    *,
    now: datetime | None = None,
    max_files: int = 1000,
) -> tuple[Path, ...]:
    """Delete expired consented ``.neeg`` files from one explicit directory.

    Invalid files and symlinks are never deleted automatically. Failure to
    honor a valid expiry is surfaced so a new recording does not silently
    proceed while expired biosignal data remains retained.
    """

    if not 1 <= max_files <= 10_000:
        raise ValueError("max_files must be between 1 and 10000")
    root = directory.expanduser().resolve()
    if not root.is_dir():
        return ()
    current = now or datetime.now(UTC)
    removed: list[Path] = []
    for candidate in sorted(root.glob("*.neeg"))[:max_files]:
        if candidate.is_symlink():
            continue
        resolved = candidate.resolve()
        if resolved.parent != root:
            continue
        try:
            with resolved.open("r", encoding="utf-8") as stream:
                header_line = stream.readline(65_537)
            if len(header_line) > 65_536:
                continue
            header = json.loads(header_line)
            if header.get("type") != "header" or header.get("cipher") != "AES-256-GCM":
                continue
            consent = NeuralRecordingConsentV1.model_validate(header["consent"])
        except (OSError, KeyError, ValueError):
            continue
        if current < consent.expires_at:
            continue
        try:
            resolved.unlink()
        except OSError as exc:
            raise NeuralRecordingError(f"could not remove expired neural recording: {resolved.name}") from exc
        removed.append(resolved)
    return tuple(removed)


def export_main() -> None:
    parser = argparse.ArgumentParser(description="Export a consented Heliox neural recording to BIDS/BrainVision")
    parser.add_argument("recording", help="Encrypted .neeg recording")
    parser.add_argument("destination", help="Empty or new BIDS dataset directory")
    args = parser.parse_args()
    recorder = EncryptedNeuralRecorder.open_existing(Path(args.recording))
    recorder.export_bids_brainvision(Path(args.destination))


if __name__ == "__main__":
    export_main()
