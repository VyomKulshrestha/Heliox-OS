"""Consent purposes and the core consent record (section 10.2-10.3 of the co-adaptation spec).

``ConsentRecord`` is the pure twin of the adapter's pydantic ``CoAdaptationConsentV1``: the same fields, the same
validation (raising ``ValueError``) and the same JSON from ``to_dict()``. SDK hosts use it directly; the Heliox
adapter converts with ``to_core()``. Nothing here reads the clock: every time is passed in.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Self
from uuid import UUID, uuid4


class CoAdaptationPurpose(StrEnum):
    SELECTION_STUDY = "selection_study"
    ADAPTIVE_ASSISTANCE = "adaptive_assistance"
    CROSS_SESSION_LEARNING = "cross_session_learning"
    GAZE_PRIOR = "gaze_prior"
    HAND_POINTER = "hand_pointer"
    RESEARCH_EXPORT = "research_export"


PARTICIPANT_KEY_PATTERN = r"^p-[0-9a-f]{8}$"
IDENTIFIER_PATTERN = r"^[A-Za-z0-9._:-]{1,128}$"
CONSENT_TEXT_VERSION_PATTERN = r"^coadapt-consent-v[0-9]+$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"

_PARTICIPANT_KEY_RE = re.compile(PARTICIPANT_KEY_PATTERN)
_IDENTIFIER_RE = re.compile(IDENTIFIER_PATTERN)
_TEXT_VERSION_RE = re.compile(CONSENT_TEXT_VERSION_PATTERN)
_SHA256_RE = re.compile(SHA256_PATTERN)
_EXPIRY_TOLERANCE = timedelta(seconds=1)
_PURPOSE_ORDER = tuple(CoAdaptationPurpose)

CONSENT_ERROR_CODES = frozenset({"consent_expired", "consent_revoked", "purpose_missing"})


class ConsentError(Exception):
    """A consent check failed. ``code`` is one of ``consent_expired``, ``consent_revoked``, ``purpose_missing``."""

    def __init__(self, code: str) -> None:
        if code not in CONSENT_ERROR_CODES:
            raise ValueError("unknown consent error code")
        super().__init__(code)
        self.code = code


def _matches(pattern: re.Pattern[str], value: object) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _is_aware(value: object) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def _strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def format_datetime(value: datetime) -> str:
    """ISO-8601 with an explicit offset, written the way pydantic's JSON mode writes it ("Z" for UTC)."""
    text = value.isoformat()
    return text[:-6] + "Z" if text.endswith("+00:00") else text


def parse_datetime(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{name} is not a valid ISO-8601 timestamp") from None
    if not _is_aware(parsed):
        raise ValueError(f"{name} must carry a UTC offset")
    return parsed


def _parse_uuid(value: object, name: str) -> UUID:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a UUID string")
    try:
        return UUID(value)
    except ValueError:
        raise ValueError(f"{name} is not a valid UUID") from None


def _check_keys(d: Mapping[str, Any], required: Iterable[str], optional: Iterable[str], record: str) -> None:
    if not isinstance(d, Mapping):
        raise ValueError(f"{record} must be an object")
    required = set(required)
    allowed = required | set(optional)
    unknown = [key for key in d if key not in allowed]
    if unknown:
        raise ValueError(f"{record} has unknown keys: {sorted(str(key) for key in unknown)}")
    missing = sorted(required - set(d))
    if missing:
        raise ValueError(f"{record} is missing keys: {missing}")


@dataclass(frozen=True, slots=True, kw_only=True)
class ConsentRecord:
    """Purpose-bound, expiring, revocable consent of one participant (section 10.3)."""

    schema_version: int = 1
    consent_id: UUID = field(default_factory=uuid4)
    participant_key: str
    purposes: frozenset[CoAdaptationPurpose]
    consent_text_version: str
    consent_text_sha256: str
    protocol_id: str | None = None
    protocol_sha256: str | None = None
    granted_at: datetime
    expires_at: datetime
    retention_days: int
    authorized: bool
    replaces_consent_id: UUID | None = None

    def __post_init__(self) -> None:
        if not _strict_int(self.schema_version) or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if not isinstance(self.consent_id, UUID):
            raise ValueError("consent_id must be a UUID")
        if not _matches(_PARTICIPANT_KEY_RE, self.participant_key):
            raise ValueError("participant_key must match ^p-[0-9a-f]{8}$")
        if not isinstance(self.purposes, frozenset) or not self.purposes:
            raise ValueError("purposes must be a non-empty set")
        if not all(isinstance(purpose, CoAdaptationPurpose) for purpose in self.purposes):
            raise ValueError("purposes must be CoAdaptationPurpose values")
        if CoAdaptationPurpose.SELECTION_STUDY not in self.purposes:
            raise ValueError("purposes must contain selection_study")
        if not _matches(_TEXT_VERSION_RE, self.consent_text_version):
            raise ValueError("consent_text_version must match ^coadapt-consent-v[0-9]+$")
        if not _matches(_SHA256_RE, self.consent_text_sha256):
            raise ValueError("consent_text_sha256 must be 64 lowercase hex characters")
        if self.protocol_id is not None and not _matches(_IDENTIFIER_RE, self.protocol_id):
            raise ValueError("protocol_id is not a valid identifier")
        if self.protocol_sha256 is not None and not _matches(_SHA256_RE, self.protocol_sha256):
            raise ValueError("protocol_sha256 must be 64 lowercase hex characters")
        if (self.protocol_id is None) != (self.protocol_sha256 is None):
            raise ValueError("protocol_sha256 must be present exactly when protocol_id is present")
        if not _strict_int(self.retention_days) or not 1 <= self.retention_days <= 365:
            raise ValueError("retention_days must be an integer in [1, 365]")
        if self.authorized is not True:
            raise ValueError("consent requires explicit authorization")
        if not _is_aware(self.granted_at) or not _is_aware(self.expires_at):
            raise ValueError("consent timestamps must include a timezone")
        if self.expires_at <= self.granted_at:
            raise ValueError("consent must expire after it is granted")
        if self.replaces_consent_id is not None and not isinstance(self.replaces_consent_id, UUID):
            raise ValueError("replaces_consent_id must be a UUID")
        retention_end = self.granted_at + timedelta(days=self.retention_days)
        if self.replaces_consent_id is None:
            if abs(self.expires_at - retention_end) > _EXPIRY_TOLERANCE:
                raise ValueError("a first grant expires exactly retention_days after it is granted")
        elif self.expires_at > retention_end + _EXPIRY_TOLERANCE:
            raise ValueError("a consent update never extends the expiry beyond the retention period")

    def require_active(
        self, purpose: CoAdaptationPurpose, *, now: datetime, revoked_at: datetime | None = None
    ) -> None:
        """Raise ``ConsentError`` unless this consent is in force at ``now`` and covers ``purpose``."""
        if revoked_at is not None and revoked_at <= now:
            raise ConsentError("consent_revoked")
        if now >= self.expires_at:
            raise ConsentError("consent_expired")
        if CoAdaptationPurpose(purpose) not in self.purposes:
            raise ConsentError("purpose_missing")

    def to_dict(self) -> dict[str, Any]:
        """The same JSON as ``CoAdaptationConsentV1.model_dump(mode="json")`` (purposes in enum order)."""
        return {
            "schema_version": self.schema_version,
            "consent_id": str(self.consent_id),
            "participant_key": self.participant_key,
            "purposes": [purpose.value for purpose in _PURPOSE_ORDER if purpose in self.purposes],
            "consent_text_version": self.consent_text_version,
            "consent_text_sha256": self.consent_text_sha256,
            "protocol_id": self.protocol_id,
            "protocol_sha256": self.protocol_sha256,
            "granted_at": format_datetime(self.granted_at),
            "expires_at": format_datetime(self.expires_at),
            "retention_days": self.retention_days,
            "authorized": self.authorized,
            "replaces_consent_id": None if self.replaces_consent_id is None else str(self.replaces_consent_id),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        _check_keys(
            d,
            required=(
                "participant_key",
                "purposes",
                "consent_text_version",
                "consent_text_sha256",
                "granted_at",
                "expires_at",
                "retention_days",
                "authorized",
            ),
            optional=("schema_version", "consent_id", "protocol_id", "protocol_sha256", "replaces_consent_id"),
            record="ConsentRecord",
        )
        purposes = d["purposes"]
        if not isinstance(purposes, list | tuple) or not all(isinstance(item, str) for item in purposes):
            raise ValueError("purposes must be a list of strings")
        try:
            parsed_purposes = frozenset(CoAdaptationPurpose(item) for item in purposes)
        except ValueError:
            raise ValueError("purposes contains an unknown purpose") from None
        for name in ("participant_key", "consent_text_version", "consent_text_sha256"):
            if not isinstance(d[name], str):
                raise ValueError(f"{name} must be a string")
        for name in ("protocol_id", "protocol_sha256"):
            if d.get(name) is not None and not isinstance(d[name], str):
                raise ValueError(f"{name} must be a string or null")
        if not isinstance(d["authorized"], bool):
            raise ValueError("authorized must be a boolean")
        for name in ("schema_version", "retention_days"):
            if name in d and not _strict_int(d[name]):
                raise ValueError(f"{name} must be an integer")
        kwargs: dict[str, Any] = {
            "participant_key": d["participant_key"],
            "purposes": parsed_purposes,
            "consent_text_version": d["consent_text_version"],
            "consent_text_sha256": d["consent_text_sha256"],
            "protocol_id": d.get("protocol_id"),
            "protocol_sha256": d.get("protocol_sha256"),
            "granted_at": parse_datetime(d["granted_at"], "granted_at"),
            "expires_at": parse_datetime(d["expires_at"], "expires_at"),
            "retention_days": d["retention_days"],
            "authorized": d["authorized"],
        }
        if "schema_version" in d:
            kwargs["schema_version"] = d["schema_version"]
        if "consent_id" in d:
            kwargs["consent_id"] = _parse_uuid(d["consent_id"], "consent_id")
        if d.get("replaces_consent_id") is not None:
            kwargs["replaces_consent_id"] = _parse_uuid(d["replaces_consent_id"], "replaces_consent_id")
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class ConsentPeriod:
    """One entry of a participant's consent history (export bundle and audit A6)."""

    record: ConsentRecord
    replaced_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.record, ConsentRecord):
            raise ValueError("record must be a ConsentRecord")
        if self.replaced_at is not None and not _is_aware(self.replaced_at):
            raise ValueError("replaced_at must include a timezone")

    def ends_at(self, *, revoked_at: datetime | None = None) -> datetime:
        """The earliest of ``replaced_at``, ``record.expires_at`` and ``revoked_at``."""
        ends = [self.record.expires_at]
        if self.replaced_at is not None:
            ends.append(self.replaced_at)
        if revoked_at is not None:
            ends.append(revoked_at)
        return min(ends)

    def in_force(self, at: datetime, *, revoked_at: datetime | None = None) -> bool:
        return self.record.granted_at <= at < self.ends_at(revoked_at=revoked_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record": self.record.to_dict(),
            "replaced_at": None if self.replaced_at is None else format_datetime(self.replaced_at),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        _check_keys(d, required=("record", "replaced_at"), optional=(), record="ConsentPeriod")
        if not isinstance(d["record"], Mapping):
            raise ValueError("record must be an object")
        replaced = d["replaced_at"]
        return cls(
            record=ConsentRecord.from_dict(d["record"]),
            replaced_at=None if replaced is None else parse_datetime(replaced, "replaced_at"),
        )


def consent_in_force(
    periods: Iterable[ConsentPeriod], at: datetime, *, revoked_at: datetime | None = None
) -> ConsentRecord | None:
    """The record in force at ``at``: granted at or before it and not yet replaced, expired or revoked.

    ``revoked_at`` is the participant's revocation time, which ends every period. When periods overlap (they should
    not), the most recently granted record wins.
    """
    in_force = [period for period in periods if period.in_force(at, revoked_at=revoked_at)]
    if not in_force:
        return None
    return max(in_force, key=lambda period: period.record.granted_at).record


def new_participant_key(existing: Collection[str]) -> str:
    """A fresh daemon-generated participant key (``p-`` and 8 hex characters), regenerated on collision."""
    taken = set(existing)
    while True:
        key = f"p-{secrets.token_hex(4)}"
        if key not in taken:
            return key
