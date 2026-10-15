"""Consent record for the co-adaptation study (spec section 10.3).

The pydantic model mirrors ``NeuralRecordingConsentV1`` in ``pilot/neural/recording.py`` without importing it:
importing anything from ``pilot.neural`` loads the authority layer (spec section 3), so ``Identifier`` and
``ParticipantKey`` are defined here with the same patterns.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated, Any, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_serializer, model_validator

from pilot.coadaptation.core.consent import CoAdaptationPurpose, ConsentRecord

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9._:-]{1,128}$")]
ParticipantKey = Annotated[str, StringConstraints(pattern=r"^p-[0-9a-f]{8}$")]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ConsentTextVersion = Annotated[str, StringConstraints(pattern=r"^coadapt-consent-v[0-9]+$")]

CONSENT_ERROR_CODES = frozenset({"consent_expired", "consent_revoked", "purpose_missing", "no_consent"})

_EXPIRY_TOLERANCE = timedelta(seconds=1)

_CONSENT_MESSAGES = {
    "consent_expired": "The study consent has expired.",
    "consent_revoked": "The study consent was withdrawn.",
    "purpose_missing": "The study consent does not cover this use.",
    "no_consent": "There is no active study consent for this participant.",
}


class CoAdaptationConsentError(Exception):
    """A write or use that the participant's consent does not cover. ``code`` names the reason."""

    def __init__(self, code: str, message: str | None = None) -> None:
        if code not in CONSENT_ERROR_CODES:
            raise ValueError("unknown consent error code")
        self.code = code
        super().__init__(message or _CONSENT_MESSAGES[code])


class CoAdaptationConsentV1(BaseModel):
    """Typed, purpose-bound, expiring and revocable consent for one participant."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: int = Field(default=1, strict=True, ge=1, le=1)
    consent_id: UUID = Field(default_factory=uuid4)
    participant_key: ParticipantKey
    purposes: frozenset[CoAdaptationPurpose]
    consent_text_version: ConsentTextVersion
    consent_text_sha256: Sha256Hex
    protocol_id: Identifier | None = None
    protocol_sha256: Sha256Hex | None = None
    granted_at: datetime
    expires_at: datetime
    retention_days: int = Field(strict=True, ge=1, le=365)
    authorized: bool = Field(strict=True)
    replaces_consent_id: UUID | None = None

    @model_validator(mode="after")
    def _validate_consent(self) -> Self:
        if self.authorized is not True:
            raise ValueError("co-adaptation consent requires explicit authorization")
        if self.granted_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("consent timestamps must include a UTC offset")
        if self.expires_at <= self.granted_at:
            raise ValueError("consent must expire after it is granted")
        limit = self.granted_at + timedelta(days=self.retention_days)
        if self.replaces_consent_id is None:
            if abs(self.expires_at - limit) > _EXPIRY_TOLERANCE:
                raise ValueError("a first consent expires exactly retention_days after it is granted")
        elif self.expires_at > limit + _EXPIRY_TOLERANCE:
            raise ValueError("a consent update never extends the expiry beyond its retention period")
        if not self.purposes:
            raise ValueError("consent needs at least one purpose")
        if CoAdaptationPurpose.SELECTION_STUDY not in self.purposes:
            raise ValueError("the selection_study purpose is required")
        if (self.protocol_id is None) != (self.protocol_sha256 is None):
            raise ValueError("protocol_sha256 must be given exactly when protocol_id is given")
        return self

    @field_serializer("purposes")
    def _serialize_purposes(self, purposes: frozenset[CoAdaptationPurpose]) -> list[str]:
        # Enum order, so the stored JSON and every summary are deterministic.
        return [purpose.value for purpose in CoAdaptationPurpose if purpose in purposes]

    def require_active(
        self,
        purpose: CoAdaptationPurpose | str,
        *,
        now: datetime,
        revoked_at: datetime | None = None,
    ) -> None:
        """Raise ``CoAdaptationConsentError`` unless this consent is in force at ``now`` and covers ``purpose``."""
        if revoked_at is not None:
            raise CoAdaptationConsentError("consent_revoked")
        if now >= self.expires_at:
            raise CoAdaptationConsentError("consent_expired")
        try:
            wanted = CoAdaptationPurpose(purpose)
        except ValueError:
            raise CoAdaptationConsentError("purpose_missing")
        if wanted not in self.purposes:
            raise CoAdaptationConsentError("purpose_missing")

    def covers(self, purpose: CoAdaptationPurpose | str) -> bool:
        try:
            return CoAdaptationPurpose(purpose) in self.purposes
        except ValueError:
            return False

    def to_core(self) -> ConsentRecord:
        """The core twin, built from the same JSON that ``ConsentRecord.to_dict()`` writes."""
        return ConsentRecord.from_dict(self.model_dump(mode="json"))


def consent_summary(record: CoAdaptationConsentV1, revoked_at: datetime | None, now: datetime) -> dict[str, Any]:
    """The consent object of the status and grant replies (spec 14.2). Holds no behavioural data."""
    return {
        "consent_id": str(record.consent_id),
        "participant_key": record.participant_key,
        "purposes": [purpose.value for purpose in CoAdaptationPurpose if purpose in record.purposes],
        "granted_at": record.granted_at.isoformat(),
        "expires_at": record.expires_at.isoformat(),
        "retention_days": record.retention_days,
        "consent_text_version": record.consent_text_version,
        "consent_text_sha256": record.consent_text_sha256,
        "protocol_id": record.protocol_id,
        "protocol_sha256": record.protocol_sha256,
        "replaces_consent_id": None if record.replaces_consent_id is None else str(record.replaces_consent_id),
        "revoked_at": None if revoked_at is None else revoked_at.isoformat(),
        "active": revoked_at is None and now < record.expires_at,
    }
