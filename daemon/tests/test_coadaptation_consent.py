"""Consent record of the co-adaptation study (spec 10.3, tests of 22.2)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from pilot.coadaptation.consent import (
    CoAdaptationConsentError,
    CoAdaptationConsentV1,
    Identifier,
    ParticipantKey,
    consent_summary,
)
from pilot.coadaptation.core.consent import ConsentRecord

GRANTED = datetime(2026, 10, 15, 10, 0, tzinfo=UTC)


def _consent(**overrides):
    data = {
        "participant_key": "p-3fa29c1e",
        "purposes": ["selection_study", "gaze_prior"],
        "consent_text_version": "coadapt-consent-v1",
        "consent_text_sha256": "a" * 64,
        "protocol_id": "coadapt-study-v1",
        "protocol_sha256": "b" * 64,
        "granted_at": GRANTED,
        "expires_at": GRANTED + timedelta(days=90),
        "retention_days": 90,
        "authorized": True,
    }
    data.update(overrides)
    return CoAdaptationConsentV1.model_validate(data)


def test_valid_first_grant_round_trips_through_json_and_core():
    record = _consent()
    assert CoAdaptationConsentV1.model_validate_json(record.model_dump_json()) == record
    core = record.to_core()
    assert isinstance(core, ConsentRecord)
    assert core.to_dict() == record.model_dump(mode="json")
    assert record.model_dump(mode="json")["purposes"] == ["selection_study", "gaze_prior"]


@pytest.mark.parametrize("value", [False, "true", 1, None])
def test_authorized_must_be_literally_true(value):
    with pytest.raises(ValidationError):
        _consent(authorized=value)


def test_timestamps_must_carry_an_offset():
    naive = datetime(2026, 10, 15, 10, 0)
    with pytest.raises(ValidationError):
        _consent(granted_at=naive, expires_at=naive + timedelta(days=90))


def test_first_grant_expires_exactly_after_retention():
    assert _consent(expires_at=GRANTED + timedelta(days=90, milliseconds=500)).retention_days == 90
    with pytest.raises(ValidationError):
        _consent(expires_at=GRANTED + timedelta(days=90, seconds=2))
    with pytest.raises(ValidationError):
        _consent(expires_at=GRANTED + timedelta(days=30))


def test_update_never_extends_the_expiry():
    replaces = uuid4()
    shorter = _consent(replaces_consent_id=replaces, expires_at=GRANTED + timedelta(days=10))
    assert shorter.replaces_consent_id == replaces
    with pytest.raises(ValidationError):
        _consent(replaces_consent_id=replaces, expires_at=GRANTED + timedelta(days=91))
    with pytest.raises(ValidationError):
        _consent(replaces_consent_id=replaces, expires_at=GRANTED - timedelta(seconds=1))


@pytest.mark.parametrize("purpose", ["approval_observation", "screen_capture", "SELECTION_STUDY"])
def test_unknown_purposes_are_rejected(purpose):
    with pytest.raises(ValidationError):
        _consent(purposes=["selection_study", purpose])


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        _consent(label="Lab A")


def test_selection_study_is_required():
    with pytest.raises(ValidationError):
        _consent(purposes=["gaze_prior"])
    with pytest.raises(ValidationError):
        _consent(purposes=[])


@pytest.mark.parametrize("key", ["alice", "p-3FA29C1E", "p-3fa29c1", "p-3fa29c1e0", "p-3fa2:c1e", "p:3fa29c1e", ""])
def test_participant_key_must_be_daemon_shaped(key):
    with pytest.raises(ValidationError):
        _consent(participant_key=key)


def test_protocol_hash_present_exactly_with_protocol_id():
    with pytest.raises(ValidationError):
        _consent(protocol_sha256=None)
    with pytest.raises(ValidationError):
        _consent(protocol_id=None)
    assert _consent(protocol_id=None, protocol_sha256=None).protocol_id is None


def test_require_active_codes():
    record = _consent()
    record.require_active("selection_study", now=GRANTED + timedelta(days=1))
    with pytest.raises(CoAdaptationConsentError) as expired:
        record.require_active("selection_study", now=GRANTED + timedelta(days=90))
    assert expired.value.code == "consent_expired"
    with pytest.raises(CoAdaptationConsentError) as revoked:
        record.require_active("selection_study", now=GRANTED + timedelta(days=1), revoked_at=GRANTED)
    assert revoked.value.code == "consent_revoked"
    with pytest.raises(CoAdaptationConsentError) as missing:
        record.require_active("research_export", now=GRANTED + timedelta(days=1))
    assert missing.value.code == "purpose_missing"


def test_consent_error_codes_are_closed():
    with pytest.raises(ValueError):
        CoAdaptationConsentError("something_else")


def test_consent_summary_is_free_of_behavioural_data():
    record = _consent()
    summary = consent_summary(record, None, GRANTED + timedelta(days=1))
    assert summary["active"] is True
    assert summary["purposes"] == ["selection_study", "gaze_prior"]
    assert summary["consent_id"] == str(record.consent_id)
    assert summary["retention_days"] == 90
    assert consent_summary(record, None, GRANTED + timedelta(days=91))["active"] is False
    assert consent_summary(record, GRANTED, GRANTED + timedelta(days=1))["active"] is False


def test_local_patterns_match_the_neural_identifier_without_importing_it():
    from pydantic import TypeAdapter

    TypeAdapter(Identifier).validate_python("coadapt-study-v1")
    with pytest.raises(ValidationError):
        TypeAdapter(Identifier).validate_python("bad id")
    TypeAdapter(ParticipantKey).validate_python("p-0123abcd")
    with pytest.raises(ValidationError):
        TypeAdapter(ParticipantKey).validate_python("p-0123ABCD")
