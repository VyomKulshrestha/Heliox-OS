"""Core consent twin: validation, history, serialization and participant keys (spec 10.2-10.3, 22.1)."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest

from pilot.coadaptation.core import consent as consent_module
from pilot.coadaptation.core.consent import (
    PARTICIPANT_KEY_PATTERN,
    CoAdaptationPurpose,
    ConsentError,
    ConsentPeriod,
    ConsentRecord,
    consent_in_force,
    new_participant_key,
)

GRANTED = datetime(2026, 10, 1, 9, 30, 15, 123456, tzinfo=UTC)
HASH = "a" * 64


def _record(**overrides) -> ConsentRecord:
    values = {
        "participant_key": "p-3fa29c1e",
        "purposes": frozenset({CoAdaptationPurpose.SELECTION_STUDY}),
        "consent_text_version": "coadapt-consent-v1",
        "consent_text_sha256": HASH,
        "granted_at": GRANTED,
        "expires_at": GRANTED + timedelta(days=90),
        "retention_days": 90,
        "authorized": True,
    }
    values.update(overrides)
    return ConsentRecord(**values)


def test_purposes_are_exactly_the_six_of_the_spec() -> None:
    assert [p.value for p in CoAdaptationPurpose] == [
        "selection_study",
        "adaptive_assistance",
        "cross_session_learning",
        "gaze_prior",
        "hand_pointer",
        "research_export",
    ]
    assert not any("approval" in p.value for p in CoAdaptationPurpose)


def test_a_valid_first_grant_and_its_defaults() -> None:
    record = _record()
    assert record.schema_version == 1
    assert record.replaces_consent_id is None
    assert record.protocol_id is None and record.protocol_sha256 is None


def test_first_grant_expires_exactly_retention_days_after_grant() -> None:
    _record(expires_at=GRANTED + timedelta(days=90, seconds=1))  # within the 1 s tolerance
    with pytest.raises(ValueError):
        _record(expires_at=GRANTED + timedelta(days=89))
    with pytest.raises(ValueError):
        _record(expires_at=GRANTED + timedelta(days=90, seconds=2))


def test_an_update_never_extends_the_expiry() -> None:
    old_id = uuid4()
    # an update may expire earlier than granted_at + retention (the original expiry is kept)
    _record(replaces_consent_id=old_id, expires_at=GRANTED + timedelta(days=30))
    _record(replaces_consent_id=old_id, expires_at=GRANTED + timedelta(days=90))
    with pytest.raises(ValueError):
        _record(replaces_consent_id=old_id, expires_at=GRANTED + timedelta(days=91))


@pytest.mark.parametrize(
    "overrides",
    [
        {"authorized": False},
        {"purposes": frozenset()},
        {"purposes": frozenset({CoAdaptationPurpose.GAZE_PRIOR})},
        {"participant_key": "p-3FA29C1E"},
        {"participant_key": "alice"},
        {"participant_key": "p-3fa29c1e:x"},
        {"consent_text_version": "v1"},
        {"consent_text_sha256": "A" * 64},
        {"consent_text_sha256": "a" * 63},
        {"protocol_id": "coadapt-study-v1"},  # protocol_sha256 missing
        {"protocol_sha256": HASH},  # protocol_id missing
        {"protocol_id": "bad id with spaces", "protocol_sha256": HASH},
        {"retention_days": 0},
        {"retention_days": 366},
        {"retention_days": True},
        {"schema_version": 2},
        {"granted_at": GRANTED.replace(tzinfo=None)},
        {"expires_at": GRANTED},
    ],
)
def test_invalid_records_raise_value_error(overrides) -> None:
    with pytest.raises(ValueError):
        _record(**overrides)


def test_protocol_pair_is_accepted_together() -> None:
    record = _record(protocol_id="coadapt-study-v1", protocol_sha256=HASH)
    assert record.protocol_id == "coadapt-study-v1"


def test_require_active_codes() -> None:
    record = _record(purposes=frozenset({CoAdaptationPurpose.SELECTION_STUDY, CoAdaptationPurpose.GAZE_PRIOR}))
    now = GRANTED + timedelta(days=1)
    record.require_active(CoAdaptationPurpose.GAZE_PRIOR, now=now)
    with pytest.raises(ConsentError) as missing:
        record.require_active(CoAdaptationPurpose.HAND_POINTER, now=now)
    assert missing.value.code == "purpose_missing"
    with pytest.raises(ConsentError) as expired:
        record.require_active(CoAdaptationPurpose.SELECTION_STUDY, now=record.expires_at)
    assert expired.value.code == "consent_expired"
    with pytest.raises(ConsentError) as revoked:
        record.require_active(CoAdaptationPurpose.SELECTION_STUDY, now=now, revoked_at=now - timedelta(hours=1))
    assert revoked.value.code == "consent_revoked"


def test_to_dict_matches_the_pydantic_json_shape() -> None:
    record = _record(
        purposes=frozenset({CoAdaptationPurpose.RESEARCH_EXPORT, CoAdaptationPurpose.SELECTION_STUDY}),
        protocol_id="coadapt-study-v1",
        protocol_sha256=HASH,
    )
    d = record.to_dict()
    assert list(d) == [
        "schema_version",
        "consent_id",
        "participant_key",
        "purposes",
        "consent_text_version",
        "consent_text_sha256",
        "protocol_id",
        "protocol_sha256",
        "granted_at",
        "expires_at",
        "retention_days",
        "authorized",
        "replaces_consent_id",
    ]
    assert d["purposes"] == ["selection_study", "research_export"]
    assert d["granted_at"] == "2026-10-01T09:30:15.123456Z"  # pydantic's JSON mode writes UTC as Z
    assert d["consent_id"] == str(record.consent_id)
    assert d["replaces_consent_id"] is None


def test_round_trips_through_json() -> None:
    tz = timezone(timedelta(hours=5, minutes=30))
    records = [
        _record(),
        _record(
            granted_at=GRANTED.astimezone(tz),
            expires_at=(GRANTED + timedelta(days=30)).astimezone(tz),
            retention_days=30,
            replaces_consent_id=uuid4(),
            purposes=frozenset(CoAdaptationPurpose),
            protocol_id="coadapt-study-v1",
            protocol_sha256=HASH,
        ),
    ]
    for record in records:
        again = ConsentRecord.from_dict(json.loads(json.dumps(record.to_dict())))
        assert again == record
        period = ConsentPeriod(record=record, replaced_at=record.granted_at + timedelta(days=3))
        assert ConsentPeriod.from_dict(json.loads(json.dumps(period.to_dict()))) == period
        open_period = ConsentPeriod(record=record, replaced_at=None)
        assert ConsentPeriod.from_dict(json.loads(json.dumps(open_period.to_dict()))) == open_period


def test_from_dict_rejects_bad_keys_and_types_without_echoing_values() -> None:
    good = _record().to_dict()
    secret = "do-not-echo-1234"
    cases = [
        {**good, "extra": secret},
        {key: value for key, value in good.items() if key != "authorized"},
        {**good, "retention_days": "90"},
        {**good, "purposes": [secret]},
        {**good, "granted_at": secret},
        {**good, "participant_key": secret},
        {**good, "authorized": "true"},
    ]
    for case in cases:
        with pytest.raises(ValueError) as error:
            ConsentRecord.from_dict(case)
        assert secret not in str(error.value)
    with pytest.raises(ValueError):
        ConsentPeriod.from_dict({"record": good, "replaced_at": None, "other": 1})


def test_consent_in_force_follows_updates_expiry_and_revocation() -> None:
    first = _record()
    update_at = GRANTED + timedelta(days=10)
    second = _record(
        granted_at=update_at,
        expires_at=first.expires_at,  # the retention clock never restarts
        replaces_consent_id=first.consent_id,
        purposes=frozenset({CoAdaptationPurpose.SELECTION_STUDY, CoAdaptationPurpose.ADAPTIVE_ASSISTANCE}),
    )
    periods = [ConsentPeriod(first, replaced_at=update_at), ConsentPeriod(second, replaced_at=None)]
    assert consent_in_force(periods, GRANTED - timedelta(seconds=1)) is None
    assert consent_in_force(periods, GRANTED) == first
    assert consent_in_force(periods, update_at - timedelta(microseconds=1)) == first
    assert consent_in_force(periods, update_at) == second
    assert consent_in_force(periods, first.expires_at - timedelta(seconds=1)) == second
    assert consent_in_force(periods, first.expires_at) is None
    revoked = GRANTED + timedelta(days=20)
    assert consent_in_force(periods, revoked - timedelta(seconds=1), revoked_at=revoked) == second
    assert consent_in_force(periods, revoked, revoked_at=revoked) is None


def test_new_participant_keys_match_the_pattern_and_avoid_collisions(monkeypatch) -> None:
    pattern = re.compile(PARTICIPANT_KEY_PATTERN)
    keys = {new_participant_key(()) for _ in range(50)}
    assert all(pattern.fullmatch(key) for key in keys)
    values = iter(["3fa29c1e", "3fa29c1e", "00000001"])
    monkeypatch.setattr(consent_module.secrets, "token_hex", lambda n: next(values))
    assert new_participant_key({"p-3fa29c1e"}) == "p-00000001"
