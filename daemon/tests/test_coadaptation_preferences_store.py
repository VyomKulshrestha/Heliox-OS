"""The co-adaptation preferences file (spec 10.5, tests of 22.2)."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta

import pytest

from pilot.coadaptation.core.preferences import PreferenceConflict, PreferenceSet
from pilot.coadaptation.preferences_store import PreferenceFileError, PreferenceStore

START = datetime(2026, 10, 15, 10, 0, tzinfo=UTC)
PK = "p-3fa29c1e"
OTHER = "p-0b5d7e11"


class _Clock:
    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        return self.now


def _rule(pref_id: str, kind: str, value, *, participant_key=None, context="study", modality=None, created=START):
    return {
        "pref_id": pref_id,
        "kind": kind,
        "scope": {"context": context, "modality": modality, "participant_key": participant_key},
        "value": value,
        "origin": "explicit_user",
        "created_at": created.isoformat(),
        "retracted_at": None,
    }


@pytest.fixture
def store(tmp_path):
    clock = _Clock()
    return PreferenceStore(tmp_path / "co_adaptation_preferences.json", clock=clock), clock


def test_load_without_file_is_empty(store):
    prefs, _clock = store
    assert isinstance(prefs.load(), PreferenceSet)
    assert prefs.entries() == ()
    assert not prefs.path.exists()


def test_add_writes_the_documented_format(store):
    prefs, _clock = store
    pref, notes = prefs.add(_rule("r1", "forbid_variant", "ask_clarification:3"))
    assert notes == ()
    assert pref.origin == "explicit_user"
    document = json.loads(prefs.path.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["preferences"] == [_rule("r1", "forbid_variant", "ask_clarification:3")]
    assert [p.pref_id for p in prefs.entries()] == ["r1"]


def test_writes_are_atomic(store, monkeypatch):
    prefs, _clock = store
    prefs.add(_rule("r1", "forbid_variant", "ask_clarification:3"))
    before = prefs.path.read_bytes()

    def failing_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError):
        prefs.add(_rule("r2", "forbid_variant", "enlarge_target"))
    assert prefs.path.read_bytes() == before
    assert [p.name for p in prefs.path.parent.iterdir()] == [prefs.path.name]


def test_origin_is_always_explicit_user(store):
    prefs, _clock = store
    learned = _rule("r1", "forbid_variant", "act")
    learned["origin"] = "learned"
    with pytest.raises(ValueError):
        prefs.add(learned)
    assert not prefs.path.exists()


def test_conflicting_rule_is_rejected_and_nothing_is_written(store):
    prefs, _clock = store
    prefs.add(_rule("min", "min_caution", "confirm_selection:second_click", participant_key=PK))
    before = prefs.path.read_bytes()
    with pytest.raises(PreferenceConflict) as conflict:
        prefs.add(_rule("frz", "freeze_adaptation", {"regime": "act"}, participant_key=PK, modality="mouse"))
    assert "min" in conflict.value.conflicting_pref_ids
    assert prefs.path.read_bytes() == before


def test_cap_with_confirm_level_minimum_is_accepted_with_a_note(store):
    prefs, _clock = store
    prefs.add(_rule("min", "min_caution", "confirm_selection:second_click"))
    _pref, notes = prefs.add(_rule("cap", "max_interruptions_per_block", 2))
    assert notes


def test_retract_keeps_the_rule_for_thirty_days(store):
    prefs, clock = store
    prefs.add(_rule("r1", "forbid_variant", "ask_clarification:3"))
    clock.now = START + timedelta(days=1)
    assert prefs.retract("r1") is True
    assert prefs.retract("r1") is False
    assert prefs.retract("missing") is False
    (entry,) = prefs.entries()
    assert entry.retracted_at == START + timedelta(days=1)
    assert prefs.load().active(context="study", modality="mouse", participant_key=PK) == ()

    clock.now = START + timedelta(days=29)
    prefs.add(_rule("r2", "forbid_variant", "enlarge_target", created=clock.now))
    assert [p.pref_id for p in prefs.entries()] == ["r1", "r2"]

    clock.now = START + timedelta(days=31, seconds=1)
    prefs.add(_rule("r3", "forbid_variant", "offer_modality", created=clock.now))
    assert [p.pref_id for p in prefs.entries()] == ["r2", "r3"]


def test_retract_at_the_given_instant_never_reads_the_clock_again(store):
    prefs, clock = store
    prefs.add(_rule("r1", "forbid_variant", "ask_clarification:3"))
    issued = START + timedelta(minutes=5)
    clock.now = issued + timedelta(microseconds=900)  # a later reading of a real clock
    assert prefs.retract("r1", at=issued) is True
    (entry,) = prefs.entries()
    assert entry.retracted_at == issued
    # At the instant the reply's fallbacks were issued, the rule is already out of force.
    assert prefs.load().as_of(issued).active(context="study", modality="mouse", participant_key=PK) == ()
    with pytest.raises(ValueError):
        prefs.retract("r1", at=datetime(2026, 10, 15, 10, 5))


def test_retract_never_precedes_the_rule(store):
    prefs, _clock = store
    prefs.add(_rule("r1", "forbid_variant", "ask_clarification:3"))
    assert prefs.retract("r1", at=START - timedelta(seconds=1)) is True
    (entry,) = prefs.entries()
    assert entry.retracted_at == START


def test_delete_for_participant_keeps_device_wide_rules(store):
    prefs, _clock = store
    prefs.add(_rule("device", "forbid_variant", "ask_clarification:3"))
    prefs.add(_rule("mine", "forbid_variant", "enlarge_target", participant_key=PK))
    prefs.add(_rule("theirs", "max_interruptions_per_block", 4, participant_key=OTHER))
    assert prefs.delete_for_participant(PK) == 1
    assert [p.pref_id for p in prefs.entries()] == ["device", "theirs"]
    assert prefs.delete_for_participant(PK) == 0
    assert PK not in prefs.path.read_text(encoding="utf-8")


def test_file_holds_participant_keys_but_never_labels(store):
    prefs, _clock = store
    prefs.add(_rule("mine", "forbid_variant", "enlarge_target", participant_key=PK))
    text = prefs.path.read_text(encoding="utf-8")
    assert PK in text
    assert "label" not in text


def test_unreadable_file_is_never_treated_as_empty(store):
    prefs, _clock = store
    prefs.path.write_text("{not json", encoding="utf-8")
    with pytest.raises(PreferenceFileError):
        prefs.load()
    prefs.path.write_text(json.dumps({"schema_version": 2, "preferences": []}), encoding="utf-8")
    with pytest.raises(PreferenceFileError):
        prefs.load()


def test_works_without_any_consent_or_store(store):
    prefs, _clock = store
    prefs.add(_rule("r1", "pin_act", True, participant_key=PK))
    assert prefs.load().pin_act(modality="mouse", participant_key=PK) is True
