"""The online learner loads from a saved state and replays only what came after it: with 100,000 events in the
ledger a start reads a handful of pages, and afterwards it decides exactly as a full replay does. A saved state that
does not match its vault key, its versions, or the allow-list is refused and the whole ledger is replayed instead."""

from __future__ import annotations

import io
import json
import pickle
import random
from datetime import UTC, datetime, timedelta

import pytest

from pilot.intelligence import learner_snapshot
from pilot.intelligence.experience import ExperienceEvent, ExperienceEventType
from pilot.intelligence.online_learning import MODEL_VERSION, VerifiedOnlineLearner

KEY = b"k" * 64
EVENTS = 100_000
ACTIONS = ("file_write", "file_read", "shell_command", "browser_navigate")
PATTERNS = ("break_reminder", "save_reminder", "focus_mode")


def _event(sequence, event_type, *, plan_id="", action_id="", source="interactive", payload=None, at=None):
    return ExperienceEvent(
        event_id=f"e{sequence}",
        sequence=sequence,
        event_type=event_type,
        occurred_at=at,
        schema_version=1,
        session_id="s",
        task_id="",
        user_id="",
        plan_id=plan_id,
        action_id=action_id,
        parent_event_id="",
        idempotency_key=f"k{sequence}",
        source=source,
        payload=payload or {},
    )


def _ledger_events(count: int) -> list[ExperienceEvent]:
    """A deterministic mix: verified one-step plans, shown and answered suggestions, and screen observations."""
    rng = random.Random(7)
    start = datetime(2026, 1, 5, 9, tzinfo=UTC)
    events: list[ExperienceEvent] = []
    plan = 0
    while len(events) < count:
        at = (start + timedelta(minutes=len(events))).isoformat()
        kind = rng.random()
        sequence = len(events) + 1
        if kind < 0.6:
            plan += 1
            action = ACTIONS[plan % len(ACTIONS)]
            ok = rng.random() < (0.85 if action != "shell_command" else 0.25)
            plan_id, action_id = f"p{plan}", f"a{plan}"
            events.append(
                _event(
                    sequence,
                    ExperienceEventType.ACTION_STARTED,
                    plan_id=plan_id,
                    action_id=action_id,
                    payload={"index": 0, "action_type": action},
                    at=at,
                )
            )
            events.append(
                _event(
                    sequence + 1,
                    ExperienceEventType.ACTION_COMPLETED,
                    plan_id=plan_id,
                    action_id=action_id,
                    payload={"index": 0, "success": ok, "executed": True, "callback_observed": True},
                    at=at,
                )
            )
            events.append(
                _event(
                    sequence + 2,
                    ExperienceEventType.OUTCOME_VERIFIED,
                    plan_id=plan_id,
                    source="verifier",
                    payload={
                        "result_count": 1,
                        "verification": {
                            "passed": ok,
                            "failed_actions": [] if ok else [0],
                            "independently_observed_actions": [0],
                        },
                    },
                    at=at,
                )
            )
        elif kind < 0.8:
            suggestion = f"sg{sequence}"
            pattern = PATTERNS[sequence % len(PATTERNS)]
            events.append(
                _event(
                    sequence,
                    ExperienceEventType.SUGGESTION_SHOWN,
                    source="proactive",
                    payload={"suggestion_id": suggestion, "pattern_id": pattern, "context_app": "code"},
                    at=at,
                )
            )
            decision = "accepted" if rng.random() < (0.8 if pattern == "focus_mode" else 0.2) else "dismissed"
            events.append(
                _event(
                    sequence + 1,
                    ExperienceEventType.SUGGESTION_FEEDBACK,
                    source="proactive",
                    payload={"suggestion_id": suggestion, "decision": decision},
                    at=at,
                )
            )
        else:
            events.append(
                _event(
                    sequence, ExperienceEventType.OBSERVATION, source="screen", payload={"active_app": "code"}, at=at
                )
            )
    return events[:count]


class _Ledger:
    """The ledger's read API over a list, counting how many pages a load reads."""

    def __init__(self, events):
        self.events = events
        self.reads = 0

    async def list_events(self, *, after_sequence=0, limit=200, **_filters):
        self.reads += 1
        start = after_sequence  # sequences are 1..n with no gaps
        return self.events[start : start + min(limit, 1000)]


@pytest.fixture(scope="module")
def events():
    return _ledger_events(EVENTS)


def _decisions(learner: VerifiedOnlineLearner) -> list:
    at = "2026-03-02T10:00:00+00:00"
    found = []
    for action in ACTIONS:
        for source in ("interactive", "mcp"):
            score = learner.score_transition(action_type=action, source=source, occurred_at=at)
            found.append(("transition", action, source, round(score.probability, 12), score.labels, score.state))
    for pattern in PATTERNS:
        score = learner.score_suggestion(pattern_id=pattern, app_name="code", priority="low", occurred_at=at)
        found.append(("suggestion", pattern, round(score.probability, 12), score.labels, score.state))
    status = learner.status()
    found.append({k: status[k] for k in ("event_cursor", "suggestions", "transitions", "workflow_patterns")})
    return found


async def test_a_start_reads_only_what_came_after_the_saved_state_and_decides_as_a_full_replay(tmp_path, events):
    full = VerifiedOnlineLearner(tmp_path / "full.json")
    full_ledger = _Ledger(events)
    await full.initialize(full_ledger)
    assert full_ledger.reads == EVENTS // 1000 + 1
    expected = _decisions(full)
    assert any(row[-1] == "promoted" for row in expected[:-1])  # the comparison is about real learned decisions

    # Yesterday: the learner had seen 95,000 events and saved its state.
    earlier = VerifiedOnlineLearner(tmp_path / "state.json")
    await earlier.initialize(_Ledger(events[:95_000]))
    snapshot = tmp_path / "online_learning_snapshot.bin"
    await earlier.save_snapshot(snapshot, KEY)

    # Today's start: the saved state, then only the 5,000 events after it.
    today = VerifiedOnlineLearner(tmp_path / "state.json")
    today.begin_loading()
    ledger = _Ledger(events)
    loaded = await today.initialize(ledger, snapshot=snapshot, key=KEY)
    assert loaded["from_snapshot"] is True and loaded["refused"] == ""
    assert loaded["replayed"] == 5_000
    assert ledger.reads == 6  # five full pages and the empty one that ends the replay
    assert today.is_ready
    assert _decisions(today) == expected


async def test_events_that_arrive_while_it_loads_are_learned_in_order_after_the_replay(tmp_path, events):
    full = VerifiedOnlineLearner(tmp_path / "full.json")
    await full.initialize(_Ledger(events[:3_000]))

    loading = VerifiedOnlineLearner(tmp_path / "loading.json")
    loading.begin_loading()
    for event in reversed(events[2_000:3_000]):  # delivered live, out of order, while the replay has not reached them
        await loading.consume(event)
    assert loading.status()["event_cursor"] == 0  # held, not learned
    await loading.initialize(_Ledger(events[:2_000]))
    assert _decisions(loading) == _decisions(full)


@pytest.mark.parametrize(
    ("spoil", "words"),
    [
        (lambda raw: raw[:-5] + b"xxxxx", "does not match its key"),
        (lambda raw: raw.replace(MODEL_VERSION.encode(), b"verified-online-v0", 1), "does not match its key"),
        (lambda raw: b"not a snapshot\n" + raw, "not in a known format"),
    ],
)
async def test_a_spoiled_snapshot_is_refused_and_the_whole_ledger_is_replayed(tmp_path, events, spoil, words):
    small = events[:4_000]
    earlier = VerifiedOnlineLearner(tmp_path / "state.json")
    await earlier.initialize(_Ledger(small))
    snapshot = tmp_path / "snap.bin"
    await earlier.save_snapshot(snapshot, KEY)
    snapshot.write_bytes(spoil(snapshot.read_bytes()))

    learner = VerifiedOnlineLearner(tmp_path / "state.json")
    ledger = _Ledger(small)
    loaded = await learner.initialize(ledger, snapshot=snapshot, key=KEY)
    assert loaded["from_snapshot"] is False and words in loaded["refused"]
    assert loaded["replayed"] == len(small)
    assert _decisions(learner) == _decisions(earlier)


async def test_a_different_key_or_river_version_is_refused(tmp_path, events, monkeypatch):
    earlier = VerifiedOnlineLearner(tmp_path / "state.json")
    await earlier.initialize(_Ledger(events[:1_000]))
    snapshot = tmp_path / "snap.bin"
    await earlier.save_snapshot(snapshot, KEY)
    with pytest.raises(learner_snapshot.SnapshotRefused, match="does not match its key"):
        learner_snapshot.read(snapshot, b"another key", model_version=MODEL_VERSION, reset_before=0)
    monkeypatch.setattr(learner_snapshot, "_versions", lambda: {"river": "0.99.0", "numpy": "9"})
    with pytest.raises(learner_snapshot.SnapshotRefused, match="different river"):
        learner_snapshot.read(snapshot, KEY, model_version=MODEL_VERSION, reset_before=0)


class _Anything:
    pass


def test_a_correctly_signed_snapshot_that_names_anything_else_is_refused_before_it_is_built(tmp_path):
    """Defence in depth: even with the right key, only the allow-listed classes can be built."""
    payload = pickle.dumps({"_cursor": _Anything()}, protocol=5)
    header = {
        "format": learner_snapshot.FORMAT,
        "model_version": MODEL_VERSION,
        "reset_before_sequence": 0,
        "sequence": 1,
        **learner_snapshot._versions(),
    }
    header["hmac"] = learner_snapshot._sign(KEY, header, payload)
    path = tmp_path / "snap.bin"
    path.write_bytes(json.dumps(header, sort_keys=True).encode() + b"\n" + payload)
    with pytest.raises(learner_snapshot.SnapshotRefused, match="never contains"):
        learner_snapshot.read(path, KEY, model_version=MODEL_VERSION, reset_before=0)
    with pytest.raises(learner_snapshot.SnapshotRefused):
        learner_snapshot._Restricted(io.BytesIO(pickle.dumps(print))).load()
