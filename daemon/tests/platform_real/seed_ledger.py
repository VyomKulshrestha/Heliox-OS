"""Fill Heliox's experience ledger with synthetic events, and optionally save the learner's state, so a CI start can
show the learner's load on a real machine. Run with the environment's Python, with HOME set to the CI's empty home.

    python seed_ledger.py 100000 [--snapshot]

The events are a deterministic mix of verified one-step plans, answered suggestions and screen observations, written
straight into the ledger's table in one transaction. ``--snapshot`` then replays them into a learner and saves its
state with the same key the daemon will use (the key file, when the job turns the keyring off), so the next start
reads only what comes after it. Prints one line: how many events, and whether a state was saved.
"""

from __future__ import annotations

import asyncio
import json
import random
import sqlite3
import sys
from datetime import UTC, datetime, timedelta

ACTIONS = ("file_write", "file_read", "shell_command", "browser_navigate")
PATTERNS = ("break_reminder", "save_reminder", "focus_mode")


def rows(count: int) -> list[tuple]:
    rng = random.Random(7)
    start = datetime(2026, 1, 5, 9, tzinfo=UTC)
    out: list[tuple] = []
    plan = 0

    def add(event_type, *, plan_id="", action_id="", source="interactive", payload=None):
        n = len(out) + 1
        at = (start + timedelta(minutes=n)).isoformat()
        out.append(
            (
                f"seed-{n}",
                event_type,
                at,
                1,
                "seed",
                "",
                "",
                plan_id,
                action_id,
                "",
                f"seed-key-{n}",
                source,
                json.dumps(payload or {}),
                "{}",
                None,
                "internal",
            )
        )

    while len(out) < count:
        kind = rng.random()
        if kind < 0.6:
            plan += 1
            action = ACTIONS[plan % len(ACTIONS)]
            ok = rng.random() < (0.85 if action != "shell_command" else 0.25)
            plan_id, action_id = f"seed-p{plan}", f"seed-a{plan}"
            add("action_started", plan_id=plan_id, action_id=action_id, payload={"index": 0, "action_type": action})
            add(
                "action_completed",
                plan_id=plan_id,
                action_id=action_id,
                payload={"index": 0, "success": ok, "executed": True, "callback_observed": True},
            )
            add(
                "outcome_verified",
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
            )
        elif kind < 0.8:
            suggestion = f"seed-s{len(out)}"
            pattern = PATTERNS[len(out) % len(PATTERNS)]
            add("suggestion_shown", source="proactive", payload={"suggestion_id": suggestion, "pattern_id": pattern})
            decision = "accepted" if rng.random() < 0.5 else "dismissed"
            add("suggestion_feedback", source="proactive", payload={"suggestion_id": suggestion, "decision": decision})
        else:
            add("observation", source="screen", payload={"active_app": "code"})
    return out[:count]


async def main(count: int, snapshot: bool) -> None:
    from pilot.config import DATA_DIR, EXPERIENCE_DB_FILE, PilotConfig
    from pilot.intelligence.experience import ExperienceLedger

    ledger = ExperienceLedger(EXPERIENCE_DB_FILE)
    await ledger.initialize()  # makes the table as the daemon does
    await ledger.close()
    with sqlite3.connect(EXPERIENCE_DB_FILE) as db:
        db.executemany(
            "INSERT INTO experience_events (event_id, event_type, occurred_at, schema_version, session_id, task_id, "
            "user_id, plan_id, action_id, parent_event_id, idempotency_key, source, payload_json, provenance_json, "
            "confidence, privacy_class) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows(count),
        )
    saved = False
    if snapshot:
        from pilot.intelligence.online_learning import VerifiedOnlineLearner
        from pilot.server import LEARNER_SNAPSHOT_FILE, PilotServer

        ledger = ExperienceLedger(EXPERIENCE_DB_FILE)
        await ledger.initialize()
        learner = VerifiedOnlineLearner(DATA_DIR / "online_learning_state.json")
        await learner.initialize(ledger)
        key = await PilotServer(PilotConfig.load())._learner_key()  # the daemon's own choice of key
        if key is not None:
            await learner.save_snapshot(DATA_DIR / LEARNER_SNAPSHOT_FILE, key)
            saved = True
        await ledger.close()
    print(f"SEED events={count} snapshot={'saved' if saved else 'none'} data={DATA_DIR}")


if __name__ == "__main__":
    from ci_only import require_runner  # here, not at the top: the tests import this file for its helpers

    require_runner("writes 100,000 synthetic events into Heliox's experience ledger in the home folder")
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 100_000, "--snapshot" in sys.argv))
