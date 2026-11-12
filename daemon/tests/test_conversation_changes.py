"""What a voice turn changed is read from a record (pilot.turn_changes), never from counting a few stores.

Every code-answered handler that changes something outside the conversation records it with the turn; a sentence
split at a pause whose first half did so is "too late", Heliox says what it did, and the change is neither made
twice nor silently lost. A guard keeps it so: every store that writes the person's data must record the write."""

from __future__ import annotations

import ast
import asyncio
import tempfile
from pathlib import Path

import pytest

from benchmarks.conversation_eval import sessions
from pilot.system.voice import voice_turn_addressed, voice_turn_id
from pilot.turn_changes import note_change, turn_changes

DAEMON = Path(__file__).resolve().parents[1]

# Writers that never run inside a person's voice turn as a code answer, and why.
NOT_A_TURN_CHANGE = {
    "pilot/workflows/task_receipts.py": "the receipt of a task the executor ran (that turn is already too late)",
    "pilot/workflows/undo_journal.py": "the undo record of an executed action (already too late)",
    "pilot/workflows/daily_brief.py": "when the scheduled brief was last sent",
    "pilot/workflows/durable_tasks.py": "the planner road's task store (executor work)",
    "pilot/workflows/checkpoints.py": "the planner road's checkpoints (executor work)",
    "pilot/workflows/autonomous_jobs.py": "background jobs (started by the planner road)",
    "pilot/workflows/recorded_workflows.py": "recordings made from the app, not by a spoken answer",
    "pilot/workflows/scheduled_runs.py": "scheduled runs (created through the planner road with approval)",
    "pilot/workflows/voice_gesture_workflows.py": "workflow state of the planner road",
    "pilot/workflows/privacy.py": "exports made from the app",
    "pilot/personal/bench.py": "a benchmark's own files",
}
WRITES = {"write_text", "write_bytes"}  # a file written; json.dump and os.replace below


def _is_write(call: ast.Call) -> bool:
    func = call.func
    if not isinstance(func, ast.Attribute):
        return False
    owner = func.value.id if isinstance(func.value, ast.Name) else ""
    return (
        func.attr in WRITES or (func.attr == "dump" and owner == "json") or (func.attr == "replace" and owner == "os")
    )


def _writes_without_record(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
        writes = any(_is_write(c) for c in calls)
        records = any(
            (isinstance(c.func, ast.Name) and c.func.id == "note_change")
            or (isinstance(c.func, ast.Attribute) and c.func.attr == "note_change")
            for c in calls
        )
        if writes and not records:
            missing.append(node.name)
    return missing


def _store_files() -> list[Path]:
    return sorted([*DAEMON.glob("pilot/workflows/*.py"), *DAEMON.glob("pilot/personal/*.py")])


@pytest.mark.parametrize("path", _store_files(), ids=lambda p: p.name)
def test_every_store_that_keeps_the_persons_data_records_the_change(path):
    rel = path.relative_to(DAEMON).as_posix()
    if rel in NOT_A_TURN_CHANGE:
        pytest.skip(NOT_A_TURN_CHANGE[rel])
    missing = [name for name in _writes_without_record(path) if not name.startswith(("_load", "export", "_export"))]
    assert missing == [], f"{rel}: {missing} write the person's data without note_change"


def test_outside_a_turn_recording_does_nothing_and_inside_it_records_once():
    note_change("set the reminder")  # no turn: nothing to record against
    changes: list[str] = []
    token = turn_changes.set(changes)
    try:
        note_change("set the reminder")
        note_change("set the reminder")
        note_change("")
    finally:
        turn_changes.reset(token)
    assert changes == ["set the reminder"]


async def test_a_store_saving_in_a_worker_thread_still_records_against_the_turn(tmp_path):
    from pilot.workflows.todos import TodoStore

    store = TodoStore(tmp_path / "todos.json")
    changes: list[str] = []
    turn_changes.set(changes)
    await asyncio.to_thread(store.answer, "add milk to my to-do list")
    assert changes == ["updated your to-do list"]


# ── a sentence split at a pause after a change kept in code ──

CHANGES = [
    ("add milk to my to-do list", "updated your to-do list", lambda s: len(s._todos.open_items())),
    ("note that the boiler code is 4417", "saved the note", lambda s: len(s._notes.list())),
    ("remind me in 20 minutes to stretch", "updated your reminders", lambda s: len(s._reminders.list())),
    ("my flight is on Friday", "remembered that", lambda s: len(s._user_profile.list())),
    ("start a pomodoro", "changed the focus session", lambda s: int(s._focus.active)),
    ("when I say start work, open VS Code", "saved the shortcut", lambda s: len(s._shortcuts.list())),
]


@pytest.mark.parametrize(("said", "what", "count"), CHANGES, ids=[c[0] for c in CHANGES])
async def test_split_after_a_change_kept_in_code_is_too_late_and_says_what_it_did(said, what, count):
    record, clock = sessions.Record(), sessions.Clock()
    with tempfile.TemporaryDirectory() as tmp:
        undo = sessions._isolate(Path(tmp))
        try:
            server = await sessions.build_server("none", {"turns": []}, Path(tmp), record, clock)
            # The rest of the sentence comes while Heliox is still answering (it yields at once): no time passes.
            server._voice_turns._clock = lambda: 0.0
            before = count(server)

            async def first_half():
                voice_turn_addressed.set("wake")
                voice_turn_id.set("v1")
                await server._voice_command_dispatch(said)

            await asyncio.ensure_future(first_half())
            once = count(server)
            result = await server.withdraw_voice_turn("v1", f"{said} please")
            await asyncio.sleep(0)
            assert result.status == "too_late"
            assert what in result.said and result.said.startswith("I've already")
            assert count(server) == once != before  # kept once: not withdrawn, not made again
            assert record.said[-1].text == result.said  # and Heliox says what it did
        finally:
            for step in undo:
                step()


async def test_a_read_only_answer_can_still_be_withdrawn():
    record, clock = sessions.Record(), sessions.Clock()
    with tempfile.TemporaryDirectory() as tmp:
        undo = sessions._isolate(Path(tmp))
        try:
            server = await sessions.build_server("none", {"turns": []}, Path(tmp), record, clock)

            async def first_half():
                voice_turn_addressed.set("wake")
                voice_turn_id.set("v2")
                await server._voice_command_dispatch("what's on my list")

            await asyncio.ensure_future(first_half())
            assert (await server.withdraw_voice_turn("v2", "what's on my list for tomorrow")).status == "withdrawn"
            await asyncio.gather(*getattr(server, "_background_dispatches", set()))  # the joined sentence's turn
        finally:
            for step in undo:
                step()
