"""Every durable task can be joined to the experience ledger: each path that creates one also writes an intent event
with the same task_id, whose provenance says which Heliox wrote it (written_by)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pilot import __version__

PILOT = Path(__file__).resolve().parents[1] / "pilot"


def test_every_durable_task_is_created_through_the_one_path_that_writes_its_intent():
    creating = [
        f"{path.relative_to(PILOT.parent).as_posix()}:{number}"
        for path in PILOT.rglob("*.py")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"_durable_tasks\.create_task\(", line)
    ]
    # The only direct call is inside PilotServer._create_durable_task.
    assert len(creating) == 1, creating
    server = (PILOT / "server.py").read_text(encoding="utf-8")
    helper = server[server.index("async def _create_durable_task") :]
    helper = helper[: helper.index("\n    async def ", 10)]
    assert "_durable_tasks.create_task(" in helper and "ExperienceEventType.INTENT" in helper


@pytest.mark.parametrize("accepted_as", ["queued", "interactive"])
async def test_a_created_task_has_an_intent_with_its_task_id_and_the_stamp(tmp_path, accepted_as):
    from pilot.config import PilotConfig
    from pilot.intelligence.experience import ExperienceLedger
    from pilot.server import PilotServer
    from pilot.workflows.durable_tasks import DurableTaskStore

    server = PilotServer(PilotConfig())
    server._durable_tasks = DurableTaskStore(tmp_path / "durable_tasks.db")
    await server._durable_tasks.initialize()
    server._experience_ledger = ExperienceLedger(tmp_path / "experience.db")
    await server._experience_ledger.initialize()
    try:
        created = await server._create_durable_task(
            accepted_as=accepted_as, session_id="s", user_id="u", user_input="test command"
        )
        events = await server._experience_ledger.list_events(task_id=created.task.task_id)
    finally:
        await server._durable_tasks.close()
        await server._experience_ledger.close()

    assert [event.event_type.value for event in events] == ["intent"]
    assert events[0].source == accepted_as and events[0].session_id == "s"
    assert events[0].provenance["written_by"] == {"version": __version__, "run": "test root"}


async def test_a_queued_task_that_never_runs_still_has_its_intent(tmp_path, monkeypatch):
    from pilot.config import PilotConfig
    from pilot.intelligence.experience import ExperienceLedger
    from pilot.server import PilotServer
    from pilot.workflows.durable_tasks import DurableTaskStore

    server = PilotServer(PilotConfig())
    server._durable_tasks = DurableTaskStore(tmp_path / "durable_tasks.db")
    await server._durable_tasks.initialize()
    server._experience_ledger = ExperienceLedger(tmp_path / "experience.db")
    await server._experience_ledger.initialize()
    server._broadcast_notification = lambda *_a, **_k: _nothing()
    monkeypatch.setattr(server, "_start_queued_execute_worker", lambda: None)  # it never runs
    try:
        queued = await server._queue_interactive_request({"input": "test command"})
        events = await server._experience_ledger.list_events(task_id=queued["task_id"])
    finally:
        await server._durable_tasks.close()
        await server._experience_ledger.close()

    assert queued["status"] == "queued"
    assert [event.event_type.value for event in events] == ["intent"]


async def _nothing() -> None:
    return None
