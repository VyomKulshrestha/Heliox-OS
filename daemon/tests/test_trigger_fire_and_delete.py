"""A trigger that fires does its job visibly, and one can be removed by saying which.

Live run, 26 November: a file_modified trigger stored the shell line a model wrote ("echo seen a change>>fired.log");
when it fired the line went to the planner as an autonomous job, which planned a shell step, waited for an approval
card nobody saw and never finished; nothing was said. "Delete the trigger that watches watched.txt" ran only
trigger_list, and deleting "nonexistent-trigger-42" said nothing useful.

Now a trigger keeps the exact steps it runs, fixed and approved when it is made; a fire runs those and nothing else,
bounded in time, with a receipt and one sentence. A step needing an approval nobody gave makes the fire wait as a
recorded request the person is told about. Delete resolves a description against the stored triggers in code.
Fakes only: no shell runs, the files are in a temporary folder, no model is called.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, FileParams, ShellCommandParams, TriggerParams
from pilot.config import PilotConfig
from pilot.system import triggers


def _shell(command: str) -> Action:
    return Action(action_type=ActionType.SHELL_COMMAND, target=command, parameters=ShellCommandParams(command=command))


def _append(path: Path) -> Action:
    return Action(
        action_type=ActionType.FILE_WRITE,
        target=str(path),
        parameters=FileParams(path=str(path), content="seen a change\n"),
    )


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> triggers.TriggerEngine:
    fresh = triggers.TriggerEngine()
    fresh.attach_store(tmp_path / "triggers.json")
    monkeypatch.setattr(triggers, "_engine", fresh)
    return fresh


class _Executor:
    """Records the plans it is given; runs nothing."""

    def __init__(self, hang: bool = False) -> None:
        self.plans: list[tuple[ActionPlan, dict[str, Any]]] = []
        self._hang = hang

    async def execute(self, plan: ActionPlan, **kwargs: Any) -> list[ActionResult]:
        self.plans.append((plan, kwargs))
        if self._hang:
            await asyncio.Event().wait()
        return [ActionResult(action=action, success=True, output="done") for action in plan.actions]


def _server(executor: Any = None) -> Any:
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server._running = True
    server._executor = executor
    server._broadcast_notification = AsyncMock()
    server._speak_companion_text = AsyncMock()
    server._task_receipts = SimpleNamespace(finalize=AsyncMock(return_value={"receipt_id": "r1"}))
    server._autonomous = SimpleNamespace(submit=AsyncMock(side_effect=AssertionError("no model plans a fixed trigger")))
    return server


def _told(server: Any) -> list[str]:
    return [
        call.args[1]["text"] for call in server._broadcast_notification.await_args_list if call.args[0] == "watch_fired"
    ]


# ── What is stored, and what making a trigger asks approval for ──────


def test_a_trigger_keeps_its_fixed_steps_and_their_approval_across_a_restart(engine, tmp_path: Path) -> None:
    made = engine.create_trigger(
        "fired_log",
        "file_modified",
        {"path": str(tmp_path / "watched.txt")},
        "append a line to fired.log",
        steps=[_shell("echo seen a change")],
        steps_approved=True,
    )
    stored = json.loads((tmp_path / "triggers.json").read_text(encoding="utf-8"))["triggers"][0]
    assert stored["steps"][0]["action_type"] == "shell_command" and stored["steps_approved"] is True

    again = triggers.TriggerEngine()
    again.attach_store(tmp_path / "triggers.json")
    loaded = again.get_trigger(made.id)
    assert loaded is not None and loaded.steps == made.steps and loaded.steps_approved is True
    assert [a.action_type for a in triggers.fire_actions(loaded)] == [ActionType.SHELL_COMMAND]


def test_a_store_written_before_steps_loads_as_words_only(tmp_path: Path) -> None:
    store = tmp_path / "triggers.json"
    entry = {
        "id": "old1",
        "name": "old",
        "trigger_type": "cpu_threshold",
        "condition": {},
        "action_command": "x",
        "enabled": True,
    }
    store.write_text(json.dumps({"version": 1, "engine_running": False, "triggers": [entry]}), encoding="utf-8")
    engine = triggers.TriggerEngine()
    engine.attach_store(store)
    old = engine.get_trigger("old1")
    assert old is not None and old.enabled and old.steps == [] and old.steps_approved is False and old.pending is None


def test_a_trigger_cannot_make_triggers() -> None:
    nested = Action(action_type=ActionType.TRIGGER_CREATE, target="x", parameters=TriggerParams(name="x"))
    with pytest.raises(ValueError, match="cannot run trigger_create"):
        triggers.validate_steps([nested])


def test_making_a_trigger_asks_for_the_approval_its_steps_need() -> None:
    from pilot.server import _approval_effect

    shell = Action(
        action_type=ActionType.TRIGGER_CREATE,
        target="fired_log",
        parameters=TriggerParams(name="fired_log", trigger_type="file_modified", steps=[_shell("echo seen a change")]),
    )
    write = shell.model_copy(update={"parameters": TriggerParams(name="n", steps=[_append(Path("fired.log"))])})
    words_only = shell.model_copy(update={"parameters": TriggerParams(name="n", action_command="echo hi")})
    assert shell.requires_confirmation is True
    assert write.requires_confirmation is False and words_only.requires_confirmation is False
    assert _approval_effect(shell) == (
        "Each time this trigger fires, Heliox runs this without asking again: shell command echo seen a change."
    )


@pytest.mark.parametrize(("approved", "kept"), [(True, True), (False, False)])
async def test_the_steps_are_kept_approved_only_when_the_plan_was_approved(engine, approved: bool, kept: bool) -> None:
    from tests.test_failed_actions_are_failures import _executor

    step = Action(
        action_type=ActionType.TRIGGER_CREATE,
        target="fired_log",
        parameters=TriggerParams(
            name="fired_log", trigger_type="cpu_threshold", action_command="echo", steps=[_shell("echo seen")]
        ),
    )
    run = _executor(ActionType.TRIGGER_CREATE, "_exec_trigger_create")
    result = await run._execute_single(step, snapshot_id=None, user_confirmed=approved)
    assert result.success, result.error
    [made] = engine._triggers.values()
    assert made.steps_approved is kept
    await engine.stop()


# ── A fire ───────────────────────────────────────────────────────────


async def test_a_fire_runs_exactly_the_approved_steps_and_says_so(engine, tmp_path: Path) -> None:
    made = engine.create_trigger(
        "fired_log",
        "file_modified",
        {"path": str(tmp_path / "watched.txt")},
        "append a line",
        steps=[_shell("echo seen a change")],
        steps_approved=True,
    )
    made.fire_count = 1
    executor = _Executor()
    server = _server(executor)

    await server._dispatch_reactive_trigger(made)

    [(plan, kwargs)] = executor.plans
    assert [a.model_dump(mode="json") for a in plan.actions] == made.steps  # what was approved, nothing else
    assert kwargs["user_confirmed"] is True and kwargs["plan_id"] == f"trigger-{made.id}-1"
    server._task_receipts.finalize.assert_awaited_once()
    assert _told(server) == [
        'The trigger "fired_log" (file modified: watched.txt) fired and ran: shell command echo seen a change.'
    ]
    assert engine.get_trigger(made.id).pending is None
    assert engine.get_trigger(made.id).last_outcome == _told(server)[0]


async def test_a_step_needing_an_approval_nobody_gave_waits_and_is_told_once_an_hour(
    engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pilot.server as server_module

    clock = [1_000_000.0]
    monkeypatch.setattr(server_module, "time", SimpleNamespace(time=lambda: clock[0], monotonic=lambda: clock[0]))
    made = engine.create_trigger(
        "fired_log",
        "file_modified",
        {"path": str(tmp_path / "watched.txt")},
        "append a line",
        steps=[_shell("echo seen a change")],
        steps_approved=False,
    )
    executor = _Executor()
    server = _server(executor)

    await server._dispatch_reactive_trigger(made)
    clock[0] += 60
    await server._dispatch_reactive_trigger(engine.get_trigger(made.id))
    clock[0] += 3600
    await server._dispatch_reactive_trigger(engine.get_trigger(made.id))

    assert executor.plans == []  # nothing ran
    sentence = (
        'The trigger "fired_log" (file modified: watched.txt) fired but did not run shell command echo seen a '
        "change, because that needs your approval and none was given when the trigger was made."
    )
    assert _told(server) == [sentence, sentence]  # the second fire, a minute later, was not said again
    pending = engine.get_trigger(made.id).pending
    assert pending is not None and pending["fires"] == 3 and pending["why"] == "needs approval"
    stored = json.loads((tmp_path / "triggers.json").read_text(encoding="utf-8"))["triggers"][0]
    assert stored["pending"]["fires"] == 3  # recorded, so it is still listed after a restart


async def test_a_fire_is_bounded_in_time(engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pilot.server as server_module

    monkeypatch.setattr(server_module, "TRIGGER_FIRE_SECONDS", 0)  # bounded() cancels at once: no real waiting
    made = engine.create_trigger(
        "slow", "cpu_threshold", {"threshold": 90}, "x", steps=[_append(tmp_path / "fired.log")], steps_approved=True
    )
    server = _server(_Executor(hang=True))

    await server._dispatch_reactive_trigger(made)

    assert _told(server) == [
        'The trigger "slow" (cpu threshold) fired, but its steps did not finish within 0 seconds, so they were stopped.'
    ]


async def test_a_trigger_stored_as_words_does_not_hold_an_approval_card(engine) -> None:
    """The run's case: the words go to the guarded autonomous job; its shell step is not waited for at a card."""
    made = engine.create_trigger("fired_log", "cpu_threshold", {"threshold": 90}, "echo seen a change>>fired.log")
    server = _server()
    job = SimpleNamespace(job_id="job-1", source="trigger", session_id=f"trigger:{made.id}")
    plan = ActionPlan(actions=[_shell("echo seen a change>>fired.log")])

    approved = await server._wait_for_autonomous_confirmation(job, plan, "autonomous-job-1:0:0")

    assert approved is False
    assert server._pending_confirms == {}  # no card waits
    methods = [call.args[0] for call in server._broadcast_notification.await_args_list]
    assert "confirm_required" not in methods
    assert _told(server) == [
        'The trigger "fired_log" (cpu threshold) fired but did not run shell command echo seen a change>>fired.log, '
        "because that needs your approval and none was given when the trigger was made."
    ]
    # The job's own end is not told a second time.
    server.__dict__["_trigger_jobs"] = {"job-1": made.id}
    await server._autonomous_broadcast(
        "autonomous_complete", {"job_id": "job-1", "source": "trigger", "status": "failed"}
    )
    assert len(_told(server)) == 1


async def test_a_file_trigger_fires_on_a_change_and_not_on_a_restart(engine, tmp_path: Path) -> None:
    watched = tmp_path / "watched.txt"
    watched.write_text("one", encoding="utf-8")
    os.utime(watched, (1_000, 1_000))
    made = engine.create_trigger("w", "file_modified", {"path": str(watched)}, "x", cooldown_seconds=0)
    assert await engine.evaluate_once() == []  # the first look sets the baseline

    restarted = triggers.TriggerEngine()
    restarted.attach_store(tmp_path / "triggers.json")
    assert await restarted.evaluate_once() == []  # a restart looks again; nothing changed, nothing fires
    os.utime(watched, (2_000, 2_000))
    assert [t.id for t in await restarted.evaluate_once()] == [made.id]


# ── Delete by description ────────────────────────────────────────────


@pytest.fixture
def two(engine, tmp_path: Path) -> tuple[triggers.Trigger, triggers.Trigger]:
    watched = engine.create_trigger(
        "watched_txt_seen_a_change", "file_modified", {"path": str(tmp_path / "watched.txt")}, "append"
    )
    cpu = engine.create_trigger("cpu_alert", "cpu_threshold", {"threshold": 90}, "tell me")
    return watched, cpu


async def _delete(description: str) -> ActionResult:
    from tests.test_failed_actions_are_failures import _run

    return await _run(ActionType.TRIGGER_DELETE, "_exec_trigger_delete", TriggerParams(name=description))


@pytest.mark.parametrize(
    "description",
    ["the trigger that watches watched.txt", "watched.txt", "stop watching that file", "WATCHED_TXT_SEEN_A_CHANGE"],
)
async def test_one_match_is_deleted_and_the_check_verifies_it(engine, two, description: str) -> None:
    from pilot.agents.postconditions.triggers_schedules import trigger_absent

    watched, cpu = two
    result = await _delete(description)

    assert result.success, result.error
    record = json.loads(result.output)  # the removed trigger's whole entry, for the reply sentence
    assert record["id"] == watched.id and record["deleted"] is True
    assert (record["trigger_type"], record["watched"], record["action_command"]) == (
        "file_modified",
        "watched.txt",
        "append",
    )
    assert (
        record["text"]
        == f'Deleted the trigger "watched_txt_seen_a_change" (file modified: watched.txt) (id {watched.id}).'
    )
    assert [t["id"] for t in engine.list_triggers()] == [cpu.id]
    outcome = await trigger_absent(result)
    assert watched.id in str(outcome)


async def test_several_matches_ask_which_and_delete_nothing(engine, tmp_path: Path) -> None:
    for name in ("notes_a", "notes_b"):
        engine.create_trigger(name, "file_modified", {"path": str(tmp_path / f"{name}.txt")}, "x")
    result = await _delete("the notes trigger")
    assert result.success is False
    assert (result.error or "").startswith('2 triggers match "the notes trigger": "notes_a"')
    assert (result.error or "").endswith("Nothing was deleted; say which one to delete.")
    assert len(engine.list_triggers()) == 2


async def test_a_name_that_matches_nothing_says_so_and_no_repair_round_guesses(engine, two) -> None:
    from pilot.server import _is_terminal_execution_failure

    result = await _delete("nonexistent-trigger-42")
    assert result.success is False
    assert result.error == (
        'No trigger matches "nonexistent-trigger-42", so nothing was deleted. The triggers there are: '
        '"watched_txt_seen_a_change" (file modified: watched.txt), "cpu_alert" (cpu threshold).'
    )
    assert len(engine.list_triggers()) == 2
    assert _is_terminal_execution_failure([result]) is True
