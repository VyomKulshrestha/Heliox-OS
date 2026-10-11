"""Scheduled agent runs: timetable, persistence, and the approved-action limit."""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, EmptyParams, ShellCommandParams
from pilot.agents.autonomous import AutonomousExecutor, AutonomousJob, JobStatus, JobStep, restrict_scope_to_actions
from pilot.config import PilotConfig
from pilot.security.gateway import DEFAULT_SOURCE_PROFILES, TaskScopeOverride, resolve_effective_profile
from pilot.server import PilotServer
from pilot.workflows.scheduled_runs import (
    MISSED_GRACE_SECONDS,
    ScheduledRun,
    ScheduledRunner,
    ScheduledRunStore,
    next_run_after,
    validate_schedule,
)

READ_ONLY = ["system_info", "disk_usage", "notify"]


def _ts(year, month, day, hour, minute) -> float:
    return datetime(year, month, day, hour, minute).timestamp()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"name": ""}, "Name"),
        ({"goal": ""}, "Goal"),
        ({"allowed_actions": []}, "at least one action"),
        ({"allowed_actions": ["system_info", "launch_rockets"]}, "Unknown action types: launch_rockets"),
        ({"every_minutes": 2}, "between 5"),
        ({"every_minutes": 0, "daily_at": ""}, "either every_minutes or daily_at"),
        ({"every_minutes": 30, "daily_at": "09:00"}, "either every_minutes or daily_at"),
        ({"every_minutes": 0, "daily_at": "25:00"}, "HH:MM"),
        ({"every_minutes": 0, "daily_at": "09:00", "weekdays": [7]}, "weekdays"),
        ({"weekdays": [1]}, "only apply to daily_at"),
    ],
)
def test_schedule_validation_explains_what_to_fix(changes, message):
    spec = {"name": "Disk report", "goal": "report free disk space", "allowed_actions": READ_ONLY, "every_minutes": 60}
    spec.update(changes)
    with pytest.raises(ValueError, match=message):
        validate_schedule(**spec)


def test_next_run_times_follow_intervals_daily_times_and_weekdays():
    interval = ScheduledRun(run_id="a", name="n", goal="g", allowed_actions=READ_ONLY, every_minutes=30)
    assert next_run_after(interval, 1000.0) == 1000.0 + 1800

    daily = ScheduledRun(run_id="b", name="n", goal="g", allowed_actions=READ_ONLY, daily_at="09:00")
    assert next_run_after(daily, _ts(2026, 10, 12, 8, 0)) == _ts(2026, 10, 12, 9, 0)  # later today
    assert next_run_after(daily, _ts(2026, 10, 12, 9, 0)) == _ts(2026, 10, 13, 9, 0)  # strictly after

    weekdays_only = ScheduledRun(
        run_id="c", name="n", goal="g", allowed_actions=READ_ONLY, daily_at="09:00", weekdays=[0, 1, 2, 3, 4]
    )
    friday_evening = _ts(2026, 10, 16, 18, 0)  # 2026-10-16 is a Friday
    assert next_run_after(weekdays_only, friday_evening) == _ts(2026, 10, 19, 9, 0)  # Monday


async def test_schedules_survive_a_restart_of_the_store(tmp_path):
    runner = ScheduledRunner(ScheduledRunStore(tmp_path / "s.db"), submit=AsyncMock(), job_is_active=lambda _id: False)
    run = await runner.create(
        name="Disk report", goal="report free disk space", allowed_actions=["disk_usage", "notify"], every_minutes=60
    )

    reopened = ScheduledRunner(ScheduledRunStore(tmp_path / "s.db"), submit=AsyncMock(), job_is_active=lambda _: False)
    [loaded] = await reopened.list()

    assert loaded == run
    assert loaded.allowed_actions == ["disk_usage", "notify"]
    assert await reopened.delete(run.run_id) is True
    assert await reopened.list() == []


async def test_tick_starts_due_runs_skips_missed_and_busy_ones(tmp_path):
    submitted: list[str] = []
    active: set[str] = set()

    async def submit(run):
        submitted.append(run.name)
        return f"job-{run.name}"

    runner = ScheduledRunner(ScheduledRunStore(tmp_path / "s.db"), submit=submit, job_is_active=active.__contains__)
    now = 1_800_000_000.0
    due = await runner.create(name="due", goal="g", allowed_actions=READ_ONLY, every_minutes=10, now=now - 600)
    missed = await runner.create(name="missed", goal="g", allowed_actions=READ_ONLY, every_minutes=5, now=now - 3600)
    busy = await runner.create(name="busy", goal="g", allowed_actions=READ_ONLY, every_minutes=10, now=now - 600)
    later = await runner.create(name="later", goal="g", allowed_actions=READ_ONLY, every_minutes=60, now=now)
    off = await runner.create(name="off", goal="g", allowed_actions=READ_ONLY, every_minutes=10, now=now - 600)
    await runner.set_enabled(off.run_id, False)
    store = ScheduledRunStore(tmp_path / "s.db")
    busy_record = await store.get(busy.run_id)
    busy_record.last_job_id = "still-going"
    await store.save(busy_record)
    active.add("still-going")
    assert missed.next_run_at < now - MISSED_GRACE_SECONDS

    started = await runner.tick(now=now)

    assert started == [due.run_id]
    assert submitted == ["due"]
    runs = {run.name: run for run in await runner.list()}
    assert runs["due"].last_status == "started" and runs["due"].last_job_id == "job-due"
    assert runs["due"].next_run_at == now + 600
    assert runs["missed"].last_status == "missed" and runs["missed"].next_run_at == now + 300
    assert runs["busy"].last_status == "skipped_still_running"
    assert runs["later"].last_status == "" and runs["later"].next_run_at == later.next_run_at
    assert runs["off"].last_status == "" and runs["off"].enabled is False


async def test_run_now_starts_immediately_and_reports_submit_failures(tmp_path):
    async def failing_submit(_run):
        raise RuntimeError("Autonomous executor not initialized")

    runner = ScheduledRunner(ScheduledRunStore(tmp_path / "s.db"), submit=failing_submit, job_is_active=lambda _: False)
    run = await runner.create(name="n", goal="g", allowed_actions=READ_ONLY, daily_at="07:30")

    result = await runner.run_now(run.run_id)

    assert result.last_status == "failed_to_start"
    assert result.next_run_at == run.next_run_at  # the timetable is unchanged
    assert await runner.run_now("missing") is None


def test_the_scope_override_denies_every_action_off_the_list_and_keeps_existing_denials():
    override = restrict_scope_to_actions(TaskScopeOverride(deny_action_types=["notify"]), ["system_info", "notify"])

    assert "shell_command" in override.deny_action_types
    assert "system_info" not in override.deny_action_types
    assert "notify" in override.deny_action_types  # a denial the caller already made stays
    profile = resolve_effective_profile(DEFAULT_SOURCE_PROFILES["autonomous"], override)
    assert "file_delete" in profile.deny_action_types


def _plan(action_type: ActionType, params) -> ActionPlan:
    return ActionPlan(actions=[Action(action_type=action_type, target="", parameters=params)], explanation="plan")


async def test_the_goal_loop_refuses_plans_outside_the_list_even_without_the_gateway():
    plans = [
        _plan(ActionType.SHELL_COMMAND, ShellCommandParams(command="del", args=["C:\\notes"])),
        _plan(ActionType.SYSTEM_INFO, EmptyParams()),
    ]
    seen_contexts: list[str] = []

    async def plan(goal, **kwargs):
        seen_contexts.append(kwargs["screen_context"])
        return plans.pop(0)

    executed: list[str] = []

    async def execute(plan, **_kwargs):
        executed.extend(action.action_type.value for action in plan.actions)
        return [ActionResult(action=action, success=True, output="CPU: 8 cores") for action in plan.actions]

    async def verify(plan, results):
        return SimpleNamespace(passed=True, details=["system info read"], independently_observed_actions=[])

    autonomous = AutonomousExecutor(
        planner=SimpleNamespace(plan=plan),
        executor=SimpleNamespace(execute=execute),
        verifier=SimpleNamespace(verify=verify),
        decomposer=None,
    )
    job = AutonomousJob(job_id="s1", goal="report system info", allowed_actions=["system_info"])
    step = JobStep(index=0, title="Execute", description=job.goal)

    await autonomous._execute_goal_loop(job, step, job.goal)

    assert executed == ["system_info"]  # the shell command never reached the executor
    assert step.status == "success"
    assert "shell_command is not on this run's approved list" in seen_contexts[1]


async def test_submit_with_an_allowed_list_rejects_unknown_actions_and_records_the_list():
    autonomous = AutonomousExecutor(planner=None, executor=None, verifier=None, decomposer=None)
    with pytest.raises(ValueError):
        await autonomous.submit("g", allowed_actions=["not_an_action"])

    job = await autonomous.submit("g", allowed_actions=["notify", "system_info", "notify"])
    task = autonomous._active_tasks.pop(job.job_id)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert job.allowed_actions == ["notify", "system_info"]
    assert "shell_command" in job.scope_override.deny_action_types
    assert AutonomousJob.from_record(job.to_record()).allowed_actions == ["notify", "system_info"]


async def test_scheduled_run_rpcs_create_list_toggle_run_and_delete(tmp_path):
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    submitted = []

    async def submit(goal, **kwargs):
        submitted.append((goal, kwargs))
        return AutonomousJob(job_id="job1", goal=goal, status=JobStatus.RUNNING)

    server._autonomous = SimpleNamespace(
        submit=submit,
        get_job=lambda job_id: AutonomousJob(job_id=job_id, status=JobStatus.SUCCESS) if job_id == "job1" else None,
    )
    server._scheduled_runs = ScheduledRunner(
        ScheduledRunStore(tmp_path / "s.db"),
        submit=server._submit_scheduled_run,
        job_is_active=server._autonomous_job_is_active,
    )
    ws = AsyncMock()

    bad = await server._handle_scheduled_run_create({"name": "x", "goal": "g", "allowed_actions": ["nope"]}, ws)
    assert bad["status"] == "error" and "Unknown action types" in bad["message"]
    bad_type = await server._handle_scheduled_run_create(
        {"name": "x", "goal": "g", "allowed_actions": READ_ONLY, "every_minutes": "60"}, ws
    )
    assert bad_type["message"] == "every_minutes must be a whole number"

    created = await server._handle_scheduled_run_create(
        {"name": "Morning report", "goal": "summarise disk use", "allowed_actions": READ_ONLY, "daily_at": "08:15"},
        ws,
    )
    run_id = created["run"]["run_id"]
    assert created["status"] == "ok"

    started = await server._handle_scheduled_run_now({"run_id": run_id}, ws)
    assert started["status"] == "ok" and started["run"]["last_job_status"] == "success"
    goal, kwargs = submitted[0]
    assert "Only these actions are approved: disk_usage, notify, system_info" in goal
    assert kwargs["allowed_actions"] == ["disk_usage", "notify", "system_info"]
    assert kwargs["source"] == "scheduled"
    server._broadcast_notification.assert_awaited_with(
        "scheduled_run_dispatched", {"run_id": run_id, "name": "Morning report", "job_id": "job1"}
    )

    toggled = await server._handle_scheduled_run_set_enabled({"run_id": run_id, "enabled": False}, ws)
    assert toggled["run"]["enabled"] is False
    assert (await server._handle_scheduled_run_set_enabled({"run_id": run_id, "enabled": "no"}, ws))[
        "status"
    ] == "error"
    listed = await server._handle_scheduled_run_list({}, ws)
    assert [run["name"] for run in listed["runs"]] == ["Morning report"]
    assert (await server._handle_scheduled_run_delete({"run_id": run_id}, ws))["deleted"] is True
    assert (await server._handle_scheduled_run_list({}, ws))["runs"] == []


async def test_a_goal_the_approved_actions_cannot_reach_fails_honestly():
    async def plan(goal, **kwargs):
        assert "GOAL_BLOCKED" in kwargs["screen_context"]  # the planner is told how to say so
        return ActionPlan(
            actions=[], explanation="GOAL_BLOCKED: only notify is approved, so the disk cannot be measured"
        )

    executor = SimpleNamespace(execute=AsyncMock())
    autonomous = AutonomousExecutor(
        planner=SimpleNamespace(plan=plan), executor=executor, verifier=SimpleNamespace(), decomposer=None
    )
    job = AutonomousJob(job_id="b1", goal="report free disk space", allowed_actions=["notify"])
    step = JobStep(index=0, title="Execute", description=job.goal)

    await autonomous._execute_goal_loop(job, step, job.goal)

    assert step.status == "failed"
    assert step.error == "Blocked: only notify is approved, so the disk cannot be measured"
    executor.execute.assert_not_awaited()
