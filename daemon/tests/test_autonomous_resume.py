"""Autonomous jobs survive a daemon restart as interrupted jobs the person can resume."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.agents.autonomous import AutonomousExecutor, AutonomousJob, JobStatus, JobStep
from pilot.agents.decomposer import Subtask, TaskDecomposer, TaskDecomposition
from pilot.config import PilotConfig
from pilot.security.gateway import TaskScopeOverride
from pilot.server import PilotServer
from pilot.workflows.autonomous_jobs import INTERRUPTED_SUMMARY, AutonomousJobStore


def _decomposer() -> SimpleNamespace:
    async def decompose(goal: str) -> TaskDecomposition:
        return TaskDecomposition(
            goal=goal,
            is_complex=True,
            subtasks=[
                Subtask(id="a1", title="Collect", description="collect the numbers", order=0),
                Subtask(id="b2", title="Report", description="write the report", depends_on=["0"], order=1),
            ],
        )

    return SimpleNamespace(
        decompose=decompose,
        get_execution_order=lambda decomposition: TaskDecomposer(MagicMock()).get_execution_order(decomposition),
    )


def _executor(store: AutonomousJobStore, calls: list, block_step: int | None = None, started=None):
    autonomous = AutonomousExecutor(
        planner=None, executor=None, verifier=None, decomposer=_decomposer(), job_store=store
    )
    autonomous.set_broadcast(AsyncMock())
    autonomous.set_speech(AsyncMock())

    async def fake_loop(job, step, goal, *, initial_progress=None, **_kwargs):
        calls.append((step.index, goal, list(initial_progress or [])))
        step.rounds = 2
        if step.index == block_step:
            started.set()
            await asyncio.sleep(60)
        step.status = "success"
        step.output = f"output of step {step.index}"

    autonomous._execute_goal_loop = fake_loop
    return autonomous


async def test_store_marks_cut_off_jobs_interrupted_and_keeps_them_through_pruning(tmp_path):
    store = AutonomousJobStore(tmp_path / "jobs.db")
    await store.save({"job_id": "done", "status": "success", "steps": []})
    await store.save({"job_id": "busy", "status": "running", "steps": [{"status": "success"}, {"status": "running"}]})

    assert await store.mark_interrupted() == ["busy"]

    records = {record["job_id"]: record for record in await store.load_recent()}
    assert records["busy"]["status"] == "interrupted"
    assert records["busy"]["result_summary"] == INTERRUPTED_SUMMARY
    assert [step["status"] for step in records["busy"]["steps"]] == ["success", "interrupted"]
    assert records["done"]["status"] == "success"
    assert await store.mark_interrupted() == []

    assert await store.prune(keep=0) == 1
    assert [record["job_id"] for record in await store.load_recent()] == ["busy"]


async def test_shutdown_interrupts_a_job_and_a_new_daemon_resumes_only_unfinished_steps(tmp_path):
    store = AutonomousJobStore(tmp_path / "jobs.db")
    first_calls: list = []
    started = asyncio.Event()
    first = _executor(store, first_calls, block_step=1, started=started)
    job = await first.submit("collect numbers then report", scope_override=TaskScopeOverride(allow_root=False))
    await started.wait()
    first._broadcast.reset_mock()

    await first.stop()

    first._broadcast.assert_not_awaited()  # shutdown stays quiet
    assert job.status == JobStatus.INTERRUPTED
    saved = (await store.load_recent())[0]
    assert saved["status"] == "interrupted"
    assert [step["status"] for step in saved["steps"]] == ["success", "interrupted"]

    second_calls: list = []
    second = _executor(store, second_calls)
    interrupted = await second.recover()

    assert [item.job_id for item in interrupted] == [job.job_id]
    restored = second.get_job(job.job_id)
    assert restored.to_dict()["resumable"] is True
    assert restored.scope_override == TaskScopeOverride(allow_root=False)
    assert any(item["job_id"] == job.job_id and item["status"] == "interrupted" for item in second.list_jobs())

    resumed = await second.resume(job.job_id)
    assert resumed is restored
    await second._active_tasks[job.job_id]

    assert restored.status == JobStatus.SUCCESS
    assert [call[0] for call in second_calls] == [1]  # step 0 finished before the shutdown
    step_index, goal, progress = second_calls[0]
    assert "output of step 0" in goal  # the finished prerequisite's output still feeds the next step
    assert "interrupted after 2 round(s)" in progress[0]
    assert restored.resumed_count == 1
    assert (await store.load_recent())[0]["status"] == "success"
    assert await second.resume(job.job_id) is None  # finished jobs cannot be resumed again


async def test_a_crash_leaves_a_running_row_that_comes_back_interrupted_and_can_be_dismissed(tmp_path):
    store = AutonomousJobStore(tmp_path / "jobs.db")
    crashed = AutonomousJob(
        job_id="crash1",
        goal="tidy the downloads folder",
        status=JobStatus.RUNNING,
        steps=[JobStep(index=0, title="Execute", description="tidy", status="running", rounds=1)],
        total_steps=1,
        started_at=1.0,
    )
    await store.save(crashed.to_record())

    autonomous = _executor(store, [])
    assert [job.job_id for job in await autonomous.recover()] == ["crash1"]

    assert await autonomous.cancel("crash1") is True
    assert autonomous.get_job("crash1").status == JobStatus.CANCELLED
    assert (await store.load_recent())[0]["status"] == "cancelled"
    assert await autonomous.resume("crash1") is None


async def test_single_step_resume_tells_the_planner_to_continue_from_what_is_on_screen(tmp_path):
    store = AutonomousJobStore(tmp_path / "jobs.db")
    interrupted = AutonomousJob(
        job_id="one",
        goal="rename the report",
        status=JobStatus.INTERRUPTED,
        steps=[JobStep(index=0, title="Execute", description="rename", status="interrupted", rounds=3)],
        total_steps=1,
        started_at=1.0,
    )
    await store.save(interrupted.to_record())
    calls: list = []
    autonomous = _executor(store, calls)
    await autonomous.recover()

    await autonomous.resume("one")
    await autonomous._active_tasks["one"]

    assert autonomous.get_job("one").status == JobStatus.SUCCESS
    assert calls[0][1] == "rename the report"
    assert "interrupted after 3 round(s)" in calls[0][2][0]


async def test_step_outputs_are_redacted_and_capped_before_they_are_saved():
    step = JobStep(
        index=0, title="t", description="d", output="api_key=sk-abcdefghijklmnopqrstuvwxyz123456 " + "x" * 5000
    )

    record = step.to_record()

    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in record["output"]
    assert len(record["output"]) <= 2100


async def test_resume_rpc_refuses_jobs_that_are_not_interrupted(tmp_path):
    server = PilotServer(PilotConfig())
    store = AutonomousJobStore(tmp_path / "jobs.db")
    await store.save(
        AutonomousJob(job_id="paused", goal="g", status=JobStatus.INTERRUPTED, steps=[], total_steps=0).to_record()
    )
    server._autonomous = _executor(store, [])
    await server._autonomous.recover()

    assert (await server._handle_autonomous_resume({"job_id": "missing"}, ws=AsyncMock()))["status"] == "error"
    assert (await server._handle_autonomous_resume({}, ws=AsyncMock()))["status"] == "error"
    result = await server._handle_autonomous_resume({"job_id": "paused"}, ws=AsyncMock())
    assert result["status"] == "resumed" and result["job"]["job_id"] == "paused"
    await server._autonomous._active_tasks["paused"]


@pytest.mark.parametrize("status", ["success", "failed", "cancelled", "running"])
async def test_only_interrupted_jobs_resume(tmp_path, status):
    store = AutonomousJobStore(tmp_path / "jobs.db")
    autonomous = _executor(store, [])
    autonomous._jobs["j"] = AutonomousJob(job_id="j", goal="g", status=JobStatus(status))

    assert await autonomous.resume("j") is None
