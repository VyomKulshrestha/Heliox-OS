"""Autonomous Executor — fire-and-forget background task pipeline.

Allows users to dispatch complex tasks that run in the background
while they continue working. Progress updates stream via WebSocket
notifications. Results queue up and are announced via TTS or UI.

Architecture:
  User: "Set up a React project and push to GitHub"
  → Decompose into subtasks
  → Execute each subtask via full ReAct pipeline (plan → execute → verify)
  → Stream progress: "Step 1/6: Creating project... done"
  → Announce completion via TTS
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from pilot.security.gateway import InvocationSource, TaskScopeOverride

if TYPE_CHECKING:
    from pilot.agents.decomposer import TaskDecomposer
    from pilot.agents.executor import Executor
    from pilot.agents.planner import Planner
    from pilot.agents.screen_vision import ScreenVisionAgent
    from pilot.agents.verifier import Verifier

logger = logging.getLogger("pilot.agents.autonomous")


class JobStatus(StrEnum):
    QUEUED = "queued"
    DECOMPOSING = "decomposing"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobStep:
    """A single step in an autonomous job."""

    index: int
    title: str
    description: str
    status: str = "pending"
    output: str = ""
    error: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0

    @property
    def duration_ms(self) -> int:
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at) * 1000)
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "output": self.output[:500] if self.output else "",
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class AutonomousJob:
    """A background job that runs a multi-step autonomous workflow."""

    job_id: str = field(default_factory=lambda: str(uuid.uuid4().hex)[:10])
    goal: str = ""
    status: JobStatus = JobStatus.QUEUED
    steps: list[JobStep] = field(default_factory=list)
    current_step: int = 0
    total_steps: int = 0
    started_at: float = 0.0
    completed_at: float = 0.0
    result_summary: str = ""
    source: str = "text"  # "text" or "voice" -- input modality, unrelated to gateway InvocationSource
    scope_override: TaskScopeOverride | None = None  # optional caller-supplied restriction (see AgentGateway)

    @property
    def duration_seconds(self) -> float:
        if self.started_at and self.completed_at:
            return round(self.completed_at - self.started_at, 1)
        elif self.started_at:
            return round(time.time() - self.started_at, 1)
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "goal": self.goal,
            "status": self.status.value,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "steps": [s.to_dict() for s in self.steps],
            "duration_seconds": self.duration_seconds,
            "result_summary": self.result_summary,
            "source": self.source,
        }


class AutonomousExecutor:
    """Manages fire-and-forget background job execution.

    Jobs are decomposed into steps, each executed through the full
    Plan → Execute → Verify pipeline. Progress is streamed to the
    UI and completion is announced via TTS.
    """

    def __init__(
        self,
        planner: Planner,
        executor: Executor,
        verifier: Verifier,
        decomposer: TaskDecomposer,
        screen_vision: ScreenVisionAgent | None = None,
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._verifier = verifier
        self._decomposer = decomposer
        self._screen_vision = screen_vision
        self._broadcast: Callable[[str, Any], Coroutine[Any, Any, None]] | None = None
        self._jobs: dict[str, AutonomousJob] = {}
        self._active_tasks: dict[str, asyncio.Task[None]] = {}

    def set_broadcast(self, fn: Callable[[str, Any], Coroutine[Any, Any, None]]) -> None:
        """Set the WebSocket broadcast function."""
        self._broadcast = fn

    async def submit(
        self, goal: str, source: str = "text", scope_override: TaskScopeOverride | None = None
    ) -> AutonomousJob:
        """Submit a new autonomous job. Returns immediately with a job handle."""
        job = AutonomousJob(goal=goal, source=source, scope_override=scope_override)
        self._jobs[job.job_id] = job

        # Launch in background — non-blocking
        task = asyncio.create_task(self._run_job(job))
        self._active_tasks[job.job_id] = task

        logger.info("Autonomous job submitted: [%s] %s", job.job_id, goal[:80])
        return job

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running job."""
        job = self._jobs.get(job_id)
        if not job:
            return False

        task = self._active_tasks.get(job_id)
        if task and not task.done():
            task.cancel()

        job.status = JobStatus.CANCELLED
        job.completed_at = time.time()
        await self._notify("autonomous_cancelled", job)
        return True

    def get_job(self, job_id: str) -> AutonomousJob | None:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        """List all jobs (recent first)."""
        sorted_jobs = sorted(self._jobs.values(), key=lambda j: j.started_at or 0, reverse=True)
        return [j.to_dict() for j in sorted_jobs[:50]]

    async def _run_job(self, job: AutonomousJob) -> None:
        """Execute a job through the full autonomous pipeline."""
        job.started_at = time.time()

        try:
            # Stage 1: Decompose the goal into subtasks
            job.status = JobStatus.DECOMPOSING
            await self._notify("autonomous_started", job)

            decomposition = await self._decomposer.decompose(job.goal)

            if decomposition.is_complex and decomposition.subtasks:
                # Complex task — execute each subtask
                job.total_steps = len(decomposition.subtasks)
                for i, subtask in enumerate(decomposition.subtasks):
                    job.steps.append(
                        JobStep(
                            index=i,
                            title=subtask.title,
                            description=subtask.description,
                        )
                    )
                await self._notify("autonomous_decomposed", job)
                await self._execute_multi_step(job, decomposition)
            else:
                # Simple task — single-step execution
                job.total_steps = 1
                job.steps.append(JobStep(index=0, title="Execute", description=job.goal))
                await self._execute_single_step(job)

            # Stage 3: Summarize results
            successes = sum(1 for s in job.steps if s.status == "success")
            if successes == job.total_steps:
                job.status = JobStatus.SUCCESS
                job.result_summary = f"All {job.total_steps} steps completed successfully."
            elif successes > 0:
                job.status = JobStatus.PARTIAL
                job.result_summary = f"{successes}/{job.total_steps} steps succeeded."
            else:
                job.status = JobStatus.FAILED
                job.result_summary = "All steps failed."

        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            job.result_summary = "Job was cancelled."
        except Exception as e:
            job.status = JobStatus.FAILED
            job.result_summary = f"Job failed: {e}"
            logger.error("Autonomous job [%s] failed: %s", job.job_id, e)

        job.completed_at = time.time()
        await self._notify("autonomous_complete", job)

        # Announce via TTS
        try:
            from pilot.system.voice import speak

            if job.status == JobStatus.SUCCESS:
                await speak(f"Task complete. {job.result_summary}")
            elif job.status == JobStatus.PARTIAL:
                await speak(f"Task partially complete. {job.result_summary}")
            else:
                await speak(f"Task failed. {job.result_summary}")
        except Exception:
            pass

        # Cleanup
        self._active_tasks.pop(job.job_id, None)

    async def _execute_single_step(self, job: AutonomousJob) -> None:
        """Execute a simple (non-decomposed) task."""
        step = job.steps[0]
        job.current_step = 0
        job.status = JobStatus.RUNNING
        step.status = "running"
        step.started_at = time.time()
        await self._notify("autonomous_step_start", job)

        try:
            # Get screen context
            screen_ctx = ""
            if self._screen_vision:
                try:
                    screen_ctx = self._screen_vision.get_context_for_planner()
                except Exception:
                    pass

            # Plan
            plan = await self._planner.plan(job.goal, screen_context=screen_ctx)
            if plan.error:
                step.status = "failed"
                step.error = plan.error
                step.completed_at = time.time()
                return

            # Execute
            results = await self._executor.execute(
                plan,
                invocation_source=InvocationSource.AUTONOMOUS,
                scope_override=job.scope_override,
            )

            # Collect output
            outputs = [r.output for r in results if r.output]
            step.output = "\n".join(outputs)
            step.status = "success" if all(r.success for r in results) else "failed"
            if not all(r.success for r in results):
                step.error = "; ".join(r.error for r in results if r.error)

        except Exception as e:
            step.status = "failed"
            step.error = str(e)

        step.completed_at = time.time()
        await self._notify("autonomous_step_complete", job)

    async def _execute_multi_step(self, job: AutonomousJob, decomposition: Any) -> None:
        """Execute a decomposed multi-step task sequentially."""
        from pilot.agents.decomposer import SubtaskStatus

        batches = self._decomposer.get_execution_order(decomposition)
        job.status = JobStatus.RUNNING

        step_idx = 0
        for batch in batches:
            for subtask in batch:
                if step_idx >= len(job.steps):
                    break

                step = job.steps[step_idx]
                job.current_step = step_idx
                step.status = "running"
                step.started_at = time.time()
                await self._notify("autonomous_step_start", job)

                try:
                    # Get screen context
                    screen_ctx = ""
                    if self._screen_vision:
                        try:
                            screen_ctx = self._screen_vision.get_context_for_planner()
                        except Exception:
                            pass

                    # Plan the subtask
                    plan = await self._planner.plan(
                        subtask.description,
                        screen_context=screen_ctx,
                    )
                    if plan.error:
                        step.status = "failed"
                        step.error = plan.error
                        subtask.status = SubtaskStatus.FAILED
                        step.completed_at = time.time()
                        await self._notify("autonomous_step_complete", job)
                        step_idx += 1
                        continue

                    # Execute
                    results = await self._executor.execute(
                        plan,
                        invocation_source=InvocationSource.AUTONOMOUS,
                        scope_override=job.scope_override,
                    )

                    # Collect
                    outputs = [r.output for r in results if r.output]
                    step.output = "\n".join(outputs)

                    if all(r.success for r in results):
                        step.status = "success"
                        subtask.status = SubtaskStatus.SUCCESS
                        subtask.output = step.output
                    else:
                        step.status = "failed"
                        step.error = "; ".join(r.error for r in results if r.error)
                        subtask.status = SubtaskStatus.FAILED
                        subtask.error = step.error

                except Exception as e:
                    step.status = "failed"
                    step.error = str(e)
                    subtask.status = SubtaskStatus.FAILED

                step.completed_at = time.time()
                await self._notify("autonomous_step_complete", job)
                step_idx += 1

    async def _notify(self, event: str, job: AutonomousJob) -> None:
        """Send a progress notification."""
        if self._broadcast:
            try:
                await self._broadcast(event, job.to_dict())
            except Exception:
                pass
