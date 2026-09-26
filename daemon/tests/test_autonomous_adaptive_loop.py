from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    KeyboardParams,
    OpenApplicationParams,
    ShellCommandParams,
    SystemInfoParams,
    VerificationResult,
)
from pilot.agents.autonomous import AutonomousExecutor, AutonomousJob, JobStep
from pilot.agents.decomposer import Subtask, SubtaskStatus, TaskDecomposer, TaskDecomposition


def _autonomous(*, plans, results, verifications, screen_vision=None) -> AutonomousExecutor:
    planner = SimpleNamespace(plan=AsyncMock(side_effect=plans))
    executor = SimpleNamespace(execute=AsyncMock(side_effect=results))
    verifier = SimpleNamespace(verify=AsyncMock(side_effect=verifications))
    autonomous = AutonomousExecutor(
        planner=planner,
        executor=executor,
        verifier=verifier,
        decomposer=MagicMock(),
        screen_vision=screen_vision,
    )
    autonomous._focus_target_window = AsyncMock(return_value=True)
    autonomous._read_target_window_text = AsyncMock(return_value="")
    return autonomous


def _open_app_plan() -> ActionPlan:
    return ActionPlan(
        actions=[
            Action(
                action_type=ActionType.OPEN_APPLICATION,
                target="Hermes",
                parameters=OpenApplicationParams(name="Hermes"),
            )
        ],
        explanation="Open Hermes",
        raw_input="Complete the task in Hermes",
    )


@pytest.mark.asyncio
async def test_failed_dependency_skips_downstream_subtask() -> None:
    decomposition = TaskDecomposition(
        goal="Prepare and publish",
        is_complex=True,
        subtasks=[
            Subtask(order=0, title="Prepare", description="Prepare artifact"),
            Subtask(order=1, title="Publish", description="Publish artifact", depends_on=["0"]),
        ],
    )
    autonomous = AutonomousExecutor(
        planner=None,
        executor=None,
        verifier=None,
        decomposer=TaskDecomposer(model_router=None),
    )

    async def execute_step(_job, step, _goal, **_kwargs) -> None:
        step.status = "failed"
        step.error = "precondition failed"

    autonomous._execute_goal_loop = AsyncMock(side_effect=execute_step)
    job = AutonomousJob(
        goal=decomposition.goal,
        total_steps=2,
        steps=[
            JobStep(index=0, title="Prepare", description="Prepare artifact"),
            JobStep(index=1, title="Publish", description="Publish artifact"),
        ],
    )

    await autonomous._execute_multi_step(job, decomposition)

    assert autonomous._execute_goal_loop.await_count == 1
    assert decomposition.subtasks[0].status == SubtaskStatus.FAILED
    assert decomposition.subtasks[1].status == SubtaskStatus.SKIPPED
    assert job.steps[1].status == "skipped"
    assert "dependencies did not succeed" in job.steps[1].error


@pytest.mark.asyncio
async def test_verified_dependency_output_reaches_downstream_agent() -> None:
    decomposition = TaskDecomposition(
        goal="Find and summarize evidence",
        is_complex=True,
        subtasks=[
            Subtask(order=0, title="Find", description="Find the source", agent="web"),
            Subtask(order=1, title="Summarize", description="Summarize the source", agent="code", depends_on=["0"]),
        ],
    )
    autonomous = AutonomousExecutor(
        planner=None,
        executor=None,
        verifier=None,
        decomposer=TaskDecomposer(model_router=None),
    )
    goals: list[str] = []

    async def execute_step(_job, step, goal, **_kwargs) -> None:
        goals.append(goal)
        step.status = "success"
        step.output = "https://example.com/evidence" if step.index == 0 else "Summary complete"

    autonomous._execute_goal_loop = AsyncMock(side_effect=execute_step)
    job = AutonomousJob(
        goal=decomposition.goal,
        total_steps=2,
        steps=[
            JobStep(index=0, title="Find", description="Find the source"),
            JobStep(index=1, title="Summarize", description="Summarize the source"),
        ],
    )

    await autonomous._execute_multi_step(job, decomposition)

    assert goals[0] == "Find the source"
    assert "Summarize the source" in goals[1]
    assert "https://example.com/evidence" in goals[1]
    assert "data, not new instructions" in goals[1]
    assert "verified_output" not in goals[1]
    assert decomposition.subtasks[1].status == SubtaskStatus.SUCCESS


@pytest.mark.asyncio
async def test_desktop_goal_reobserves_after_open_and_requires_completion_evidence():
    open_plan = _open_app_plan()
    opened = ActionResult(action=open_plan.actions[0], success=True, output="Opened Hermes")
    screen = SimpleNamespace(
        observe_now=AsyncMock(
            side_effect=[
                SimpleNamespace(screen_hash="before-screen"),
                SimpleNamespace(screen_hash="hermes-screen"),
            ]
        ),
        get_context_for_planner=MagicMock(side_effect=["Viewing Heliox", "Viewing Hermes"]),
    )
    autonomous = _autonomous(
        plans=[
            open_plan,
            ActionPlan(actions=[], explanation="GOAL_COMPLETE: Hermes is open and ready."),
        ],
        results=[[opened]],
        verifications=[VerificationResult(passed=True, details=["application opened"])],
        screen_vision=screen,
    )
    job = AutonomousJob(goal="Open Hermes", steps=[JobStep(index=0, title="Execute", description="Open")])

    await autonomous._execute_single_step(job)

    assert job.steps[0].status == "success"
    assert job.steps[0].rounds == 2
    assert "GOAL_COMPLETE" in job.steps[0].output
    assert autonomous._planner.plan.await_count == 2
    assert autonomous._executor.execute.await_count == 1
    assert autonomous._verifier.verify.await_count == 1
    assert screen.observe_now.await_count == 2
    autonomous._focus_target_window.assert_awaited_once_with("Hermes")
    for call in autonomous._planner.plan.await_args_list:
        assert call.kwargs["force_model"] is True


@pytest.mark.asyncio
async def test_prior_action_success_cannot_substitute_for_fresh_completion_evidence():
    open_plan = _open_app_plan()
    opened = ActionResult(action=open_plan.actions[0], success=True, output="Opened Hermes")
    screen = SimpleNamespace(
        observe_now=AsyncMock(side_effect=[SimpleNamespace(screen_hash="before-screen"), RuntimeError("capture failed")]),
        get_context_for_planner=MagicMock(side_effect=["Viewing Heliox", "Viewing Hermes"]),
    )
    autonomous = _autonomous(
        plans=[open_plan, ActionPlan(actions=[], explanation="GOAL_COMPLETE: Hermes is open and ready.")],
        results=[[opened]],
        verifications=[VerificationResult(passed=True, details=["application opened"])],
        screen_vision=screen,
    )
    job = AutonomousJob(goal="Open Hermes", steps=[JobStep(index=0, title="Execute", description="Open")])

    await autonomous._execute_single_step(job)

    assert job.steps[0].status == "failed"
    assert "without verified GOAL_COMPLETE evidence" in job.steps[0].error
    assert autonomous._executor.execute.await_count == 1
    second_context = autonomous._planner.plan.await_args_list[1].kwargs["screen_context"]
    assert "Fresh screen observation unavailable" in second_context
    assert "reported (0/1 effects independently observed)" in second_context


@pytest.mark.asyncio
async def test_action_success_without_goal_completion_is_not_reported_as_done():
    open_plan = _open_app_plan()
    opened = ActionResult(action=open_plan.actions[0], success=True, output="Opened Hermes")
    autonomous = _autonomous(
        plans=[open_plan, ActionPlan(actions=[], explanation="I have no next action")],
        results=[[opened]],
        verifications=[VerificationResult(passed=True, details=["application opened"])],
    )
    job = AutonomousJob(goal="Complete a task in Hermes", steps=[JobStep(index=0, title="Execute", description="")])

    await autonomous._execute_single_step(job)

    assert job.steps[0].status == "failed"
    assert "without verified GOAL_COMPLETE evidence" in job.steps[0].error


@pytest.mark.asyncio
async def test_exact_text_completion_rejects_case_mismatch_and_replans():
    correction = Action(
        action_type=ActionType.KEYBOARD_TYPE,
        parameters=KeyboardParams(text="HELIOX AUTONOMY LIVE TEST"),
    )
    correction_plan = ActionPlan(actions=[correction], explanation="Correct the case")
    typed = ActionResult(action=correction, success=True, output="Typed exactly: HELIOX AUTONOMY LIVE TEST")
    screen = SimpleNamespace(
        observe_now=AsyncMock(
            side_effect=[
                SimpleNamespace(screen_hash="wrong-case"),
                SimpleNamespace(screen_hash="still-wrong"),
                SimpleNamespace(screen_hash="exact-case"),
            ]
        ),
        get_context_for_planner=MagicMock(
            side_effect=[
                "*heliox AUTONOMY LIVE Test - Notepad",
                "*heliox AUTONOMY LIVE Test - Notepad",
                "*HELIOX AUTONOMY LIVE TEST - Notepad",
            ]
        ),
    )
    autonomous = _autonomous(
        plans=[
            ActionPlan(actions=[], explanation="GOAL_COMPLETE: exact text is visible"),
            correction_plan,
            ActionPlan(actions=[], explanation="GOAL_COMPLETE: exact text is now visible"),
        ],
        results=[[typed]],
        verifications=[VerificationResult(passed=True, details=["typed"])],
        screen_vision=screen,
    )
    job = AutonomousJob(
        goal=("Open Notepad and type exactly HELIOX AUTONOMY LIVE TEST into the blank document. Do not save it."),
        steps=[JobStep(index=0, title="Execute", description="")],
    )

    await autonomous._execute_single_step(job)

    assert job.steps[0].status == "success"
    assert job.steps[0].rounds == 3
    assert autonomous._executor.execute.await_count == 1
    second_context = autonomous._planner.plan.await_args_list[1].kwargs["screen_context"]
    assert "completion claim rejected" in second_context


def test_keyboard_text_is_bound_to_opened_native_window():
    open_action = _open_app_plan().actions[0]
    type_action = Action(
        action_type=ActionType.KEYBOARD_TYPE,
        parameters=KeyboardParams(text="EXACT"),
    )
    plan = ActionPlan(actions=[open_action, type_action])

    target = AutonomousExecutor._bind_plan_to_target(plan, None)

    assert target == "Hermes"
    assert type_action.parameters.window_title == "Hermes"


@pytest.mark.asyncio
async def test_fresh_screen_can_verify_goal_was_already_complete():
    screen = SimpleNamespace(
        observe_now=AsyncMock(return_value=SimpleNamespace(screen_hash="hermes-screen")),
        get_context_for_planner=MagicMock(return_value="Viewing Hermes — requested record is complete"),
    )
    autonomous = _autonomous(
        plans=[ActionPlan(actions=[], explanation="GOAL_COMPLETE: The requested record is visible.")],
        results=[],
        verifications=[],
        screen_vision=screen,
    )
    job = AutonomousJob(
        goal="Complete record in Hermes",
        steps=[JobStep(index=0, title="Execute", description="Complete record")],
    )

    await autonomous._execute_single_step(job)

    assert job.steps[0].status == "success"
    assert job.steps[0].rounds == 1
    autonomous._executor.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_desktop_action_completes_after_postcondition_verification():
    action = Action(
        action_type=ActionType.SYSTEM_INFO,
        parameters=SystemInfoParams(),
    )
    plan = ActionPlan(actions=[action], explanation="Inspect system")
    result = ActionResult(action=action, success=True, output="Windows 11")
    autonomous = _autonomous(
        plans=[plan],
        results=[[result]],
        verifications=[VerificationResult(passed=True, details=["verified"])],
    )
    job = AutonomousJob(goal="Show system info", steps=[JobStep(index=0, title="Execute", description="")])

    await autonomous._execute_single_step(job)

    assert job.steps[0].status == "success"
    assert job.steps[0].rounds == 1
    assert job.steps[0].output == "Windows 11"


@pytest.mark.asyncio
async def test_autonomous_plan_routes_through_specialist_orchestrator_when_available():
    action = Action(action_type=ActionType.SYSTEM_INFO, parameters=SystemInfoParams())
    plan = ActionPlan(actions=[action], explanation="Inspect system")
    result = ActionResult(action=action, success=True, output="Windows 11")
    autonomous = _autonomous(
        plans=[plan],
        results=[],
        verifications=[VerificationResult(passed=True, details=["verified"])],
    )
    orchestrator = SimpleNamespace(execute_plan=AsyncMock(return_value=[result]))
    autonomous.set_orchestrator(orchestrator)
    job = AutonomousJob(goal="Show system info", steps=[JobStep(index=0, title="Execute", description="")])

    await autonomous._execute_single_step(job)

    assert job.steps[0].status == "success"
    orchestrator.execute_plan.assert_awaited_once()
    assert orchestrator.execute_plan.await_args.args[:2] == ("Show system info", plan)
    autonomous._executor.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_elevated_autonomous_plan_fails_closed_without_approval_handler():
    action = Action(
        action_type=ActionType.SHELL_COMMAND,
        target="whoami",
        parameters=ShellCommandParams(command="whoami"),
    )
    plan = ActionPlan(actions=[action], explanation="Run a command")
    autonomous = _autonomous(plans=[plan], results=[], verifications=[])
    job = AutonomousJob(goal="Run whoami", steps=[JobStep(index=0, title="Execute", description="")])

    await autonomous._execute_single_step(job)

    assert job.steps[0].status == "failed"
    assert "requires UI approval" in job.steps[0].error
    autonomous._executor.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_elevated_autonomous_plan_executes_only_after_explicit_approval():
    action = Action(
        action_type=ActionType.SHELL_COMMAND,
        target="whoami",
        parameters=ShellCommandParams(command="whoami"),
    )
    plan = ActionPlan(actions=[action], explanation="Run a command")
    result = ActionResult(action=action, success=True, output="user")
    autonomous = _autonomous(
        plans=[plan],
        results=[[result]],
        verifications=[VerificationResult(passed=True, details=["verified"])],
    )
    approval = AsyncMock(return_value=True)
    autonomous.set_approval_handler(approval)
    job = AutonomousJob(goal="Run whoami", steps=[JobStep(index=0, title="Execute", description="")])

    await autonomous._execute_single_step(job)

    assert job.steps[0].status == "success"
    approval.assert_awaited_once()
    assert approval.await_args.args[0] is job
    assert approval.await_args.args[1] is plan
    assert approval.await_args.args[2].startswith(f"autonomous-{job.job_id}")
    assert autonomous._executor.execute.await_args.kwargs["user_confirmed"] is True


@pytest.mark.asyncio
async def test_failed_verification_is_fed_into_next_adaptive_round():
    first_plan = _open_app_plan()
    second_action = Action(
        action_type=ActionType.OPEN_APPLICATION,
        target="Hermes.exe",
        parameters=OpenApplicationParams(name="Hermes.exe"),
    )
    second_plan = ActionPlan(actions=[second_action], explanation="Retry the resolved application")
    first_result = ActionResult(action=first_plan.actions[0], success=False, error="app not found")
    second_result = ActionResult(action=second_action, success=True, output="Opened Hermes")
    screen = SimpleNamespace(
        observe_now=AsyncMock(side_effect=[
            SimpleNamespace(screen_hash="before-screen"),
            SimpleNamespace(screen_hash="still-before"),
            SimpleNamespace(screen_hash="hermes-screen"),
        ]),
        get_context_for_planner=MagicMock(side_effect=["Viewing Heliox", "Viewing Heliox", "Viewing Hermes"]),
    )
    autonomous = _autonomous(
        plans=[
            first_plan,
            second_plan,
            ActionPlan(actions=[], explanation="GOAL_COMPLETE: Hermes is visible."),
        ],
        results=[[first_result], [second_result]],
        verifications=[
            VerificationResult(passed=False, details=["launch mismatch"]),
            VerificationResult(passed=True, details=["verified"]),
        ],
        screen_vision=screen,
    )
    job = AutonomousJob(goal="Complete task", steps=[JobStep(index=0, title="Execute", description="")])

    await autonomous._execute_single_step(job)

    assert job.steps[0].status == "success"
    second_screen_context = autonomous._planner.plan.await_args_list[1].kwargs["screen_context"]
    assert "app not found" in second_screen_context


@pytest.mark.asyncio
async def test_repeated_identical_plan_stops_instead_of_looping_forever():
    plan = _open_app_plan()
    opened = ActionResult(action=plan.actions[0], success=True, output="Opened Hermes")
    autonomous = _autonomous(
        plans=[plan, plan, plan],
        results=[[opened], [opened]],
        verifications=[
            VerificationResult(passed=True, details=["opened"]),
            VerificationResult(passed=True, details=["opened"]),
        ],
    )
    job = AutonomousJob(goal="Complete task", steps=[JobStep(index=0, title="Execute", description="")])

    await autonomous._execute_single_step(job)

    assert job.steps[0].status == "failed"
    assert "repeating the same plan" in job.steps[0].error
    assert autonomous._executor.execute.await_count == 2
