from __future__ import annotations

import asyncio
import time
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from pilot.actions import Action, ActionResult, ActionType, EmptyParams
from pilot.agents.destructive_critic import PlanRiskAssessment
from pilot.config import PilotConfig
from pilot.neural.audit import NeuralAuditStore
from pilot.neural.controller import NeuralControlError, NeuralController
from pilot.neural.gate import NeuralIntentGate, NeuralIntentGateConfig, NeuralIntentSigner
from pilot.neural.goals import NeuralGoalDefinition, NeuralGoalError, NeuralGoalRegistry
from pilot.neural.protocol import (
    NeuralEvidenceKind,
    NeuralIntentClass,
    NeuralIntentV1,
    NeuralParadigm,
    NeuralScope,
    NeuralStimulusEvent,
    NeuralStreamDescriptorV1,
    NeuralTransport,
    SignalQuality,
)
from pilot.security.gateway import InvocationSource

CALIBRATION = "b" * 64
SUBJECT = "local-subject"


def _safe_assessment(*, requires_confirmation: bool = False) -> PlanRiskAssessment:
    score = 0.5 if requires_confirmation else 0.0
    return PlanRiskAssessment(
        heuristic_score=0.0,
        world_model_score=score,
        combined_score=score,
        reasons=["test warning"] if requires_confirmation else [],
        prediction_sources=["rules"],
        weights_loaded=False,
        model_version="rule-fallback-v1",
    )


async def _controller(
    scope: NeuralScope,
    *,
    audit_store: NeuralAuditStore | None = None,
    fusion_snapshot=None,
    task_dispatcher=None,
) -> tuple[NeuralController, NeuralIntentSigner, uuid.UUID, AsyncMock]:
    signer = NeuralIntentSigner(b"k" * 32)
    session_id = uuid.uuid4()
    goals = NeuralGoalRegistry()
    gate = NeuralIntentGate(
        signer=signer,
        safe_goals={command_id: command_id for command_id in goals.command_ids},
        config=NeuralIntentGateConfig(cancellation_window_ns=1, cooldown_ns=1),
    )
    executor = AsyncMock()
    controller = NeuralController(
        config=PilotConfig(),
        gate=gate,
        executor=executor,
        goals=goals,
        audit_store=audit_store,
        fusion_snapshot=fusion_snapshot,
        task_dispatcher=task_dispatcher,
    )
    descriptor = NeuralStreamDescriptorV1(
        session_id=session_id,
        source_id="synthetic-test",
        board_kind="synthetic",
        transport=NeuralTransport.SYNTHETIC,
        evidence_kind=NeuralEvidenceKind.SYNTHETIC,
        sample_rate_hz=250,
        channel_count=3,
        channel_names=("O1", "Oz", "O2"),
        reference="synthetic-common-reference",
        sequence_start=0,
        started_monotonic_ns=time.monotonic_ns(),
    )
    await controller.connect(descriptor)
    await controller.begin_calibration(session_id)
    await controller.finish_calibration(session_id, calibration_id=CALIBRATION, subject_key=SUBJECT)
    await controller.arm(session_id, scope=scope, non_neural_authorized=True)
    return controller, signer, session_id, executor


async def _intent(
    controller: NeuralController,
    signer: NeuralIntentSigner,
    session_id: uuid.UUID,
    *,
    intent_class: NeuralIntentClass,
    scope: NeuralScope,
    command_id: str | None = None,
) -> NeuralIntentV1:
    status = await controller.status()
    now = time.monotonic_ns()
    unsigned = NeuralIntentV1(
        session_id=session_id,
        intent_id=uuid.uuid4(),
        sequence=int(status["last_sequence"]) + 1,
        window_start_ns=now - 300_000_000,
        window_end_ns=now - 100_000_000,
        expires_at_ns=now + 2_000_000_000,
        paradigm=NeuralParadigm.SYNTHETIC,
        intent_class=intent_class,
        command_id=command_id,
        posterior_permille=950,
        margin_permille=400,
        signal_quality=SignalQuality.GOOD,
        dwell_windows=4,
        decoder_version="a" * 64,
        calibration_id=CALIBRATION,
        subject_key=SUBJECT,
        requested_scope=scope,
        state_revision=int(status["state_revision"]),
        signature="0" * 64,
    )
    return unsigned.model_copy(update={"signature": signer.sign(unsigned)})


def test_goal_registry_rejects_tier_two_or_dynamic_authority() -> None:
    with pytest.raises(NeuralGoalError, match="Tier 0/1"):
        NeuralGoalDefinition(
            "click",
            "Click",
            "Unsafe dynamic click",
            Action(action_type=ActionType.BROWSER_CLICK, target="anything", parameters=EmptyParams()),
        )
    registry = NeuralGoalRegistry()
    assert "system-overview" in registry.command_ids
    assert registry.resolve("system-overview").plan().actions[0].action_type == ActionType.SYSTEM_INFO


@pytest.mark.asyncio
async def test_navigation_commit_never_reaches_executor() -> None:
    controller, signer, session_id, executor = await _controller(NeuralScope.NAVIGATE)
    intent = await _intent(
        controller,
        signer,
        session_id,
        intent_class=NeuralIntentClass.SELECT,
        scope=NeuralScope.NAVIGATE,
    )
    preview = await controller.preview(intent)
    await asyncio.sleep(0.02)
    result = await controller.commit(
        uuid.UUID(str(preview["preview_id"])),
        expected_revision=int(preview["state_revision"]),
        world_model_approved=False,
    )
    assert result["canonical_goal"] == "neural_ui.select"
    executor.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_safe_desktop_goal_rechecks_world_model_and_uses_neural_gateway() -> None:
    controller, signer, session_id, executor = await _controller(NeuralScope.SAFE_DESKTOP)
    intent = await _intent(
        controller,
        signer,
        session_id,
        intent_class=NeuralIntentClass.SAFE_GOAL,
        scope=NeuralScope.SAFE_DESKTOP,
        command_id="system-overview",
    )
    executor.execute.return_value = [
        ActionResult(
            action=NeuralGoalRegistry().resolve("system-overview").plan().actions[0],
            success=True,
            output="system data",
        )
    ]
    with patch("pilot.neural.controller.assess_plan_risk", return_value=_safe_assessment()) as assess:
        preview = await controller.preview(intent)
        await asyncio.sleep(0.02)
        result = await controller.commit(
            uuid.UUID(str(preview["preview_id"])),
            expected_revision=int(preview["state_revision"]),
            world_model_approved=False,
        )
    assert assess.call_count == 2
    assert result["status"] == "completed"
    kwargs = executor.execute.await_args.kwargs
    assert kwargs["invocation_source"] == InvocationSource.NEURAL
    assert kwargs["critic_already_reviewed"] is False
    assert kwargs["user_confirmed"] is False


@pytest.mark.asyncio
async def test_safe_desktop_select_resolves_only_the_daemon_owned_focused_goal() -> None:
    controller, signer, session_id, executor = await _controller(NeuralScope.SAFE_DESKTOP)
    focused = str((await controller.status())["focused_command_id"])
    intent = await _intent(
        controller,
        signer,
        session_id,
        intent_class=NeuralIntentClass.SELECT,
        scope=NeuralScope.SAFE_DESKTOP,
    )
    executor.execute.return_value = [
        ActionResult(
            action=NeuralGoalRegistry().resolve(focused).plan().actions[0],
            success=True,
            output="safe result",
        )
    ]
    with patch("pilot.neural.controller.assess_plan_risk", return_value=_safe_assessment()):
        preview = await controller.preview(intent)
        assert preview["resolved_command_id"] == focused
        await asyncio.sleep(0.02)
        result = await controller.commit(
            uuid.UUID(str(preview["preview_id"])),
            expected_revision=int(preview["state_revision"]),
            world_model_approved=False,
        )
    assert result["command_id"] == focused
    assert executor.execute.await_count == 1


@pytest.mark.asyncio
async def test_explicitly_staged_task_launches_autonomous_pipeline_from_neural_select() -> None:
    dispatcher = AsyncMock(return_value={"job_id": "job-1", "status": "pending", "total_steps": 0})
    controller, signer, session_id, executor = await _controller(
        NeuralScope.SAFE_DESKTOP,
        task_dispatcher=dispatcher,
    )
    staged = await controller.stage_task(
        label="Research and report",
        goal="Research the topic, compare the evidence, and save a verified report.",
        session_id="chat-42",
    )
    task = staged["staged_task"]
    assert staged["focused_command_id"] == task["command_id"]
    assert staged["capabilities"]["free_form_thought_decoding"] is False

    intent = await _intent(
        controller,
        signer,
        session_id,
        intent_class=NeuralIntentClass.SELECT,
        scope=NeuralScope.SAFE_DESKTOP,
    )
    preview = await controller.preview(intent)
    assert preview["resolved_command_id"] == task["command_id"]
    assert preview["staged_task"]["goal"] == task["goal"]
    assert preview["requires_non_neural_approval"] is False

    await asyncio.sleep(0.02)
    result = await controller.commit(
        uuid.UUID(str(preview["preview_id"])),
        expected_revision=int(preview["state_revision"]),
        world_model_approved=False,
    )
    assert result["status"] == "submitted"
    assert result["job"]["job_id"] == "job-1"
    dispatched_task, scope_override = dispatcher.await_args.args
    assert dispatched_task.goal == task["goal"]
    assert dispatched_task.session_id == "chat-42"
    assert scope_override.allow_root is False
    assert "power_shutdown" in scope_override.deny_action_types
    assert (await controller.status())["staged_tasks"] == []
    executor.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_staged_task_dispatch_is_visible_and_kept_for_retry() -> None:
    dispatcher = AsyncMock(side_effect=RuntimeError("autonomous queue unavailable"))
    controller, signer, session_id, executor = await _controller(
        NeuralScope.SAFE_DESKTOP,
        task_dispatcher=dispatcher,
    )
    staged = await controller.stage_task(
        label="Research and report",
        goal="Research the topic and save a verified report.",
        session_id="chat-42",
    )
    intent = await _intent(
        controller,
        signer,
        session_id,
        intent_class=NeuralIntentClass.SELECT,
        scope=NeuralScope.SAFE_DESKTOP,
    )
    preview = await controller.preview(intent)
    await asyncio.sleep(0.02)

    result = await controller.commit(
        uuid.UUID(str(preview["preview_id"])),
        expected_revision=int(preview["state_revision"]),
        world_model_approved=False,
    )

    assert result["status"] == "failed"
    assert result["retry_allowed"] is True
    assert result["error"] == "autonomous queue unavailable"
    assert [task["task_id"] for task in (await controller.status())["staged_tasks"]] == [
        staged["staged_task"]["task_id"]
    ]
    executor.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_staged_task_queue_is_bounded() -> None:
    controller, _, _, _ = await _controller(NeuralScope.SAFE_DESKTOP)
    for index in range(8):
        await controller.stage_task(
            label=f"Task {index + 1}",
            goal=f"Inspect and verify bounded task number {index + 1}.",
            session_id="chat-bounded",
        )

    with pytest.raises(NeuralControlError, match="at most 8"):
        await controller.stage_task(
            label="Task 9",
            goal="This task must not enter the bounded neural queue.",
            session_id="chat-bounded",
        )
    assert len((await controller.status())["staged_tasks"]) == 8


@pytest.mark.asyncio
async def test_world_model_warning_requires_ui_approval_and_unavailable_model_disarms() -> None:
    controller, signer, session_id, executor = await _controller(NeuralScope.SAFE_DESKTOP)
    intent = await _intent(
        controller,
        signer,
        session_id,
        intent_class=NeuralIntentClass.SAFE_GOAL,
        scope=NeuralScope.SAFE_DESKTOP,
        command_id="battery-status",
    )
    with patch(
        "pilot.neural.controller.assess_plan_risk",
        return_value=_safe_assessment(requires_confirmation=True),
    ):
        preview = await controller.preview(intent)
        with pytest.raises(NeuralControlError, match="non-neural approval"):
            await controller.commit(
                uuid.UUID(str(preview["preview_id"])),
                expected_revision=int(preview["state_revision"]),
                world_model_approved=False,
            )
    executor.execute.assert_not_awaited()

    controller2, signer2, session2, _ = await _controller(NeuralScope.SAFE_DESKTOP)
    intent2 = await _intent(
        controller2,
        signer2,
        session2,
        intent_class=NeuralIntentClass.SAFE_GOAL,
        scope=NeuralScope.SAFE_DESKTOP,
        command_id="battery-status",
    )
    unavailable = _safe_assessment().to_dict()
    unavailable["model_version"] = "evaluation-error"
    with (
        patch(
            "pilot.neural.controller.assess_plan_risk",
            return_value=PlanRiskAssessment(
                **{key: value for key, value in unavailable.items() if key != "requires_confirmation"}
            ),
        ),
        pytest.raises(NeuralControlError, match="unavailable"),
    ):
        await controller2.preview(intent2)
    assert (await controller2.status())["armed_scope"] == "observe"


@pytest.mark.asyncio
async def test_controller_audit_links_accepted_window_to_executed_plan(tmp_path) -> None:
    audit = NeuralAuditStore(tmp_path / "neural.db", tmp_path / "neural.key")
    controller, signer, session_id, executor = await _controller(
        NeuralScope.SAFE_DESKTOP,
        audit_store=audit,
    )
    intent = await _intent(
        controller,
        signer,
        session_id,
        intent_class=NeuralIntentClass.SAFE_GOAL,
        scope=NeuralScope.SAFE_DESKTOP,
        command_id="system-overview",
    )
    executor.execute.return_value = [
        ActionResult(
            action=NeuralGoalRegistry().resolve("system-overview").plan().actions[0],
            success=True,
            output="system data",
        )
    ]
    with patch("pilot.neural.controller.assess_plan_risk", return_value=_safe_assessment()):
        preview = await controller.preview(intent)
        await asyncio.sleep(0.02)
        result = await controller.commit(
            uuid.UUID(str(preview["preview_id"])),
            expected_revision=int(preview["state_revision"]),
            world_model_approved=False,
        )

    events = list(reversed(await audit.list_events(intent_id=str(intent.intent_id))))
    assert [event["stage"] for event in events] == [
        "intent_accepted",
        "preview_created",
        "commit_authorized",
        "result",
    ]
    assert events[-1]["preview_id"] == preview["preview_id"]
    assert events[-1]["plan_id"] == result["plan_id"]
    assert events[0]["window_start_ns"] == intent.window_start_ns
    assert (await audit.verify_chain()).valid is True


@pytest.mark.asyncio
async def test_multimodal_context_is_visible_but_cannot_expand_neural_authority() -> None:
    fusion = AsyncMock(
        return_value={
            "modalities": ["voice", "gesture", "gaze"],
            "cancellation_present": False,
            "voice": {"transcript": "open something else"},
        }
    )
    controller, signer, session_id, executor = await _controller(
        NeuralScope.NAVIGATE,
        fusion_snapshot=fusion,
    )
    intent = await _intent(
        controller,
        signer,
        session_id,
        intent_class=NeuralIntentClass.SELECT,
        scope=NeuralScope.NAVIGATE,
    )
    preview = await controller.preview(intent)
    assert preview["fusion"] == {
        "modalities": ["voice", "gesture", "gaze"],
        "cancellation_present": False,
        "raw_media_excluded": True,
    }
    assert "voice" not in preview["fusion"]
    executor.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_simultaneous_cancel_after_preview_disarms_before_commit() -> None:
    fusion = AsyncMock(
        side_effect=[
            {"modalities": ["gaze"], "cancellation_present": False},
            {"modalities": ["voice", "gaze"], "cancellation_present": True},
        ]
    )
    controller, signer, session_id, executor = await _controller(
        NeuralScope.NAVIGATE,
        fusion_snapshot=fusion,
    )
    intent = await _intent(
        controller,
        signer,
        session_id,
        intent_class=NeuralIntentClass.SELECT,
        scope=NeuralScope.NAVIGATE,
    )
    preview = await controller.preview(intent)
    await asyncio.sleep(0.02)
    with pytest.raises(NeuralControlError, match="cancellation disarmed"):
        await controller.commit(
            uuid.UUID(str(preview["preview_id"])),
            expected_revision=int(preview["state_revision"]),
            world_model_approved=False,
        )
    assert (await controller.status())["armed_scope"] == "observe"
    executor.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_stimulus_markers_are_daemon_stamped_bounded_and_audited(tmp_path) -> None:
    audit = NeuralAuditStore(tmp_path / "neural.db", tmp_path / "neural.key")
    controller, _, session_id, _ = await _controller(NeuralScope.NAVIGATE, audit_store=audit)
    marker = await controller.record_stimulus_marker(
        session_id,
        target_id="focus_left",
        event=NeuralStimulusEvent.TARGET_ON,
        client_performance_ms=25.5,
    )
    assert marker["sequence"] == 0
    assert marker["received_monotonic_ns"] > 0
    assert await controller.stimulus_markers(after_sequence=-1) == (marker,)
    assert await controller.stimulus_markers(after_sequence=0) == ()
    events = await audit.list_events()
    assert events[0]["stage"] == "stimulus_marker"
    with pytest.raises(NeuralControlError, match="registered SSVEP"):
        await controller.record_stimulus_marker(
            session_id,
            target_id="dynamic-command",
            event=NeuralStimulusEvent.TARGET_ON,
            client_performance_ms=26.0,
        )


@pytest.mark.asyncio
async def test_gate_state_reports_disconnected_armed_and_preview_without_verifying_the_audit_chain(tmp_path) -> None:
    signer = NeuralIntentSigner(b"k" * 32)
    goals = NeuralGoalRegistry()
    idle = NeuralController(
        config=PilotConfig(),
        gate=NeuralIntentGate(signer=signer, safe_goals={command_id: command_id for command_id in goals.command_ids}),
        executor=AsyncMock(),
        goals=goals,
        audit_store=NeuralAuditStore(tmp_path / "idle.db", tmp_path / "idle.key"),
    )
    audit = NeuralAuditStore(tmp_path / "neural.db", tmp_path / "neural.key")
    controller, signer, session_id, _ = await _controller(NeuralScope.NAVIGATE, audit_store=audit)
    intent = await _intent(
        controller,
        signer,
        session_id,
        intent_class=NeuralIntentClass.SELECT,
        scope=NeuralScope.NAVIGATE,
    )

    with patch.object(
        NeuralAuditStore, "verify_chain", new=AsyncMock(side_effect=AssertionError("audit chain verified"))
    ) as verify_chain:
        disconnected = await idle.gate_state()
        armed = await controller.gate_state()
        await controller.preview(intent)
        previewed = await controller.gate_state()

    assert disconnected == "disconnected"
    assert armed == "armed_safe_ui"
    assert previewed == "previewed"
    verify_chain.assert_not_awaited()
    assert all(isinstance(state, str) for state in (disconnected, armed, previewed))
