"""Heliox-side neural preview and guarded execution controller."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from uuid import UUID, uuid4

from pilot.actions import ActionPlan, PermissionTier
from pilot.agents.destructive_critic import PlanRiskAssessment, assess_plan_risk
from pilot.config import PilotConfig
from pilot.neural.audit import NeuralAuditStore
from pilot.neural.gate import NeuralCommit, NeuralIntentGate, NeuralPreview
from pilot.neural.goals import NeuralGoalRegistry
from pilot.neural.protocol import (
    NeuralCalibrationMetricsV1,
    NeuralIntentClass,
    NeuralIntentV1,
    NeuralScope,
    NeuralStimulusEvent,
    NeuralStimulusMarkerV1,
    NeuralStreamDescriptorV1,
)
from pilot.neural.quality import SignalQualitySummary
from pilot.security.gateway import (
    DEFAULT_SOURCE_PROFILES,
    InvocationSource,
    TaskScopeOverride,
)


class NeuralControlError(ValueError):
    pass


MAX_STAGED_NEURAL_TASKS = 8
STAGED_TASK_COMMAND_PREFIX = "staged-task:"


@dataclass(frozen=True, slots=True)
class StagedNeuralTask:
    """A non-neurally authored goal that neural input may only select and launch."""

    task_id: UUID
    label: str
    goal: str
    session_id: str
    created_at_ns: int

    @property
    def command_id(self) -> str:
        return f"{STAGED_TASK_COMMAND_PREFIX}{self.task_id}"

    def public_summary(self) -> dict[str, object]:
        return {
            "task_id": str(self.task_id),
            "command_id": self.command_id,
            "label": self.label,
            "goal": self.goal,
            "session_id": self.session_id,
            "created_at_ns": self.created_at_ns,
            "authority": "explicit_non_neural_staging",
        }


@dataclass(frozen=True, slots=True)
class PendingNeuralExecution:
    preview: NeuralPreview
    plan: ActionPlan | None
    world_model: PlanRiskAssessment | None
    resolved_command_id: str | None = None
    fusion: dict[str, object] | None = None
    staged_task: StagedNeuralTask | None = None


class NeuralController:
    """Convert a committed neural preview into UI navigation or one fixed plan."""

    def __init__(
        self,
        *,
        config: PilotConfig,
        gate: NeuralIntentGate,
        executor: Any,
        goals: NeuralGoalRegistry | None = None,
        broadcast: Callable[[str, Any], Awaitable[None]] | None = None,
        audit_store: NeuralAuditStore | None = None,
        fusion_snapshot: Callable[[], Awaitable[dict[str, Any]]] | None = None,
        task_dispatcher: Callable[[StagedNeuralTask, TaskScopeOverride], Awaitable[dict[str, object]]] | None = None,
    ) -> None:
        self._config = config
        self._gate = gate
        self._executor = executor
        self._goals = goals or NeuralGoalRegistry()
        self._broadcast = broadcast
        self._audit_store = audit_store
        self._fusion_snapshot = fusion_snapshot
        self._task_dispatcher = task_dispatcher
        self._staged_tasks: dict[UUID, StagedNeuralTask] = {}
        self._pending: PendingNeuralExecution | None = None
        self._last_observation: dict[str, object] | None = None
        self._focus_index = 0
        self._stimulus_markers: deque[NeuralStimulusMarkerV1] = deque(maxlen=512)
        self._stimulus_sequence = 0
        self._lock = asyncio.Lock()

    def set_task_dispatcher(
        self,
        dispatcher: Callable[[StagedNeuralTask, TaskScopeOverride], Awaitable[dict[str, object]]],
    ) -> None:
        """Connect staged neural selections to the normal autonomous task engine."""

        self._task_dispatcher = dispatcher

    async def stage_task(self, *, label: str, goal: str, session_id: str) -> dict[str, object]:
        """Stage an explicit UI-authored goal for later neural focus/select.

        Neural acquisition never supplies ``goal`` or action parameters. The UI
        must create the task first, making neural input a bounded launcher rather
        than an unverified free-form thought decoder.
        """

        normalized_goal = " ".join(str(goal).split())
        normalized_label = " ".join(str(label).split()) or normalized_goal[:72]
        normalized_session = " ".join(str(session_id).split()) or "neural"
        if not (3 <= len(normalized_goal) <= 2000):
            raise NeuralControlError("staged neural task goal must be 3-2000 characters")
        if not (1 <= len(normalized_label) <= 80):
            raise NeuralControlError("staged neural task label must be 1-80 characters")
        if len(normalized_session) > 128:
            raise NeuralControlError("staged neural task session id is too long")

        async with self._lock:
            if len(self._staged_tasks) >= MAX_STAGED_NEURAL_TASKS:
                raise NeuralControlError(f"at most {MAX_STAGED_NEURAL_TASKS} neural tasks may be staged")
            task = StagedNeuralTask(
                task_id=uuid4(),
                label=normalized_label,
                goal=normalized_goal,
                session_id=normalized_session,
                created_at_ns=time.monotonic_ns(),
            )
            self._staged_tasks[task.task_id] = task
            self._focus_index = len(self._focusable_command_ids()) - 1

        status = await self.status()
        await self._emit("neural_status", status)
        return {**status, "staged_task": task.public_summary()}

    async def remove_staged_task(self, task_id: UUID) -> dict[str, object]:
        async with self._lock:
            if self._pending is not None and self._pending.staged_task is not None:
                if self._pending.staged_task.task_id == task_id:
                    raise NeuralControlError("cannot remove a staged task while its neural preview is active")
            if self._staged_tasks.pop(task_id, None) is None:
                raise NeuralControlError("staged neural task was not found")
            focusable = self._focusable_command_ids()
            self._focus_index = min(self._focus_index, max(0, len(focusable) - 1))

        status = await self.status()
        await self._emit("neural_status", status)
        return status

    async def connect(self, descriptor: NeuralStreamDescriptorV1) -> dict[str, object]:
        async with self._lock:
            self._pending = None
            self._last_observation = None
            status = await self._gate.connect(descriptor)
            await self._emit("neural_status", status)
            return status

    async def begin_calibration(self, session_id: UUID) -> dict[str, object]:
        async with self._lock:
            self._pending = None
            status = await self._gate.begin_calibration(session_id)
            await self._emit("neural_status", status)
            return status

    async def finish_calibration(
        self,
        session_id: UUID,
        *,
        calibration_id: str,
        subject_key: str,
        decoder_version: str = "",
        metrics: NeuralCalibrationMetricsV1 | None = None,
    ) -> dict[str, object]:
        async with self._lock:
            status = await self._gate.finish_calibration(
                session_id,
                calibration_id=calibration_id,
                subject_key=subject_key,
                decoder_version=decoder_version,
                metrics=metrics,
            )
            await self._emit("neural_status", status)
            return status

    async def arm(
        self,
        session_id: UUID,
        *,
        scope: NeuralScope,
        non_neural_authorized: bool,
    ) -> dict[str, object]:
        async with self._lock:
            self._pending = None
            status = await self._gate.arm(
                session_id,
                scope=scope,
                non_neural_authorized=non_neural_authorized,
            )
            await self._emit("neural_status", status)
            return status

    async def preview(self, intent: NeuralIntentV1) -> dict[str, object]:
        async with self._lock:
            fusion = await self._capture_fusion()
            await self._enforce_fusion_safety(fusion, reason="multimodal_cancel_before_preview")
            preview = await self._gate.preview(intent)
            await self._audit(
                stage="intent_accepted",
                intent=intent,
                outcome="cancelled" if preview is None else "accepted",
                metadata={
                    "intent_class": intent.intent_class.value,
                    "requested_scope": intent.requested_scope.value,
                    "signal_quality": intent.signal_quality.value,
                    "dwell_windows": intent.dwell_windows,
                    "posterior_permille": intent.posterior_permille,
                    "margin_permille": intent.margin_permille,
                },
            )
            if preview is None:
                self._pending = None
                status = await self._gate.status()
                await self._emit("neural_disarmed", {**status, "reason": "neural_cancel"})
                return {"status": "cancelled", **status}

            plan: ActionPlan | None = None
            world_model: PlanRiskAssessment | None = None
            staged_task: StagedNeuralTask | None = None
            resolved_command_id = preview.command_id
            try:
                if (
                    preview.intent_class == NeuralIntentClass.SELECT
                    and preview.requested_scope == NeuralScope.SAFE_DESKTOP
                ):
                    resolved_command_id = self._focused_command_id()
                if resolved_command_id:
                    staged_task = self._resolve_staged_task(resolved_command_id)
                    if staged_task is None:
                        plan = self._goals.resolve(resolved_command_id).plan()
                        world_model = self._assess_world_model(plan)
            except Exception:
                self._pending = None
                await self._gate.disarm(reason="preview_safety_failure")
                raise

            self._pending = PendingNeuralExecution(
                preview,
                plan,
                world_model,
                resolved_command_id,
                fusion,
                staged_task,
            )
            payload = self._preview_payload(self._pending)
            await self._audit(
                stage="preview_created",
                intent=intent,
                preview=preview,
                outcome="previewed",
                metadata={
                    "canonical_goal": preview.canonical_goal,
                    "command_id": resolved_command_id,
                    "world_model": world_model.to_dict() if world_model else None,
                    "fusion": self._public_fusion(fusion),
                    "staged_task_id": str(staged_task.task_id) if staged_task else None,
                },
            )
            await self._emit("neural_preview", payload)
            return payload

    async def commit(
        self,
        preview_id: UUID,
        *,
        expected_revision: int,
        world_model_approved: bool,
    ) -> dict[str, object]:
        async with self._lock:
            pending = self._pending
            if pending is None or pending.preview.preview_id != preview_id:
                raise NeuralControlError("preview is missing or no longer current")
            current_fusion = await self._capture_fusion()
            await self._enforce_fusion_safety(
                current_fusion,
                reason="multimodal_cancel_during_preview",
            )
            plan = pending.plan
            staged_task = pending.staged_task
            assessment = pending.world_model
            if plan is not None:
                assessment = self._assess_world_model(plan)
                if assessment.requires_confirmation and not world_model_approved:
                    raise NeuralControlError("world-model warning requires a non-neural approval")
                effect_tier = int(plan.max_tier)
            elif staged_task is not None:
                if self._task_dispatcher is None:
                    raise NeuralControlError("autonomous task engine is unavailable; neural task launch fails closed")
                effect_tier = int(PermissionTier.READ_ONLY)
            else:
                effect_tier = int(PermissionTier.READ_ONLY)

            commit = await self._gate.commit(
                preview_id,
                expected_revision=expected_revision,
                effect_tier=effect_tier,
            )
            self._pending = None
            plan_id = f"neural-{commit.intent_id}" if plan is not None else ""
            await self._audit(
                stage="commit_authorized",
                preview=pending.preview,
                commit=commit,
                plan_id=plan_id,
                outcome="authorized",
                metadata={
                    "canonical_goal": commit.canonical_goal,
                    "command_id": pending.resolved_command_id,
                    "effect_tier": effect_tier,
                    "fusion": self._public_fusion(current_fusion),
                },
            )
            if plan is None:
                if staged_task is not None:
                    autonomous_profile = self._config.gateway.source_profiles.get(
                        "autonomous", DEFAULT_SOURCE_PROFILES["autonomous"]
                    )
                    scope_override = TaskScopeOverride(
                        max_tier=autonomous_profile.max_tier,
                        deny_action_types=autonomous_profile.deny_action_types,
                        allow_root=autonomous_profile.allow_root,
                    )
                    try:
                        job = await self._task_dispatcher(staged_task, scope_override)
                    except Exception as exc:
                        payload = {
                            "status": "failed",
                            "preview_id": str(commit.preview_id),
                            "intent_id": str(commit.intent_id),
                            "canonical_goal": commit.canonical_goal,
                            "command_id": pending.resolved_command_id,
                            "staged_task": staged_task.public_summary(),
                            "error": str(exc),
                            "retry_allowed": True,
                        }
                        await self._audit(
                            stage="result",
                            preview=pending.preview,
                            commit=commit,
                            outcome="failed",
                            metadata={"staged_task_id": str(staged_task.task_id)},
                        )
                        await self._emit("neural_result", payload)
                        return payload

                    self._staged_tasks.pop(staged_task.task_id, None)
                    self._focus_index = min(
                        self._focus_index,
                        max(0, len(self._focusable_command_ids()) - 1),
                    )
                    payload = {
                        "status": "submitted",
                        "preview_id": str(commit.preview_id),
                        "intent_id": str(commit.intent_id),
                        "canonical_goal": commit.canonical_goal,
                        "command_id": pending.resolved_command_id,
                        "staged_task": staged_task.public_summary(),
                        "job": job,
                        "retry_allowed": False,
                    }
                    await self._audit(
                        stage="task_submitted",
                        preview=pending.preview,
                        commit=commit,
                        outcome="submitted",
                        metadata={
                            "staged_task_id": str(staged_task.task_id),
                            "job_id": str(job.get("job_id") or ""),
                        },
                    )
                    await self._emit("neural_result", payload)
                    return payload

                self._apply_navigation_commit(commit)
                payload = self._navigation_commit_payload(commit)
                payload["focused_command_id"] = self._focused_command_id()
                await self._audit(
                    stage="result",
                    preview=pending.preview,
                    commit=commit,
                    outcome="committed",
                    metadata={"canonical_goal": commit.canonical_goal},
                )
                await self._emit("neural_navigation", payload)
                return payload

            profile = self._config.gateway.source_profiles.get("neural", DEFAULT_SOURCE_PROFILES["neural"])
            results = await self._executor.execute(
                plan,
                plan_id=plan_id,
                invocation_source=InvocationSource.NEURAL,
                scope_override=TaskScopeOverride(
                    max_tier=profile.max_tier,
                    deny_action_types=profile.deny_action_types,
                    allow_root=profile.allow_root,
                ),
                critic_already_reviewed=False,
                user_confirmed=bool(assessment and assessment.requires_confirmation and world_model_approved),
            )
            payload = {
                "status": "completed" if results and all(result.success for result in results) else "failed",
                "preview_id": str(commit.preview_id),
                "intent_id": str(commit.intent_id),
                "plan_id": plan_id,
                "canonical_goal": commit.canonical_goal,
                "command_id": pending.resolved_command_id,
                "world_model": assessment.to_dict() if assessment else None,
                "results": [result.model_dump(mode="json") for result in results],
                "retry_allowed": False,
            }
            await self._audit(
                stage="result",
                preview=pending.preview,
                commit=commit,
                plan_id=plan_id,
                outcome=str(payload["status"]),
                metadata={
                    "canonical_goal": commit.canonical_goal,
                    "command_id": pending.resolved_command_id,
                    "result_count": len(results),
                },
            )
            await self._emit("neural_result", payload)
            return payload

    async def disarm(self, *, reason: str) -> dict[str, object]:
        async with self._lock:
            self._pending = None
            status = await self._gate.disarm(reason=reason)
            await self._emit("neural_disarmed", status)
            return status

    async def update_observation(
        self,
        summary: SignalQualitySummary,
        *,
        buffered_samples: int,
        dropped_samples: int,
        observed_at_ns: int,
    ) -> dict[str, object]:
        async with self._lock:
            self._last_observation = {
                "quality": summary.model_dump(mode="json"),
                "buffered_samples": max(0, buffered_samples),
                "dropped_samples": max(0, dropped_samples),
                "observed_at_ns": max(0, observed_at_ns),
            }
            await self._emit("neural_observation", self._last_observation)
            return dict(self._last_observation)

    async def status(self) -> dict[str, object]:
        status = await self._gate.status()
        audit_status: dict[str, object] = {"enabled": self._audit_store is not None}
        if self._audit_store is not None:
            verification = await self._audit_store.verify_chain()
            audit_status.update(
                valid=verification.valid,
                checked_entries=verification.checked_entries,
                error=verification.error,
            )
        return {
            **status,
            "capabilities": {
                "observe": True,
                "calibrate": True,
                "ui_navigation": True,
                "safe_desktop": True,
                "physical_control": False,
                "destructive_approval": False,
                "staged_autonomous_tasks": self._task_dispatcher is not None,
                "free_form_thought_decoding": False,
            },
            "safe_goals": self._goals.public_summaries(),
            "staged_tasks": [task.public_summary() for task in self._staged_tasks.values()],
            "focused_command_id": self._focused_command_id(),
            "last_observation": self._last_observation,
            "audit": audit_status,
            "latest_stimulus_marker": (
                self._stimulus_markers[-1].model_dump(mode="json") if self._stimulus_markers else None
            ),
        }

    async def gate_state(self) -> str:
        """The gate's session state only. Unlike status(), it does not verify the audit chain, so it is cheap
        enough to check on every selection-study trial."""
        return str((await self._gate.status())["state"])

    async def record_stimulus_marker(
        self,
        session_id: UUID,
        *,
        target_id: str | None,
        event: NeuralStimulusEvent,
        client_performance_ms: float,
    ) -> dict[str, object]:
        async with self._lock:
            status = await self._gate.status()
            if not status.get("connected") or str(status.get("session_id")) != str(session_id):
                raise NeuralControlError("stimulus marker does not match the active neural session")
            if target_id is not None and target_id not in {"focus_left", "focus_right", "select", "cancel"}:
                raise NeuralControlError("stimulus target is not in the registered SSVEP grid")
            marker = NeuralStimulusMarkerV1(
                session_id=session_id,
                sequence=self._stimulus_sequence,
                target_id=target_id,
                event=event,
                received_monotonic_ns=time.monotonic_ns(),
                client_performance_ms=client_performance_ms,
            )
            self._stimulus_sequence += 1
            self._stimulus_markers.append(marker)
            if self._audit_store is not None:
                await self._audit_store.record_event(
                    stage="stimulus_marker",
                    session_id=str(session_id),
                    outcome=event.value,
                    metadata={"target_id": target_id, "sequence": marker.sequence},
                )
            return marker.model_dump(mode="json")

    async def stimulus_markers(self, *, after_sequence: int) -> tuple[dict[str, object], ...]:
        async with self._lock:
            return tuple(
                marker.model_dump(mode="json") for marker in self._stimulus_markers if marker.sequence > after_sequence
            )

    async def _audit(
        self,
        *,
        stage: str,
        intent: NeuralIntentV1 | None = None,
        preview: NeuralPreview | None = None,
        commit: NeuralCommit | None = None,
        plan_id: str = "",
        outcome: str = "",
        metadata: dict[str, object] | None = None,
    ) -> None:
        if self._audit_store is None:
            return
        source = intent or preview or commit
        if source is None:
            raise NeuralControlError("neural audit event is missing provenance")
        await self._audit_store.record_event(
            stage=stage,
            session_id=str(source.session_id),
            intent_id=str(source.intent_id),
            preview_id=str(getattr(preview or commit, "preview_id", "")),
            plan_id=plan_id,
            window_start_ns=int(getattr(intent or preview, "window_start_ns", 0)),
            window_end_ns=int(getattr(intent or preview, "window_end_ns", 0)),
            outcome=outcome,
            metadata=metadata,
        )

    def _assess_world_model(self, plan: ActionPlan) -> PlanRiskAssessment:
        if not self._config.gateway.risk_gate_enabled:
            raise NeuralControlError("world model is disabled; neural desktop control fails closed")
        assessment = assess_plan_risk(plan, self._config)
        if assessment.model_version in {"disabled", "evaluation-error"}:
            raise NeuralControlError("world model is unavailable; neural desktop control fails closed")
        return assessment

    @staticmethod
    def _preview_payload(pending: PendingNeuralExecution) -> dict[str, object]:
        preview = pending.preview
        assessment = pending.world_model
        return {
            "status": "previewed",
            "preview_id": str(preview.preview_id),
            "intent_id": str(preview.intent_id),
            "intent_class": preview.intent_class.value,
            "command_id": preview.command_id,
            "resolved_command_id": pending.resolved_command_id,
            "canonical_goal": preview.canonical_goal,
            "requested_scope": preview.requested_scope.value,
            "state_revision": preview.state_revision,
            "created_at_ns": preview.created_at_ns,
            "eligible_at_ns": preview.eligible_at_ns,
            "expires_at_ns": preview.expires_at_ns,
            "world_model": assessment.to_dict() if assessment else None,
            "requires_non_neural_approval": bool(assessment and assessment.requires_confirmation),
            "fusion": NeuralController._public_fusion(pending.fusion),
            "staged_task": pending.staged_task.public_summary() if pending.staged_task else None,
        }

    async def _capture_fusion(self) -> dict[str, Any]:
        if self._fusion_snapshot is None:
            return {"modalities": [], "cancellation_present": False}
        snapshot = await self._fusion_snapshot()
        if not isinstance(snapshot, dict):
            raise NeuralControlError("multimodal fusion snapshot is unavailable")
        return snapshot

    async def _enforce_fusion_safety(self, snapshot: dict[str, Any], *, reason: str) -> None:
        if snapshot.get("cancellation_present") is True:
            self._pending = None
            await self._gate.disarm(reason=reason)
            raise NeuralControlError("a simultaneous voice or gesture cancellation disarmed neural control")

    @staticmethod
    def _public_fusion(snapshot: dict[str, object] | None) -> dict[str, object]:
        snapshot = snapshot or {}
        modalities = snapshot.get("modalities")
        return {
            "modalities": list(modalities) if isinstance(modalities, list) else [],
            "cancellation_present": snapshot.get("cancellation_present") is True,
            "raw_media_excluded": True,
        }

    @staticmethod
    def _navigation_commit_payload(commit: NeuralCommit) -> dict[str, object]:
        return {
            "status": "committed",
            "preview_id": str(commit.preview_id),
            "intent_id": str(commit.intent_id),
            "canonical_goal": commit.canonical_goal,
            "requested_scope": commit.requested_scope.value,
            "committed_at_ns": commit.committed_at_ns,
        }

    def _focused_command_id(self) -> str:
        command_ids = self._focusable_command_ids()
        if not command_ids:
            raise NeuralControlError("no safe neural goals are registered")
        return command_ids[self._focus_index % len(command_ids)]

    def _apply_navigation_commit(self, commit: NeuralCommit) -> None:
        command_ids = self._focusable_command_ids()
        if not command_ids:
            return
        if commit.canonical_goal == "neural_ui.focus_left":
            self._focus_index = (self._focus_index - 1) % len(command_ids)
        elif commit.canonical_goal == "neural_ui.focus_right":
            self._focus_index = (self._focus_index + 1) % len(command_ids)

    def _focusable_command_ids(self) -> tuple[str, ...]:
        return self._goals.command_ids + tuple(task.command_id for task in self._staged_tasks.values())

    def _resolve_staged_task(self, command_id: str) -> StagedNeuralTask | None:
        if not command_id.startswith(STAGED_TASK_COMMAND_PREFIX):
            return None
        try:
            task_id = UUID(command_id.removeprefix(STAGED_TASK_COMMAND_PREFIX))
        except ValueError as exc:
            raise NeuralControlError("staged neural task command is invalid") from exc
        task = self._staged_tasks.get(task_id)
        if task is None:
            raise NeuralControlError("staged neural task is no longer available")
        return task

    async def _emit(self, method: str, payload: dict[str, object]) -> None:
        if self._broadcast is not None:
            await self._broadcast(method, payload)
