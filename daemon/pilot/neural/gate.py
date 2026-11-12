"""Fail-closed neural session, preview, and commit authority."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable
from uuid import UUID, uuid4

from pydantic import TypeAdapter

from pilot.neural.protocol import (
    ArtifactHash,
    Identifier,
    NeuralCalibrationMetricsV1,
    NeuralIntentClass,
    NeuralIntentV1,
    NeuralScope,
    NeuralStreamDescriptorV1,
    SignalQuality,
)

_ARTIFACT_HASH = TypeAdapter(ArtifactHash)
_IDENTIFIER = TypeAdapter(Identifier)


class NeuralGateError(ValueError):
    """A neural candidate failed a deterministic boundary."""


class NeuralSessionState(StrEnum):
    DISCONNECTED = "disconnected"
    CONNECTED_UNCALIBRATED = "connected_uncalibrated"
    CALIBRATING = "calibrating"
    OBSERVE_ONLY = "observe_only"
    ARMED_SAFE_UI = "armed_safe_ui"
    ARMED_SAFE_DESKTOP = "armed_safe_desktop"
    CANDIDATE_INTENT = "candidate_intent"
    PREVIEWED = "previewed"
    COMMITTED = "committed"
    ABSTAINED = "abstained"
    COOLDOWN = "cooldown"


@dataclass(frozen=True, slots=True)
class NeuralIntentGateConfig:
    min_posterior_permille: int = 750
    min_margin_permille: int = 150
    min_dwell_windows: int = 3
    max_window_ns: int = 5_000_000_000
    max_window_age_ns: int = 5_000_000_000
    max_future_skew_ns: int = 500_000_000
    cancellation_window_ns: int = 800_000_000
    cooldown_ns: int = 1_000_000_000
    max_seen_intents: int = 2048

    def __post_init__(self) -> None:
        if not 0 <= self.min_margin_permille <= self.min_posterior_permille <= 1000:
            raise ValueError("neural confidence thresholds must be ordered permille values")
        for field_name in (
            "min_dwell_windows",
            "max_window_ns",
            "max_window_age_ns",
            "max_future_skew_ns",
            "cancellation_window_ns",
            "cooldown_ns",
            "max_seen_intents",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")


@dataclass(frozen=True, slots=True)
class NeuralPreview:
    session_id: UUID
    preview_id: UUID
    intent_id: UUID
    window_start_ns: int
    window_end_ns: int
    intent_class: NeuralIntentClass
    command_id: str | None
    canonical_goal: str
    requested_scope: NeuralScope
    state_revision: int
    created_at_ns: int
    eligible_at_ns: int
    expires_at_ns: int


@dataclass(frozen=True, slots=True)
class NeuralCommit:
    session_id: UUID
    preview_id: UUID
    intent_id: UUID
    window_start_ns: int
    window_end_ns: int
    canonical_goal: str
    requested_scope: NeuralScope
    committed_at_ns: int


@dataclass(slots=True)
class _Session:
    descriptor: NeuralStreamDescriptorV1
    state: NeuralSessionState = NeuralSessionState.CONNECTED_UNCALIBRATED
    calibration_id: str = ""
    decoder_version: str = ""
    calibration_metrics: NeuralCalibrationMetricsV1 | None = None
    subject_key: str = ""
    armed_scope: NeuralScope = NeuralScope.OBSERVE
    state_revision: int = 0
    last_sequence: int = -1
    last_commit_ns: int = 0
    seen_intents: set[UUID] = field(default_factory=set)
    seen_order: deque[UUID] = field(default_factory=deque)
    pending: NeuralPreview | None = None


class NeuralIntentSigner:
    """HMAC authority shared only with the least-privileged neural gateway."""

    def __init__(self, key: bytes) -> None:
        if len(key) < 32:
            raise ValueError("neural signing keys must contain at least 32 bytes")
        self._key = bytes(key)

    def sign(self, intent: NeuralIntentV1) -> str:
        return hmac.new(self._key, intent.signing_payload(), hashlib.sha256).hexdigest()

    def verify(self, intent: NeuralIntentV1) -> bool:
        return hmac.compare_digest(self.sign(intent), intent.signature)


class NeuralIntentGate:
    """Own the one active neural controller session.

    The gate never executes actions. It can release only a pre-authored goal
    after signed evidence passes freshness, signal-quality, dwell, confidence,
    replay, cancellation, revision, cooldown, scope, and tier constraints.
    """

    _SCOPE_RANK = {
        NeuralScope.OBSERVE: 0,
        NeuralScope.NAVIGATE: 1,
        NeuralScope.SAFE_DESKTOP: 2,
        NeuralScope.PHYSICAL_GOAL: 3,
    }

    def __init__(
        self,
        *,
        signer: NeuralIntentSigner,
        safe_goals: dict[str, str] | None = None,
        config: NeuralIntentGateConfig | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._signer = signer
        self._safe_goals = dict(safe_goals or {})
        self._config = config or NeuralIntentGateConfig()
        self._monotonic_ns = monotonic_ns
        self._session: _Session | None = None
        self._lock = asyncio.Lock()

    async def connect(self, descriptor: NeuralStreamDescriptorV1) -> dict[str, object]:
        async with self._lock:
            self._session = _Session(
                descriptor=descriptor,
                last_sequence=descriptor.sequence_start - 1,
            )
            return self._status_locked()

    async def begin_calibration(self, session_id: UUID) -> dict[str, object]:
        async with self._lock:
            session = self._require_session(session_id)
            session.state = NeuralSessionState.CALIBRATING
            session.pending = None
            session.state_revision += 1
            return self._status_locked()

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
            session = self._require_session(session_id)
            if session.state != NeuralSessionState.CALIBRATING:
                raise NeuralGateError("calibration was not active")
            session.calibration_id = _ARTIFACT_HASH.validate_python(calibration_id)
            session.subject_key = _IDENTIFIER.validate_python(subject_key)
            session.decoder_version = _ARTIFACT_HASH.validate_python(decoder_version) if decoder_version else ""
            session.calibration_metrics = metrics
            session.state = NeuralSessionState.OBSERVE_ONLY
            session.state_revision += 1
            return self._status_locked()

    async def arm(
        self,
        session_id: UUID,
        *,
        scope: NeuralScope,
        non_neural_authorized: bool,
    ) -> dict[str, object]:
        async with self._lock:
            session = self._require_session(session_id)
            if not session.calibration_id:
                raise NeuralGateError("a current calibration is required")
            if not non_neural_authorized:
                raise NeuralGateError("arming requires a non-neural user action")
            if scope == NeuralScope.PHYSICAL_GOAL:
                raise NeuralGateError("physical neural control is disabled")
            if scope == NeuralScope.OBSERVE:
                raise NeuralGateError("observe-only sessions are not armed")
            session.armed_scope = scope
            session.pending = None
            session.state = (
                NeuralSessionState.ARMED_SAFE_UI
                if scope == NeuralScope.NAVIGATE
                else NeuralSessionState.ARMED_SAFE_DESKTOP
            )
            session.state_revision += 1
            return self._status_locked()

    async def disarm(self, *, reason: str = "user_disarm") -> dict[str, object]:
        async with self._lock:
            if self._session is None:
                return {"state": NeuralSessionState.DISCONNECTED.value, "reason": reason}
            self._session.pending = None
            self._session.armed_scope = NeuralScope.OBSERVE
            self._session.state = NeuralSessionState.OBSERVE_ONLY
            self._session.state_revision += 1
            status = self._status_locked()
            status["reason"] = reason
            return status

    async def preview(self, intent: NeuralIntentV1, *, now_ns: int | None = None) -> NeuralPreview | None:
        async with self._lock:
            now = self._monotonic_ns() if now_ns is None else now_ns
            session = self._require_session(intent.session_id)
            self._verify_authenticated_envelope(session, intent, now)
            self._consume_identity(session, intent)

            if intent.intent_class == NeuralIntentClass.CANCEL:
                session.pending = None
                session.armed_scope = NeuralScope.OBSERVE
                session.state = NeuralSessionState.OBSERVE_ONLY
                session.state_revision += 1
                return None

            self._verify_candidate_policy(session, intent, now)
            canonical_goal = self._resolve_goal(session, intent)
            session.state = NeuralSessionState.CANDIDATE_INTENT
            session.state_revision += 1
            preview = NeuralPreview(
                session_id=intent.session_id,
                preview_id=uuid4(),
                intent_id=intent.intent_id,
                window_start_ns=intent.window_start_ns,
                window_end_ns=intent.window_end_ns,
                intent_class=intent.intent_class,
                command_id=intent.command_id,
                canonical_goal=canonical_goal,
                requested_scope=intent.requested_scope,
                state_revision=session.state_revision,
                created_at_ns=now,
                eligible_at_ns=now + self._config.cancellation_window_ns,
                expires_at_ns=intent.expires_at_ns,
            )
            session.pending = preview
            session.state = NeuralSessionState.PREVIEWED
            return preview

    async def commit(
        self,
        preview_id: UUID,
        *,
        expected_revision: int,
        effect_tier: int,
        now_ns: int | None = None,
    ) -> NeuralCommit:
        async with self._lock:
            now = self._monotonic_ns() if now_ns is None else now_ns
            session = self._require_active_session()
            preview = session.pending
            if preview is None or preview.preview_id != preview_id:
                raise NeuralGateError("preview is missing or no longer current")
            if expected_revision != session.state_revision or expected_revision != preview.state_revision:
                raise NeuralGateError("neural state changed after preview")
            if effect_tier not in {0, 1}:
                raise NeuralGateError("neural commit authority is limited to Tier 0/1 effects")
            if now < preview.eligible_at_ns:
                raise NeuralGateError("the cancellation window is still open")
            if now > preview.expires_at_ns:
                session.pending = None
                session.state = NeuralSessionState.ABSTAINED
                session.state_revision += 1
                raise NeuralGateError("preview expired before commit")
            if session.last_commit_ns and now - session.last_commit_ns < self._config.cooldown_ns:
                raise NeuralGateError("neural controller is cooling down")

            session.pending = None
            session.last_commit_ns = now
            session.state = NeuralSessionState.COMMITTED
            session.state_revision += 1
            commit = NeuralCommit(
                session_id=preview.session_id,
                preview_id=preview.preview_id,
                intent_id=preview.intent_id,
                window_start_ns=preview.window_start_ns,
                window_end_ns=preview.window_end_ns,
                canonical_goal=preview.canonical_goal,
                requested_scope=preview.requested_scope,
                committed_at_ns=now,
            )
            session.state = NeuralSessionState.COOLDOWN
            return commit

    async def status(self) -> dict[str, object]:
        async with self._lock:
            if self._session is not None:
                now = self._monotonic_ns()
                self._refresh_cooldown(self._session, now)
                self._refresh_lapsed_preview(self._session, now)
            return self._status_locked()

    def _refresh_lapsed_preview(self, session: _Session, now: int) -> None:
        """A preview nobody committed in time, or one whose commit came too late, is dropped and the session is
        armed again, so the bridge (which decodes only in an armed state) can offer the next intent.

        Before this a lapsed preview left the session in "previewed" or "abstained" for good: no new previews,
        neural control dead until it was armed again by hand. Dropping it commits nothing; every new preview still
        needs its own commit after the cancellation window, so this fails closed.
        """
        lapsed = session.state == NeuralSessionState.ABSTAINED or (
            session.state == NeuralSessionState.PREVIEWED
            and (session.pending is None or now > session.pending.expires_at_ns)
        )
        if not lapsed:
            return
        session.pending = None
        session.state = {
            NeuralScope.NAVIGATE: NeuralSessionState.ARMED_SAFE_UI,
            NeuralScope.SAFE_DESKTOP: NeuralSessionState.ARMED_SAFE_DESKTOP,
        }.get(session.armed_scope, NeuralSessionState.OBSERVE_ONLY)
        session.state_revision += 1

    def _refresh_cooldown(self, session: _Session, now: int) -> None:
        if (
            session.state == NeuralSessionState.COOLDOWN
            and session.last_commit_ns
            and now - session.last_commit_ns >= self._config.cooldown_ns
        ):
            session.state = (
                NeuralSessionState.ARMED_SAFE_UI
                if session.armed_scope == NeuralScope.NAVIGATE
                else NeuralSessionState.ARMED_SAFE_DESKTOP
            )
            session.state_revision += 1

    def _verify_authenticated_envelope(self, session: _Session, intent: NeuralIntentV1, now: int) -> None:
        if not self._signer.verify(intent):
            raise NeuralGateError("invalid neural intent signature")
        if intent.calibration_id != session.calibration_id:
            raise NeuralGateError("intent calibration does not match the active session")
        if intent.subject_key != session.subject_key:
            raise NeuralGateError("intent subject does not match the active session")
        if intent.state_revision != session.state_revision:
            raise NeuralGateError("intent was decoded against a stale controller state")
        if intent.sequence <= session.last_sequence:
            raise NeuralGateError("neural sequence was replayed or reordered")
        if intent.intent_id in session.seen_intents:
            raise NeuralGateError("neural intent_id was replayed")
        if intent.window_end_ns - intent.window_start_ns > self._config.max_window_ns:
            raise NeuralGateError("neural evidence window is too large")
        if intent.window_start_ns > now + self._config.max_future_skew_ns:
            raise NeuralGateError("neural evidence timestamp is in the future")
        if intent.window_end_ns < now - self._config.max_window_age_ns:
            raise NeuralGateError("neural evidence window is stale")
        if intent.expires_at_ns < now:
            raise NeuralGateError("neural intent expired")

    def _consume_identity(self, session: _Session, intent: NeuralIntentV1) -> None:
        session.last_sequence = intent.sequence
        session.seen_intents.add(intent.intent_id)
        session.seen_order.append(intent.intent_id)
        while len(session.seen_order) > self._config.max_seen_intents:
            expired = session.seen_order.popleft()
            session.seen_intents.discard(expired)

    def _verify_candidate_policy(self, session: _Session, intent: NeuralIntentV1, now: int) -> None:
        if session.state not in {
            NeuralSessionState.ARMED_SAFE_UI,
            NeuralSessionState.ARMED_SAFE_DESKTOP,
            NeuralSessionState.PREVIEWED,
            NeuralSessionState.COOLDOWN,
        }:
            raise NeuralGateError("neural control is not armed")
        if session.last_commit_ns and now - session.last_commit_ns < self._config.cooldown_ns:
            raise NeuralGateError("neural controller is cooling down")
        if intent.signal_quality != SignalQuality.GOOD:
            raise NeuralGateError("signal quality requires abstention")
        if intent.artifact_flags:
            raise NeuralGateError("artifact flags require abstention")
        if intent.posterior_permille < self._config.min_posterior_permille:
            raise NeuralGateError("neural confidence is below threshold")
        if intent.margin_permille < self._config.min_margin_permille:
            raise NeuralGateError("neural class margin is below threshold")
        if intent.dwell_windows < self._config.min_dwell_windows:
            raise NeuralGateError("neural dwell requirement is not met")
        if self._SCOPE_RANK[intent.requested_scope] > self._SCOPE_RANK[session.armed_scope]:
            raise NeuralGateError("intent exceeds the armed neural scope")
        if intent.requested_scope == NeuralScope.PHYSICAL_GOAL:
            raise NeuralGateError("physical neural control is disabled")

    def _resolve_goal(self, session: _Session, intent: NeuralIntentV1) -> str:
        ui_goals = {
            NeuralIntentClass.FOCUS_LEFT: "neural_ui.focus_left",
            NeuralIntentClass.FOCUS_RIGHT: "neural_ui.focus_right",
            NeuralIntentClass.SELECT: "neural_ui.select",
        }
        if intent.intent_class in ui_goals:
            if intent.requested_scope not in {NeuralScope.NAVIGATE, NeuralScope.SAFE_DESKTOP}:
                raise NeuralGateError("UI navigation intents require navigate or safe_desktop scope")
            return ui_goals[intent.intent_class]
        if intent.intent_class == NeuralIntentClass.SAFE_GOAL:
            if session.armed_scope != NeuralScope.SAFE_DESKTOP:
                raise NeuralGateError("desktop goals require safe_desktop arming")
            goal = self._safe_goals.get(intent.command_id or "")
            if not goal:
                raise NeuralGateError("command_id is not in the compiled safe-goal allow-list")
            return goal
        raise NeuralGateError("intent class cannot produce a commit")

    def _require_session(self, session_id: UUID) -> _Session:
        session = self._require_active_session()
        if session.descriptor.session_id != session_id:
            raise NeuralGateError("session_id does not match the active neural controller")
        return session

    def _require_active_session(self) -> _Session:
        if self._session is None:
            raise NeuralGateError("no neural session is connected")
        return self._session

    def _status_locked(self) -> dict[str, object]:
        session = self._session
        if session is None:
            return {"state": NeuralSessionState.DISCONNECTED.value, "connected": False}
        return {
            "state": session.state.value,
            "connected": True,
            "session_id": str(session.descriptor.session_id),
            "source_id": session.descriptor.source_id,
            "board_kind": session.descriptor.board_kind,
            "transport": session.descriptor.transport.value,
            "evidence_kind": session.descriptor.evidence_kind.value,
            "calibrated": bool(session.calibration_id),
            "calibration_id": session.calibration_id,
            "decoder_version": session.decoder_version,
            "calibration_metrics": (
                session.calibration_metrics.model_dump(mode="json") if session.calibration_metrics else None
            ),
            "armed_scope": session.armed_scope.value,
            "state_revision": session.state_revision,
            "last_sequence": session.last_sequence,
            "pending_preview": str(session.pending.preview_id) if session.pending else None,
        }
