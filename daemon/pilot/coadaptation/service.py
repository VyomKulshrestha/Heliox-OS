"""CoAdaptationService: the daemon side of the consented selection study (spec section 13.3).

The service sits between the RPC handlers and the pure core engine. It receives only a settings provider, a gaze flag
provider, paths, a key provider, a broadcast function and a clock; it holds no reference to approvals, permissions,
execution, memory or the experience ledger, and nothing it does changes what Heliox is allowed to run. Every
recommendation it returns is presentation-only.

One ``asyncio.Lock`` serialises every mutating call. Each RPC method takes the raw params dict and returns the result
dict of spec section 14; it never raises. Engines are kept per participant in memory, restored from the stored
snapshot when it still matches the rows, and otherwise rebuilt by replaying the stored rows (spec 10.8). A rebuild or
restore never runs while the participant has an open session.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import math
import secrets
import statistics
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import ValidationError

from pilot.coadaptation.consent import CoAdaptationConsentError, CoAdaptationConsentV1, consent_summary
from pilot.coadaptation.core import SDK_VERSION
from pilot.coadaptation.core.consent import CoAdaptationPurpose, new_participant_key
from pilot.coadaptation.core.constraints import fallbacks as compute_fallbacks
from pilot.coadaptation.core.engine import (
    CoAdaptationEngine,
    EngineEvent,
    EngineSettings,
    NoAllowedRegime,
)
from pilot.coadaptation.core.evidence import EvidenceKind
from pilot.coadaptation.core.outcome_model import MODEL_VERSION, CalibrationMonitor
from pilot.coadaptation.core.policy import POLICY_VERSION
from pilot.coadaptation.core.preferences import Preference, PreferenceConflict, PreferenceSet, preference_to_dict
from pilot.coadaptation.core.priors import PRIOR_VERSION
from pilot.coadaptation.core.records import BurdenReport, DecisionRecord, SessionRow, StoredTrial, vetoed_variant
from pilot.coadaptation.core.state import FEATURE_SCHEMA, BlockContext, ClockContext, day_index_for
from pilot.coadaptation.core.study_plan import (
    PROTOCOL_HYPERPARAMETERS,
    STUDY_PROTOCOL_ID,
    build_session_plan,
    plan_seed,
    rank_fixed_regimes,
    study_protocol_sha256,
)
from pilot.coadaptation.core.utility import pointing_throughput
from pilot.coadaptation.core.vocabulary import ASSISTANCE_OFF, STRATEGY_VERSION, DecisionLevel, acting, l1_arms
from pilot.coadaptation.keys import CoAdaptationKeyUnavailable, KeyProvider, OsKeyringKeyProvider
from pilot.coadaptation.preferences_store import PreferenceFileError, PreferenceStore
from pilot.coadaptation.schemas import (
    PAYLOAD_LIMIT_BYTES,
    TRIAL_PAYLOAD_LIMIT_BYTES,
    AmbiguityTableContextV1,
    BlockRegimeContextV1,
    ConsentGrantParams,
    EmptyParams,
    EraseParams,
    OfferContextV1,
    ParticipantParams,
    PayloadTooLarge,
    PreferenceRetractParams,
    PreferenceSetParams,
    PreferenceV1,
    RecommendParams,
    RecordTrialParams,
    ReportBurdenParams,
    ResultsParams,
    SessionBeginParams,
    SessionEndParams,
    SessionRefParams,
    TrialV1,
    check_trial_spread,
    validation_message,
    without_ineffective_fields,
)
from pilot.coadaptation.store import CoAdaptationStore, FallbackRelabel, SessionRows

logger = logging.getLogger("pilot.coadaptation.service")
_T = TypeVar("_T")

CONSENT_TEXT_VERSION = "coadapt-consent-v1"
MODALITIES = ("mouse", "touchpad", "pen", "touch", "keyboard", "hand_pointer")
FLOOR_PX = {"mouse": 1.0, "touchpad": 1.5, "pen": 1.5, "touch": 3.0, "hand_pointer": 6.0}
GAZE_ACCURACY_MIN = 0.6
PRUNE_INTERVAL_S = 24 * 60 * 60
PROPOSAL_WINDOW = timedelta(days=7)
PROPOSAL_VETOES = 3
# A session_begin for a participant whose open session saw no study call for this long ends that session as
# abandoned (the UI lost it: a reload or a webview crash); a younger one is still refused with session_active.
STALE_SESSION_AFTER = timedelta(seconds=60)
LAYOUT_SEED_LIMIT = 2**31 - 100_000
KEYRING_MESSAGE = "Secure OS key storage is unavailable; co-adaptation stays off."
SESSION_ACTIVE_MESSAGE = "Stop the running session before removing a permission."
AUTHORITY = {"authority": "presentation_only", "execution_authority": False}
_TABLE_BASELINE = {"level": "ambiguity_table", "cells": []}
_VERIFICATION_VALUES = ("ok", "fallback", "draw_mismatch", "posterior_mismatch", "interrupted")


class CoAdaptationSettingsView(Protocol):
    """The 13 ``co_adaptation`` config fields the service reads (satisfied by ``pilot.config.CoAdaptationConfig``)."""

    enabled: bool
    adaptive_enabled: bool
    learn_across_sessions: bool
    gaze_prior_enabled: bool
    hand_pointer_enabled: bool
    ambiguity_threshold: float
    unintended_activation_tolerance: float
    noninferiority_margin: float
    upper_bound_credibility: float
    exploration_floor: float
    switching_penalty: float
    burden_prompt_every_blocks: int
    day_boundary_hour: int


SETTINGS_FIELDS = (
    "enabled",
    "adaptive_enabled",
    "learn_across_sessions",
    "gaze_prior_enabled",
    "hand_pointer_enabled",
    "ambiguity_threshold",
    "unintended_activation_tolerance",
    "noninferiority_margin",
    "upper_bound_credibility",
    "exploration_floor",
    "switching_penalty",
    "burden_prompt_every_blocks",
    "day_boundary_hour",
)


@dataclasses.dataclass(frozen=True, slots=True)
class _Effective:
    """Purposes in effective use: granted, switched on in config and, for gaze, gaze tracking enabled (spec 10.2)."""

    adaptive: bool
    cross_session: bool
    gaze: bool
    hand: bool
    export: bool


@dataclasses.dataclass(slots=True)
class _LiveSession:
    """In-memory state of one open session. Lost on restart: start() closes sessions a restart left open."""

    session_id: str
    participant_key: str
    row: SessionRow
    modality: str
    gaze_used: bool
    hand_used: bool
    days_since_last_session: float | None
    last_activity: datetime
    next_seq: int = 1
    decision_ids: set[str] = dataclasses.field(default_factory=set)
    l1_decisions: set[str] = dataclasses.field(default_factory=set)
    tables: dict[str, str] = dataclasses.field(default_factory=dict)
    trial_replies: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    veto_counts: Counter[str] = dataclasses.field(default_factory=Counter)

    def take_seq(self) -> int:
        seq = self.next_seq
        self.next_seq += 1
        return seq


def _plain(value: Any) -> Any:
    """JSON form of a core result object (to_dict where it exists, else field by field)."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _bins(tau: float) -> dict[str, list[float]]:
    return {
        "b1": [tau, max(tau, 0.4)],
        "b2": [max(tau, 0.4), max(tau, 0.6)],
        "b3": [max(tau, 0.6), 1.0],
    }


def _labels_reply(labels: Any) -> dict[str, Any] | None:
    """The ``labels`` object of the record_trial reply (spec 14.3)."""
    if labels is None:
        return None
    return {name: getattr(labels, name) for name in ("complete", "ua", "correction", "veto", "time_ms")}


def _label(labels: Any, name: str) -> Any:
    return getattr(labels, name, None)


def _mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return float(ordered[rank - 1])


class CoAdaptationService:
    """See the module docstring and spec section 13.3."""

    def __init__(
        self,
        *,
        settings_provider: Callable[[], CoAdaptationSettingsView],
        gaze_tracking_enabled: Callable[[], bool],
        db_path: Path,
        preferences_path: Path,
        exports_dir: Path,
        key_provider: KeyProvider | None = None,
        broadcast_fn: Callable[[str, dict], Awaitable[None]] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._settings_provider = settings_provider
        self._gaze_tracking_enabled = gaze_tracking_enabled
        self._keys: KeyProvider = key_provider if key_provider is not None else OsKeyringKeyProvider()
        self._broadcast_fn = broadcast_fn
        self._clock = clock
        self._store = CoAdaptationStore(db_path, exports_dir, self._keys, clock=clock)
        self._prefs_store = PreferenceStore(preferences_path, clock=clock)
        self._prefs = PreferenceSet(())
        self._prefs_error = False
        self._lock = asyncio.Lock()
        self._engines: dict[str, CoAdaptationEngine] = {}
        self._sessions: dict[str, _LiveSession] = {}
        self._refresh_seq: dict[str, int] = {}
        self._known_proposals: dict[str, set[str]] = {}
        self._gone: set[str] = set()
        self._gone_sessions: dict[str, str] = {}
        self._closed_sessions: set[str] = set()
        self._active_consents: list[CoAdaptationConsentV1] = []
        self._prune_task: asyncio.Task[None] | None = None
        # Notifications queued while the lock is held; sent after it is released (see _locked).
        self._outbox: list[tuple[str, str | None, int | None]] = []
        # Service calls run as tasks the RPC task cannot cancel half way (see _call).
        self._calls: set[asyncio.Task[dict]] = set()
        # participant_key -> (loaded_since, [(recorded_at, vetoed variant)]) for cross-session proposals.
        self._veto_log: dict[str, tuple[datetime, list[tuple[datetime, str]]]] = {}

    # ── lifecycle ────────────────────────────────────────────────────────────

    @property
    def store(self) -> CoAdaptationStore:
        return self._store

    async def start(self) -> None:
        """Load rules, open an existing store, prune expired consents, close sessions a restart left open.

        Never raises. An error for one participant is logged with its key and the exception type and skips that
        participant's step. Without a usable keyring the service starts degraded: status, revoke, erase and prune
        work (deleting rows needs no key); every other call is refused with ``keyring_unavailable``. Erasure work an
        earlier run could not finish (pending-erasure list) is retried here, in every prune and on a repeated revoke.
        """
        await self._locked(self._start_unlocked)
        if self._prune_task is None:
            self._prune_task = asyncio.create_task(self._prune_loop())

    async def _start_unlocked(self) -> None:
        try:
            self._load_preferences()
        except Exception as exc:
            logger.warning("Co-adaptation preferences could not be loaded (%s)", type(exc).__name__)
        try:
            await self._store.open(create=False)
        except Exception as exc:
            logger.warning("Co-adaptation store could not be opened (%s)", type(exc).__name__)
        now = self._clock()
        try:
            await self._prune(now)
        except Exception as exc:
            logger.warning("Co-adaptation prune at start failed (%s)", type(exc).__name__)
        try:
            await self._close_leftover_sessions(now)
        except Exception as exc:
            logger.warning("Co-adaptation restart clean-up failed (%s)", type(exc).__name__)
        with contextlib.suppress(Exception):
            await self._refresh_consent_cache()

    async def close(self) -> None:
        """Cancel the prune task, save snapshots of engines whose participant still consents, close the store."""
        task, self._prune_task = self._prune_task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await self._locked(self._close_unlocked)

    async def _close_unlocked(self) -> None:
        now = self._clock()
        for participant_key, engine in list(self._engines.items()):
            if self._open_session_of(participant_key) is not None:
                # Rebuilt from rows next time; a snapshot is only taken between sessions.
                continue
            try:
                consent, _reason = await self._consent_for(participant_key, now)
                if consent is not None:
                    await self._save_snapshot(participant_key, engine, now)
            except Exception as exc:
                logger.warning(
                    "Co-adaptation snapshot at close failed for %s (%s)", participant_key, type(exc).__name__
                )
        self._engines.clear()
        self._sessions.clear()
        self._veto_log.clear()
        with contextlib.suppress(Exception):
            await self._store.close()

    def has_active_consent(self) -> bool:
        now = self._clock()
        return any(
            record.participant_key not in self._gone and now < record.expires_at for record in self._active_consents
        )

    async def on_config_changed(self, previous: CoAdaptationSettingsView) -> None:
        """Turning the feature off ends open sessions; every other change applies at the next session_begin."""
        await self._locked(self._on_config_changed_unlocked)

    async def _on_config_changed_unlocked(self) -> None:
        if self._view().enabled:
            return
        now = self._clock()
        for live in list(self._sessions.values()):
            try:
                await self._end_live_session(live, now, "disabled")
            except Exception as exc:
                logger.warning(
                    "Co-adaptation could not end a session for %s (%s)", live.participant_key, type(exc).__name__
                )
                self._drop_session(live)

    async def rebuild(self, participant_key: str) -> CoAdaptationEngine:
        """Replay the participant's retained rows into a fresh engine (spec 10.8), install it and store its snapshot."""

        async def run() -> CoAdaptationEngine:
            now = self._clock()
            consent, reason = await self._consent_for(participant_key, now)
            if consent is None:
                raise CoAdaptationConsentError(reason or "no_consent")
            return await self._rebuild_unlocked(participant_key, consent, now)

        return await self._locked(run)

    async def _locked(self, fn: Callable[[], Awaitable[_T]]) -> _T:
        """Run ``fn`` under the service lock, then send the notifications it queued once the lock is released.

        A UI client that stops reading its socket can stall a notification send; it never holds the lock, so the
        study's recommend and record_trial calls from a healthy client go on.
        """
        outbox: list[tuple[str, str | None, int | None]] = []
        try:
            async with self._lock:
                try:
                    return await fn()
                finally:
                    outbox, self._outbox = self._outbox, []
        finally:
            await self._send(outbox)

    # ── RPC methods (spec 14.2 and 14.3) ─────────────────────────────────────

    async def status(self, params: dict) -> dict:
        return await self._call("status", EmptyParams, params, self._status, works_degraded=True)

    async def grant_consent(self, params: dict) -> dict:
        return await self._call("grant_consent", ConsentGrantParams, params, self._grant_consent, works_degraded=True)

    async def revoke_consent(self, params: dict) -> dict:
        return await self._call("revoke_consent", ParticipantParams, params, self._revoke_consent, works_degraded=True)

    async def set_preference(self, params: dict) -> dict:
        return await self._call("set_preference", PreferenceSetParams, params, self._set_preference)

    async def retract_preference(self, params: dict) -> dict:
        return await self._call("retract_preference", PreferenceRetractParams, params, self._retract_preference)

    async def reset_learning(self, params: dict) -> dict:
        return await self._call("reset_learning", ParticipantParams, params, self._reset_learning)

    async def erase(self, params: dict) -> dict:
        return await self._call("erase", EraseParams, params, self._erase, works_degraded=True)

    async def results(self, params: dict) -> dict:
        return await self._call("results", ResultsParams, params, self._results)

    async def export(self, params: dict) -> dict:
        return await self._call("export", ParticipantParams, params, self._export)

    async def begin_session(self, params: dict) -> dict:
        return await self._call("begin_session", SessionBeginParams, params, self._begin_session)

    async def resume_session(self, params: dict) -> dict:
        return await self._call("resume_session", SessionRefParams, params, self._resume_session)

    async def recommend(self, params: dict) -> dict:
        return await self._call("recommend", RecommendParams, params, self._recommend)

    async def record_trial(self, params: dict) -> dict:
        return await self._call(
            "record_trial", RecordTrialParams, params, self._record_trial, limit=TRIAL_PAYLOAD_LIMIT_BYTES
        )

    async def report_burden(self, params: dict) -> dict:
        return await self._call("report_burden", ReportBurdenParams, params, self._report_burden)

    async def end_session(self, params: dict) -> dict:
        return await self._call("end_session", SessionEndParams, params, self._end_session)

    async def _call(
        self,
        name: str,
        model: Any,
        params: Any,
        handler: Callable[[Any], Awaitable[dict]],
        *,
        limit: int = PAYLOAD_LIMIT_BYTES,
        works_degraded: bool = False,
    ) -> dict:
        try:
            request = model.parse(params if isinstance(params, dict) else {}, limit=limit)
        except PayloadTooLarge as exc:
            return {"status": "error", "reason": "payload_too_large", "message": str(exc)}
        except ValidationError as exc:
            return {"status": "error", "message": validation_message(exc)}
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        if not works_degraded and not self._keyring_ok():
            # Degraded mode (spec 13.3): only status, revoke, erase and prune run without the OS keyring.
            return self._refused("keyring_unavailable", KEYRING_MESSAGE)
        # The handler runs in a task of its own. The server cancels a connection's request tasks when the UI socket
        # closes; that cancels this await only, never a handler half way (a trial the engine learned but the store
        # rolled back, a decision applied but not stored). The handler finishes and its reply is cached as usual.
        task = asyncio.create_task(self._run(name, handler, request))
        self._calls.add(task)
        task.add_done_callback(self._calls.discard)
        return await asyncio.shield(task)

    async def _run(self, name: str, handler: Callable[[Any], Awaitable[dict]], request: Any) -> dict:
        try:
            return await self._locked(lambda: handler(request))
        except CoAdaptationKeyUnavailable:
            return self._refused("keyring_unavailable", KEYRING_MESSAGE)
        except CoAdaptationConsentError as exc:
            return {"status": "ignored", "reason": exc.code}
        except Exception as exc:
            logger.warning("Co-adaptation %s failed (%s)", name, type(exc).__name__)
            return {"status": "error", "message": f"Co-adaptation could not complete {name}."}

    # ── status ───────────────────────────────────────────────────────────────

    async def _status(self, _request: EmptyParams) -> dict:
        now = self._clock()
        view = self._view()
        await self._prune(now)
        keyring_ok = self._keyring_ok()
        labels = await self._store.participant_labels() if keyring_ok else {}
        consents = await self._store.consents()
        participants: list[dict[str, Any]] = []
        proposals: list[dict[str, Any]] = []
        for participant_key in await self._store.participants():
            current = [
                (record, revoked)
                for record, revoked, replaced in consents
                if record.participant_key == participant_key and replaced is None
            ]
            record, revoked = current[-1] if current else (None, None)
            active = record is not None and revoked is None and now < record.expires_at
            model: dict[str, Any] | None = None
            fallbacks: dict[str, Any] = {}
            if keyring_ok and active and not self._prefs_error:
                try:
                    engine = await self._engine(participant_key, record, now)
                    model = self._model_status(engine)
                    fallbacks = {modality: engine.fallbacks(modality, now=now).to_dict() for modality in MODALITIES}
                    for variant in await self._proposals_for(participant_key, record, now):
                        proposals.append(
                            {
                                "participant_key": participant_key,
                                "kind": "forbid_variant",
                                "variant": variant,
                                "reason": "three_vetoes",
                            }
                        )
                except Exception as exc:
                    logger.warning("Co-adaptation status failed for %s (%s)", participant_key, type(exc).__name__)
            infos = await self._store.sessions_summary(participant_key)
            participants.append(
                {
                    "participant_key": participant_key,
                    "label": labels.get(participant_key) if keyring_ok else None,
                    "consent": None if record is None else consent_summary(record, revoked, now),
                    "model": model,
                    "fallbacks": fallbacks,
                    "sessions": len(infos),
                    "last_session_at": infos[-1].started_at.isoformat() if infos else None,
                }
            )
        try:
            preferences = [preference_to_dict(pref) for pref in self._prefs_store.entries()]
        except PreferenceFileError:
            preferences = []
        return {
            "status": "ok",
            "available": True,
            "config": self._config_dict(view),
            "keyring_available": keyring_ok,
            # True while a revoke, expiry or erase left an export file, a keyring key or a rule to delete later; the
            # prune this call just ran retried it. The only way the person learns that an expiry could not finish.
            "erasure_pending": not self._store.pending_erasure().empty,
            "store": await self._store.counts(),
            "participants": participants,
            "preferences": preferences,
            "proposals": proposals,
            "protocol": {"protocol_id": STUDY_PROTOCOL_ID, "sha256": study_protocol_sha256()},
            "versions": {**self._versions(), "consent_text": CONSENT_TEXT_VERSION},
            **AUTHORITY,
        }

    # ── consent ──────────────────────────────────────────────────────────────

    async def _grant_consent(self, request: ConsentGrantParams) -> dict:
        now = self._clock()
        if request.participant_key is None:
            return await self._grant_new(request, now)
        return await self._update_consent(request, now)

    async def _grant_new(self, request: ConsentGrantParams, now: datetime) -> dict:
        if not self._keyring_ok():
            return {"status": "error", "reason": "keyring_unavailable", "message": KEYRING_MESSAGE}
        try:
            # A key whose erasure is still pending is never reused: its old keyring entry may still exist.
            pending = self._store.pending_erasure()
            existing = (
                set(await self._store.participants()) | self._gone | set(pending.participants) | set(pending.rules)
            )
            participant_key = new_participant_key(existing)
            record = CoAdaptationConsentV1(
                participant_key=participant_key,
                purposes=frozenset(request.purposes),
                consent_text_version=request.consent_text_version,
                consent_text_sha256=request.consent_text_sha256,
                protocol_id=request.protocol_id,
                protocol_sha256=request.protocol_sha256,
                granted_at=now,
                expires_at=now + timedelta(days=request.retention_days),
                retention_days=request.retention_days,
                authorized=True,
            )
        except ValidationError as exc:
            return {"status": "error", "message": validation_message(exc)}
        # The database file and the keyring entry come into existence only here (spec 10.1).
        await self._store.open(create=True)
        try:
            await self._store.add_participant(participant_key, now, request.label)
            await self._store.put_consent(record)
            for pref in self._prefs_store.entries():
                if pref.scope.participant_key is None:
                    await self._store.put_rule(participant_key, pref)
        except Exception:
            with contextlib.suppress(Exception):
                await self._store.erase_participant(participant_key)
            raise
        await self._refresh_consent_cache()
        await self._broadcast("consent", participant_key)
        return {
            "status": "ok",
            "participant_key": participant_key,
            "consent": consent_summary(record, None, now),
            "removed": {
                "purposes": [],
                "export_files": 0,
                "trials_rewritten": 0,
                "sessions_rewritten": 0,
                "export_files_left": 0,
            },
        }

    async def _update_consent(self, request: ConsentGrantParams, now: datetime) -> dict:
        participant_key = request.participant_key
        assert participant_key is not None
        consent, _reason = await self._consent_for(participant_key, now)
        if consent is None:
            return {"status": "error", "reason": "unknown_participant", "message": "Unknown participant."}
        if request.label is not None:
            return {"status": "error", "message": "A display label can be given only for a new participant."}
        if not self._keyring_ok():
            return {"status": "error", "reason": "keyring_unavailable", "message": KEYRING_MESSAGE}
        purposes = frozenset(request.purposes)
        removed = frozenset(purpose for purpose in consent.purposes if purpose not in purposes)
        if removed and not await self._clear_stale_session(participant_key, now):
            return {"status": "refused", "reason": "session_active", "message": SESSION_ACTIVE_MESSAGE}
        await self._finish_pending_erasures()
        try:
            record = CoAdaptationConsentV1(
                participant_key=participant_key,
                purposes=purposes,
                consent_text_version=request.consent_text_version,
                consent_text_sha256=request.consent_text_sha256,
                protocol_id=request.protocol_id,
                protocol_sha256=request.protocol_sha256,
                granted_at=now,
                # The retention clock never restarts: an update can shorten the expiry, never extend it (10.3).
                expires_at=min(consent.expires_at, now + timedelta(days=request.retention_days)),
                retention_days=request.retention_days,
                authorized=True,
                replaces_consent_id=consent.consent_id,
            )
        except ValidationError as exc:
            return {"status": "error", "message": validation_message(exc)}
        result = await self._store.put_consent(record, removed_purposes=frozenset(p.value for p in removed))
        if CoAdaptationPurpose.HAND_POINTER in removed:
            # quality_deficit reads the hand summaries that were just deleted, so the model is recomputed.
            self._engines.pop(participant_key, None)
            await self._rebuild_unlocked(participant_key, record, now)
        await self._refresh_consent_cache()
        await self._broadcast("consent", participant_key)
        return {
            "status": "ok",
            "participant_key": participant_key,
            "consent": consent_summary(record, None, now),
            "removed": {
                "purposes": list(result.purposes),
                "export_files": result.export_files,
                "trials_rewritten": result.trials_rewritten,
                "sessions_rewritten": result.sessions_rewritten,
                "export_files_left": result.export_files_left,
            },
        }

    async def _revoke_consent(self, request: ParticipantParams) -> dict:
        participant_key = request.participant_key
        pending = self._store.pending_erasure()
        unfinished = participant_key in pending.participants or participant_key in pending.rules
        if participant_key not in await self._store.participants() and not unfinished:
            return {"status": "ignored", "reason": "no_consent"}
        # Evict first, then delete: no queued write or snapshot can recreate anything for this key. A repeated revoke
        # of a participant whose rows are gone completes what an earlier one could not (export files, key, rules).
        self._evict(participant_key)
        try:
            erased = await self._store.erase_participant(participant_key)
        finally:
            # The rules, the consent cache and the notification never depend on every file being deletable.
            rules_deleted = self._delete_participant_rules(participant_key)
            with contextlib.suppress(Exception):
                await self._refresh_consent_cache()
            await self._broadcast("revoked", participant_key)
        rules_left = participant_key in self._store.pending_erasure().rules
        return {
            "status": "ok",
            "erased": {
                "rows": erased.rows,
                "key_destroyed": erased.key_destroyed,
                "export_files": erased.export_files,
                "rules_deleted": rules_deleted,
                "export_files_left": erased.export_files_left,
                # True while something could not be deleted yet; Heliox retries it at start, status and every prune.
                "pending": not erased.complete or rules_left,
            },
        }

    # ── preferences ──────────────────────────────────────────────────────────

    async def _set_preference(self, request: PreferenceSetParams) -> dict:
        now = self._clock()
        scope_key = request.scope.participant_key
        if scope_key is not None and scope_key not in await self._store.participants():
            return {"status": "error", "reason": "unknown_participant", "message": "Unknown participant."}
        try:
            pref_model = PreferenceV1(
                pref_id=str(uuid.uuid4()),
                kind=request.kind,
                scope=request.scope,
                value=request.value,
                origin="explicit_user",
                created_at=now,
            )
            pref, notes = self._prefs_store.add(pref_model.to_json_dict())
        except PreferenceConflict as conflict:
            return self._conflict_reply(conflict)
        except PreferenceFileError:
            return {"status": "error", "message": "The co-adaptation rules file cannot be read."}
        await self._apply_rule_change(pref, now)
        return {
            "status": "ok",
            "preference": preference_to_dict(pref),
            "notes": list(notes),
            **self._refresh_fields(scope_key, request.scope.modality, now),
        }

    async def _retract_preference(self, request: PreferenceRetractParams) -> dict:
        now = self._clock()
        try:
            existing = self._prefs_store.get(request.pref_id)
            # retracted_at is exactly the issued_at of the reply's fallbacks: a reconstruction at that instant
            # (PreferenceSet.as_of, Preference.active_at) must already leave the rule out.
            if existing is None or existing.retracted_at is not None:
                return {"status": "error", "message": "Unknown rule"}
            if not self._prefs_store.retract(request.pref_id, at=now):
                return {"status": "error", "message": "Unknown rule"}
            pref = self._prefs_store.get(request.pref_id)
        except PreferenceFileError:
            return {"status": "error", "message": "The co-adaptation rules file cannot be read."}
        if pref is None:
            return {"status": "error", "message": "Unknown rule"}
        await self._apply_rule_change(pref, now)
        return {"status": "ok", **self._refresh_fields(pref.scope.participant_key, pref.scope.modality, now)}

    async def _apply_rule_change(self, pref: Preference, now: datetime) -> None:
        """Rule history, new PreferenceSet, stale marks, refresh_seq and broadcasts for one rule change (13.3)."""
        self._load_preferences()
        for participant_key in self._consenting_participants(now):
            if pref.scope.participant_key not in (None, participant_key):
                continue
            try:
                await self._store.put_rule(participant_key, pref)
            except Exception as exc:
                logger.warning(
                    "Co-adaptation rule history write failed for %s (%s)", participant_key, type(exc).__name__
                )
        marked: list[str] = []
        for participant_key, engine in self._engines.items():
            engine.set_preferences(self._prefs)
            # The change counts for every session whose current decisions are now stale, also when an earlier change
            # already marked them: the study's pause replacement (retract, then set) is two changes with two values.
            if engine.refresh_required():
                marked.append(participant_key)
        for participant_key in marked:
            seq = self._bump_refresh_seq(participant_key)
            await self._broadcast("preferences", participant_key, seq)
        if not marked:
            await self._broadcast("preferences", pref.scope.participant_key)

    def _refresh_fields(self, participant_key: str | None, modality: str | None, now: datetime) -> dict[str, Any]:
        if participant_key is None:
            return {"refresh_required": False, "refresh_seq": None, "fallbacks": None}
        engine = self._engines.get(participant_key)
        live = self._open_session_of(participant_key)
        modality = live.modality if live is not None else modality
        refresh_required = bool(engine is not None and live is not None and engine.refresh_required())
        fallbacks = None
        if engine is not None and modality is not None:
            fallbacks = engine.fallbacks(modality, now=now).to_dict()
        return {
            "refresh_required": refresh_required,
            "refresh_seq": self._refresh_seq.get(participant_key, 0),
            "fallbacks": fallbacks,
        }

    def _conflict_reply(self, conflict: PreferenceConflict) -> dict:
        ids = tuple(getattr(conflict, "conflicting_pref_ids", ()) or ())
        try:
            by_id = {pref.pref_id: pref for pref in self._prefs_store.entries()}
        except PreferenceFileError:
            by_id = {}
        return {
            "status": "error",
            "reason": "preference_conflict",
            "message": str(getattr(conflict, "message", "") or conflict),
            "conflicting": [{"pref_id": pref_id, "summary": self._rule_summary(by_id.get(pref_id))} for pref_id in ids],
        }

    @staticmethod
    def _rule_summary(pref: Preference | None) -> str:
        if pref is None:
            return "A rule that no longer exists"
        return f"{pref.summary()} ({pref.scope.describe()})"

    # ── learning reset and erase ─────────────────────────────────────────────

    async def _reset_learning(self, request: ParticipantParams) -> dict:
        participant_key = request.participant_key
        now = self._clock()
        consent, reason = await self._consent_for(participant_key, now)
        if consent is None:
            return {"status": "ignored", "reason": reason}
        session_ended = False
        live = self._open_session_of(participant_key)
        if live is not None:
            await self._end_live_session(live, now, "learning_reset", save=False)
            session_ended = True
        await self._store.mark_learning_reset(participant_key, now)
        await self._store.put_event(participant_key, "learning_reset", {}, now)
        self._engines.pop(participant_key, None)
        engine = await self._rebuild_unlocked(participant_key, consent, now)
        await self._broadcast("reset", participant_key)
        return {"status": "ok", "model": self._model_status(engine), "session_ended": session_ended}

    async def _erase(self, request: EraseParams) -> dict:
        participant_key = request.participant_key
        now = self._clock()
        consent, reason = await self._consent_for(participant_key, now)
        if consent is None:
            return {"status": "ignored", "reason": reason}
        session_ended = False
        live = self._open_session_of(participant_key)
        if live is not None:
            await self._end_live_session(live, now, "erased", save=False)
            session_ended = True
        # Evict the engine with its cached decisions and tables before rows are deleted.
        self._engines.pop(participant_key, None)
        self._veto_log.pop(participant_key, None)
        # An export an earlier erase-today could not delete is deleted now, also when today has no rows left.
        await self._finish_pending_erasures()
        today = day_index_for(now.astimezone(), self._view().day_boundary_hour)
        # erase_day deletes the model snapshot with the rows (no key needed), so a degraded erase never leaves state
        # learned from the erased trials behind; the next engine load rebuilds from the retained rows.
        result = await self._store.erase_day(participant_key, today)
        if self._keyring_ok():
            await self._store.put_event(
                participant_key,
                "erase_today",
                {"day_index": today, "rows": result.rows, "export_files": result.export_files},
                now,
            )
            await self._rebuild_unlocked(participant_key, consent, now)
        await self._broadcast("erased", participant_key)
        return {
            "status": "ok",
            "erased": {
                "rows": result.rows,
                "export_files": result.export_files,
                "export_files_left": result.export_files_left,
            },
            "session_ended": session_ended,
        }

    # ── results and export ───────────────────────────────────────────────────

    async def _results(self, request: ResultsParams) -> dict:
        participant_key = request.participant_key
        now = self._clock()
        consent, reason = await self._consent_for(participant_key, now)
        if consent is None:
            return {"status": "ignored", "reason": reason}
        view = self._view()
        day_index = day_index_for(now.astimezone(), view.day_boundary_hour) if request.scope == "today" else None
        rows = await self._store.results_rows(participant_key, day_index=day_index)
        engine = await self._engine(participant_key, consent, now)
        status = engine.status()
        calibration = await self._calibration(
            participant_key, consent, {session.session_id for session in rows.sessions}
        )
        latest = rows if rows.sessions else await self._store.results_rows(participant_key, day_index=None)
        modality = latest.sessions[-1].device.modality if latest.sessions else "mouse"
        candidates = [variant for variant in l1_arms(modality) if variant != ASSISTANCE_OFF]
        eligible, excluded = self._prefs.apply(
            DecisionLevel.BLOCK_REGIME,
            candidates,
            context="study",
            modality=modality,
            participant_key=participant_key,
            interruptions_used=0,
        )
        excluded_out = {variant: _enum_value(why) for variant, why in excluded.items()}
        eligible_out = list(eligible)
        if status.calibration_suspended:
            for variant in list(eligible_out):
                if acting(variant):
                    eligible_out.remove(variant)
                    excluded_out[variant] = "calibration_suspended"
        return {
            "status": "ok",
            "evidence_kind": EvidenceKind.LIVE_HUMAN.value,
            "claim": "Your data on this device; not a general result.",
            "per_regime": self._per_regime(rows.trials, rows.burden),
            "calibration": calibration,
            "probe_trend": self._probe_trend(rows.sessions, rows.trials),
            "drift_flags": list(status.drift_flags),
            "allowed_now": {"eligible": eligible_out, "excluded": excluded_out},
            "predictions_now": self._predictions_now(engine, latest, now),
        }

    async def _calibration(
        self, participant_key: str, consent: CoAdaptationConsentV1, session_ids: set[str]
    ) -> dict[str, Any]:
        """Prequential calibration from a temporary replay; the live engine is never touched (spec 13.3)."""
        monitor = CalibrationMonitor()

        def observe(engine: CoAdaptationEngine, stored: StoredTrial) -> None:
            if stored.verification != "ok" or stored.trial.session_id not in session_ids or stored.labels is None:
                return
            if stored.trial.task == "idle":
                return
            prediction = engine.prequential_prediction(stored)
            labels = stored.labels
            monitor.add_binary("ua", prediction["ua"], bool(labels.ua))
            monitor.add_binary("complete", prediction["complete"], bool(labels.complete))
            # time_ms is the time to the correct commit, or the trial limit when the time is censored (spec 7.1).
            monitor.add_time(
                prediction["log_time_mean"],
                prediction["log_time_sd"],
                math.log(max(float(labels.time_ms), 1.0) / 1000.0),
                censored=bool(labels.censored),
            )

        await self._replay_engine(participant_key, self._base_settings(consent), observer=observe)
        return monitor.report()

    def _per_regime(self, trials: Iterable[StoredTrial], burden: Iterable[BurdenReport]) -> list[dict[str, Any]]:
        by_regime: dict[str, list[StoredTrial]] = {}
        block_regime: dict[tuple[str, int], str] = {}
        for stored in trials:
            trial = stored.trial
            by_regime.setdefault(trial.applied_regime, []).append(stored)
            if trial.task != "idle":
                block_regime.setdefault((trial.session_id, trial.block_index), trial.applied_regime)
        ratings: dict[str, dict[str, list[float]]] = {}
        for report in burden:
            if report.scope != "block" or report.block_index is None:
                continue
            regime = block_regime.get((report.session_id, report.block_index))
            if regime is None:
                continue
            for item in ("effort", "intrusiveness", "control"):
                value = getattr(report, item)
                if value is not None:
                    ratings.setdefault(regime, {}).setdefault(item, []).append(float(value))
        out: list[dict[str, Any]] = []
        for regime in sorted(by_regime):
            rows = by_regime[regime]
            active = [s for s in rows if s.labels is not None and s.trial.task != "idle"]
            idle = [s for s in rows if s.labels is not None and s.trial.task == "idle"]
            times = [
                float(_label(s.labels, "time_ms"))
                for s in active
                if _label(s.labels, "complete") and _label(s.labels, "time_ms") is not None
            ]
            idle_minutes = sum(float(s.labels.idle_minutes) for s in idle)
            ua_rate = _mean([1.0 if _label(s.labels, "ua") else 0.0 for s in active])
            verification = Counter(_enum_value(s.verification) for s in rows)
            out.append(
                {
                    "variant": regime,
                    "trials": len(rows),
                    "completion": _mean([1.0 if _label(s.labels, "complete") else 0.0 for s in active]),
                    "ua_per_100": None if ua_rate is None else 100.0 * ua_rate,
                    "idle_ua_per_min": (
                        sum(int(s.labels.idle_commits) for s in idle) / idle_minutes if idle_minutes > 0 else None
                    ),
                    "corrections_per_trial": _mean([float(s.trial.corrections) for s in active]),
                    "median_time_ms": float(statistics.median(times)) if times else None,
                    "p95_time_ms": _percentile(times, 0.95),
                    "throughput_bits_s": pointing_throughput(rows),
                    "burden": {
                        item: _mean(ratings.get(regime, {}).get(item, []))
                        for item in ("effort", "intrusiveness", "control")
                    },
                    "verification": {value: verification.get(value, 0) for value in _VERIFICATION_VALUES},
                }
            )
        return out

    @staticmethod
    def _probe_trend(sessions: Iterable[SessionRow], trials: Iterable[StoredTrial]) -> list[dict[str, Any]]:
        trials = list(trials)
        out: list[dict[str, Any]] = []
        for session in sessions:
            probe_blocks = {block.index for block in session.plan.blocks if block.phase == "probe"}
            if not probe_blocks:
                continue
            probe = [
                s
                for s in trials
                if s.trial.session_id == session.session_id
                and s.trial.block_index in probe_blocks
                and s.trial.task != "idle"
            ]
            if not probe:
                continue
            labelled = [s for s in probe if s.labels is not None]
            ua_rate = _mean([1.0 if _label(s.labels, "ua") else 0.0 for s in labelled])
            out.append(
                {
                    "session_number": session.session_number,
                    "throughput_bits_s": pointing_throughput(probe),
                    "ua_per_100": None if ua_rate is None else 100.0 * ua_rate,
                }
            )
        return out

    @staticmethod
    def _predictions_now(engine: CoAdaptationEngine, rows: Any, now: datetime) -> dict[str, Any]:
        """Predictions over the layouts of the participant's most recent recorded block (spec 13.3)."""
        active = [s for s in rows.trials if s.trial.task != "idle"]
        if not active or not rows.sessions:
            return {}
        last = active[-1].trial
        session = next((s for s in rows.sessions if s.session_id == last.session_id), None)
        if session is None:
            return {}
        layouts = tuple(
            s.trial.layout
            for s in active
            if s.trial.session_id == last.session_id and s.trial.block_index == last.block_index
        )[:8]
        phase = next((block.phase for block in session.plan.blocks if block.index == last.block_index), "B")
        block = BlockContext(
            session_id=session.session_id,
            block_index=last.block_index,
            phase=phase,
            task=last.task,
            layouts=layouts,
            device=session.device,
            clock=ClockContext(
                day_index=session.day_index, session_minute=0.0, trials_today=0, days_since_last_session=None
            ),
            previous_regime=None,
            hand_quality_recent=None,
        )
        try:
            predictions = engine.predictions_for_block(block, now=now)
        except Exception as exc:
            logger.warning("Co-adaptation predictions for results failed (%s)", type(exc).__name__)
            return {}
        return {variant: prediction.to_dict() for variant, prediction in predictions.items()}

    async def _export(self, request: ParticipantParams) -> dict:
        participant_key = request.participant_key
        now = self._clock()
        consent, reason = await self._consent_for(participant_key, now)
        if consent is None:
            return {"status": "ignored", "reason": reason}
        if not consent.covers(CoAdaptationPurpose.RESEARCH_EXPORT):
            return self._refused("purpose_missing")
        path, digest, size = await self._store.export(participant_key, now)
        return {"status": "ok", "path": str(path), "sha256": digest, "bytes": size}

    # ── study sessions ───────────────────────────────────────────────────────

    async def _begin_session(self, request: SessionBeginParams) -> dict:
        view = self._view()
        participant_key = request.participant_key
        if not view.enabled:
            return self._refused("disabled")
        now = self._clock()
        await self._prune(now)
        consent, reason = await self._consent_for(participant_key, now)
        if consent is None:
            return self._refused(reason or "no_consent")
        if not consent.covers(CoAdaptationPurpose.SELECTION_STUDY):
            return self._refused("purpose_missing")
        if request.consent_text_sha256 != consent.consent_text_sha256:
            return self._refused("consent_text_changed")
        protocol_sha = study_protocol_sha256()
        if (
            request.protocol_id != STUDY_PROTOCOL_ID
            or consent.protocol_id != STUDY_PROTOCOL_ID
            or consent.protocol_sha256 != protocol_sha
        ):
            return self._refused("protocol_changed")
        if not await self._clear_stale_session(participant_key, now):
            # One open session per participant (14.3). _clear_stale_session has already ended an open session that
            # went quiet (the UI lost it), so only a session still in use is refused.
            return self._refused("session_active")
        if self._prefs_error:
            return self._refused("preferences_unreadable")
        effective = self._effective(consent, view)
        device_in = request.device
        modality = device_in.modality
        if (modality == "hand_pointer" or device_in.hand_pointer) and not effective.hand:
            return self._refused("hand_pointer_not_allowed")

        # Decision cap (spec 10.6), only here and before the new session opens; a deletion forces a rebuild.
        pruned = await self._store.apply_decision_cap(participant_key)
        if pruned:
            self._engines.pop(participant_key, None)
            self._veto_log.pop(participant_key, None)
            await self._store.put_event(participant_key, "prune", {"sessions": pruned}, now)
        engine = await self._engine(participant_key, consent, now)

        accuracy = request.gaze_check.accuracy if request.gaze_check is not None else None
        gaze_used = effective.gaze and accuracy is not None and accuracy >= GAZE_ACCURACY_MIN
        hand_used = effective.hand and modality == "hand_pointer"
        settings, deviations = self._derive_settings(effective, view, gaze_used=gaze_used)
        # The only place the machine's time zone is read (spec 8); the engine and replay use the stored integer.
        day_index = day_index_for(now.astimezone(), view.day_boundary_hour)

        status = engine.status()
        suspended = bool(
            status.calibration_suspended and status.last_day_index == day_index and settings.learn_across_sessions
        )
        practice = compute_fallbacks(
            self._prefs, modality, participant_key=participant_key, calibration_suspended=suspended, now=now
        ).practice
        if practice is None:
            return self._refused("no_allowed_regime")

        infos = await self._store.sessions_summary(participant_key)
        session_number = 1 + sum(1 for info in infos if info.trials > 0)
        days_since = (now - infos[-1].started_at).total_seconds() / 86400.0 if infos else None
        ranking: tuple[str, ...] = ()
        if settings.adaptive_allowed and session_number >= 2:
            # Comparator ranking from session 1 (adaptive_assistance purpose), also without cross-session learning.
            first: SessionRows | None = await self._store.session_rows(participant_key, 1)
            if first is not None:
                ranking = tuple(rank_fixed_regimes(first.trials, first.burden, first.decisions, modality=modality))
        plan = build_session_plan(
            session_number=session_number,
            adaptive_allowed=settings.adaptive_allowed,
            comparator_available=bool(ranking),
            plan_seed=plan_seed(participant_key, session_number),
            burden_every_blocks=view.burden_prompt_every_blocks,
            n_arms=len(l1_arms(modality)),
        )
        device = dataclasses.replace(device_in.to_core(), gaze_available=gaze_used, hand_pointer=hand_used)
        session_id = str(uuid.uuid4())
        row = SessionRow(
            session_id=session_id,
            participant_key=participant_key,
            context="study",
            day_index=day_index,
            session_number=session_number,
            protocol_id=STUDY_PROTOCOL_ID,
            protocol_sha256=protocol_sha,
            evidence_kind=EvidenceKind.LIVE_HUMAN,
            started_at=now,
            ended_at=None,
            end_reason=None,
            device=device,
            plan=plan,
            layout_seed=secrets.randbelow(LAYOUT_SEED_LIMIT),
            seed_salt=secrets.token_hex(16),
            gaze_check_accuracy=accuracy if effective.gaze else None,
            comparator_ranking=ranking,
            settings=dataclasses.asdict(settings),
            protocol_deviations=deviations,
            versions=self._versions(),
        )
        await self._store.put_session(row)
        try:
            engine.begin_session(row)
        except Exception:
            self._engines.pop(participant_key, None)
            with contextlib.suppress(Exception):
                await self._store.close_session(session_id, now, "interrupted")
            raise
        self._sessions[session_id] = _LiveSession(
            session_id=session_id,
            participant_key=participant_key,
            row=row,
            modality=modality,
            gaze_used=gaze_used,
            hand_used=hand_used,
            days_since_last_session=days_since,
            last_activity=now,
        )
        fallbacks = engine.fallbacks(modality, now=now)
        if fallbacks.practice is None:
            # The engine's own view after begin_session wins; nothing runs without an allowed regime.
            await self._end_live_session(self._sessions[session_id], now, "interrupted")
            return self._refused("no_allowed_regime")
        spread: dict[str, float] = {}
        floors: dict[str, float] = {}
        if modality != "keyboard":
            r_hat = engine.spread().get(modality)
            if r_hat is not None:
                spread[modality] = float(r_hat)
                floors[modality] = FLOOR_PX[modality]
        tau = float(settings.ambiguity_threshold)
        return {
            "status": "ok",
            "session_id": session_id,
            "participant_key": participant_key,
            "day_index": day_index,
            "session_number": session_number,
            "plan": _plain(plan),
            "layout_seed": row.layout_seed,
            "spread": spread,
            "floors": floors,
            "thresholds": {"ambiguity": tau, "bins": _bins(tau)},
            "gaze_prior": gaze_used,
            "hand_pointer": hand_used,
            "burden_prompt_every_blocks": view.burden_prompt_every_blocks,
            "evidence_kind": EvidenceKind.LIVE_HUMAN.value,
            "fallbacks": fallbacks.to_dict(),
            "refresh_seq": self._refresh_seq.get(participant_key, 0),
        }

    async def _resume_session(self, request: SessionRefParams) -> dict:
        live = self._sessions.get(request.session_id)
        if live is None:
            return {"status": "ignored", "reason": self._missing_session_reason(request.session_id)}
        now = self._clock()
        consent, reason = await self._live_consent(live, now)
        if consent is None:
            return {"status": "ignored", "reason": reason}
        live.last_activity = now
        engine = self._engines[live.participant_key]
        return {
            "status": "ok",
            "refresh_required": bool(engine.refresh_required()),
            "refresh_seq": self._refresh_seq.get(live.participant_key, 0),
            "fallbacks": engine.fallbacks(live.modality, now=now).to_dict(),
        }

    async def _recommend(self, request: RecommendParams) -> dict:
        level = request.decision_level()
        live = self._sessions.get(request.session_id)
        if not self._view().enabled:
            return self._baseline(level, "disabled")
        if live is None:
            return self._baseline(level, self._missing_session_reason(request.session_id))
        now = self._clock()
        consent, reason = await self._live_consent(live, now)
        if consent is None:
            return self._baseline(level, reason or "no_consent")
        live.last_activity = now
        # Checked before the engine decides: a decision the store would refuse must never touch session state.
        if request.decision_id in live.decision_ids or await self._store.decision_exists(request.decision_id):
            return {"status": "error", "message": "Duplicate decision_id"}
        try:
            context = request.parsed_context()
        except PayloadTooLarge as exc:
            return {"status": "error", "reason": "payload_too_large", "message": str(exc)}
        except ValidationError as exc:
            return {"status": "error", "message": validation_message(exc)}
        engine = self._engines[live.participant_key]
        row = live.row
        try:
            if isinstance(context, OfferContextV1):
                decision = engine.decide_offer(
                    decision_id=request.decision_id,
                    context=context.to_core(session_id=live.session_id, modality=live.modality),
                    now=now,
                )
            else:
                block = self._block_context(live, context)
                if block is None:
                    return {"status": "error", "message": "The block_index is not part of this session's plan."}
                if isinstance(context, AmbiguityTableContextV1):
                    if context.block_decision_id not in live.l1_decisions:
                        return {"status": "error", "message": "Unknown block_decision_id"}
                    decision = engine.decide_ambiguity_table(
                        decision_id=request.decision_id,
                        block_decision_id=context.block_decision_id,
                        block=block,
                        start_trial_index=context.start_trial_index,
                        trials_in_block=context.trials_in_block,
                        now=now,
                    )
                else:
                    decision = engine.decide_block(decision_id=request.decision_id, block=block, now=now)
        except NoAllowedRegime:
            return self._refused("no_allowed_regime")
        except ValueError as exc:
            # The core rejects contexts it cannot decide on; its messages name the rule, never a value.
            return {"status": "error", "message": str(exc)}
        record = DecisionRecord.from_decision(
            decision, session_id=row.session_id, seq=live.take_seq(), behavior="policy", created_at=now
        )
        supersedes = None
        if isinstance(context, AmbiguityTableContextV1):
            # A new table for the same block closes the old one (spec 9.3), in the same transaction.
            supersedes = live.tables.get(context.block_decision_id)
        try:
            await self._store.put_decision(record, participant_key=live.participant_key, supersedes=supersedes)
        except BaseException:
            # decide_* already registered the decision, consumed the C5 vetoes of its level and advanced the
            # permuted round. A decision the person never receives must not stay applied: drop the session and its
            # engine, so the model is rebuilt from the stored rows (as record_trial does).
            await self._abandon_session(live, now)
            raise
        live.decision_ids.add(request.decision_id)
        if level == DecisionLevel.BLOCK_REGIME:
            live.l1_decisions.add(request.decision_id)
        elif level == DecisionLevel.AMBIGUITY_RESPONSE:
            assert isinstance(context, AmbiguityTableContextV1)
            live.tables[context.block_decision_id] = request.decision_id
        return {"status": "ok", "logged": True, "recommendation": decision.to_dict()}

    def _block_context(
        self, live: _LiveSession, context: BlockRegimeContextV1 | AmbiguityTableContextV1
    ) -> BlockContext | None:
        plan_block = next((block for block in live.row.plan.blocks if block.index == context.block_index), None)
        if plan_block is None:
            return None
        return context.to_core(
            session_id=live.session_id,
            phase=plan_block.phase,
            task=plan_block.task,
            device=live.row.device,
            day_index=live.row.day_index,
            days_since_last_session=live.days_since_last_session,
        )

    async def _record_trial(self, request: RecordTrialParams) -> dict:
        live = self._sessions.get(request.session_id)
        if live is None:
            return {"status": "ignored", "reason": self._missing_session_reason(request.session_id)}
        if not self._view().enabled:
            return {"status": "ignored", "reason": "disabled"}
        now = self._clock()
        consent, reason = await self._live_consent(live, now)
        if consent is None:
            return {"status": "ignored", "reason": reason}
        live.last_activity = now
        trial: TrialV1 = request.trial
        cached = live.trial_replies.get(trial.trial_id)
        if cached is not None:
            # Idempotent per trial_id: the stored result, with the refresh_seq the UI already applied or passed.
            return dict(cached)
        participant_key = live.participant_key
        engine = self._engines[participant_key]
        owner = await self._store.trial_owner(trial.trial_id)
        if owner is not None:
            # Stored but not cached: a repeat is answered from the row and never learned a second time (spec 14.3).
            if owner != (participant_key, live.session_id):
                return {"status": "error", "message": "Duplicate trial_id"}
            stored_row = await self._store.read_trial(participant_key, trial.trial_id)
            if stored_row is None:
                return {"status": "error", "message": "Duplicate trial_id"}
            return self._cache_trial_reply(live, engine, trial.trial_id, stored_row, now)
        try:
            check_trial_spread(trial, live.modality)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        trial = without_ineffective_fields(trial, gaze_effective=live.gaze_used, hand_effective=live.hand_used)
        record = trial.to_core(session_id=live.session_id, recorded_at=now, seq=live.take_seq())
        observation = engine.observe_trial(record)
        try:
            # The trial, its alarm events and the relabelling of every level that fell back are one transaction.
            await self._store.put_trial(
                observation, participant_key=participant_key, relabels=self._fallback_relabels(live, trial)
            )
        except BaseException:
            # The engine learned from a row that is not stored: drop the session so the model is rebuilt from rows.
            await self._abandon_session(live, now)
            raise
        # Everything below runs after the commit and cannot lose the trial: the reply is cached and refresh_seq raised
        # first, so a repeat of this trial_id returns this result and never observes the trial again.
        if observation.refresh_required:
            refresh_seq: int | None = self._bump_refresh_seq(participant_key)
        else:
            refresh_seq = None
        for event in observation.events:
            await self._broadcast(event.kind, participant_key, refresh_seq)
        variant = vetoed_variant(record) if trial.veto is not None else None
        if variant is not None:
            live.veto_counts[variant] += 1
            self._note_veto(participant_key, record.recorded_at, variant)
        reply = self._cache_trial_reply(live, engine, trial.trial_id, observation.stored, now)
        if trial.veto is not None:
            try:
                await self._announce_new_proposals(participant_key, consent, now)
            except Exception as exc:
                # A proposal is only a suggestion; the next status call recomputes it.
                logger.warning("Co-adaptation proposal check failed for %s (%s)", participant_key, type(exc).__name__)
        return reply

    def _cache_trial_reply(
        self, live: _LiveSession, engine: CoAdaptationEngine, trial_id: str, stored: StoredTrial, now: datetime
    ) -> dict:
        refresh_required = bool(engine.refresh_required())
        reply = {
            "status": "recorded",
            "verification": _enum_value(stored.verification),
            "labels": _labels_reply(stored.labels),
            "preference_violation": bool(stored.preference_violations),
            "refresh_required": refresh_required,
            "refresh_seq": self._refresh_seq.get(live.participant_key, 0),
            "fallbacks": engine.fallbacks(live.modality, now=now).to_dict() if refresh_required else None,
        }
        live.trial_replies[trial_id] = reply
        return dict(reply)

    @staticmethod
    def _fallback_relabels(live: _LiveSession, trial: TrialV1) -> list[FallbackRelabel]:
        """The decision of each level that fell back in the UI, and no other (spec 9.7).

        A stale_table fallback names the table requested after a veto, which had not arrived yet. When it arrives the
        running trial adopts it, so a table this trial also applied from its draws stays a policy decision."""
        relabels: list[FallbackRelabel] = []
        applied_tables = {
            activation.l2.decision_id
            for activation in trial.activations
            if activation.l2 is not None and activation.l2.source == "table"
        }
        if trial.l1_fallback_reason is not None and trial.regime_decision_id in live.decision_ids:
            relabels.append(
                FallbackRelabel(
                    decision_id=trial.regime_decision_id,
                    applied=trial.applied_regime,
                    fallback_reason=trial.l1_fallback_reason,
                )
            )
        relabelled: set[str] = set()
        for activation in trial.activations:
            l2 = activation.l2
            if l2 is None or l2.source != "fallback" or l2.fallback_reason is None:
                continue
            if l2.fallback_reason == "stale_table" and l2.decision_id in applied_tables:
                continue
            if l2.decision_id in relabelled or l2.decision_id not in live.decision_ids:
                continue
            relabelled.add(l2.decision_id)
            relabels.append(
                FallbackRelabel(decision_id=l2.decision_id, applied=None, fallback_reason=l2.fallback_reason)
            )
        for offer in trial.offers:
            if offer.fallback_reason is None or offer.decision_id not in live.decision_ids:
                continue
            if offer.decision_id in relabelled:
                continue
            relabelled.add(offer.decision_id)
            relabels.append(
                FallbackRelabel(
                    decision_id=offer.decision_id, applied=offer.variant, fallback_reason=offer.fallback_reason
                )
            )
        return relabels

    async def _report_burden(self, request: ReportBurdenParams) -> dict:
        live = self._sessions.get(request.session_id)
        if live is None:
            return {"status": "ignored", "reason": self._missing_session_reason(request.session_id)}
        if not self._view().enabled:
            return {"status": "ignored", "reason": "disabled"}
        now = self._clock()
        consent, reason = await self._live_consent(live, now)
        if consent is None:
            return {"status": "ignored", "reason": reason}
        live.last_activity = now
        report = request.report.to_core(
            report_id=str(uuid.uuid4()), session_id=live.session_id, recorded_at=now, seq=live.take_seq()
        )
        await self._store.put_burden(report, participant_key=live.participant_key)
        try:
            self._engines[live.participant_key].observe_burden(report)
        except Exception:
            await self._abandon_session(live, now)
            raise
        return {"status": "recorded"}

    async def _end_session(self, request: SessionEndParams) -> dict:
        live = self._sessions.get(request.session_id)
        if live is None:
            return {"status": "ignored", "reason": self._missing_session_reason(request.session_id)}
        now = self._clock()
        consent, reason = await self._live_consent(live, now)
        if consent is None:
            return {"status": "ignored", "reason": reason}
        summary = await self._end_live_session(live, now, request.reason)
        return {"status": "ok", "summary": _plain(summary) if summary is not None else None}

    # ── engines, sessions and replay ─────────────────────────────────────────

    def _open_session_of(self, participant_key: str) -> _LiveSession | None:
        for live in self._sessions.values():
            if live.participant_key == participant_key:
                return live
        return None

    async def _clear_stale_session(self, participant_key: str, now: datetime) -> bool:
        """True when the participant has no open session (any more).

        An open session without a study call for STALE_SESSION_AFTER is one the UI lost (a window reload or a webview
        crash keeps no session id), so it is ended as abandoned instead of blocking session_begin and consent updates
        until the daemon restarts. A session in use is left alone and the caller refuses with session_active.
        """
        live = self._open_session_of(participant_key)
        if live is None:
            return True
        if now - live.last_activity < STALE_SESSION_AFTER:
            return False
        try:
            await self._end_live_session(live, now, "abandoned")
        except Exception as exc:
            logger.warning(
                "Co-adaptation could not end a stale session for %s (%s)", participant_key, type(exc).__name__
            )
            self._drop_session(live)
        return True

    async def _engine(self, participant_key: str, consent: CoAdaptationConsentV1, now: datetime) -> CoAdaptationEngine:
        engine = self._engines.get(participant_key)
        if engine is not None:
            return engine
        return await self._load_engine(participant_key, consent, now)

    async def _load_engine(
        self, participant_key: str, consent: CoAdaptationConsentV1, now: datetime
    ) -> CoAdaptationEngine:
        """Restore from the stored snapshot when rows and versions match, else rebuild (spec 10.6, 10.8)."""
        if self._open_session_of(participant_key) is not None:
            raise RuntimeError("an engine is never loaded while its participant has an open session")
        # A session a failed restart clean-up left open is closed before anything is replayed.
        for owner, session_id in await self._store.open_sessions():
            if owner == participant_key and session_id not in self._sessions:
                await self._store.close_session(session_id, now, "daemon_restart")
                self._closed_sessions.add(session_id)
        settings = self._base_settings(consent)
        snapshot = await self._store.load_snapshot(participant_key)
        if snapshot is not None:
            state, rows_hash, model_version, prior_version = snapshot
            if (
                model_version == MODEL_VERSION
                and prior_version == PRIOR_VERSION
                and rows_hash == await self._store.rows_hash(participant_key)
            ):
                try:
                    engine = CoAdaptationEngine.restore(
                        state, participant_key=participant_key, settings=settings, preferences=self._prefs
                    )
                except Exception as exc:
                    logger.warning("Co-adaptation snapshot rejected for %s (%s)", participant_key, type(exc).__name__)
                else:
                    self._engines[participant_key] = engine
                    return engine
        return await self._rebuild_unlocked(participant_key, consent, now)

    async def _rebuild_unlocked(
        self, participant_key: str, consent: CoAdaptationConsentV1, now: datetime
    ) -> CoAdaptationEngine:
        if self._open_session_of(participant_key) is not None:
            raise RuntimeError("a rebuild never runs while the participant has an open session")
        engine = await self._replay_engine(participant_key, self._base_settings(consent))
        self._engines[participant_key] = engine
        await self._save_snapshot(participant_key, engine, now)
        return engine

    async def _replay_engine(
        self,
        participant_key: str,
        settings: EngineSettings,
        *,
        observer: Callable[[CoAdaptationEngine, StoredTrial], None] | None = None,
    ) -> CoAdaptationEngine:
        """A fresh engine at the prior with every retained row replayed in the order of spec 10.8."""
        engine = CoAdaptationEngine(participant_key=participant_key, settings=settings, preferences=self._prefs)
        current: SessionRow | None = None
        async for item in self._store.replay_rows(participant_key):
            if isinstance(item, SessionRow):
                self._finish_replayed_session(engine, current)
                engine.begin_session(item)
                current = item
            elif isinstance(item, StoredTrial):
                if observer is not None:
                    observer(engine, item)
                engine.replay_trial(item)
            elif isinstance(item, BurdenReport):
                engine.observe_burden(item)
            elif isinstance(item, EngineEvent):
                engine.apply_event(item)
        self._finish_replayed_session(engine, current)
        return engine

    @staticmethod
    def _finish_replayed_session(engine: CoAdaptationEngine, row: SessionRow | None) -> None:
        if row is not None and row.ended_at is not None:
            engine.end_session(ended_at=row.ended_at, reason=row.end_reason or "completed")

    async def _save_snapshot(self, participant_key: str, engine: CoAdaptationEngine, now: datetime) -> None:
        rows_hash = await self._store.rows_hash(participant_key)
        await self._store.save_snapshot(participant_key, engine.snapshot(), rows_hash, now)

    async def _end_live_session(self, live: _LiveSession, now: datetime, reason: str, *, save: bool = True) -> Any:
        await self._store.close_session(live.session_id, now, reason)
        engine = self._engines.get(live.participant_key)
        summary = engine.end_session(ended_at=now, reason=reason) if engine is not None else None
        self._sessions.pop(live.session_id, None)
        self._closed_sessions.add(live.session_id)
        if save and engine is not None:
            await self._save_snapshot(live.participant_key, engine, now)
        return summary

    def _drop_session(self, live: _LiveSession) -> None:
        self._sessions.pop(live.session_id, None)
        self._closed_sessions.add(live.session_id)
        self._engines.pop(live.participant_key, None)

    async def _abandon_session(self, live: _LiveSession, now: datetime) -> None:
        self._drop_session(live)
        with contextlib.suppress(Exception):
            await self._store.close_session(live.session_id, now, "interrupted")

    def _evict(self, participant_key: str) -> None:
        """Remove every in-memory trace of a participant before its rows are deleted (spec 10.7)."""
        self._engines.pop(participant_key, None)
        for session_id, live in list(self._sessions.items()):
            if live.participant_key == participant_key:
                self._sessions.pop(session_id, None)
                self._gone_sessions[session_id] = "no_consent"
        self._known_proposals.pop(participant_key, None)
        self._veto_log.pop(participant_key, None)
        self._gone.add(participant_key)

    def _missing_session_reason(self, session_id: str) -> str:
        if session_id in self._gone_sessions:
            return self._gone_sessions[session_id]
        if session_id in self._closed_sessions:
            return "session_closed"
        if not self.has_active_consent():
            return "no_consent"
        return "unknown_session"

    async def _close_leftover_sessions(self, now: datetime) -> None:
        if not self._store.is_open:
            return
        consenting = set(self._consenting_from(await self._store.consents(), now))
        for participant_key, session_id in await self._store.open_sessions():
            if participant_key not in consenting:
                continue
            try:
                await self._store.close_session(session_id, now, "daemon_restart")
                self._closed_sessions.add(session_id)
            except Exception as exc:
                logger.warning(
                    "Co-adaptation could not close a session for %s (%s)", participant_key, type(exc).__name__
                )

    # ── consent bookkeeping and pruning ──────────────────────────────────────

    async def _consent_for(
        self, participant_key: str, now: datetime
    ) -> tuple[CoAdaptationConsentV1 | None, str | None]:
        if participant_key in self._gone:
            return None, "no_consent"
        for record, revoked_at, replaced_at in reversed(await self._store.consents()):
            if record.participant_key != participant_key or replaced_at is not None:
                continue
            if revoked_at is not None:
                return None, "no_consent"
            if now >= record.expires_at:
                return None, "consent_expired"
            return record, None
        return None, "no_consent"

    async def _live_consent(self, live: _LiveSession, now: datetime) -> tuple[CoAdaptationConsentV1 | None, str | None]:
        """Consent check for a call on an open session; an expired consent is erased at once (spec 10.7)."""
        consent, reason = await self._consent_for(live.participant_key, now)
        if consent is None and reason == "consent_expired":
            await self._prune(now)
        elif consent is not None and not consent.covers(CoAdaptationPurpose.SELECTION_STUDY):
            return None, "purpose_missing"
        return consent, reason

    @staticmethod
    def _consenting_from(
        consents: list[tuple[CoAdaptationConsentV1, datetime | None, datetime | None]], now: datetime
    ) -> list[str]:
        keys: list[str] = []
        for record, revoked_at, replaced_at in consents:
            if revoked_at is None and replaced_at is None and now < record.expires_at:
                keys.append(record.participant_key)
        return keys

    def _consenting_participants(self, now: datetime) -> list[str]:
        return [
            record.participant_key
            for record in self._active_consents
            if record.participant_key not in self._gone and now < record.expires_at
        ]

    async def _refresh_consent_cache(self) -> None:
        self._active_consents = [
            record
            for record, revoked_at, replaced_at in await self._store.consents()
            if revoked_at is None and replaced_at is None
        ]

    async def _prune(self, now: datetime) -> list[str]:
        """Erase every participant whose consent expired, evicting its in-memory state first (spec 10.7).

        Then retries the erasure work an earlier revoke, expiry or erase could not finish (pending-erasure list).
        """
        erased: list[str] = []
        if self._store.is_open:
            consenting = set(self._consenting_from(await self._store.consents(), now))
            doomed = [key for key in await self._store.participants() if key not in consenting]
            if doomed:
                for participant_key in doomed:
                    self._evict(participant_key)
                erased = await self._store.prune(now)
                for participant_key in erased:
                    self._delete_participant_rules(participant_key)
                    await self._broadcast("expired", participant_key)
                await self._refresh_consent_cache()
        await self._finish_pending_erasures()
        return erased

    async def _finish_pending_erasures(self) -> None:
        """Complete erasures a file lock, a locked keyring or an unreadable rules file left unfinished (spec 10.7).

        A listed participant is evicted and erased again (rows, export files, keyring key), single export files are
        deleted, and participant-scoped rules are deleted from the preferences file. Whatever still fails stays
        listed for the next start, status call or prune. Never raises.
        """
        pending = self._store.pending_erasure()
        if pending.empty:
            return
        for participant_key in pending.participants:
            self._evict(participant_key)
            try:
                await self._store.erase_participant(participant_key)
            except Exception as exc:
                logger.warning("Co-adaptation pending erasure failed for %s (%s)", participant_key, type(exc).__name__)
        if pending.files:
            self._store.retry_pending_files()
        for participant_key in self._store.pending_erasure().rules:
            self._delete_participant_rules(participant_key)
        with contextlib.suppress(Exception):
            await self._refresh_consent_cache()

    async def _prune_loop(self) -> None:
        while True:
            await asyncio.sleep(PRUNE_INTERVAL_S)
            try:
                await self._locked(lambda: self._prune(self._clock()))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Co-adaptation scheduled prune failed (%s)", type(exc).__name__)

    def _delete_participant_rules(self, participant_key: str) -> int:
        """Hard-delete the participant's scoped rules; on failure they stay on the pending-erasure list."""
        try:
            deleted = self._prefs_store.delete_for_participant(participant_key)
        except Exception as exc:
            logger.warning("Co-adaptation could not delete the rules of %s (%s)", participant_key, type(exc).__name__)
            return 0
        self._store.rules_erased(participant_key)
        with contextlib.suppress(Exception):
            self._load_preferences()
        return deleted

    # ── proposals ────────────────────────────────────────────────────────────

    async def _proposals_for(self, participant_key: str, consent: CoAdaptationConsentV1, now: datetime) -> list[str]:
        """Variants vetoed three times: across sessions within 7 days only with cross-session learning (9.7)."""
        counts: Counter[str] = Counter()
        if self._effective(consent, self._view()).cross_session:
            since = now - PROPOSAL_WINDOW
            for recorded_at, variant in await self._vetoes_since(participant_key, since):
                if recorded_at >= since:
                    counts[variant] += 1
        else:
            live = self._open_session_of(participant_key)
            if live is not None:
                counts = Counter(live.veto_counts)
        return sorted(
            variant
            for variant, n in counts.items()
            if n >= PROPOSAL_VETOES and not self._already_forbidden(participant_key, variant)
        )

    async def _vetoes_since(self, participant_key: str, since: datetime) -> list[tuple[datetime, str]]:
        """(recorded_at, vetoed variant) of the participant's trials since ``since``, cached per participant.

        The first call reads only the trials of the proposal window (a narrow query, decrypted off the event loop);
        record_trial then appends each new veto, and every deletion of trials drops the cache. A status call or a
        "Not this time" therefore never decrypts the participant's whole history.
        """
        cached = self._veto_log.get(participant_key)
        if cached is None or since < cached[0]:
            trials = await self._store.trials_since(participant_key, since)
            entries = [
                (stored.trial.recorded_at, variant)
                for stored in trials
                if (variant := vetoed_variant(stored.trial)) is not None
            ]
            cached = (since, entries)
            self._veto_log[participant_key] = cached
        return cached[1]

    def _note_veto(self, participant_key: str, recorded_at: datetime, variant: str) -> None:
        cached = self._veto_log.get(participant_key)
        if cached is not None:
            cached[1].append((recorded_at, variant))

    def _already_forbidden(self, participant_key: str, variant: str) -> bool:
        family = variant.split(":", 1)[0]
        try:
            entries = self._prefs_store.entries()
        except PreferenceFileError:
            return False
        for pref in entries:
            if pref.retracted_at is not None or _enum_value(pref.kind) != "forbid_variant":
                continue
            if pref.scope.participant_key not in (None, participant_key):
                continue
            if pref.value in (variant, family):
                return True
        return False

    async def _announce_new_proposals(
        self, participant_key: str, consent: CoAdaptationConsentV1, now: datetime
    ) -> None:
        current = set(await self._proposals_for(participant_key, consent, now))
        known = self._known_proposals.setdefault(participant_key, set())
        if current - known:
            await self._broadcast("proposal", participant_key)
        self._known_proposals[participant_key] = current

    # ── small helpers ────────────────────────────────────────────────────────

    def _view(self) -> CoAdaptationSettingsView:
        return self._settings_provider()

    def _keyring_ok(self) -> bool:
        try:
            return bool(self._keys.available())
        except Exception:
            return False

    def _load_preferences(self) -> None:
        try:
            self._prefs = self._prefs_store.load()
            self._prefs_error = False
        except PreferenceFileError:
            # An unreadable rules file never loosens anything: studies are refused until it is fixed.
            self._prefs_error = True
            raise

    def _bump_refresh_seq(self, participant_key: str) -> int:
        value = self._refresh_seq.get(participant_key, 0) + 1
        self._refresh_seq[participant_key] = value
        return value

    def _effective(self, consent: CoAdaptationConsentV1, view: CoAdaptationSettingsView) -> _Effective:
        try:
            gaze_tracking = bool(self._gaze_tracking_enabled())
        except Exception:
            gaze_tracking = False
        return _Effective(
            adaptive=consent.covers(CoAdaptationPurpose.ADAPTIVE_ASSISTANCE) and bool(view.adaptive_enabled),
            cross_session=consent.covers(CoAdaptationPurpose.CROSS_SESSION_LEARNING)
            and bool(view.learn_across_sessions),
            gaze=consent.covers(CoAdaptationPurpose.GAZE_PRIOR) and bool(view.gaze_prior_enabled) and gaze_tracking,
            hand=consent.covers(CoAdaptationPurpose.HAND_POINTER) and bool(view.hand_pointer_enabled),
            export=consent.covers(CoAdaptationPurpose.RESEARCH_EXPORT),
        )

    def _derive_settings(
        self, effective: _Effective, view: CoAdaptationSettingsView, *, gaze_used: bool | None = None
    ) -> tuple[EngineSettings, tuple[str, ...]]:
        """Study settings: protocol-pinned floor and penalty, the stricter of protocol and config for the rest.

        Returns the settings and the names of the settings whose config value differs from the protocol.
        """
        protocol = PROTOCOL_HYPERPARAMETERS
        values = {
            "ambiguity_threshold": min(float(view.ambiguity_threshold), float(protocol["ambiguity_threshold"])),
            "unintended_activation_tolerance": min(
                float(view.unintended_activation_tolerance), float(protocol["unintended_activation_tolerance"])
            ),
            "noninferiority_margin": min(float(view.noninferiority_margin), float(protocol["noninferiority_margin"])),
            "upper_bound_credibility": max(
                float(view.upper_bound_credibility), float(protocol["upper_bound_credibility"])
            ),
            "exploration_floor": float(protocol["exploration_floor"]),
            "switching_penalty": float(protocol["switching_penalty"]),
        }
        deviations = tuple(sorted(name for name in values if float(getattr(view, name)) != float(protocol[name])))
        settings = EngineSettings(
            **values,
            adaptive_allowed=effective.adaptive,
            learn_across_sessions=effective.cross_session,
            gaze_allowed=effective.gaze if gaze_used is None else gaze_used,
            hand_pointer_allowed=effective.hand,
            thompson_draws=int(protocol["thompson_draws"]),
            day_boundary_hour=int(view.day_boundary_hour),
        )
        return settings, deviations

    def _base_settings(self, consent: CoAdaptationConsentV1) -> EngineSettings:
        view = self._view()
        return self._derive_settings(self._effective(consent, view), view)[0]

    @staticmethod
    def _versions() -> dict[str, str]:
        return {
            "strategy": STRATEGY_VERSION,
            "feature_schema": FEATURE_SCHEMA,
            "model": MODEL_VERSION,
            "prior": PRIOR_VERSION,
            "policy": POLICY_VERSION,
            "sdk": SDK_VERSION,
        }

    @staticmethod
    def _config_dict(view: CoAdaptationSettingsView) -> dict[str, Any]:
        return {name: getattr(view, name) for name in SETTINGS_FIELDS}

    @staticmethod
    def _model_status(engine: CoAdaptationEngine) -> dict[str, Any]:
        status = engine.status()
        return {
            "trials_seen": status.trials_seen,
            "sessions_seen": status.sessions_seen,
            "last_day_index": status.last_day_index,
            "drift_flags": list(status.drift_flags),
            "calibration_suspended": bool(status.calibration_suspended),
            "learn_across_sessions_effective": bool(status.learn_across_sessions_effective),
        }

    @staticmethod
    def _refused(reason: str, message: str | None = None) -> dict:
        reply: dict[str, Any] = {"status": "refused", "reason": reason}
        if message is not None:
            reply["message"] = message
        return reply

    @staticmethod
    def _baseline(level: DecisionLevel, reason: str) -> dict:
        if level == DecisionLevel.BLOCK_REGIME:
            recommendation: dict[str, Any] = {"variant": ASSISTANCE_OFF}
        elif level == DecisionLevel.AMBIGUITY_RESPONSE:
            recommendation = dict(_TABLE_BASELINE)
        else:
            recommendation = {"variant": "no_offer"}
        return {
            "status": "ignored",
            "reason": reason,
            "logged": False,
            "recommendation": {**recommendation, **AUTHORITY},
        }

    async def _broadcast(self, reason: str, participant_key: str | None, refresh_seq: int | None = None) -> None:
        """Queue a co_adaptation_updated notification; _locked sends it once the service lock is released."""
        self._outbox.append((reason, participant_key, refresh_seq))

    async def _send(self, outbox: list[tuple[str, str | None, int | None]]) -> None:
        if self._broadcast_fn is None:
            return
        for reason, participant_key, refresh_seq in outbox:
            try:
                await self._broadcast_fn(
                    "co_adaptation_updated",
                    {"reason": reason, "participant_key": participant_key, "refresh_seq": refresh_seq},
                )
            except Exception as exc:
                logger.warning("Co-adaptation notification failed (%s)", type(exc).__name__)
