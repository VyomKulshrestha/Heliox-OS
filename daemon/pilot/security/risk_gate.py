"""Learned Risk Gate — ties Layers 1/3/4/5 together for a whole plan.

Mirrors Ferrum-OS's cognitive/world_model/mod.rs's evaluate_action(), which
composes the encoder/transition/safety layers for one proposed action.
Heliox's RiskGate.evaluate_plan() captures one OS snapshot, runs every
action through both learned and deterministic transitions, scores the
riskier result, and advances a conservative simulated state. This preserves
the "one bad action anywhere in the plan is enough" rule while also catching
resource impact that compounds across actions already present in the plan.

Strictly additive: if no weights are staged, evaluate_plan() still runs
using deterministic transitions. When a learned model is staged, a learned
prediction can add caution but cannot replace a stronger rule prediction.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from pilot.actions import ActionType
from pilot.intelligence.world_model import (
    HistoricalFailureRisk,
    HybridWorldModel,
    RiskEvidence,
    UiJepaPredictor,
    WorldPrediction,
    WorldState,
)
from pilot.security.risk_model import EMBEDDING_SIZE, LEARNABLE_ACTION_TYPES, RiskTransitionModel
from pilot.security.risk_observation import (
    NOMINAL_PROC_CAPACITY,
    ON_LOOP_SCAN_BUDGET_SECONDS,
    PLAN_SCAN_BUDGET_SECONDS,
    OsSnapshot,
    TreeSize,
    VolumeUsage,
    capture_os_snapshot,
    measure_tree_size,
    volume_of,
)
from pilot.security.risk_safety import DISK_USAGE_RISK_THRESHOLD, DISK_USAGE_RISK_WEIGHT, score_outcome

if TYPE_CHECKING:
    from pilot.actions import Action, ActionPlan
    from pilot.config import PilotConfig
    from pilot.intelligence.online_learning import VerifiedOnlineLearner


# ── Deterministic disk-size lookahead ──
#
# The transition model's disk inputs are the current usage fraction and the
# action type; nothing tells it how many bytes an action writes. Its rule
# table adds a fixed fraction of the system drive per action (copy +1%,
# write +2%, download +3%) and the learned MLP, trained on 1 KB to 512 KB
# sandbox files, predicts about +0.00001%. Both are blind to size. Where the
# size is knowable before running, the gate measures it here, and that
# measurement replaces the rule table's fixed guess for that action.

# Adds an approval (destructive_critic.HEURISTIC_RISK_THRESHOLD is 0.3)
# without claiming a fill the gate cannot prove; a proven fill scores
# DISK_USAGE_RISK_WEIGHT (0.8).
UNMEASURED_SIZE_RISK = 0.5
# Other deterministic findings on the riskiest action at least this strong are reported next to the top one.
SUPPORTING_REASON_FLOOR = 0.5
# Same 1e-6 fraction risk_safety uses to decide a prediction "added" disk
# usage: about 1 MB on a 1 TB drive. Smaller writes never flag.
_MATERIAL_FRACTION = 1e-6
_WRITE_PLACEHOLDER = re.compile(r"\{[A-Z][A-Z0-9_]*\}")
_PATH_TOKEN = r"(\"[^\"]+\"|'[^']+'|[^\s;|&\"']+)"
_FSUTIL_CREATENEW = re.compile(
    r"\bfsutil(?:\.exe)?\s+file\s+createnew\s+" + _PATH_TOKEN + r"\s+(\d+)(?![\w.])",
    re.IGNORECASE,
)
_FALLOCATE_LENGTH = re.compile(r"\bfallocate\s+(?:-l\s*|--length[\s=]+)(\d+)([KMGTPE])?(iB|B)?\s+" + _PATH_TOKEN)
_UNIT_POWER = {"K": 1, "M": 2, "G": 3, "T": 4, "P": 5, "E": 6}


@dataclass(frozen=True)
class _DiskWrite:
    """Bytes one action is expected to add to the volume of `destination`."""

    verb: str
    noun: str
    label: str
    destination: str
    bytes: int
    complete: bool
    entries: int = 1
    files: int = 0
    stopped_early: bool = False
    source: str | None = None


def _format_bytes(size: int) -> str:
    """Binary units with the labels Windows Explorer shows (1 GB = 1024^3)."""
    value = float(max(0, size))
    if value < 1024:
        return f"{int(value)} bytes"
    for unit in ("KB", "MB", "GB", "TB"):
        value /= 1024
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if value < 10 else f"{value:.0f} {unit}"
    return f"{value:.0f} TB"


def _format_fraction(fraction: float) -> str:
    text = f"{fraction:.0%}"
    if text == f"{DISK_USAGE_RISK_THRESHOLD:.0%}":
        text = f"{fraction:.1%}"
    return text


def _volume_name(root: str) -> str:
    stripped = root.rstrip("\\/")
    return stripped or root


def _unquote(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    return token


def _label(path: str) -> str:
    """Last path component for a reason, splitting on both \\ and / so a
    Windows path reads the same on every platform."""
    stripped = path.rstrip("\\/")
    return re.split(r"[\\/]", stripped)[-1] if stripped else path


# Paths in actions are paths on the machine the daemon runs on, so they are
# read with that platform's rules (ntpath on Windows, posixpath elsewhere).
# Tests pass the other flavour explicitly to check both on any OS.
_PATH_FLAVOUR: Any = os.path


def _shell_created_files(
    text: str,
    working_directory: str | None,
    flavour: Any = None,
) -> list[tuple[str, int]]:
    """Files whose exact size the shell text states, for commands that exist
    on the platform: `fsutil file createnew <path> <bytes>` on Windows and
    `fallocate -l <size> <path>` elsewhere. Both allocate real space. A path
    that is not a real path for the platform (a `C:\\...` path under
    posixpath) is never resolved, and a relative path is joined to the
    command's working directory when that is absolute, or to the daemon's
    own when the command names none and the flavour is the running one.
    `truncate -s` is not counted (it makes sparse files) and `dd` is not
    counted (its size depends on bs, count and the input)."""
    flavour = flavour or _PATH_FLAVOUR
    windows = flavour.__name__ == "ntpath"
    created: list[tuple[str, int]] = []
    if windows:
        for match in _FSUTIL_CREATENEW.finditer(text):
            created.append((_unquote(match.group(1)), int(match.group(2))))
    else:
        for match in _FALLOCATE_LENGTH.finditer(text):
            number, unit, suffix, token = match.groups()
            base = 1000 if suffix == "B" else 1024
            created.append((_unquote(token), int(number) * base ** _UNIT_POWER.get(unit or "", 0)))
    resolved: list[tuple[str, int]] = []
    for path, size in created:
        if not flavour.isabs(path):
            if working_directory and flavour.isabs(working_directory):
                path = flavour.join(working_directory, path)
            elif not working_directory and flavour is os.path:
                path = flavour.join(os.getcwd(), path)
            else:
                continue
        resolved.append((path, size))
    return resolved


def _scan_budget_seconds() -> float:
    """Full budget off the event loop; a short one if a caller still runs
    evaluate_plan() on a loop thread, so the loop is never held for long."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return PLAN_SCAN_BUDGET_SECONDS
    return ON_LOOP_SCAN_BUDGET_SECONDS


class _PlanDiskLookahead:
    """Per-plan, ordered disk ledger keyed by volume.

    Only proven growth is added. Space that a delete or a cross-volume move
    would free is never credited, so a delete earlier in the plan cannot
    make a later fill look safe (the delete may fail, go to a recycle bin,
    or be declined while the later step still runs).
    """

    def __init__(
        self,
        snapshot: OsSnapshot,
        *,
        measure: Callable[..., TreeSize | None],
        volume_lookup: Callable[[str], VolumeUsage | None],
    ) -> None:
        self._measure = measure
        self._budget_seconds = _scan_budget_seconds()
        self._volume_lookup = volume_lookup
        self._deadline: float | None = None
        self._sizes: dict[str, TreeSize | None] = {}
        self._volumes: dict[str, dict[str, Any]] = {}
        self._os_key: str | None = None
        os_volume = self._lookup(snapshot.disk_path) if snapshot.disk_path else None
        if os_volume is not None:
            # The plan's snapshot is authoritative for its own volume.
            self._os_key = os_volume.key
            self._volumes[os_volume.key] = {
                "root": os_volume.root,
                "total": os_volume.total,
                "used": round(snapshot.disk_usage_fraction * os_volume.total),
            }

    def _lookup(self, path: str) -> VolumeUsage | None:
        try:
            return self._volume_lookup(path)
        except Exception:
            return None

    def _volume_state(self, path: str) -> dict[str, Any] | None:
        volume = self._lookup(path)
        if volume is None or volume.total <= 0:
            return None
        return self._volumes.setdefault(
            volume.key,
            {"root": volume.root, "total": volume.total, "used": volume.used},
        )

    def sync_system_volume(self, fraction: float) -> None:
        """Carry rule or learned growth on the snapshot volume into the ledger."""
        state = self._volumes.get(self._os_key) if self._os_key else None
        if state is not None:
            state["used"] = max(state["used"], round(fraction * state["total"]))

    def system_volume_fraction(self) -> float | None:
        state = self._volumes.get(self._os_key) if self._os_key else None
        return state["used"] / state["total"] if state is not None else None

    def _size_of(self, path: str) -> TreeSize | None:
        key = os.path.normcase(os.path.abspath(os.path.expanduser(path)))
        if key not in self._sizes:
            if self._deadline is None:
                self._deadline = time.perf_counter() + self._budget_seconds
            try:
                self._sizes[key] = self._measure(path, deadline=self._deadline)
            except Exception:
                self._sizes[key] = None
        return self._sizes[key]

    def _volume_root_size(self, path: str) -> TreeSize | None:
        """A whole volume holds exactly its used bytes (plus file-system
        metadata the copy skips), so copying its root needs no scan. Only
        exact volume roots qualify; ordinary folders are always walked."""
        try:
            root = os.path.abspath(os.path.expanduser(path))
            if not os.path.ismount(root):
                return None
        except Exception:
            return None
        volume = self._lookup(root)
        if volume is None:
            return None
        return TreeSize(bytes=volume.used, entries=0, complete=True, files=0)

    def estimate(self, action: Action) -> list[_DiskWrite]:
        """Bytes the action would add, where knowable without running it."""
        try:
            return self._estimate(action)
        except Exception:
            return []

    def _estimate(self, action: Action) -> list[_DiskWrite]:
        params = action.parameters
        action_type = action.action_type
        if action_type in (ActionType.FILE_COPY, ActionType.FILE_MOVE):
            source = str(getattr(params, "path", "") or action.target or "")
            destination = str(getattr(params, "destination", "") or "")
            if not source or not destination:
                return []
            destination_volume = self._lookup(destination)
            if destination_volume is None:
                return []  # no local volume to charge: keep the rule table, skip the scan
            if action_type == ActionType.FILE_MOVE:
                source_volume = self._lookup(source)
                if source_volume is None:
                    return []
                if source_volume.key == destination_volume.key:
                    return []  # a rename on one volume adds nothing
            size = self._volume_root_size(source) or self._size_of(source)
            if size is None:
                return []
            copying = action_type == ActionType.FILE_COPY
            return [
                _DiskWrite(
                    verb="copying" if copying else "moving",
                    noun="the copy" if copying else "the move",
                    label=_label(source),
                    destination=destination,
                    bytes=size.bytes,
                    complete=size.complete,
                    entries=size.entries,
                    files=size.files,
                    stopped_early=size.stopped_early,
                    source=source,
                )
            ]
        if action_type == ActionType.FILE_WRITE:
            path = str(getattr(params, "path", "") or action.target or "")
            content = getattr(params, "content", None)
            if not path or not isinstance(content, str):
                return []
            if action.use_previous_output or _WRITE_PLACEHOLDER.search(content):
                return []  # the executor substitutes earlier output of unknown size
            size = len(content.encode("utf-8"))
            if sys.platform == "win32":
                size += content.count("\n")  # text-mode write turns \n into \r\n
            return [_DiskWrite("writing", "the write", _label(path), path, size, True)]
        if action_type in (ActionType.SHELL_COMMAND, ActionType.SHELL_SCRIPT):
            text = str(getattr(params, "command", "") or getattr(params, "script", "") or "")
            args = getattr(params, "args", None)
            if isinstance(args, list):
                text = " ".join([text, *(str(arg) for arg in args)])
            working_directory = getattr(params, "working_directory", None)
            return [
                _DiskWrite("creating", "the new file", _label(path), path, size, True)
                for path, size in _shell_created_files(text, working_directory)
            ]
        return []

    def apply(self, writes: list[_DiskWrite]) -> tuple[float, str, list[dict[str, Any]], bool]:
        """Add each write to its volume in plan order.

        Returns (risk, reason, records, resolved). `resolved` is True when
        every write landed on a known local volume, so the measurement can
        stand in for the rule table's fixed guess.
        """
        risk = 0.0
        reason = ""
        records: list[dict[str, Any]] = []
        resolved = bool(writes)
        for write in writes:
            state = self._volume_state(write.destination)
            if state is None:
                resolved = False
                continue
            total = state["total"]
            before = state["used"]
            place = _volume_name(state["root"])
            state["used"] = before + write.bytes
            after = state["used"]
            free_before = max(0, total - before)
            material = write.bytes > _MATERIAL_FRACTION * total
            size_text = f"{'about' if write.complete else 'at least'} {_format_bytes(write.bytes)}"
            write_risk = 0.0
            write_reason = ""
            if material and after / total > DISK_USAGE_RISK_THRESHOLD:
                write_risk = DISK_USAGE_RISK_WEIGHT
                if after > total:
                    write_reason = (
                        f"{write.verb} {write.label} ({size_text}) needs more than the "
                        f"{_format_bytes(free_before)} free on {place}"
                    )
                else:
                    write_reason = (
                        f"{write.verb} {write.label} ({size_text}) would take {place} from "
                        f"{_format_fraction(before / total)} to {'' if write.complete else 'at least '}"
                        f"{_format_fraction(after / total)}, past the safe level of "
                        f"{DISK_USAGE_RISK_THRESHOLD:.0%}"
                    )
            elif not write.complete and not self._fits_whole_source_volume(write, before, total):
                write_risk = UNMEASURED_SIZE_RISK
                why = "is too large to measure quickly" if write.stopped_early else "could not be fully measured"
                write_reason = (
                    f"{write.label} {why} (at least {_format_bytes(write.bytes)} in "
                    f"{write.files:,} files so far). {place} has "
                    f"{_format_bytes(free_before)} free, so {write.noun} may not fit."
                )
            records.append(
                {
                    "path_label": write.label,
                    "bytes": write.bytes,
                    "size_is_exact": write.complete,
                    "entries_scanned": write.entries,
                    "volume": state["root"],
                    "usage_before": before / total,
                    "usage_after": after / total,
                    "risk": write_risk,
                }
            )
            if write_risk > risk:
                risk = write_risk
                reason = write_reason
        return risk, reason, records, resolved

    def _fits_whole_source_volume(self, write: _DiskWrite, used_before: int, total: int) -> bool:
        """Cheap upper bound for an unfinished scan: a source tree cannot
        hold more than its volume's used bytes (ignoring hard links and
        sparse or compressed files, which a copy can expand). If even that
        much stays under the safe level, the copy is known to fit."""
        if write.source is None:
            return False
        source_volume = self._lookup(write.source)
        if source_volume is None:
            return False
        remaining = max(0, source_volume.used - write.bytes)
        return (used_before + write.bytes + remaining) / total <= DISK_USAGE_RISK_THRESHOLD


def _with_supporting_reasons(reasons: list[str], prediction: dict[str, Any]) -> list[str]:
    """Add the riskiest action's other strong deterministic findings after its top reason.

    Only the single strongest reason used to reach the person, so a generic one ("Action
    requests root privileges") could hide a measured one ("copying C: (about 692 GB) needs
    more than the 260 GB free on C:"). This only adds wording; the score is unchanged.
    """
    combined = list(reasons)
    evidence = prediction.get("risk_evidence") or []
    ranked = sorted(
        (item for item in evidence if isinstance(item, dict)),
        key=lambda item: -float(item.get("score") or 0.0),
    )
    for item in ranked:
        reason = str(item.get("reason") or "").strip()
        score = item.get("score")
        if (
            reason
            and item.get("deterministic") is True
            and isinstance(score, (int, float))
            and score >= SUPPORTING_REASON_FLOOR
            and reason not in combined
        ):
            combined.append(reason)
    return combined


def _highest_risk_evidence(prediction: WorldPrediction) -> RiskEvidence | None:
    """Return the strongest well-formed advisory evidence from a predictor."""
    highest: RiskEvidence | None = None
    highest_score = -1.0
    for evidence in prediction.risk_evidence:
        raw_score = evidence.score
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            continue
        try:
            score = float(raw_score)
        except (OverflowError, TypeError, ValueError):
            continue
        if (
            not math.isfinite(score)
            or not 0.0 <= score <= 1.0
            or not isinstance(evidence.deterministic, bool)
            or not isinstance(evidence.source, str)
            or not evidence.source.strip()
            or not isinstance(evidence.reason, str)
            or not evidence.reason.strip()
        ):
            continue
        if score > highest_score:
            highest = evidence
            highest_score = score
    return highest


# While the online learner loads after start-up (pilot.server, after ready), every action with a source gets the
# largest risk the learner can ever add (0.7, above the 0.3 approval threshold in destructive_critic), so the approval
# it might have added is there; the reason says the learner was still loading.
LEARNER_LOADING_RISK = 0.7
LEARNER_LOADING_REASON = (
    "Heliox was still loading what it has learned from past outcomes, so it asks for your approval to be safe."
)


class RiskGate:
    """Owns the (optional) learned transition model and evaluates whole
    plans against it. One instance is enough for the daemon's lifetime —
    see get_risk_gate() below; construct your own only in tests."""

    def __init__(self, weights_path: str | None = None) -> None:
        self._transition = RiskTransitionModel(weights_path)
        self._world_model = HybridWorldModel()
        self._history = HistoricalFailureRisk()
        self._online_learner: VerifiedOnlineLearner | None = None
        self._ui_observation_provider: Callable[[], dict[str, Any] | None] | None = None
        self._last_evaluation: dict[str, object] | None = None
        # evaluate_plan() may run in worker threads (the daemon calls it via
        # asyncio.to_thread) and overlap with another plan's evaluation. One
        # lock serializes evaluations; a thread-local copy lets a caller read
        # back the evaluation its own call produced, not a newer one.
        self._evaluate_lock = threading.Lock()
        self._history_lock = threading.Lock()
        self._thread_state = threading.local()
        self._measure_tree_size: Callable[..., TreeSize | None] = measure_tree_size
        self._volume_of: Callable[[str], VolumeUsage | None] = volume_of

    def set_online_learner(self, learner: VerifiedOnlineLearner) -> None:
        """Attach verified outcome evidence without changing policy authority."""
        self._online_learner = learner

    def set_ui_observation_provider(
        self,
        provider: Callable[[], dict[str, Any] | None] | None,
    ) -> None:
        """Use an existing fresh screen observation; never capture in this gate."""
        self._ui_observation_provider = provider

    def configure_ui_jepa_artifact(
        self,
        weights_path: str | Path,
        *,
        weight_loader: Callable[[Path], bytes] | None = None,
    ) -> None:
        """Load an optional visual candidate while preserving structured rules."""
        self._world_model.set_visual_predictor(UiJepaPredictor(weights_path, weight_loader=weight_loader))

    def record_outcome(self, action_type: object, success: bool) -> None:
        """Feed only observed execution outcomes into historical caution."""
        with self._history_lock:
            self._history.record(action_type, success)

    @property
    def available(self) -> bool:
        """True once a learned transition model is actually staged — the
        rule-based fallback runs regardless, so this only reports whether
        predictions for LEARNABLE_ACTION_TYPES come from real training or
        the honest rule-table default."""
        return self._transition.is_loaded

    @property
    def last_evaluation(self) -> dict[str, object] | None:
        """Return a copy of the assessment made by this thread's latest
        evaluate_plan() call, or the latest from any thread if this thread
        has made none."""
        evaluation = getattr(self._thread_state, "evaluation", None) or self._last_evaluation
        return dict(evaluation) if evaluation is not None else None

    def _history_status(self) -> dict[str, Any]:
        with self._history_lock:
            return self._history.status()

    def _history_score(self, action_type: object) -> tuple[float, str]:
        with self._history_lock:
            return self._history.score(action_type)

    def _online_adaptation(self, action_type: str, source: str) -> Any:
        """The learner is updated on the event loop; reading it from a worker
        thread can meet a deque or dict mid-update, which raises RuntimeError.
        Retry a few times; if it keeps failing, raise so assess_plan_risk()
        treats the evaluation as unavailable (it escalates, never relaxes)."""
        assert self._online_learner is not None
        for attempt in range(5):
            try:
                return self._online_learner.score_transition(action_type=action_type, source=source)
            except RuntimeError:
                if attempt == 4:
                    raise
                time.sleep(0.001)
        return None

    def status(self, enabled: bool) -> dict[str, object]:
        """Return user-safe runtime/model metadata for Settings."""
        validation_mae = self._transition.validation_mae
        baseline_mae = self._transition.baseline_mae
        return {
            "enabled": enabled,
            "weights_loaded": self.available,
            "model_version": self._transition.model_version,
            "training_samples": self._transition.training_samples,
            "validation_samples": self._transition.validation_samples,
            "calibrated": self._transition.is_calibrated,
            "validation_mae": (
                {"disk_delta": validation_mae[0], "process_delta": validation_mae[1]} if validation_mae else None
            ),
            "baseline_mae": (
                {"disk_delta": baseline_mae[0], "process_delta": baseline_mae[1]} if baseline_mae else None
            ),
            "embedding_size": EMBEDDING_SIZE,
            "learnable_action_types": sorted(action_type.value for action_type in LEARNABLE_ACTION_TYPES),
            "prediction_contract": self._world_model.status(),
            "historical_risk": self._history_status(),
            "verified_online_transition": (
                self._online_learner.status()["transitions"] if self._online_learner is not None else None
            ),
            "last_evaluation": self._last_evaluation,
        }

    def evaluate_plan(self, plan: ActionPlan, config: PilotConfig, *, source: str = "") -> tuple[float, list[str]]:
        """Returns (worst risk in [0,1] seen across the plan's actions,
        the reasons that fired for that worst action, empty if none)."""
        with self._evaluate_lock:
            result = self._evaluate_plan_locked(plan, config, source=source)
            self._thread_state.evaluation = self._last_evaluation
        return result

    def _evaluate_plan_locked(
        self, plan: ActionPlan, config: PilotConfig, *, source: str = ""
    ) -> tuple[float, list[str]]:
        if not plan.actions:
            self._last_evaluation = {
                "evaluated_at": datetime.now(UTC).isoformat(),
                "action_count": 0,
                "risk_score": 0.0,
                "reasons": [],
                "worst_action_type": None,
                "prediction_sources": [],
                "prediction_confidence": None,
                "predictions": [],
            }
            return 0.0, []

        # One snapshot for the whole plan: this runs before execution
        # (predicting what WOULD happen), not interleaved with it, so OS
        # state isn't expected to shift meaningfully action-to-action —
        # and psutil calls aren't free, so one call beats N.
        snapshot = capture_os_snapshot()
        ui_observation: dict[str, Any] = {}
        if self._ui_observation_provider is not None:
            try:
                ui_observation = self._ui_observation_provider() or {}
            except Exception:
                # Screen availability must not decide whether policy runs.
                ui_observation = {}

        simulated = snapshot
        lookahead = _PlanDiskLookahead(
            snapshot,
            measure=self._measure_tree_size,
            volume_lookup=self._volume_of,
        )
        cumulative_proc_delta = 0.0
        worst_risk = 0.0
        worst_reasons: list[str] = []
        worst_action_type: str | None = None
        worst_sources: list[str] = []
        worst_prediction_confidence: float | None = None
        world_predictions: list[dict[str, object]] = []
        rollout_ui = ui_observation
        rollout_browser: dict[str, Any] = {}
        worst_step: int | None = None
        for step_index, action in enumerate(plan.actions):
            contract_prediction = self._world_model.predict(
                WorldState.from_os_snapshot(simulated, ui=rollout_ui, browser=rollout_browser),
                action,
            )
            worst_before_step = (worst_risk, worst_action_type)
            contract_evidence = _highest_risk_evidence(contract_prediction)
            rollout_ui = dict(contract_prediction.predicted_state.ui)
            rollout_browser = dict(contract_prediction.predicted_state.browser)
            learned_or_rule = self._transition.predict(simulated, action)
            deterministic = self._transition.predict_rule(simulated, action)
            lookahead.sync_system_volume(simulated.disk_usage_fraction)
            disk_writes = lookahead.estimate(action)
            size_risk, size_reason, size_records, size_resolved = lookahead.apply(disk_writes)
            if size_resolved:
                # A measured size replaces the rule table's fixed per-action
                # disk guess; the lookahead ledger now carries that growth.
                deterministic = replace(deterministic, disk_usage_after=simulated.disk_usage_fraction)
                if learned_or_rule.source == "rule":
                    learned_or_rule = deterministic
            candidates = [learned_or_rule]
            if learned_or_rule.source != "rule":
                candidates.append(deterministic)

            step_proc_delta = max(
                (outcome.proc_count_delta_normalized for outcome in candidates),
                key=abs,
            )
            cumulative_proc_delta += step_proc_delta

            action_risk = 0.0
            action_reasons: list[str] = []
            for outcome in candidates:
                cumulative_outcome = replace(outcome, proc_count_delta_normalized=cumulative_proc_delta)
                risk, reasons = score_outcome(
                    action,
                    cumulative_outcome,
                    config,
                    baseline_disk_usage=simulated.disk_usage_fraction,
                )
                if risk > action_risk or not action_reasons:
                    action_risk = risk
                    action_reasons = reasons
                if risk > worst_risk or worst_action_type is None:
                    worst_risk = risk
                    worst_reasons = reasons
                    worst_action_type = action.action_type.value
                    worst_sources = sorted({candidate.source for candidate in candidates})
                    worst_prediction_confidence = (
                        learned_or_rule.confidence if learned_or_rule.source != "rule" else None
                    )

            transition_risk = action_risk
            transition_reasons = list(action_reasons)

            if size_risk > action_risk:
                action_risk = size_risk
                action_reasons = [size_reason]
            if size_risk > worst_risk:
                worst_risk = size_risk
                worst_reasons = [size_reason]
                worst_action_type = action.action_type.value
                worst_sources = ["size_estimate"]
                worst_prediction_confidence = None

            if contract_evidence is not None and contract_evidence.score > action_risk:
                action_risk = contract_evidence.score
                action_reasons = [contract_evidence.reason]
            if contract_evidence is not None and contract_evidence.score > worst_risk:
                worst_risk = contract_evidence.score
                worst_reasons = [contract_evidence.reason]
                worst_action_type = action.action_type.value
                worst_sources = sorted(
                    {
                        *contract_prediction.sources,
                        *(candidate.source for candidate in candidates),
                        contract_evidence.source,
                    }
                )
                worst_prediction_confidence = (
                    None
                    if contract_evidence.deterministic
                    else max(0.0, min(1.0, 1.0 - contract_prediction.uncertainty))
                )

            historical_risk, historical_reason = self._history_score(action.action_type)
            if historical_risk > action_risk:
                action_risk = historical_risk
                action_reasons = [historical_reason]
            if historical_risk > worst_risk:
                worst_risk = historical_risk
                worst_reasons = [historical_reason]
                worst_action_type = action.action_type.value
                worst_sources = ["execution_history"]
                worst_prediction_confidence = min(
                    1.0,
                    self._history_status()["samples"] / 10.0,
                )

            online_risk = 0.0
            online_reason = ""
            if self._online_learner is not None and source and not getattr(self._online_learner, "is_ready", True):
                # Still loading after start-up: absent means more careful, never neutral. Add the most the learner
                # could ever add, which always asks for approval, and say why.
                online_risk = LEARNER_LOADING_RISK
                online_reason = LEARNER_LOADING_REASON
            elif self._online_learner is not None and source:
                adaptation = self._online_adaptation(action.action_type.value, source)
                if adaptation.state == "promoted" and adaptation.probability < 0.42:
                    online_risk = min(0.7, max(0.0, (0.5 - adaptation.probability) * 2.0))
                    online_reason = (
                        f"{adaptation.labels} independently verified {action.action_type.value} "
                        "outcomes suggest elevated failure risk"
                    )
            if online_risk > action_risk:
                action_risk = online_risk
                action_reasons = [online_reason]
            if online_risk > worst_risk:
                worst_risk = online_risk
                worst_reasons = [online_reason]
                worst_action_type = action.action_type.value
                worst_sources = ["verified_online_transition"]
                worst_prediction_confidence = None

            contract_payload = contract_prediction.to_dict()
            contract_payload["input_state_source"] = "observed" if step_index == 0 else "predicted"
            contract_payload["sources"] = list(
                dict.fromkeys(
                    [
                        *contract_payload["sources"],
                        *(candidate.source for candidate in candidates),
                    ]
                )
            )
            contract_payload["risk_score"] = action_risk
            if historical_risk > 0:
                contract_payload["sources"] = list(dict.fromkeys([*contract_payload["sources"], "execution_history"]))
                contract_payload["risk_evidence"].append(
                    {
                        "source": "execution_history",
                        "score": historical_risk,
                        "reason": historical_reason,
                        "deterministic": False,
                    }
                )
            if online_risk > 0:
                contract_payload["sources"] = list(
                    dict.fromkeys([*contract_payload["sources"], "verified_online_transition"])
                )
                contract_payload["risk_evidence"].append(
                    {
                        "source": "verified_online_transition",
                        "score": online_risk,
                        "reason": online_reason,
                        "deterministic": False,
                    }
                )
            if size_records:
                contract_payload["sources"] = list(dict.fromkeys([*contract_payload["sources"], "size_estimate"]))
                contract_payload["size_estimates"] = size_records
                if size_risk > 0:
                    contract_payload["risk_evidence"].append(
                        {
                            "source": "size_estimate",
                            "score": size_risk,
                            "reason": size_reason,
                            "deterministic": True,
                        }
                    )
            contract_payload["risk_evidence"].extend(
                {
                    "source": "risk_transition",
                    "score": transition_risk,
                    "reason": reason,
                    "deterministic": "rule" in contract_payload["sources"],
                }
                for reason in transition_reasons
            )
            world_predictions.append(contract_payload)
            if (worst_risk, worst_action_type) != worst_before_step:
                worst_step = step_index

            # The lookahead never lowers predicted usage: freed space is not
            # credited, so a delete cannot hide a later fill in the plan.
            disk_after = max(simulated.disk_usage_fraction, *(outcome.disk_usage_after for outcome in candidates))
            system_volume_fraction = lookahead.system_volume_fraction()
            if system_volume_fraction is not None:
                disk_after = max(disk_after, system_volume_fraction)
            simulated = replace(
                simulated,
                disk_usage_fraction=min(1.0, disk_after),
                proc_count=max(
                    0,
                    round(snapshot.proc_count + cumulative_proc_delta * NOMINAL_PROC_CAPACITY),
                ),
            )

        if worst_step is not None and worst_reasons:
            worst_reasons = _with_supporting_reasons(worst_reasons, world_predictions[worst_step])
        self._last_evaluation = {
            "evaluated_at": datetime.now(UTC).isoformat(),
            "action_count": len(plan.actions),
            "risk_score": worst_risk,
            "reasons": worst_reasons,
            "worst_action_type": worst_action_type,
            "prediction_sources": worst_sources,
            "prediction_confidence": worst_prediction_confidence,
            "predictions": world_predictions,
        }
        return worst_risk, worst_reasons


_gate: RiskGate | None = None


def get_risk_gate() -> RiskGate:
    """Lazily-constructed process-wide singleton — the learned weights (if
    any) only need loading once per daemon lifetime."""
    global _gate
    if _gate is None:
        _gate = RiskGate()
    return _gate
