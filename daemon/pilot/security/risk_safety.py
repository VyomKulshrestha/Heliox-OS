"""Learned Risk Gate — Layer 5: Safety Rules.

Mirrors Ferrum-OS's cognitive/world_model/safety.rs: scores a *predicted*
outcome (risk_model.py's PredictedOutcome) for risk using hardcoded,
human-readable rules — never a learned model. The ML in this feature is
scoped entirely to predicting WHAT WOULD HAPPEN (risk_model.py); this
module decides whether that's dangerous, the same separation of concerns
Ferrum's design uses so the actual block/allow decision stays fully
auditable regardless of how good or bad the prediction turns out to be.

This is a second, *predictive* check — it runs before, and independently
of, the existing reactive PermissionChecker/ConfirmationGate. Both must
still pass; neither replaces the other. See destructive_critic.py's
risk_score() for how this composes with the existing heuristic_risk().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pilot.security.risk_model import PredictedOutcome
from pilot.security.risk_observation import NOMINAL_PROC_CAPACITY

if TYPE_CHECKING:
    from pilot.actions import Action
    from pilot.config import PilotConfig

# Mirrors safety.rs's BLOCK_THRESHOLD/disk-usage/fork-bomb constants —
# named so each figure is traceable to a specific, auditable rule rather
# than a bare literal buried in a formula.
DISK_USAGE_RISK_THRESHOLD = 0.95
DISK_USAGE_RISK_WEIGHT = 0.8

# 50 processes in one step, normalized the same way risk_observation.py's
# NOMINAL_PROC_CAPACITY scales proc_count — matches Ferrum's own
# "FORK_BOMB_DELTA_FRACTION = 50.0 / 64.0" reasoning, rescaled to Heliox's
# own nominal capacity constant.
FORK_BOMB_DELTA_THRESHOLD = 50.0 / NOMINAL_PROC_CAPACITY
FORK_BOMB_RISK_WEIGHT = 0.7

PROTECTED_PATH_RISK_WEIGHT = 0.9


def _path_like_params(action: Action) -> list[str]:
    """Every string-valued field on the action's parameters that plausibly
    holds a filesystem path or package name — best-effort, mirrors
    destructive_critic.py's own _format_action_list relevant-field list."""
    candidates: list[str] = []
    if action.target:
        candidates.append(action.target)

    params = action.parameters
    if params is None:
        return candidates
    try:
        param_dict = params.model_dump(exclude_none=True)
    except Exception:
        return candidates

    for key in ("path", "destination", "name", "package", "command", "script"):
        value = param_dict.get(key)
        if isinstance(value, str) and value:
            candidates.append(value)

    return candidates


def touches_protected_path(action: Action, config: PilotConfig) -> str | None:
    """Returns the matched protected folder/package string if this action's
    target or parameters touch one, else None. Heliox's direct analog of
    Ferrum's `deletes_own_config` check — a concrete fact about THIS
    action's own arguments, checked directly regardless of how the
    predicted-outcome model scores the action otherwise."""
    restrictions = config.restrictions
    protected = list(restrictions.protected_folders) + list(restrictions.protected_packages)
    if not protected:
        return None

    for candidate in _path_like_params(action):
        normalized = candidate.replace("\\", "/").lower()
        for guard in protected:
            guard_normalized = guard.replace("\\", "/").lower()
            if guard_normalized and guard_normalized in normalized:
                return guard
    return None


def score_outcome(action: Action, outcome: PredictedOutcome, config: PilotConfig) -> tuple[float, list[str]]:
    """Scores a single action's predicted outcome for risk. Returns
    (risk in [0,1], human-readable reasons — empty list if none fired)."""
    risk = 0.0
    reasons: list[str] = []

    if outcome.disk_usage_after > DISK_USAGE_RISK_THRESHOLD:
        risk += DISK_USAGE_RISK_WEIGHT
        reasons.append(f"predicted disk usage {outcome.disk_usage_after:.0%} exceeds the safe threshold")

    if abs(outcome.proc_count_delta_normalized) > FORK_BOMB_DELTA_THRESHOLD:
        risk += FORK_BOMB_RISK_WEIGHT
        reasons.append("predicted process-count change looks like a fork-bomb pattern")

    matched = touches_protected_path(action, config)
    if matched:
        risk += PROTECTED_PATH_RISK_WEIGHT
        reasons.append(f"touches a protected path/package: {matched}")

    return min(1.0, risk), reasons
