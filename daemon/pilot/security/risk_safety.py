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

import os
import re
from typing import TYPE_CHECKING

from pilot.security.path_rules import rules_for
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


def _params(action: Action) -> dict:
    params = action.parameters
    if params is None:
        return {}
    try:
        return params.model_dump(exclude_none=True)
    except Exception:
        return {}


def _strings(values: dict, keys: tuple[str, ...]) -> list[str]:
    return [value for key in keys if isinstance(value := values.get(key), str) and value.strip()]


def _resolved(path: str) -> str:
    try:
        return os.path.realpath(os.path.expanduser(path.strip()))
    except (OSError, ValueError):
        return path


def _is_absolute(path: str) -> bool:
    expanded = os.path.expanduser(path.strip())
    return os.path.isabs(expanded) or bool(re.match(r"^[A-Za-z]:[\\/]", expanded))


def _fold(text: str) -> str:
    """Case is folded where the file system ignores it (Windows, macOS), and kept on Linux."""
    text = text.replace("\\", "/")
    return text.casefold() if rules_for().case_insensitive else text


# Where a protected folder written in command text ends: the end, a path separator, a space, a quote or a shell
# operator. So "/etc" matches "cat /etc/hosts" but not "/etcetera" or "/home/sam/etc-notes".
_FOLDER_END = r"(?=$|[/\s'\"`;|&)<>])"
_FOLDER_START = r"(?<![\w.~-])"
# A package name in command text is a whole token: "git" matches "apt remove git" but not "github" or "gitk".
_PACKAGE_EDGE_BEFORE = r"(?<![\w.+-])"
_PACKAGE_EDGE_AFTER = r"(?![\w.+-])"
_REMOVAL_WORDS = re.compile(r"\b(uninstall|remove|purge|erase|autoremove)\b", re.IGNORECASE)


def _folder_in_text(text: str, folder: str) -> bool:
    folded = _fold(text)
    forms = {_fold(folder.strip().rstrip("/\\")), _fold(_resolved(folder).rstrip("/\\"))}
    return any(form and re.search(_FOLDER_START + re.escape(form) + _FOLDER_END, folded) is not None for form in forms)


def _package_in_text(text: str, package: str) -> bool:
    """A protected package named as a whole token in a command that removes packages."""
    if not _REMOVAL_WORDS.search(text):
        return False
    from pilot.security.package_names import names_for

    folded = text.casefold()
    return any(
        re.search(_PACKAGE_EDGE_BEFORE + re.escape(form) + _PACKAGE_EDGE_AFTER, folded) is not None
        for form in names_for(package)
    )


def touches_protected_path(action: Action, config: PilotConfig) -> str | None:
    """Returns the matched protected folder/package string if this action's
    target or parameters touch one, else None. Heliox's direct analog of
    Ferrum's `deletes_own_config` check — a concrete fact about THIS
    action's own arguments, checked directly regardless of how the
    predicted-outcome model scores the action otherwise.

    The matching rule (it was a substring of any field, so a protected "git" scored every path with "git" in it,
    and "/etc" scored "/home/sam/etcetera"):
    - a protected folder: an absolute path in the target, ``path`` or ``destination`` that is the folder or inside
      it, after "~" and links are resolved (case folded on Windows and macOS); or the folder, as written or
      resolved, at a path boundary in a ``command`` or ``script``;
    - a protected package: the package action's name, the same package by ``package_names.is_protected``
      (ignoring case; on Windows a winget id or the name after its last dot); or one of those as a whole token in a
      ``command`` or ``script`` that removes packages (uninstall, remove, purge, erase, autoremove)."""
    from pilot.actions import PackageParams
    from pilot.security.package_names import is_protected
    from pilot.security.path_rules import path_within

    restrictions = config.restrictions
    folders = [folder for folder in restrictions.protected_folders if str(folder).strip()]
    packages = [package for package in restrictions.protected_packages if str(package).strip()]
    if not folders and not packages:
        return None

    values = _params(action)
    is_package = isinstance(action.parameters, PackageParams)
    paths = _strings(values, ("path", "destination"))
    if action.target and not is_package:
        paths.append(action.target)
    texts = _strings(values, ("script",))
    command = values.get("command")
    args = values.get("args")
    if isinstance(command, str) and command.strip():
        # A command and its arguments read as one line, so "apt" with ["remove", "git"] is seen whole (the
        # arguments were not looked at before).
        words = [str(arg) for arg in args] if isinstance(args, list) else []
        texts.append(" ".join([command, *words]))
    names = _strings(values, ("name", "package")) if is_package else []
    if is_package and action.target:
        names.append(action.target)

    for folder in folders:
        guard = _resolved(folder)
        if any(_is_absolute(path) and path_within(_resolved(path), guard) for path in paths):
            return folder
        if any(_folder_in_text(text, folder) for text in texts):
            return folder
    for package in packages:
        if any(is_protected(name, [package]) for name in names):
            return package
        if any(_package_in_text(text, package) for text in texts):
            return package
    return None


def score_outcome(
    action: Action,
    outcome: PredictedOutcome,
    config: PilotConfig,
    *,
    baseline_disk_usage: float | None = None,
) -> tuple[float, list[str]]:
    """Scores a single action's predicted outcome for risk. Returns
    (risk in [0,1], human-readable reasons — empty list if none fired)."""
    risk = 0.0
    reasons: list[str] = []

    # A pre-existing nearly full drive is important system health evidence,
    # but a read-only or disk-neutral action must not be treated as causing
    # it. The plan gate supplies the observed pre-action baseline; direct
    # callers without a baseline retain the conservative absolute check.
    disk_increased = baseline_disk_usage is None or outcome.disk_usage_after > baseline_disk_usage + 1e-6
    if disk_increased and outcome.disk_usage_after > DISK_USAGE_RISK_THRESHOLD:
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
