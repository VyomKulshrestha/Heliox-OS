"""Unpromoted Jev-scope student using a frozen public NLI backbone.

The NLI weights are not Jev weights and are not bundled. The trained artifact
contains only numeric logistic coefficients. This module never grants action
authority and is not imported by the production execution companion.
"""

from __future__ import annotations

import re
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from scipy.sparse import csr_matrix, hstack

from benchmarks.scope_decision_experiment import FEATURE_COUNT, apply_target_guard
from benchmarks.scope_decision_experiment import feature_matrix as lexical_features
from benchmarks.scope_effects import effect_conflict, normalize_request_for_guard
from benchmarks.scope_nli_features import REVISION, NliFeatureExtractor, cached_paths

V3_VERSION = "scope-decision-semantic-v3"
V4_VERSION = "scope-decision-semantic-v4"
MODEL_VERSION = "scope-decision-semantic-v5"
_OBSERVE = re.compile(r"\b(read|show|list|check|inspect|find|search|scan|review|report)\b")
_UNDO = re.compile(r"\b(revert|rollback|roll back|undo|restore)\b")
_DEVICE = re.compile(r"\b(wi-?fi|bluetooth|camera|microphone|speaker|display|screen)\b")
_EXACT_TARGET_V2 = re.compile(r"\b\d+\.\d+\.\d+\b|\bissue\s+\d+\b|\bcell\s+[a-z]+\d+\b")
_EXACT_TARGET = re.compile(r"\b\d+\.\d+\.\d+\b|\b(?:issue|ticket)\s+\d+\b|\bcell\s+[a-z]+\d+\b")
_BROADEN = re.compile(r"\b(all|every|entire|whole|everyone|anyone)\b")
_COLUMN = re.compile(r"\b(?:across|whole|entire)\s+column\b")
_COLUMN_V4 = re.compile(r"\b(?:across|down|whole|entire)\s+column\b")
_DIFFERENT = re.compile(r"\b(?:different|another|other)\b")
_SPECIFIC = re.compile(r"\b(?:selected|this|only|one|named)\b")
_APP = re.compile(r"\b(?:app|application)\b")
_OPERATING_SYSTEM = re.compile(r"\b(?:operating system|system-wide|os-wide)\b")
_CLOSE = re.compile(r"\b(?:close|resolve)\b")
_EXTERNAL_ACTION = re.compile(r"(?:^\s*|\b(?:and|then|also)\s+)(?:publish|post|upload|share|send|email|forward|deploy|release)\b")
_EXTERNAL_REQUEST = re.compile(r"\b(?:publish|post|upload|share|send|email|forward|deploy|release)\b")
_UNSUBSCRIBE = re.compile(r"\bunsubscribe\b")
_DESTRUCTIVE_ALIAS = (
    (re.compile(r"\b(?:stop|pause)\b"), re.compile(r"^\s*(?:cancel|stop)\b")),
    (re.compile(r"\bpause\b"), re.compile(r"^\s*stop\b")),
    (re.compile(r"\bclear\b"), re.compile(r"^\s*remove\b")),
)
_MUTATING_CLAUSE = re.compile(
    r"(?:^\s*(?:please\s+)?|\b(?:and|then|also|to)\s+|[,;]\s*)"
    r"(?:edit|change|update|rewrite|set|create|apply|fix|draw|add|delete|remove|erase|stop|terminate|kill|restart|install|clear|send|publish|push|post|upload|share|move|copy|rename|export|save|download|turn|mute|crop|resize|format|transcribe|record)\b"
)


def combined_features(pairs: list[tuple[str, str]], nli: np.ndarray) -> csr_matrix:
    if nli.shape != (len(pairs), 6) or not np.all(np.isfinite(nli)):
        raise ValueError("Expected six finite NLI scores per request/action pair")
    semantic = np.log(np.clip(nli, 1e-4, 1.0))
    return hstack((lexical_features(pairs), csr_matrix(semantic)), format="csr")


def guarded_probabilities(pairs: list[tuple[str, str]], values: np.ndarray) -> np.ndarray:
    return np.asarray([
        semantic_guard(request, action, float(probability))
        for (request, action), probability in zip(pairs, values, strict=True)
    ])


def semantic_guard(request: str, action: str, probability: float, *, version: str = MODEL_VERSION) -> float:
    """Add caution on explicit undo and named device/identifier expansion."""
    if version == MODEL_VERSION:
        if effect_conflict(request, action):
            return min(probability, 0.05)
        return semantic_guard(normalize_request_for_guard(request, action), action, probability, version=V4_VERSION)
    request_lower, action_lower = request.lower(), action.lower()
    if _OBSERVE.search(request_lower) and _UNDO.search(action_lower) and not _UNDO.search(request_lower):
        return min(probability, 0.05)
    request_devices = {name.replace("-", "") for name in _DEVICE.findall(request_lower)}
    action_devices = {name.replace("-", "") for name in _DEVICE.findall(action_lower)}
    if version == V4_VERSION:
        request_devices = {"display" if name == "screen" else name for name in request_devices}
        action_devices = {"display" if name == "screen" else name for name in action_devices}
    if request_devices and action_devices and not action_devices <= request_devices:
        return min(probability, 0.05)
    target_pattern = _EXACT_TARGET if version in {V3_VERSION, V4_VERSION} else _EXACT_TARGET_V2
    request_targets = set(target_pattern.findall(request_lower))
    action_targets = set(target_pattern.findall(action_lower))
    if request_targets and action_targets and not action_targets <= request_targets:
        return min(probability, 0.05)
    if version in {V3_VERSION, V4_VERSION}:
        if version == V4_VERSION:
            if _BROADEN.search(action_lower) and not _BROADEN.search(request_lower):
                return min(probability, 0.05)
            if _SPECIFIC.search(request_lower) and _DIFFERENT.search(action_lower) and not _DIFFERENT.search(request_lower):
                return min(probability, 0.05)
            if _APP.search(request_lower) and _OPERATING_SYSTEM.search(action_lower) and not _OPERATING_SYSTEM.search(request_lower):
                return min(probability, 0.05)
            if _CLOSE.search(action_lower) and not _CLOSE.search(request_lower):
                return min(probability, 0.05)
            if _EXTERNAL_ACTION.search(action_lower) and not _EXTERNAL_REQUEST.search(request_lower):
                return min(probability, 0.05)
            if _UNSUBSCRIBE.search(action_lower) and not _UNSUBSCRIBE.search(request_lower):
                return min(probability, 0.05)
            if any(target.startswith("cell ") for target in request_targets) and _COLUMN_V4.search(action_lower) and not action_targets:
                return min(probability, 0.05)
        if request_targets and _BROADEN.search(action_lower) and not _BROADEN.search(request_lower):
            return min(probability, 0.05)
        if any(target.startswith("cell ") for target in request_targets) and _COLUMN.search(action_lower) and not action_targets:
            return min(probability, 0.05)
        destructive_request = None
        if version == V4_VERSION:
            destructive_request = any(
                request_verb.search(request_lower) and action_verb.search(action_lower)
                for request_verb, action_verb in _DESTRUCTIVE_ALIAS
            )
            destructive_request = destructive_request or None
        guard_request = re.sub(r"\bscreen\b", "display", request) if version == V4_VERSION else request
        guard_action = re.sub(r"\bscreen\b", "display", action) if version == V4_VERSION else action
        return apply_target_guard(guard_request, guard_action, probability,
                                  mutating_action=bool(_MUTATING_CLAUSE.search(action_lower)),
                                  destructive_request=destructive_request)
    return apply_target_guard(request, action, probability)


class SemanticScopeHead:
    def __init__(self, weights_path: Path, *, extractor: NliFeatureExtractor | None = None) -> None:
        self._weights: np.ndarray | None = None
        self._intercept = 0.0
        self._extractor = extractor
        self._version = MODEL_VERSION
        try:
            with np.load(weights_path, allow_pickle=False) as data:
                weights = np.asarray(data["weights"], dtype=np.float64)
                intercept = float(data["intercept"])
                version = str(data["model_version"])
                backbone_revision = str(data["backbone_revision"])
            if (weights.shape == (FEATURE_COUNT + 6,)
                and np.all(np.isfinite(weights)) and np.isfinite(intercept)
                and version in {"scope-decision-semantic-v2", V3_VERSION, V4_VERSION, MODEL_VERSION} and backbone_revision == REVISION):
                self._weights = weights
                self._intercept = intercept
                self._version = version
        except (FileNotFoundError, KeyError, OSError, ValueError, EOFError, BadZipFile):
            pass

    @property
    def available(self) -> bool:
        return self._weights is not None

    def predict_many_with_raw(self, pairs: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray] | None:
        """Return numeric and guarded scores for audit; neither is authority."""
        if self._weights is None:
            return None
        if self._extractor is None:
            self._extractor = NliFeatureExtractor(*cached_paths())
        nli = self._extractor.transform(pairs)
        logits = np.asarray(combined_features(pairs, nli) @ self._weights).ravel() + self._intercept
        raw = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        if self._version == MODEL_VERSION:
            return raw, guarded_probabilities(pairs, raw)
        guarded = np.asarray([
            semantic_guard(request, action, float(probability), version=self._version)
            for (request, action), probability in zip(pairs, raw, strict=True)
        ])
        return raw, guarded

    def predict_many(self, pairs: list[tuple[str, str]]) -> np.ndarray | None:
        scores = self.predict_many_with_raw(pairs)
        return None if scores is None else scores[1]
