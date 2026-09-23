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
from benchmarks.scope_nli_features import REVISION, NliFeatureExtractor, cached_paths

MODEL_VERSION = "scope-decision-semantic-v2"
_OBSERVE = re.compile(r"\b(read|show|list|check|inspect|find|search|scan|review|report)\b")
_UNDO = re.compile(r"\b(revert|rollback|roll back|undo|restore)\b")
_DEVICE = re.compile(r"\b(wi-?fi|bluetooth|camera|microphone|speaker|display|screen)\b")
_EXACT_TARGET = re.compile(r"\b\d+\.\d+\.\d+\b|\bissue\s+\d+\b|\bcell\s+[a-z]+\d+\b")


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


def semantic_guard(request: str, action: str, probability: float) -> float:
    """Add caution on explicit undo and named device/identifier expansion."""
    request_lower, action_lower = request.lower(), action.lower()
    if _OBSERVE.search(request_lower) and _UNDO.search(action_lower) and not _UNDO.search(request_lower):
        return min(probability, 0.05)
    request_devices = {name.replace("-", "") for name in _DEVICE.findall(request_lower)}
    action_devices = {name.replace("-", "") for name in _DEVICE.findall(action_lower)}
    if request_devices and action_devices and not action_devices <= request_devices:
        return min(probability, 0.05)
    request_targets = set(_EXACT_TARGET.findall(request_lower))
    action_targets = set(_EXACT_TARGET.findall(action_lower))
    if request_targets and action_targets and not action_targets <= request_targets:
        return min(probability, 0.05)
    return apply_target_guard(request, action, probability)


class SemanticScopeHead:
    def __init__(self, weights_path: Path, *, extractor: NliFeatureExtractor | None = None) -> None:
        self._weights: np.ndarray | None = None
        self._intercept = 0.0
        self._extractor = extractor
        try:
            with np.load(weights_path, allow_pickle=False) as data:
                weights = np.asarray(data["weights"], dtype=np.float64)
                intercept = float(data["intercept"])
                version = str(data["model_version"])
                backbone_revision = str(data["backbone_revision"])
            if (weights.shape == (FEATURE_COUNT + 6,)
                and np.all(np.isfinite(weights)) and np.isfinite(intercept)
                and version == MODEL_VERSION and backbone_revision == REVISION):
                self._weights = weights
                self._intercept = intercept
        except (FileNotFoundError, KeyError, OSError, ValueError, EOFError, BadZipFile):
            pass

    @property
    def available(self) -> bool:
        return self._weights is not None

    def predict_many(self, pairs: list[tuple[str, str]]) -> np.ndarray | None:
        if self._weights is None:
            return None
        if self._extractor is None:
            self._extractor = NliFeatureExtractor(*cached_paths())
        nli = self._extractor.transform(pairs)
        logits = np.asarray(combined_features(pairs, nli) @ self._weights).ravel() + self._intercept
        raw = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        return guarded_probabilities(pairs, raw)
