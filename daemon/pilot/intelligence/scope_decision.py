"""Small local, typed scope-decision experiment distilled from Jev samples.

This is advisory, not a permission check. The existing gateway, confirmation,
and verification systems remain authoritative. Weights are numeric-only NPZ;
no pickle or remote model call is used during prediction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import HashingVectorizer

MODEL_VERSION = "scope-decision-v1"
HASH_DIM = 2048
_VECTORIZER = HashingVectorizer(
    n_features=HASH_DIM,
    ngram_range=(1, 2),
    alternate_sign=False,
    binary=True,
    norm="l2",
    lowercase=True,
)
_WORDS = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "for",
        "from",
        "i",
        "in",
        "into",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "the",
        "this",
        "to",
        "what",
        "which",
        "with",
        "you",
    ]
)
_OPERATIONS = (
    re.compile(r"\b(read|show|list|check|inspect|find|search|report|summari[sz]e|analy[sz]e|tell|status)\b"),
    re.compile(r"\b(edit|write|update|rewrite|set|create|move|switch|connect|change|draft|commit)\b"),
    re.compile(r"\b(delete|remove|stop|kill|terminate|uninstall|cancel|disconnect|wipe|clear|mute|sleep)\b"),
    re.compile(r"\b(send|publish|push|post|upload|purchase|buy|submit|share)\b"),
    re.compile(r"\b(run|execute|launch|open|start|capture|take)\b"),
)
_NEGATION = re.compile(r"\b(do not|don't|never|without|only|read.only|make no changes)\b")
_TARGETS = re.compile(r"\bv\d+(?:\.\d+)+\b|\b(?:heliox|ferrum|nucle)\b|\b[a-z0-9_-]+\.(?:txt|md|py|json|toml|csv)\b")


def _terms(text: str) -> set[str]:
    return set(_WORDS.findall(text.lower())) - _STOP


def _operation_vector(text: str) -> list[float]:
    value = text.lower()
    return [float(bool(pattern.search(value))) for pattern in _OPERATIONS]


def feature_matrix(pairs: list[tuple[str, str]]) -> csr_matrix:
    """Fixed-layout lexical and relation features; no train-time vocabulary."""
    requests = [request for request, _ in pairs]
    actions = [action for _, action in pairs]
    lexical_request = _VECTORIZER.transform(requests)
    lexical_action = _VECTORIZER.transform(actions)
    relational: list[list[float]] = []
    for request, action in pairs:
        request_terms, action_terms = _terms(request), _terms(action)
        overlap = len(request_terms & action_terms)
        request_ops, action_ops = _operation_vector(request), _operation_vector(action)
        request_targets = set(_TARGETS.findall(request.lower()))
        action_targets = set(_TARGETS.findall(action.lower()))
        relational.append(
            [
                overlap / max(1, len(request_terms)),
                overlap / max(1, len(action_terms)),
                float(bool(request_targets and action_targets and not action_targets <= request_targets)),
                float(bool(_NEGATION.search(request.lower()))),
                float(bool(_NEGATION.search(action.lower()))),
                float(request_ops == action_ops),
                *request_ops,
                *action_ops,
                *(float(left and right) for left, right in zip(request_ops, action_ops, strict=True)),
            ]
        )
    return hstack([lexical_request, lexical_action, csr_matrix(np.asarray(relational))], format="csr")


FEATURE_COUNT = 2 * HASH_DIM + 6 + 3 * len(_OPERATIONS)


def apply_target_guard(request: str, action: str, probability: float) -> float:
    """Only lower confidence on an explicit named-target mismatch."""
    request_targets = set(_TARGETS.findall(request.lower()))
    action_targets = set(_TARGETS.findall(action.lower()))
    if request_targets and action_targets and not action_targets <= request_targets:
        return min(probability, 0.05)
    return probability


@dataclass(frozen=True)
class ScopeDecision:
    in_scope_probability: float
    confidence: float
    source: str = MODEL_VERSION


class LocalScopeDecisionHead:
    """Predicts a scope probability; callers must not treat it as approval."""

    def __init__(self, weights_path: str | Path | None = None) -> None:
        self._weights: np.ndarray | None = None
        self._intercept = 0.0
        path = (
            Path(weights_path) if weights_path is not None else Path(__file__).with_name("scope_decision_weights.npz")
        )
        try:
            with np.load(path, allow_pickle=False) as data:
                weights = np.asarray(data["weights"], dtype=np.float64)
                intercept = float(data["intercept"])
                version = str(data["model_version"])
            if (
                weights.shape == (FEATURE_COUNT,)
                and np.all(np.isfinite(weights))
                and np.isfinite(intercept)
                and version == MODEL_VERSION
            ):
                self._weights = weights
                self._intercept = intercept
        except (FileNotFoundError, KeyError, OSError, ValueError, BadZipFile, EOFError):
            pass

    @property
    def available(self) -> bool:
        return self._weights is not None

    def predict(self, request: str, action: str) -> ScopeDecision | None:
        if self._weights is None:
            return None
        matrix = feature_matrix([(request, action)])
        logit = float((matrix @ self._weights)[0] + self._intercept)
        probability = float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))
        # This explicit mismatch was missed by the small student on its
        # development set. It may only add an advisory, never grant one.
        probability = apply_target_guard(request, action, probability)
        return ScopeDecision(in_scope_probability=probability, confidence=abs(2 * probability - 1))
