"""Unpromoted local scope-decision experiment distilled from Jev samples.

This is advisory, not a permission check. The existing gateway, confirmation,
and verification systems remain authoritative. Weights are numeric-only NPZ;
no pickle or remote model call is used during prediction. This module is not
imported by the production companion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import HashingVectorizer

MODEL_VERSION = "scope-decision-v7"
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
    re.compile(r"\b(read|show|list|check|inspect|find|search|scan|filter|look up|report|summari[sz]e|analy[sz]e|tell|status|compare|review|identify|measure|extract)\b"),
    re.compile(r"\b(edit|write|update|rewrite|set|create|apply|fix|draw|add|move|switch|connect|change|draft|commit|save|export|convert|crop|rename|copy|turn|adjust|prepare|download|resize|format|transcribe|record)\b"),
    re.compile(r"\b(delete|remove|erase|stop|kill|terminate|uninstall|cancel|disconnect|wipe|clear|mute|sleep|reset|revert|disable)\b"),
    re.compile(r"\b(send|publish|push|post|upload|purchase|buy|submit|share|deploy|stream|forward|email)\b"),
    re.compile(r"\b(run|execute|launch|open|start|capture|take|navigate|display|preview|play)\b"),
)
_NEGATION = re.compile(r"\b(do not|don't|never|without|only|read.only|make no changes)\b")
_TARGETS = re.compile(
    r"\bv\d+(?:\.\d+)+\b|\b(?:heliox|ferrum|nucle|staging|production|qa|cache|database|display|screen|speaker|microphone|camera|captions|language|timezone)\b"
    r"|\b[a-z0-9_-]+\.(?:txt|md|py|json|toml|csv|png|pdf|docx|xlsx|svg)\b"
)
_BROADEN = re.compile(r"\b(all|every|entire|whole|everyone|anyone|public|publicly)\b")
_SINGULAR = re.compile(r"\b(this|one|only|exact|selected|named|current)\b")
_DRAFT = re.compile(r"\b(draft|unsent)\b")
_EXTERNAL = re.compile(r"^\s*(?:send|publish|post|upload|share|stream|forward|email)\b|\b(?:and|then)\s+(?:send|publish|post|upload|share|stream|forward|email)\b")
_DRAFT_THEN_SEND = re.compile(r"\b(?:and|then|also)\s+(?:send|post|publish|share|upload)\b")
_TURN_OFF = re.compile(r"\b(turn off|shut down|power off)\b")
_SENSITIVE = re.compile(r"\b(credentials?|passwords?|api key|private key|secret)\b")
_EXPORT = re.compile(r"^\s*(?:export|upload|send|share|forward|publish)\b")
_READ_ONLY_INTENT = re.compile(r"\b(read|show|list|check|find|search|scan|filter|look up|report|summari[sz]e|analy[sz]e|inspect|extract|tell|open|navigate)\b")
_MUTATING_INTENT = re.compile(r"\b(edit|change|update|rewrite|set|create|apply|fix|draw|add|delete|remove|erase|stop|terminate|kill|restart|install|clear|send|publish|push|post|upload|share|move|copy|rename|export|save|download|turn|mute|crop|resize|format|transcribe|record)\b")
_EXECUTE_CODE = re.compile(r"^\s*(?:run|execute|launch)\b.*\b(?:code|script|command)\b")
_EXECUTE_INTENT = re.compile(r"\b(run|execute|launch)\b")
_PRIVATE = re.compile(r"\b(private|locally|local|unsent)\b")
_PUBLIC = re.compile(r"\b(public|publicly|online|internet)\b")
_RECORD = re.compile(r"\b(record|capture)\b")
_PLAY = re.compile(r"\b(play|listen)\b")
_DESTRUCTIVE = re.compile(r"\b(delete|remove|erase|wipe|clear|uninstall|kill|terminate|cancel|format)\b")


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


def apply_target_guard(
    request: str, action: str, probability: float, *,
    mutating_action: bool | None = None, destructive_request: bool | None = None,
) -> float:
    """Only lower confidence on explicit scope expansion, never grant authority."""
    request_targets = set(_TARGETS.findall(request.lower()))
    action_targets = set(_TARGETS.findall(action.lower()))
    if request_targets and action_targets and not action_targets <= request_targets:
        return min(probability, 0.05)
    if (request_targets or _SINGULAR.search(request.lower())) and _BROADEN.search(action.lower()) and not _BROADEN.search(request.lower()):
        return min(probability, 0.05)
    if _DRAFT.search(request.lower()) and _EXTERNAL.search(action.lower()) and not _DRAFT_THEN_SEND.search(request.lower()):
        return min(probability, 0.05)
    if _TURN_OFF.search(action.lower()) and not _TURN_OFF.search(request.lower()):
        return min(probability, 0.05)
    if _SENSITIVE.search(action.lower()) and _EXPORT.search(action.lower()) and not _SENSITIVE.search(request.lower()):
        return min(probability, 0.05)
    action_mutates = bool(_MUTATING_INTENT.search(action.lower())) if mutating_action is None else mutating_action
    if _READ_ONLY_INTENT.search(request.lower()) and not _MUTATING_INTENT.search(request.lower()) and action_mutates:
        return min(probability, 0.05)
    if _EXECUTE_CODE.search(action.lower()) and not _EXECUTE_INTENT.search(request.lower()):
        return min(probability, 0.05)
    if _PRIVATE.search(request.lower()) and _PUBLIC.search(action.lower()) and not _PUBLIC.search(request.lower()):
        return min(probability, 0.05)
    if _RECORD.search(action.lower()) and _PLAY.search(request.lower()) and not _RECORD.search(request.lower()):
        return min(probability, 0.05)
    request_allows_destructive = (
        bool(_DESTRUCTIVE.search(request.lower())) if destructive_request is None else destructive_request
    )
    if _DESTRUCTIVE.search(action.lower()) and not request_allows_destructive:
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
        path = Path(weights_path) if weights_path is not None else Path(__file__).with_name("scope_decision_candidate.npz")
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
