"""Research-only learning from explicit outcomes of saved Heliox workflows."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, recall_score
from sklearn.model_selection import GroupShuffleSplit

OUTCOMES = ("achieved", "partial", "not_achieved")
MINIMUM_EXAMPLES = 24
MINIMUM_GROUPS = 8
BASE_FEATURES = frozenset({
    "step_count_log1p", "omitted_fraction", "placeholder_fraction",
    "overlap_warning", "incomplete",
})


def fit_workflow_outcome_model(
    examples: list[dict[str, Any]], *, seed: int = 20260930,
    minimum_examples: int = MINIMUM_EXAMPLES, minimum_groups: int = MINIMUM_GROUPS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit a de-identified model and evaluate it on held-out recipe groups.

    A saved recipe's explicit outcome labels all stay in one split. This model
    is a local research candidate only; it is never an execution or approval
    signal.
    """
    if len(examples) < minimum_examples:
        raise ValueError(f"Need at least {minimum_examples} user-rated runs; found {len(examples)}")

    groups: list[str] = []
    labels: list[str] = []
    for example in examples:
        if set(example) != {"group_id", "features", "label"}:
            raise ValueError("Outcome examples must contain only group_id, features, and label")
        group_id, label, features = example["group_id"], example["label"], example["features"]
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("Every outcome example needs a recipe group")
        if label not in OUTCOMES:
            raise ValueError("Outcome label is not supported")
        if not isinstance(features, dict) or not features:
            raise ValueError("Outcome features must be a non-empty numeric mapping")
        if any(not isinstance(key, str)
               or (key not in BASE_FEATURES
                   and re.fullmatch(r"action_family_[a-z][a-z0-9_-]{0,31}", key) is None)
               or not isinstance(value, (int, float)) or isinstance(value, bool)
               or not math.isfinite(value)
               for key, value in features.items()):
            raise ValueError("Outcome features must contain finite numeric values only")
        groups.append(group_id)
        labels.append(label)

    unique_groups = sorted(set(groups))
    classes = sorted(set(labels))
    if len(unique_groups) < minimum_groups:
        raise ValueError(f"Need at least {minimum_groups} independent recipe groups; found {len(unique_groups)}")
    if len(classes) < 2:
        raise ValueError("Need at least two distinct user-reported outcome classes")

    label_values = np.asarray(labels)
    group_values = np.asarray(groups)
    all_classes = set(classes)
    splitter = GroupShuffleSplit(
        n_splits=128,
        test_size=max(2, math.ceil(len(unique_groups) * 0.25)),
        random_state=seed,
    )
    selected_split = None
    for train_indices, test_indices in splitter.split(
        np.zeros((len(examples), 1)), label_values, group_values
    ):
        if (all_classes.issubset(set(label_values[train_indices]))
                and len(set(label_values[test_indices])) >= 2):
            selected_split = train_indices, test_indices
            break
    if selected_split is None:
        raise ValueError("Could not form a group-disjoint holdout containing multiple outcome classes")
    train_indices, test_indices = selected_split
    train_groups = set(group_values[train_indices])
    test_groups = set(group_values[test_indices])
    if train_groups & test_groups:
        raise RuntimeError("Recipe group leaked between training and evaluation")

    feature_names = sorted({
        key for index in train_indices for key in examples[index]["features"]
    })

    def matrix(indices: np.ndarray) -> np.ndarray:
        return np.asarray([
            [float(examples[index]["features"].get(name, 0.0)) for name in feature_names]
            for index in indices
        ], dtype=np.float64)

    train_x = matrix(train_indices)
    test_x = matrix(test_indices)
    train_y = label_values[train_indices]
    test_y = label_values[test_indices]
    group_counts = Counter(group_values[train_indices])
    weights = np.asarray([1.0 / group_counts[group] for group in group_values[train_indices]])
    weights *= len(weights) / weights.sum()

    model = LogisticRegression(
        C=0.5, class_weight="balanced", max_iter=1000, random_state=seed,
    )
    model.fit(train_x, train_y, sample_weight=weights)
    probabilities = model.predict_proba(test_x)
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    majority = Counter(train_y).most_common(1)[0][0]
    test_classes = np.unique(test_y)
    candidate_balanced_accuracy = recall_score(
        test_y, predictions, labels=test_classes, average="macro", zero_division=0,
    )
    majority_predictions = np.full(len(test_y), majority)
    majority_balanced_accuracy = recall_score(
        test_y, majority_predictions, labels=test_classes, average="macro", zero_division=0,
    )

    artifact = {
        "schema_version": 1,
        "model_type": "group_weighted_logistic_regression",
        "feature_names": feature_names,
        "classes": model.classes_.tolist(),
        "coefficients": model.coef_.tolist(),
        "intercepts": model.intercept_.tolist(),
        "research_only": True,
        "execution_authority": "none",
    }
    report = {
        "schema_version": 1,
        "examples": len(examples),
        "recipe_groups": len(unique_groups),
        "classes": classes,
        "train_examples": len(train_indices),
        "test_examples": len(test_indices),
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),
        "train_label_counts": {str(label): int(count) for label, count in Counter(train_y).items()},
        "test_label_counts": {str(label): int(count) for label, count in Counter(test_y).items()},
        "candidate": {
            "accuracy": float(accuracy_score(test_y, predictions)),
            "balanced_accuracy": float(candidate_balanced_accuracy),
            "log_loss": float(log_loss(test_y, probabilities, labels=model.classes_)),
        },
        "majority_baseline": {
            "class": majority,
            "accuracy": float(accuracy_score(test_y, majority_predictions)),
            "balanced_accuracy": float(majority_balanced_accuracy),
        },
        "seed": seed,
        "split": "group-disjoint recipe holdout",
        "privacy": "Only coarse action-family structure and user outcome labels are used; no goals, names, parameters, screens, or executor status.",
        "limitation": "Small local sample; research only. Not Jev parity, safety evidence, or execution authority.",
    }
    return artifact, report
