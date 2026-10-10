"""Score frozen research candidate against local labels without exporting text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.scope_nli_features import NliFeatureExtractor, cached_paths  # noqa: E402
from benchmarks.scope_semantic_head import SemanticScopeHead  # noqa: E402
from pilot.config import EXPERIENCE_DB_FILE  # noqa: E402
from scripts.review_local_scope import LABELS_FILE, read_cases, read_labels  # noqa: E402


def summarize(labels: np.ndarray, scores: np.ndarray) -> dict:
    predicted = scores >= 0.5
    positives = int(np.sum(labels))
    negatives = len(labels) - positives
    return {
        "cases": len(labels),
        "reviewed_in_scope": positives,
        "reviewed_out_of_scope": negatives,
        "agreement": float(np.mean(predicted == labels)),
        "false_allowances": int(np.sum(predicted & ~labels)),
        "false_warnings": int(np.sum(~predicted & labels)),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=EXPERIENCE_DB_FILE)
    parser.add_argument("--labels", type=Path, default=LABELS_FILE)
    parser.add_argument("--reviewer", choices=("user", "codex"), default="user")
    parser.add_argument("--candidate", type=Path, default=root / "benchmarks/scope_semantic_candidate_v5.npz")
    parser.add_argument("--max-length", type=int, choices=(128, 512), default=512)
    args = parser.parse_args()
    cases = read_cases(args.database)
    labels_by_id = read_labels(args.labels, cases, reviewer=args.reviewer)
    selected = {}
    for case in cases:
        label = labels_by_id.get(case.case_id)
        if label in {"in_scope", "out_of_scope"}:
            selected.setdefault(case.source_hash, (case, label))
    if not selected:
        parser.error("No decisive local labels; review cases before evaluating")
    pairs = [(case.request, case.action) for case, _ in selected.values()]
    labels = np.asarray([label == "in_scope" for _, label in selected.values()], dtype=bool)
    head = SemanticScopeHead(
        args.candidate,
        extractor=NliFeatureExtractor(*cached_paths(), max_length=args.max_length),
    )
    result = head.predict_many_with_raw(pairs)
    if result is None:
        parser.error("Candidate is unavailable or incompatible")
    raw, guarded = result
    print(
        json.dumps(
            {
                "reviewer": args.reviewer,
                "max_length": args.max_length,
                "raw": summarize(labels, raw),
                "guarded": summarize(labels, guarded),
                "limitation": "Local review labels are not independent expert or user labels unless reviewer=user.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
