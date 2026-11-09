"""Gnomon round 7: runtime-shaped training families, the everyday-review gate, and a code-only Kaggle bundle."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.desktop_decision_corpus import EVAL_LEXICON, TRAIN_LEXICON, VERSIONS, generate  # noqa: E402
from benchmarks.runtime_review_families import runtime_harm, runtime_scope  # noqa: E402
from pilot.intelligence.gnomon_reviewer import HARM_LEVELS, HARM_QUESTION, SCOPE_QUESTION  # noqa: E402
from scripts import build_kaggle_notebook, check_gnomon_everyday_review  # noqa: E402

REVIEW_SET = ROOT / "benchmarks" / "gnomon_everyday_review_v1.json"


def test_runtime_families_ask_exactly_what_the_reviewer_asks_with_a_near_miss_for_each_plan():
    rows = list(runtime_scope(random.Random(1), TRAIN_LEXICON, 0, 3))
    assert [row.target_index for row in rows] == [1, 0]  # the correct plan, then its near miss
    assert all(row.question == SCOPE_QUESTION and set(row.state) == {"proposed_action", "user_message"} for row in rows)
    assert rows[0].state["user_message"] == rows[1].state["user_message"]
    assert rows[0].state["proposed_action"] != rows[1].state["proposed_action"]
    harm = list(runtime_harm(random.Random(1), TRAIN_LEXICON, 0, 10))  # opening a site and a lookalike login page
    assert all(row.question == HARM_QUESTION and row.options == list(HARM_LEVELS) for row in harm)
    assert harm[0].target_index == 0 and harm[1].target_index == 2


def test_corpus_v5_adds_the_runtime_families_and_keeps_the_v4_held_out_families():
    name, families, held_out, scale = VERSIONS["v5"]
    assert name == "heliox_desktop_decisions_v5" and {"runtime_scope", "runtime_harm"} <= set(families)
    assert held_out == VERSIONS["v4"][2] and scale["runtime_scope"] == 3.0
    parts = generate(seed=20261002, version="v5")
    train_families = {row["family"] for row in parts["train"]}
    assert {"runtime_scope", "runtime_harm"} <= train_families
    assert not {row["family"] for row in parts["family_ood"]} & {"runtime_scope", "runtime_harm"}
    assert generate(seed=20261002, version="v5")["train"] == parts["train"]  # deterministic


def test_training_never_sees_the_everyday_review_set():
    review = json.loads(REVIEW_SET.read_text(encoding="utf-8"))["cases"]
    requests = {case["request"] for case in review}
    parts = generate(seed=20261002, version="v5")
    seen = {json.loads(row["state_json"]).get("user_message") for split in parts.values() for row in split}
    assert not requests & seen
    # Nor the eval lexicon's entities, which the held-out surfaces use.
    assert not {doc for doc in EVAL_LEXICON.documents} & {doc for doc in TRAIN_LEXICON.documents}


def test_the_gate_counts_false_alarms_and_misses_and_spares_asked_for_deletions():
    rows = [
        {"set": "dev", "label": "correct", "request": "a", "escalate": True},
        {"set": "dev", "label": "correct", "request": "b", "escalate": False},
        {"set": "dev", "label": "correct", "request": "delete it", "escalate": True, "needs_approval_anyway": True},
        {"set": "heldout", "label": "bad", "request": "c", "escalate": True},
        {"set": "heldout", "label": "bad", "request": "d", "escalate": False},
    ]
    result = check_gnomon_everyday_review.score(rows)
    assert result["false_alarms"] == 1 and result["correct_plans"] == 2
    assert result["misses"] == 1 and not result["passed"]
    review = json.loads(REVIEW_SET.read_text(encoding="utf-8"))["cases"]
    assert sum(c["label"] == "correct" and not c.get("needs_approval_anyway") for c in review) == 27
    assert sum(c["label"] == "bad" for c in review) == 12


def test_the_kaggle_bundle_is_tracked_code_only():
    payload, files = build_kaggle_notebook.bundle()
    assert files and all(name.endswith(".py") or name in build_kaggle_notebook.EXTRA_FILES for name in files)
    assert not set(files) & build_kaggle_notebook._excluded()
    assert not any("/.bench-state/" in name for name in files)
    assert "daemon/benchmarks/runtime_review_families.py" in files
