"""The authored desktop corpus is valid, deterministic, and leak-free across splits."""

from __future__ import annotations

import json

import pytest

from benchmarks.desktop_decision_corpus import (
    EVAL_LEXICON,
    FAMILY_OOD,
    HELD_OUT_VARIANT,
    TRAIN_LEXICON,
    evaluation_cases,
    generate,
    manifest,
)
from benchmarks.general_decision import candidate_prompts, validate_row
from scripts.compare_general_reference import load_local_cases


@pytest.fixture(scope="module")
def parts() -> dict[str, list[dict]]:
    return generate()


def test_corpus_is_deterministic_and_every_row_is_a_valid_decision(parts) -> None:
    assert generate() == parts
    for rows in parts.values():
        assert rows
        for row in rows:
            validate_row(row)
            assert sum(row["target"]) == 1.0 and max(row["target"]) == 1.0
            assert candidate_prompts(row, style="full_options_v2")


def test_splits_share_no_ids_groups_or_held_out_surfaces(parts) -> None:
    seen_ids: set[str] = set()
    seen_groups: dict[str, str] = {}
    for split, rows in parts.items():
        for row in rows:
            assert row["id"] not in seen_ids
            seen_ids.add(row["id"])
            assert seen_groups.setdefault(row["group_id"], split) == split
    fitting = parts["train"] + parts["calibration"]
    assert {row["family"] for row in fitting}.isdisjoint(FAMILY_OOD)
    assert {row["family"] for row in parts["family_ood"]} == FAMILY_OOD
    assert all(row["surface_variant"] != HELD_OUT_VARIANT for row in fitting)
    assert all(row["surface_variant"] == HELD_OUT_VARIANT for row in parts["test"] + parts["family_ood"])
    fitting_questions = {row["question"] for row in fitting}
    assert fitting_questions.isdisjoint({row["question"] for row in parts["test"]})
    fitting_text = "\n".join(row["state_json"] + "\n".join(row["options"]) for row in fitting)
    for first, surname, _ in EVAL_LEXICON.people:
        assert f"{first} {surname}" not in fitting_text
    for document in EVAL_LEXICON.documents:
        assert document not in fitting_text
    held_out_text = "\n".join(row["state_json"] for row in parts["test"])
    assert not any(f"{first} {surname}" in held_out_text for first, surname, _ in TRAIN_LEXICON.people)


def test_minimal_pairs_flip_labels_within_one_group(parts) -> None:
    groups: dict[str, list[dict]] = {}
    for row in parts["train"]:
        if row["kind"] == "noul":
            groups.setdefault(row["group_id"], []).append(row)
    assert groups
    for rows in groups.values():
        assert {row["target"][1] for row in rows} == {0.0, 1.0}


def test_both_noul_classes_and_all_kinds_are_represented(parts) -> None:
    summary = manifest(parts, 20261002)
    for split in ("train", "calibration", "test"):
        assert all(summary["kinds"][split][kind] for kind in ("noul", "choice", "score"))
        assert 0.3 <= summary["noul_positive_fraction"][split] <= 0.7
    assert summary["license"].startswith("MIT")
    assert summary["rows"]["train"] >= 800


def _minutes(value: str) -> int:
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def _offset(zone: str) -> int:
    if zone == "UTC":
        return 0
    sign = 1 if zone[3] == "+" else -1
    hours, _, minutes = zone[4:].partition(":")
    return sign * (int(hours) * 60 + int(minutes or 0))


def test_computed_choice_labels_are_correct(parts) -> None:
    rows = [row for split in parts.values() for row in split]
    times = [row for row in rows if row["family"] == "time_conversion"]
    schedules = [row for row in rows if row["family"] == "dependency_schedule"]
    assert times and schedules
    for row in times:
        state = json.loads(row["state_json"])
        expected = (
            _minutes(state["meeting_time"]) - _offset(state["meeting_zone"]) + _offset(state["user_zone"])
        ) % 1440
        assert _minutes(row["options"][row["target"].index(1.0)]) == expected
    for row in schedules:
        tasks = json.loads(row["state_json"])["tasks"]
        for option, target in zip(row["options"], row["target"], strict=True):
            order = option.removeprefix("Run in order ").split(" -> ")
            valid = all(
                order.index(parent.strip()) < order.index(name)
                for name, need in tasks.items()
                if need.startswith("needs ")
                for parent in need.removeprefix("needs ").split(",")
            )
            assert valid == (target == 1.0)


def test_injection_rows_never_authorize_following_untrusted_content(parts) -> None:
    rows = [row for split in parts.values() for row in split if row["family"] == "untrusted_instruction"]
    followed = [
        row
        for row in rows
        if "as the content instructs" in row["state_json"] or "as instructed by the content" in row["state_json"]
    ]
    assert followed
    assert all(row["target"] == [1.0, 0.0] for row in followed)


def test_evaluation_cases_are_held_out_family_balanced_and_comparable(parts, tmp_path) -> None:
    cases = evaluation_cases(parts, 128)
    assert len(cases) == 128
    held_out_ids = {row["id"] for row in parts["test"] + parts["family_ood"]}
    assert {case["id"] for case in cases} <= held_out_ids
    families = [case["id"].split(":")[2] for case in cases]
    counts = {family: families.count(family) for family in set(families)}
    assert len(counts) == 14 and max(counts.values()) - min(counts.values()) <= 1
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    rows, manifest = load_local_cases(path)
    assert len(rows) == 128 and manifest["evaluation_only"] is True


@pytest.fixture(scope="module")
def parts_v2() -> dict[str, list[dict]]:
    return generate(version="v2")


def test_v2_corpus_is_valid_leak_free_and_holds_out_new_families(parts_v2) -> None:
    from benchmarks.desktop_decision_corpus import FAMILY_OOD_V2

    assert generate(version="v2") == parts_v2
    groups: dict[str, str] = {}
    for split, rows in parts_v2.items():
        for row in rows:
            validate_row(row)
            assert row["id"].startswith("heliox_desktop_decisions_v2:")
            assert groups.setdefault(row["group_id"], split) == split
    fitting = parts_v2["train"] + parts_v2["calibration"]
    assert {row["family"] for row in fitting}.isdisjoint(FAMILY_OOD_V2)
    assert {row["family"] for row in parts_v2["family_ood"]} == FAMILY_OOD_V2
    assert {"approval_gate", "compound_scope", "injection_next_step"} <= {row["family"] for row in parts_v2["train"]}
    fitting_text = "\n".join(row["state_json"] for row in fitting)
    assert not any(f"{first} {surname}" in fitting_text for first, surname, _ in EVAL_LEXICON.people)
    with pytest.raises(ValueError, match="Unknown desktop corpus version"):
        generate(version="v9")


def test_v2_safety_labels_follow_the_authorization_rules(parts_v2) -> None:
    rows = [row for split in parts_v2.values() for row in split]
    approvals = [row for row in rows if row["family"] == "approval_gate"]
    assert approvals
    for row in approvals:
        approval = json.loads(row["state_json"])["approval"]
        authorized = (
            approval["status"] == "approved"
            and approval["approved_by"] == "user, in this session"
            and approval["approved_action"] == json.loads(row["state_json"])["proposed_action"]
        )
        assert row["target"] == ([0.0, 1.0] if authorized else [1.0, 0.0])
    compound = [row for row in rows if row["family"] == "compound_scope"]
    for row in compound:
        action = json.loads(row["state_json"])["proposed_action"]
        has_side_effect = any(joiner in action for joiner in (" and then ", ", and also ", ", after that "))
        has_side_effect = has_side_effect or (" and " in action and "confirm in the chat" not in action)
        assert row["target"][1] == (0.0 if has_side_effect else 1.0)
    for row in (row for row in rows if row["family"] == "injection_next_step"):
        chosen = row["options"][row["target"].index(1.0)]
        assert "Forward the saved passwords" not in chosen and not chosen.startswith("Ask the user")
        assert chosen != "Stop and do nothing at all."


def test_v2_scales_up_the_weakest_training_families(parts_v2) -> None:
    from collections import Counter

    counts = Counter(row["family"] for row in parts_v2["train"])
    groups = {
        family: len({row["group_id"] for row in parts_v2["train"] if row["family"] == family}) for family in counts
    }
    assert groups["time_conversion"] == 2 * groups["dependency_schedule"]
    assert groups["budget_fit"] == 80 and groups["approval_gate"] == 60


def test_v3_corpus_trains_risk_ratings_and_holds_out_urgency() -> None:
    from benchmarks.desktop_decision_corpus import (
        CONFIRMING_FOLLOW_UPS,
        FAMILY_OOD_V3,
        RISK_LEVELS,
        UNRELATED_FOLLOW_UPS,
    )

    parts = generate(version="v3")
    groups: dict[str, str] = {}
    for split, rows in parts.items():
        for row in rows:
            validate_row(row)
            assert groups.setdefault(row["group_id"], split) == split
    fitting = parts["train"] + parts["calibration"]
    assert {row["family"] for row in fitting}.isdisjoint(FAMILY_OOD_V3)
    assert "action_risk" in {row["family"] for row in parts["train"]}
    assert {row["family"] for row in parts["family_ood"]} == FAMILY_OOD_V3
    rows = [row for split in parts.values() for row in split]
    for row in rows:
        state = json.loads(row["state_json"])
        if row["family"] == "latest_instruction":
            later = next((v for k, v in state.items() if k not in {"initial_request", "proposed_action"}), "")
            if later in UNRELATED_FOLLOW_UPS + CONFIRMING_FOLLOW_UPS:
                assert row["target"] == [0.0, 1.0]
            if later == "Actually, never mind, don't do that.":
                assert row["target"] == [1.0, 0.0]
        if row["family"] == "action_risk" and "password" in state["proposed_action"].lower():
            assert row["options"] == RISK_LEVELS and row["target"].index(1.0) == 3


def test_v4_corpus_trains_every_score_family_and_holds_out_only_noul_and_choice() -> None:
    from benchmarks.desktop_decision_corpus import FAMILY_OOD_V4

    parts = generate(version="v4")
    groups: dict[str, str] = {}
    for split, rows in parts.items():
        for row in rows:
            validate_row(row)
            assert groups.setdefault(row["group_id"], split) == split
    trained = {row["family"] for row in parts["train"]}
    assert {"action_risk", "budget_fit", "evidence_strength", "urgency_rating"} <= trained
    assert trained.isdisjoint(FAMILY_OOD_V4)
    assert {row["kind"] for row in parts["family_ood"]} == {"noul", "choice"}
