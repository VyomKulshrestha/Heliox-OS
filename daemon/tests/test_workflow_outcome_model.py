"""Outcome candidates use group-held-out, de-identified examples only."""

import json
import math

import pytest

from pilot.intelligence.workflow_outcome_model import fit_workflow_outcome_model
from scripts.train_recorded_workflow_outcome import _write_new


def _examples() -> list[dict]:
    rows = []
    families = (
        ("file", "achieved"),
        ("browser", "partial"),
        ("mouse", "not_achieved"),
    )
    for family, label in families:
        for index in range(4):
            group = f"opaque-recipe-{family}-{index}"
            for _ in range(3):
                rows.append(
                    {
                        "group_id": group,
                        "features": {
                            f"action_family_{family}": 1.0,
                            "step_count_log1p": math.log1p(index + 1),
                        },
                        "label": label,
                    }
                )
    return rows


def test_outcome_candidate_is_fitted_and_research_only() -> None:
    artifact, report = fit_workflow_outcome_model(_examples(), seed=17, minimum_examples=12, minimum_groups=8)

    assert artifact["research_only"] is True
    assert artifact["execution_authority"] == "none"
    assert artifact["schema_version"] == 1
    assert report["recipe_groups"] == 12
    assert report["train_groups"] + report["test_groups"] == 12
    assert report["split"] == "group-disjoint recipe holdout"
    assert all(count > 0 for count in report["test_class_coverage"].values())
    assert set(report["candidate"]["class_recall"]) == set(report["classes"])
    assert report["candidate"]["accuracy"] >= report["majority_baseline"]["accuracy"]
    assert math.isfinite(report["candidate"]["log_loss"])
    assert "opaque-recipe" not in str(artifact)
    assert "opaque-recipe" not in str(report)
    json.dumps(artifact, allow_nan=False)
    json.dumps(report, allow_nan=False)


def test_outcome_candidate_rejects_too_few_real_examples() -> None:
    with pytest.raises(ValueError, match="user-rated runs"):
        fit_workflow_outcome_model(_examples()[:3])


def test_outcome_candidate_rejects_class_with_only_one_independent_recipe() -> None:
    examples = [row for row in _examples() if row["label"] != "not_achieved" or row["group_id"].endswith("-0")]

    with pytest.raises(ValueError, match="at least two independent recipe groups"):
        fit_workflow_outcome_model(examples, minimum_examples=12, minimum_groups=8)


def test_outcome_candidate_rejects_extra_private_fields() -> None:
    examples = _examples()
    examples[0]["goal"] = "must not enter training"
    with pytest.raises(ValueError, match="only group_id"):
        fit_workflow_outcome_model(examples, minimum_examples=12, minimum_groups=8)


def test_candidate_writer_never_overwrites_an_existing_artifact(tmp_path) -> None:
    path = tmp_path / "candidate.enc"
    path.write_bytes(b"keep")

    with pytest.raises(FileExistsError):
        _write_new(path, b"replace")

    assert path.read_bytes() == b"keep"
    fresh = tmp_path / "new.enc"
    _write_new(fresh, b"encrypted candidate")
    assert fresh.read_bytes() == b"encrypted candidate"
