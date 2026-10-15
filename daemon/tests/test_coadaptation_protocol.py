"""The frozen simulation protocol and the draft human study protocol (spec sections 18.4 and 21.1)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from pilot.coadaptation.core.simulator import FAMILY_FORMS, FAMILY_SPLITS
from pilot.coadaptation.core.study_plan import LAYOUT_GRID, PROTOCOL_HYPERPARAMETERS
from scripts.evaluate_coadaptation_study import AnalysisRefused, analyse, protocol_sha256

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
HUMAN = json.loads((BENCHMARKS / "coadaptation_protocol_v1.json").read_text(encoding="utf-8"))
SIM = json.loads((BENCHMARKS / "coadaptation_sim_protocol_v1.json").read_text(encoding="utf-8"))

HUMAN_KEYS = {
    "protocol_id",
    "status",
    "source_commit",
    "pinned_hashes",
    "evidence_kind",
    "population",
    "design",
    "hyperparameters",
    "utility_weights",
    "hypotheses",
    "ope",
    "ope_validation",
    "gates",
    "analysis",
    "exclusions",
    "do_not_tune_on",
    "failure_handling",
    "limitations",
    "promoted_to_runtime",
}


def _hypothesis(identifier: str) -> dict:
    return next(h for h in HUMAN["hypotheses"] if h["id"] == identifier)


def test_human_protocol_has_every_required_key() -> None:
    assert set(HUMAN) == HUMAN_KEYS
    assert "ope_in_sample" not in json.dumps(HUMAN)
    assert HUMAN["protocol_id"] == "coadapt-study-v1"
    assert HUMAN["status"] == "draft_pending_pilot"
    assert HUMAN["source_commit"] is None
    assert set(HUMAN["pinned_hashes"]) == {
        "prior_table",
        "policy",
        "analysis_script",
        "consent_text",
    }
    assert HUMAN["evidence_kind"] == "live_human"
    assert HUMAN["promoted_to_runtime"] is False
    population = HUMAN["population"]
    assert population["confirmatory_n"] == 16 and population["pilot_n"] == 3
    assert "F2, F4, F6" in population["n_rule"]
    assert "never the tuning families" in population["n_rule"]


def test_human_protocol_design_matches_the_study_plan() -> None:
    design = HUMAN["design"]
    assert design["layout_grid"] == LAYOUT_GRID
    assert design["layout_grid"]["menu"]["n_items"] == [6, 12, 16]
    assert design["sessions"] == 3 and design["block_trials"] == 8
    assert "3 adaptive, 1 probe, 1 comparator" in design["later_sessions"]
    assert "rank_fixed_regimes" in design["comparator"]
    assert design["retention_days_min"] >= 90
    assert HUMAN["hyperparameters"] == PROTOCOL_HYPERPARAMETERS
    weights = HUMAN["utility_weights"]
    assert "logged" in weights["primary"] and weights["secondary"] == [0.5, 0.5, 0.5]


def test_human_protocol_hypotheses() -> None:
    assert [h["id"] for h in HUMAN["hypotheses"]] == [f"H{i}" for i in range(1, 10)]
    for identifier in ("H2", "H3", "H6"):
        assert _hypothesis(identifier)["burn_in_trials"] == 3
    assert _hypothesis("H2")["ni_margin"] == 0.05
    assert _hypothesis("H5")["probe_throughput"] == "descriptive"
    assert _hypothesis("H6")["idle_ua"] == "descriptive"
    assert _hypothesis("H6")["margin_pp"] == 2.0
    assert _hypothesis("H7")["exploratory"] is True
    h9 = _hypothesis("H9")
    assert h9["session_only_keeps"] == ["practice_counters", "n_item_eff"]
    assert h9["per_outcome"] is True


def test_human_protocol_ope_and_restated_g5() -> None:
    ope = HUMAN["ope"]
    assert ope["clip"] == 20 and ope["min_ess"] == 30
    assert "per-decision" in ope["weights"]
    assert "never re-run" in ope["refusal"]
    validation = HUMAN["ope_validation"]
    assert set(validation) == {
        "n_sim",
        "estimator_check_coverage",
        "in_sample",
        "out_of_sample_S_2",
        "power_run_refusal_rates",
    }
    assert set(validation["in_sample"]) == {"S_2", "S_3"}
    summary_keys = {"coverage", "coverage_not_refused", "mean_bias", "refusal_rate", "median_ess"}
    assert set(validation["in_sample"]["S_2"]) == summary_keys
    assert set(validation["out_of_sample_S_2"]) == summary_keys
    g5 = HUMAN["gates"]["G5"]
    assert g5.startswith("OPE validity") and "43/50" in g5 and "not refused" in g5
    assert set(HUMAN["gates"]) == {f"G{i}" for i in range(1, 7)}


def test_simulation_protocol_is_frozen_with_the_simulator_forms() -> None:
    assert SIM["status"] == "frozen"
    assert SIM["evidence_kind"] == "simulated"
    assert SIM["family_forms"] == {family: asdict(forms) for family, forms in FAMILY_FORMS.items()}
    assert SIM["splits"] == {split: list(families) for split, families in FAMILY_SPLITS.items()}
    assert SIM["population_seed"] == 20261015 and SIM["runner_seed"] == 7
    assert SIM["people_per_family"] == 40 and SIM["sessions"] == 3
    assert SIM["hyperparameters"] == PROTOCOL_HYPERPARAMETERS
    assert SIM["do_not_tune_on"] == ["F2", "F4", "F6"]
    assert SIM["holdout_looks_allowed"] == 1
    assert "no claim about people" in SIM["claim_boundary"]
    assert SIM["promoted_to_runtime"] is False
    assert SIM["metrics"]


def test_the_analysis_refuses_a_confirmatory_run_under_the_draft_protocol() -> None:
    with pytest.raises(AnalysisRefused, match="frozen_before_collection"):
        analyse([], HUMAN, exploratory=False)


def test_results_cite_a_stable_protocol_hash() -> None:
    digest = protocol_sha256(HUMAN)
    assert len(digest) == 64
    assert digest == protocol_sha256(json.loads(json.dumps(HUMAN, indent=4)))
