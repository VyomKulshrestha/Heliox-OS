"""The OPE validation of section 19 on simulated studies sized to the human design (spec section 22.4).

50 seeded studies of ``N_sim`` people (the human protocol's ``population.confirmatory_n``, 16 in the draft), F1 and F3
in equal numbers, three sessions with the block counts of section 12.3, switch cost and practice on, truth from
``counterfactual_block_value`` with each block's ``R_b``. The estimator check is asserted. The analysis's own targets
(in-sample S_2 and S_3, out-of-sample S_2) are computed and written as ``ope_validation`` under
``daemon/benchmarks/.bench-state/`` without a pass threshold. The budget is 10 minutes at N = 16 (8-17 minutes in
practice, depending on machine load), so it is opt-in: set ``HELIOX_RUN_SLOW_TESTS=1`` and run this file as its own
command (``--timeout=1800``). The default suite and CI skip it.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import pytest

from benchmarks.coadaptation_suite import (
    COVERAGE_MIN,
    DEFAULT_STATE_DIR,
    HUMAN_PROTOCOL_PATH,
    _simulate_person,
    _study_people,
    confirmatory_n,
    load_protocol,
    ope_validation,
    protocol_sha256,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("HELIOX_RUN_SLOW_TESTS") != "1",
    reason="slow simulated OPE validation; set HELIOX_RUN_SLOW_TESTS=1 to run it",
)

SIM = load_protocol()
HUMAN = load_protocol(HUMAN_PROTOCOL_PATH)
FAMILIES = ("F1", "F3")
SUMMARY_KEYS = {
    "studies",
    "covered",
    "coverage",
    "coverage_not_refused",
    "mean_bias",
    "refusal_rate",
    "median_ess",
}


@pytest.fixture(scope="module")
def validation() -> dict:
    started = time.perf_counter()
    result = ope_validation(
        n_studies=int(SIM["ope_validation"]["studies"]),
        n_sim=confirmatory_n(),
        families=FAMILIES,
        population_seed=int(SIM["population_seed"]),
        runner_seed=int(SIM["runner_seed"]),
        replays=int(SIM["ope_validation"]["counterfactual_replays"]),
    )
    result["elapsed_s"] = round(time.perf_counter() - started, 1)
    result["protocol_sha256"] = protocol_sha256(SIM)
    path = DEFAULT_STATE_DIR / f"coadaptation_ope_validation_{result['protocol_sha256'][:12]}.json"
    DEFAULT_STATE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"ope_validation": result}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["written_to"] = str(path)
    return result


def test_the_studies_are_sized_to_the_human_design(validation: dict) -> None:
    assert validation["n_sim"] == HUMAN["population"]["confirmatory_n"] == 16
    assert validation["studies"] == SIM["ope_validation"]["studies"] == 50
    assert validation["families"] == list(FAMILIES)
    people = _study_people(0, validation["n_sim"], int(SIM["population_seed"]), FAMILIES)
    assert Counter(person.family for person in people) == {"F1": 8, "F3": 8}
    _runner, result = _simulate_person(people[0], int(SIM["runner_seed"]))
    phases = [Counter(block.phase for block in row.plan.blocks) for row in result.sessions]
    assert len(phases) == 3
    assert phases[0]["A"] == 12
    for later in phases[1:]:
        assert later["A_lite"] == 6 and later["B"] + later["probe"] + later["comparator"] == 10


def test_the_estimator_check_covers_the_truth(validation: dict) -> None:
    check = validation["estimator_check"]
    assert check["studies"] == 50 and check["required"] == COVERAGE_MIN == 43
    assert check["covered"] >= 43, check
    errors = check["mean_errors"]
    for key in ("dr_unclipped", "ips_unclipped"):
        assert abs(errors[key]["mean"]) <= 3 * errors[key]["mc_se"], (key, errors[key])
    for key in ("dr", "ips", "snips"):
        assert abs(errors[key]["mean"]) <= 0.02, (key, errors[key])
    assert check["unbiased_within_3_mc_se"] and check["clipped_and_snips_within_0_02"]
    assert check["passed"]
    assert validation["estimator_check_coverage"] == check["coverage"]


def test_the_analysis_targets_are_computed_and_written(validation: dict) -> None:
    targets = {
        "in_sample_S_2": validation["in_sample"]["S_2"],
        "in_sample_S_3": validation["in_sample"]["S_3"],
        "out_of_sample_S_2": validation["out_of_sample_S_2"],
    }
    for name, summary in targets.items():
        assert set(summary) == SUMMARY_KEYS, name
        assert summary["studies"] == 50, name
        assert 0.0 <= summary["refusal_rate"] <= 1.0, name
        assert summary["median_ess"] is not None and summary["median_ess"] >= 0.0, name
    assert set(HUMAN["ope_validation"]["in_sample"]["S_2"]) == SUMMARY_KEYS - {"studies", "covered"}
    assert validation["evidence_kind"] == "simulated" and validation["promoted_to_runtime"] is False
    assert "no claim about people" in validation["claim_boundary"]
    stored = json.loads(Path(validation["written_to"]).read_text(encoding="utf-8"))
    written = stored["ope_validation"]
    assert written["in_sample"] == validation["in_sample"]
    assert written["out_of_sample_S_2"] == validation["out_of_sample_S_2"]
    assert written["protocol_sha256"] == protocol_sha256(SIM)
