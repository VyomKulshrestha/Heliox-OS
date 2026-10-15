"""The simulated-user benchmark: the quick CI run and the holdout-look accounting (spec sections 18.2-18.4)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from benchmarks import coadaptation_suite
from benchmarks.coadaptation_suite import (
    LOOKS_FILE,
    HoldoutLookRefused,
    SuiteConfig,
    benchmark,
    load_protocol,
    protocol_sha256,
    run_config,
)
from pilot.coadaptation.core.outcome_model import ModelVariant

REPORT_KEYS = {
    "schema_version",
    "benchmark",
    "evidence_kind",
    "scope",
    "families",
    "split",
    "in_sample",
    "environment",
    "protocol_sha256",
    "metrics",
    "agency_audit",
    "limitations",
    "claim_boundary",
    "promoted_to_runtime",
}


@pytest.fixture(scope="module")
def quick_reports() -> tuple[dict, dict, float]:
    started = time.perf_counter()
    first = benchmark(quick=True)
    elapsed = time.perf_counter() - started
    second = benchmark(quick=True)
    return first, second, elapsed


def test_quick_output_shape_and_labels(quick_reports) -> None:
    report, _, _ = quick_reports
    assert set(report) == REPORT_KEYS
    assert report["schema_version"] == "1.0.0"
    assert report["benchmark"] == "coadaptation_simulated_users"
    assert report["evidence_kind"] == "simulated"
    assert report["families"] == ["F1", "F3"]
    assert report["split"] == "tune"
    assert report["in_sample"] is True
    assert "tuning families, in-sample" in report["claim_boundary"]
    boundary = "simulated users only; tuning families, in-sample; no claim about people"
    assert report["claim_boundary"] == boundary
    assert report["promoted_to_runtime"] is False
    assert report["protocol_sha256"] == protocol_sha256(load_protocol())
    for family in ("F1", "F3"):
        entry = report["metrics"]["by_family"][family]
        assert entry["label"] == "tuning families, in-sample"
        assert set(entry["policies"]) == {
            "1_fixed_assistance_off",
            "4_uniform_random",
            "9_engine_full",
        }
        assert entry["sessions"] == 2 and entry["people"] == 5
        # FULL against NO_TODAY per family.
        assert entry["prediction"]["full_vs_no_today"]["people"] == 5
        for metrics in entry["policies"].values():
            assert metrics["constraint_violations"] == 0
            assert metrics["acting_under_ambiguity"] == 0
            assert metrics["realized_utility_per_trial"]["mean"] is not None
        blocks = entry["policies"]["9_engine_full"]["blocks_by_phase"]
        assert all(blocks.get(phase, 0) > 0 for phase in ("B", "probe", "comparator"))
    assert set(report["metrics"]["by_split"]) == {"tune"}


def test_quick_agency_counters_are_zero(quick_reports) -> None:
    report, _, _ = quick_reports
    for policy, audit in report["agency_audit"].items():
        assert audit["passed"], policy
        for key, value in audit["counts"].items():
            assert value == (1.0 if key in ("A7", "A8") else 0), (policy, key)


def test_quick_is_deterministic_and_fast(quick_reports) -> None:
    first, second, elapsed = quick_reports
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert elapsed < 60.0


def _holdout_config() -> SuiteConfig:
    return SuiteConfig(
        families=("F4", "F6"),
        people_per_family=1,
        sessions=2,
        policies=("9_engine_full",),
        bootstrap_n=20,
        population_seed=20261015,
        runner_seed=7,
        prediction_variants=(ModelVariant.FULL, ModelVariant.NO_TODAY),
        workers=1,
    )


def test_a_second_holdout_look_is_refused_without_allow_extra_look(tmp_path: Path) -> None:
    protocol = load_protocol()
    first = run_config(_holdout_config(), protocol, state_dir=tmp_path)
    assert first["split"] == "holdout"
    assert first["in_sample"] is False
    assert "extra_holdout_look" not in first
    looks = (tmp_path / LOOKS_FILE).read_text(encoding="utf-8").splitlines()
    assert len(looks) == 1
    look = json.loads(looks[0])
    assert look["protocol_sha256"] == protocol_sha256(protocol)
    assert look["families"] == ["F4", "F6"]
    with pytest.raises(HoldoutLookRefused):
        run_config(_holdout_config(), protocol, state_dir=tmp_path)
    assert len((tmp_path / LOOKS_FILE).read_text(encoding="utf-8").splitlines()) == 1
    extra = run_config(_holdout_config(), protocol, state_dir=tmp_path, allow_extra_look=True)
    assert extra["extra_holdout_look"] is True
    assert len((tmp_path / LOOKS_FILE).read_text(encoding="utf-8").splitlines()) == 2


def test_tuning_runs_log_no_look_and_the_cli_writes_under_the_state_dir(
    tmp_path: Path,
    capsys,
) -> None:
    assert coadaptation_suite.main(["--quick", "--json", "--state-dir", str(tmp_path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["split"] == "tune"
    assert not (tmp_path / LOOKS_FILE).exists()
    assert list(tmp_path.glob("coadaptation_quick_*.json"))


def test_ablations_are_benchmark_only() -> None:
    pilot_root = Path(__file__).resolve().parents[1] / "pilot"
    for path in pilot_root.rglob("*.py"):
        assert "coadaptation_ablations" not in path.read_text(encoding="utf-8"), path
