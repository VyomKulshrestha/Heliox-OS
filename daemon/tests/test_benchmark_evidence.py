from __future__ import annotations

import json
from pathlib import Path


def test_committed_software_benchmark_bundle_has_truthful_passing_contracts() -> None:
    evidence_dir = Path(__file__).resolve().parents[2] / "docs" / "evidence"
    path = sorted(evidence_dir.glob("software-benchmarks-*.json"))[-1]
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["schema_version"] == "1.0.0"
    assert len(report["source_commit"]) == 40
    assert report["guarded_cpu_request"]["model_generate_calls"] == 0
    assert report["local_status_suite"]["model_generate_calls"] == 0
    assert report["intent_dispatch"]["failed"] == 0
    assert report["intent_dispatch"]["accuracy"] == 1.0
    world_model = report["learned_risk_world_model"]
    assert world_model["direction_checks_passed"] == len(world_model["direction_checks"])
    assert world_model["validation"]["disk_improvement_percent"] > 0
    assert world_model["validation"]["process_improvement_percent"] > 0
    assert "no claim" in report["claim_boundary"].lower()
    if "coadaptation_simulated_users" in report:
        _assert_coadaptation_contracts(report["coadaptation_simulated_users"])
    if "local_commands" in report:
        _assert_local_command_contracts(report["local_commands"])


def _assert_local_command_contracts(entry: dict) -> None:
    """Every command-guide example is handled in code, with no model call and no action."""
    assert entry["benchmark"] == "local_commands"
    assert entry["handled"] == entry["examples"] > 0
    assert entry["unhandled"] == []
    assert entry["model_generate_calls"] == 0 and entry["actions_executed"] == 0
    assert "no claim" in entry["claim_boundary"]


def test_local_command_contracts_hold_for_a_quick_run() -> None:
    import asyncio

    from benchmarks.local_commands_suite import benchmark_async

    _assert_local_command_contracts(asyncio.run(benchmark_async(iterations=1, warmup=0)))


def _assert_coadaptation_contracts(entry: dict) -> None:
    """Bundles captured after the co-adaptation suite landed carry its quick run: simulated users on the tuning
    families, labelled in-sample, every hard agency invariant at zero and nothing promoted to runtime."""
    assert entry["benchmark"] == "coadaptation_simulated_users"
    assert entry["evidence_kind"] == "simulated"
    assert entry["split"] == "tune" and entry["in_sample"] is True
    assert entry["families"] == ["F1", "F3"]
    boundary = "simulated users only; tuning families, in-sample; no claim about people"
    assert entry["claim_boundary"] == boundary
    assert entry["promoted_to_runtime"] is False
    assert len(entry["protocol_sha256"]) == 64
    for family in entry["families"]:
        assert entry["metrics"]["by_family"][family]["label"] == "tuning families, in-sample"
    for policy, audit in entry["agency_audit"].items():
        assert audit["passed"], policy


def test_coadaptation_contracts_hold_for_a_quick_run_entry() -> None:
    from benchmarks.coadaptation_suite import benchmark

    _assert_coadaptation_contracts(benchmark(quick=True))


def test_committed_subscription_planning_evidence_is_bounded_and_non_destructive() -> None:
    evidence_dir = Path(__file__).resolve().parents[2] / "docs" / "evidence"
    path = sorted(evidence_dir.glob("subscription-planning-*.json"))[-1]
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["schema_version"] == "1.0.0"
    assert len(report["source_commit"]) == 40
    assert report["passed"] == report["case_count"] == len(report["results"])
    assert report["scope"] == "side-effect-free planning only; no action was executed"
    assert all(result["passed"] for result in report["results"])
    assert all(result["destructive_actions"] == 0 for result in report["results"])
    assert "one developer-machine" in report["claim_boundary"].lower()
