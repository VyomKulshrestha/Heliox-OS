"""The pre-registered analysis script (spec section 21.2) on synthetic encrypted bundles.

The bundles come from simulated people relabelled as recorded data and are built only with the records' to_dict()
methods, never hand-written dictionaries; the encryption follows the export format of section 10.9.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import scripts.evaluate_coadaptation_study as study
from pilot.coadaptation.core.evidence import EvidenceKind
from pilot.coadaptation.core.preferences import Preference, PreferenceKind, PreferenceScope
from pilot.coadaptation.core.simulator import (
    SIM_EPOCH,
    EnginePolicy,
    SimulatedPerson,
    SimulatedStudyRunner,
    make_population,
)
from pilot.coadaptation.core.utility import UtilityWeights
from scripts.evaluate_coadaptation_study import (
    AnalysisRefused,
    ScoredTrial,
    audit_bundles,
    block_type_switch_shares,
    decrypt_bundle,
    gate_g5,
    h2_ope_report,
    phase_b_contrasts,
    protocol_sha256,
    read_human_bundle,
    run,
    score_trials,
    verify_logs,
)

PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "coadaptation_protocol_v1.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
KEY = bytes(range(32))


def _priorities(participant_key: str) -> Preference:
    return Preference(
        pref_id=f"prio-{participant_key}",
        kind=PreferenceKind.PRIORITIES,
        scope=PreferenceScope(context="study", modality=None, participant_key=participant_key),
        value={"fewer_mistakes": 1.0, "faster": 0.0, "fewer_interruptions": 1.0},
        origin="explicit_user",
        created_at=SIM_EPOCH - timedelta(hours=12),
    )


@pytest.fixture(scope="module")
def bundles() -> list[dict[str, Any]]:
    out = []
    population = make_population(["F1"], 10, 31)
    plain_people = [person for person in population if not person.stated_preferences][:2]
    for index, person in enumerate(plain_people):
        if index == 0:
            person = replace(
                person,
                stated_preferences=(
                    *person.stated_preferences,
                    _priorities(person.participant_key),
                ),
            )
        result = SimulatedStudyRunner(SimulatedPerson(person), lambda: EnginePolicy(), seed=7).run(
            sessions=3,
        )
        human = EvidenceKind.RECORDED_HUMAN
        result.sessions = [replace(row, evidence_kind=human) for row in result.sessions]
        out.append(result.bundle())
    return out


def _encrypt(bundle: dict[str, Any], path: Path, *, key: bytes = KEY) -> Path:
    participant_key = bundle["sessions"][0]["participant_key"]
    nonce = os.urandom(12)
    header = {
        "schema_version": 1,
        "participant_key": participant_key,
        "created_at": "2026-10-15T10:00:00+00:00",
        "cipher": "AES-256-GCM",
        "nonce": base64.b64encode(nonce).decode("ascii"),
    }
    header_line = json.dumps(header, sort_keys=True, separators=(",", ":"))
    plaintext = json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, header_line.encode("utf-8"))
    path.write_bytes(
        (header_line + "\n" + base64.b64encode(ciphertext).decode("ascii") + "\n").encode("utf-8"),
    )
    return path


def test_decrypt_bundle_reads_the_export_format(tmp_path: Path, bundles) -> None:
    path = _encrypt(bundles[0], tmp_path / "export.enc")
    assert decrypt_bundle(path, KEY) == json.loads(json.dumps(bundles[0]))
    encoded = base64.b64encode(KEY).decode("ascii")
    assert decrypt_bundle(path, key_lookup=lambda participant: encoded) == decrypt_bundle(path, KEY)
    with pytest.raises(AnalysisRefused):
        decrypt_bundle(path, key_lookup=lambda participant: None)


@pytest.mark.parametrize(
    ("key", "value"),
    [("bundle_schema", 2), ("record_schema", "coadapt-records-v0")],
)
def test_unknown_schemas_are_refused(tmp_path: Path, bundles, key: str, value: Any) -> None:
    path = _encrypt({**bundles[0], key: value}, tmp_path / "export.enc")
    with pytest.raises(AnalysisRefused):
        decrypt_bundle(path, KEY)


def test_simulated_bundles_are_refused(bundles) -> None:
    simulated = json.loads(json.dumps(bundles[0]))
    for row in simulated["sessions"]:
        row["evidence_kind"] = "simulated"
    with pytest.raises(AnalysisRefused, match="simulated"):
        read_human_bundle(simulated)


def test_outputs_are_never_overwritten(tmp_path: Path, bundles) -> None:
    path = _encrypt(bundles[0], tmp_path / "export.enc")
    output = tmp_path / "out"
    output.mkdir()
    (output / "results.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AnalysisRefused, match="fresh"):
        run([path], PROTOCOL_PATH, output, exploratory=True, keys={"": KEY})


def test_tampered_seeds_and_draws_are_reported(bundles) -> None:
    clean = verify_logs(read_human_bundle(bundles[0]))
    assert clean["passed"], clean["mismatches"]
    tampered = json.loads(json.dumps(bundles[0]))
    first_l1 = next(d for d in tampered["decisions"] if d["level"] == "block_regime")
    first_l1["payload"]["seed"] = "0" * 16
    table = next(d for d in tampered["decisions"] if d["level"] == "ambiguity_response")
    table["payload"]["draws"][0][0] = 0.123456789
    report = verify_logs(read_human_bundle(tampered))
    assert not report["passed"]
    assert first_l1["decision_id"] in report["mismatches"]["seed"]
    assert table["decision_id"] in report["mismatches"]["table_draws"]


def test_realized_utility_uses_logged_priorities_first_and_protocol_weights_second(bundles) -> None:
    rows = read_human_bundle(bundles[0])
    scored, counts = score_trials(rows)
    assert scored
    assert any(abs(t.utility_participant - t.utility_protocol) > 1e-9 for t in scored)
    assert set(counts) == {"imputed_blocks_by_regime", "missingness_by_regime"}
    logged = UtilityWeights.from_priorities(1.0, 0.0, 1.0)
    plain = [t for t in scored if t.regime == "assistance_off"]
    assert plain
    reports = {(r.session_id, r.block_index): r for r in rows.burden}
    numbers = {row.session_number: row.session_id for row in rows.sessions}
    checked = 0
    for trial in plain:
        report = reports.get((numbers[trial.session_number], trial.block_index))
        if report is None or report.effort is None:
            continue
        # Intrusiveness is 1 by definition after an assistance_off block (B2 is not asked there).
        expected = logged.w_b * ((report.effort - 4) / 3 + (1 - 4) / 3)
        assert trial.burden_participant == pytest.approx(expected)
        checked += 1
    assert checked
    without = json.loads(json.dumps(bundles[0]))
    burden = without["burden"]
    rated = next(i for i, r in enumerate(burden) if None not in (r["effort"], r["intrusiveness"]))
    del burden[rated]
    _, fewer = score_trials(read_human_bundle(without))
    imputed = sum(counts["imputed_blocks_by_regime"].values())
    assert sum(fewer["imputed_blocks_by_regime"].values()) == imputed + 1


def _scored(
    participant: str,
    phase: str,
    trial_index: int,
    utility: float,
    *,
    switched: bool = True,
) -> ScoredTrial:
    return ScoredTrial(
        participant_key=participant,
        session_number=2,
        block_index={"B": 8, "probe": 9, "comparator": 10, "A": 1}[phase],
        phase=phase,
        trial_index=trial_index,
        regime="act",
        verification="ok",
        modality="mouse",
        utility_participant=utility,
        utility_protocol=utility / 2,
        burden_participant=0.0,
        burden_protocol=0.0,
        ua=utility < 0,
        override=False,
        switched=switched,
    )


def test_phase_b_contrasts_use_concurrent_blocks_and_the_burn_in() -> None:
    scored = []
    for participant, shift in (("p-00000001", 0.0), ("p-00000002", 0.2)):
        for t in range(8):
            scored.append(_scored(participant, "B", t, -10.0 if t < 3 else 1.0 + shift))
            scored.append(_scored(participant, "probe", t, 0.0))
            scored.append(_scored(participant, "comparator", t, 0.5))
            scored.append(_scored(participant, "A", t, 100.0))  # Phase A never enters the contrasts
    primary = phase_b_contrasts(scored, {}, burn_in=3, bootstrap_n=200)
    assert primary["h2_vs_probe"]["mean"] == pytest.approx(1.1)
    assert primary["h2_vs_comparator"]["mean"] == pytest.approx(0.6)
    every = phase_b_contrasts(scored, {}, burn_in=0, bootstrap_n=200)
    assert every["h2_vs_probe"]["mean"] == pytest.approx((3 * -10.0 + 5 * 1.1) / 8)
    protocol = phase_b_contrasts(scored, {}, burn_in=3, weights="protocol", bootstrap_n=200)
    assert protocol["h2_vs_probe"]["mean"] == pytest.approx(0.55)
    assert primary["h6_ua"]["mean"] == pytest.approx(0.0)


def test_block_type_switch_shares_are_reported(bundles) -> None:
    shares = block_type_switch_shares([read_human_bundle(b) for b in bundles], burn_in=3)
    assert {"B:session_2", "probe:session_2", "comparator:session_2"} <= set(shares)
    for value in shares.values():
        assert 0.0 <= value["followed_regime_change"] <= 1.0
        assert value["mean_switch_age_at_first_scored_trial"] >= 3


VALID = {
    "estimator_check_coverage": 45 / 50,
    "in_sample": {"S_2": {"coverage": 44 / 50}, "S_3": {"coverage": 43 / 50}},
}


def _ope(ess2: float = 40.0, ess3: float = 50.0, oos_ess: float = 35.0) -> dict[str, Any]:
    def entry(ess: float) -> dict[str, Any]:
        refused = "low_ess" if ess < 30 else None
        return {
            "estimate": {"value": None if refused else 0.4, "refused": refused, "ess": ess},
            "ess": ess,
        }

    return {
        "in_sample_S_2": entry(ess2),
        "in_sample_S_3": entry(ess3),
        "out_of_sample_S_2": entry(oos_ess),
    }


def test_g5_passes_only_when_every_condition_holds() -> None:
    assert gate_g5(VALID, _ope())["passed"]
    failures = {
        "a_estimator_check": ({**VALID, "estimator_check_coverage": 42 / 50}, _ope()),
        "b_in_sample_S_2_coverage": (
            {**VALID, "in_sample": {**VALID["in_sample"], "S_2": {"coverage": 0.8}}},
            _ope(),
        ),
        "b_in_sample_S_3_coverage": (
            {**VALID, "in_sample": {**VALID["in_sample"], "S_3": {"coverage": None}}},
            _ope(),
        ),
        "c_human_in_sample_S_2_computed": (VALID, _ope(ess2=12.0)),
        "c_human_in_sample_S_3_computed": (VALID, _ope(ess3=29.9)),
    }
    for condition, (validation, ope) in failures.items():
        result = gate_g5(validation, ope)
        assert result["result"] == "not passed"
        assert result["failed_conditions"] == [condition]
    # The draft has no validation yet.
    assert gate_g5(PROTOCOL["ope_validation"], _ope())["result"] == "not passed"


def test_h2_ope_reporting_rules() -> None:
    passed = gate_g5(VALID, _ope())
    report = h2_ope_report(passed, _ope())
    assert report["in_sample_S_2"]["value"] == 0.4 and report["out_of_sample_S_2"]["g5"] == "passed"
    failed = gate_g5(VALID, _ope(ess2=10.0))
    report = h2_ope_report(failed, _ope(ess2=10.0))
    assert report["in_sample_S_2"]["reported"] is False and report["in_sample_S_2"]["ess"] == 10.0
    # Reported when the estimator check passed.
    assert report["out_of_sample_S_2"]["g5"] == "not passed"
    low = h2_ope_report(passed, _ope(oos_ess=11.0))
    assert low["out_of_sample_S_2"]["reported"] is False
    assert "not estimated" in low["out_of_sample_S_2"]["reason"]
    no_check = gate_g5({**VALID, "estimator_check_coverage": 0.5}, _ope())
    assert h2_ope_report(no_check, _ope())["out_of_sample_S_2"]["reported"] is False


def test_the_audit_receives_consents_rules_events_and_the_reset_time(bundles, monkeypatch) -> None:
    rows = read_human_bundle(bundles[0])
    seen: dict[str, Any] = {}
    real = study.agency_audit

    def recorder(*args: Any, **kwargs: Any) -> Any:
        seen["args"] = args
        seen["kwargs"] = kwargs
        return real(*args, **kwargs)

    monkeypatch.setattr(study, "agency_audit", recorder)
    report = audit_bundles([rows])
    assert report["passed"], report
    decisions, trials, sessions, burden, preferences, consents = seen["args"]
    assert consents == rows.consents and preferences == rows.preferences
    assert seen["kwargs"] == {"events": rows.events, "learning_reset_at": rows.learning_reset_at}
    unconsented = json.loads(json.dumps(bundles[0]))
    unconsented["consents"] = []
    assert audit_bundles([read_human_bundle(unconsented)])["counts"]["A6"] > 0


def test_end_to_end_exploratory_run(tmp_path: Path, bundles) -> None:
    paths = [_encrypt(bundle, tmp_path / f"export_{i}.enc") for i, bundle in enumerate(bundles)]
    keys = {bundle["sessions"][0]["participant_key"]: KEY for bundle in bundles}
    output = tmp_path / "results"
    results = run(paths, PROTOCOL_PATH, output, exploratory=True, keys=keys)
    assert (output / "results.json").exists() and (output / "results.md").exists()
    written = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert written["protocol_sha256"] == protocol_sha256(PROTOCOL)
    assert written["protocol_sha256"] in (output / "results.md").read_text(encoding="utf-8")
    assert written["mode"] == "exploratory" and "exploratory" in written["exploratory_label"]
    assert written["promoted_to_runtime"] is False
    assert set(written["gates"]) == {"G1", "G2", "G3", "G4", "G5", "G6", "G5_detail"}
    assert all(isinstance(written["gates"][g], bool) for g in ("G1", "G2", "G3", "G4", "G5", "G6"))
    assert written["gates"]["G5"] is False  # the draft protocol holds no validation figures
    assert written["h2_ope"]["in_sample_S_2"]["reported"] is False
    assert all(v["passed"] for v in written["verification"]["draws"].values())
    assert all(v["passed"] for v in written["verification"]["posteriors"].values())
    assert written["agency_audit"]["passed"]
    assert written["evidence_kind"] == ["recorded_human"]
    hypotheses = written["hypotheses"]
    assert hypotheses["phase_b_primary"]["burn_in_trials"] == 3
    assert {"all_trials", "protocol_weights", "switched_blocks_only", "ok_trials_only"} <= set(
        hypotheses["phase_b_secondary"]
    )
    assert {
        "H1_vs_physics_only",
        "H1_vs_person_static",
        "H4_vs_no_today",
        "H8_vs_population",
        "H9_vs_session_only",
    } <= set(hypotheses["prediction"])
    assert set(hypotheses["prediction"]["H9_vs_session_only"]) == {
        "complete",
        "ua",
        "correction",
        "veto",
        "log_time",
        "total",
    }
    assert results["realized_utility"]["counts"]
    for name in ("in_sample_S_2", "in_sample_S_3", "out_of_sample_S_2"):
        entry = written["ope"][name]
        assert "ess" in entry
        if entry["estimate"] and entry["estimate"].get("refused"):
            assert "not estimated" in entry["estimate"]["note"]
