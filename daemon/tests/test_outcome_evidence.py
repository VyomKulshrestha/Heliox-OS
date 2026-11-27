"""What the world model learns from finished steps: only harm in the same place may ask for approval.

Fakes and a held clock only; nothing is copied, written or run.
"""

from __future__ import annotations

import os

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    BrowserParams,
    FileParams,
    VerificationResult,
)
from pilot.agents.destructive_critic import HEURISTIC_RISK_THRESHOLD
from pilot.config import PilotConfig
from pilot.intelligence.experience import experience_scope
from pilot.security.outcome_evidence import (
    HALF_LIFE_SECONDS,
    PAUSE_SCORE,
    Outcome,
    OutcomeEvidence,
    failure_outcome,
    outcomes_of,
    scope_of,
)
from pilot.security.risk_gate import RiskGate
from pilot.security.risk_observation import OsSnapshot

WORK = os.path.abspath(os.path.join("work"))
NOTES = os.path.join(WORK, "notes")
BIG = os.path.join(WORK, "big")


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def _copy(source: str, destination: str) -> Action:
    return Action(
        action_type=ActionType.FILE_COPY,
        target=source,
        parameters=FileParams(path=source, destination=destination),
    )


def _failed(action: Action, error: str, *, executed: bool = True) -> ActionResult:
    return ActionResult(action=action, success=False, executed=executed, error=error)


def _checked(action: Action, *, observed: bool, passed: bool) -> tuple[list[ActionResult], VerificationResult]:
    results = [ActionResult(action=action, success=True, output="Copied")]
    verification = VerificationResult(
        passed=passed,
        failed_actions=[] if passed else [0],
        independently_observed_actions=[0] if observed else [],
    )
    return results, verification


@pytest.fixture
def gate(monkeypatch) -> RiskGate:
    monkeypatch.setattr(
        "pilot.security.risk_gate.capture_os_snapshot",
        lambda: OsSnapshot(proc_count=100, disk_usage_fraction=0.5, memory_usage_fraction=0.5, disk_path="/"),
    )
    gate = RiskGate(weights_path="/nonexistent.npz")
    # No folder is measured: the size lookahead stays out of these plans.
    gate._measure_tree_size = lambda *args, **kwargs: None
    gate._volume_of = lambda path: None
    return gate


def _risk(gate: RiskGate, action: Action) -> tuple[float, list[str]]:
    return gate.evaluate_plan(ActionPlan(actions=[action], raw_input="copy"), PilotConfig())


def test_the_pause_score_asks_exactly_as_other_world_model_pauses_do():
    assert PAUSE_SCORE >= HEURISTIC_RISK_THRESHOLD


# ── classifying a failure ──


@pytest.mark.parametrize(
    ("error", "executed"),
    [
        ("Validation failed: Action [0]: Empty target", False),
        ("Source not found: C:\\work\\notes\\does_not_exist.txt", True),
        ("[Errno 13] Permission denied: 'C:\\\\work\\\\readonly\\\\blocked.txt'", True),
        ("[WinError 5] Access is denied", True),
        ("Refused by the person", True),
        ("Timed out", False),
    ],
)
def test_a_failure_that_stopped_before_writing_changed_nothing(error, executed):
    outcome, _cause = failure_outcome(_failed(_copy("a", "b"), error, executed=executed))
    assert outcome is Outcome.FAILED_UNCHANGED


@pytest.mark.parametrize("error", ["[Errno 28] No space left on device", "Timed out after 30 s", "Broken pipe"])
def test_a_failure_of_unknown_cause_after_starting_may_have_left_something_half_done(error):
    outcome, _cause = failure_outcome(_failed(_copy("a", "b"), error))
    assert outcome is Outcome.FAILED_PARTIAL


def test_successful_steps_take_their_outcome_from_the_check_not_from_the_handler():
    a, b, c, d = (_copy(f"s{i}", f"d{i}") for i in range(4))
    results = [
        ActionResult(action=a, success=True, output="ok"),
        ActionResult(action=b, success=True, output="ok"),
        ActionResult(action=c, success=True, output="ok"),
        ActionResult(action=d, success=False, error="Source not found"),
    ]
    verification = VerificationResult(passed=False, failed_actions=[1, 3], independently_observed_actions=[0, 1])

    assert [outcome for _action, outcome in outcomes_of(results, verification)] == [
        Outcome.VERIFIED,
        Outcome.MISMATCH,
        Outcome.UNCHECKED,
    ]  # the failed step is recorded as it fails, not here


def test_a_dry_run_or_a_step_still_being_checked_teaches_nothing():
    action = _copy("a", "b")
    results = [ActionResult(action=action, success=True, output="(dry run) would copy")]
    assert outcomes_of(results, VerificationResult(passed=True)) == []
    results = [ActionResult(action=action, success=True, output="Copied")]
    assert outcomes_of(results, VerificationResult(passed=True, pending_actions=[0])) == []


# ── the run of 26 November ──


def test_four_harmless_copy_failures_then_a_copy_of_an_existing_file_shows_no_card(gate):
    missing = _copy(os.path.join(NOTES, "does_not_exist.txt"), os.path.join(NOTES, "x.txt"))
    gate.record_step(_failed(missing, "Validation failed: Action [0]: Empty target", executed=False))
    for _ in range(3):
        gate.record_step(_failed(missing, f"Source not found: {missing.target}"))
    readonly = Action(
        action_type=ActionType.FILE_WRITE,
        target=os.path.join(WORK, "readonly", "blocked.txt"),
        parameters=FileParams(path=os.path.join(WORK, "readonly", "blocked.txt"), content="hello"),
    )
    for _ in range(4):
        gate.record_step(_failed(readonly, "[Errno 13] Permission denied"))

    existing = _copy(os.path.join(BIG, "big300.bin"), os.path.join(BIG, "big300_copy.bin"))
    risk, reasons = _risk(gate, existing)
    assert risk < HEURISTIC_RISK_THRESHOLD
    assert not any("callbacks" in reason for reason in reasons)

    # Even the same missing file again asks nothing: the failure is said in its own reply.
    risk, _reasons = _risk(gate, missing)
    assert risk < HEURISTIC_RISK_THRESHOLD
    risk, _reasons = _risk(gate, readonly)
    assert risk < HEURISTIC_RISK_THRESHOLD
    notes = gate.last_evaluation["predictions"][0].get("notes")
    assert notes and "failed before changing anything" in notes[0] and "(no permission)" in notes[0]


# ── harm pauses, in the person's words ──


def test_two_mismatched_copies_into_one_folder_pause_the_next_copy_there_with_a_plain_sentence(gate):
    copy = _copy(os.path.join(NOTES, "a.txt"), os.path.join(NOTES, "b.txt"))
    gate.record_checked(*_checked(copy, observed=True, passed=False))
    risk, _reasons = _risk(gate, copy)
    assert risk < HEURISTIC_RISK_THRESHOLD  # one is not enough to speak

    gate.record_checked(*_checked(copy, observed=True, passed=False))
    another = _copy(os.path.join(WORK, "elsewhere.txt"), os.path.join(NOTES, "c.txt"))
    risk, reasons = _risk(gate, another)

    assert risk >= HEURISTIC_RISK_THRESHOLD
    assert reasons[0] == "The last two copies into this folder ended with a different file than the source."
    assert "%" not in reasons[0] and "callback" not in reasons[0]


def test_harm_in_one_folder_says_nothing_about_another_folder(gate):
    copy = _copy(os.path.join(NOTES, "a.txt"), os.path.join(NOTES, "b.txt"))
    for _ in range(3):
        gate.record_checked(*_checked(copy, observed=True, passed=False))

    risk, _reasons = _risk(gate, _copy(os.path.join(NOTES, "a.txt"), os.path.join(BIG, "a.txt")))
    assert risk < HEURISTIC_RISK_THRESHOLD


def test_steps_that_stopped_part_way_pause(gate):
    copy = _copy(os.path.join(NOTES, "a.txt"), os.path.join(BIG, "a.txt"))
    for _ in range(2):
        gate.record_step(_failed(copy, "No space left on device"))
    risk, reasons = _risk(gate, copy)
    assert risk >= HEURISTIC_RISK_THRESHOLD
    assert reasons[0] == "The last two copies into this folder stopped part way and may have left something half done."


def test_steps_the_person_undid_pause(gate):
    copy = _copy(os.path.join(NOTES, "a.txt"), os.path.join(BIG, "a.txt"))
    gate.record_undone(copy)
    gate.record_undone(copy)
    risk, reasons = _risk(gate, copy)
    assert risk >= HEURISTIC_RISK_THRESHOLD
    assert reasons[0] == "The last two copies into this folder were undone by you afterwards."


def test_unchecked_successes_count_neither_way(gate):
    copy = _copy(os.path.join(NOTES, "a.txt"), os.path.join(BIG, "a.txt"))
    for _ in range(10):
        gate.record_checked(*_checked(copy, observed=False, passed=True))
    risk, _reasons = _risk(gate, copy)
    assert risk < HEURISTIC_RISK_THRESHOLD


def test_verified_runs_and_approvals_count_the_other_way(gate):
    copy = _copy(os.path.join(NOTES, "a.txt"), os.path.join(BIG, "a.txt"))
    gate.record_checked(*_checked(copy, observed=True, passed=False))
    gate.record_checked(*_checked(copy, observed=True, passed=False))
    assert _risk(gate, copy)[0] >= HEURISTIC_RISK_THRESHOLD

    gate.record_checked(*_checked(copy, observed=True, passed=True))
    gate.record_approved([copy])
    assert _risk(gate, copy)[0] < HEURISTIC_RISK_THRESHOLD  # two good against two harmful: harm no longer outweighs


def test_interleaved_outcomes_are_not_called_the_last_ones():
    clock = _Clock()
    evidence = OutcomeEvidence(clock=clock)
    copy = _copy("a", os.path.join(BIG, "a.txt"))
    evidence.record(copy, Outcome.MISMATCH)
    evidence.record(copy, Outcome.MISMATCH)
    evidence.record(copy, Outcome.VERIFIED)
    assessment = evidence.assess(copy)
    assert assessment.reason == "Two recent copies into this folder ended with a different file than the source."


def test_old_evidence_fades():
    clock = _Clock()
    evidence = OutcomeEvidence(clock=clock)
    copy = _copy("a", os.path.join(BIG, "a.txt"))
    evidence.record(copy, Outcome.MISMATCH)
    evidence.record(copy, Outcome.MISMATCH)
    assert evidence.assess(copy).score >= HEURISTIC_RISK_THRESHOLD

    clock.now += HALF_LIFE_SECONDS  # each now weighs one half: 1.0 in all, below the 1.5 needed
    assert evidence.assess(copy).score == 0.0


def test_scopes_follow_where_the_step_acts():
    left = scope_of(_copy(os.path.join(NOTES, "a.txt"), os.path.join(BIG, "a.txt")))
    right = scope_of(_copy(os.path.join(WORK, "b.txt"), os.path.join(BIG, "b.txt")))
    assert left == right  # both land in the same folder
    web = Action(
        action_type=ActionType.BROWSER_NAVIGATE,
        target="https://Example.com/page",
        parameters=BrowserParams(url="https://Example.com/page"),
    )
    assert scope_of(web).place == "example.com"


# ── one real outcome is one piece of evidence (the second live run, 27 November) ──


def _plan_checks(gate: RiskGate, actions: list[Action], kinds: list[str]) -> None:
    """One plan's reply-time checks: ``verified``, ``mismatch``, ``could_not_look`` or ``still_checking`` per step."""
    results = [ActionResult(action=action, success=True, output="Copied") for action in actions]
    verification = VerificationResult(
        passed="mismatch" not in kinds,
        failed_actions=[i for i, kind in enumerate(kinds) if kind == "mismatch"],
        independently_observed_actions=[i for i, kind in enumerate(kinds) if kind in ("verified", "mismatch")],
        pending_actions=[i for i, kind in enumerate(kinds) if kind == "still_checking"],
    )
    gate.record_checked(results, verification)


def test_the_runs_copies_into_one_folder_count_each_real_step_once(gate):
    source = os.path.join(BIG, "big300.bin")
    tampered = _copy(source, os.path.join(BIG, "big300_tampered.bin"))
    with experience_scope(task_id="f8"):
        _plan_checks(gate, [_copy(source, os.path.join(BIG, "big300_copy.bin"))], ["could_not_look"])
    with experience_scope(task_id="f9"):
        _plan_checks(gate, [tampered], ["mismatch"])
        # The repair round re-runs the same copy: not paused by the mismatch it is repairing, and its own mismatch
        # is the same step's outcome, not a second copy.
        assert _risk(gate, tampered)[0] < HEURISTIC_RISK_THRESHOLD
        _plan_checks(gate, [tampered], ["mismatch"])
        assert _risk(gate, tampered)[0] < HEURISTIC_RISK_THRESHOLD
    late = [_copy(source, os.path.join(BIG, f"late_{name}.bin")) for name in "abc"]
    with experience_scope(task_id="f10"):
        assert _risk(gate, late[0])[0] < HEURISTIC_RISK_THRESHOLD  # one real mismatch so far
        _plan_checks(gate, late, ["could_not_look"] * 3)  # "not verified in full" is not harm
    late_f = _copy(source, os.path.join(BIG, "late_tampered_f.bin"))
    with experience_scope(task_id="f11"):
        _plan_checks(
            gate,
            [_copy(source, os.path.join(BIG, "late_d.bin")), _copy(source, os.path.join(BIG, "late_e.bin")), late_f],
            ["could_not_look", "could_not_look", "mismatch"],
        )
        assert _risk(gate, late_f)[0] < HEURISTIC_RISK_THRESHOLD  # its repair: not paused by itself
        _plan_checks(gate, [late_f], ["mismatch"])
    with experience_scope(task_id="f12"):
        risk, reasons = _risk(gate, _copy(source, os.path.join(BIG, "slowdisk_copy.bin")))
        _plan_checks(gate, [_copy(source, os.path.join(BIG, "slowdisk_copy.bin"))], ["still_checking"])

    # Two copies into big ended with a different file (f9's and f11's), whatever their repairs did: said as two.
    assert risk >= HEURISTIC_RISK_THRESHOLD
    assert reasons[0] == "Two recent copies into this folder ended with a different file than the source."
    assert gate.status(enabled=True)["historical_risk"]["outcomes"]["mismatch"] == 2


def test_one_mismatch_reported_by_several_paths_is_one_piece_of_harm():
    evidence = OutcomeEvidence(clock=_Clock())
    copy = _copy("a", os.path.join(BIG, "a.txt"))
    other = _copy("b", os.path.join(BIG, "b.txt"))
    # The check, a late check, the repair's re-run and an undo of the same step in one task.
    for outcome in (Outcome.MISMATCH, Outcome.MISMATCH, Outcome.MISMATCH, Outcome.UNDONE):
        evidence.record(copy, outcome, task="t1")
    evidence.record(copy, Outcome.APPROVED, task="t1")
    evidence.record(copy, Outcome.APPROVED, task="t1")
    assessment = evidence.assess(other, task="t2")
    assert (assessment.harmful, assessment.supporting, assessment.score) == (1, 1, 0.0)


def test_a_count_that_is_no_longer_exact_is_said_as_several():
    evidence = OutcomeEvidence(clock=_Clock())
    copy = _copy("a", os.path.join(BIG, "a.txt"))
    for index in range(2):
        evidence.record(_copy("a", os.path.join(BIG, f"old{index}.txt")), Outcome.MISMATCH, task=f"old{index}")
    for index in range(48):
        evidence.record(_copy("a", os.path.join(BIG, f"n{index}.txt")), Outcome.UNCHECKED, task=f"n{index}")
    for index in range(3):  # the first two harmful records fall out of the scope's 50
        evidence.record(_copy("a", os.path.join(BIG, f"new{index}.txt")), Outcome.MISMATCH, task=f"new{index}")
    assessment = evidence.assess(copy, task="now")
    assert assessment.score > 0
    assert assessment.reason == "Several recent copies into this folder ended with a different file than the source."
