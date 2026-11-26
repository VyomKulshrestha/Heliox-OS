"""Verification does not hold up the reply: checks past the budget go on behind it and only inform.

The budget's wait is replaced by one that returns at a moment the test chooses (an event a fake check sets), so
nothing here depends on how fast the machine is.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, ApiRequestParams, FileParams
from pilot.agents import postconditions
from pilot.agents import verifier as verifier_module
from pilot.agents.late_checks import FollowUp, LateReport, LateVerifications
from pilot.agents.postconditions import Outcome, mismatch, register, verified
from pilot.agents.verifier import Verifier
from pilot.response_contract import success_message


@pytest.fixture
def scratch_registry(monkeypatch: pytest.MonkeyPatch) -> dict:
    registry = dict(postconditions._REGISTRY)
    monkeypatch.setattr(postconditions, "_REGISTRY", registry)
    return registry


def _register(monkeypatch: pytest.MonkeyPatch, action_type: ActionType, check) -> None:
    postconditions._REGISTRY.pop(action_type, None)
    register(action_type, method="test_postcondition")(check)
    monkeypatch.setattr(verifier_module, "_AREA_VERIFIERS", postconditions.registrations())


def _mail() -> Action:
    return Action(
        action_type=ActionType.API_SEND_EMAIL,
        target="sam@example.com",
        parameters=ApiRequestParams(to="sam@example.com", subject="Late", message="Running late"),
    )


def _write(path: str = "notes.txt") -> Action:
    return Action(action_type=ActionType.FILE_WRITE, target=path, parameters=FileParams(path=path, content="x"))


class _Slow:
    """A check that starts, says so, and finishes only when released, with the outcome it is given."""

    def __init__(self, outcome: Outcome) -> None:
        self.outcome = outcome
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False

    async def __call__(self, _result: ActionResult) -> Outcome:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return self.outcome


def _budget_until(event: asyncio.Event):
    async def wait(task: asyncio.Task, _budget: float) -> None:
        waiter = asyncio.ensure_future(event.wait())
        await asyncio.wait({waiter, task}, return_when=asyncio.FIRST_COMPLETED)
        waiter.cancel()

    return wait


async def _late_plan(monkeypatch: pytest.MonkeyPatch, slow: _Slow):
    async def quick(_result: ActionResult) -> Outcome:
        return verified("notes.txt holds what was written")

    _register(monkeypatch, ActionType.FILE_WRITE, quick)
    _register(monkeypatch, ActionType.API_SEND_EMAIL, slow)
    monkeypatch.setattr(Verifier, "_wait_budget", staticmethod(_budget_until(slow.started)))
    actions = [_write(), _mail()]
    results = [ActionResult(action=action, success=True, output="done") for action in actions]
    plan = ActionPlan(actions=actions)
    verification, late = await Verifier(MagicMock()).verify_within(plan, results, budget=3.0)
    return plan, results, verification, late


class _Sink:
    def __init__(self) -> None:
        self.recorded: list[LateReport] = []
        self.told: list[str] = []
        self.shown: list[str] = []
        self.done = asyncio.Event()

    async def record(self, report: LateReport) -> None:
        self.recorded.append(report)
        self.done.set()

    async def tell(self, sentence: str, report: LateReport, speak: bool) -> None:
        (self.told if speak else self.shown).append(sentence)


@pytest.mark.asyncio
async def test_a_check_past_the_budget_is_reported_as_still_being_checked(
    monkeypatch: pytest.MonkeyPatch, scratch_registry: dict
) -> None:
    slow = _Slow(verified("found in Sent"))
    plan, results, verification, late = await _late_plan(monkeypatch, slow)

    assert late is not None and late.pending == [1]
    assert verification.passed is True
    assert verification.independently_observed_actions == [0]
    assert verification.pending_actions == [1]
    assert verification.reason_kinds == {1: "still_checking"}
    assert "CHECKING" in verification.details[1]
    assert success_message(plan, results, verification, dry_run=False).endswith(
        "Note: 1 step is still being checked; details in the receipt."
    )
    await late.cancel()
    assert slow.cancelled is True


@pytest.mark.asyncio
async def test_a_late_mismatch_on_a_sent_mail_is_told_once_and_only_informs(
    monkeypatch: pytest.MonkeyPatch, scratch_registry: dict
) -> None:
    slow = _Slow(mismatch("no copy with that Message-ID is in Sent after 4 searches"))
    _plan, _results, verification, late = await _late_plan(monkeypatch, slow)
    sink = _Sink()
    follower = LateVerifications(record=sink.record, tell=sink.tell)
    follow_up = FollowUp(task_key="task-1", plan_id="plan-1", request="email Sam", receipt_id="abc")

    follower.follow(late, follow_up)
    slow.release.set()
    await sink.done.wait()
    await follower.stop()

    assert sink.told == ["I checked the mail I sent: its copy in Sent is not what was sent."]
    [report] = sink.recorded
    assert report.follow_up == follow_up
    assert report.checks == [
        {
            "step": 2,
            "action_type": "api_send_email",
            "kind": "mismatch",
            "reason": "no copy with that Message-ID is in Sent after 4 searches",
            "late": True,
        }
    ]
    assert report.summary == "Checked after the reply: 1 not as asked."
    # The reply's own result is untouched: a late outcome never fails the plan or starts a repair.
    assert verification.passed is True and verification.failed_actions == []


@pytest.mark.asyncio
async def test_a_late_verified_is_kept_and_not_said(monkeypatch: pytest.MonkeyPatch, scratch_registry: dict) -> None:
    slow = _Slow(verified("The copy with Message-ID <a@b> is in Sent"))
    _plan, _results, _verification, late = await _late_plan(monkeypatch, slow)
    sink = _Sink()
    follower = LateVerifications(record=sink.record, tell=sink.tell)

    follower.follow(late, FollowUp(task_key="t", plan_id="p", request="email Sam"))
    slow.release.set()
    await sink.done.wait()
    await follower.stop()

    assert sink.told == [] and sink.shown == []
    assert [check["kind"] for check in sink.recorded[0].checks] == ["verified"]


@pytest.mark.asyncio
async def test_a_late_upgrade_with_packages_left_shows_them_and_says_nothing(
    monkeypatch: pytest.MonkeyPatch, scratch_registry: dict
) -> None:
    left = "2 packages were not upgraded: Spotify, Discord"
    slow = _Slow(verified(f"The upgrade ran; {left}", notice=left))
    _register(monkeypatch, ActionType.PACKAGE_UPDATE, slow)
    monkeypatch.setattr(Verifier, "_wait_budget", staticmethod(_budget_until(slow.started)))
    from pilot.actions import PackageParams

    action = Action(action_type=ActionType.PACKAGE_UPDATE, target="all", parameters=PackageParams())
    _verification, late = await Verifier(MagicMock()).verify_within(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=True, output="done")], budget=3.0
    )
    sink = _Sink()
    follower = LateVerifications(record=sink.record, tell=sink.tell)

    follower.follow(late, FollowUp(task_key="t", plan_id="p", request="update my apps"))
    slow.release.set()
    await sink.done.wait()
    await follower.stop()

    assert sink.told == []
    assert sink.shown == [left]
    assert sink.recorded[0].checks[0]["notice"] == left


@pytest.mark.asyncio
async def test_a_late_mismatch_on_a_step_that_changes_nothing_for_others_is_not_said(
    monkeypatch: pytest.MonkeyPatch, scratch_registry: dict
) -> None:
    slow = _Slow(mismatch("Copy destination content mismatch"))
    _register(monkeypatch, ActionType.FILE_COPY, slow)
    monkeypatch.setattr(Verifier, "_wait_budget", staticmethod(_budget_until(slow.started)))
    action = Action(action_type=ActionType.FILE_COPY, target="a", parameters=FileParams(path="a", destination="b"))
    _verification, late = await Verifier(MagicMock()).verify_within(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=True, output="Copied")], budget=3.0
    )
    sink = _Sink()
    follower = LateVerifications(record=sink.record, tell=sink.tell)

    follower.follow(late, FollowUp(task_key="t", plan_id="p", request="copy a"))
    slow.release.set()
    await sink.done.wait()
    await follower.stop()

    assert sink.told == []
    assert sink.recorded[0].checks[0]["kind"] == "mismatch"


@pytest.mark.asyncio
async def test_stopping_cancels_late_checks_and_waits_for_them(
    monkeypatch: pytest.MonkeyPatch, scratch_registry: dict
) -> None:
    slow = _Slow(verified("never reached"))
    _plan, _results, _verification, late = await _late_plan(monkeypatch, slow)
    sink = _Sink()
    follower = LateVerifications(record=sink.record, tell=sink.tell)
    follower.follow(late, FollowUp(task_key="t", plan_id="p", request="email Sam"))

    await follower.stop()

    assert slow.cancelled is True
    assert late.task.done()
    assert follower.running == 0
    assert sink.recorded == [] and sink.told == []

    # Nothing starts after a stop: a later follow cancels its checks at once.
    second = _Slow(verified("never reached"))
    _plan, _results, _verification, late_after = await _late_plan(monkeypatch, second)
    follower.follow(late_after, FollowUp(task_key="t2", plan_id="p2", request="again"))
    await asyncio.wait({late_after.task})
    assert late_after.task.cancelled()
    assert follower.running == 0


@pytest.mark.asyncio
async def test_a_late_check_past_the_ceiling_is_kept_as_not_finished(
    monkeypatch: pytest.MonkeyPatch, scratch_registry: dict
) -> None:
    slow = _Slow(verified("never reached"))
    _plan, _results, _verification, late = await _late_plan(monkeypatch, slow)
    sink = _Sink()
    follower = LateVerifications(record=sink.record, tell=sink.tell, ceiling=0)

    follower.follow(late, FollowUp(task_key="t", plan_id="p", request="email Sam"))
    await sink.done.wait()
    await follower.stop()

    assert slow.cancelled is True
    assert sink.told == []
    assert sink.recorded[0].checks[0]["kind"] == "could_not_look"
    assert "did not finish" in sink.recorded[0].checks[0]["reason"]


@pytest.mark.asyncio
async def test_each_follow_up_keeps_its_own_plan(monkeypatch: pytest.MonkeyPatch, scratch_registry: dict) -> None:
    first = _Slow(mismatch("no copy in Sent"))
    _p, _r, _v, late_one = await _late_plan(monkeypatch, first)
    second = _Slow(verified("found"))
    _p, _r, _v, late_two = await _late_plan(monkeypatch, second)
    sink = _Sink()
    follower = LateVerifications(record=sink.record, tell=sink.tell)

    follower.follow(late_one, FollowUp(task_key="t1", plan_id="p1", request="email Sam"))
    follower.follow(late_two, FollowUp(task_key="t2", plan_id="p2", request="email Ana"))
    second.release.set()
    first.release.set()
    while len(sink.recorded) < 2:
        sink.done.clear()
        await sink.done.wait()
    await follower.stop()

    by_plan = {report.follow_up.plan_id: report for report in sink.recorded}
    assert by_plan["p1"].checks[0]["kind"] == "mismatch" and by_plan["p1"].told
    assert by_plan["p2"].checks[0]["kind"] == "verified" and not by_plan["p2"].told
    assert len(sink.told) == 1


@pytest.mark.asyncio
async def test_a_cancelled_verification_stops_its_own_checks(
    monkeypatch: pytest.MonkeyPatch, scratch_registry: dict
) -> None:
    slow = _Slow(verified("never reached"))
    _register(monkeypatch, ActionType.API_SEND_EMAIL, slow)

    async def never(task: asyncio.Task, _budget: float) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(Verifier, "_wait_budget", staticmethod(never))
    action = _mail()
    verifying = asyncio.ensure_future(
        Verifier(MagicMock()).verify_within(
            ActionPlan(actions=[action]), [ActionResult(action=action, success=True, output="sent")], budget=3.0
        )
    )
    await slow.started.wait()
    verifying.cancel()
    await asyncio.wait({verifying})

    assert verifying.cancelled()
    assert slow.cancelled is True


@pytest.mark.asyncio
async def test_without_a_budget_every_check_is_waited_for(
    monkeypatch: pytest.MonkeyPatch, scratch_registry: dict
) -> None:
    async def done(_result: ActionResult) -> Outcome:
        return verified("found")

    _register(monkeypatch, ActionType.API_SEND_EMAIL, done)
    action = _mail()

    verification, late = await Verifier(MagicMock()).verify_within(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=True, output="sent")], budget=None
    )

    assert late is None
    assert verification.independently_observed_actions == [0]
    assert verification.pending_actions == []


# ── the checks' own cost ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_large_copy_is_compared_up_to_its_cap_and_says_so(tmp_path: Path, monkeypatch) -> None:
    from pilot.agents.postconditions import files

    monkeypatch.setattr(files, "CHUNK_BYTES", 1024)
    monkeypatch.setattr(files, "COPY_COMPARE_BYTES", 4096)
    source = tmp_path / "video.mp4"
    destination = tmp_path / "copy.mp4"
    source.write_bytes(b"\x01" * 10_000)
    destination.write_bytes(b"\x01" * 10_000)
    action = Action(
        action_type=ActionType.FILE_COPY,
        target=str(source),
        parameters=FileParams(path=str(source), destination=str(destination)),
    )

    verification = await Verifier(MagicMock()).verify(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=True, output="Copied")]
    )

    assert verification.passed is True
    assert verification.independently_observed_actions == []
    assert "Not verified in full: compared the first 4,096 bytes and the sizes" in verification.reasons[0]
    assert verification.reason_kinds == {0: "could_not_look"}

    # A difference inside the compared part is still a mismatch.
    destination.write_bytes(b"\x02" + b"\x01" * 9_999)
    verification = await Verifier(MagicMock()).verify(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=True, output="Copied")]
    )
    assert verification.passed is False


@pytest.mark.asyncio
async def test_a_large_file_is_not_hashed_a_second_time_when_size_and_time_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pilot.agents.postconditions import reads
    from pilot.system import filesystem

    path = tmp_path / "disk.iso"
    path.write_bytes(b"\x00" * 2048)
    returned = await filesystem.file_hash(str(path))
    monkeypatch.setattr(reads, "HASH_AGAIN_UP_TO_BYTES", 1024)

    def second_digest(*_args: object) -> str:
        raise AssertionError("hashed a second time")

    monkeypatch.setattr(reads, "_second_digest", second_digest)
    action = Action(action_type=ActionType.FILE_HASH, target=str(path), parameters=FileParams(path=str(path)))

    verification = await Verifier(MagicMock()).verify(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=True, output=returned)]
    )

    assert verification.independently_observed_actions == [0]  # and the second digest was never made
    assert verification.passed is True

    # Changed since it was hashed: the second digest is made (here it would raise, which fails the check closed).
    path.write_bytes(b"\x01" * 4096)
    verification = await Verifier(MagicMock()).verify(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=True, output=returned)]
    )
    assert verification.passed is False
