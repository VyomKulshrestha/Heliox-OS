"""Checks that go on behind the reply, and what is done with their outcome when it arrives.

The reply waits at most ``unconfirmed.VERIFY_BUDGET_SECONDS`` for a plan's checks (``Verifier.verify_within``). A
check still running then (a large file compared, a mail server asked for its copy in Sent, a package listing) goes
on here, bounded by its own time limit and by ``LATE_CEILING_SECONDS`` for all of a plan's late checks together.

When the outcome arrives it only informs:

- every outcome goes to a receipt of its own that follows the task's receipt, and to the timeline;
- a mismatch on a step that sends something to other people or cannot be undone is told to the person in one
  sentence, through the same channel as other notices that come after a reply (a watcher that fired, a reminder);
- a late "verified" is silent; a notice it carries (packages an upgrade left behind) is shown, not said.

Nothing here plans, retries or repairs: the plan is over. Nothing here reads or writes the state of a later plan:
each follow-up carries the values it was given when the reply went out. ``stop`` cancels every check still running
and every report not yet written, and waits for them, before the daemon goes on stopping.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pilot.agents.unconfirmed import CONSEQUENTIAL_ACTIONS, Kind, Step, late_sentence, mask
from pilot.agents.verifier import LateChecks, LateOutcome
from pilot.timeouts import bounded

logger = logging.getLogger("pilot.agents.late_checks")

# All of one plan's late checks together; each check also has its own limit (postconditions.register's timeout).
LATE_CEILING_SECONDS = 600.0
VERIFIED = "verified"


@dataclass(frozen=True)
class FollowUp:
    """What a late outcome is filed under: the task and plan it belongs to, as they were when the reply went out."""

    task_key: str
    plan_id: str
    request: str
    receipt_id: str | None = None


@dataclass
class LateReport:
    """The outcome of one plan's late checks."""

    follow_up: FollowUp
    checks: list[dict[str, Any]] = field(default_factory=list)
    told: str = ""
    shown: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        verified = sum(check["kind"] == VERIFIED for check in self.checks)
        mismatched = sum(check["kind"] == Kind.MISMATCH.value for check in self.checks)
        other = len(self.checks) - verified - mismatched
        parts = [
            f"{verified} verified" if verified else "",
            f"{mismatched} not as asked" if mismatched else "",
            f"{other} not confirmed" if other else "",
        ]
        return "Checked after the reply: " + ", ".join(part for part in parts if part) + "."


Record = Callable[[LateReport], Awaitable[None]]
# (sentence, report, speak): speak is False for a notice that is only shown.
Tell = Callable[[str, LateReport, bool], Awaitable[None]]


def _entry(late: LateOutcome) -> dict[str, Any]:
    action_type = late.result.action.action_type
    if late.kind is None:
        entry: dict[str, Any] = {
            "step": late.index + 1,
            "action_type": action_type.value,
            "kind": VERIFIED,
            "reason": mask(late.outcome.detail)[:500],
            "late": True,
        }
    else:
        entry = Step(late.index, action_type, late.kind, late.outcome.detail, True).for_receipt()
    if late.outcome.notice:
        entry["notice"] = late.outcome.notice[:500]
    return entry


class LateVerifications:
    """Follows the checks a reply did not wait for. ``record`` writes a report to the receipt and the timeline;
    ``tell`` gives the person one sentence. Both are the daemon's; tests pass their own."""

    def __init__(self, *, record: Record, tell: Tell, ceiling: float = LATE_CEILING_SECONDS) -> None:
        self._record = record
        self._tell = tell
        self._ceiling = ceiling
        self._checks: set[asyncio.Task[Any]] = set()
        self._reports: set[asyncio.Task[None]] = set()
        self._stopping = False

    def hold(self, late: LateChecks) -> None:
        """Keep ``late``'s task among those stopped at shutdown, from the moment it exists."""
        if self._stopping:
            late.task.cancel()
            return
        self._checks.add(late.task)
        late.task.add_done_callback(self._check_done)

    def _check_done(self, task: asyncio.Task[Any]) -> None:
        self._checks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.warning("A check that ran behind the reply failed: %s", task.exception())

    def follow(self, late: LateChecks, follow_up: FollowUp) -> None:
        """Report ``late``'s outcome when it arrives (the reply has gone out)."""
        self.hold(late)
        if self._stopping:
            return
        task = asyncio.create_task(self._report(late, follow_up), name=f"late-checks-{follow_up.plan_id}")
        self._reports.add(task)
        task.add_done_callback(self._report_done)

    def _report_done(self, task: asyncio.Task[None]) -> None:
        self._reports.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.warning("Reporting a late check failed: %s", task.exception())

    async def drop(self, late: LateChecks | None) -> None:
        """Stop ``late``'s checks and wait for them: the plan is being retried or was abandoned."""
        if late is not None:
            await late.cancel()

    async def _report(self, late: LateChecks, follow_up: FollowUp) -> None:
        try:
            outcomes = await bounded(late.outcomes(), self._ceiling)
        except TimeoutError:
            await late.cancel()
            outcomes = []
        except asyncio.CancelledError:
            await late.cancel()
            raise
        report = LateReport(follow_up, [_entry(outcome) for outcome in outcomes])
        finished = {outcome.index for outcome in outcomes}
        for index in late.pending:
            if index not in finished:
                action_type = late.results[index].action.action_type
                reason = f"The check did not finish within {self._ceiling:g} s after the reply"
                report.checks.append(Step(index, action_type, Kind.COULD_NOT_LOOK, reason, True).for_receipt())
        report.checks.sort(key=lambda check: check["step"])
        mismatch = next(
            (
                outcome
                for outcome in outcomes
                if outcome.kind is Kind.MISMATCH and outcome.result.action.action_type in CONSEQUENTIAL_ACTIONS
            ),
            None,
        )
        if mismatch is not None:
            report.told = late_sentence(mismatch.result.action.action_type)
        report.shown = [outcome.outcome.notice for outcome in outcomes if outcome.outcome.notice]
        if report.checks:
            await self._record(report)
        if report.told:
            await self._tell(report.told, report, True)
        for notice in report.shown:
            await self._tell(notice, report, False)

    async def stop(self) -> None:
        """Cancel every late check and report, and wait for them. Nothing starts after this."""
        self._stopping = True
        tasks = [*self._reports, *self._checks]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @property
    def running(self) -> int:
        return len(self._checks) + len(self._reports)


__all__ = ["LATE_CEILING_SECONDS", "FollowUp", "LateReport", "LateVerifications"]
