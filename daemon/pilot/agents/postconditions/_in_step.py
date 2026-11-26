"""Browser checks read right after their step, not after the whole plan.

A plan such as "scroll down and open the second article" leaves the page before the verifier runs, so a check made
then reads the wrong page: the article sits at the top, as the old page did before the scroll, and the scroll looked
as if it never moved. The executor therefore calls ``observe`` right after each browser step, while the page is
still the one the step acted on, and keeps the outcome in the result's ``observation`` (never shown, spoken or
stored). The verifier reports that outcome later (``recalled``). A browser step whose result carries no such reading
(a result replayed from storage, or one that did not come through the executor) is checked when the verifier runs;
each check then compares the page's identity (a hash of its address and its document's load time) with the one
recorded before the step, and says "not verified", never a mismatch, when the page has changed since.

The reading is bounded (``IN_STEP_SECONDS``) so it adds little to each step; a reading that breaks or runs out of
time is "not verified", never a failed step.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar

from pilot.actions import ActionResult, ActionType
from pilot.agents.postconditions import Outcome, load_area_modules, unobserved
from pilot.agents.unconfirmed import Kind

logger = logging.getLogger("pilot.agents.postconditions.in_step")

STEP_CHECK_MARKER = "[STEP_CHECK]"
IN_STEP_SECONDS = 4.0

# Browser steps whose check reads the live page: every one of them is read right after it runs.
IN_STEP_ACTIONS = frozenset(
    {
        ActionType.BROWSER_NAVIGATE,
        ActionType.BROWSER_TYPE,
        ActionType.BROWSER_SELECT,
        ActionType.BROWSER_FILL_FORM,
        ActionType.BROWSER_CLICK,
        ActionType.BROWSER_CLICK_TEXT,
        ActionType.BROWSER_NEW_TAB,
        ActionType.BROWSER_SWITCH_TAB,
        ActionType.BROWSER_CLOSE_TAB,
        ActionType.BROWSER_CLOSE,
        ActionType.BROWSER_BACK,
        ActionType.BROWSER_FORWARD,
        ActionType.BROWSER_REFRESH,
        ActionType.BROWSER_HOVER,
        ActionType.BROWSER_SCROLL,
    }
)

_RIGHT_AFTER_STEP: ContextVar[bool] = ContextVar("browser_check_right_after_step", default=False)


def right_after_step() -> bool:
    """True while a check runs right after its step (from ``observe``); False when it runs later, after the plan."""
    return _RIGHT_AFTER_STEP.get()


async def observe(result: ActionResult) -> str | None:
    """Run the step's check now and return the line that keeps its outcome, or None for other steps."""
    from pilot.timeouts import bounded

    if not result.success or result.action.action_type not in IN_STEP_ACTIONS:
        return None
    registration = load_area_modules().get(result.action.action_type)
    if registration is None:
        return None
    token = _RIGHT_AFTER_STEP.set(True)
    try:
        outcome = await bounded(registration.check(result), IN_STEP_SECONDS)
    except TimeoutError:
        outcome = unobserved("The page took too long to read after this step", kind=Kind.COULD_NOT_LOOK)
    except Exception:
        logger.debug("Reading the page after %s failed", result.action.action_type.value, exc_info=True)
        outcome = unobserved("The page could not be read after this step", kind=Kind.COULD_NOT_LOOK)
    finally:
        _RIGHT_AFTER_STEP.reset(token)
    record = {
        "passed": outcome.passed,
        "detail": outcome.detail,
        "observed": outcome.observed,
        "kind": outcome.kind,
        "notice": outcome.notice,
    }
    return f"{STEP_CHECK_MARKER} {json.dumps(record, separators=(',', ':'))}"


def recalled(result: ActionResult) -> Outcome | None:
    """The outcome read right after the step, or None when the step was not read then."""
    for line in reversed(str(result.observation or "").splitlines()):
        if line.startswith(STEP_CHECK_MARKER):
            try:
                record = json.loads(line[len(STEP_CHECK_MARKER) :].strip())
                return Outcome(
                    bool(record["passed"]),
                    str(record["detail"]),
                    bool(record["observed"]),
                    str(record.get("kind") or ""),
                    str(record.get("notice") or ""),
                )
            except (ValueError, KeyError, TypeError):
                return None
    return None
