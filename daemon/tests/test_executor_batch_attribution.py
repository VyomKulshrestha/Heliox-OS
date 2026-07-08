"""Regression test for batch exception attribution

When an action in a parallel batch raises, the failure must be attributed to
the action that actually raised — not always batch[0]. asyncio.gather preserves
input order, so batch_results[i] corresponds to batch[i]; a raised exception
carries no index, so position is the only link back to the action.
"""

from pilot.actions import Action, ActionResult, ActionType, EmptyParams
from pilot.agents.executor import Executor


def _make_action(target: str) -> Action:
    return Action(action_type=ActionType.FILE_READ, target=target, parameters=EmptyParams())


def _bare_executor() -> Executor:
    # Build an Executor without running __init__ (which constructs snapshot
    # managers, dispatch tables, etc.) — _collect_batch_results only touches
    # _last_output / _largest_output.
    ex = object.__new__(Executor)
    ex._last_output = ""
    ex._largest_output = ""
    return ex


def test_exception_attributed_to_failing_action_not_first():
    ex = _bare_executor()
    batch = [_make_action("action0"), _make_action("action1"), _make_action("action2")]
    # gather returns positionally: action0 ok, action1 raised, action2 ok.
    # Success items are (idx, ActionResult) tuples; the exception is the bare exc.
    batch_results = [
        (0, ActionResult(action=batch[0], success=True, output="ok0")),
        ValueError("action1 blew up"),
        (2, ActionResult(action=batch[2], success=True, output="ok2")),
    ]
    results: list[ActionResult] = []
    failed = ex._collect_batch_results(batch, batch_results, results)

    assert failed is True
    # The failure must name action1 (the one that raised), not action0 (batch[0]).
    failures = [r for r in results if not r.success]
    assert len(failures) == 1
    assert failures[0].action.target == "action1"
    assert "action1 blew up" in (failures[0].error or "")


def test_first_action_exception_still_correct():
    # Guard the boundary: when batch[0] is the one that raises, it should still
    # be attributed to batch[0] (the old code happened to be right only here).
    ex = _bare_executor()
    batch = [_make_action("action0"), _make_action("action1")]
    batch_results = [
        RuntimeError("action0 failed"),
        (1, ActionResult(action=batch[1], success=True, output="ok1")),
    ]
    results: list[ActionResult] = []
    failed = ex._collect_batch_results(batch, batch_results, results)

    assert failed is True
    failures = [r for r in results if not r.success]
    assert len(failures) == 1
    assert failures[0].action.target == "action0"


def test_all_success_returns_not_failed():
    ex = _bare_executor()
    batch = [_make_action("a"), _make_action("b")]
    batch_results = [
        (0, ActionResult(action=batch[0], success=True, output="x")),
        (1, ActionResult(action=batch[1], success=True, output="y")),
    ]
    results: list[ActionResult] = []
    failed = ex._collect_batch_results(batch, batch_results, results)

    assert failed is False
    assert len(results) == 2
    assert all(r.success for r in results)
