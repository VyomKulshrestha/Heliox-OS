"""Every registered check states the kind of a result it could not confirm, and the everyday cases still come out
as the reply rules say, with the sentences the desktop and browser checks write today.

The kind decides what the person hears, sees and finds in the receipt (pilot.agents.unconfirmed). Reading it from the
verifier's sentence (DETAIL_KINDS) is kept only as a fallback for results built elsewhere: a reworded sentence must
never change what is said.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    ApiRequestParams,
    BrowserParams,
    KeyboardParams,
    MouseParams,
    OpenApplicationParams,
)
from pilot.agents import postconditions
from pilot.agents import verifier as verifier_module
from pilot.agents.postconditions import Outcome, register, unobserved, verified
from pilot.agents.unconfirmed import (
    SPOKEN_PHRASES,
    Kind,
)
from pilot.agents.verifier import Verifier
from pilot.conversation.action_talk import result_sentence
from pilot.response_contract import success_message

DAEMON = Path(__file__).resolve().parents[1]
KINDS = {kind.value for kind in Kind}


# ── The contract: a not-verified result always names its kind ────────────────────────────────


def _module_paths() -> list[Path]:
    """Every module a registered verifier lives in, and the desktop readings the desktop checks pass on."""
    modules = {registration.module for registration in postconditions.load_area_modules().values()}
    modules |= {"pilot.agents.postconditions._in_step", "pilot.system.desktop_observe"}
    return sorted(Path(importlib.import_module(name).__file__) for name in modules)


def _kind_problem(call: ast.Call) -> str:
    kinds = [keyword for keyword in call.keywords if keyword.arg == "kind"]
    if not kinds:
        return "names no kind"
    value = kinds[0].value
    if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and value.value.id == "Kind":
        return "" if value.attr in Kind.__members__ else f"Kind.{value.attr} is not a kind"
    if isinstance(value, ast.Constant):
        return "" if value.value in KINDS else f"{value.value!r} is not a kind"
    return ""  # computed (a reading's own kind, with a default): checked at run time below


def test_every_not_verified_result_of_a_registered_check_names_its_kind() -> None:
    problems = []
    for path in _module_paths():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            recorded_unconfirmed = (
                node.func.id == "_record"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in {"partial", "unobservable"}
            )
            if node.func.id == "unobserved" or recorded_unconfirmed:
                problem = _kind_problem(node)
                if problem:
                    problems.append(f"{path.name}:{node.lineno} {node.func.id}(...) {problem}")
    assert problems == []
    assert len(_module_paths()) > 10


def test_the_registry_walk_covers_every_registered_check() -> None:
    covered = {path.stem for path in _module_paths()}
    for registration in postconditions.load_area_modules().values():
        assert registration.module.rsplit(".", 1)[-1] in covered


@pytest.mark.asyncio
async def test_a_reading_made_in_the_step_keeps_its_kind_and_notice() -> None:
    from pilot.agents.postconditions import _in_step

    result = ActionResult(
        action=Action(action_type=ActionType.BROWSER_SCROLL, parameters=BrowserParams(direction="down")),
        success=True,
    )

    async def check(_result: ActionResult) -> Outcome:
        return unobserved("The page did not move when scrolled down", kind=Kind.NOTHING_HAPPENED)

    registry = {ActionType.BROWSER_SCROLL: SimpleNamespace(check=check)}
    original = _in_step.load_area_modules
    _in_step.load_area_modules = lambda: registry  # type: ignore[assignment]
    try:
        result.observation = await _in_step.observe(result)
    finally:
        _in_step.load_area_modules = original  # type: ignore[assignment]

    recalled = _in_step.recalled(result)
    assert recalled is not None and recalled.kind == Kind.NOTHING_HAPPENED

    fits = verified("The page fits in the window, so there was nothing to scroll", notice="fits")
    result.observation = _in_step.STEP_CHECK_MARKER + ' {"passed":true,"detail":"x","observed":true,"notice":"fits"}'
    assert _in_step.recalled(result).notice == fits.notice


@pytest.mark.parametrize(
    ("record_kind", "expected"),
    [("not_observable", Kind.NOT_OBSERVABLE), ("changed_later", Kind.CHANGED_LATER), (None, Kind.COULD_NOT_LOOK)],
)
@pytest.mark.asyncio
async def test_the_desktop_checks_pass_on_the_kind_of_their_reading(record_kind, expected) -> None:
    from pilot.agents.postconditions import desktop_input
    from pilot.system import desktop_observe

    action = Action(action_type=ActionType.MOUSE_MOVE, parameters=MouseParams(x=10, y=10))
    facts = {"kind": record_kind} if record_kind else {}
    desktop_observe.remember(desktop_observe.StepRecord(action, "partial", "the mouse may have been moved", facts))
    try:
        outcome = await desktop_input.pointer_moved(ActionResult(action=action, success=True))
    finally:
        desktop_observe.forget_all()
    assert outcome.kind == expected


# ── The everyday cases, with the sentences the checks write today ────────────────────────────


def _kind_at(module: str, phrase: str) -> str:
    """The kind stated by the call whose sentence contains ``phrase`` (an unobserved(...) or a desktop _record)."""
    path = Path(importlib.import_module(module).__file__)
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id not in {"unobserved", "_record"}:
            continue
        text = "".join(
            part.value
            for argument in node.args
            for part in ast.walk(argument)
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
        if phrase in text:
            value = next(keyword.value for keyword in node.keywords if keyword.arg == "kind")
            return Kind[value.attr].value if isinstance(value, ast.Attribute) else str(value.value)
    raise AssertionError(f"no call in {module} says {phrase!r}")


@pytest.fixture
def scratch_registry(monkeypatch: pytest.MonkeyPatch) -> dict:
    registry = dict(postconditions._REGISTRY)
    monkeypatch.setattr(postconditions, "_REGISTRY", registry)
    return registry


async def _through_the_reply(
    monkeypatch, action: Action, output: str, outcome: Outcome, notice: str | None = None
) -> tuple[str, str, dict]:
    async def check(_result: ActionResult) -> Outcome:
        return outcome

    postconditions._REGISTRY.pop(action.action_type, None)
    register(action.action_type, method="test_postcondition")(check)
    monkeypatch.setattr(verifier_module, "_AREA_VERIFIERS", postconditions.registrations())
    plan = ActionPlan(actions=[action])
    results = [ActionResult(action=action, success=True, output=output, notice=notice)]
    verification = await Verifier(MagicMock()).verify(plan, results)
    reply = success_message(plan, results, verification, dry_run=False)
    return reply, result_sentence("success", reply), dict(verification.reason_kinds)


ONE_LINE = "Note: 1 step could not be checked; details in the receipt."

EVERYDAY = {
    "an app that lives in the tray": (
        "pilot.agents.postconditions.desktop_input",
        "lives in the tray",
        Action(action_type=ActionType.OPEN_APPLICATION, target="Discord", parameters=OpenApplicationParams(name="x")),
        "Discord is open.",
        Kind.COULD_NOT_LOOK,
        ONE_LINE,
        "",
    ),
    "a click with no visible change": (
        "pilot.agents.postconditions.browser_ui",
        "no change was observed",
        Action(action_type=ActionType.BROWSER_CLICK, target="Play", parameters=BrowserParams(selector="#play")),
        "Clicked Play.",
        Kind.NOTHING_HAPPENED,
        ONE_LINE,
        "",
    ),
    "a scroll that did not move": (
        "pilot.agents.postconditions.browser_ui",
        "The page did not move when scrolled",
        Action(action_type=ActionType.BROWSER_SCROLL, target="page", parameters=BrowserParams(direction="down")),
        "Scrolled down by 500px.",
        Kind.NOTHING_HAPPENED,
        ONE_LINE,
        "",
    ),
    "a field that read empty": (
        "pilot.system.desktop_observe",
        "which apps that paste late, complete or reformat text also cause",
        Action(action_type=ActionType.KEYBOARD_TYPE, target="Notepad", parameters=KeyboardParams(text="hi")),
        "Typed it.",
        Kind.NOT_OBSERVABLE,
        "",
        "",
    ),
    "a pointer the person moved": (
        "pilot.system.desktop_observe",
        "the mouse may have been moved since",
        Action(action_type=ActionType.MOUSE_CLICK, target="Save", parameters=MouseParams(x=5, y=5)),
        "Clicked.",
        Kind.CHANGED_LATER,
        ONE_LINE,
        "",
    ),
    "a Slack post without read-back set up": (
        "pilot.agents.postconditions.messaging",
        "add a bot token and the channel ID in",
        Action(action_type=ActionType.API_SLACK, target="team", parameters=ApiRequestParams(message="Deploy done")),
        "Posted to Slack.",
        Kind.RECEIPT_ONLY,
        f"Note: {SPOKEN_PHRASES[Kind.RECEIPT_ONLY]}; details in the receipt.",
        SPOKEN_PHRASES[Kind.RECEIPT_ONLY],
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", list(EVERYDAY))
async def test_everyday_cases_hold_with_todays_sentences(case, monkeypatch, scratch_registry) -> None:
    module, phrase, action, output, kind, seen, spoken = EVERYDAY[case]
    stated = _kind_at(module, phrase)
    assert stated == kind.value

    reply, heard, kinds = await _through_the_reply(
        monkeypatch, action, output, unobserved(f"a sentence that says {phrase}", kind=stated)
    )

    assert kinds == {0: kind.value}
    assert reply == output + (f"\n{seen}" if seen else "")
    if spoken:
        assert heard.count(spoken) == 1
    else:
        assert heard == output


@pytest.mark.asyncio
async def test_a_page_that_fits_the_window_is_verified_and_its_notice_is_shown_not_spoken(
    monkeypatch, scratch_registry
) -> None:
    from pilot.system.browser import scroll_end_already_reached

    sentence = scroll_end_already_reached("down", {"y": 0, "max_y": 0})
    assert sentence and "fits" in sentence
    action = Action(action_type=ActionType.BROWSER_SCROLL, target="page", parameters=BrowserParams(direction="down"))

    # The scroll handler gives the sentence as its notice; the check verifies the scroll.
    reply, heard, kinds = await _through_the_reply(
        monkeypatch, action, "Scrolled down by 500px.", verified(sentence), notice=sentence
    )

    assert kinds == {}
    assert f"Note: {sentence}" in reply
    assert "fits" not in heard
