"""What a person hears, sees and finds in the receipt for everyday requests whose step was not plainly verified.

Each case gives the step the verifier's real sentence for that situation (copied from the verifier that writes it),
then follows it through the verification result, the reply, the spoken sentence and the receipt entry.
"""

from __future__ import annotations

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
    VolumeParams,
)
from pilot.agents import postconditions
from pilot.agents import verifier as verifier_module
from pilot.agents.postconditions import Outcome, mismatch, register, unobserved, verified
from pilot.agents.unconfirmed import (
    COUNTED_KINDS,
    SPOKEN_PHRASES,
    Kind,
    classify,
    mask,
    receipt_checks,
    speakable_message,
    window_line,
)
from pilot.agents.verifier import Verifier
from pilot.conversation.action_talk import result_sentence
from pilot.response_contract import partial_failure_message, success_message

NOTEPAD_TYPED = (
    'Could not verify: "Untitled - Notepad" (Notepad.exe) kept the focus; the field\'s text was not read: the window '
    "was right; the field's content was not read (no UI Automation reader was running, and starting one takes about "
    "a second, spent only when the next step presses Enter)"
)
WAYLAND_TYPED = (
    "Could not verify: Wayland does not let a program ask which window has the focus, and AT-SPI (the accessibility "
    "bus) did not answer: accessibility may be off in this session, or the window's app is not on the bus"
)
NO_IMAP = (
    "The mail server accepted the message, but Heliox could not check its copy in Sent: reading mail (IMAP) is not "
    "set up in Settings."
)


@pytest.fixture
def scratch_registry(monkeypatch: pytest.MonkeyPatch) -> dict:
    registry = dict(postconditions._REGISTRY)
    monkeypatch.setattr(postconditions, "_REGISTRY", registry)
    return registry


def _use(monkeypatch: pytest.MonkeyPatch, action_type: ActionType, outcome: Outcome) -> None:
    async def check(_result: ActionResult) -> Outcome:
        return outcome

    postconditions._REGISTRY.pop(action_type, None)
    register(action_type, method="test_postcondition")(check)
    monkeypatch.setattr(verifier_module, "_AREA_VERIFIERS", postconditions.registrations())


CASES = {
    "typing without Enter, Windows": (
        Action(action_type=ActionType.KEYBOARD_TYPE, target="Notepad", parameters=KeyboardParams(text="see you at 6")),
        "Typed it.",
        unobserved(NOTEPAD_TYPED),
        Kind.NOT_OBSERVABLE,
    ),
    "scroll down": (
        Action(action_type=ActionType.MOUSE_SCROLL, target="page", parameters=MouseParams()),
        "Scrolled down.",
        unobserved("The scroll went to wherever the pointer was; how far the content moved cannot be read"),
        Kind.NOT_OBSERVABLE,
    ),
    "play or pause": (
        Action(action_type=ActionType.KEYBOARD_PRESS, target="media", parameters=KeyboardParams(key="playpause")),
        "Pressed playpause.",
        unobserved("The state of playpause cannot be read here"),
        Kind.NOT_OBSERVABLE,
    ),
    "volume up": (
        Action(action_type=ActionType.VOLUME_SET, target="volume", parameters=VolumeParams(step=10)),
        "Volume is at 40.",
        unobserved("A volume step has no fixed target to compare with"),
        Kind.NOT_OBSERVABLE,
    ),
    "a search on a site": (
        Action(
            action_type=ActionType.BROWSER_NAVIGATE,
            target="youtube.com",
            parameters=BrowserParams(url="https://youtube.com"),
        ),
        "Opened youtube.com.",
        unobserved(
            "The site showed another page on youtube.com (for example a sign-in page) instead of the one asked for"
        ),
        Kind.CHANGED_LATER,
    ),
    "a browser click with no visible change": (
        Action(action_type=ActionType.BROWSER_CLICK, target="Play", parameters=BrowserParams(selector="#play")),
        "Clicked Play.",
        unobserved("The click landed on the element; no change was observed after it"),
        Kind.NOTHING_HAPPENED,
    ),
    "mail without read-back set up": (
        Action(
            action_type=ActionType.API_SEND_EMAIL,
            target="sam@example.com",
            parameters=ApiRequestParams(to="sam@example.com", subject="Late", message="Running late"),
        ),
        "Email sent: Late.",
        unobserved(NO_IMAP),
        Kind.COULD_NOT_LOOK,
    ),
    "mail with read-back set up": (
        Action(
            action_type=ActionType.API_SEND_EMAIL,
            target="sam@example.com",
            parameters=ApiRequestParams(to="sam@example.com", subject="Late", message="Running late"),
        ),
        "Email sent: Late.",
        verified("The copy with Message-ID <a@b> is in Sent"),
        None,
    ),
    "a desktop step on Wayland": (
        Action(action_type=ActionType.KEYBOARD_TYPE, target="editor", parameters=KeyboardParams(text="hello")),
        "Typed it.",
        unobserved(WAYLAND_TYPED),
        Kind.NOT_OBSERVABLE,
    ),
    "open an app that sits in the tray": (
        Action(action_type=ActionType.OPEN_APPLICATION, target="Discord", parameters=OpenApplicationParams(name="x")),
        "Discord is open.",
        unobserved("Discord is running but shows no window after 10 s: an app that lives in the tray, or one still"),
        Kind.COULD_NOT_LOOK,
    ),
}

# What the person hears and sees, case by case.
HEARD_AND_SEEN = {
    "typing without Enter, Windows": ("Typed it.", ""),
    "scroll down": ("Scrolled down.", ""),
    "play or pause": ("Pressed playpause.", ""),
    "volume up": ("Volume is at 40.", ""),
    "a search on a site": ("Opened youtube.com.", "Note: 1 step could not be checked; details in the receipt."),
    "a browser click with no visible change": (
        "Clicked Play.",
        "Note: 1 step could not be checked; details in the receipt.",
    ),
    "mail without read-back set up": (
        "Email sent: Late. I couldn't check that it went through.",
        "Note: I couldn't check that it went through; details in the receipt.",
    ),
    "mail with read-back set up": ("Email sent: Late.", ""),
    "a desktop step on Wayland": ("Typed it.", ""),
    "open an app that sits in the tray": (
        "Discord is open.",
        "Note: 1 step could not be checked; details in the receipt.",
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", list(CASES))
async def test_everyday_requests_are_said_shown_and_kept_by_kind(
    case: str, monkeypatch: pytest.MonkeyPatch, scratch_registry: dict
) -> None:
    action, output, outcome, kind = CASES[case]
    _use(monkeypatch, action.action_type, outcome)
    plan = ActionPlan(actions=[action])
    results = [ActionResult(action=action, success=True, output=output)]

    verification = await Verifier(MagicMock()).verify(plan, results)
    reply = success_message(plan, results, verification, dry_run=False)
    heard, seen = HEARD_AND_SEEN[case]

    assert verification.passed is True
    assert verification.reason_kinds.get(0) == (kind.value if kind else None)
    assert result_sentence("success", reply) == heard
    assert reply == output + (f"\n{seen}" if seen else "")
    # Nothing the verifier wrote is in the reply or said aloud.
    assert outcome.detail not in reply
    checks = receipt_checks(results, verification)
    if kind is None:
        assert checks == []
    else:
        assert checks == [
            {
                "step": 1,
                "action_type": action.action_type.value,
                "kind": kind.value,
                "reason": mask(outcome.detail),
                "late": False,
            }
        ]


def test_the_receipt_keeps_the_reason_with_the_window_title_masked() -> None:
    kept = mask(NOTEPAD_TYPED)

    assert "Untitled - Notepad" not in kept
    assert 'Notepad.exe window "Unt…"' in kept
    assert "the field's content was not read" in kept


def test_file_names_and_long_quoted_texts_are_masked_for_the_receipt() -> None:
    kept = mask(r"File does not exist after write: C:\Users\someone\Documents\salaries-2026.xlsx.")
    assert "salaries" not in kept and "someone" not in kept
    assert kept.endswith("…/sal….xlsx.")
    assert mask('The tab shows "Quarterly results draft"') == 'The tab shows "Qua…"'
    # Web addresses are not paths.
    assert mask("The page shown is on youtube.com/watch, not example.com") == (
        "The page shown is on youtube.com/watch, not example.com"
    )
    # The window keeps the file name the person gave; only a title is masked there.
    shown = mask('"Inbox - Re: salary review" (OUTLOOK.EXE) is in front; C:\\x\\a.txt', files=False)
    assert "salary" not in shown and "C:\\x\\a.txt" in shown


def test_a_consequential_step_is_spoken_only_for_its_kind() -> None:
    assert window_line([(ActionType.API_SLACK, Kind.RECEIPT_ONLY)]) == (
        "Note: It was accepted, but I can't read it back; details in the receipt."
    )
    assert window_line(
        [(ActionType.GIT_PUSH, Kind.COULD_NOT_LOOK), (ActionType.MOUSE_CLICK, Kind.NOTHING_HAPPENED)]
    ) == ("Note: I couldn't check that it went through, 1 more step could not be checked; details in the receipt.")
    # A receipt-only step that changes nothing for others is counted, never said.
    assert window_line([(ActionType.BROWSER_CLICK, Kind.RECEIPT_ONLY)]) == (
        "Note: 1 step could not be checked; details in the receipt."
    )
    # What by nature cannot be read is not counted at all.
    assert window_line([(ActionType.VOLUME_SET, Kind.NOT_OBSERVABLE)] * 3) == ""
    assert Kind.NOT_OBSERVABLE not in COUNTED_KINDS


def test_a_check_still_running_is_shown_and_not_said() -> None:
    line = window_line([(ActionType.FILE_COPY, Kind.STILL_CHECKING)])
    assert line == "Note: 1 step is still being checked; details in the receipt."
    assert result_sentence("success", f"Copied it.\n{line}") == "Copied it."


def test_a_long_reply_read_as_it_stands_says_the_phrase_not_the_line() -> None:
    reply = (
        "Execution results:\n- a: one\n- b: two\nNote: I couldn't check that it went through; details in the receipt."
    )
    said = speakable_message(reply, limit=60)

    assert "details in the receipt" not in said
    assert said.endswith("I couldn't check that it went through.")
    assert len(said) <= 60


def test_a_mismatch_is_said_in_its_fixed_phrase_and_the_window_title_is_masked() -> None:
    action = Action(action_type=ActionType.KEYBOARD_TYPE, target="Slack", parameters=KeyboardParams(text="hi"))
    from pilot.actions import VerificationResult

    detail = 'The text may have been typed somewhere else: "General - Acme - Slack" (slack.exe) took the focus'
    verification = VerificationResult(
        passed=False, details=[f"Action 0 (keyboard_type): MISMATCH — {detail}"], failed_actions=[0]
    )
    message = partial_failure_message([ActionResult(action=action, success=True, output="Typed")], verification)

    assert "General - Acme" not in message
    assert 'slack.exe window "Gen…"' in message
    phrase = SPOKEN_PHRASES[Kind.MISMATCH]
    assert (
        result_sentence("partial_failure", message) == f"That didn't work for Slack: {phrase[:1].lower()}{phrase[1:]}."
    )


def test_kinds_named_by_the_verifier_win_over_the_sentence() -> None:
    assert classify(ActionType.API_SLACK, "anything", observed=False, passed=True, hint="receipt_only") is (
        Kind.RECEIPT_ONLY
    )
    assert classify(ActionType.API_SLACK, "x", observed=True, passed=True) is None
    assert classify(ActionType.API_SLACK, "x", observed=True, passed=False) is Kind.MISMATCH
    assert (
        classify(
            ActionType.API_SLACK, "Slack accepted the message (receipt, no read-back)", observed=False, passed=True
        )
        is Kind.RECEIPT_ONLY
    )
    assert (
        classify(
            ActionType.API_SLACK,
            "Slack accepted the message (receipt); slack.com could not be reached",
            observed=False,
            passed=True,
        )
        is Kind.COULD_NOT_LOOK
    )


@pytest.mark.asyncio
async def test_a_time_limit_is_a_check_that_could_not_look(
    monkeypatch: pytest.MonkeyPatch, scratch_registry: dict
) -> None:
    import asyncio

    started = asyncio.Event()

    async def slow(_result: ActionResult) -> Outcome:
        started.set()
        await asyncio.Event().wait()
        return mismatch("never")

    postconditions._REGISTRY.pop(ActionType.API_SEND_EMAIL, None)
    register(ActionType.API_SEND_EMAIL, method="test_postcondition", timeout=0.001)(slow)
    monkeypatch.setattr(verifier_module, "_AREA_VERIFIERS", postconditions.registrations())
    action = CASES["mail without read-back set up"][0]

    verification = await Verifier(MagicMock()).verify(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=True, output="Email sent")]
    )

    assert verification.reason_kinds == {0: "could_not_look"}
