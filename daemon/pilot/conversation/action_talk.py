"""What Heliox may say about an action, chosen from the action's state, never from how the request was worded.

An action is asked for, waits for approval, runs, and ends done, refused or failed. Only a running action may be
described in the present tense ("Opening Notepad."), only a finished one as done ("Notepad is open."). Before
anything runs, Heliox says what it will do ("I'll open your launch folder."), or nothing at all when the request is
one whose plan will almost always need the person's approval (deleting, sending, installing, shutting down): there
the first thing said is the approval sentence, so the person never hears "Deleting that now." for something waiting
on a dialog.

The quick road (pilot.conversation.quick) builds one checked, allow-listed action that needs no approval and runs at
once; its sentence is said as it starts, so it is the running state's. The planner's road has no plan yet when the
person finishes speaking, so its first sentence is the asked-for state's.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pilot.conversation.router import ACTION_VERBS


class ActionState(StrEnum):
    ASKED = "asked"  # handed over, nothing planned or run yet
    AWAITING_APPROVAL = "awaiting_approval"  # planned, waiting on the person's dialog; voice can deny, never approve
    RUNNING = "running"
    DONE = "done"
    REFUSED = "refused"  # the person declined, withdrew or stopped it, or a check refused it: nothing ran
    FAILED = "failed"


# Verbs whose plans need the person's approval nearly always (pilot.actions: destructive, system-changing or sent
# beyond this computer, so requires_confirmation or is_irreversible). Their asked-for state says nothing: the next
# sentence is the approval one, or the result when an approval was waived.
APPROVAL_VERBS = frozenset(
    {
        "delete",
        "remove",
        "erase",
        "wipe",
        "format",
        "empty",
        "clear",
        "clean",
        "uninstall",
        "install",
        "update",
        "upgrade",
        "kill",
        "shut",
        "restart",
        "reboot",
        "close",
        "quit",
        "exit",
        "send",
        "email",
        "message",
        "reply",
        "post",
        "publish",
        "book",
        "buy",
        "pay",
        "run",
    }
)

_GERUNDS = frozenset(gerund.split()[0].casefold() for gerund in ACTION_VERBS.values())
_OPENER = re.compile(r"^(?:okay|ok|sure|right|alright)[,.!]?\s+", re.IGNORECASE)
_DONE = re.compile(
    r"\b(?:is|are) (?:now )?(?:open|closed|deleted|removed|sent|installed|done|muted)\b|\bI've (?!dropped|left)\w+"
    r"|\bhas been\b|^done\b",
    re.IGNORECASE,
)


def claims_running(sentence: str) -> bool:
    """ "Deleting that now.", "Okay, messaging your manager.": the sentence says an action is under way."""
    words = _OPENER.sub("", sentence.strip()).split()
    return bool(words) and words[0].casefold().strip(".,!") in _GERUNDS


def claims_done(sentence: str) -> bool:
    """ "Notepad is open.", "Done.", "I've sent it.": the sentence says an action finished."""
    return bool(_DONE.search(_OPENER.sub("", sentence.strip())))


def may_say(state: ActionState, sentence: str) -> bool:
    """The rule itself: an under-way claim only while running, a finished claim only when done."""
    if claims_done(sentence):
        return state is ActionState.DONE
    if claims_running(sentence):
        return state is ActionState.RUNNING
    return True


def said_again(sentence: str) -> str:
    """ "I'll open Notepad." said twice running: "Okay, I'll open Notepad." ("I" keeps its capital)."""
    first = sentence[:1] if re.match(r"I\b", sentence) else sentence[:1].lower()
    return "Okay, " + first + sentence[1:]


def needs_approval_first(verb: str) -> bool:
    return verb.casefold() in APPROVAL_VERBS


_FUTURE = re.compile(r"^(?:okay,?\s+|sure,?\s+)?(?:i'll|i will|let me)\b", re.IGNORECASE)


def asked_sentence(verb: str, target: str, model_say: str = "") -> str:
    """The first sentence for a request handed to the planner: what Heliox will do, or "" when the plan will need an
    approval first. ``target`` is already in the person's terms ("your launch folder"). The fast model's own sentence
    is used only when it says what will happen ("I'll ...", "Let me ..."), never when it claims progress."""
    verb = verb.casefold()
    if needs_approval_first(verb):
        return ""
    if model_say and _FUTURE.match(model_say.strip()) and may_say(ActionState.ASKED, model_say):
        return model_say.strip()
    words = target.split()
    if verb == "remind":
        return "I'll set that reminder."
    if verb == "check":
        return "I'll check."
    if not target or len(words) > 6:
        return "I'll get that started."
    if verb in {"find", "locate"}:
        return f"I'll look for {target}."
    if verb in {"search", "look"}:
        joined = target if words[0] in {"for", "up", "at", "in", "through"} else f"for {target}"
        return f"I'll {verb} {joined}."
    if verb in {"launch", "visit", "browse"}:
        verb = "open"
    if verb == "screenshot":
        return f"I'll take a screenshot of {target}."
    if verb == "backup":
        return f"I'll back up {target}."
    return f"I'll {verb} {target}."


_UNVERIFIED = re.compile(
    r"\s*Note: \d+ action effects? (?:was|were) reported successful but not independently verified\.\s*$"
)
_FAILED = re.compile(r"^Task did not complete successfully:[^.]*\.\s*")
_SPOKEN_LIMIT = 20  # words; longer results stay in the window and are left to the server's own wording


def result_sentence(status: str, message: str) -> str:
    """The planner road's outcome in a sentence, from its end state: done (the result's one line, and that it could
    not be checked when the verifier saw nothing), or failed (what went wrong). "" leaves the server's wording:
    a refusal or cancel is already said in plain words, and a long or listed result is not read out here."""
    text = message.strip()
    if status == "success":
        unverified = bool(_UNVERIFIED.search(text))
        body = _UNVERIFIED.sub("", text).strip()
        if not body or "\n" in body or len(body.split()) > _SPOKEN_LIMIT:
            return ""
        sentence = body if body.endswith((".", "!", "?")) else body + "."
        return f"{sentence} I couldn't check that it worked." if unverified else sentence
    if status in {"partial_failure", "partial", "error", "failed"}:
        found = _FAILED.match(text)
        if found is None:
            return ""
        issue = text[found.end() :].strip()
        if not issue or "\n" in issue or len(issue.split()) > _SPOKEN_LIMIT:
            return "That didn't work."
        return f"That didn't work: {issue[:1].lower() + issue[1:]}"
    return ""
