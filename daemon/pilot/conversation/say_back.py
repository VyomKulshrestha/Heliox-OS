"""Saying back a destructive step that came in as the answer to Heliox's own question, before its dialog opens.

After Heliox asks a question, the next thing it hears is taken as the answer, whoever says it: a television's line,
someone else in the room. When that answer would start something destructive by the planner's own classification,
Heliox says it back first, in the person's words with the target named ("Delete the reports folder, yes?"), once.
A spoken "yes" opens the approval dialog and nothing more: voice never approves, and the dialog still needs the
click. A "no", silence past the follow-up window, or anything else drops it with one short sentence. A request the
person started with the wake phrase is never said back. The same answer arriving again unconfirmed is dropped
without a word.
"""

from __future__ import annotations

import re
from pathlib import PureWindowsPath
from typing import Any

from pilot.conversation.router import ACTION_VERBS, normalize

DROPPED = "Okay, I've left it."
_YES = frozenset(
    {
        "yes", "yeah", "yep", "sure", "yes please", "please do", "do it", "go ahead", "yes do it",
        "yes go ahead", "go for it", "correct", "that's right", "right", "yes that's right",
    }
)  # fmt: skip
_PRONOUNS = {"it", "them", "that", "those", "this", "these", "all", "everything"}


def destructive(actions: list[Any]) -> bool:
    """The planner's own classification: a destructive tier or an action no snapshot can reverse."""
    from pilot.actions import PermissionTier

    for action in actions:
        try:
            if action.permission_tier >= PermissionTier.DESTRUCTIVE or action.is_irreversible or action.destructive:
                return True
        except Exception:
            return True
    return False


def is_yes(words: str) -> bool:
    return normalize(words) in _YES


def sentence(words: str, actions: list[Any]) -> str:
    """ "Delete the reports folder, yes?": the person's words, with what it would act on named when they only
    pointed at it ("delete them all")."""
    said = " ".join(words.strip().rstrip(".!?").split())
    parts = said.split()
    named = [str(getattr(a, "target", "") or "") for a in actions if getattr(a, "target", "")]
    if parts and (len(parts) == 1 or all(w.casefold().strip(",") in _PRONOUNS for w in parts[1:])) and named:
        verb = parts[0].casefold() if parts[0].casefold() in ACTION_VERBS else "go ahead with"
        names = [PureWindowsPath(target.replace("/", "\\")).name or target for target in named[:2]]
        said = f"{verb} {' and '.join(re.sub(r'[-_]+', ' ', n) for n in names)}"
    said = said[:1].upper() + said[1:]
    return f"{said}, yes?"
