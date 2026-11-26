"""A talk reply never says work is being done, is queued or was done when no such task exists.

The talk model only talks. In the live run of 26 November it answered typed requests with "Heliox is already on that
fetch, and the page title will show on screen once it's back", "The commit, branch and push requests are queued after
it", "Heliox is planning a safe read-only check for it now": no task existed, and nothing ever showed. Its context now
holds the truth (pilot.conversation.context.describe_task), and this module checks each sentence of its reply against
that truth before the sentence is said or shown, the way pilot.conversation.debate.proposes_heliox_act checks offers.

A claim is one of:
- under way or about to be: "Heliox is checking it", "I'm on it", "it's planning that now", "I'll have Heliox look";
- waiting: "queued", "in progress", "the request is still open", "lined up";
- coming: "will show on screen", "the result will appear once it finishes";
- done: "I've sent it", "Heliox has posted it", "that's been copied".

A claim is backed when the task it talks about exists: something being worked on or waiting in the queue for a claim
of work, a finished task for a claim that something was done. When the claim names what it is about, one of those
words must appear in the task ("the append is queued" while only a git status runs is not backed). Questions, offers,
sentences about the person ("You copied it yesterday") and ordinary talk ("I'm doing well", "that's interesting",
"it's written in Python") are never claims: only a work verb after Heliox, "I", the planner or the request counts.

Pure text processing, microseconds per sentence: it runs on every sentence the talk model writes.
"""

from __future__ import annotations

import re
from typing import Any

# What doing the work is called. A gerund outside this list ("doing well", "thinking", "nothing") is never a claim.
_WORK = (
    r"planning|checking|fetching|looking (?:into|up|at|for)|reading|appending|posting|sending|hashing|computing|"
    r"calculating|scraping|copying|moving|writing|saving|listing|watching|monitoring|setting (?:it |that |this )?up|"
    r"handling|processing|preparing|working on|queu(?:e)?ing|pulling|pushing|staging|committing|searching|finding|"
    r"downloading|installing|creating|deleting|taking care of|running (?:it|that|the|a|your)|carrying (?:it |that )?out|"
    r"lining (?:it |that )?up|getting (?:the|that|your|it)"
)
_UNDER_WAY = re.compile(
    rf"\b(?:heliox|i|the planner|it|that|this|everything|the (?:task|request|job|check|plan))(?:'s| is|'m| am)\s+"
    rf"(?:already\s+|still\s+|now\s+|currently\s+|busy\s+|right now\s+)*(?:{_WORK})\b"
    r"|\b(?:heliox|i)(?:'s| is|'m| am)\s+(?:already\s+|still\s+|now\s+)*on (?:it|that|this)\b"
    rf"|\bthe (?:task|request|job|check|plan)\b[^!?]{{0,40}}?\b(?:is|are)\s+(?:already\s+|still\s+|now\s+)*(?:{_WORK})\b",
    re.IGNORECASE,
)
_FUTURE = re.compile(
    r"\b(?:i'?ll|i will|heliox will|heliox'?ll|let me|i'?m going to|heliox is going to)\s+"
    r"(?:now\s+|go ahead and\s+|just\s+|quickly\s+)?"
    r"(?:have heliox|ask heliox|get heliox|ask the planner|queue|check|look|fetch|get|run|read|post|send|append|hash|"
    r"scrape|compute|calculate|set up|watch|monitor|copy|move|write|stage|commit|push|create|delete|list|find|search|"
    r"open|pull)\b",
    re.IGNORECASE,
)
_WAITING = re.compile(
    r"\b(?:queued|in the queue|in progress|underway|under way|in the works|lined up|being (?:handled|done|"
    r"worked on|processed|checked|planned|set up))\b"
    r"|\b(?:request|requests|task|tasks|job|jobs|step|steps|check|fetch|post|append|hash|scrape)\b[^!?]{0,60}?"
    r"\b(?:is|are)\s+(?:still\s+|also\s+)*(?:open|pending|running|waiting)\b",
    re.IGNORECASE,
)
_COMING = re.compile(
    r"\b(?:will|should|'ll)\s+(?:show|appear|pop up|come up|be shown|be displayed)(?: up)?\s+"
    r"(?:on (?:the |your )?screen|here|in the (?:window|chat))"
    r"|\b(?:result|results|figure|answer|list|output|title|hash|value|reading|level|status|it)\s+(?:will|'ll)\s+"
    r"(?:show|appear|pop up|come up|be shown|be displayed)\b"
    r"|\bonce (?:it|that|the planner|heliox|the check|the task|the fetch|the request|the plan)\s+"
    r"(?:finishes|is done|is ready|completes|comes back|is back|returns|runs)\b"
    r"|\bonce (?:it|that)'s (?:active|done|ready|set up|back|sent|finished|planned)\b"
    r"|\b(?:i'?ll|i will|heliox will|it will|that will)\s+(?:show|display|put)\b[^.!?]{0,50}?\bon (?:the |your )?screen\b",
    re.IGNORECASE,
)
_DONE = re.compile(
    r"\b(?:i'?ve|i have|heliox has|heliox's)\s+(?:just\s+|already\s+)?(?:sent|posted|appended|copied|written|saved|"
    r"deleted|created|fetched|checked|hashed|scraped|staged|committed|pushed|ran|run|set up|finished|completed)\b"
    r"|\b(?:it|that|this)(?:'s| has) (?:now |already )?been (?:sent|posted|appended|copied|written|saved|deleted|"
    r"created|fetched|staged|committed|pushed|done|finished|completed)\b",
    re.IGNORECASE,
)
_QUESTION_OR_OFFER = re.compile(
    r"^(?:do you want|would you like|want me to|should i|shall i|i can(?!'t|not)|i could(?!n't)|if you (?:want|like))\b",
    re.IGNORECASE,
)
# Words that say nothing about which task: a claim made of only these is about whatever runs.
_PLAIN = frozenset(
    re.findall(
        r"[\w']+",
        """
    heliox already still now currently busy right work working planning plan checking check doing handling getting
    them those that this these will show appear shown screen result results once finish finished done ready back
    queue queued progress open pending running request requests task tasks step steps planner thing safe safest read
    only figure answer list output when soon shortly also earlier last previous same from here there then just with
    into onto have been being going come comes return returns which what while after before quick quickly moment
    second minute lined line underway under handle handled process processed waiting wait your their it's that's
    can't won't don't care take taking carry carrying should appear could would until the and for you are was not
    but its all can any our out his her him has had get got via per new old way now
""",
    )
)


def claim(sentence: str) -> str:
    """ "under_way", "waiting", "coming" or "done" when the sentence says that about work, else ""."""
    text = " ".join(str(sentence or "").replace("’", "'").split())
    if not text or text.rstrip().endswith("?") or _QUESTION_OR_OFFER.match(text):
        return ""
    if _DONE.search(text):
        return "done"
    if _WAITING.search(text):
        return "waiting"
    if _COMING.search(text):
        return "coming"
    if _UNDER_WAY.search(text):
        return "under_way"
    if _FUTURE.search(text) and not _CONDITION.search(text):
        return "under_way"  # "I'll delete them once you confirm" waits on the person: not a claim
    return ""


_CONDITION = re.compile(
    r"\b(?:if|unless|once you|when you|after you|as soon as you|whenever you)\b|\bonce (?:you|it'?s confirmed)\b",
    re.IGNORECASE,
)


def _stem(word: str) -> str:
    """ "appending" -> "append", "copied" -> "copy", "running" -> "run": enough to match a claim to its task."""
    if word.endswith("ied") and len(word) > 4:
        word = word[:-3] + "y"
    else:
        word = re.sub(r"(?:ing|ed|es|s)$", "", word) if len(word) > 4 else word
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiouls":
        word = word[:-1]  # "runn" -> "run", "stopp" -> "stop"
    return word[:6]


def _stems(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", str(text or "").casefold())
    return {_stem(word) for word in words if len(word) >= 3 and word not in _PLAIN and not word.isdigit()}


def backed(sentence: str, kind: str, state: dict[str, Any] | None) -> bool:
    """Whether the task a claim of ``kind`` talks about exists in ``state`` (adapters._task_state's form: "running",
    "queued", "approval_waiting", "last_result"). With no task at all, nothing is backed."""
    state = state if isinstance(state, dict) else {}
    if kind == "done":
        tasks = [str(state.get("last_result") or "")]
    else:
        tasks = [str(state.get("running") or ""), *(str(item) for item in state.get("queued") or [])]
        waiting = state.get("approval_waiting")
        if waiting:
            tasks.append(waiting if isinstance(waiting, str) else "an approval waiting on the screen")
    tasks = [task for task in tasks if task.strip()]
    if not tasks:
        return False
    named = _stems(sentence) - {"i", "i'm", "i'll", "i've", "you", "you're"}
    if not named:
        return True  # "Heliox is still on it": about whatever runs, and something does
    # What the claim is about, as the action catalog names things ("Slack", "the clipboard", "git status"): a task
    # that names none of them is another task ("your Discord post is queued" while a Slack post waits).
    subjects = _subjects(sentence)
    return any(named & _stems(task) and all(subject in _subjects(task) for subject in subjects) for task in tasks)


def _subjects(text: str) -> set[str]:
    from pilot.conversation import catalog_words
    from pilot.conversation.router import normalize

    words = normalize(text)
    found: set[str] = set()
    for pattern in (catalog_words.ANCHOR, catalog_words.STRONG, catalog_words.WEAK):
        found.update(match.group(0) for match in pattern.finditer(words))
    return found


def unbacked(sentence: str, state: dict[str, Any] | None) -> str:
    """The claim's kind when the sentence claims work that does not exist, else ""."""
    kind = claim(sentence)
    return kind if kind and not backed(sentence, kind, state) else ""


_SENTENCES = re.compile(r"(?<=[.!?])(\s+)")


def without_unbacked(text: str, state: dict[str, Any] | None) -> tuple[str, list[str]]:
    """Text with each unbacked claim's sentence removed, keeping the rest's layout; and the removed sentences."""
    kept: list[str] = []
    dropped: list[str] = []
    for part in _SENTENCES.split(text):
        if part.strip() and unbacked(part, state):
            dropped.append(part.strip())
            continue
        kept.append(part)
    return "".join(kept), dropped


__all__ = ["backed", "claim", "unbacked", "without_unbacked"]
