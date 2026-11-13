"""Answers Heliox gives in code that hold the person's own data, and what the conversation may do with them.

A model in the cloud is told only that such a turn was answered on this computer, never its words (pilot.server,
CODE_ANSWERS_ANY_MODEL): the line it sees names the kind of answer from the fixed list below, keyed by the handler
that gave it, so nothing of the answer can ride along. A model on this computer is sent every answer in words.

So a short follow-up straight after one of these answers ("and tomorrow?") reaches a cloud model without what was
shown. When the follow-up only swaps a day, a time or a number in the question just answered, Heliox asks its own
handler again in code (``rewrite``); when that does not clearly apply, the turn goes to the model as before, with
the kind as its only hint.
"""

from __future__ import annotations

import re

# handler (pilot.server._control_reply) or "_local_answer:<source class>" -> what kind of answer it gave
KINDS: dict[str, str] = {
    "_getting_to_know_request": "getting to know the person",
    "_offer_reply": "an offer",
    "_live_reply": "a Live note",
    "_say_again_request": "the last reply, said again",
    "_profile_request": "what Heliox knows about the person",
    "_goal_request": "a goal",
    "_companion_request": "a companion setting",
    "_suggestion_request": "a suggestion",
    "_day_request": "the day's plan",
    "_forget_request": "something forgotten",
    "_focus_request": "a focus session",
    "_settings_request": "a setting",
    "_workspace_request": "a workspace",
    "_undo_request": "an undo",
    "_routine_request": "a routine",
    "_shortcut_request": "a voice shortcut",
    "_memory_talk_request": "something remembered",
    "_local_answer:ActivityTimeline": "the activity timeline",
    "_local_answer:TaskReceipts": "a past task",
    "_local_answer:UndoJournal": "an undo",
    "_local_answer:AppReminders": "an app reminder",
    "_local_answer:ReminderStore": "a reminder",
    "_local_answer:NoteStore": "a note",
    "_local_answer:TodoStore": "a to-do",
    "_local_answer:UserProfile": "what Heliox knows about the person",
    "_local_answer:GoalStore": "a goal",
    "_local_answer:HabitStore": "a habit",
    "_local_answer:FocusLog": "focus time",
    "_local_answer:NudgeStore": "a nudge",
    "_local_answer:FocusGuard": "focus",
    "_local_answer:WatcherStore": "a watcher",
    "_local_answer:ClipboardHistory": "the clipboard",
    "_local_answer:SnippetStore": "a snippet",
    "_local_answer:SystemHealth": "the computer's health",
    "_local_answer:FileFinder": "a file search",
    "_local_answer:Generators": "something generated",
}

FOLLOW_UP_SECONDS = 30.0  # "and tomorrow?" belongs to the answer just given, not to one from minutes ago

_WEEKDAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_DAY = rf"today|tonight|tomorrow|yesterday|this week|next week|(?:on )?(?:{_WEEKDAYS})"
_TIME = r"at (?:\d{1,2}(?::\d{2})?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)(?: ?[ap]\.? ?m\.?)?"
_COUNT = r"\d+|one|two|three|four|five|six|seven|eight|nine|ten"
_LEAD = r"^(?:and|what about|how about|and what about)?\s*(?:for\s+)?"
_FOLLOW_DAY = re.compile(_LEAD + rf"(?P<slot>{_DAY})\s*\??$", re.IGNORECASE)
_FOLLOW_TIME = re.compile(_LEAD + rf"(?P<slot>{_TIME})\s*\??$", re.IGNORECASE)
_FOLLOW_COUNT = re.compile(_LEAD + rf"(?:the )?(?:last |next |first )?(?P<slot>{_COUNT})\s*\??$", re.IGNORECASE)
_SLOTS = {
    "day": re.compile(rf"\b(?:{_DAY})\b", re.IGNORECASE),
    "time": re.compile(rf"\b{_TIME}\b", re.IGNORECASE),
    "count": re.compile(rf"\b(?:{_COUNT})\b", re.IGNORECASE),
}
# The day's plan, asked as "good morning": the question it answers, so a day can be swapped into it.
_ASKED_AS = {"_day_request": "what's on today"}


def rewrite(previous: str, by: str, follow_up: str) -> str | None:
    """The question just answered in code, with the one day, time or number the follow-up carries swapped in, or
    None when that does not clearly apply: the follow-up carries anything else, or the question has none or more
    than one of that kind of slot."""
    words = " ".join(follow_up.strip().split())
    for kind, pattern in (("day", _FOLLOW_DAY), ("time", _FOLLOW_TIME), ("count", _FOLLOW_COUNT)):
        found = pattern.match(words)
        if found is None:
            continue
        slot = re.sub(r"^on ", "", found.group("slot"), flags=re.IGNORECASE)
        places = list(_SLOTS[kind].finditer(previous))
        if not places and by in _ASKED_AS:  # "good morning": the question it answers holds the day
            previous = _ASKED_AS[by]
            places = list(_SLOTS[kind].finditer(previous))
        if len(places) != 1:
            return None
        place = places[0]
        return previous[: place.start()] + slot + previous[place.end() :]
    return None
