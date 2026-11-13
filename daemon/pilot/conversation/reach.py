"""What a follow-up said without the wake phrase may reach once the room is not quiet: the one list.

Heliox cannot tell who is speaking. The wake check is words only, so the wake phrase is not authentication: anyone
who says it is "the person". The follow-up window is a convenience that assumes the same person is still talking.
Once the listener has heard speech it did not take since Heliox last spoke (``overheard_since_heliox``: a second
voice, a television, Heliox's own echo), that assumption is weak. A follow-up then keeps its reach for talk but
loses it for anything that changes or reveals the person's own data. "Say Hey Heliox first for that." is said once
per follow-up window, and with the wake phrase it goes ahead. Stop, no and wait never need the wake phrase, from
anyone.

The rule is off by default (``conversation.second_voice_guard``) until the count is measured on recordings of a
person alone. The count is logged on every follow-up either way. See docs/INTERACTION_MODEL.md, "A second voice".
"""

from __future__ import annotations

import re

SAY_WAKE_FIRST = "Say Hey Heliox first for that."

# Handlers (pilot.server._control_reply, and "_local_answer:<source>") whose answers change or reveal the person's
# own data. A follow-up that one of them would take loses its reach. Kept: sums, units, dates and the time
# (QuickAnswers), "what can I say", the computer's health, and Live's replies to its own card.
DATA_HANDLERS = frozenset(
    {
        "_getting_to_know_request",  # the profile
        "_offer_reply",  # a yes sends a request in the person's name
        "_say_again_request",  # repeats the last reply, which may hold their data
        "_profile_request",
        "_goal_request",
        "_companion_request",
        "_suggestion_request",
        "_day_request",  # the day summary: reminders, to-dos, habits
        "_forget_request",
        "_focus_request",
        "_settings_request",
        "_workspace_request",
        "_undo_request",
        "_routine_request",
        "_shortcut_request",
        "_memory_talk_request",
        "_local_answer:ActivityTimeline",
        "_local_answer:TaskReceipts",
        "_local_answer:UndoJournal",
        "_local_answer:AppReminders",
        "_local_answer:ReminderStore",
        "_local_answer:NoteStore",
        "_local_answer:TodoStore",
        "_local_answer:UserProfile",
        "_local_answer:GoalStore",
        "_local_answer:HabitStore",
        "_local_answer:FocusLog",
        "_local_answer:NudgeStore",
        "_local_answer:FocusGuard",
        "_local_answer:WatcherStore",
        "_local_answer:ClipboardHistory",
        "_local_answer:SnippetStore",
        "_local_answer:FileFinder",
        "_local_answer:Generators",
    }
)

# The same list in words, read before any handler runs (a handler that changes something cannot be asked first and
# overruled after): what a follow-up says when it asks for the person's data or to change or send something.
_DATA_WORDS = re.compile(
    r"\b(?:remind|reminders?|to-?dos?|todo list|my list|on my list|to my list|notes?|note that|remember|forget|"
    r"what do you know|about me|my name|call me|my (?:goals?|habits?|day|calendar|schedule|meetings?|emails?|"
    r"mail|messages?|inbox|files?|folders?|documents?|downloads|desktop|photos?|clipboard|history|timeline|"
    r"settings?|profile|passwords?|contacts?|address|location|plan|plans)|clipboard|copied|paste|"
    r"what did i (?:do|say|tell you|copy|ask)|what have i|did i|good morning|good night|what'?s on|"
    r"undo|send|email|e-mail|message|text (?:him|her|them)|reply|post|delete|remove|erase|wipe|empty|"
    r"settings?|turn (?:on|off)|stop suggesting|suggest|offers?|shortcut|routine|workspace|focus|pomodoro|"
    r"files?|folders?|documents?)\b",
    re.IGNORECASE,
)
# Never held back, from anyone: stopping, saying no, asking to wait.
_STOP_WORDS = re.compile(
    r"^(?:stop|stop it|stop that|no|nope|no thanks|wait|hold on|cancel|cancel it|never mind|nevermind|enough|"
    r"that's enough|quiet|be quiet|shh+|not now)$",
    re.IGNORECASE,
)


def keeps_reach(words: str) -> bool:
    """Stop, no and wait: never held back."""
    return bool(_STOP_WORDS.match(" ".join(words.casefold().strip(" .!?,").split())))


def asks_for_data(words: str) -> bool:
    """The follow-up asks for, changes or sends something of the person's: it loses reach in a room that is not
    quiet. Talk, the time, sums and general questions do not."""
    return not keeps_reach(words) and bool(_DATA_WORDS.search(words))


def overheard_reason(overheard: int, threshold: int) -> str:
    """The listener's count as a reason the room is not quiet ("" when it is below the threshold). Other signals (a
    voice unlike the one that woke Heliox, if one earns its place) give their own reasons the same way."""
    return f"{overheard} utterance(s) left out since Heliox spoke" if overheard >= max(1, threshold) else ""


def guarded(*, enabled: bool, addressed: str, reasons: list[str] | tuple[str, ...]) -> bool:
    """Whether this turn's reach is limited: the rule is on, the turn came through the follow-up window (not the wake
    phrase, the button or a continuation), and some signal says the room is not quiet (``reasons``, from outside)."""
    return enabled and addressed == "follow_up" and any(reasons)
