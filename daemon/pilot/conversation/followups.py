"""Follow-ups on what Heliox has just said, resolved in code when they need no model.

What "again" means depends on the last turn, as the server recorded it (``last_turn``): "action" when the executor
or the quick road carried out an action in that turn, "answer" when Heliox only spoke, "" when it is not known.

  Only about the words, at any time:
    "say that again", "what did you say", "what was that", "come again", "pardon"   ->  the last words again
  Either, within the follow-up window (FOLLOW_UP_SECONDS):
    "again", "once more", "one more time", "repeat", "repeat that", "repeat it"
      after an answer   ->  the words again
      after an action   ->  the action again, through the normal road (review and approvals as for any request)
      unknown           ->  as before round 8: "repeat that"/"repeat it" run the last task, the rest go to the model
  "Shorter", "in short", "the short version", within the window, after either   ->  the first sentence of what was
      said ("That was the short version." when it was one sentence). After an action report this is still about
      the words: "shorter" never repeats or changes an action.
  After the window, "repeat that" and "do that again" keep their meaning: run the last task again.
Follow-ups that change the content ("and tomorrow?", "what about the second one?") are left to the fast model, which
has the last answer in its conversation memory.
"""

from __future__ import annotations

import re

FOLLOW_UP_SECONDS = 8.0  # pilot.system.addressing.FOLLOW_UP_SECONDS: how long a reply needs no wake phrase

WORDS_ONLY = re.compile(
    r"^(?:please )?(?:say (?:that|it) again|what did you say|what was that|come again|pardon|sorry what)(?: please)?$"
)
EITHER = re.compile(r"^(?:please )?(?:again|once more|one more time|repeat|repeat (?:that|it))(?: please)?$")
TASK_AGAIN = re.compile(  # pilot.workflows.routines.REPEAT_REQUEST: always the task
    r"^(?:please )?(?:do (?:that|it) again|run (?:that|it) again|repeat (?:that|it|the last task|my last task))$"
)
SHORTER = re.compile(
    r"^(?:please )?(?:shorter|make it shorter|say it shorter|in short|briefly|short version|the short version|"
    r"just the short version|tl ?dr|give me the short version|the gist|just the gist)(?: please)?$"
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
SHORT_ALREADY = "That was the short version."


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", text.casefold()).replace("'", " ").split())


def first_sentence(text: str) -> str:
    return _SENTENCE_END.split(text.strip(), maxsplit=1)[0]


def repeat_meaning(text: str, *, last_turn: str, seconds_since: float) -> str:
    """ "words", "action" (run the last task again through the normal road), "shorter", or "" (not decided here)."""
    said = normalize(text)
    if WORDS_ONLY.match(said):
        return "words"
    inside = seconds_since <= FOLLOW_UP_SECONDS
    if SHORTER.match(said):
        return "shorter" if inside else ""
    if EITHER.match(said) and inside and last_turn in ("answer", "action"):
        return "words" if last_turn == "answer" else "action"
    if TASK_AGAIN.match(said):
        return "action"
    return ""


def follow_up_reply(text: str, *, last_spoken: str, seconds_since: float, last_turn: str = "") -> str | None:
    """What to say in code for a follow-up about the last words, or None when this is not one."""
    if not last_spoken:
        return None
    meaning = repeat_meaning(text, last_turn=last_turn, seconds_since=seconds_since)
    if meaning == "words":
        return last_spoken
    if meaning == "shorter":
        short = first_sentence(last_spoken)
        return short if short != last_spoken.strip() else SHORT_ALREADY
    return None
