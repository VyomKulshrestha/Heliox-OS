"""Decide in code what an addressed utterance is, before any model is asked.

The five kinds:

* ``control``: stop, wait, no, undo, a correction. The server's own handlers act on these.
* ``local``: something Heliox answers itself in code (what can I say, the to-do list, reminders, the day).
* ``conversation``: a question, a remark, a follow-up, a request for an opinion or an explanation; words, no action.
* ``action``: a request to do something on the computer. It goes to the planner.
* ``unclear``: too short, garbled or ambiguous to act on. The fast model asks, it never guesses an action.

The rules are ordered and cheap (a few regular expressions over at most a sentence), so a decision takes well under
a millisecond. They err towards ``conversation`` and ``unclear``: a question answered in words costs a second, an
action taken on a misheard sentence costs much more.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class TurnKind(StrEnum):
    CONTROL = "control"
    LOCAL = "local"
    CONVERSATION = "conversation"
    ACTION = "action"
    UNCLEAR = "unclear"


@dataclass(frozen=True, slots=True)
class Route:
    kind: TurnKind
    reason: str
    verb: str = ""  # the action verb, for the spoken acknowledgement
    target: str = ""  # what the verb applies to, as said ("notepad", "my launch folder")


def normalize(text: str) -> str:
    """Lower case, curly quotes made straight, punctuation dropped, single spaces."""
    text = text.casefold().replace("’", "'").replace("‘", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    return " ".join(text.split())


# Whole utterances that stop, deny, pause or undo. The server's handlers own what they do.
_CONTROL_WHOLE = frozenset(
    {
        "stop",
        "stop it",
        "stop stop",
        "stop that",
        "wait",
        "wait wait",
        "wait stop",
        "hold on",
        "hang on",
        "no",
        "nope",
        "no stop",
        "cancel",
        "cancel that",
        "cancel it",
        "never mind",
        "nevermind",
        "forget it",
        "undo",
        "undo that",
        "undo it",
        "that's enough",
        "enough",
        "be quiet",
        "quiet",
        "shut up",
        "pause",
        "don't",
        "don't do that",
        "don't do it",
    }
)
# A sentence that only stops, skips or forbids. Corrections that choose something else ("no, the second one",
# "not that folder, the other one") are not here: they refer to the conversation and go to the fast model.
_CONTROL_START = re.compile(
    r"^(?:skip (?:the|that|it)|don't (?:do|delete|send|open|run|touch)|undo the last|stop (?:doing|talking))\b"
)
# A control phrase at the start of a longer sentence: "that's enough for the backup folder, now zip it",
# "stop there and then open the report". What follows it decides the route.
_CONNECTIVE = r"(?:and |but |so )?(?:then|now|instead|next|also)\b"
_CLAUSE_END = r"(?=\s*[,;.]|\s+" + _CONNECTIVE + r"|\s*$)"
# Matched on the words with their commas kept, so "enough of that, what's next" ends the phrase at the comma.
_LEADING_CONTROL = re.compile(
    r"^(?:(?:okay|ok|alright)\s*,?\s+)?"
    r"(?:(?:that's |thats )?enough(?: (?:for|of|with) (?P<object>[^,;.]{1,40}?))?|"
    r"stop(?: there| that| it| here)?|wait(?: wait)?|hold on|hang on|never mind(?: that)?|forget (?:it|that)|"
    r"cancel (?:that|it))" + _CLAUSE_END + r"\s*[,;.]?\s*(?:" + _CONNECTIVE + r")?\s*"
)
_EMPTY_REST = frozenset({"", "now", "please", "right now", "then", "thanks", "thank you"})


def _with_commas(text: str) -> str:
    text = text.casefold().replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"[^\w\s',;.]", " ", text)
    return " ".join(text.split())


# What Heliox answers itself (pilot.server._control_reply and its handlers). Listed so the router can say so; in
# the voice path those handlers run before this layer is asked.
_LOCAL = re.compile(
    r"^(?:what can i (?:say|ask|do)(?: to you)?|help|what can you do|"
    r"what(?:'s| is) on my (?:to ?do|todo)(?: list)?|(?:read|show) (?:me )?my (?:to ?do|todo)s?(?: list)?|"
    r"what (?:are|were) my (?:to ?dos|todos|reminders)|what reminders do i have|"
    r"what did (?:you|i) do (?:today|yesterday|this morning)|what have you done today|"
    r"say (?:that|it) again|what did you say|come again|pardon|"
    r"what do you know about me|who am i|call me \w+|"
    r"what'?s my (?:name|goal|goals)|"
    r"(?:start|stop) (?:a |the )?(?:pomodoro|focus(?:ing)?(?: session)?)|"
    r"why did you say that|thanks|thank you|thanks heliox|not now|don't mention (?:these|this|that)|fix it)$"
)

# Requests to act. Each verb maps to the gerund used in the acknowledgement.
ACTION_VERBS: dict[str, str] = {
    "open": "Opening",
    "launch": "Opening",
    "start": "Starting",
    "close": "Closing",
    "quit": "Closing",
    "exit": "Closing",
    "minimize": "Minimizing",
    "minimise": "Minimizing",
    "maximize": "Maximizing",
    "maximise": "Maximizing",
    "find": "Looking for",
    "search": "Searching",
    "look": "Looking",
    "locate": "Looking for",
    "create": "Creating",
    "make": "Making",
    "write": "Writing",
    "draft": "Drafting",
    "type": "Typing",
    "delete": "Deleting",
    "remove": "Removing",
    "move": "Moving",
    "copy": "Copying",
    "rename": "Renaming",
    "zip": "Zipping",
    "compress": "Compressing",
    "extract": "Extracting",
    "unzip": "Unzipping",
    "download": "Downloading",
    "install": "Installing",
    "uninstall": "Uninstalling",
    "update": "Updating",
    "upgrade": "Updating",
    "run": "Running",
    "set": "Setting",
    "turn": "Turning",
    "switch": "Switching",
    "mute": "Muting",
    "unmute": "Unmuting",
    "play": "Playing",
    "send": "Sending",
    "email": "Emailing",
    "reply": "Replying",
    "schedule": "Scheduling",
    "book": "Booking",
    "add": "Adding",
    "organize": "Organizing",
    "organise": "Organizing",
    "sort": "Sorting",
    "clean": "Cleaning",
    "clear": "Clearing",
    "save": "Saving",
    "back": "Backing",
    "backup": "Backing up",
    "take": "Taking",
    "screenshot": "Taking a screenshot of",
    "go": "Going",
    "navigate": "Going",
    "visit": "Opening",
    "browse": "Opening",
    "click": "Clicking",
    "press": "Pressing",
    "scroll": "Scrolling",
    "check": "Checking",
    "show": "Showing",
    "list": "Listing",
    "count": "Counting",
    "convert": "Converting",
    "resize": "Resizing",
    "print": "Printing",
    "translate": "Translating",
    "summarize": "Summarizing",
    "summarise": "Summarizing",
    "read": "Reading",
    "fix": "Fixing",
    "restart": "Restarting",
    "reboot": "Restarting",
    "shut": "Shutting",
    "lock": "Locking",
    "kill": "Stopping",
    "pin": "Pinning",
    "share": "Sharing",
    "upload": "Uploading",
    "commit": "Committing",
    "push": "Pushing",
    "pull": "Pulling",
    "remind": "Setting a reminder",
    "record": "Recording",
    "plan": "Planning",
    "arrange": "Arranging",
}

# Polite or indirect ways of asking for an action: "can you open…", "please open…", "I need you to open…".
_REQUEST_PREFIX = re.compile(
    r"^(?:(?:hey |ok |okay |so |and |now |then |also |alright |right |well )*(?:please |kindly )?"
    r"(?:can you |could you |would you |will you |can u |could u |would you mind |"
    r"i (?:want|need|would like|'d like) (?:you )?to |i'd like you to |"
    r"go ahead and |let's |lets |help me |try to |try and |just |quickly |please )*)"
)
# A verb that reads as an action but here asks for words: "explain", "tell me about", "what do you think of".
_TALK_VERBS = re.compile(
    r"^(?:explain|describe|tell me|talk|say|think|guess|suggest|recommend|remind me what|remind me how|"
    r"remind me why|teach|define|compare|help me understand|help me think|walk me through|imagine)\b"
)
_QUESTION_START = re.compile(
    r"^(?:what|what's|whats|why|how|how's|hows|who|who's|whose|where|where's|when|which|is|are|am|was|were|do|"
    r"does|did|can|could|should|would|will|have|has|any|isn't|aren't|don't|doesn't|didn't|shall)\b"
)
# Questions whose honest answer needs Heliox to look at or do something on the computer.
_LOOKUP_QUESTION = re.compile(
    r"^(?:what(?:'s| is) (?:the )?(?:size|weather|disk|battery|ip|cpu|memory|ram|free space)|"
    r"how much (?:disk|space|memory|ram|battery|storage)|how many (?:files|tabs|windows)|"
    r"is (?:my |the )?\w+ (?:running|open|installed|connected|online|up to date)|"
    r"which (?:files|apps|programs|processes)|where(?:'s| is) (?:my|the) \w+ (?:file|folder|document))"
)
_SOCIAL = re.compile(
    r"^(?:hi|hello|hey|hiya|yo|good (?:morning|afternoon|evening|night)|morning|evening|how are you|how's it going|"
    r"hows it going|what's up|whats up|sup|nice|cool|great|awesome|perfect|wow|interesting|ha|haha|lol|"
    r"ok|okay|alright|right|sure|yes|yeah|yep|got it|i see|makes sense|fair enough|sorry|my bad|bye|goodbye|"
    r"see you|good job|well done|nice one|love it|that's great|that's cool|that's interesting|that's weird|"
    r"that's wrong|that's not right|i'm (?:back|tired|bored|done|here|stuck|confused|lost|annoyed|happy)|"
    r"i am (?:back|tired|bored|done|here|stuck|confused|lost)|i think|i feel|i guess|i wonder|i don't know|"
    r"i don't understand|i'm not sure|never heard|it's|it is|this is|that is|that was|that's)\b"
)
# Pronouns that point back at earlier talk: "open it", "send that to her". With the conversation they refer to, the
# fast model resolves them and hands over what it understood; without one, Heliox asks instead of guessing.
_DEICTIC = re.compile(r"\b(?:it|that|this|those|these|them|one|her|him|there|same|again)\b")
# Words that name nothing on their own: a target made only of these ("it", "up", "for your own") is no target.
_EMPTY_TARGET = frozenset(
    {
        "it",
        "that",
        "this",
        "those",
        "these",
        "them",
        "one",
        "there",
        "here",
        "same",
        "again",
        "up",
        "on",
        "for",
        "at",
        "in",
        "the",
        "a",
        "an",
        "your",
        "my",
        "own",
        "me",
        "us",
        "now",
        "please",
        "then",
        "out",
        "to",
        "of",
        "and",
        "thing",
        "stuff",
        "something",
        "look",
        "go",
        "ahead",
        "back",
    }
)
_FILLER = frozenset({"um", "uh", "erm", "hmm", "mm", "ah", "eh", "oh", "like", "so", "and", "the", "a"})
_WAKE_ONLY = frozenset({"heliox", "hey heliox", "hello heliox", "hi heliox", "ok heliox", "okay heliox", "pilot"})
# Requests whose verb alone is the whole request.
_SINGLE_WORD_ACTIONS = frozenset(
    {"mute", "unmute", "screenshot", "lock", "minimize", "minimise", "maximize", "maximise"}
)
# "take" acts only on these ("take a screenshot", "take a note"); "take a look" is talk.
_TAKE_OBJECTS = re.compile(r"\b(?:screenshot|screen shot|photo|picture|note|break|backup)\b")


def _strip_prefix(normalized: str) -> str:
    match = _REQUEST_PREFIX.match(normalized)
    return normalized[match.end() :] if match else normalized


def _looks_garbled(normalized: str) -> bool:
    words = normalized.split()
    if not words:
        return True
    content = [word for word in words if word not in _FILLER]
    if not content:
        return True
    # The recogniser repeating one word ("Google, Google, Google", "open, close, close, close") is noise.
    most = max(words.count(word) for word in set(words))
    return most >= 3 and most * 2 >= len(words)


def route(text: str, *, conversation_open: bool = False) -> Route:
    """Classify one addressed utterance. ``conversation_open``: earlier turns exist that pronouns can refer to."""
    normalized = normalize(text)
    if not normalized or normalized in _WAKE_ONLY:
        return Route(TurnKind.UNCLEAR, "nothing but the wake phrase")
    if normalized in _CONTROL_WHOLE or _CONTROL_START.match(normalized):
        return Route(TurnKind.CONTROL, "control word")
    lead = _LEADING_CONTROL.match(_with_commas(text))
    if lead:
        rest_text = _with_commas(text)[lead.end() :]
        if normalize(rest_text) in _EMPTY_REST:
            return Route(TurnKind.CONTROL, "control phrase")
        # Routed by the whole sentence: the instruction after the control phrase is what the person wants, and
        # "it" in it can be what the phrase named ("that's enough for the backup folder, now zip it").
        rest = route(rest_text, conversation_open=conversation_open)
        named = (lead.group("object") or "").strip()
        if named and rest.reason == "the target is only a pronoun":
            return Route(TurnKind.ACTION, "after a control phrase, on what it named", verb=rest.verb, target=named)
        if rest.kind is not TurnKind.UNCLEAR:
            return Route(rest.kind, f"after a control phrase: {rest.reason}", verb=rest.verb, target=rest.target)
        return Route(TurnKind.CONVERSATION, "a control phrase and something unclear")
    if _looks_garbled(normalized):
        return Route(TurnKind.UNCLEAR, "repeated or filler words")
    if _LOCAL.match(normalized):
        return Route(TurnKind.LOCAL, "Heliox answers this in code")

    body = _strip_prefix(normalized)
    words = body.split()
    if not words:
        return Route(TurnKind.UNCLEAR, "only a polite prefix")
    if _TALK_VERBS.match(body):
        return Route(TurnKind.CONVERSATION, "asks for words")
    if _LOOKUP_QUESTION.match(normalized):
        return Route(TurnKind.ACTION, "a question only the computer can answer", verb="check", target=body)

    verb = words[0]
    if verb in ACTION_VERBS:
        target = " ".join(words[1:])
        if verb == "look" and not re.match(r"^(?:for|up)\b", target):
            return Route(TurnKind.CONVERSATION, "look as a remark")
        if verb == "take" and not _TAKE_OBJECTS.search(target):
            return Route(TurnKind.CONVERSATION, "take as a figure of speech")
        if verb == "show" and target.startswith("me how"):
            return Route(TurnKind.CONVERSATION, "asks to be shown how, in words")
        if len(words) == 1:
            if verb in _SINGLE_WORD_ACTIONS:
                return Route(TurnKind.ACTION, "a one-word request", verb=verb)
            return Route(TurnKind.UNCLEAR, "a verb with nothing to act on", verb=verb)  # "Open." "Launch."
        if not [word for word in words[1:] if word not in _EMPTY_TARGET]:
            # "open it", "launch that", "open up": what "it" is lives in the conversation, or nowhere.
            kind = TurnKind.CONVERSATION if conversation_open else TurnKind.UNCLEAR
            return Route(kind, "the target is only a pronoun", verb=verb, target=target)
        if conversation_open and _DEICTIC.search(" ".join(words[1:4])) and len(words) <= 6:
            return Route(TurnKind.CONVERSATION, "refers back to the conversation", verb=verb, target=target)
        return Route(TurnKind.ACTION, "imperative", verb=verb, target=target)

    if _SOCIAL.match(normalized) or _QUESTION_START.match(normalized) or text.strip().endswith("?"):
        return Route(TurnKind.CONVERSATION, "question or remark")
    if len(words) <= 1:
        return Route(TurnKind.UNCLEAR, "too short to tell")
    # A statement that is neither a request nor a question ("the build failed again"): talk about it.
    return Route(TurnKind.CONVERSATION, "statement")
