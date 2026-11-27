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
    "message": "Messaging",
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
    "arrange": "Arranging",
}

# Polite or indirect ways of asking for an action: "can you open…", "please open…", "I need you to open…".
_REQUEST_PREFIX = re.compile(
    r"^(?:(?:hey |ok |okay |so |and |now |then |also |alright |right |well )*(?:please |kindly )?"
    r"(?:would you mind |can you |could you |would you |will you |can u |could u |"
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


# Media and volume requests that do not start with an action verb ("next song", "pause the music").
MEDIA_PATTERNS = (
    (re.compile(r"^(?:pause|play|resume|unpause)(?: the)? (?:music|song|track|video|playback|audio)$"), "playpause"),
    (re.compile(r"^(?:next|skip to the next) (?:song|track)$|^play the next (?:song|track)$"), "nexttrack"),
    (re.compile(r"^(?:previous|last|go back a) (?:song|track)$|^play the previous (?:song|track)$"), "prevtrack"),
    (re.compile(r"^(?:turn )?(?:the )?volume up$|^turn (?:it|the volume|the sound) up$|^louder$"), "volumeup"),
    (re.compile(r"^(?:turn )?(?:the )?volume down$|^turn (?:it|the volume|the sound) down$|^quieter$"), "volumedown"),
)
VOLUME_LEVEL = re.compile(r"^set (?:the )?volume to (\d{1,3})(?: percent| %)?$")


# ── the net from the action catalog (pilot.conversation.catalog_words) ──
# Reasons the net gives, which the layer reads (a read goes to the quick road only on its subject's words).
READ_REASON = "names something Heliox can read"
CATALOG_IMPERATIVE_REASON = "imperative from the action catalog"
_URL_IN_TEXT = re.compile(r"\bhttps?://\S+|\bwww\.[\w-]+\.\S+", re.IGNORECASE)
_PATH_IN_TEXT = re.compile(
    r"\b[a-z]:\\|\\\\[\w.$-]+\\|(?<![\w.])~?/(?:[\w.-]+/)+[\w.-]*|"
    r"\b[\w-]+\.(?:txt|md|log|csv|tsv|json|ya?ml|toml|ini|cfg|conf|py|js|ts|html?|css|xml|pdf|docx?|xlsx?|pptx?|png|"
    r"jpe?g|gif|zip|tar|gz|bin|exe|sh|ps1|bat|sql|db)\b",
    re.IGNORECASE,
)
# Questions and remarks that stay talk even when they name something an action reads or does: whether to, what Heliox
# thinks, how something works, what a word means, what happened in the past.
_STAYS_TALK = re.compile(
    r"^(?:should|shall) (?:i|we)\b|^(?:do|would|did|will) you (?:think|like|know|prefer|recommend|use|remember|mind|"
    r"believe|agree)\b|^what do you (?:think|reckon|recommend|suggest|make|know)\b|^what would you\b|"
    r"^which (?:is|one is|would be|one would be|of them is) (?:better|best|faster|safer)\b|"
    r"^is it (?:worth|safe|a good idea|ok|okay|bad|normal|possible|true|wise|smart)\b|^why\b|"
    r"^how (?:do|does|did|can|could|should|would|is it|are) \w+|^(?:can|could|may|might) (?:i|we)\b|"
    r"^do i (?:need|have to|want)\b|^what(?:'s| s| is) the (?:best|difference|point|deal|purpose|use)\b|"
    r"^what does\b|^if\b|^when (?:should|do|did|would|will|was|is)\b|^who\b|"
    r"^tell me (?:about|how|why|a |an |more|what you|something|if you)"
)
# The talk forms that ask about this computer when they name a reading plainly (a read subject, a path, an address).
_OVERRIDABLE_TALK = re.compile(
    r"^(?:who\b|what does\b|tell me about\b|(?:can|could) i (?:get|find|see)\b|when did (?:my|this|the)\b)"
)
# "What is git?", "what's a checksum?": a word asked about, not this computer's state.
_DEFINITION = re.compile(r"^what(?:'s| s| is| are) (?:a |an )?(?P<rest>.+)$")
_NOT_A_DEFINITION = re.compile(r"^(?:the|my|on|in|this|that|your|our|using|taking|left|running|open|currently)\b")
_HERE_AND_NOW = re.compile(
    r"\b(?:my|mine|this (?:computer|machine|pc|laptop|mac)|right now|currently|at the moment|now|left|free|"
    r"remaining|available|in use|used|level|running|installed|connected|today|tomorrow|open|do i have|have i got)\b|"
    r"^how (?:much|many|full|big|large)\b"
)
# Asking for words that are a reading: "tell me the volume level", "give me the git status", "I'd like to know ...".
_ASKS_FOR_A_READING = re.compile(
    r"^(?:tell me|give me|let me know|i(?: want| need| would like|'d like) to (?:know|see|check)|i wonder|"
    r"i (?:want|need) (?:the|my)\b|know (?:how|what|which|if|whether)\b)"
)
_SUBJECT_FILLER = frozenset({"the", "my", "a", "an", "please", "now", "right", "current", "currently", "and", "your"})
# What a reading is called after its subject: "clipboard contents", "battery level", "git status please".
_READING_WORDS = frozenset(
    {"contents", "content", "level", "status", "usage", "info", "information", "reading", "value", "percentage"}
)


def _anchored(text: str, words: str) -> bool:
    """Something an action works on: a subject of any action, a path, a file name or a web address."""
    from pilot.conversation import catalog_words

    return bool(catalog_words.names_anything(words) or _URL_IN_TEXT.search(text) or _PATH_IN_TEXT.search(text))


def _matched_subject(words: str) -> str:
    from pilot.conversation import catalog_words

    found = catalog_words.STRONG.search(words) or catalog_words.WEAK.search(words)
    return found.group(0) if found else ""


def catalog_net(text: str, normalized: str, body: str, *, conversation_open: bool = False) -> Route | None:
    """A turn the rules left as talk that names what an action reads or asks for what an action does, as the route
    it should have had; None when it is talk after all. Pure and cheap: a few compiled patterns, no model."""
    from pilot.conversation import catalog_words

    if not body:
        return None
    if _STAYS_TALK.match(normalized) or _STAYS_TALK.match(body):
        # "Who made the last commit?", "what does the file C:\\a.txt say?", "tell me about process 880": a talk form
        # around something only this computer can answer stays a reading.
        located_here = bool(_URL_IN_TEXT.search(text) or _PATH_IN_TEXT.search(text))
        # A read subject counts only with a sign of here and now ("in my repo", "on this computer"), a number (a
        # process id), or as what "can I get" asks for: "who invented the clipboard?" stays talk.
        plainly = located_here or bool(
            catalog_words.STRONG.search(normalized)
            and (
                _HERE_AND_NOW.search(normalized)
                or re.search(r"\b\d+\b", normalized)
                or re.match(r"(?:can|could) i ", normalized)
            )
        )
        if not (plainly and (_OVERRIDABLE_TALK.match(normalized) or _OVERRIDABLE_TALK.match(body))):
            return None
    words = body.split()
    verb = words[0]
    if verb in catalog_words.VERBS and verb not in ACTION_VERBS and len(words) > 1:
        target = " ".join(words[1:])
        if not [word for word in words[1:] if word not in _EMPTY_TARGET]:
            # "post it", "hash that": what "it" is lives in the conversation, or Heliox asks.
            kind = TurnKind.CONVERSATION if conversation_open else TurnKind.UNCLEAR
            return Route(kind, "the target is only a pronoun", verb=verb, target=target)
        if verb in catalog_words.ANCHORED_VERBS and not _anchored(text, target):
            return None
        if conversation_open and _DEICTIC.search(" ".join(words[1:4])) and len(words) <= 6:
            return Route(TurnKind.CONVERSATION, "refers back to the conversation", verb=verb, target=target)
        return Route(TurnKind.ACTION, CATALOG_IMPERATIVE_REASON, verb=verb, target=target)
    strong = bool(catalog_words.STRONG.search(normalized))
    weak = bool(catalog_words.WEAK.search(normalized))
    located = bool(_URL_IN_TEXT.search(text) or _PATH_IN_TEXT.search(text))
    asks = bool(
        _QUESTION_START.match(normalized)
        or text.strip().endswith("?")
        or _ASKS_FOR_A_READING.match(body)
        or _ASKS_FOR_A_READING.match(normalized)
    )
    if asks and (strong or weak or located):
        definition = _DEFINITION.match(normalized)
        # "what is process 880 doing" names a reading plainly; "what's a checksum" asks what a word means.
        plain_reading = strong and not re.match(r"^what(?:'s| s| is| are) (?:a|an) ", normalized)
        if definition and not located and not plain_reading and not _NOT_A_DEFINITION.match(definition.group("rest")):
            if not _HERE_AND_NOW.search(definition.group("rest")):
                return None  # "what is git", "what's a checksum": the word, not this computer
        if strong or located or _HERE_AND_NOW.search(normalized):
            return Route(TurnKind.ACTION, READ_REASON, verb="check", target=_matched_subject(normalized) or body)
        return None
    rest = " ".join(word for word in normalized.split() if word not in _SUBJECT_FILLER)
    if rest and len(rest.split()) <= 3:
        reading = " ".join(word for word in rest.split() if word not in _READING_WORDS)
        if catalog_words.STRONG.fullmatch(rest) or (reading and catalog_words.STRONG.fullmatch(reading)):
            return Route(TurnKind.ACTION, READ_REASON, verb="check", target=rest)  # "disk space?", "my battery"
        if catalog_words.WEAK.fullmatch(rest) or catalog_words.ANCHOR.fullmatch(rest):
            return Route(TurnKind.UNCLEAR, "names something Heliox could act on, and asks nothing")
    return None


def asks_for_work(text: str, decided: Route) -> bool:
    """Whether words routed as talk still ask for something to be done: the router read a verb in them (a pronoun or
    a sentence referring back, for the talk model to resolve), or they start with a verb Heliox acts on. Used when
    the talk model's reply claimed work that does not exist (pilot.conversation.work_claims): such words go to the
    planner; a question about something that is not running does not."""
    if decided.verb:
        return True
    normalized = normalize(text)
    if _STAYS_TALK.match(normalized):
        return False
    words = _strip_prefix(normalized).split()
    if not words:
        return False
    from pilot.conversation import catalog_words

    return words[0] in ACTION_VERBS or words[0] in catalog_words.VERBS


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
    for pattern, key in MEDIA_PATTERNS:
        if pattern.search(body):
            return Route(TurnKind.ACTION, "a media or volume key", verb="media", target=key)
    if VOLUME_LEVEL.match(body):
        return Route(TurnKind.ACTION, "a volume level", verb="set", target=body[4:])
    if _TALK_VERBS.match(body):
        reading = catalog_net(text, normalized, body) if _ASKS_FOR_A_READING.match(body) else None
        return reading or Route(TurnKind.CONVERSATION, "asks for words")
    if _LOOKUP_QUESTION.match(normalized):
        return Route(TurnKind.ACTION, "a question only the computer can answer", verb="check", target=body)

    verb = words[0]
    if verb.endswith("ing") and verb not in ACTION_VERBS and body != normalized:
        # "Would you mind muting the sound?", "could you try opening it": the gerund names the action.
        stem = verb[:-3]
        for candidate in (stem, stem + "e", stem[:-1]):
            if candidate in ACTION_VERBS:
                verb, words = candidate, [candidate, *words[1:]]
                body = " ".join(words)
                break
    if verb == "put" and words[1:2] == ["on"] and len(words) > 2:
        # "Put on some music", "put on my playlist": a request, worked out by the planner.
        return Route(TurnKind.ACTION, "put on", verb="play", target=" ".join(words[2:]))
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

    # The rules found no request. The action catalog's own words have the last say before talk (catalog_net): a
    # question about something a read action answers, or a verb that maps to an action, is a request after all.
    netted = catalog_net(text, normalized, body, conversation_open=conversation_open)
    if netted is not None:
        return netted
    if _SOCIAL.match(normalized) or _QUESTION_START.match(normalized) or text.strip().endswith("?"):
        return Route(TurnKind.CONVERSATION, "question or remark")
    if len(words) <= 1:
        return Route(TurnKind.UNCLEAR, "too short to tell")
    # A statement that is neither a request nor a question ("the build failed again"): talk about it.
    return Route(TurnKind.CONVERSATION, "statement")


# Right after the wake phrase or the button, a command verb with no subject in the past or -s form is the command,
# misheard: the recogniser turns "open Notepad" into "opened Notepad" and "mute" into "muted". Only then: "I opened
# Notepad earlier" has a subject and stays a remark, and words nobody addressed to Heliox never reach this.
_MISHEARD_VERBS = {
    "opened": "open",
    "opens": "open",
    "closed": "close",
    "closes": "close",
    "launched": "launch",
    "launches": "launch",
    "played": "play",
    "plays": "play",
    "paused": "pause",
    "pauses": "pause",
    "muted": "mute",
    "mutes": "mute",
    "unmuted": "unmute",
    "unmutes": "unmute",
    "skipped": "skip",
    "skips": "skip",
    "searched": "search",
    "searches": "search",
}
_NEEDS_NO_OBJECT = {"mute", "unmute", "pause", "play", "skip"}
_MISHEARD_ADDRESSING = {"wake", "button"}


def misheard_command(text: str, *, addressed: str) -> str | None:
    """The imperative a subjectless "opened Notepad" stands for, when the turn was addressed by the wake phrase or the
    button; None otherwise (or when the words are not that shape)."""
    if addressed not in _MISHEARD_ADDRESSING:
        return None
    words = normalize(text).split()
    if not words or words[0] not in _MISHEARD_VERBS:
        return None
    verb = _MISHEARD_VERBS[words[0]]
    rest = words[1:]
    if not rest and verb not in _NEEDS_NO_OBJECT:
        return None
    if rest and rest[0] in {"it", "that", "this", "them", "and", "but", "so", "because", "earlier", "yesterday"}:
        return None  # "opened it earlier", "closed and ...": a remark about something already done
    if len(rest) > 6:
        return None  # a long sentence is talk, not a misheard command
    original = text.strip().split(maxsplit=1)
    tail = original[1] if len(original) > 1 else ""
    return f"{verb.capitalize()} {tail}".strip()
