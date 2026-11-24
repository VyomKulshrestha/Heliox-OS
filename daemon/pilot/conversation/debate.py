"""Heliox disagrees with a decision when it has a reason it can name, holds that for a moment, then gives way.

The setting ``conversation.disagree``: "off", "when_it_matters" (the default) or "often". Nothing here calls a model:
the rules ride in the talk prompt the layer already sends (``prompt_section``), and the state of a disagreement
reaches the model as one short context line (``Debate.context_line``).

How the layer knows the model disagreed: a reply that disagrees ends with a last line ``STANCE: <topic>``, which is
never spoken (``StanceFilter`` takes it out of the stream; it comes after the words, so it adds nothing before the
first sound). The layer keeps the topic, what Heliox said and how many replies held the point (``Stance``):

- the first reply that disagrees opens the stance (count 1);
- a reply that holds the point on the same topic counts 2;
- with count 2, the next reply on it is told to give way, and the stance is settled whatever the model writes, so a
  topic takes at most three exchanges, the last one a deferral;
- "go ahead anyway" (``go_ahead``) settles it at once: the reply gives way and does what was asked;
- a reply with no STANCE line (a concession, or another subject) settles it too, and a settled topic is not raised
  again in this conversation.

For a request to act, the model writes its reason in the ACT line's "concern". A small step that can be undone is
done, and the concern is said with the acknowledgement. A step that cannot be undone, costs money or affects other
people (``costly``) is asked about once first, as an offer: a yes or "do it anyway" hands it over through the usual
road. Nothing here approves, refuses or changes what a gate decides: the planner, its reviews and the approvals run
as for any request, and stop, wait and no never reach this code.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

SETTINGS = ("off", "when_it_matters", "often")
DEFAULT = "when_it_matters"
MAX_EXCHANGES = 3  # replies on one topic, the last of them giving way
SETTLED_KEPT = 4  # settled topics kept for "don't raise it again"
TOPIC_CHARS = 60
POSITION_CHARS = 160
# A reply cut at a question or the word cap: how long the layer waits, saying nothing, for the stream's STANCE line.
STANCE_WAIT_SECONDS = 1.0
RISKS = frozenset({"cannot_undo", "costs_money", "affects_others"})
# Steps whose effect cannot be taken back or reaches beyond the computer: a concern about one is said before it is
# done, whatever the model marked. The model's "risk" adds steps this list does not know.
COSTLY_VERBS = frozenset(
    {
        "delete", "remove", "erase", "wipe", "empty", "format", "uninstall", "overwrite", "purge", "clear",
        "send", "email", "e-mail", "message", "text", "forward", "reply", "post", "publish", "share", "upload",
        "pay", "buy", "purchase", "order", "book", "subscribe", "transfer", "donate", "cancel", "decline",
        "shutdown", "shut", "restart", "reset", "kill",
    }
)  # fmt: skip


def setting_value(raw: object) -> str:
    """The setting as one of SETTINGS; anything else is the default."""
    value = str(raw or "").strip().casefold().replace(" ", "_").replace("-", "_")
    return value if value in SETTINGS else DEFAULT


def effective(setting: str, *, local: bool) -> str:
    """What a model is asked for. A model on this computer may be small, and small models told to speak up "often"
    argue on weak grounds: for one, "often" is asked as "when_it_matters". The hard lines are code, so this is about
    how it sounds, never about what it may do."""
    value = setting_value(setting)
    return "when_it_matters" if local and value == "often" else value


_LEVEL = {
    "when_it_matters": "Speak up only when it matters: the clash or the risk is real and easy to miss. "
    "Let small things go without comment.",
    "often": "Speak up whenever you have such a reason, even a small one, and name the trade-off. Still let pure taste "
    "go.",
}


def prompt_section(setting: str, *, spoken: bool = True) -> str:
    """The rules added to the talk prompt for this setting ("" when off)."""
    value = setting_value(setting)
    if value == "off":
        return ""
    how = "one or two short spoken sentences" if spoken else "one or two short sentences"
    return (
        "\nWhen you disagree:\n"
        f"- When the person tells you a decision or asks for something and you have a reason against it, say so in {how} "
        'with the reason, about 15 words in all, the way a friend would: "Your review is at three, so a long lunch at '
        'half two is tight."\n'
        "- Only for a reason you can name: something you are told about them (their calendar, to-dos, habits, files, "
        "what they said earlier), a named risk (it cannot be undone, it costs money, it affects other people), or a "
        "trade-off you state. No reason, no disagreement. Never disagree for show, over taste, or on a guess.\n"
        f"- {_LEVEL[value]}\n"
        '- Say it to them, as "you": "That\'s your only copy.", never "their" or "the person". State the '
        'fact; never judge them (no "you may not have weighed", no "please be careful").\n'
        "- Every reply that pushes back ends with one more line: STANCE: <the topic in two to five words>. Heliox "
        "does not say that line aloud. Never write it in a reply that agrees or gives way.\n"
        '- If they give a new reason, weigh it. If it answers your point, say so in a few words, like "Fair, that '
        'changes it.", and drop it.\n'
        '- If they say to go ahead anyway, give way in a few words, like "Okay, your call.", and do what they asked.\n'
        "- Disagreeing never stops Heliox doing what they ask. When they ask for something to be done and you have a "
        'reason against it, still write the ACT line, with your reason in "concern" (one short sentence). If the '
        'step cannot be undone, costs money or affects other people, also add "risk": "cannot_undo", "costs_money" '
        'or "affects_others"; Heliox then says the concern and asks once before doing it.\n'
        "- Never argue with stop, wait, cancel or no. Voice still cannot approve anything.\n"
    )


def system_prompt(base: str, setting: str, *, local: bool = False, spoken: bool = True) -> str:
    """The talk prompt with this setting's rules. Off gives ``base`` unchanged."""
    return base + prompt_section(effective(setting, local=local), spoken=spoken)


# ── the STANCE line, never spoken ──

MARK = "STANCE:"
_MARK = re.compile(r"\[?STANCE\]?\s*:")
# A model that writes the line in another case puts it on a line of its own; such a sentence is never said either.
_MARK_LINE = re.compile(r"^\s*\[?stance\]?\s*:\s*(?P<topic>.*)$", re.IGNORECASE | re.DOTALL)
_HOLDABLE = ("[STANCE]", "[STANCE:", MARK, "STANCE :")


class StanceFilter:
    """Takes the STANCE line out of a streamed reply. ``feed`` gives back the text to speak; the last few characters
    when they could be the start of the line ("S", "STAN") are held until the next piece shows they are not. A
    sentence is cut only at its end followed by a space, so holding them never keeps a sentence from the voice."""

    def __init__(self) -> None:
        self._held = ""
        self._tag: str | None = None
        self._ended = False  # the line ended: what follows it (an ACT line) is the reply's again

    def feed(self, fragment: str) -> str:
        if self._tag is not None and not self._ended:
            self._tag += fragment
            body = self._tag.lstrip(" \t")
            end = body.find("\n")
            if end < 0:
                return ""
            self._tag, self._ended = body[:end], True
            fragment = body[end + 1 :]
            if not fragment:
                return ""
        text = self._held + fragment
        self._held = ""
        match = _MARK.search(text) if self._tag is None else None
        if match is not None:
            before = text[: match.start()]
            self._tag = ""
            return before + self.feed(text[match.end() :])
        held = _held_suffix(text)
        if held:
            self._held = text[-held:]
            return text[:-held]
        return text

    def finish(self) -> str:
        """The end of the stream: what was held turned out to be words."""
        rest, self._held = self._held, ""
        return rest

    @property
    def topic(self) -> str:
        return clean_topic(self._tag or "")


def _held_suffix(text: str) -> int:
    for size in range(min(len(text), max(len(mark) for mark in _HOLDABLE)), 0, -1):
        tail = text[-size:]
        if any(mark.startswith(tail) for mark in _HOLDABLE):
            return size
    return 0


def stance_sentence(sentence: str) -> str | None:
    """The topic when a whole sentence is a STANCE line (any case), else None. Such a sentence is never said."""
    match = _MARK_LINE.match(sentence)
    return clean_topic(match.group("topic")) if match else None


def clean_topic(text: str) -> str:
    first = text.strip().splitlines()[0] if text.strip() else ""
    first = re.sub(r"[\[\]{}<>\"*`]", "", first).strip(" .,:;-")
    words = first.split()[:8]
    return " ".join(words)[:TOPIC_CHARS]


_SMALL = frozenset({"the", "a", "an", "my", "your", "to", "of", "on", "for", "and", "with", "about", "it", "this"})


def _content(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", text.casefold())
    return {word.rstrip("s") for word in words if len(word) >= 4 and word not in _SMALL}


def same_topic(a: str, b: str) -> bool:
    """Two names for one topic: the same words, or a word of four letters or more in common ("deleting the backups",
    "the old backup folder")."""
    if not a or not b:
        return False
    if a.casefold() == b.casefold():
        return True
    return bool(_content(a) & _content(b))


# "Do it anyway", "go ahead", "I know, just do it", "I'm sure": the person keeps their decision. Never "no", "stop",
# "don't" or "wait": those stop or deny, and the server and the router own them.
_GO_AHEAD = re.compile(
    r"^(?:(?:yes|yeah|yep|ok|okay|fine|look|well|no|nah|honestly|seriously|i know|i hear you|i get it|understood|"
    r"noted|thanks|thank you|still|but)\s+)*"
    r"(?:(?:just |please |still )*(?:do it|go ahead|go for it|carry on|proceed|do that|send it|book it|delete it)"
    r"(?: (?:anyway|anyways|regardless|please|now|then|all the same))*"
    r"|i'?m sure|i am sure|i'?m doing it anyway|i'?m going anyway|"
    r"i(?:'ll| will|'m going to|'m gonna|m going to|ll) (?:\w+ ){0,3}?\w+ (?:anyway|anyways|regardless)|"
    r"(?:it'?s )?my (?:call|decision|choice)|i(?:'ve| have) decided|i still want (?:to|it)|do it anyway|"
    r"(?:just )?\w+ (?:it|them|that|those)(?: \w+)? (?:anyway|anyways|regardless))"
    r"(?: anyway| please| thanks)*$"
)


def go_ahead(words: str) -> bool:
    """The person keeps their decision after a disagreement (``words`` already normalized: lower case, no
    punctuation)."""
    return bool(_GO_AHEAD.match(" ".join(words.split())))


def costly(verb: str, risk: str) -> bool:
    """A step whose concern is said before it is done: the model marked a risk, or the verb is one of COSTLY_VERBS."""
    return risk in RISKS or verb.casefold().strip() in COSTLY_VERBS


# ── the second signal: a reply that pushes back without the STANCE line ──
#
# The model does not always write the line. A reply is also counted as holding a point when its words push back on
# the decision: a clash, a cost to the person, a step that cannot be undone, or Heliox saying it would not. Read from
# the reply's own text after it was written, so it costs no call and nothing waits for it. Kept narrow on purpose: a
# reply counted by mistake only makes Heliox give way sooner; it never makes it argue or hold anything back.
_PUSHES_BACK = re.compile(
    r"\b(?:"
    r"(?<!sleep )(?<!hold )tight|won't work|but you(?:'d| would|'ll| will)|^maybe,? but|"
    r"you(?:'d| would|'ll| will) be (?:up late|tired|exhausted|short|rushed|starting)|"
    r"(?:less|little|short on|lose|losing) sleep|keeps? you (?:up|awake)|short of (?:the|your)|"
    r"i'?d (?:put|do|start with|finish) (?:that|it|the \w+(?: \w+)?) first|"
    r"(?:isn't|is not|aren't|are not) done|hasn't (?:been )?(?:signed|approved|reviewed)|signs? off|"
    r"(?:is|that's|it's) a lot (?:to|for)|still want (?:it|to|me to)\b|"
    r"no time (?:to|for|left)|not enough time|(?:too|very) (?:late|close) (?:for|to|before)|clash(?:es)? with|"
    r"conflicts? with|overlaps? with|double[- ]booked|"
    r"(?:would|will|could|might) (?:permanently )?(?:delete|erase|wipe|lose|cost you|miss|eat into|cut into)|"
    r"(?:cost|costs|costing) you|leaves? (?:you )?(?:no|very little|little|barely)|"
    r"you(?:'ll| will| would|'d) (?:be late|miss|lose|struggle|be wiped|be exhausted|regret)|"
    r"or you(?:'ll| will) (?:be late|miss|lose)|(?:gets?|will be) missed|"
    r"(?:can't|cannot|can not) be undone|no other copy|(?:the )?only copy|no backup|"
    r"i(?:'d| would) (?:hold off|wait|skip|rethink|reconsider|not|avoid|think twice)|i wouldn't|"
    r"might want to (?:rethink|reconsider|wait|hold off|move|skip)|are you sure|"
    r"(?:is|isn't) that (?:wise|a lot|too)|that's a lot|(?:that's|it's|that is|it is|a bit) risky|risky to|"
    r"not (?:a good|the best) (?:idea|time)|before (?:legal|the review|it'?s (?:signed|reviewed|approved))"
    r")\b",
    re.IGNORECASE,
)
# A reply that opens by giving way ("Okay, your call.", "Fair, that changes it.") is not holding a point, whatever it
# adds after ("Keep an eye on the time.").
_GIVES_WAY = re.compile(
    r"^\W*(?:okay|ok|alright|all right|sure|fine)?[,.!]?\s*(?:your call|fair(?: enough)?\b|that changes it|"
    r"i'll drop it|good point|you're right|makes sense|go for it|enjoy)",
    re.IGNORECASE,
)


def gives_way(reply: str) -> bool:
    """A reply that opens by giving way ("Okay, your call.", "Fair, that changes it.")."""
    return bool(_GIVES_WAY.match(" ".join(str(reply or "").split())))


def pushes_back(reply: str) -> bool:
    """Whether a reply's words push back on a decision (the second signal; the STANCE line is the first)."""
    text = " ".join(str(reply or "").split())
    if not text or _GIVES_WAY.match(text):
        return False
    return bool(_PUSHES_BACK.search(text))


# A reply that opens by agreeing ("Okay, you're leaving at two.", "Good, the early train it is.").
_AGREES = re.compile(r"^\W*(?:okay|ok|good|great|sounds good|perfect|lovely|sure|got it|nice|noted)\b", re.IGNORECASE)


def false_mark(reply: str) -> bool:
    """A STANCE line on a reply that agrees: it opens by agreeing and no words push back. Small local models write
    the line on such replies (seen with a 2B model on 7 of 16 replies); it is not counted."""
    text = " ".join(str(reply or "").split())
    return bool(_AGREES.match(text)) and not _PUSHES_BACK.search(text)


def topic_of(words: str) -> str:
    """A topic named from the person's own words, for a disagreement the model did not mark (normalized words in)."""
    content = [word for word in words.split() if len(word) >= 4 and word not in _SMALL and word not in _PLAN_WORDS]
    return " ".join(content[:5])[:TOPIC_CHARS]


_PLAN_WORDS = frozenset({"going", "gonna", "think", "want", "need", "will", "it'll", "i'll", "i'm", "let's", "just"})
# The person telling Heliox their own plan ("I'll book the early train"): the step is theirs, not one to hand over.
_OWN_PLAN = re.compile(
    r"^(?:so |and |well |ok |okay )?(?:i'?ll|i will|i'?m going to|i am going to|i'?m gonna|i think i'?ll|i plan to|i might)\b"
)


# Words that make the plan a request after all: "I'm going to need you to delete it", "I'll have Heliox send it".
_ASKS_HELIOX = re.compile(r"\b(?:you|heliox)\b")
_ADDRESS = re.compile(r"^(?:(?:hey|hi|ok|okay) )?heliox ")  # "Hey Heliox, I'm going to ..." when the name was kept


def own_plan(words: str) -> bool:
    """The person told Heliox what they will do themselves ("I'm going to delete my only copy of the thesis", "I'll
    book the early train"), and did not ask Heliox to do it. ``words`` normalized."""
    text = _ADDRESS.sub("", " ".join(words.split()))
    return bool(text and _OWN_PLAN.match(text)) and not _ASKS_HELIOX.search(text)


def said_already(concern: str, said: str) -> bool:
    """The reply's own words already gave the concern: at least half of its words of four letters or more are in what
    was said ("It is the only copy and cannot be undone." after "That is your only copy, so it cannot be undone.")."""
    wanted = _content(concern)
    return bool(wanted) and 2 * len(wanted & _content(said)) >= len(wanted)


def _forms(verb: str) -> set[str]:
    forms = {verb, verb + "s", verb + "ed", verb + "ing"}
    if verb.endswith("e"):
        forms |= {verb + "d", verb[:-1] + "ing"}
    if verb.endswith("y"):
        forms |= {verb[:-1] + "ied", verb[:-1] + "ies"}
    return forms


_COSTLY_FORMS = frozenset(
    {form for verb in COSTLY_VERBS for form in _forms(verb)} | {"sent", "paid", "bought", "shut down", "emptied"}
)


def asks_for_costly_step(words: str) -> str:
    """The costly verb when the person asked Heliox for a step that cannot be undone, costs money or affects others
    ("I want the recycle bin emptied", "send the contract to Dana"); "" otherwise, and "" for their own plan ("I'll
    book the early train"). ``words`` normalized."""
    text = " ".join(words.split())
    if not text or own_plan(text):
        return ""
    return next((word for word in text.split() if word in _COSTLY_FORMS), "")


# The person holding to their decision after Heliox spoke ("It'll be fine.", "I really need to.", "But the report can
# wait."): each such turn is one more exchange on the same thread, whatever Heliox's reply was marked. Read from the
# person's words before the model is asked, so it needs nothing from the model; "do it anyway" counts too (go_ahead).
_INSISTS = re.compile(
    r"^(?:(?:no|nah|yes|yeah|yep|look|well|okay|ok|honestly|seriously|still|and)\s+)*"
    r"(?:it'?ll be (?:fine|okay|ok|alright)|it'?s (?:fine|okay|ok|alright|not a problem|no problem|worth it)|"
    r"i'?ll (?:manage|be fine|be okay|be ok|cope|make it|sort it|handle it)|"
    r"i (?:really |still |just )?(?:need|want|have) to\b|i (?:really|still|just) (?:need|want)\b|"
    r"i don'?t care|don'?t worry|trust me|"
    r"i know what i'?m doing|i'?ve got (?:it|this)|i can handle it|i'?m still going|i'?m going anyway|"
    r"i'?m (?:still )?doing it|"
    r"i still (?:want|need|think)|that'?s fine|it doesn'?t matter|i'?d rather|"
    r"but\b|because\b|i know\b|i hear you\b|i get it\b|yeah but|yes but)"
)


_WITHDRAWS = re.compile(
    r"\b(?:don'?t|do not|never ?mind|cancel|stop|wait|leave (?:it|them)|keep (?:it|them)|hold off|not now|"
    r"restore|check (?:it|the|that) first|let me check|on second thought|you'?re right)\b"
)


def withdraws(words: str) -> bool:
    """The person backs off the step (``words`` normalized): it is then never done on Heliox's giving way."""
    return bool(_WITHDRAWS.search(" ".join(words.split())))


def insists(words: str) -> bool:
    """The person holds to their decision (``words`` normalized)."""
    return bool(_INSISTS.match(" ".join(words.split())))


@dataclass
class Stance:
    topic: str
    position: str  # what Heliox said when it disagreed
    count: int = 1  # replies that held the point
    offered: bool = False  # a costly step asked about once: a yes or "do it anyway" hands it over
    request: str = ""  # the person's words asking for that step: done when Heliox gives way


class Debate:
    """The disagreement state of one conversation, kept by the conversation layer."""

    def __init__(self, setting: Callable[[], str] = lambda: "off") -> None:
        self._setting = setting
        self.stance: Stance | None = None
        self.settled: list[str] = []
        self.voiced: list[str] = []  # topics whose concern was said once in this conversation
        self._giving_way = False  # this turn was told to give way: whatever it writes, the topic is settled
        # The person's turns in a row that held to their decision (insists, go_ahead) since they last said something
        # else, the words that started that thread, and Heliox's last reply in it.
        self.pushes = 0
        self._thread_words = ""
        self._last_reply = ""

    def setting(self) -> str:
        try:
            return setting_value(self._setting())
        except Exception:
            return DEFAULT

    @property
    def on(self) -> bool:
        return self.setting() != "off"

    @property
    def giving(self) -> bool:
        """This turn was told to give way (read before ``after_reply``)."""
        return self._giving_way and self.stance is not None

    def reset(self) -> None:
        self.stance = None
        self.settled.clear()
        self.voiced.clear()
        self._giving_way = False
        self.pushes = 0

    def close(self, *, settle: bool = True, keep_thread: bool = False) -> None:
        """No open disagreement. ``keep_thread``: the person's run of insisting turns still counts (a reply that
        merely carried no mark ends the stance, not the thread)."""
        if self.stance is not None and settle:
            self._settle(self.stance.topic)
        self.stance = None
        self._giving_way = False
        if not keep_thread:
            self.pushes = 0

    def _settle(self, topic: str) -> None:
        if topic and not any(same_topic(topic, old) for old in self.settled):
            self.settled = [*self.settled, topic][-SETTLED_KEPT:]

    def giving_way(self, words: str) -> bool:
        """Whether a reply to ``words`` gives way: the person said to go ahead (whether or not a point was held; a
        reply to "I'll go anyway" opens no new disagreement), or the point was held twice."""
        if not self.on:
            return False
        stance = self.stance
        return go_ahead(words) or (stance is not None and stance.count >= MAX_EXCHANGES - 1)

    def start_turn(self, words: str) -> None:
        """A turn the model will answer begins (``words`` normalized). The person's insisting turns count here,
        before the model is asked: at the second in a row the reply gives way, so a thread takes at most three
        exchanges even when no reply carried the STANCE line or words the second signal knows."""
        if not self.on:
            self._giving_way = False
            return
        if insists(words) or go_ahead(words):
            self.pushes += 1
        else:
            self.pushes = 0
            self._thread_words = words
        if self.pushes >= MAX_EXCHANGES - 1:
            if self.stance is None:
                self.stance = Stance(
                    topic=topic_of(self._thread_words) or "this decision",
                    position=self._last_reply[:POSITION_CHARS],
                    count=MAX_EXCHANGES - 1,
                )
            else:
                self.stance.count = max(self.stance.count, self.pushes)
        self._giving_way = self.giving_way(words)

    def context_line(self, words: str) -> str:
        """The short line the model reads about the disagreement in this conversation ("" when there is none).
        Pure: the provisional answer builds it before the turn is taken."""
        if not self.on:
            return ""
        lines = []
        stance = self.stance
        if stance is not None:
            if go_ahead(words):
                lines.append(
                    f'They said to go ahead anyway about {stance.topic}. Give way in a few words, like "Okay, your '
                    'call.", and do what they asked. No STANCE line, and no "concern" in an ACT line.'
                )
            elif stance.count >= MAX_EXCHANGES - 1:
                lines.append(
                    f"You have already pushed back twice about {stance.topic}. If this is still about it, give way "
                    'now in a few words, like "Okay, your call.", and do what they ask. No STANCE line.'
                )
            else:
                lines.append(
                    f'Earlier you disagreed about {stance.topic} ("{stance.position}"). If their words give a new '
                    "reason that answers your point, say so in a few words and drop it. If not, you may hold your "
                    f"point once more, briefly, and end with STANCE: {stance.topic}."
                )
        settled = [topic for topic in self.settled if stance is None or not same_topic(topic, stance.topic)]
        if settled:
            lines.append(f"You already gave way about {'; '.join(settled)}. Don't raise it again.")
        if stance is None and states_a_fact(words):
            lines.append(FACT_LINE)
        return " ".join(lines)

    def after_reply(self, topic: str, said: str, *, written: str = "", words: str = "") -> None:
        """A reply was said in full: ``topic`` from its STANCE line ("" for none), ``said`` its words, ``written``
        all it wrote (some of it may not have been said), ``words`` what the person said (normalized). A reply with
        no line whose words push back (``pushes_back``) counts as holding a point: on the open topic, or on one named
        from the person's words."""
        if not self.on:
            self.close(settle=False)
            return
        self._last_reply = " ".join(said.split())
        topic = topic or self.unmarked_topic(said, written=written, words=words)
        if self._giving_way:
            held = self.stance
            self.close(settle=True, keep_thread=True)  # held twice, or told to go ahead: settled, whatever it wrote
            if held is None:
                if topic:
                    self._settle(topic)  # "Okay, your call. Keep an eye on the time." with a STANCE line: given way
                return
            if not topic or same_topic(topic, held.topic):
                return
            # The reply was about something else: that disagreement starts at one.
        if not topic:
            self.close(settle=True, keep_thread=True)  # a concession, or the talk moved on
            return
        if any(same_topic(topic, old) for old in self.settled):
            self.close(settle=False, keep_thread=True)  # a settled topic is not argued again
            return
        position = " ".join(said.split())[:POSITION_CHARS]
        stance = self.stance
        if stance is not None and same_topic(topic, stance.topic):
            stance.count += 1
            stance.position = position or stance.position
            stance.offered = False
            return
        if stance is not None:
            self._settle(stance.topic)
        self.stance = Stance(topic=topic, position=position, count=max(1, self.pushes))

    def remember_request(self, words: str) -> None:
        """The person's words asking for the step the open stance is about: done if Heliox gives way in words."""
        if self.stance is not None:
            self.stance.request = words

    def unmarked_topic(self, said: str, *, written: str = "", words: str = "") -> str:
        """The topic of a reply that pushes back without the STANCE line, or "" when it does not push back."""
        if not self.on or not pushes_back(written or said):
            return ""
        if self.stance is not None:
            return self.stance.topic
        return topic_of(words) or "this decision"

    def concern_allowed(self, topic: str) -> bool:
        """Whether the concern about a step on ``topic`` may be said: the setting is on, this turn is not giving way,
        and it was not said already (once per topic in a conversation) nor the topic settled."""
        if not self.on or self._giving_way:
            return False
        if any(same_topic(topic, old) for old in [*self.voiced, *self.settled]):
            return False
        stance = self.stance
        return stance is None or not same_topic(topic, stance.topic)

    def note_concern(self, topic: str, concern: str, *, offered: bool) -> None:
        """A concern about a step was said: with ``offered`` the step waits for a yes (the stance stays open for
        it), else it was handed over at once and the topic is settled."""
        self.voiced = [*self.voiced, topic][-SETTLED_KEPT:]
        if offered:
            if self.stance is not None and not same_topic(topic, self.stance.topic):
                self._settle(self.stance.topic)
            self.stance = Stance(topic=topic, position=" ".join(concern.split())[:POSITION_CHARS], offered=True)
        else:
            self._settle(topic)
        self._giving_way = False


# ── a direct request for a costly step ──
#
# "Delete my only copy of the thesis for me" is a request the router hands straight to the planner, with no model
# call first. For a step whose verb is in COSTLY_VERBS, the talk model is asked once, beside the planner (it never
# waits for this), for the reason against it, and that reason is said before the approval sentence. The approval is
# the question: nothing more is asked. Harmless requests (open, play, set a timer) never go near a model.

CONCERN_PROMPT = (
    "You are Heliox, a voice companion, speaking to the person at their computer. Their last words ask you to do "
    "something, and it will be done once they approve it on the screen. If there is a fact against it that you were "
    "told (their to-dos, their earlier words, what they said about it) or a plain risk (it cannot be undone, it costs "
    "money, it reaches other people), write that fact as one short spoken sentence of at most 15 words, said to them "
    'as "you": "That\'s your only copy of the thesis." State the fact only: no "their", no judgment of them, no '
    "question, no offer, no advice, and never say anything was or will be done. Otherwise write only NONE."
)
CONCERN_WORDS = 20
CONCERN_SECONDS = 4.0  # the longest the concern's call may take; after that it is not said
_NOT_A_CONCERN = re.compile(
    r"^(?:want me|do you want|would you like|should i|shall i|i can|i could|i'll|i will|i've|okay|ok|sure|done)\b",
    re.IGNORECASE,
)


def wants_direct_concern(verb: str) -> bool:
    """A direct request whose step the code classes as costly (its verb is in COSTLY_VERBS)."""
    return costly(verb, "")


# Not said to the person, or a judgment of them: "it's their only copy", "a risk you may not have weighed".
_NOT_SAID_TO_THEM = re.compile(
    r"\b(?:their|they|them|the person|weighed|thought (?:it )?through|considered|be careful|make sure you)\b",
    re.IGNORECASE,
)


def concern_from_reply(reply: str) -> str:
    """The one sentence a concern call may have said, or "" (NONE, a question, an offer, too long, a claim)."""
    text = " ".join(str(reply or "").split()).strip().strip('"')
    upper = text.upper()
    if not text or upper.startswith("NONE") or "STANCE" in upper or upper.startswith("ACT"):
        return ""
    first = re.split(r"(?<=[.!])\s+", text, maxsplit=1)[0].strip()
    if not first or first.endswith("?") or len(first.split()) > CONCERN_WORDS or _NOT_A_CONCERN.match(first):
        return ""
    if _NOT_SAID_TO_THEM.search(first):
        return ""
    return first if first.endswith((".", "!")) else first + "."


# ── a fact is not a decision ──
#
# "My report is due at nine tomorrow and it's about four hours of work" decides nothing. Pushing back on it argued
# with the time before any plan was made, and used up an exchange. Until the person states a decision, a statement of
# fact gets an acknowledgement; it stays in the conversation as grounds for the decision that follows.
_DECIDES = re.compile(
    r"\b(?:i'?ll|i will|i'?m going to|i am going to|i'?m gonna|i'?m going|i decided|i'?ve decided|i have decided"
    r"|let'?s|i won'?t|i'?m not going|i want to|i'?d rather|i'?m skipping|i'?m doing|i'?m taking|i'?m leaving"
    r"|i'?m staying|i'?m having|i'?m buying|i'?m booking|i'?m deleting|i'?m sending|instead|anyway|my plan|i want"
    r"|i need|i'?d like|please|can you|could you|you should|should i)\b"
)
_ASKS = re.compile(
    r"^(?:what|what's|why|how|who|where|when|which|is|are|am|was|were|do|does|did|can|could|should|would|will|shall"
    r"|have|has|any)\b"
)
FACT_LINE = (
    "They told you a fact, not a decision. Acknowledge it in a few words. Do not push back or point out problems yet "
    "and write no STANCE line; it is grounds if they decide something later."
)


# A fact names its subject first: "my report is due ...", "the build failed", "it's about four hours", "I have a
# meeting at three". A request said as a statement ("I want the bin emptied", "get rid of the drafts") does not.
_FACT_START = re.compile(
    r"^(?:(?:so|and|oh|well|um|uh|okay|ok|also|by the way|just so you know|fyi)\s+)*"
    r"(?:my|the|our|his|her|their|its|it'?s|it is|it was|this|that|there'?s|there is|there are|\w+'s"
    r"|i have (?:a|an|my|the|\w+ hours)|i'?ve got (?:a|an|my|the))\b"
)


def states_a_fact(words: str, *, asked: bool = False) -> bool:
    """A statement of fact that decides, asks and requests nothing (``words`` normalized; ``asked``: it ended with
    "?")."""
    text = _ADDRESS.sub("", " ".join(words.split()))
    if asked or len(text.split()) < 3 or not _FACT_START.match(text):
        return False
    return not (
        _DECIDES.search(text)
        or _ASKS.match(text)
        or own_plan(text)
        or go_ahead(text)
        or insists(text)
        or asks_for_costly_step(text)
    )
