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
    "when_it_matters": "Speak up only when it matters: the clash or the risk is real and they may not have seen it. "
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
    r"(?:it'?s )?my (?:call|decision|choice)|i(?:'ve| have) decided|i still want (?:to|it)|do it anyway)"
    r"(?: anyway| please| thanks)*$"
)


def go_ahead(words: str) -> bool:
    """The person keeps their decision after a disagreement (``words`` already normalized: lower case, no
    punctuation)."""
    return bool(_GO_AHEAD.match(" ".join(words.split())))


def costly(verb: str, risk: str) -> bool:
    """A step whose concern is said before it is done: the model marked a risk, or the verb is one of COSTLY_VERBS."""
    return risk in RISKS or verb.casefold().strip() in COSTLY_VERBS


@dataclass
class Stance:
    topic: str
    position: str  # what Heliox said when it disagreed
    count: int = 1  # replies that held the point
    offered: bool = False  # a costly step asked about once: a yes or "do it anyway" hands it over


class Debate:
    """The disagreement state of one conversation, kept by the conversation layer."""

    def __init__(self, setting: Callable[[], str] = lambda: "off") -> None:
        self._setting = setting
        self.stance: Stance | None = None
        self.settled: list[str] = []
        self.voiced: list[str] = []  # topics whose concern was said once in this conversation
        self._giving_way = False  # this turn was told to give way: whatever it writes, the topic is settled

    def setting(self) -> str:
        try:
            return setting_value(self._setting())
        except Exception:
            return DEFAULT

    @property
    def on(self) -> bool:
        return self.setting() != "off"

    def reset(self) -> None:
        self.stance = None
        self.settled.clear()
        self.voiced.clear()
        self._giving_way = False

    def close(self, *, settle: bool = True) -> None:
        if self.stance is not None and settle:
            self._settle(self.stance.topic)
        self.stance = None
        self._giving_way = False

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
        """A turn the model will answer begins (``words`` normalized)."""
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
        return " ".join(lines)

    def after_reply(self, topic: str, said: str) -> None:
        """A reply was said in full: ``topic`` from its STANCE line ("" for none), ``said`` its words."""
        if not self.on:
            self.close(settle=False)
            return
        if self._giving_way:
            held = self.stance
            self.close(settle=True)  # held twice, or told to go ahead: settled whatever the reply wrote
            if held is None:
                if topic:
                    self._settle(topic)  # "Okay, your call. Keep an eye on the time." with a STANCE line: given way
                return
            if not topic or same_topic(topic, held.topic):
                return
            # The reply was about something else: that disagreement starts at one.
        if not topic:
            self.close(settle=True)  # a concession, or the talk moved on
            return
        if any(same_topic(topic, old) for old in self.settled):
            self.close(settle=False)  # a settled topic is not argued again
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
        self.stance = Stance(topic=topic, position=position)

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
