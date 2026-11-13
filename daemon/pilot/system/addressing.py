"""When Heliox takes speech as meant for it. One rule, pure, with an injected clock; the listener asks it about every
utterance. The rule in words is in docs/INTERACTION_MODEL.md ("When Heliox takes speech as meant for it").

Outside an open conversation only two things open one: the wake phrase in the vocative position at the very start
of an utterance, or the microphone button (or hotkey). Nothing is taken as addressed because it sounds like a
command.

Inside an open conversation (Heliox has just answered someone who addressed it, or a task they started is running)
a follow-up needs no wake phrase only if it starts within FOLLOW_UP_SECONDS of Heliox's last word, did not begin
while Heliox was still talking, and is plausibly meant for Heliox: a control word, a reply to the question Heliox
just asked, or a direct request or question (imperative or second person). Third-person talk, narration and long
monologues do not count. A follow-up that might be for Heliox but is not clearly so gets one short "Sorry?" and does
not keep the conversation open. Only a confidently addressed utterance keeps it open; CLOSE_SECONDS without one
closes it, and a call closes it at once.

Language: the wake phrase and the button work in any language. The words of a follow-up are read in English
(imperatives, question starts, narration), so in another language a follow-up is more often "Sorry?" or nothing.
One part needs no words: a short reply that starts within QUICK_REPLY_SECONDS of Heliox's own question, and did
not begin over Heliox's voice, is a reply in any language ("Ja, die blaue.", "हाँ, नीली वाली", "はい、青いほう").
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from pilot.system import voice_words

FOLLOW_UP_SECONDS = 8.0  # people answer within about a second; 8 s leaves room to think, 30 s let a room in
CLOSE_SECONDS = 20.0  # no confidently addressed utterance for this long: the wake phrase is needed again
MONOLOGUE_SECONDS = 8.0  # a follow-up with more voice than this is talk, not a reply
CUT_OFF_SECONDS = 1.6  # a word or two said over Heliox's voice is "stop", whatever it reads as (voice.is_cut_off_stop)
QUICK_REPLY_SECONDS = 2.0  # an answer to Heliox's question starts within this, in any language
QUICK_REPLY_VOICE_SECONDS = 4.0  # and is short: an answer, not a story
# "Thank you." is also what Whisper writes for silence and noise, so it counts as a reply only with this much voice
# in it, and only inside the follow-up window after Heliox spoke. Measured 13 November through the listener's own
# recorder: Whisper small wrote "Thank you." for clicks and room noise with 0.05 to 0.2 s of voice; Kokoro's
# "Thank you." (four voices, three speeds) had 0.40 to 0.70 s, and "Thanks." 0.35 to 0.55 s. Synthetic voices.
THANKS_VOICE_SECONDS = 0.3
THANKS = re.compile(r"^(?:thank you(?: so much| very much)?|thanks(?: a lot| so much)?)$", re.I)
LIVE_REPLY_SECONDS = 6.0  # after a spoken Live remark: its own replies ("why?", "not now") need no wake phrase
ELLIPTICAL_SECONDS = 4.0  # "and tomorrow?", "again", "shorter": said right after Heliox's answer, not later
ELLIPTICAL_WORDS = 6  # "what about the second one?" is short; a sentence starting with "and" is narration
EXACT_NAME = re.compile(r"^heliox(?:'s)?$", re.I)
LOOSE_NAME = re.compile(r"^h[ei]l{1,2}[iy][a-z]{0,5}$", re.I)  # "Heliocs", "Helios", "Hilliox": recogniser spellings
LEADS = {"hey", "hay", "hi", "hei", "okay", "ok", "oh", "here", "dear", "hello", "a", "eh", "ay"} | set(
    voice_words.GREETING_WORDS
)  # and "Oye Heliox", "Hallo Heliox": the greetings of other languages before the name
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# After the name: words that make it third-person talk about Heliox, not words said to it.
THIRD_PERSON = {
    "is", "was", "has", "had", "does", "did", "will", "would", "should", "can", "could", "might", "may",
    "says", "said", "thinks", "knows", "seems", "looks", "sounds", "keeps", "gets", "got", "uses", "used",
    "and", "or", "thing", "app", "itself", "isn't", "wasn't", "doesn't", "didn't", "won't", "can't",
}  # fmt: skip
SECOND_PERSON = re.compile(r"\b(?:you|your|yours|yourself)\b", re.I)
QUESTION_START = re.compile(
    r"^(?:what|who|where|when|why|how|which|whose|is|are|was|were|do|does|did|can|could|would|will|should|shall|"
    r"may|have|has)\b",
    re.I,
)
IMPERATIVE_START = re.compile(
    r"^(?:open|close|show|find|search|look|tell|explain|summari[sz]e|read|write|create|launch|start|stop|cancel|"
    r"continue|go|play|pause|send|type|scroll|switch|check|remind|schedule|turn|set|run|use|save|delete|remove|"
    r"move|copy|paste|undo|redo|add|make|give|put|take|call|text|email|book|note|list|repeat|say|help|try|keep|"
    r"skip|wait|hold|mute|unmute|raise|lower|increase|decrease|translate|draft|reply|answer|let's|please|"
    r"click|press|choose|select|navigate|visit|browse|inspect|research|execute|approve|deny|download|upload|"
    r"install|uninstall|zip|rename|share|print|record|pin|archive|minimi[sz]e|maximi[sz]e|refresh|reload)\b",
    re.I,
)
CONTROL = re.compile(
    r"^(?:stop|wait|hold on|no|nope|cancel|never ?mind|undo|skip|don'?t|do not|not (?:that|now)|actually|instead|"
    r"go ahead|continue|that'?s (?:it|enough|all)|enough)\b",
    re.I,
)
# Acknowledgements are replies only as a whole short utterance: "Okay, thanks" is to Heliox, "okay so anyway we went
# to lunch" is not.
# Asking for the rest of a reply that was cut short ("Sorry, what were you saying?", "go on"): a follow-up only
# within RESUME_SECONDS of the cut, when the conversation layer still holds the unsaid rest (Addressing.reply_cut_at),
# so these words do not widen the follow-up rule otherwise.
RESUME = re.compile(  # the same phrases the conversation layer answers from code (its WHAT_WERE_YOU_SAYING)
    r"^(?:sorry\s+)?(?:what were you saying|you were saying|go on|carry on|keep going|continue|"
    r"finish what you were saying|what were you going to say|go ahead finish)$",
    re.I,
)
RESUME_SECONDS = 120.0
# The person telling Heliox about themselves in an open follow-up window ("I start work at nine." after "Call me
# Sam."): taken when the personal layer recognises one of the forms it keeps (Addressing.about_me, injected). Each
# is said back, which restarts the window, so a few in a row need no wake phrase; after ABOUT_ME_IN_ROW the wake
# phrase is needed again. Three covers telling a name, hours and one more thing in one go (the Live day), and keeps a
# room's run of "I ..." sentences from being swept in one after another.
ABOUT_ME_IN_ROW = 3
# A reply to an addressed turn comes within this: when its voice ends, the conversation's quiet is counted from
# then, even if the reply (thinking, then a long answer) took longer than CLOSE_SECONDS.
REPLY_OWED_SECONDS = 60.0
SHORT_REPLY = re.compile(r"^(?:yes|yeah|yep|sure|okay|ok|thanks|thank you|great|perfect|got it|sorry)\b", re.I)
# Follow-ups on Heliox's last answer that read as talk out of context. The repeat and shorter phrases are the ones
# the conversation layer answers in code (pilot/conversation/followups.py, AGAIN and SHORTER; copied, not imported:
# importing pilot.conversation loads the whole layer; its WORDS_ONLY, EITHER and TASK_AGAIN together are AGAIN here).
AGAIN = re.compile(
    r"^(?:please )?(?:say (?:that|it) again|repeat (?:that|it)|repeat|again|once more|one more time|"
    r"what did you say|what was that|come again|pardon|sorry what|do (?:that|it) again|run (?:that|it) again|"
    r"repeat (?:the|my) last task)(?: please)?$"
)
SHORTER = re.compile(
    r"^(?:please )?(?:shorter|make it shorter|say it shorter|in short|briefly|short version|the short version|"
    r"just the short version|tl ?dr|give me the short version|the gist|just the gist)(?: please)?$"
)
CONTINUATION = re.compile(r"^(?:and|or|what about|how about|what if)\b")


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", text.casefold()).replace("'", " ").split())


def elliptical_kind(text: str) -> str:
    """ "again" or "shorter" (the conversation layer answers these in code), "continuation" for a short "and
    tomorrow?" / "what about the second one?", or "" when it is neither."""
    said = _normalize(text)
    if AGAIN.match(said):
        return "again"
    if SHORTER.match(said):
        return "shorter"
    if CONTINUATION.match(said) and len(said.split()) <= ELLIPTICAL_WORDS:
        return "continuation"
    return ""


NARRATION = re.compile(r"^(?:i|we|he|she|they|it|this|that|there|so|and|but|then|now|first|next|well)\b", re.I)


def _words(text: str) -> list[re.Match[str]]:
    return list(WORD.finditer(text))


@dataclass(frozen=True)
class Wake:
    """A wake phrase in the vocative position: ``command`` is what follows it ("" for the phrase alone)."""

    prefix: str
    command: str
    exact: bool


def vocative_wake(text: str, other_text: str = "") -> Wake | None:
    """The wake phrase at the very start of ``text``, said to Heliox, or None.

    Counts: "Hey Heliox, open notes", "Heliox, what time is it?", "Hey Heliox." Does not count: "Heliox is the app I
    use", "Heliox's settings", "that Heliox thing" (not at the start), "Helix" or "Hello Alex" (not the name). A
    recogniser spelling of the name ("Heliocs", "Helios") counts only when ``other_text``, the second recogniser's
    words for the same sound, starts the same way after the name; the exact name needs no second opinion.
    """
    words = _words(text[:80])
    if not words:
        return None
    index = 0
    if words[0].group(0).casefold() in LEADS and len(words) > 1:
        index = 1
    name = words[index].group(0)
    exact = bool(EXACT_NAME.match(name))
    if not exact and not LOOSE_NAME.match(name):
        return None
    if name.casefold().endswith("'s"):
        return None  # "Heliox's settings": a possessive, not a call
    if index == 0 and not exact and len(name) <= 5:
        return None  # a bare short word ("Helix", "Hello") is not the name without a "Hey" before it
    end = words[index].end()
    rest = text[end:]
    after = _words(rest)
    pause = bool(re.match(r"^\s*[,.!?;:–-](?:\s|$)", rest))  # "Heliox.com" is not a pause
    command = rest.strip(" \t,.:;!?-–")
    if after:
        first = after[0].group(0).casefold()
        if first in THIRD_PERSON and not SECOND_PERSON.search(rest) and not rest.rstrip().endswith("?"):
            # "Heliox is ...", "Heliox can do that", and with a pause the recogniser put in, "Heliox, is a nice
            # name for a project" (the bench's synthetic meeting): talk about Heliox. "Heliox, is it raining?" and
            # "Heliox, can you ..." are questions to it.
            return None
        if index == 0 and not pause and not (IMPERATIVE_START.match(command) or QUESTION_START.match(command)):
            return None  # bare name then ordinary words: "Heliox thing", "Helios mission"
    if not exact and not _agrees(command, other_text):
        return None
    return Wake(text[:end], command, exact)


def _agrees(command: str, other_text: str) -> bool:
    """The second recogniser heard the same start after the name: its first five words hold the first word here."""
    if not command:
        return False
    first = _words(command)[0].group(0).casefold().split("'")[0]
    return first in [m.group(0).casefold().split("'")[0] for m in _words(other_text)[:6]]


def plausibly_addressed(text: str, *, question_open: bool, elliptical: bool = True, language: str = "en") -> str:
    """How a follow-up reads: "control", "reply", "request", "question" (confident), "unclear", or "talk".
    A stop or deny word of the person's voice language (voice_words) is "control" in any script."""
    from pilot.system import voice_words

    if voice_words.is_control_word(text, language):
        return "control"
    words = _words(text)
    if not words:
        return "talk"
    normalized = " ".join(w.group(0) for w in words)
    from pilot.system.voice import looks_hallucinated

    if looks_hallucinated(text):
        return "talk"  # "Thanks for watching", "Open, open, open, open": noise read as words
    if CONTROL.match(normalized):
        return "control"
    if len(words) <= 3 and SHORT_REPLY.match(normalized):
        return "control"
    if elliptical and elliptical_kind(text):
        return "elliptical"  # taken only right after Heliox's answer (Addressing.decide); otherwise talk
    if question_open and len(words) <= 12:
        return "reply"
    if MEMORY_COMMAND.match(SOFTENER.sub("", normalized)) and len(words) <= 12:
        return "request"  # "Forget that I'm a founder." right after Heliox spoke: about what Heliox keeps
    if NARRATION.match(normalized):
        return "talk"
    if IMPERATIVE_START.match(SOFTENER.sub("", normalized)):
        return "request"  # "Just delete all of that.", "Okay, just open the deck."
    if QUESTION_START.match(normalized) and (SECOND_PERSON.search(normalized) or len(words) <= 10):
        return "question"
    if SECOND_PERSON.search(normalized) and len(words) <= 12:
        return "unclear"  # might be said to Heliox, might not
    return "talk"


# What Heliox keeps about the person, said right after it spoke: "Forget that I'm a founder.", "Forget what I said.",
# "Don't keep that.", "Remember that." Only these shapes. "Forget it." and "Forget about it, ..." are said to other
# people too, so "forget" counts only before that, this, what, my or everything; and "Remember to save your work
# before you close the editor." is in the unaddressed corpus, so "remember" counts only as "remember that/this/it"
# alone.
MEMORY_COMMAND = re.compile(
    r"^(?:please\s+)?(?:forget\s+(?:that|this|what|my|everything)\b|"
    r"don'?t\s+(?:keep|remember|store|save)\s+(?:that|it|this)\b|remember\s+(?:that|this|it)\s*$)",
    re.I,
)

CONFIDENT = {"control", "reply", "request", "question"}

SOFTENER = re.compile(r"^(?:(?:just|okay|ok|so|right|then|now|and)\s+)+", re.I)  # "okay so just open ..."
# Heliox's own sentence names something to say: `Say "more" for others.`, "Just say yes or no."
_INVITES = re.compile(r"(?:^|[.!?]\s+)(?:just\s+|you can\s+)?say\s+[\"“'‘]?\w", re.I)


def invites_reply(heliox_said: str) -> bool:
    """Whether what Heliox just said asks for a reply: a question, or a sentence that names a word to say
    (`Say "more" for others.`). After it, a short answer in the follow-up window is a reply."""
    text = heliox_said.strip()
    return text.endswith("?") or bool(_INVITES.search(text))


@dataclass(frozen=True)
class Heard:
    """One utterance as the listener heard it (monotonic times)."""

    text: str
    started: float
    ended: float
    voiced_seconds: float = 0.0
    other_text: str = ""  # the second recogniser's words (Parakeet beside Whisper), for the agreement rule
    wake_text: str = ""  # the wake check's own words (Whisper), when the words acted on come from another recogniser
    by_button: bool = False
    heliox_was_speaking: bool = False  # the voice began while Heliox was still talking
    in_call: bool = False


@dataclass(frozen=True)
class Decision:
    act: bool  # hand the command on
    command: str = ""
    why: str = ""  # wake, button, follow-up kind, or the reason it was not taken
    say_sorry: bool = False
    hint: str = ""  # said once in the voice bar when the person clearly tried and was not taken


@dataclass
class Addressing:
    clock: Callable[[], float] = time.monotonic
    follow_up_seconds: float = FOLLOW_UP_SECONDS
    close_seconds: float = CLOSE_SECONDS
    open_since: float | None = None  # when the conversation last had a confidently addressed utterance
    heliox_ended_at: float | None = None
    question_open: bool = False
    task_open: bool = False  # a task the person started is running or an approval waits
    # When a reply was cut off with words still unsaid (the conversation layer's cut_reply_at, time.monotonic), or
    # None: "what were you saying?" is taken within RESUME_SECONDS of it (_resume).
    reply_cut_at: float | None = None
    language: str = "en"  # the person's voice language (voice_words.voice_language), for its stop and deny words
    _sorry_given: bool = False
    _at: float | None = None
    # When Heliox Live last finished a spoken remark (live_spoke), and the test for its own closed list of replies
    # (pilot.live.engine.is_live_reply, injected: importing the Live stack here would load it with the listener).
    live_spoke_at: float | None = None
    live_reply: Callable[[str], bool] | None = (
        None  # while deciding: when the utterance began (capture clock), not when it is processed
    )
    _hint_given: bool = False
    history: list[str] = field(default_factory=list)
    # How many utterances were not taken since Heliox last finished speaking (any reason, over its voice too): a
    # count only, no words. The conversation layer reads it when a follow-up is taken, as a sign that someone else,
    # a television or the person's own muttering was heard in between (its second-voice rule).
    overheard_since_heliox: int = 0
    # The personal layer's test for a sentence telling Heliox about the person (pilot.workflows.user_profile.
    # tells_about_self, injected by the listener so the patterns live in one place), and the second-voice rule's
    # threshold when it is on (0 when off): with that many utterances left out since Heliox spoke, a statement
    # about the person is not taken without the wake phrase.
    about_me: Callable[[str], bool] | None = None
    second_voice_threshold: int = 0
    _about_me_row: int = 0
    _reply_owed_since: float | None = None

    def _now(self) -> float:
        return self._at if self._at is not None else self.clock()

    @property
    def conversation_open(self) -> bool:
        if self.task_open:
            return True
        if self.open_since is None:
            return False
        if self._now() - self.open_since > self.close_seconds:
            self.close("quiet")
            return False
        return True

    def close(self, why: str) -> None:
        if self.open_since is not None:
            self.history.append(f"closed: {why}")
        self.open_since = None
        self.heliox_ended_at = None
        self.question_open = False
        self._sorry_given = False
        self.live_spoke_at = None
        self._about_me_row = 0
        self._reply_owed_since = None

    def live_spoke(self) -> None:
        """Heliox Live just finished a spoken remark: for LIVE_REPLY_SECONDS its own replies are taken without the
        wake phrase, even with no conversation open, and open nothing."""
        self.live_spoke_at = self.clock()

    def note_not_understood(self) -> None:
        """The conversation layer could not understand the last turn ("Sorry?"): the window closes."""
        self.close("not understood")

    def heliox_finished(self, *, asked: bool = False) -> None:
        """Heliox's voice stopped. Only within an open conversation does this start a follow-up window."""
        self.overheard_since_heliox = 0
        now = self.clock()
        owed = self._reply_owed_since
        if self.open_since is not None and owed is not None and 0 <= now - owed <= REPLY_OWED_SECONDS:
            # Part of the reply to what the person said has just ended (a reply spoken sentence by sentence ends
            # several times): a long one, or a long wait for it, does not use up the conversation. Its quiet runs
            # from Heliox's last word, as the follow-up window does. A "Sorry?" after speech that was not taken is
            # no such reply (decide clears the debt), so it keeps nothing open.
            self.open_since = now
        if self.conversation_open:
            self.heliox_ended_at = now
            self.question_open = asked

    def decide(self, heard: Heard) -> Decision:
        """The decision for one utterance, judged at the moment it began (``heard.started``, capture clock): a
        recognition backlog of seconds must not close the window or the conversation under it."""
        self._at = heard.started
        try:
            decision = self._decide(heard)
        finally:
            self._at = None
        if not decision.act:
            self.overheard_since_heliox += 1  # counted, never kept: nothing of what was said stays here
            self._reply_owed_since = None  # what Heliox says next answers nobody's request
        return decision

    def _decide(self, heard: Heard) -> Decision:
        if heard.in_call:
            self.close("call")
            return Decision(False, why="in a call")
        if heard.by_button:
            self._addressed()
            self._about_me_row = 0
            return Decision(True, heard.text.strip(), "button")
        if (
            heard.heliox_was_speaking
            and 0 < heard.voiced_seconds <= CUT_OFF_SECONDS
            and vocative_wake(heard.wake_text or heard.text, heard.other_text) is None
        ):
            # Heliox was talking and the person cut it off: it stops and acknowledges; the words go nowhere else.
            # "Hey Heliox" over Heliox is not a stop: its voice has already stopped for it, and the request follows.
            return Decision(True, "Stop", "cut-off")
        wake = vocative_wake(heard.wake_text or heard.text, heard.other_text)
        if wake is None and heard.wake_text and heard.text != heard.wake_text:
            # The wake check's own words can misread what follows the name ("Hey Heliox and do that." for "Hey
            # Heliox, undo that.", which reads as talk about Heliox); the words acted on, which take the name from
            # it and the rest from the second recogniser, are judged by the same rule.
            wake = vocative_wake(heard.text, heard.other_text)
        if wake is not None:
            self._addressed()
            self._about_me_row = 0
            # The command in the words acted on (Parakeet's), the wake phrase decided by the wake check's.
            spoken = vocative_wake(heard.text) if heard.wake_text else None
            return Decision(True, spoken.command if spoken is not None else wake.command, "wake")
        if self._resume(heard):
            self._addressed()
            return Decision(True, heard.text.strip(" \t,.:;!?-"), "follow-up: resume")
        if not self.conversation_open and self._live_reply(heard):
            self.live_spoke_at = None  # one reply per remark; Live speaking again opens the next
            return Decision(True, heard.text.strip(" \t,.:;!?-"), "live reply")
        if not self.conversation_open:
            return self._not_taken(heard, "no conversation open")
        kind = plausibly_addressed(heard.text, question_open=self.question_open, language=self.language)
        if kind != "control" and self._quick_reply(heard):
            kind = "reply"  # read without the words: any language
        if kind == "elliptical" and not self._soon_after_answer(heard):
            # Not right after Heliox's answer: read as any other follow-up ("What about tomorrow?" is a question).
            kind = plausibly_addressed(
                heard.text, question_open=self.question_open, elliptical=False, language=self.language
            )
        if kind in {"talk", "unclear"} and self._thanks(heard):
            kind = "control"  # "Thank you." said with voice right after Heliox spoke: an acknowledgement
        if kind not in CONFIDENT and self._tells_about_self(heard):
            kind = "about me"  # said back at once, so a wrong take is heard and undone with "no"
        if heard.heliox_was_speaking and kind != "control":
            return Decision(False, why="began while Heliox was talking")
        if not self.task_open:
            if self.heliox_ended_at is None or heard.started - self.heliox_ended_at > self.follow_up_seconds:
                return self._not_taken(heard, "too long after Heliox spoke")
        if heard.voiced_seconds > MONOLOGUE_SECONDS and kind not in {"control"}:
            return Decision(False, why="talk, not a reply")
        if kind in CONFIDENT or kind in ("elliptical", "about me"):
            self._addressed()
            if kind == "about me":
                self._about_me_row += 1
            return Decision(True, heard.text.strip(" \t,.:;!?-"), f"follow-up: {kind}")
        if kind == "unclear" and not self._sorry_given:
            self._sorry_given = True
            return Decision(False, why="unclear", say_sorry=True)
        return Decision(False, why=f"follow-up: {kind}")

    def _live_reply(self, heard: Heard) -> bool:
        """A reply to Live's remark: one of its own replies, begun after its voice ended (a sentence begun over it
        is not; a word or two over it is a cut-off, decided above) and within LIVE_REPLY_SECONDS, on the capture
        clock."""
        spoke, test = self.live_spoke_at, self.live_reply
        if spoke is None or test is None or heard.heliox_was_speaking or heard.in_call:
            return False
        if not 0.0 <= heard.started - spoke <= LIVE_REPLY_SECONDS:
            return False
        try:
            return bool(test(heard.text))
        except Exception:
            return False

    def _thanks(self, heard: Heard) -> bool:
        """ "Thank you." with real voice in it (THANKS_VOICE_SECONDS) inside the follow-up window after Heliox spoke,
        not over its voice: the hallucination guard keeps "Thank you." from silence and noise out."""
        text = " ".join(w.group(0) for w in _words(heard.text))
        if not THANKS.match(text) or heard.heliox_was_speaking or self.heliox_ended_at is None:
            return False
        if not 0 <= heard.started - self.heliox_ended_at <= self.follow_up_seconds:
            return False
        return heard.voiced_seconds >= THANKS_VOICE_SECONDS

    def _tells_about_self(self, heard: Heard) -> bool:
        """A sentence telling Heliox about the person, in a form the personal layer keeps: not over Heliox's voice,
        not more than ABOUT_ME_IN_ROW in a row without the wake phrase, and not when the second-voice rule is on and
        the room was heard talking since Heliox spoke (it is data about the person)."""
        test = self.about_me
        if test is None or heard.heliox_was_speaking or self._about_me_row >= ABOUT_ME_IN_ROW:
            return False
        if self.second_voice_threshold and self.overheard_since_heliox >= self.second_voice_threshold:
            return False
        try:
            return bool(test(heard.text))
        except Exception:
            return False

    def _resume(self, heard: Heard) -> bool:
        """ "What were you saying?", "go on": asked within RESUME_SECONDS of a reply cut short, not over Heliox's voice."""
        cut = self.reply_cut_at
        if cut is None or heard.heliox_was_speaking or not 0 <= heard.started - cut <= RESUME_SECONDS:
            return False
        return bool(RESUME.match(" ".join(w.group(0) for w in _words(heard.text))))

    def _quick_reply(self, heard: Heard) -> bool:
        """A short answer that starts right after Heliox's own question and not over its voice."""
        if not self.question_open or heard.heliox_was_speaking or self.heliox_ended_at is None:
            return False
        if not 0 <= heard.started - self.heliox_ended_at <= QUICK_REPLY_SECONDS:
            return False
        text = heard.text.strip()
        if not text:
            return False
        from pilot.system.voice import looks_hallucinated

        if looks_hallucinated(text):
            # Noise, or Heliox's own voice through open speakers, right after its question is when Whisper writes
            # "Thanks for watching." or "Open, open, open": never a reply.
            return False
        if heard.voiced_seconds:
            return heard.voiced_seconds <= QUICK_REPLY_VOICE_SECONDS
        return len(text.split()) <= 12 and len(text) <= 80  # no voice measure: judge by length (no spaces in some)

    def _soon_after_answer(self, heard: Heard) -> bool:
        """An elliptical follow-up counts only right after Heliox's own answer and not over its voice."""
        if heard.heliox_was_speaking or self.heliox_ended_at is None:
            return False
        return 0 <= heard.started - self.heliox_ended_at <= ELLIPTICAL_SECONDS

    def _addressed(self) -> None:
        self.open_since = self._now()
        self._reply_owed_since = self.open_since
        self._sorry_given = False
        self._hint_given = False

    def _not_taken(self, heard: Heard, why: str) -> Decision:
        """Said once, only when the person clearly tried: a direct request or question in the second person."""
        kind = plausibly_addressed(heard.text, question_open=False)
        tried = kind in {"request", "question"} and bool(SECOND_PERSON.search(heard.text)) and not heard.in_call
        hint = ""
        if tried and not self._hint_given:
            self._hint_given = True
            hint = "Say “Hey Heliox” first."
        return Decision(False, why=why, hint=hint)


def hand_about_me(rule: Addressing, config: object) -> None:
    """Give the rule the personal layer's own test for a statement about the person, and the second-voice rule's
    threshold from the settings (0 when the rule is off). The listener and the session runner both call this."""
    if rule.about_me is None:
        try:
            from pilot.workflows.user_profile import tells_about_self

            rule.about_me = tells_about_self
        except Exception:  # pragma: no cover - the profile module is part of the daemon
            rule.about_me = None
    conversation = getattr(config, "conversation", None)
    try:
        on = bool(getattr(conversation, "second_voice_guard", False))
        rule.second_voice_threshold = max(1, int(getattr(conversation, "second_voice_threshold", 1))) if on else 0
    except (TypeError, ValueError):
        rule.second_voice_threshold = 0


# ── the Conversation Bench's transcript replay (benchmarks/conversation_bench/replay.py) ──

_REPLAY: dict[str, tuple[list[float], Addressing]] = {}


def decide(utterance: dict, context: dict) -> dict[str, str]:
    """The bench's replay entry point: one utterance of a stream, in order, with the bench's context.

    Keeps one Addressing per stream, on the stream's own clock (the utterance times), and feeds it what the
    listener would know: when Heliox's reply ended, whether Heliox was talking when the voice began, both
    recognisers' words. Returns {"act": "dispatch" | "sorry" | "overheard" | "ignore", "why": ...}.
    """
    from pilot.system.parakeet_asr import merge_wake
    from pilot.system.voice import loose_wake_prefix

    stream = str(utterance.get("stream", ""))
    if stream not in _REPLAY or not context.get("history"):
        now = [0.0]
        _REPLAY[stream] = (now, Addressing(clock=lambda: now[0]))
        hand_about_me(_REPLAY[stream][1], None)
    now, rule = _REPLAY[stream]
    start = float(utterance.get("t_start", 0.0))
    since = context.get("since_heliox_ended_s")
    if since is not None and rule.heliox_ended_at is None:
        now[0] = start - float(since)
        rule.heliox_finished()
    now[0] = start
    whisper = str(utterance.get("whisper") or "")
    parakeet = str(utterance.get("parakeet") or "")
    words = merge_wake(whisper, parakeet, loose_wake_prefix) if (whisper and parakeet) else (parakeet or whisper)
    heard = Heard(
        text=words,
        started=start,
        ended=float(utterance.get("t_end", start)),
        voiced_seconds=float(utterance.get("voiced_s") or 0.0),
        other_text=parakeet if whisper else "",
        wake_text=whisper,
        heliox_was_speaking=bool(utterance.get("heliox_speaking_at_start") or context.get("heliox_speaking")),
    )
    was_open = rule.conversation_open
    decision = rule.decide(heard)
    if decision.act:
        rule.heliox_ended_at = None  # the reply that follows sets it again
        return {"act": "dispatch", "why": decision.why}
    if decision.say_sorry:
        rule.note_not_understood()
        return {"act": "sorry", "why": decision.why}
    return {"act": "overheard" if was_open else "ignore", "why": decision.why}
