"""The first sound of a model answer: the first clause goes to the voice before its sentence is complete, and the
holding phrase is said only when the model is slow (13 November).

The splitter is tried on the first sentences Claude Haiku actually wrote in the paid conversation runs (12 and 13
November) and on texts made to break it: lists, "No, ..." openers, times, decimals, thousands, dates, abbreviations,
quotations, a URL, Spanish questions, Japanese and Hindi."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from pilot.conversation import layer as layer_module
from pilot.conversation.layer import HOLD_PHRASES, ConversationLayer
from pilot.conversation.style import SentenceSplitter, early_clause


def pieces(text: str, *, step: int = 1) -> list[tuple[str, bool]]:
    """The pieces the splitter gives for ``text`` streamed ``step`` characters at a time: (text, ends)."""
    splitter = SentenceSplitter(early=True)
    out = []
    for at in range(0, len(text), step):
        out += splitter.feed(text[at : at + step])
    out += splitter.flush()
    return [(str(piece), piece.ends) for piece in out]


# ── what Claude Haiku wrote (paid runs, 12 and 13 November): the first piece the voice now gets ──


@pytest.mark.parametrize(
    ("reply", "first"),
    [
        (
            "It's nine o'clock on Thursday, the twelfth of November. Anything else?",
            "It's nine o'clock on Thursday,",
        ),
        (
            "A loud fan usually means the processor or graphics card is working hard, so check Task Manager.",
            "A loud fan usually means the processor or graphics card is working hard,",
        ),
        (
            "Sunlight scatters off air molecules, and blue light scatters the most.",
            "Sunlight scatters off air molecules,",
        ),
        (
            "Keep it easy this week, Sam: a few short runs and one long run on the weekend.",
            "Keep it easy this week, Sam:",
        ),
        (
            "I can't see your hardware from here, so I can't tell for sure.",
            "I can't see your hardware from here,",
        ),
        (
            "Nothing is on your calendar that I can see, so it's up to you.",
            "Nothing is on your calendar that I can see,",
        ),
        # Too short to cut, or no pause: whole, as before.
        ("Canberra, the capital of Australia, not Sydney.", "Canberra, the capital of Australia, not Sydney."),
        ("Probably a busy program or dust in the vents.", "Probably a busy program or dust in the vents."),
        ("Today is Thursday, the twelfth of November.", "Today is Thursday, the twelfth of November."),
        ("Do you want me to open Notepad?", "Do you want me to open Notepad?"),
        # A list stays whole until its end.
        (
            "Biscuit, Juniper, or Pepper would all suit a dog well.",
            "Biscuit, Juniper, or Pepper would all suit a dog well.",
        ),
        (
            "Open Task Manager with Control, Shift and Escape, then sort by CPU.",
            "Open Task Manager with Control, Shift and Escape,",
        ),
    ],
)
def test_the_first_piece_of_replies_the_model_wrote(reply, first):
    assert pieces(reply)[0][0] == first
    assert pieces(reply, step=7)[0][0] == first  # streamed in other sized pieces, the same cut


# ── made to break it ──


@pytest.mark.parametrize(
    ("text", "first"),
    [
        ("No, that's wrong, the capital is Canberra.", "No, that's wrong, the capital is Canberra."),
        ("The meeting is at 5:14 tomorrow, so leave by five.", "The meeting is at 5:14 tomorrow,"),
        ("It costs 3.5 dollars, or 1,000 yen in Tokyo today.", "It costs 3.5 dollars,"),
        ("On June 5, 2026 the office closes early for the holiday.", None),
        ("Your rooms are 4, 5 and 6 on the second floor today.", None),
        ("Use a short name, e.g., Rex, for a small dog like yours.", "Use a short name, e.g., Rex,"),
        ('He said "wait, not yet, please" and left the room quietly.', None),
        (
            "Visit https://example.com/a,b for details, then come back here later.",
            "Visit https://example.com/a,b for details,",
        ),
        ("Buy milk, eggs, bread and butter on the way home tonight.", None),
        ("Call Priya, Dana and Omar before the meeting this afternoon.", None),
        ("¿Quieres que abra el Bloc de notas, o prefieres que espere un momento?", None),
        ("Je pense que oui, mais il faut vérifier les fichiers d'abord.", "Je pense que oui,"),
        (
            "Ich glaube schon, dass die Datei noch geöffnet ist, aber prüfe es bitte.",
            "Ich glaube schon, dass die Datei noch geöffnet ist,",
        ),
        (
            "It's 12:30 now; the train leaves at 13:05 from platform two.",
            "It's 12:30 now; the train leaves at 13:05 from platform two.",
        ),
    ],
)
def test_no_cut_where_a_person_would_not_pause(text, first):
    got = pieces(text)[0][0]
    assert got == (first if first is not None else text)


def test_japanese_and_hindi_sentences_end_without_a_space():
    """Before 13 November a reply in these languages was not cut until 240 characters or its end."""
    assert pieces("今日は木曜日です、十一月十二日ですね。ほかに何かありますか。") == [
        ("今日は木曜日です、", False),
        ("十一月十二日ですね。", True),
        ("ほかに何かありますか。", True),
    ]
    assert pieces("आज गुरुवार है। और कुछ?") == [("आज गुरुवार है।", True), ("और कुछ?", True)]


def test_a_cut_needs_words_after_it_so_a_short_sentence_is_never_split():
    assert early_clause("It's nine o'clock on Thursday, the") == 0
    assert early_clause("It's nine o'clock on Thursday, the twelfth of") > 0


def test_the_early_clause_is_off_by_default():
    text = "It's nine o'clock on Thursday, the twelfth of November. "
    assert [str(p) for p in SentenceSplitter().feed(text)] == [text.strip()]


def test_only_the_reply_s_first_sentence_is_cut_early():
    got = pieces("Fan noise is normal. A busy program is the usual reason, so check Task Manager.")
    assert got == [
        ("Fan noise is normal.", True),
        ("A busy program is the usual reason, so check Task Manager.", True),
    ]


# ── in the layer: a cut sentence is said whole, a statement's one sentence included ──


class Model:
    name, provider, streams, local = "stub", "claude", True, False

    def __init__(self, pieces: list, delay: float = 0.0, *, by_word: bool = False) -> None:
        self.delay, self.calls = delay, []
        # By word, as a model streams: the splitter sees the words after a pause arrive one by one.
        self.pieces = [w for p in pieces for w in (p.split(" ") if by_word and isinstance(p, str) else [p])]
        if by_word:
            self.pieces = [w if i == 0 else " " + w for i, w in enumerate(self.pieces) if w != ""]

    async def stream(self, system: str, messages: list) -> AsyncIterator[str]:
        await asyncio.sleep(self.delay)
        for piece in self.pieces:
            if isinstance(piece, float):
                await asyncio.sleep(piece)
                continue
            yield piece


class Ear:
    def __init__(self) -> None:
        self.heard: list[tuple[str, str]] = []

    async def __call__(self, text: str, channel: str, key: str) -> str:
        self.heard.append((text, channel))
        return "spoken"


LONG = "A long run on the weekend is a great way to build your distance slowly and safely, then rest."
LONG_HALF = "A long run on the weekend is a great way to build your distance slowly and safely,"


async def test_a_statement_gets_its_whole_first_sentence_when_it_was_cut_at_a_comma():
    """A first sentence over FIRST_CLAUSE_MIN characters is cut at a comma: before 13 November a statement turn (one
    sentence) said only its first half, and the voice stopped mid-sentence."""
    ear = Ear()
    model = Model([LONG + " Stretch afterwards too."], by_word=True)
    layer = ConversationLayer(sink=ear, model=lambda: model, hold_after=60.0)
    await layer.take("I'm going for a long run on the weekend.")  # a statement: one sentence
    assert [text for text, _ch in ear.heard] == [LONG_HALF, "then rest."]


async def test_the_word_cap_never_stops_a_sentence_after_its_first_clause(monkeypatch):
    ear = Ear()
    monkeypatch.setattr(layer_module, "SPOKEN_WORD_CAP", 8)
    model = Model([LONG + " More."], by_word=True)
    layer = ConversationLayer(sink=ear, model=lambda: model, hold_after=60.0)
    layer._word_cap = lambda: 8
    await layer.take("How should I train this week?")
    assert [text for text, _ch in ear.heard] == [LONG_HALF, "then rest."]


# ── the holding phrase ──


async def test_no_holding_phrase_when_text_comes_in_time():
    ear = Ear()
    layer = ConversationLayer(sink=ear, model=lambda: Model(["About ten minutes."], delay=0.05), hold_after=0.3)
    await layer.take("How long does pasta take to cook?")
    assert [text for text, _ch in ear.heard] == ["About ten minutes."]


async def test_a_holding_phrase_when_no_text_has_come_and_never_twice_in_a_row():
    ear = Ear()
    slow = Model(["About ten minutes."], delay=0.3)
    layer = ConversationLayer(sink=ear, model=lambda: slow, hold_after=0.1)
    await layer.take("How long does pasta take to cook?")
    assert [text for text, _ch in ear.heard] == [HOLD_PHRASES[0], "About ten minutes."]
    ear.heard.clear()
    slow.pieces = ["Rice takes about fifteen."]
    await layer.take("And rice?")
    assert [text for text, _ch in ear.heard] == ["Rice takes about fifteen."]  # the turn after a hold: none


async def test_text_without_a_sentence_waits_longer_before_holding(monkeypatch):
    """Text came by HOLD_NO_TEXT_SECONDS: the holding phrase waits for HOLD_AFTER_SECONDS instead."""
    monkeypatch.setattr(layer_module, "HOLD_NO_TEXT_SECONDS", 0.05)
    ear = Ear()
    model = Model(["About", 0.15, " ten minutes."])
    layer = ConversationLayer(sink=ear, model=lambda: model, hold_after=0.5)
    await layer.take("How long does pasta take to cook?")
    assert [text for text, _ch in ear.heard] == ["About ten minutes."]


async def test_no_second_holding_phrase_within_a_minute():
    ear = Ear()
    slow = Model(["About ten minutes."], delay=0.3)
    layer = ConversationLayer(sink=ear, model=lambda: slow, hold_after=0.1)
    await layer.take("How long does pasta take to cook?")
    await layer.take("And rice?")  # no hold: the turn after one
    ear.heard.clear()
    layer._last_hold_at = layer._clock()  # as if the hold had been a moment ago, two turns back
    layer._held_last_turn = False
    await layer.take("And potatoes?")
    assert not set(HOLD_PHRASES) & {text for text, _ch in ear.heard}
