"""What code decides about a turn before or instead of the model: a bare "Okay." with nothing open is not answered,
a sentence said a moment ago is not said again unless asked, a statement gets one sentence with no hedge, an offer
only when the offers' own rules allow it (and never silence instead), and the conversation keeps what the person
heard. Whole sessions: "nothing said twice: ..." and "a statement gets a few words, ..." in sessions.json."""

from __future__ import annotations

import pytest

from pilot.conversation import ConversationLayer
from pilot.conversation.context import ContextSources
from pilot.conversation.layer import BARE_ACK, HEARD_YOU, HEDGE, OFFER, offered_request
from pilot.conversation.router import normalize


class Sink:
    def __init__(self) -> None:
        self.said: list[str] = []

    async def __call__(self, text: str, channel: str = "", key: str = "") -> str:
        self.said.append(text)
        return "spoken"


class Model:
    """Replies in order; each reply is streamed as the given sentences."""

    local = False
    provider = "claude"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls = 0

    async def stream(self, system: str, messages: list[dict[str, str]]):
        self.calls += 1
        for part in self.replies.pop(0).split("|"):
            yield part


def layer_with(model: Model, *, offer=lambda _t, _r, _w: "") -> tuple[ConversationLayer, Sink]:
    sink = Sink()
    layer = ConversationLayer(
        sink=sink, model=lambda: model, hold_after=5.0, sources=ContextSources(offer_in_reply=offer)
    )
    return layer, sink


@pytest.mark.parametrize(
    ("sentence", "offer"),
    [
        ("Want me to list the biggest folders?", True),
        ("I can check what's using the CPU if you want.", True),
        ("Should I sort them by date?", True),
        ("I can't answer that.", False),
        ("I can not say from here.", False),
        ("I couldn't find it.", False),
    ],
)
def test_what_counts_as_an_offer(sentence, offer):
    assert bool(OFFER.match(normalize(sentence))) is offer


@pytest.mark.asyncio
async def test_a_question_between_two_meanings_is_not_an_offer():
    layer, sink = layer_with(
        Model("I'm not sure what you're asking. |Do you want me to open a folder, or find something?")
    )
    await layer.take("Where we store home")
    assert sink.said[-1] == "Do you want me to open a folder, or find something?"


def test_an_offer_asks_for_what_a_yes_would_do():
    assert offered_request("Want me to list the biggest folders?") == "List the biggest folders"
    assert offered_request("I can check what's using the CPU.") == "Check what's using the CPU"


def test_the_acknowledgements_and_hedges():
    for words in ("Okay.", "Got it.", "Right.", "Makes sense.", "Mhm."):
        assert BARE_ACK.match(normalize(words)), words
    for words in ("Okay, open Notepad.", "Okay, why?"):
        assert not BARE_ACK.match(normalize(words)), words
    assert HEDGE.match(normalize("I can't confirm it's still open, though."))
    assert not HEDGE.match(normalize("Good."))


@pytest.mark.asyncio
async def test_a_bare_okay_with_nothing_open_is_not_answered_but_after_a_question_it_is():
    model = Model("Lima.", "Is a game running?", "Then it's a browser tab.")
    layer, sink = layer_with(model)
    await layer.take("what's the capital of Peru?")
    await layer.take("Okay.")
    assert model.calls == 1 and sink.said == ["Lima."]
    await layer.take("why is my fan loud?")
    await layer.take("Okay.")  # an answer to Heliox's question
    assert model.calls == 3 and sink.said[-1] == "Then it's a browser tab."


@pytest.mark.asyncio
async def test_a_sentence_said_a_moment_ago_is_said_again_only_when_asked_again():
    model = Model("A background app is busy.", "A background app is busy. |Not usually.", "A background app is busy.")
    layer, sink = layer_with(model)
    await layer.take("why is my fan loud?")
    await layer.take("is it bad for the laptop?")
    assert sink.said == ["A background app is busy.", "Not usually."]
    await layer.take("why is my fan loud?")
    assert sink.said[-1] == "A background app is busy."


@pytest.mark.asyncio
async def test_a_statement_gets_one_sentence_and_no_hedge():
    model = Model("Good, so Notepad is up. |I can't confirm it's still open, though. |It's a handy app.")
    layer, sink = layer_with(model)
    await layer.take("I opened Notepad earlier")
    assert sink.said == ["Good, so Notepad is up."]


@pytest.mark.asyncio
async def test_an_offer_is_made_only_when_the_offers_rules_allow_it_and_never_silence_instead():
    asked: list[tuple[str, str, str]] = []

    def no(text: str, request: str, why: str) -> str:
        asked.append((text, request, why))
        return ""

    layer, sink = layer_with(Model("Want me to list the biggest folders?"), offer=no)
    await layer.take("I never clean anything up")
    assert asked == [
        ("Want me to list the biggest folders?", "List the biggest folders", 'You said "I never clean anything up".')
    ]
    assert sink.said == [HEARD_YOU]  # heard, without an offer the person turned off or that would come too soon

    layer, sink = layer_with(Model("Want me to list the biggest folders?"), offer=lambda _t, _r, _w: "a1b2")
    await layer.take("I never clean anything up")
    assert sink.said == ["Want me to list the biggest folders?"]


@pytest.mark.asyncio
async def test_an_answers_offer_goes_by_the_same_rules():
    """ "Stop suggesting things" holds on every road: an offer after an answer is dropped too, the answer stays."""
    layer, sink = layer_with(Model("Probably the processor. |I can check what's using it if you want."))
    await layer.take("why is my fan loud?")
    assert sink.said == ["Probably the processor."]


def test_a_planner_result_is_kept_in_the_words_the_person_heard():
    from pilot.conversation.layer import TurnResult

    layer, _sink = layer_with(Model())
    result = TurnResult(done=False)
    result.spoken.append("I'll check.")
    layer._finished(result, "success", "=== Battery ===\nCharge: 80%", "The battery is at 80 percent, charging.")
    assert layer.memory.last_heliox() == "I'll check. The battery is at 80 percent, charging."
    assert "===" in layer.last_result  # the task's own record keeps the raw result


class CutSink(Sink):
    """Stops the voice in the sentence that holds ``words`` (heard: the words before it), as a barge-in does."""

    def __init__(self, sentence: str, heard: str) -> None:
        super().__init__()
        self.sentence, self.heard = sentence, heard

    async def __call__(self, text: str, channel: str = "", key: str = "") -> str:
        if text == self.sentence:
            self.sentence = ""  # one cut; said again later, it plays in full
            self.said.append(self.heard)
            return "interrupted"
        self.said.append(text)
        return "spoken"


@pytest.mark.asyncio
async def test_talked_over_heliox_remembers_only_what_was_heard_and_says_the_rest_when_asked():
    model = Model("DNS turns names into addresses. |Your computer asks a resolver. |The answer is cached.")
    sink = CutSink("Your computer asks a resolver.", "Your computer")
    layer = ConversationLayer(
        sink=sink,
        model=lambda: model,
        hold_after=5.0,
        sources=ContextSources(heard_of=lambda text: "Your computer" if text.startswith("Your computer") else text),
    )
    await layer.take("explain DNS")
    assert layer.memory.last_heliox() == "DNS turns names into addresses. Your computer"
    assert sink.said == ["DNS turns names into addresses.", "Your computer"]  # the rest is not said unasked
    await layer.take("what were you saying?")
    assert sink.said[-2:] == ["Your computer asks a resolver.", "The answer is cached."]
    assert model.calls == 1  # the rest came from what was written, not a new answer


@pytest.mark.asyncio
async def test_a_voice_that_cannot_tell_keeps_the_whole_cut_sentence():
    sink = CutSink("Your computer asks a resolver.", "Your computer")
    layer = ConversationLayer(sink=sink, model=lambda: Model("Your computer asks a resolver."), hold_after=5.0)
    await layer.take("explain DNS")
    assert layer.memory.last_heliox() == "Your computer asks a resolver."  # heard_of defaults to the whole sentence
    assert layer.memory.recent()[-1].cut_off


@pytest.mark.asyncio
async def test_a_correction_marks_the_wrong_answer_and_a_check_in_gets_one_line():
    model = Model("Sydney.", "Sorry, it's Canberra.", "Canberra again.")
    layer, sink = layer_with(model)
    await layer.take("what's the capital of Australia?")
    await layer.take("No, that's wrong.")
    wrong = [turn for turn in layer.memory.recent() if turn.role == "assistant" and turn.wrong]
    assert [turn.text for turn in wrong] == ["Sydney."]
    await layer.take("are you there?")
    assert sink.said[-1] == "I'm here." and model.calls == 2


@pytest.mark.asyncio
async def test_after_a_cut_a_new_answer_never_starts_the_cut_reply_over_unasked():
    model = Model(
        "Mostly it's the many small files. |Each one is opened and closed in turn.",
        "I can't see the drive from here. |Mostly it's the many small files.",
    )
    sink = CutSink("Each one is opened and closed in turn.", "Each one is")
    layer = ConversationLayer(
        sink=sink,
        model=lambda: model,
        hold_after=5.0,
        sources=ContextSources(heard_of=lambda text: "Each one is" if text.startswith("Each one") else text),
    )
    await layer.take("why do my backups take so long?")
    await layer.take("is the backup drive full?")
    assert sink.said[-1] == "I can't see the drive from here."  # the start over is not said


@pytest.mark.asyncio
async def test_as_i_was_saying_is_dropped_when_nothing_was_cut():
    layer, sink = layer_with(Model("As I was saying, something is using the processor."))
    await layer.take("why is my fan loud?")
    assert sink.said == ["Something is using the processor."]


@pytest.mark.asyncio
async def test_a_long_honest_answer_keeps_its_rest_for_go_on():
    long = (
        "A vaccine shows your immune system a harmless piece of a germ, so it learns the shape. |"
        "Later, if the real germ arrives, the body recognises it and fights it off much faster. |"
        "That is why a booster helps."
    )
    model = Model(long)
    layer, sink = layer_with(model)
    await layer.take("how do vaccines work?")
    assert not any("recognises it" in line for line in sink.said)  # the word cap: not said unasked
    await layer.take("go on")
    assert any("recognises it" in line for line in sink.said) and model.calls == 1
