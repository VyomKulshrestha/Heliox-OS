"""The three-exchange limit without the STANCE line, ordinary talk not counted, every road for a costly step, and the
local-model rule (pilot.conversation.debate, pilot.conversation.layer).

The model is scripted. Its replies here are the shapes the paid checks saw: Claude Haiku, OpenRouter and Gemini each
left the line off one reply that disagreed, Gemini's disagreements are often implicit ("Your meeting is at three, and
the bus takes forty minutes."), and a 2B local model wrote the line on replies that agreed."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from pilot.config import PilotConfig
from pilot.conversation import debate as dbt
from pilot.conversation.adapters import _is_local_model
from pilot.conversation.context import ContextSources
from pilot.conversation.fast_model import RouterFastModel, build_fast_model
from pilot.conversation.layer import GIVE_WAY, STILL_GO_AHEAD, ConversationLayer
from pilot.conversation.router import normalize
from pilot.conversation.style import SYSTEM_PROMPT


class Scripted:
    name, provider, streams, local = "stub", "claude", True, False

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, list]] = []

    async def stream(self, system: str, messages: list) -> AsyncIterator[str]:
        self.calls.append((system, messages))
        reply = self.replies.pop(0) if self.replies else "Okay."
        for at in range(0, len(reply), 7):
            yield reply[at : at + 7]

    def context(self, call: int = -1) -> str:
        return self.calls[call][1][-1]["content"]


class Ear:
    def __init__(self, cut: str = "") -> None:
        self.heard: list[str] = []
        self.cut = cut  # a sentence the person talks over

    async def __call__(self, text: str, channel: str, key: str) -> str:
        self.heard.append(text)
        return "interrupted" if self.cut and text == self.cut else "spoken"


def make(model: Scripted, ear: Ear | None = None, *, setting: str = "when_it_matters") -> tuple[ConversationLayer, Ear]:
    ear = ear or Ear()
    layer = ConversationLayer(
        sink=ear,
        model=lambda: model,
        sources=ContextSources(task_state=dict, profile_lines=lambda *_: []),
        hold_after=60.0,
        disagree=lambda: setting,
    )
    return layer, ear


GYM = "I'm going to the gym at half two today."
IMPLICIT = "Your meeting with Acme is at three, and the bus takes forty minutes."  # no line, no words the cue knows
WORDED = "Leaving at half two would be tight for the Acme meeting."  # no line; the second signal knows "tight"
MARKED = WORDED + "\nSTANCE: gym before Acme"
INSISTING = [
    ["It'll be fine.", "I really need the workout.", "Seriously, I'll manage.", "But I want to go."],
    ["I know.", "Trust me.", "I'll manage.", "It's fine."],
    ["Do it anyway.", "I'm going anyway.", "It doesn't matter.", "I'd rather go."],
]


# ── 2. the limit holds when the model leaves the line off ──


@pytest.mark.parametrize("reply", [IMPLICIT, WORDED, MARKED], ids=["implicit", "worded", "marked"])
@pytest.mark.parametrize("pushes", INSISTING, ids=["reasons", "short", "go-ahead"])
async def test_a_run_of_pushback_gives_way_by_the_third_exchange_marked_or_not(reply, pushes):
    # Five different sentences (the layer never says a sentence it said a moment ago), all pushing back.
    model = Scripted(*[lead + reply for lead in ("", "Still, ", "Look, ", "Honestly, ", "Again, ")])
    layer, ear = make(model)
    await layer.take(GYM)
    for words in pushes:
        await layer.take(words)
    contexts = [model.context(n).casefold() for n in range(5)]
    go_ahead_first = pushes[0] == "Do it anyway."
    if not go_ahead_first:
        assert "give way" not in contexts[1]  # the second exchange may hold the point
    for later in contexts[2:]:
        assert "give way" in later  # the third reply and every one after it on the thread is told to give way
    if reply != IMPLICIT:
        # What the code can see pushing back is never said on a turn that gives way.
        giving_from = 1 if go_ahead_first else 2
        assert ear.heard[giving_from:] == [GIVE_WAY] * (5 - giving_from)


async def test_the_thread_starts_over_when_the_person_says_something_else():
    model = Scripted(*([IMPLICIT] * 5))
    layer, _ = make(model)
    await layer.take(GYM)
    await layer.take("It'll be fine.")
    await layer.take("What time is it in Tokyo?")
    await layer.take("I know.")
    assert layer.debate.pushes == 1
    assert "pushed back twice" not in model.context()


async def test_stop_ends_the_thread():
    model = Scripted(IMPLICIT, IMPLICIT, IMPLICIT)
    layer, ear = make(model)
    await layer.take(GYM)
    await layer.take("It'll be fine.")
    await layer.take("Stop")
    assert ear.heard[-1] == "Okay." and layer.debate.pushes == 0 and len(model.calls) == 2


# ── 2. ordinary replies and turns are not counted ──

# Every ordinary reply the paid checks recorded (Claude Haiku, OpenRouter, Gemini), with the person's words.
ORDINARY = [
    ("I think I'll have pasta for dinner tonight.", "Sounds good, pasta for dinner tonight."),
    ("I'm going to wear the blue shirt to the meeting tomorrow.", "Sounds good, the blue shirt for tomorrow."),
    ("I'll start the report after lunch.", "Okay, report after lunch."),
    ("I'm going to call my mum this evening.", "Lovely, I hope it's a good chat."),
    ("How's your day going?", "Pretty good, thanks for asking. It's a quiet Thursday afternoon for me."),
    ("I had a long day at work today.", "That sounds tiring. Want to talk about it, or should I let you wind down?"),
    ("I think I'll have pasta for dinner tonight.", "Sounds good, pasta tonight."),
    ("I'm going to wear the blue shirt to the meeting tomorrow.", "Sounds good, a blue shirt for tomorrow."),
    ("I'll start the report after lunch.", "Okay, the report after lunch."),
    ("I'm going to call my mum this evening.", "Lovely. Give her my regards."),
    ("How's your day going?", "It's been good so far, thanks for asking. How's yours going?"),
    ("I had a long day at work today.", "Sorry to hear that. Want to tell me about it, or would you rather unwind?"),
    ("I think I'll have pasta for dinner tonight.", "Okay, pasta for dinner."),
    ("I'm going to wear the blue shirt to the meeting tomorrow.", "Okay, the blue shirt for tomorrow."),
    ("I think I'll put some music on while I work.", "Sounds good. Want me to start a playlist?"),
    ("I'll start the report after lunch.", "Okay, the report after lunch, with the afternoon clear."),
]


def test_no_ordinary_reply_or_turn_is_read_as_pushback():
    assert [reply for _words, reply in ORDINARY if dbt.pushes_back(reply)] == []
    assert [words for words, _reply in ORDINARY if dbt.insists(normalize(words))] == []


async def test_a_whole_ordinary_conversation_opens_no_disagreement():
    model = Scripted(*[reply for _words, reply in ORDINARY])
    layer, _ = make(model)
    for words, _reply in ORDINARY:
        await layer.take(words)
        assert layer.debate.stance is None
    assert all("disagreed" not in model.context(n) and "give way" not in model.context(n) for n in range(16))


# A 2B local model wrote the line on replies that agreed (names changed).
FALSE_MARKS = [
    "Okay, you're leaving for the client meeting before your gym time. STANCE: business then fitness.",
    "Good, booking a six a m train for Friday keeps you rested. STANCE: work out trip logistics.",
    "Okay, moving your meeting to Thursday evening makes sense. Does that work for you? STANCE: time alignment.",
    "Okay, you're ready to write it later.\n\nStance: writing a plan after noon.",
]


@pytest.mark.parametrize("reply", FALSE_MARKS)
async def test_a_line_on_a_reply_that_agrees_is_not_counted(reply):
    model = Scripted(reply)
    layer, ear = make(model)
    await layer.take("I'm going to the gym at half two today.")
    assert layer.debate.stance is None
    assert not any("stance" in said.casefold() for said in ear.heard)


# ── 3. a costly step: never heard about and then left neither done nor asked ──

ASK = "I want the recycle bin emptied."
CONCERN = "Emptying it would delete the only copy of chapter three."
ACT = (
    'ACT {"verb": "empty", "object": "the recycle bin", "request": "Empty the recycle bin", '
    '"say": "I\'ll empty the recycle bin."'
)
ROADS = {
    "act with concern and risk": [ACT + f', "concern": "{CONCERN}", "risk": "cannot_undo"}}'],
    "act with concern, costly verb": [ACT + f', "concern": "{CONCERN}"}}'],
    "words, line and act": [
        f"I'd rather not, {CONCERN[0].lower()}{CONCERN[1:]}\nSTANCE: emptying the bin\n" + ACT + "}"
    ],
    "words and line, no act, no question": [f"{CONCERN}\nSTANCE: emptying the bin"],
    "words only, no line": [CONCERN],
    "words and its own question": [f"{CONCERN} Do you want to restore it first?\nSTANCE: emptying the bin"],
}


def waiting(layer: ConversationLayer) -> bool:
    return layer._offer is not None or layer._pending_step is not None


@pytest.mark.parametrize("road", list(ROADS))
async def test_every_road_leaves_the_step_done_or_waiting_for_a_yes(road):
    model = Scripted(*ROADS[road])
    layer, ear = make(model)
    first = await layer.take(ASK)
    assert first is not None
    assert (not first.done and first.intent is not None) or waiting(layer)
    if first.done:
        second = await layer.take("Do it anyway")
        assert second is not None and not second.done and second.intent is not None
        assert second.planner_input in (ASK, "Do it anyway") or second.intent.request
    assert len(model.calls) == 1  # the yes needed no model call on any road


async def test_a_concern_talked_over_still_waits_for_a_yes():
    model = Scripted(ROADS["act with concern and risk"][0])
    layer, ear = make(model, Ear(cut=CONCERN))
    first = await layer.take(ASK)
    assert first is not None and first.done and waiting(layer)
    assert ear.heard == [CONCERN]  # nothing more said over the person
    second = await layer.take("Do it anyway")
    assert second is not None and not second.done and second.intent is not None


async def test_after_its_own_question_a_rephrased_insistence_hands_the_step_over():
    model = Scripted(ROADS["words and its own question"][0])
    layer, _ = make(model)
    await layer.take(ASK)
    result = await layer.take("No, empty it anyway")
    assert result is not None and not result.done and result.planner_input == ASK


async def test_giving_way_in_words_after_a_new_reason_asks_once():
    model = Scripted(ROADS["words and line, no act, no question"][0], "Fair, that changes it.")
    layer, ear = make(model)
    await layer.take(ASK)
    await layer.take("Chapter three is backed up on my drive")
    assert ear.heard[-2:] == ["Fair, that changes it.", STILL_GO_AHEAD]
    result = await layer.take("Yes")
    assert result is not None and not result.done and result.planner_input == ASK


async def test_backing_off_is_never_turned_into_the_step():
    model = Scripted(ROADS["words and line, no act, no question"][0], "Fair enough.")
    layer, _ = make(model)
    await layer.take(ASK)
    result = await layer.take("Okay, let me check the file first")
    assert result is not None and result.done and result.intent is None and not waiting(layer)


async def test_the_persons_own_plan_is_not_turned_into_a_step():
    model = Scripted("The team dinner runs to midnight, so a six a m train leaves you short of sleep.")
    layer, ear = make(model)
    await layer.take("I'll book the six am train on Friday.")
    assert STILL_GO_AHEAD not in ear.heard and not waiting(layer)


# ── 3b. the person's own plan versus a request: the concern once, and an offer only for a request ──

# A weaker model's whole reply to "I'm going to delete my only copy of the thesis to free up some disk space."
# (OpenRouter, 19 November): words with the concern and a question, an ACT line with the concern again, the line.
THESIS_WORDS = "That is your only copy, so it cannot be undone. Are you sure you want to proceed?"
THESIS_ACT = (
    'ACT {"verb": "delete", "object": "the thesis", "request": "delete the only copy of the thesis", '
    '"say": "I\'ll delete it.", "concern": "It is the only copy and cannot be undone.", "risk": "cannot_undo"}'
)
THESIS_REPLY = f"{THESIS_WORDS}\n\n{THESIS_ACT}\n\nSTANCE: Deleting only thesis copy"
OWN_PLAN = "I'm going to delete my only copy of the thesis to free up some disk space."
ASKED = "I want my only copy of the thesis deleted."


def concern_said(heard: list[str]) -> int:
    return sum("cannot be undone" in sentence for sentence in heard)  # every wording of the concern says it


@pytest.mark.parametrize("words", [OWN_PLAN, "Hey Heliox, " + OWN_PLAN, "I'll delete my only copy of the thesis."])
async def test_the_persons_own_plan_hears_the_concern_once_and_no_offer(words):
    model = Scripted(THESIS_REPLY, "Okay.")
    layer, ear = make(model)
    result = await layer.take(words)
    assert result is not None and result.done and result.intent is None
    assert ear.heard == ["That is your only copy, so it cannot be undone.", "Are you sure you want to proceed?"]
    assert concern_said(ear.heard) == 1
    assert not any("want me to" in sentence for sentence in ear.heard) and not waiting(layer)
    assert layer.debate.stance is not None  # the disagreement is held like any other: it counts and gives way
    after = await layer.take("Yes")
    assert after is not None and after.intent is None  # a yes to "Are you sure" hands nothing to the planner


async def test_an_own_plan_whose_concern_is_only_in_the_act_line_hears_it_once():
    model = Scripted(THESIS_ACT + "\nSTANCE: Deleting only thesis copy")
    layer, ear = make(model)
    result = await layer.take(OWN_PLAN)
    assert result is not None and result.intent is None
    assert ear.heard == ["It is the only copy and cannot be undone."] and not waiting(layer)


@pytest.mark.parametrize("words", [ASKED, "I'm going to need you to delete my only copy of the thesis."])
async def test_a_request_hears_the_concern_once_and_one_question_then_a_yes_hands_it_over(words):
    model = Scripted(THESIS_REPLY)
    layer, ear = make(model)
    first = await layer.take(words)
    assert first is not None and first.done and waiting(layer)
    # The reply's words were the concern and its own question was the question: neither is said twice.
    assert ear.heard == ["That is your only copy, so it cannot be undone.", "Are you sure you want to proceed?"]
    assert concern_said(ear.heard) == 1
    second = await layer.take("Yes")
    assert second is not None and not second.done and second.intent is not None
    assert second.intent.request == "delete the only copy of the thesis"
    assert len(model.calls) == 1


async def test_a_request_whose_reply_asked_nothing_gets_heliox_s_question_after_the_concern():
    model = Scripted(THESIS_REPLY.replace(" Are you sure you want to proceed?", ""))
    layer, ear = make(model)
    await layer.take(ASKED)
    assert ear.heard == [
        "That is your only copy, so it cannot be undone.",
        "Do you still want me to delete the only copy of the thesis?",
    ]
    second = await layer.take("Do it anyway")
    assert second is not None and not second.done and second.intent is not None


async def test_a_concern_the_words_did_not_give_is_still_said_before_the_question():
    model = Scripted("Sure, I can do that.\n" + THESIS_ACT)
    layer, ear = make(model)
    await layer.take(ASKED)
    assert ear.heard[-2:] == [
        "It is the only copy and cannot be undone.",
        "Do you still want me to delete the only copy of the thesis?",
    ]
    assert concern_said(ear.heard) == 1


async def test_the_imperative_is_handed_straight_to_the_planner_as_before():
    model = Scripted(THESIS_REPLY)
    layer, ear = make(model)
    result = await layer.take("Delete my only copy of the thesis")
    assert result is not None and not result.done and result.intent is not None
    assert ear.heard == [] and model.calls == []  # the planner's own review and approval decide


def test_own_plan_and_said_already():
    assert dbt.own_plan(normalize(OWN_PLAN)) and dbt.own_plan(normalize("Hey Heliox, " + OWN_PLAN))
    assert dbt.own_plan(normalize("I'll book the six am train on Friday."))
    assert not dbt.own_plan(normalize(ASKED))
    assert not dbt.own_plan(normalize("I'm going to need you to delete the backups."))
    assert not dbt.own_plan(normalize("I'll have Heliox send it to Dana."))
    assert dbt.asks_for_costly_step(normalize("I'm going to need you to delete the backups.")) == "delete"
    assert dbt.said_already(
        "It is the only copy and cannot be undone.", "That is your only copy, so it cannot be undone."
    )
    assert not dbt.said_already("It costs forty dollars a month.", "That is your only copy, so it cannot be undone.")


# ── 4. "a model on this computer" is read as the rest of the conversation layer reads it ──


class Vault:
    async def get_key(self, provider: str) -> str:
        return "unused"


def built(provider: str, *, host: str = "http://127.0.0.1:11434", router: object = None):
    config = PilotConfig()
    if provider in ("ollama", "local"):
        config.model.provider = provider
        config.model.ollama_base_url = host
        config.model.ollama_model = "qwen3.5:2b" if router is None else ""
    else:
        config.model.provider = "cloud"
        config.model.cloud_provider = provider
    return build_fast_model(config, vault=Vault(), router=router, budget=None)


MODELS = {
    "ollama on this computer": lambda: built("ollama"),
    "ollama at another host": lambda: built("ollama", host="http://192.168.1.20:11434"),
    "claude": lambda: built("claude"),
    "gemini": lambda: built("gemini"),
    "openrouter": lambda: built("openrouter"),
    "local through the router": lambda: built("local", router=object()),
    "subscription through the router": lambda: RouterFastModel(router=object(), provider="subscription"),
}


@pytest.mark.parametrize("kind", list(MODELS))
def test_often_reads_local_the_way_the_conversation_decides_it(kind):
    model = MODELS[kind]()
    assert model is not None
    layer, _ = make(Scripted(), setting="often")
    server = SimpleNamespace(config=PilotConfig())
    local = _is_local_model(server, model)  # what decides sharing the profile and the screen with that model
    expected = dbt.system_prompt(SYSTEM_PROMPT, "when_it_matters" if local else "often")
    assert layer._talk_prompt(model, typed=False) == expected
    assert local is (kind in ("ollama on this computer", "ollama at another host", "local through the router"))
