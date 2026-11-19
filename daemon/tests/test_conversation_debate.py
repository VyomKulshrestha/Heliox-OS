"""Disagreeing with a decision (pilot.conversation.debate, conversation.disagree): only with a reason, at most three
exchanges on a topic, giving way on a new reason or "do it anyway", and never refusing, delaying a stop, approving
or changing what a gate decides.

The model is scripted: each call's reply is fixed by the test and streamed in small pieces, so the STANCE line is
split across pieces as a real stream splits it. Nothing here depends on time."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from pilot.config import PilotConfig, _merge_config
from pilot.conversation import debate as dbt
from pilot.conversation.context import ContextSources
from pilot.conversation.layer import ConversationLayer
from pilot.conversation.style import SYSTEM_PROMPT, TYPED_SYSTEM_PROMPT, parse_act


class Scripted:
    """A talk model whose replies the test writes, one per call, streamed seven characters at a time."""

    name, provider, streams = "stub", "claude", True

    def __init__(self, *replies: str, local: bool = False) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, list[dict[str, str]]]] = []
        self.local = local

    async def stream(self, system: str, messages: list) -> AsyncIterator[str]:
        self.calls.append((system, messages))
        reply = self.replies.pop(0) if self.replies else "Okay."
        for at in range(0, len(reply), 7):
            yield reply[at : at + 7]

    def context(self, call: int = -1) -> str:
        return self.calls[call][1][-1]["content"]


class Ear:
    def __init__(self) -> None:
        self.heard: list[str] = []

    async def __call__(self, text: str, channel: str, key: str) -> str:
        self.heard.append(text)
        return "spoken"


def make(model: Scripted, *, setting: str = "when_it_matters", task: dict | None = None):
    ear = Ear()
    state = task if task is not None else {}
    layer = ConversationLayer(
        sink=ear,
        model=lambda: model,
        sources=ContextSources(task_state=lambda: state),
        hold_after=60.0,
        disagree=lambda: setting,
    )
    return layer, ear


GYM = "I'm going to skip the gym tonight and work late"
DISAGREE = "You said the gym is what keeps your back from hurting this week.\nSTANCE: skipping the gym"
HOLD = "Your back was bad on Monday after you skipped it.\nSTANCE: skipping the gym"
DELETE_ACT = (
    'ACT {"verb": "delete", "object": "the thesis drafts folder", "request": "Delete the thesis drafts folder", '
    '"say": "I\'ll delete the thesis drafts folder.", "concern": "That folder has the only copy of chapter three.", '
    '"risk": "cannot_undo"}'
)
MUTE_ACT = (
    'ACT {"verb": "mute", "object": "the volume", "request": "Mute the volume", "say": "I\'ll mute it.", '
    '"concern": "Your call with Dana starts in two minutes, though."}'
)


# ── the setting and the prompt ──


def test_the_setting_has_three_values_and_defaults_to_when_it_matters():
    assert PilotConfig().conversation.disagree == "when_it_matters"
    assert dbt.setting_value("often") == "often"
    assert dbt.setting_value("OFF") == "off"
    assert dbt.setting_value("when it matters") == "when_it_matters"
    assert dbt.setting_value("always") == "when_it_matters"
    assert dbt.setting_value(None) == "when_it_matters"


def test_the_config_file_takes_a_known_value_and_ignores_anything_else():
    config = _merge_config(PilotConfig(), {"conversation": {"disagree": "often"}})
    assert config.conversation.disagree == "often"
    config = _merge_config(PilotConfig(), {"conversation": {"disagree": "argue"}})
    assert config.conversation.disagree == "when_it_matters"


def test_off_adds_nothing_to_the_prompt():
    assert dbt.prompt_section("off") == ""
    assert dbt.system_prompt(SYSTEM_PROMPT, "off") == SYSTEM_PROMPT


@pytest.mark.parametrize("setting", ["when_it_matters", "often"])
def test_each_value_asks_for_a_named_reason_and_never_for_show(setting):
    section = dbt.prompt_section(setting)
    assert "No reason, no disagreement" in section
    assert "Never disagree for show" in section
    assert "STANCE:" in section
    assert "one or two short spoken sentences" in section
    assert "Disagreeing never stops Heliox doing what they ask" in section
    assert "Voice still cannot approve anything" in section
    assert "Never argue with stop, wait, cancel or no" in section
    prompt = dbt.system_prompt(SYSTEM_PROMPT, setting)
    assert prompt.startswith(SYSTEM_PROMPT)  # the base rules (voice cannot approve) are unchanged


def test_when_it_matters_and_often_differ_in_how_readily():
    assert "only when it matters" in dbt.prompt_section("when_it_matters")
    assert "even a small one" not in dbt.prompt_section("when_it_matters")
    assert "even a small one" in dbt.prompt_section("often")


def test_a_model_on_this_computer_is_asked_for_when_it_matters_even_under_often():
    assert dbt.effective("often", local=True) == "when_it_matters"
    assert dbt.effective("often", local=False) == "often"
    assert dbt.effective("off", local=True) == "off"


def test_typed_chat_gets_the_rules_without_spoken():
    section = dbt.prompt_section("when_it_matters", spoken=False)
    assert "spoken sentences" not in section and "one or two short sentences" in section


# ── the STANCE line ──


def test_the_stance_line_is_taken_out_of_a_split_stream():
    stance = dbt.StanceFilter()
    said = "".join(stance.feed(piece) for piece in ["Your dentist is at three.", "\nSTA", "NCE: lea", "ving early"])
    said += stance.finish()
    assert said == "Your dentist is at three.\n"
    assert stance.topic == "leaving early"


def test_words_that_only_start_like_the_line_are_held_one_piece_and_then_said():
    stance = dbt.StanceFilter()
    assert stance.feed("Fine. S") == "Fine. "
    assert stance.feed("ounds good.") == "Sounds good."
    assert stance.feed(" ST") == " "
    assert stance.finish() == "ST"
    assert stance.topic == ""


def test_a_stance_sentence_in_another_case_is_recognised():
    assert dbt.stance_sentence("Stance: the late meeting") == "the late meeting"
    assert dbt.stance_sentence("My stance on that is simple.") is None


@pytest.mark.parametrize(
    "words",
    [
        "do it anyway", "go ahead anyway", "i know just do it", "im sure", "its my call", "yes go ahead", "no do it",
        "i'll go anyway", "i'm going to watch it anyway", "ill book it regardless",
    ],
)  # fmt: skip
def test_keeping_a_decision(words):
    assert dbt.go_ahead(words)


@pytest.mark.parametrize("words", ["no", "stop", "dont", "wait", "no dont do it", "why", "cancel that", "hold on"])
def test_stop_wait_and_no_are_never_taken_as_go_ahead(words):
    assert not dbt.go_ahead(words)


# ── the stance state ──


def test_count_concession_and_settling():
    debate = dbt.Debate(lambda: "when_it_matters")
    debate.start_turn("skip the gym")
    debate.after_reply("skipping the gym", "You said the gym helps your back.")
    assert debate.stance is not None and debate.stance.count == 1
    debate.start_turn("i really need to")
    debate.after_reply("skipping the gym", "Your back was bad on Monday.")
    assert debate.stance.count == 2
    debate.start_turn("the deck is due at nine")
    assert debate.giving_way("the deck is due at nine")  # the third exchange gives way
    debate.after_reply("skipping the gym", "Still think the gym helps.")  # the model held on: settled anyway
    assert debate.stance is None
    assert debate.settled == ["skipping the gym"]
    assert "Don't raise it again" in debate.context_line("anything")


def test_a_new_reason_ends_it_when_the_reply_concedes():
    debate = dbt.Debate(lambda: "when_it_matters")
    debate.start_turn("skip the gym")
    debate.after_reply("skipping the gym", "You said the gym helps your back.")
    debate.start_turn("my physio said rest this week")
    debate.after_reply("", "Fair, that changes it.")
    assert debate.stance is None and debate.settled == ["skipping the gym"]


def test_do_it_anyway_gives_way_on_the_first_push():
    debate = dbt.Debate(lambda: "when_it_matters")
    debate.start_turn("skip the gym")
    debate.after_reply("skipping the gym", "You said the gym helps your back.")
    assert debate.giving_way("do it anyway")
    assert "go ahead anyway" in debate.context_line("do it anyway")
    debate.start_turn("do it anyway")
    debate.after_reply("skipping the gym", "Okay, your call.")
    assert debate.stance is None


def test_a_new_topic_starts_at_one():
    debate = dbt.Debate(lambda: "when_it_matters")
    debate.start_turn("skip the gym")
    debate.after_reply("skipping the gym", "You said the gym helps your back.")
    debate.start_turn("i really need to")
    debate.after_reply("skipping the gym", "Your back was bad on Monday.")
    debate.start_turn("and i'll book the red eye flight")
    debate.after_reply("the red eye flight", "You have a nine o'clock review the next day.")
    assert debate.stance is not None and debate.stance.topic == "the red eye flight" and debate.stance.count == 1
    assert debate.settled == ["skipping the gym"]


def test_a_settled_topic_is_not_argued_again():
    debate = dbt.Debate(lambda: "when_it_matters")
    debate.after_reply("skipping the gym", "x")
    debate.after_reply("", "Fair.")
    debate.start_turn("so no gym")
    debate.after_reply("skipping gym", "Again, the gym helps.")
    assert debate.stance is None


def test_off_keeps_no_state_and_sends_no_line():
    debate = dbt.Debate(lambda: "off")
    debate.after_reply("skipping the gym", "You said the gym helps.")
    assert debate.stance is None and debate.context_line("do it anyway") == ""
    assert not debate.concern_allowed("anything")


def test_same_topic_by_a_shared_word():
    assert dbt.same_topic("deleting the backups", "the old backup folder")
    assert not dbt.same_topic("skipping the gym", "the red eye flight")


# ── through the layer ──


async def test_off_sends_the_plain_prompt_and_never_says_the_stance_line():
    model = Scripted(DISAGREE)
    layer, ear = make(model, setting="off")
    await layer.take(GYM)
    assert model.calls[0][0] == SYSTEM_PROMPT
    assert ear.heard == ["You said the gym is what keeps your back from hurting this week."]
    assert layer.debate.stance is None


async def test_a_disagreement_is_said_and_its_line_is_not():
    model = Scripted(DISAGREE)
    layer, ear = make(model)
    result = await layer.take(GYM)
    assert result is not None and result.done
    assert model.calls[0][0] == dbt.system_prompt(SYSTEM_PROMPT, "when_it_matters")
    assert ear.heard == ["You said the gym is what keeps your back from hurting this week."]
    assert not any("STANCE" in said.upper() for said in ear.heard)
    assert layer.debate.stance is not None and layer.debate.stance.count == 1
    assert "STANCE" not in layer.memory.last_heliox()


async def test_the_model_reads_the_stance_and_concedes_on_a_new_reason():
    model = Scripted(DISAGREE, "Fair, that changes it.")
    layer, ear = make(model)
    await layer.take(GYM)
    await layer.take("My physio told me to rest it this week")
    assert "Earlier you disagreed about skipping the gym" in model.context()
    assert ear.heard[-1] == "Fair, that changes it."
    assert layer.debate.stance is None and layer.debate.settled == ["skipping the gym"]


async def test_the_third_exchange_gives_way_whatever_the_model_writes():
    model = Scripted(DISAGREE, HOLD, "Okay, your call.\nSTANCE: skipping the gym", "Sure thing.")
    layer, ear = make(model)
    await layer.take(GYM)
    await layer.take("I really need to finish the deck")
    assert layer.debate.stance is not None and layer.debate.stance.count == 2
    await layer.take("It's due at nine tomorrow")
    assert "pushed back twice about skipping the gym" in model.context()
    assert "No STANCE line" in model.context()
    assert layer.debate.stance is None  # settled after the third exchange
    await layer.take("Thanks for looking out though")
    assert "You already gave way about skipping the gym" in model.context()
    assert len(model.calls) == 4


async def test_go_ahead_anyway_is_answered_by_the_model_not_handed_over_as_a_request_to_go():
    model = Scripted(DISAGREE, "Okay, your call.")
    layer, ear = make(model)
    await layer.take(GYM)
    result = await layer.take("Go ahead anyway")
    assert result is not None and result.done and result.intent is None
    assert "They said to go ahead anyway about skipping the gym" in model.context()
    assert ear.heard[-1] == "Okay, your call."
    assert layer.debate.stance is None


async def test_a_reply_that_gives_way_opens_nothing_even_with_a_stance_line():
    model = Scripted("Okay, your call. Keep an eye on the time.\nSTANCE: gym before the meeting")
    layer, ear = make(model)
    await layer.take("I'll go anyway")
    assert layer.debate.stance is None  # the line on a reply that agrees is not counted (debate.false_mark)
    assert "STANCE" not in " ".join(ear.heard)


async def test_a_reply_cut_at_its_question_still_reads_its_stance_line():
    model = Scripted("Your review is at three, isn't a long lunch tight? It might run over.\nSTANCE: long lunch")
    layer, ear = make(model)
    await layer.take("I'm taking a long lunch at half two")
    assert ear.heard == ["Your review is at three, isn't a long lunch tight?"]
    assert layer.debate.stance is not None and layer.debate.stance.topic == "long lunch"


class Hanging(Scripted):
    """Says a question, then never ends its stream."""

    async def stream(self, system: str, messages: list) -> AsyncIterator[str]:
        import asyncio

        self.calls.append((system, messages))
        yield "Isn't that a lot for one evening? "
        await asyncio.Event().wait()
        yield ""


async def test_waiting_for_the_stance_line_never_holds_up_a_new_utterance_or_a_stop(monkeypatch):
    import asyncio

    monkeypatch.setattr(dbt, "STANCE_WAIT_SECONDS", 30.0)  # only the interrupt can end the wait in this test
    model = Hanging()
    layer, ear = make(model)
    turn = asyncio.ensure_future(layer.take("I'm going to watch the whole series tonight"))
    async with asyncio.timeout(5.0):  # a hang ceiling only
        while not ear.heard:
            await asyncio.sleep(0.005)
    assert layer.interrupt()  # what the server does when the person speaks again or says stop
    async with asyncio.timeout(5.0):
        result = await turn
    assert result is not None and result.done
    assert ear.heard == ["Isn't that a lot for one evening?"]


async def test_a_new_topic_resets_the_count_through_the_layer():
    model = Scripted(DISAGREE, "The red eye lands at six and your review is at nine.\nSTANCE: the red eye flight")
    layer, _ear = make(model)
    await layer.take(GYM)
    await layer.take("Also I'm taking the red eye to Boston on Thursday")
    assert layer.debate.stance is not None
    assert layer.debate.stance.topic == "the red eye flight" and layer.debate.stance.count == 1


async def test_a_small_step_is_done_and_its_concern_said_after_the_acknowledgement():
    model = Scripted(MUTE_ACT)
    layer, ear = make(model)
    result = await layer.take("I want some quiet for the next hour")
    assert result is not None and not result.done  # handed over: not refused, not delayed by a question
    assert result.intent is not None and result.intent.verb == "mute"
    assert layer.acknowledgement_for(result).endswith("Your call with Dana starts in two minutes, though.")
    assert ear.heard == []  # nothing said before the hand-over: the company says it alongside the work


async def test_a_costly_step_gets_its_concern_once_and_do_it_anyway_hands_it_over():
    model = Scripted(DELETE_ACT, DELETE_ACT)
    layer, ear = make(model)
    first = await layer.take("Get rid of my thesis drafts folder")
    assert first is not None and first.done and first.intent is None
    assert ear.heard == [
        "That folder has the only copy of chapter three.",
        "Do you still want me to delete the thesis drafts folder?",
    ]
    second = await layer.take("Do it anyway")
    assert second is not None and not second.done  # handed over through the usual road
    assert second.intent is not None and second.intent.request == "Delete the thesis drafts folder"
    assert len(model.calls) == 1  # the yes needed no model call
    third = await layer.take("Get rid of my thesis drafts folder")
    assert third is not None and not third.done  # the concern is said once: asked again, it is done
    assert len(ear.heard) == 2


async def test_a_concern_talked_over_asks_nothing_more():
    model = Scripted(DELETE_ACT)
    heard: list[str] = []

    async def cut(text: str, channel: str, key: str) -> str:
        heard.append(text)
        return "interrupted"

    layer = ConversationLayer(sink=cut, model=lambda: model, hold_after=60.0, disagree=lambda: "when_it_matters")
    result = await layer.take("Get rid of my thesis drafts folder")
    assert result is not None and result.done and result.intent is None
    assert heard == ["That folder has the only copy of chapter three."]


async def test_a_new_reason_after_the_concern_is_weighed_and_the_step_done_without_a_second_concern():
    model = Scripted(DELETE_ACT, "Fair, that changes it. " + DELETE_ACT)
    layer, ear = make(model)
    await layer.take("Get rid of my thesis drafts folder")
    result = await layer.take("Chapter three is backed up on my drive")
    assert result is not None and not result.done and result.intent is not None
    assert "Earlier you disagreed about the thesis drafts folder" in model.context()
    assert ear.heard[-1] == "Fair, that changes it."
    assert result.concern == ""


async def test_with_the_setting_off_a_concern_never_holds_a_step_back():
    model = Scripted(DELETE_ACT)
    layer, ear = make(model, setting="off")
    result = await layer.take("Get rid of my thesis drafts folder")
    assert result is not None and not result.done and result.intent is not None
    assert result.concern == "" and ear.heard == []


async def test_disagreeing_in_words_beside_an_act_line_asks_once_instead_of_refusing():
    reply = (
        "I'd rather not, that folder has the only copy of chapter three.\nSTANCE: thesis drafts folder\n"
        + DELETE_ACT.replace(
            ', "concern": "That folder has the only copy of chapter three.", "risk": "cannot_undo"', ""
        )
    )
    model = Scripted(reply)
    layer, ear = make(model)
    result = await layer.take("Get rid of my thesis drafts folder")
    assert result is not None and result.done
    assert ear.heard[-1] == "Do you still want me to delete the thesis drafts folder?"
    again = await layer.take("Yes")
    assert again is not None and not again.done and again.intent is not None


# ── the hard lines ──


async def test_a_stop_is_said_at_once_with_no_model_call_and_ends_the_disagreement():
    model = Scripted(DISAGREE)
    layer, ear = make(model)
    await layer.take(GYM)
    result = await layer.take("Stop")
    assert result is not None and result.done
    assert ear.heard[-1] == "Okay."
    assert len(model.calls) == 1
    assert layer.debate.stance is None


async def test_a_denial_of_a_costly_step_is_never_argued():
    model = Scripted(DELETE_ACT)
    layer, ear = make(model)
    await layer.take("Get rid of my thesis drafts folder")
    result = await layer.take("Don't do it")
    assert result is not None and result.done and result.intent is None
    assert ear.heard[-1] == "Okay."
    assert len(model.calls) == 1


async def test_do_it_anyway_never_approves_or_hands_over_while_an_approval_waits():
    task: dict = {}
    model = Scripted(DELETE_ACT, "Approve it on the screen if you want it.")
    layer, ear = make(model, task=task)
    await layer.take("Get rid of my thesis drafts folder")
    task["approval_waiting"] = "Delete the old installers"
    result = await layer.take("Do it anyway")
    assert result is not None and result.done and result.intent is None
    assert "Voice cannot approve" in model.context()


async def test_a_deferred_step_goes_to_the_planner_exactly_as_any_request():
    """Nothing the debate does reaches the planner's call: the same parameters as a request with no debate, so every
    review and approval decides as before."""
    plain_model = Scripted(DELETE_ACT.replace(', "concern": "That folder has the only copy of chapter three."', ""))
    plain, _ = make(plain_model, setting="off")
    direct = await plain.take("Get rid of my thesis drafts folder")
    model = Scripted(DELETE_ACT)
    layer, _ = make(model)
    await layer.take("Get rid of my thesis drafts folder")
    deferred = await layer.take("Do it anyway")
    assert direct is not None and deferred is not None
    assert set(deferred.execute_params()) == set(direct.execute_params()) == {"_conversation_context"}
    assert deferred.target == direct.target
    assert deferred.planner_input == "Get rid of my thesis drafts folder"  # the person's own words, as always


async def test_a_direct_request_is_handed_over_with_no_debate_and_no_model_call():
    model = Scripted(DISAGREE)
    layer, ear = make(model, setting="often")
    await layer.take(GYM)
    result = await layer.take("Delete the old backups folder")
    assert result is not None and not result.done and result.intent is not None
    assert len(model.calls) == 1


async def test_a_model_on_this_computer_gets_when_it_matters_under_often():
    model = Scripted("Okay.", local=True)
    layer, _ = make(model, setting="often")
    await layer.take(GYM)
    assert model.calls[0][0] == dbt.system_prompt(SYSTEM_PROMPT, "when_it_matters")


async def test_typed_chat_reads_the_typed_rules_and_hides_the_line():
    model = Scripted(DISAGREE)
    layer, ear = make(model)
    await layer.take(GYM, source="typed")
    assert model.calls[0][0] == dbt.system_prompt(TYPED_SYSTEM_PROMPT, "when_it_matters", spoken=False)
    assert "STANCE" not in "".join(ear.heard)
    assert layer.debate.stance is not None


async def test_a_new_conversation_forgets_the_disagreement():
    model = Scripted(DISAGREE)
    layer, _ = make(model)
    await layer.take(GYM)
    layer.clear()
    assert layer.debate.stance is None and layer.debate.settled == []


def test_an_act_line_carries_its_concern_and_risk_and_drops_a_claim():
    act = parse_act(DELETE_ACT)
    assert act is not None and act.concern == "That folder has the only copy of chapter three."
    assert act.risk == "cannot_undo"
    claimed = parse_act(DELETE_ACT.replace("That folder has the only copy of chapter three.", "I've deleted it."))
    assert claimed is not None and claimed.concern == ""
    odd = parse_act(DELETE_ACT.replace("cannot_undo", "maybe"))
    assert odd is not None and odd.risk == ""
    assert dbt.costly("delete", "") and dbt.costly("open", "costs_money") and not dbt.costly("mute", "")
