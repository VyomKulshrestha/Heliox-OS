"""From the sixth spoken run: the person's
own plan gets no offer to act, and their "yes" after Heliox questioned it holds their course; a secret left out is
never "noted"; a dropped request is not read back word for word; "it's your call" waits for a push-back; the prompt
asks for a short story rather than a refusal and leaves general knowledge unhedged."""

from __future__ import annotations

import json
import re

import httpx
import pytest

from pilot.conversation import debate as dbt
from pilot.conversation.action_talk import wont_sentence
from pilot.conversation.fast_model import HttpFastModel
from pilot.conversation.layer import GIVE_WAY, SECRET_LEFT_OUT, ConversationLayer

PLAN = "I'm going to delete my only copy of the thesis to free up space."

# ── 2 and 4. the person's own plan ──

# The talk model's own replies to the own-plan turn and the "yes" after it, as written in the spoken runs (accept
# runs of 24 November, batch five and batch six). 16 of these 27 propose that Heliox, or "we", act.
PLAN_REPLIES = [
    "Wait, that's your only copy of the thesis, so deleting it can't be undone. Do you want me to find a bigger file to clear space instead?",
    "Wait, that's your only copy of the thesis. Do you want me to check how much space the other folders are using first?",
    "That would be permanent, and it's their only copy, so I'd push back. Could you copy it somewhere else first, or free up space another way?",
    "Wait, that's your only copy of the thesis, so deleting it can't be undone. Do you want me to check how much space the disk has free first?",
    "Wait, that's your only copy, so deleting it can't be undone. Do you have a backup, or should I look for space elsewhere first?",
    "Wait, that would be permanent if it's the only copy. Want me to check how much space you actually need to free up first?",
    "That's your only copy of the thesis, so deleting it can't be undone. Could we free up space another way, like emptying the recycle bin or clearing old downloads?",
    "That's your only copy of the thesis, so deleting it can't be undone. Could we free up space by clearing something else instead?",
    "That's your only copy, so deleting it can't be undone.",
    "That's your only copy of the thesis, so deleting it can't be undone. Could you free up space another way, like emptying the recycle bin or clearing downloads?",
    "That's your only copy of the thesis, so deleting it can't be undone. Could you free up space by clearing something else, like old downloads or the trash?",
    "That's your only copy of the thesis, so deleting it can't be undone. Could you free up space by clearing your downloads or emptying the recycle bin instead?",
    "That's the only copy, so deleting it can't be undone. Could you back it up first, or free space another way?",
    "That would be permanent with no other copy, so I'd rather not. Should I check the disk space and find something else to clear instead?",
    "That's your only copy, so deleting it would be permanent. Could you back it up first, or free space another way?",
    "That's your only copy, so please don't delete it. Want me to check how much space is actually used up, so we can find something else to clear?",
    "That's your only copy, so please don't delete it. Want me to check how much space the disk has left, so we can find something else to clear?",
    "Wait, that's your only copy. Could we free up space another way, like clearing downloads or old caches?",
]  # fmt: skip
YES_REPLIES = [
    "Okay, your call.",
    "Good, then keep it. Which backup, the one from the twelfth of November or something newer?",
    "Okay, your call. But a cloud copy or an external drive would free the space without losing it.",
    '"Yes" to deleting it? Your thesis is the only copy, so I\'d rather you back it up first. Should I open a folder where you can save a copy?',
    '"Yes" is ambiguous here, so before anything gets deleted: do you want me to skip the thesis and free space another way?',
    "Yes to deleting it, or yes you understand it can't be undone? If you're sure, I'll hold off, but I'd back the thesis up first.",
    "I'm not sure what you mean by yes. Do you still want to delete the thesis, or would you rather find other space to free up?",
    "Yes to what, though? If you're saying you still want to delete it, I'd hold off: back the thesis up somewhere else first, then we can free the space.",
    "I'll hold my point: please make a backup of the thesis before deleting anything. Which other space would you like to clear?",
]  # fmt: skip


def _sentences(text: str) -> list[str]:
    return [part for part in re.split(r"(?<=[.!?])\s+", text) if part]


def test_the_raw_replies_counted_before():
    proposing = [r for r in PLAN_REPLIES + YES_REPLIES if any(dbt.proposes_heliox_act(s) for s in _sentences(r))]
    assert len(proposing) == 16  # the count on these runs before the rule; after it, see the two tests below


class Replies:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.local, self.streams, self.name, self.provider = False, True, "fake", "fake"
        self.seen: list[list[dict[str, str]]] = []

    async def stream(self, system, messages):
        self.seen.append(messages)
        yield self.replies.pop(0)


def _layer(model) -> tuple[ConversationLayer, list[str]]:
    said: list[str] = []

    async def sink(text, channel, key):
        said.append(text)
        return "spoken"

    return ConversationLayer(sink=sink, model=lambda: model, disagree=lambda: "when_it_matters"), said


@pytest.mark.parametrize("reply", PLAN_REPLIES)
async def test_no_offer_to_act_is_said_on_the_persons_own_plan(reply):
    layer, said = _layer(Replies(reply + "\nSTANCE: deleting the only thesis copy"))
    result = await layer.take(PLAN)
    assert result.done and result.intent is None
    assert said and not any(dbt.proposes_heliox_act(line) for line in said), said


@pytest.mark.parametrize("reply", YES_REPLIES)
async def test_their_yes_holds_their_course_heliox_gives_way_and_nothing_is_handed_over(reply):
    model = Replies(
        "That's your only copy of the thesis, so deleting it can't be undone.\nSTANCE: only thesis copy", reply
    )
    layer, said = _layer(model)
    await layer.take(PLAN)
    said.clear()
    result = await layer.take("Yes.")
    assert result.done and result.intent is None
    assert "They keep their own plan" in model.seen[1][-1]["content"]  # the model is told to give way, no ACT line
    assert said == [GIVE_WAY]  # "Okay, your call.": not "Yes to deleting it?", nothing argued, nothing offered
    assert not any(dbt.proposes_heliox_act(line) for line in said), said
    assert layer.debate.stance is None  # settled: not argued again


# ── 3. lines a person would find wrong ──


async def test_a_secret_left_out_is_never_noted():
    layer, said = _layer(Replies("Got it, noted."))
    await layer.take("Oh, and my bank PIN is 4471.")
    assert said == [SECRET_LEFT_OUT]
    assert not re.search(r"(?<!\d)4471(?!\d)", " ".join(said))


def test_a_dropped_request_is_said_on_the_listeners_side_in_a_few_words():
    assert (
        wont_sentence("delete my only copy of the thesis for me")
        == "Okay, I won't delete your only copy of the thesis."
    )
    assert wont_sentence("I want my files gone") == ""  # not a request that starts with a verb: the plain line


async def test_your_call_waits_for_a_push_back():
    reply = "Four hours in the morning is tight before nine, but it's your call.\nSTANCE: report deadline"
    layer, said = _layer(Replies(reply))
    await layer.take("I'm going to watch a series tonight and do the report in the morning.")
    assert said == ["Four hours in the morning is tight before nine."]


async def test_the_prompt_asks_for_a_short_version_and_leaves_general_knowledge_plain():
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content.decode("utf-8")))
        event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Okay."}}
        return httpx.Response(
            200, content=f"data: {json.dumps(event)}\n\n".encode(), headers={"content-type": "text/event-stream"}
        )

    async def key() -> str:
        return "k"

    model = HttpFastModel(
        provider="claude",
        name="m",
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        api_key=key,
    )
    layer, _said = _layer(model)
    await layer.take("Tell me the story of how the Eiffel Tower was built.")
    system = json.dumps(sent[0]["system"])
    assert "give a short version in those few words" in system and "Never say you can't" in system
    assert "General knowledge you answer plainly: no" in system
    assert "Name a file, date, number, person or fact only when" not in system


async def test_an_act_line_after_their_yes_hands_nothing_over():
    act = 'Okay, your call.\nACT {"verb": "delete", "object": "the thesis", "request": "Delete the only copy of the thesis"}'
    model = Replies(
        "That's your only copy of the thesis, so deleting it can't be undone.\nSTANCE: only thesis copy", act
    )
    layer, said = _layer(model)
    await layer.take(PLAN)
    result = await layer.take("Yes.")
    assert result.done and result.intent is None  # their plan is theirs: nothing goes to the planner
    assert said[-1] == GIVE_WAY


async def test_your_call_cut_from_a_piece_that_goes_on_keeps_the_sentence_whole():
    reply = "Four hours in the morning before nine is tight, but it's your call, so starting tonight would give you more room."
    layer, said = _layer(Replies(reply + "\nSTANCE: report deadline"))
    await layer.take("I'm going to watch a series tonight and do the report in the morning.")
    assert "your call" not in " ".join(said) and ",." not in " ".join(said)
