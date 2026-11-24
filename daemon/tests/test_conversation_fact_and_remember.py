"""Two small rules from the spoken acceptance run: a statement of fact is acknowledged, not argued with, until a
decision is made (pilot.conversation.debate.states_a_fact); and "remember that ..." that nothing kept is never answered
with an instruction to say "remember that"."""

from __future__ import annotations

import asyncio

import pytest

from pilot.config import PilotConfig
from pilot.conversation import debate as dbt
from pilot.conversation import layer_for
from pilot.conversation.layer import GOT_IT, NOT_KEPT_AS_SAID, ConversationLayer
from pilot.server import PilotServer
from pilot.system.companion_speech import SpeechChannel, SpeechOutcome
from pilot.system.voice import voice_turn_addressed
from pilot.workflows.user_profile import UserProfile

FACT = "My report is due at nine tomorrow, and it's about four hours of work."
DECISION = "I'm going to watch a series tonight and do the report in the morning."
PUSH = "Four hours of work before a nine a m deadline leaves you very little margin."


@pytest.mark.parametrize(
    ("words", "fact"),
    [
        (FACT, True),
        ("The build failed again.", True),
        (DECISION, False),
        ("I'll have pasta tonight.", False),
        ("What's the capital of Australia?", False),
        ("It'll be fine.", False),  # holding to a decision
        ("Do it anyway.", False),
    ],
)
def test_what_counts_as_a_fact(words, fact):
    from pilot.conversation.router import normalize

    assert dbt.states_a_fact(normalize(words), asked=words.endswith("?")) is fact


class Model:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.local, self.streams, self.name, self.provider = False, True, "fake", "fake"
        self.seen: list[list[dict[str, str]]] = []

    async def stream(self, system, messages):
        self.seen.append(messages)
        yield self.replies.pop(0)


async def test_a_fact_is_acknowledged_and_the_decision_after_it_may_be_argued():
    said: list[str] = []

    async def sink(text, channel, key):
        said.append(text)
        return "spoken"

    model = Model(
        PUSH + "\nSTANCE: report deadline",
        "Watching tonight squeezes four hours into the morning.\nSTANCE: report deadline",
    )
    layer = ConversationLayer(sink=sink, model=lambda: model, disagree=lambda: "when_it_matters")
    await layer.take(FACT)
    assert dbt.FACT_LINE in model.seen[0][-1]["content"]  # the request told the model it is a fact
    assert said == [GOT_IT]  # the argument the model wrote anyway is not said
    assert layer.debate.stance is None  # and it opened no disagreement
    await layer.take(DECISION)
    assert dbt.FACT_LINE not in model.seen[1][-1]["content"]
    assert said[-1].startswith("Watching tonight")  # the decision can be pushed back on, with the fact as grounds
    assert layer.debate.stance is not None


def test_remember_that_is_never_answered_with_say_remember_that():
    layer = ConversationLayer(sink=None, model=lambda: None)
    reply = layer._kept_or_changed("Remember that my sister Ana lives in Lisbon.", None)
    assert reply == NOT_KEPT_AS_SAID and 'Say "remember that"' not in reply
    assert layer._kept_or_changed("Keep in mind I like it short", None) == NOT_KEPT_AS_SAID


async def _nothing(*_args, **_kwargs):
    return None


async def test_with_the_profile_there_it_is_kept_in_code_and_read_back(tmp_path):
    server = PilotServer(PilotConfig())
    heard: list[str] = []

    async def speak(text, channel, dedupe_key="", recorder=None):
        heard.append(text)
        return SpeechOutcome("spoken", SpeechChannel(str(channel)), "")

    async def no_model_or_plan(*_args, **_kwargs):
        raise AssertionError("no model or plan for this")

    server._speech_coordinator.speak = speak
    server._broadcast_notification = _nothing
    server._handle_execute = no_model_or_plan
    server._model_router = object()
    server._user_profile = UserProfile(tmp_path / "profile.json", remember=_nothing, forget=_nothing)
    layer_for(server)._model = lambda: None
    voice_turn_addressed.set("wake")
    await server._voice_command_dispatch("Remember that my sister Ana lives in Lisbon.")
    assert any("Lisbon" in str(item.get("text")) for item in server._user_profile.get().get("about") or [])
    await server._voice_command_dispatch("Where does my sister live?")
    assert "Lisbon" in heard[-1]
    await asyncio.sleep(0)
