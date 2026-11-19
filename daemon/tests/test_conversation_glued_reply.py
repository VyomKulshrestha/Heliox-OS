"""A second answer a host joins onto the first with no space ("Canberra.Canberra.") is left out; a repetition meant
to be heard is not.

The joined replies are the ones gpt-oss gave through Groq and Cerebras on 19 November, fed whole, split at every
point and one character at a time, then through the real HttpFastModel and conversation layer in OpenRouter's
stream shape.
"""

from __future__ import annotations

import json

import httpx
import pytest

from pilot.conversation.fast_model import GluedMessages, HttpFastModel
from pilot.conversation.layer import ConversationLayer

JOINED = [
    ("Canberra.Canberra.", "Canberra."),
    (
        "Canberra is the capital of Australia.Canberra is the capital of Australia.",
        "Canberra is the capital of Australia.",
    ),
    (
        "Set a timer for 25 minutes and focus only on one task.Set a timer for twenty-five minutes and focus only on "
        "one task.",
        "Set a timer for 25 minutes and focus only on one task.",
    ),
    (
        "Because short-wave blue light scatters more than other colors in the atmosphere.Okay.Sorry, I should not add "
        "extra.The sky appears blue due to scattering.",
        "Because short-wave blue light scatters more than other colors in the atmosphere.",
    ),
    (
        "The sky looks blue because sunlight hits the air; the short blue waves bounce more.The sky’s blue hue comes "
        "from scattering.",
        "The sky looks blue because sunlight hits the air; the short blue waves bounce more.",
    ),
]
KEPT = [
    "No, no.",
    "No. No.",
    "No! No! Not that one.",
    "Canberra. Canberra is the capital.",
    "It's about 3.5 gigabytes.",
    "The U.S. Army band.",
    "Use Node.JS for that.",
    'He said "Ash". Then he left.',
    "Wait...Then try again.",
    "Okay.\nThe next one is Thursday.",
]


def run(pieces: list[str]) -> str:
    glued = GluedMessages()
    return "".join(glued.feed(piece) for piece in pieces) + glued.finish()


def splits(text: str) -> list[list[str]]:
    return [[text], list(text), *([text[:i], text[i:]] for i in range(1, len(text)))]


@pytest.mark.parametrize(("joined", "first"), JOINED)
def test_a_second_answer_joined_with_no_space_is_left_out_wherever_the_pieces_break(joined, first):
    for pieces in splits(joined):
        assert run(pieces) == first, pieces


@pytest.mark.parametrize("text", KEPT)
def test_text_with_a_space_or_no_capital_after_the_stop_is_passed_on_unchanged(text):
    for pieces in splits(text):
        assert run(pieces) == text, pieces


def test_pieces_pass_unchanged_and_only_a_capital_straight_after_a_stop_waits():
    glued = GluedMessages()
    assert glued.feed("It is Canberra.") == "It is Canberra."  # the stop is not held
    assert glued.feed(" Anything") == " Anything"
    assert glued.feed(" else?C") == " else?"  # "C" could start a joined answer
    assert glued.feed("anberra.") == ""  # it did
    assert glued.cut and glued.finish() == ""
    other = GluedMessages()
    assert other.feed("It is Canberra.C") == "It is Canberra."
    assert other.feed("ANBERRA") == "CANBERRA" and other.finish() == ""


# ── through the model and the layer, as a person hears it ──


def openrouter_stream(pieces: list[str]) -> bytes:
    chunks = [{"choices": [{"delta": {"role": "assistant", "content": piece}}]} for piece in pieces]
    chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    chunks.append({"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 9}})
    return b"".join(f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks) + b"data: [DONE]\n\n"


def model_saying(pieces: list[str], provider: str = "openrouter", name: str = "openai/gpt-oss-120b") -> HttpFastModel:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=openrouter_stream(pieces), headers={"content-type": "text/event-stream"})

    async def key() -> str:
        return "test-key-not-real"

    return HttpFastModel(
        provider=provider,
        name=name,
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        api_key=key,
    )


async def heard(model: HttpFastModel, words: str) -> list[str]:
    said: list[str] = []

    async def sink(text, channel, key):
        said.append(text)
        return "spoken"

    await ConversationLayer(sink=sink, model=lambda: model).take(words)
    return said


async def test_a_doubled_answer_is_heard_once():
    model = model_saying(["Can", "berra", ".", "Can", "berra", "."])
    assert await heard(model, "What's the capital of Australia?") == ["Canberra."]
    assert model.calls[0].chars == len("Canberra.")


async def test_a_repetition_meant_to_be_heard_is_said_in_full():
    model = model_saying(["No, no.", " That's the old", " folder."])
    said = await heard(model, "Is that the new folder?")
    assert " ".join(said) == "No, no. That's the old folder."


async def test_the_planner_hand_over_line_is_untouched():
    act = 'ACT {"verb": "open", "object": "Notepad", "request": "Open Notepad.", "say": "I\'ll open Notepad."}'
    model = model_saying([act[:20], act[20:]])
    said = await heard(model, "I'd like Notepad open.")
    assert model.calls[0].chars == len(act)  # every character reached the layer
    assert all("ACT" not in line for line in said)
