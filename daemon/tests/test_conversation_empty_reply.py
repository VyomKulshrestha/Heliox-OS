"""A talk turn that ends with no words is never silent, and a reply the model spent on thinking is asked for once more.

Each provider's stream is faked in its own documented shape (Anthropic's named events, OpenRouter's chat chunks with
``delta.reasoning``, Gemini's ``alt=sse`` parts with ``thought``) over httpx's mock transport, and the real
``HttpFastModel`` and conversation layer handle it. Order is checked by counting requests and spoken lines, never by
comparing clock stamps.
"""

from __future__ import annotations

import json

import httpx
import pytest

from pilot.conversation.fast_model import (
    EMPTY_RETRY_WITHIN,
    HttpFastModel,
    RouterFastModel,
    claude_payload,
    gemini_thinking_off,
)
from pilot.conversation.layer import AGAIN_ALTERNATES, FALLBACK, FALLBACK_AGAIN, GIVE_WAY, ConversationLayer
from pilot.models.catalog import TALK_REASONING

ANSWER = "It's sunny."


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def sse(data: dict) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode()


def claude_text(text: str) -> list[bytes]:
    return [
        sse({"type": "message_start", "message": {"usage": {"input_tokens": 20, "output_tokens": 1}}}),
        sse({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        sse({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
        sse({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}),
    ]


def claude_thinking_only(max_tokens: int = 400) -> list[bytes]:
    """Thinking with its text omitted (the default display), then the token limit: no text block at all."""
    return [
        sse({"type": "message_start", "message": {"usage": {"input_tokens": 20, "output_tokens": 1}}}),
        sse({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
        sse({"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "abc"}}),
        sse({"type": "content_block_stop", "index": 0}),
        sse({"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": max_tokens}}),
    ]


def openrouter_reasoning_only() -> list[bytes]:
    chunk = {"choices": [{"delta": {"role": "assistant", "content": "", "reasoning": "Think about it."}}]}
    end = {"choices": [{"delta": {}, "finish_reason": "length"}]}
    usage = {"choices": [], "usage": {"prompt_tokens": 30, "completion_tokens": 400}}
    return [sse(chunk), sse(end), sse(usage), b"data: [DONE]\n\n"]


def openrouter_text(text: str) -> list[bytes]:
    chunk = {"choices": [{"delta": {"role": "assistant", "content": text}}]}
    usage = {"choices": [], "usage": {"prompt_tokens": 30, "completion_tokens": 6}}
    return [sse(chunk), sse(usage), b"data: [DONE]\n\n"]


def gemini_thought_only() -> list[bytes]:
    return [
        sse(
            {
                "candidates": [{"content": {"parts": [{"text": "Considering.", "thought": True}]}}],
                "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 0, "thoughtsTokenCount": 400},
            }
        )
    ]


def gemini_text(text: str) -> list[bytes]:
    return [
        sse(
            {
                "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 4},
            }
        )
    ]


class Script:
    """Answers each request with the next canned stream, keeps every request body, and can move the clock."""

    def __init__(self, streams: list[list[bytes]], clock: Clock, advance: float = 0.0) -> None:
        self.streams, self.clock, self.advance = streams, clock, advance
        self.bodies: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        self.clock.t += self.advance
        chunks = self.streams.pop(0) if self.streams else []
        return httpx.Response(200, content=b"".join(chunks), headers={"content-type": "text/event-stream"})


def model_for(provider: str, name: str, script: Script) -> HttpFastModel:
    async def key() -> str:
        return "test-key-not-real"

    return HttpFastModel(
        provider=provider,
        name=name,
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(script.handler)),
        api_key=key,
        clock=script.clock,
    )


def layer_with(model, said: list[str]) -> ConversationLayer:
    async def sink(text, channel, key):
        said.append(text)
        return "spoken"

    return ConversationLayer(sink=sink, model=lambda: model, hold_after=5.0, first_text_deadline=5.0)


# ── asked once more, then said plainly ──


async def test_a_claude_reply_spent_on_thinking_is_asked_again_with_twice_the_allowance():
    clock = Clock()
    script = Script([claude_thinking_only(), claude_text(ANSWER)], clock)
    model = model_for("claude", "claude-haiku-5-5", script)
    said: list[str] = []
    await layer_with(model, said).take("Why is the sky blue?")
    assert said == [ANSWER]
    assert [body["max_tokens"] for body in script.bodies] == [400, 800]
    assert model.calls[0].reasoned and model.calls[0].chars == 0
    assert len(model.calls) == 2  # both calls recorded on the same model, for the budget and the log


async def test_two_replies_with_no_text_end_in_one_short_honest_line():
    clock = Clock()
    script = Script([claude_thinking_only(), claude_thinking_only(800)], clock)
    said: list[str] = []
    await layer_with(model_for("claude", "claude-haiku-5-5", script), said).take("Why is the sky blue?")
    assert len(script.bodies) == 2
    assert said == [FALLBACK["empty"]]


async def test_no_retry_once_the_retry_window_has_passed():
    clock = Clock()
    script = Script([claude_thinking_only(), claude_text(ANSWER)], clock, advance=EMPTY_RETRY_WITHIN + 0.5)
    said: list[str] = []
    await layer_with(model_for("claude", "claude-haiku-5-5", script), said).take("Why is the sky blue?")
    assert len(script.bodies) == 1  # a second call could run past the first-text deadline: not asked
    assert said == [FALLBACK["empty"]]


async def test_an_openrouter_reply_that_only_reasoned_is_asked_again_with_low_effort():
    clock = Clock()
    script = Script([openrouter_reasoning_only(), openrouter_text(ANSWER)], clock)
    model = model_for("openrouter", "deepseek/deepseek-v4", script)
    said: list[str] = []
    await layer_with(model, said).take("Why is the sky blue?")
    assert said == [ANSWER]
    assert "reasoning" not in script.bodies[0]
    assert script.bodies[1]["reasoning"] == {"effort": "low"} and script.bodies[1]["max_tokens"] == 800
    assert "provider" not in script.bodies[1]  # not required of the host: a model that does not reason still answers
    assert model.calls[0].reasoned


async def test_a_gemini_reply_of_thoughts_only_is_asked_again():
    clock = Clock()
    script = Script([gemini_thought_only(), gemini_text(ANSWER)], clock)
    model = model_for("gemini", "gemini-3.5-flash-lite", script)
    said: list[str] = []
    await layer_with(model, said).take("Why is the sky blue?")
    assert said == [ANSWER]
    assert [b["generationConfig"]["maxOutputTokens"] for b in script.bodies] == [400, 800]
    assert model.calls[0].reasoned


async def test_spaces_and_line_breaks_are_not_a_first_word():
    clock = Clock()
    blank = [sse({"type": "content_block_delta", "delta": {"type": "text_delta", "text": " \n\n "}})]
    script = Script([blank, blank], clock)
    model = model_for("claude", "claude-haiku-5-5", script)
    said: list[str] = []
    await layer_with(model, said).take("Why is the sky blue?")
    assert said == [FALLBACK["empty"]]
    assert all(call.first_text_at is None for call in model.calls)  # the first-text deadline kept running


async def test_spaces_before_the_first_word_are_kept_with_it():
    clock = Clock()
    lead = [sse({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "\n"}}), *claude_text(ANSWER)]
    script = Script([lead], clock)
    said: list[str] = []
    await layer_with(model_for("claude", "claude-haiku-5-5", script), said).take("Why is the sky blue?")
    assert said == [ANSWER] and len(script.bodies) == 1


# ── the floor in the layer: no words to say is never silence ──


async def test_a_reply_that_is_only_a_stance_line_is_not_silence():
    clock = Clock()
    script = Script([claude_text("STANCE: the long lunch\n")], clock)
    said: list[str] = []
    await layer_with(model_for("claude", "claude-haiku-5-5", script), said).take("I'll take a long lunch today.")
    assert said == [FALLBACK["empty"]]


async def test_a_router_model_that_returns_nothing_is_not_silence():
    class Router:
        async def generate(self, messages, system="", temperature=0.3):
            return ""

    said: list[str] = []
    await layer_with(RouterFastModel(router=Router()), said).take("Why is the sky blue?")
    assert said == [FALLBACK["empty"]]


async def test_a_turn_that_gives_way_and_is_talked_over_is_not_an_empty_reply():
    # On a turn that gives way, a sentence still pushing back is held back and "Okay, your call." is said instead.
    # When the person talks over that line, the reply had words: nothing more is said, least of all the empty line.
    class Pushback:
        name, provider, streams, local = "stub", "claude", True, False

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, system, messages):
            self.calls += 1
            lead = "" if self.calls == 1 else "Still, "
            yield lead + "leaving at half two would be tight for the Acme meeting."

    heard: list[str] = []

    async def sink(text, channel, key):
        heard.append(text)
        return "interrupted" if text == GIVE_WAY else "spoken"

    model = Pushback()
    layer = ConversationLayer(sink=sink, model=lambda: model, hold_after=60.0, disagree=lambda: "when_it_matters")
    await layer.take("I'm going to the gym at half two today.")
    await layer.take("Do it anyway.")
    assert model.calls == 2
    assert heard[-1] == GIVE_WAY
    assert FALLBACK["empty"] not in heard and len(heard) == 2


async def test_empty_replies_in_a_row_are_said_with_short_lines_that_differ():
    clock = Clock()
    script = Script([claude_thinking_only()] * 6, clock)
    said: list[str] = []
    layer = layer_with(model_for("claude", "claude-haiku-5-5", script), said)
    for question in ("Why is the sky blue?", "And at night?", "And at dawn?"):
        await layer.take(question)
    assert said == [FALLBACK["empty"], FALLBACK_AGAIN["empty"], AGAIN_ALTERNATES["empty"]]
    assert len(script.bodies) == 6  # each turn: one call and one retry


# ── the least thinking each provider allows, in its own parameter shape ──


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-haiku-5-5", {"type": "disabled"}),
        ("claude-sonnet-5", {"type": "disabled"}),
        ("claude-sonnet-5-5", {"type": "between_tools"}),  # disabled is a 400 there
        ("claude-opus-5-5", None),  # no switch-off at all: left out, effort low
        ("claude-fable-5-2", None),
    ],
)
def test_claude_talk_asks_each_model_for_the_least_thinking_it_accepts(model, expected):
    payload = claude_payload(model, 400, "system", [{"role": "user", "content": "hi"}], "low", True)
    assert payload.get("thinking") == expected
    assert payload["output_config"] == {"effort": "low"}


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gemini-2.5-flash", {"thinkingBudget": 0}),
        ("gemini-3.5-flash-lite", {"thinkingLevel": "minimal"}),
        ("gemini-3.1-flash-lite", {"thinkingLevel": "minimal"}),
        ("gemini-3.8-flash", {"thinkingLevel": "low"}),  # minimal is not offered there
        ("gemini-4-flash", {"thinkingLevel": "low"}),
        ("gemini-flash-latest", {"thinkingLevel": "minimal"}),
        ("gemini-3.5-pro", None),
    ],
)
def test_gemini_talk_asks_each_model_for_the_least_thinking_it_accepts(model, expected):
    assert gemini_thinking_off(model) == expected


@pytest.mark.parametrize(("model", "effort"), sorted(TALK_REASONING.items()))
def test_openrouter_reasoning_models_get_their_least_effort_from_hosts_that_take_it(model, effort):
    script = Script([], Clock())
    _url, _headers, body = model_for("openrouter", model, script).request("system", [], "k")
    assert body["reasoning"] == {"effort": effort}
    assert body["provider"] == {"require_parameters": True}


def test_the_stricter_request_leaves_a_plain_openai_model_alone():
    script = Script([], Clock())
    plain = model_for("openai", "gpt-4o-mini", script).stricter()
    _url, _headers, body = plain.request("system", [], "k")
    assert "reasoning_effort" not in body and body["max_completion_tokens"] == 800
    reasoning = model_for("openai", "gpt-6-astra", script).stricter()
    assert reasoning.request("system", [], "k")[2]["reasoning_effort"] == "low"
    luna = model_for("openai", "gpt-6-luna", script).stricter()
    assert luna.request("system", [], "k")[2]["reasoning_effort"] == "none"  # its own setting stays
