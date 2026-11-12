"""Bring the conversation layer up in the bench's server, with a stubbed fast model per kind of model.

The server runs the conversation layer only when its model stack is up (``_conversation_ready()``: a model router
exists). The bench never starts the real stack, which would build real model clients; instead it marks the stack
present and points the layer's model at a stub that streams a plain canned reply with the chosen kind's
first-token and token-rate delays (``models.KINDS``), or returns None for a person with no model, which is what
the real builder returns when nothing is set up. No model is ever called.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from .models import ModelKind

CANNED_REPLY = "Sure. Your next meeting is the launch call at ten, then the design review at two."


class StubFastModel:
    def __init__(self, kind: ModelKind, text: str = CANNED_REPLY) -> None:
        self.kind = kind
        self.text = text
        self.name = f"bench-{kind.name}"
        self.provider = "bench"
        self.streams = True
        self.local = kind.name == "local"
        self.calls: list[Any] = []

    async def warm(self) -> None:
        return None

    async def stream(self, system: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        self.calls.append({"system_chars": len(system), "messages": len(messages)})
        await asyncio.sleep(self.kind.talk_first_token_s)
        for word in self.text.split(" "):
            yield word + " "
            await asyncio.sleep(1.0 / self.kind.talk_tokens_per_s)


class _BenchRouter:
    """Stands for the model stack being up; anything that reaches for a real model through it is refused."""

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(f"the bench has no real model router ({name})")


def enable(harness: Any) -> dict[str, StubFastModel]:
    """Call after ``Harness.build()``: the conversation layer is active and its fast model follows ``harness.model``."""
    from pilot.conversation.adapters import layer_for

    server = harness.server
    server._model_router = _BenchRouter()
    models: dict[str, StubFastModel] = {}

    def model() -> Any:
        kind: ModelKind = harness.model
        if not kind.available:
            return None  # what the real builder returns when nothing is set up
        return models.setdefault(kind.name, StubFastModel(kind))

    layer_for(server)._model = model
    return models  # their ``calls`` show the layer really answered


def select(harness: Any, kind: ModelKind) -> None:
    """Switch the kind of model; the layer asks for its model on every turn, so the next turn uses this one."""
    harness.model = kind
