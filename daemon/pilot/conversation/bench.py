"""One reply from the fast layer for a benchmark script, without audio or a running daemon.

``bench_reply`` is the entry point the Conversation Bench calls (benchmarks/conversation_bench/quality.py). It runs
the real router, context, prompt, streaming, cleaning and hand-over decision; only the talk model and the speaker
are replaced: ``stream`` is a callable that takes the prompt and yields text (a stub with chosen delays), None for
a person with no model, or "configured" for the person's configured provider (real calls, counted in the budget).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from pilot.conversation.context import ContextSources
from pilot.conversation.layer import ConversationLayer
from pilot.conversation.router import TurnKind


class _StubModel:
    """A talk model made of a callable that yields text for a prompt."""

    def __init__(self, stream: Callable[[str], AsyncIterator[str]]) -> None:
        self._stream = stream
        self.name = "bench-stub"
        self.provider = "stub"
        self.streams = True
        self.local = False
        self.calls: list[Any] = []

    async def stream(self, system: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        prompt = system + "\n\n" + "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
        async for piece in self._stream(prompt):
            yield piece


async def _configured_model() -> Any:
    from pilot.config import DB_FILE, PilotConfig
    from pilot.conversation.fast_model import build_fast_model
    from pilot.models.budget_tracker import BudgetTracker
    from pilot.security.vault import KeyVault

    config = PilotConfig.load()
    budget = BudgetTracker(config.model, str(DB_FILE))
    await budget.initialize()
    return build_fast_model(config, vault=KeyVault(config), router=None, budget=budget)


async def bench_reply(
    utterance: str,
    *,
    turns: list[dict[str, str]],
    context: dict[str, Any],
    stream: Callable[[str], AsyncIterator[str]] | str | None,
) -> dict[str, Any]:
    """{"route": "answer"|"action"|"control", "spoken": [sentences], "first_sentence_s": seconds or None}."""
    if stream == "configured":
        model: Any = await _configured_model()
    elif callable(stream):
        model = _StubModel(stream)
    else:
        model = None
    spoken: list[str] = []
    first: list[float] = []
    started = time.monotonic()

    async def sink(text: str, channel: str, key: str) -> str:
        if not first:
            first.append(time.monotonic() - started)
        spoken.append(text)
        return "spoken"

    screen = str(context.get("screen") or "")
    profile = str(context.get("profile") or "")
    sources = ContextSources(
        profile_lines=lambda: [profile] if profile else [],
        extra=[lambda: f"On their screen: {screen}" if screen else ""],
    )
    layer = ConversationLayer(sink=sink, model=lambda: model, sources=sources, hold_after=60.0)
    for turn in turns:
        if turn.get("who") == "heliox":
            layer.memory.add_heliox(str(turn.get("text") or ""))
        else:
            layer.memory.add_user(str(turn.get("text") or ""))
    result = await layer.take(utterance)
    if result is None:
        return {"route": "answer", "spoken": [], "first_sentence_s": None}
    if result.route is not None and result.route.kind is TurnKind.CONTROL:
        route = "control"
    else:
        route = "answer" if result.done else "action"
    if route == "action" and not spoken:
        # A hand-over: what the person hears at once is the acknowledgement keep_company would say.
        await sink(layer.acknowledgement_for(result), "task_narration", "bench-ack")
    return {"route": route, "spoken": list(spoken), "first_sentence_s": first[0] if first else None}
