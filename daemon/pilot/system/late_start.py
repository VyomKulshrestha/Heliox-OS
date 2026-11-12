"""Parts of Heliox that no first request needs start after it is ready, one after another, with a pause between.

Start-up used to open every part before answering anything: the agent mesh's history, strategy evolution and the
evolution harness, the prompt improver, the subconscious agent, screen vision and proactive suggestions. None of
them is needed to answer the first request. They now start after ``ready``, in that order, with a short pause between
each so the first minute stays responsive, and the online learner loads before all of them (it decides approvals).

While a part is still starting, a request that needs it is answered with one plain sentence (``status:
"starting"``), never an exception; if a part fails to start, requests that need it say so and Heliox carries on
without it, as before.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("pilot.system.late_start")

PAUSE_BETWEEN_SECONDS = 2.0


@dataclass
class LatePart:
    name: str
    label: str  # "Strategy evolution"
    words: str  # what a request that needs it is told while it starts
    methods: frozenset[str]
    start: Callable[[], Awaitable[Any]] | None = None
    state: str = "waiting"  # waiting -> starting -> ready | failed
    reason: str = ""


@dataclass
class LateStarts:
    pause: float = PAUSE_BETWEEN_SECONDS
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep
    parts: dict[str, LatePart] = field(default_factory=dict)

    def add(
        self,
        name: str,
        label: str,
        words: str,
        methods: Iterable[str] = (),
        start: Callable[[], Awaitable[Any]] | None = None,
    ) -> None:
        """Register a part. ``start`` may be given later (``set_start``) or never (the part is then ready at once)."""
        self.parts[name] = LatePart(name, label, words, frozenset(methods), start)

    def set_start(self, name: str, start: Callable[[], Awaitable[Any]]) -> None:
        self.parts[name].start = start

    @staticmethod
    def _managed(part: LatePart) -> bool:
        """A part counts as starting only once the server has given it a start (or marked it): one that was never
        built here (a server that did not run initialize, as in many tests) is not held back."""
        return part.state != "waiting" or part.start is not None

    def ready(self, name: str) -> bool:
        part = self.parts.get(name)
        return part is None or part.state == "ready" or not self._managed(part)

    def mark(self, name: str, state: str, reason: str = "") -> None:
        part = self.parts[name]
        part.state, part.reason = state, reason

    def waiting_reply(self, method: str) -> dict[str, Any] | None:
        """For a request that needs a part which has not started (yet): the plain answer. None: go ahead."""
        for part in self.parts.values():
            if method in part.methods and part.state != "ready" and self._managed(part):
                if part.state == "failed":
                    return {
                        "status": "error",
                        "message": f"{part.label} could not start ({part.reason}). "
                        "Heliox works without it; restarting Heliox tries again.",
                    }
                return {"status": "starting", "message": part.words}
        return None

    async def run(self, names: Iterable[str] | None = None) -> None:
        """Start the parts in order, pausing between; one that fails is reported and the rest still start."""
        first = True
        for name in list(names) if names is not None else list(self.parts):
            part = self.parts[name]
            if part.state != "waiting":
                continue
            if not first:
                await self.sleep(self.pause)
            first = False
            if part.start is None:
                part.state = "ready"
                continue
            part.state = "starting"
            try:
                await part.start()
            except asyncio.CancelledError:
                part.state = "waiting"
                raise
            except Exception as error:
                part.state, part.reason = "failed", f"{type(error).__name__}: {str(error)[:120]}"
                logger.warning("%s did not start after ready", name, exc_info=True)
                continue
            part.state = "ready"
            logger.info("Started after ready: %s", name)

    def snapshot(self) -> dict[str, str]:
        return {name: part.state for name, part in self.parts.items()}


AGAIN = "Ask again in a moment."

# (name, label, what a request that needs it gets while it starts, the RPCs that need it), in start order.
HELIOX_PARTS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "online_learning",
        "Learning from past outcomes",
        "Heliox is still loading what it has learned from past outcomes; until it has, it asks for approval a "
        f"little more often. {AGAIN}",
        ("online_learning_status", "online_learning_reset"),
    ),
    (
        "agent_mesh",
        "The agent mesh",
        f"The agent mesh is still loading how its agents did before; requests are routed meanwhile. {AGAIN}",
        ("agent_mesh_status",),
    ),
    (
        "prompt_improver",
        "The prompt improver",
        f"The prompt improver is still starting. {AGAIN}",
        ("prompt_stats", "prompt_strategies"),
    ),
    (
        "strategy_evolution",
        "Strategy evolution",
        f"Strategy evolution is still starting; until it has, plans use the standard planner instructions. {AGAIN}",
        (
            "strategy_evolution_status",
            "strategy_candidates",
            "strategy_propose",
            "strategy_reflect",
            "strategy_start_shadow",
            "strategy_record_isolated",
            "strategy_record_shadow",
            "strategy_start_canary",
            "strategy_record_canary",
            "strategy_promote",
            "strategy_rollback",
        ),
    ),
    (
        "evolution_harness",
        "The evolution harness",
        f"The evolution harness is still starting. {AGAIN}",
        (
            "evolution_status",
            "evolution_runs",
            "evolution_candidates",
            "evolution_create_run",
            "evolution_generate_candidates",
            "evolution_evaluate",
            "evolution_request_promotion",
        ),
    ),
    (
        "subconscious",
        "Long-term preferences",
        f"Heliox's long-term preferences are still loading; requests until then are planned without them. {AGAIN}",
        ("persona_rules", "persona_consolidate", "persona_add_preference", "subconscious_stats"),
    ),
    (
        "screen_vision",
        "Screen awareness",
        f"Screen awareness is still starting, so Heliox cannot say what is on screen yet. {AGAIN}",
        ("screen_context", "screen_current_app", "screen_vision_stats", "screen_vision_toggle"),
    ),
    (
        "proactive",
        "Proactive suggestions",
        f"Proactive suggestions are still starting. {AGAIN}",
        (
            "proactive_start",
            "proactive_stop",
            "proactive_stats",
            "proactive_accept",
            "proactive_dismiss",
            "proactive_learning_status",
            "proactive_learning_reset",
        ),
    ),
)


class LearnerLoading:
    """Stands in for the online learner until it has loaded after ready, without importing its model library.

    Readers ask ``is_ready`` first: the risk gate then adds the approval the learner could have added (absent means
    more careful, never neutral) and proactive suggestions wait. It never claims to know anything.
    """

    is_ready = False

    def status(self) -> dict[str, Any]:
        return {"enabled": True, "loading": True, "suggestions": None, "transitions": None}

    def snapshot_due(self, every: int = 0) -> bool:
        return False


def heliox_parts(**options: Any) -> LateStarts:
    """The parts above, all waiting; the server gives each its start while it builds them."""
    starts = LateStarts(**options)
    for name, label, words, methods in HELIOX_PARTS:
        starts.add(name, label, words, methods)
    return starts
