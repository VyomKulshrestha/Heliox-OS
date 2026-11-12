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
    # Told every part's state whenever one changes; the server sends it to the window as a ``late_parts``
    # notification, so a panel that was told "starting" asks again once its part is ready.
    on_change: Callable[[dict[str, str]], None] | None = None

    def _changed(self) -> None:
        if self.on_change is not None:
            try:
                self.on_change(self.snapshot())
            except Exception:
                logger.debug("Could not announce the late parts' states", exc_info=True)

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
        self._changed()

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
                self._changed()
                continue
            part.state = "starting"
            self._changed()
            try:
                await part.start()
            except asyncio.CancelledError:
                part.state = "waiting"
                raise
            except Exception as error:
                part.state, part.reason = "failed", f"{type(error).__name__}: {str(error)[:120]}"
                logger.warning("%s did not start after ready", name, exc_info=True)
                self._changed()
                continue
            part.state = "ready"
            self._changed()
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
        "neural",
        "Neural input",
        f"Neural input is still starting, so a headset cannot connect or be armed yet. {AGAIN}",
        (
            "neural_status",
            "neural_stage_task",
            "neural_remove_staged_task",
            "neural_connect",
            "neural_begin_calibration",
            "neural_finish_calibration",
            "neural_arm",
            "neural_intent_preview",
            "neural_observation",
            "neural_commit",
            "neural_disarm",
            "neural_stimulus_marker",
            "neural_stimulus_markers",
        ),
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


# How long the risk gate holds a decision for the learner, from the moment it is asked, before it adds the
# approval the learner could have added.
LEARNER_WAIT_SECONDS = 5.0
WAITING_LINE = "One moment, Heliox is still starting."


class LearnerLoading:
    """Stands in for the online learner until it has loaded after ready, without importing its model library.

    Readers ask ``is_ready`` first. The risk gate then holds its decision for the learner, bounded
    (``wait_until_ready``), and only if it is still loading adds the approval the learner could have added: absent
    means more careful, never neutral. Proactive suggestions wait. It never claims to know anything.
    """

    is_ready = False

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        import threading
        import time

        self._ready = threading.Event()
        self._clock = clock or time.monotonic
        self.expected_ready_at: float | None = None  # monotonic; set once the learner knows what it must read
        self.on_wait: Callable[[], None] | None = None  # says WAITING_LINE once; set by the server
        self._said = False

    def status(self) -> dict[str, Any]:
        return {"enabled": True, "loading": True, "suggestions": None, "transitions": None}

    def snapshot_due(self, every: int = 0) -> bool:
        return False

    def expect(self, seconds: float) -> None:
        """The learner's estimate of how long it still needs (a saved state is quick; a full replay is not)."""
        self.expected_ready_at = self._clock() + max(0.0, seconds)

    def set_ready(self) -> None:
        self._ready.set()

    def wait_until_ready(self, asked_at: float, window: float | None = None) -> bool:
        """Hold a decision for the learner: at most ``window`` from ``asked_at``, and not at all when the learner
        will not be ready by then. Only from a worker thread: on the event loop's own thread waiting would hold up
        the loading it waits for. True when the learner is ready."""
        if self._ready.is_set():
            return True
        try:
            asyncio.get_running_loop()
            return False  # on the loop thread: never block it
        except RuntimeError:
            pass
        deadline = asked_at + (LEARNER_WAIT_SECONDS if window is None else window)
        if self.expected_ready_at is not None:
            if self.expected_ready_at > deadline:
                return False  # it will not make it: ask for approval now rather than after a pointless wait
            deadline = min(deadline, self.expected_ready_at + 1.0)
        remaining = deadline - self._clock()
        if remaining <= 0:
            return False
        if not self._said and self.on_wait is not None:
            self._said = True
            try:
                self.on_wait()
            except Exception:
                logger.debug("Could not say the waiting line", exc_info=True)
        return self._ready.wait(remaining)


def heliox_parts(**options: Any) -> LateStarts:
    """The parts above, all waiting; the server gives each its start while it builds them."""
    starts = LateStarts(**options)
    for name, label, words, methods in HELIOX_PARTS:
        starts.add(name, label, words, methods)
    return starts
