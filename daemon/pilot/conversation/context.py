"""What the fast model is told besides the person's words, within a small budget.

Sources, in the order they are dropped when the budget is tight (last first):

1. the last few turns of this conversation (kept here, cleared by "new conversation");
2. what Heliox is doing or has just done (a running task, a pending approval, the last result);
3. the time and day;
4. the person's profile lines (name, preferences, things they asked to be remembered);
5. what is in front of them: by default only the focused app's display name. The text of the focused field is
   added only when ``conversation.share_screen_text`` is on, and only through Heliox Live's own gates (never a
   password field, Heliox's own window, a password manager or an app excluded from the timeline), and never when
   it looks like it holds a key or password.

Only addressed turns are kept: speech Heliox overheard never reaches this module.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

MAX_TURNS = 6
TURN_CHARS = 320
CONTEXT_CHARS = 1400  # about 350 tokens of context on top of the system prompt and the turns
PROFILE_LINES = 8
SCREEN_CHARS = 400
CONVERSATION_IDLE_SECONDS = 600.0  # after ten quiet minutes, earlier turns no longer frame a new question


@dataclass(slots=True)
class Turn:
    role: str  # "user" or "assistant"
    text: str
    at: float
    cut_off: bool = False  # Heliox was talked over: ``text`` is what was heard, and no more


class ConversationMemory:
    """The short-term conversation, held only in memory."""

    def __init__(self, *, max_turns: int = MAX_TURNS, clock: Callable[[], float] = time.monotonic) -> None:
        self._turns: deque[Turn] = deque(maxlen=max_turns * 2)
        self._max_turns = max_turns
        self._clock = clock

    def add_user(self, text: str) -> None:
        self._add(Turn("user", " ".join(text.split())[:TURN_CHARS], self._clock()))

    def add_heliox(self, text: str, *, cut_off: bool = False) -> None:
        clean = " ".join(text.split())[:TURN_CHARS]
        if clean:
            self._add(Turn("assistant", clean, self._clock(), cut_off=cut_off))

    def _add(self, turn: Turn) -> None:
        if self._turns and self._clock() - self._turns[-1].at > CONVERSATION_IDLE_SECONDS:
            self._turns.clear()
        self._turns.append(turn)

    def clear(self) -> None:
        self._turns.clear()

    @property
    def open(self) -> bool:
        return bool(self._turns) and self._clock() - self._turns[-1].at <= CONVERSATION_IDLE_SECONDS

    def recent(self) -> list[Turn]:
        if not self.open:
            return []
        return list(self._turns)[-self._max_turns * 2 :]

    def last_heliox(self) -> str:
        for turn in reversed(self._turns):
            if turn.role == "assistant":
                return turn.text
        return ""


@dataclass(slots=True)
class Screen:
    app: str = ""  # display name only ("VS Code"); never the window title
    excerpt: str = ""  # only with share_screen_text on, already gated


@dataclass(slots=True)
class ContextSources:
    """Callables the layer reads at the start of each turn. Each may fail; a failing source is left out."""

    profile_lines: Callable[[], list[str]] = list
    task_state: Callable[[], dict[str, Any]] = dict
    screen: Callable[[], Screen | None] = lambda: None
    # Asked once per turn with whether the model runs on this machine; it decides what may be shared.
    screen_async: Callable[[bool], Awaitable[Screen | None]] | None = None
    now: Callable[[], datetime] = datetime.now
    extra: list[Callable[[], str]] = field(default_factory=list)


def _safe(call: Callable[[], Any], default: Any) -> Any:
    try:
        value = call()
    except Exception:
        return default
    return default if value is None else value


def describe_task(state: dict[str, Any]) -> list[str]:
    lines = []
    if state.get("approval_waiting"):
        lines.append(
            "An approval is waiting on the screen"
            + (f" for: {state['approval_waiting']}" if isinstance(state["approval_waiting"], str) else "")
            + ". Voice cannot approve; the person presses Approve there."
        )
    if state.get("running"):
        lines.append(
            f"Heliox is working on: {state['running']}" + (f" ({state['phase']})" if state.get("phase") else "")
        )
    if state.get("last_result"):
        lines.append(f"Heliox's last task result: {state['last_result']}")
    return lines


_UNSET: Any = object()


def assemble(
    sources: ContextSources, memory: ConversationMemory, *, budget: int = CONTEXT_CHARS, screen: Any = _UNSET
) -> tuple[str, list[dict[str, str]]]:
    """The context block and the earlier turns as Messages API turns (alternating, starting with the person)."""
    now = _safe(sources.now, datetime.now())
    sections: list[str] = [f"Now: {now.strftime('%A %d %B %Y, %H:%M')}."]
    task = _safe(sources.task_state, {})
    sections += describe_task(task) if isinstance(task, dict) else []
    if screen is _UNSET:
        screen = _safe(sources.screen, None)
    if isinstance(screen, Screen) and screen.app:
        sections.append(f"The person is in {screen.app}.")
        if screen.excerpt:
            sections.append(f"End of the text they are writing there: {screen.excerpt!r}")
    profile = [str(line) for line in _safe(sources.profile_lines, [])][:PROFILE_LINES]
    if profile:
        sections.append("About the person: " + " ".join(profile))
    for extra in sources.extra:
        line = _safe(extra, "")
        if line:
            sections.append(str(line))
    block = ""
    for section in sections:
        if len(block) + len(section) + 1 > budget:
            break  # later sections (profile, screen) give way first
        block += section + "\n"

    messages: list[dict[str, str]] = []
    for turn in memory.recent():
        text = turn.text + (" [the person talked over Heliox here; the rest was not said]" if turn.cut_off else "")
        if messages and messages[-1]["role"] == turn.role:
            messages[-1]["content"] += " " + text
        else:
            messages.append({"role": turn.role, "content": text})
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    return block.strip(), messages


def build_messages(
    text: str,
    sources: ContextSources,
    memory: ConversationMemory,
    *,
    budget: int = CONTEXT_CHARS,
    screen: Any = _UNSET,
) -> list[dict[str, str]]:
    """Messages for one fast-model call: earlier turns, then the context and the new words as the last user turn."""
    block, history = assemble(sources, memory, budget=budget, screen=screen)
    final = f"[Context, not said by the person]\n{block}\n\n[The person said]\n{text.strip()}"
    if history and history[-1]["role"] == "user":
        history[-1]["content"] += "\n\n" + final
    else:
        history.append({"role": "user", "content": final})
    return history
