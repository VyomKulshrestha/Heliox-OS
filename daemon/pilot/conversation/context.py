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
    source: str = ""  # "live": Heliox Live spoke up on its own about something the person wrote or said


class ConversationMemory:
    """The short-term conversation, held only in memory."""

    def __init__(self, *, max_turns: int = MAX_TURNS, clock: Callable[[], float] = time.monotonic) -> None:
        self._turns: deque[Turn] = deque(maxlen=max_turns * 2)
        self._max_turns = max_turns
        self._clock = clock

    def add_user(self, text: str) -> None:
        self._add(Turn("user", " ".join(text.split())[:TURN_CHARS], self._clock()))

    def add_heliox(self, text: str, *, cut_off: bool = False, source: str = "") -> None:
        """Something Heliox said. ``source="live"`` for a remark Heliox Live made on its own."""
        clean = " ".join(text.split())[:TURN_CHARS]
        if clean:
            self._add(Turn("assistant", clean, self._clock(), cut_off=cut_off, source=source))

    def _add(self, turn: Turn) -> None:
        if self._turns and self._clock() - self._turns[-1].at > CONVERSATION_IDLE_SECONDS:
            self._turns.clear()
        self._turns.append(turn)

    def clear(self) -> None:
        self._turns.clear()

    def snapshot(self) -> tuple[Turn, ...]:
        """The conversation now, to go back to if what follows is withdrawn."""
        return tuple(self._turns)

    def restore(self, snapshot: tuple[Turn, ...]) -> int:
        """Back to ``snapshot``: the turns added since are dropped (a first half replaced by the whole sentence).
        Returns how many were dropped."""
        kept = [turn for turn in self._turns if any(turn is old for old in snapshot)]
        dropped = len(self._turns) - len(kept)
        self._turns.clear()
        self._turns.extend(kept)
        return dropped

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

    # The person's profile lines; a source may take the words being answered and give only what bears on them.
    profile_lines: Callable[..., list[str]] = list
    task_state: Callable[[], dict[str, Any]] = dict
    screen: Callable[[], Screen | None] = lambda: None
    # Asked once per turn with whether the model runs on this machine; it decides what may be shared.
    screen_async: Callable[[bool], Awaitable[Screen | None]] | None = None
    now: Callable[[], datetime] = datetime.now
    # Further context lines (personal memory, Live). Each takes no argument or the words being answered.
    extra: list[Callable[..., str]] = field(default_factory=list)
    # How long the person asked answers to be: "short", "detailed" or "" (the profile's answer style).
    answer_length: Callable[[], str] = lambda: ""
    # The people the person named, relation -> name ("manager" -> "Priya"), for a hand-over that says "my manager".
    people: Callable[[], dict[str, str]] = dict


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


# Never sent to a model as a fact about the person, even when something stored it: health, money, someone else's
# private details, and anything that names or looks like a secret (pilot.personal.passing, pilot.workflows.user_profile).
def never_sent(line: str) -> bool:
    try:
        from pilot.personal import passing
        from pilot.workflows.clipboard_history import looks_secret
        from pilot.workflows.user_profile import NAMES_A_SECRET
    except Exception:
        return False
    for name in ("_HEALTH", "_MONEY", "_PRIVATE_OF_OTHERS"):
        pattern = getattr(passing, name, None)
        if pattern is not None and pattern.search(line):
            return True
    return bool(NAMES_A_SECRET.search(line) or looks_secret(line))


LENGTH_LINES = {
    "short": "They asked for short answers: at most about 12 words.",
    "detailed": "They asked for more detail: up to about 50 words is fine, still spoken, no lists.",
}


def _call_extra(extra: Callable[..., Any], text: str) -> Any:
    """An extra source may take the words being answered (one argument) or nothing."""
    import inspect

    try:
        takes_text = len(inspect.signature(extra).parameters) >= 1
    except (TypeError, ValueError):
        takes_text = False
    return _safe((lambda: extra(text)) if takes_text else extra, "")


def assemble(
    sources: ContextSources,
    memory: ConversationMemory,
    *,
    budget: int = CONTEXT_CHARS,
    screen: Any = _UNSET,
    text: str = "",
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
    last = memory.last_heliox()
    opening = " ".join(last.split()[:2]) if last else ""
    if opening and memory.open:
        sections.append(f"Your last reply began {opening!r}; begin this one differently.")
    chosen = _call_extra(sources.profile_lines, text) if text else _safe(sources.profile_lines, [])
    profile = [str(line) for line in (chosen or []) if not never_sent(str(line))][:PROFILE_LINES]
    if profile:
        sections.append("About the person: " + " ".join(profile))
    length = LENGTH_LINES.get(str(_safe(sources.answer_length, "")))
    if length:
        sections.append(length)
    for extra in sources.extra:
        line = _without(_call_extra(extra, text), profile)
        if line:
            sections.append(str(line))
    block = ""
    for section in sections:
        if len(block) + len(section) + 1 > budget:
            break  # later sections (profile, screen) give way first
        block += section + "\n"

    messages: list[dict[str, str]] = []
    for turn in memory.recent():
        said = turn.text + (" [the person talked over Heliox here; the rest was not said]" if turn.cut_off else "")
        if turn.source == "live":
            said = f"[Heliox Live spoke up on its own about something the person wrote or said:] {said}"
        if messages and messages[-1]["role"] == turn.role:
            messages[-1]["content"] += " " + said
        else:
            messages.append({"role": turn.role, "content": said})
    # The Messages API starts with the person. Something Heliox said first (a Live remark) goes in the context.
    opening = []
    while messages and messages[0]["role"] != "user":
        opening.append(messages.pop(0)["content"])
    if opening:
        block = f"Heliox said before this: {' '.join(opening)[:TURN_CHARS]}\n" + block
    return block.strip(), messages


def _without(line: str, sent: list[str]) -> str:
    """An extra line ("Also relevant about the person: A. B.") without what the profile lines already said and
    without anything never sent; "" when nothing is left."""
    if not isinstance(line, str) or not line:
        return ""
    head, colon, body = line.partition(": ")
    if not colon or not head.startswith("Also relevant"):
        return "" if never_sent(line) else line
    already = {piece.rstrip(".").casefold() for piece in sent}
    pieces = [piece.strip().rstrip(".") for piece in body.split(". ") if piece.strip()]
    kept = [piece for piece in pieces if piece.casefold() not in already and not never_sent(piece)]
    return f"{head}: " + ". ".join(kept) + "." if kept else ""


def build_messages(
    text: str,
    sources: ContextSources,
    memory: ConversationMemory,
    *,
    budget: int = CONTEXT_CHARS,
    screen: Any = _UNSET,
) -> list[dict[str, str]]:
    """Messages for one fast-model call: earlier turns, then the context and the new words as the last user turn."""
    block, history = assemble(sources, memory, budget=budget, screen=screen, text=text)
    final = f"[Context, not said by the person]\n{block}\n\n[The person said]\n{text.strip()}"
    if history and history[-1]["role"] == "user":
        history[-1]["content"] += "\n\n" + final
    else:
        history.append({"role": "user", "content": final})
    return history
