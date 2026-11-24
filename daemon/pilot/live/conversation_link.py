"""Heliox Live as part of the conversation: what it says is a Heliox turn, and the fast model knows about it.

Before this, a Live remark lived only in Live: the conversation layer's memory did not have it, so "what do you
mean?" phrased any way Live's own handlers did not expect reached the fast model with no idea what Heliox had just
said, and a card on screen was invisible to it. Now:

* a spoken Live remark is recorded in ``ConversationMemory`` as a Heliox turn (cut off when the person talked over
  it), so the follow-up is part of one conversation;
* a context line (``ContextSources.extra``) tells the fast model what Live raised in the last 90 seconds, aloud or
  as a card, when that is not already in the turns it gets. Live's own replies ("why", "thanks", "not now", "don't
  mention these", "fix it") stay local and answer first.

What reaches a model follows the conversation layer's screen rule (``conversation.share_screen_text``). A remark
about what the person *said* to Heliox quotes addressed speech, which that model already gets, so it is shared as
said. A remark about what they *typed* quotes text from their screen: it is shared word for word only with a model
on this machine (``auto``) or when they turned sharing on (``on``). Otherwise the model gets only the kind of catch
and the app ("Heliox Live pointed out a weekday that doesn't match its date in Notepad"), and the details stay
here; "why?" is answered locally from the evidence.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from pilot.live.detectors import KIND_LABELS

logger = logging.getLogger("pilot.live.conversation_link")

REPLY_WINDOW_SECONDS = 90.0
MAX_LINE_CHARS = 420


def shareable(entry: dict[str, Any], *, setting: str, model_local: bool) -> bool:
    """Whether this catch's words may go to the configured conversation model. Pure."""
    if entry.get("source") == "speech":
        return True
    return setting == "on" or (setting == "auto" and model_local)


def redacted(entry: dict[str, Any]) -> str:
    label = KIND_LABELS.get(str(entry.get("kind", "")), "something worth a look").lower()
    app = str(entry.get("app") or "")
    where = f" in {app}" if app else ""
    return f"Heliox Live pointed out {label}{where}; the details stay on this computer."


def remark_for_model(text: str, entries: list[dict[str, Any]], *, setting: str, model_local: bool) -> str:
    """The remark as the model may see it: as said, or by kind when it quotes screen text it may not get."""
    if entries and not all(shareable(e, setting=setting, model_local=model_local) for e in entries):
        return " ".join(dict.fromkeys(redacted(e) for e in entries))
    return text


class LiveConversationLink:
    """Glue between ``LiveCompanion`` and the conversation layer, each reached through a getter that may give None."""

    def __init__(
        self,
        *,
        live: Callable[[], Any],
        layer: Callable[[], Any],
        setting: Callable[[], str],
        model_local: Callable[[], bool],
        clock: Callable[[], float],
    ) -> None:
        self._live = live
        self._layer = layer
        self._setting = setting
        self._model_local = model_local
        self._clock = clock
        self._recorded: set[str] = set()  # correction ids already a Heliox turn
        self._cleared_at = float("-inf")  # the last "new conversation": anything Live raised before it is not given

    def _privacy(self) -> tuple[str, bool]:
        try:
            setting = str(self._setting() or "auto")
        except Exception:
            setting = "auto"
        try:
            local = bool(self._model_local())
        except Exception:
            local = False
        return setting, local

    def record_spoken(self, text: str, outcome: str) -> bool:
        """Live said ``text`` aloud (outcome "spoken" or "interrupted"): add it to the conversation as Heliox's turn."""
        if outcome not in ("spoken", "interrupted") or not text.strip():
            return False
        live, layer = self._live(), self._layer()
        memory = getattr(layer, "memory", None)
        if memory is None:
            return False
        entries = list(live.last_spoken_entries()) if live is not None else []
        setting, local = self._privacy()
        said = remark_for_model(text, entries, setting=setting, model_local=local)
        try:
            # Marked as Live's own remark, so the model explains it from the evidence instead of taking the blame.
            memory.add_heliox(said, cut_off=outcome == "interrupted", source="live")
        except TypeError:  # a ConversationMemory without ``source``
            memory.add_heliox(said, cut_off=outcome == "interrupted")
        self._recorded.update(str(e.get("id", "")) for e in entries)
        return True

    def new_conversation(self) -> None:
        """The "new conversation" call: what Live raised before it belongs to the conversation that ended."""
        self._cleared_at = self._clock()
        self._recorded.clear()

    def context_line(self, text: str = "") -> str:
        """For ``ContextSources.extra``: what Live raised lately in this conversation, when the turns the model gets do
        not show it and the person's words bear on it (``bears_on``; no words known: as before)."""
        live = self._live()
        if live is None:
            return ""
        entry = live.last_raised()
        if not entry:
            return ""
        raised = float(entry.get("at", 0.0))
        age = self._clock() - raised
        if age > REPLY_WINDOW_SECONDS or raised <= self._cleared_at:
            return ""
        if text and not bears_on(text, entry):
            return ""  # the person is talking about something else: Live's card is not brought up
        layer = self._layer()
        memory = getattr(layer, "memory", None)
        if entry.get("id") in self._recorded and memory is not None and _shown_in_turns(memory):
            return ""  # already a Heliox turn the model will read
        setting, local = self._privacy()
        how = "said aloud" if entry.get("spoken") else "showed as a card on screen"
        if shareable(entry, setting=setting, model_local=local):
            from pilot.live.engine import because

            title = str(entry.get("title") or "").rstrip(".")
            body = f"{title}. Why: {because(entry)}"
        else:
            body = redacted(entry)
        line = (
            f"Heliox Live, on its own, {how} {int(age)} seconds ago: {body} If the person refers to it, this is what "
            "they mean. It is there already: never repeat it or offer it again."
        )
        return line[:MAX_LINE_CHARS]


# A turn that refers to what Live just raised without naming it: a short question or answer ("why?", "what was
# that?", "yes", "not now"), or words about a card, a reminder or what Heliox said.
_REFERS = re.compile(
    r"^(?:why|what|huh|sorry|really|yes|yeah|yep|sure|ok|okay|no|nope|not now|thanks|thank you|fix it|yes please"
    r"|no thanks|why not|why that|why is that|why did you say that|what do you mean|what was that)$"
)
_ABOUT_IT = re.compile(
    r"\b(?:card|reminder|remind|you (?:just )?(?:said|showed|mean|meant|offered|asked|pointed)|what was that)\b"
)
_SMALL = frozenset(
    {
        "that", "this", "with", "from", "have", "will", "what", "when", "your", "about", "there", "their", "they",
        "them", "then", "than", "been", "were", "would", "could", "should", "going", "want", "need", "just", "also",
        "into", "over", "some", "more", "most", "very", "much", "make", "made", "like", "said", "says", "today",
        "tonight", "tomorrow",
    }
)  # fmt: skip


def _content(text: str) -> set[str]:
    return {
        w.rstrip("s")
        for w in re.findall(r"[a-z0-9']+", text.casefold())
        if len(w) >= 4 and "'" not in w and w not in _SMALL
    }


def bears_on(text: str, entry: dict[str, Any]) -> bool:
    """Whether the person's words are about what Live raised: they share a word of four letters or more with it, or
    they refer to it ("why?", "yes", "what reminder?"). Pure."""
    words = " ".join(re.sub(r"[^\w\s']", " ", text.casefold()).split())
    if not words:
        return True
    if _REFERS.match(words) or _ABOUT_IT.search(words):
        return True
    raised = " ".join(str(entry.get(key) or "") for key in ("title", "message", "evidence", "detail", "fix"))
    return bool(_content(words) & _content(raised))


def _layer_keeps_opening_remarks() -> bool:
    """Whether the conversation layer passes a Heliox turn that opens the conversation on to the model (newer
    layers put it in the context block as "Heliox said before this"; older ones dropped it)."""
    import inspect

    try:
        from pilot.conversation import context
    except ImportError:
        return False
    try:
        return "Heliox said before this" in inspect.getsource(context)
    except (OSError, TypeError):
        return False


def _shown_in_turns(memory: Any) -> bool:
    """The latest Heliox turn will reach the model."""
    try:
        turns = memory.recent()
    except Exception:
        return False
    if not turns or turns[-1].role != "assistant":
        return False
    return any(turn.role == "user" for turn in turns) or _layer_keeps_opening_remarks()


__all__ = ["LiveConversationLink", "bears_on", "redacted", "remark_for_model", "shareable"]
