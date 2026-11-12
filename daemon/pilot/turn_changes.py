"""What a voice turn changed outside the conversation, recorded by the code that changed it.

A handler that answers in code may keep something: a reminder, a to-do, a note, a setting, a habit logged, a focus
session started, an offer silenced, the clipboard written, a preference remembered. When the listener splits a
sentence at a pause and asks to withdraw the first half (pilot.conversation.turns), whether it can is read from this
record, never guessed from counting a few stores: anything recorded here makes it too late, and Heliox says what it
did. Each place that changes something calls ``note_change`` once, with what it did in a few words ("set the
reminder"); outside a voice turn the call does nothing.

Nothing is imported here, so any store can call it without pulling in the voice or the server.
"""

from __future__ import annotations

from contextvars import ContextVar

# The current voice turn's list of changes (set by pilot.conversation.turns.TurnLog.begin). asyncio copies it into
# every task and thread the turn starts (asyncio.to_thread copies the context), so a store saving in a worker thread
# still records against the turn that asked for it.
turn_changes: ContextVar[list[str] | None] = ContextVar("turn_changes", default=None)


def note_change(what: str) -> None:
    """Record that this turn changed something outside the conversation ("set the reminder"). No turn: nothing."""
    changes = turn_changes.get()
    if changes is not None and what and what not in changes:
        changes.append(what)
