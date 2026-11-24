"""What a bench run puts in place of the person's own computer before the server is built.

- The foreground application. The conversation layer tells the talk model "The person is in <app>." from the focused
  window (pilot.conversation.adapters.FocusSource), and nothing stubbed it in a bench run, so the display name of
  whatever app was in front on the machine running the bench went into the prompts. A run now reads a made-up name
  (BENCH_APP) unless it asks for the real one. Only a name was ever read: no window title and no screen text.
- The stores the server's own code handlers answer from (to-dos, reminders, the profile). The server builds them in
  ``start()``, which a bench never runs, so "add ... to my to-do list", "remind me ..." and "remember that ..." fell
  through to the planner or the conversation layer. They are made in the run's own scratch folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

BENCH_APP = "Quillpad"  # made up: no real application has this name


def stub_foreground(*, real: bool = False) -> str:
    """The app name the conversation layer reads from now on: BENCH_APP, or "real" when the run asked for it."""
    if real:
        return "real"
    from pilot.conversation import adapters

    adapters._active_window = lambda: (BENCH_APP, "")
    return BENCH_APP


def give_stores(server: Any, data_dir: Path) -> list[str]:
    """To-dos, reminders and the profile in ``data_dir`` (the run's scratch folder), for the ones the server lacks.
    The reminder store is not started: nothing comes due during a run. Returns the names made."""
    made = []
    data_dir = Path(data_dir)
    if getattr(server, "_todos", None) is None:
        from pilot.workflows.todos import TodoStore

        server._todos = TodoStore(data_dir / "todos.json")
        made.append("todos")
    if getattr(server, "_reminders", None) is None:
        from pilot.workflows.reminders import ReminderStore

        server._reminders = ReminderStore(data_dir / "reminders.json")
        made.append("reminders")
    if getattr(server, "_user_profile", None) is None:
        from pilot.workflows.user_profile import UserProfile

        async def nothing(*_args: Any, **_kwargs: Any) -> None:
            return None

        server._user_profile = UserProfile(data_dir / "user_profile.json", remember=nothing, forget=nothing)
        made.append("profile")
    return made
