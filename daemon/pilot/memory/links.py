"""Links between the people the person mentioned and what Heliox remembers about them.

Memory is searched by the words of the request. "What did my manager decide about the hiring?" shares no word with
the episode "Rahul approved hiring two bakers", so a plain search misses it, although Heliox knows Rahul is the
manager. This module follows that one link: a relation word in the request ("manager", "sister", "landlord") is
joined by the name the person gave for it, so the search finds what is remembered about that person.

It is a small, explicit graph (relation to person, person to everything that names them) built from the profile
the person wrote, not from guesses, and it only adds words to the search; it never adds a memory by itself.
"""

from __future__ import annotations

import re
from collections.abc import Callable


def expand_query(query: str, people: dict[str, str]) -> str:
    """The query, plus the name behind each relation it mentions ("my manager" -> "Rahul")."""
    added = []
    for relation, name in people.items():
        if not relation or not name:
            continue
        if re.search(rf"\b{re.escape(relation)}s?\b", query, re.IGNORECASE) and name.casefold() not in query.casefold():
            added.append(name)
    return f"{query} {' '.join(added)}" if added else query


class PeopleLinks:
    """Reads relation -> name from the profile each time, so a changed manager is followed at once."""

    def __init__(self, people: Callable[[], dict[str, str]]) -> None:
        self._people = people

    def expand(self, query: str) -> str:
        try:
            return expand_query(query, self._people())
        except Exception:
            return query


__all__ = ["PeopleLinks", "expand_query"]
