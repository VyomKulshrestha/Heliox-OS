"""Saying a known person back by name: "to my manager" read back as "to your manager, Priya".

What is stored stays the person's own words ("send the deck to my manager"): they may change who their manager is,
and a reminder set for next month follows the profile as it is then. What Heliox says and shows resolves the person
at the moment it speaks, from the profile's people (UserProfile.people_by_relation: a relation with exactly one
name). It keeps the person's words when the profile has two people for that relation (people_by_relation leaves it
out), or when that person was told within the last FRESH_SECONDS (the same exchange, possibly misheard).

Where it is used: everything Heliox says about things to do (pilot.server): a reminder's confirmation and the
reminder when it comes due, the to-do list's confirmations and read-back, and the reminders and to-dos the morning,
night and tomorrow summaries read out. Never where the person's own words are given back to them: notes, "what did
I tell you about ...", goal titles and updates, and the lists in Settings show and say what they said.
"""

from __future__ import annotations

import re
from typing import Any

FRESH_SECONDS = 10.0


def known_people(profile: Any) -> dict[str, str]:
    """Relation -> name for the people Heliox may name back (one person for the relation, not just told)."""
    if profile is None:
        return {}
    try:
        people = dict(profile.people_by_relation())
        fresh = set(profile.relations_told_within(FRESH_SECONDS))
    except Exception:
        return {}
    return {relation: name for relation, name in people.items() if relation and name and relation not in fresh}


def said_with_people(text: str, profile: Any) -> str:
    """The one call for every place Heliox reads the person's words about people back: ``text`` with each known
    person named (resolve_people over known_people), as the profile is at this moment."""
    people = known_people(profile)
    return resolve_people(text, people) if people else text


def resolve_people(text: str, people: dict[str, str]) -> str:
    """ "send the deck to my manager" -> "send the deck to your manager, Priya" for each known relation; the text
    unchanged when no known relation is named, or when the name is already there."""
    out = text
    for relation, name in sorted(people.items(), key=lambda item: -len(item[0])):
        if name.casefold() in out.casefold():
            continue
        pattern = re.compile(rf"\b(?:my|your) {re.escape(relation)}\b(?![,'’])", re.IGNORECASE)
        out = pattern.sub(f"your {relation}, {name}", out, count=1)
    return out


__all__ = ["FRESH_SECONDS", "known_people", "resolve_people", "said_with_people"]
