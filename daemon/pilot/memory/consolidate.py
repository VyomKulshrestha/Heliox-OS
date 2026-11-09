"""Consolidation: the memory upkeep Heliox does while idle, in code and without a model.

Memory that only grows gets worse: the same "check the disk space" turn stored forty times takes room a useful
memory could have had, and a chatty answer from last spring outranks nothing but still has to be scanned. Every few
hours this pass:

1. deletes working memory whose time is up (it was only for one task);
2. merges episodes that say nearly the same thing (nine in ten of their words shared and the same kind of outcome),
   keeping the newest and making it a little more important each time it repeats, since repetition is a sign it
   matters;
3. lets answer-only episodes older than 90 days go. Their request and answer stay in conversation history; episodes
   of actions that ran, which carry verified outcomes, are kept.

Facts are left alone: they change only through what the person says, with evidence, never through upkeep.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from pilot.memory.assembler import memory_terms, term_similarity

logger = logging.getLogger("pilot.memory.consolidate")

DUPLICATE_SIMILARITY = 0.9
FADE_AFTER_DAYS = 90
FADING_OUTCOMES = frozenset({"answered"})
IMPORTANCE_PER_REPEAT = 0.05
MAX_IMPORTANCE = 0.95
LOOK_AT = 2000  # the newest episodes; older ones were consolidated in earlier passes


async def consolidate(temporal: Any, *, now: datetime | None = None) -> dict[str, int]:
    now = now or datetime.now(UTC)
    report = {"working_expired": await temporal.purge_expired_working(now=now), "merged": 0, "faded": 0}
    episodes = await temporal.recent_episodes(limit=LOOK_AT)  # newest first
    cutoff = (now - timedelta(days=FADE_AFTER_DAYS)).isoformat()
    faded = [e["episode_id"] for e in episodes if e["outcome"] in FADING_OUTCOMES and e["occurred_at"] < cutoff]
    gone = set(faded)
    kept: list[tuple[dict[str, Any], frozenset[str]]] = []
    merged: list[str] = []
    for episode in episodes:
        if episode["episode_id"] in gone:
            continue
        terms = memory_terms(episode["summary"])
        twin = next(
            (
                newer
                for newer, newer_terms in kept
                if newer["outcome"] == episode["outcome"]
                and term_similarity(terms, newer_terms) >= DUPLICATE_SIMILARITY
            ),
            None,
        )
        if twin is None:
            kept.append((episode, terms))
            continue
        merged.append(episode["episode_id"])
        twin["importance"] = min(MAX_IMPORTANCE, twin["importance"] + IMPORTANCE_PER_REPEAT)
        twin["_bumped"] = True
    for episode, _ in kept:
        if episode.get("_bumped"):
            await temporal.set_episode_importance(episode["episode_id"], episode["importance"])
    report["merged"] = await temporal.delete_episodes(merged)
    report["faded"] = await temporal.delete_episodes(faded)
    if any(report.values()):
        logger.info("Memory consolidated: %s", report)
    return report


__all__ = ["consolidate"]
