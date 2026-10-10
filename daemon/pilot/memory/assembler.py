"""Strictly bounded context assembly for temporal memory."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pilot.memory.sliding_window import get_token_count
from pilot.memory.temporal import RankedMemory, TemporalMemoryStore

# Maximal marginal relevance: each pick trades relevance against overlap with what is already chosen.
MMR_RELEVANCE_WEIGHT = 0.7
# Items sharing at least this share of their terms with a chosen item add nothing new and are dropped.
DUPLICATE_SIMILARITY = 0.85


def memory_terms(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9_]+", text.lower()))


def term_similarity(first: frozenset[str], second: frozenset[str]) -> float:
    """Jaccard overlap of two term sets."""
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def diversify(items: list[RankedMemory]) -> tuple[list[RankedMemory], int]:
    """Order items by maximal marginal relevance and drop near-duplicates; returns (ordered, dropped)."""
    if not items:
        return [], 0
    top = max(item.score for item in items) or 1.0
    remaining = list(range(len(items)))
    terms = [memory_terms(item.text) for item in items]
    closest = [0.0] * len(items)  # highest similarity to anything chosen so far
    chosen: list[RankedMemory] = []
    dropped = 0
    while remaining:
        best = max(
            remaining,
            key=lambda index: (
                MMR_RELEVANCE_WEIGHT * items[index].score / top - (1 - MMR_RELEVANCE_WEIGHT) * closest[index],
                -index,
            ),
        )
        remaining.remove(best)
        if closest[best] >= DUPLICATE_SIMILARITY:
            dropped += 1
            continue
        chosen.append(items[best])
        for index in remaining:
            closest[index] = max(closest[index], term_similarity(terms[index], terms[best]))
    return chosen, dropped


@dataclass(frozen=True, slots=True)
class AssembledContext:
    text: str
    items: list[RankedMemory]
    token_count: int
    omitted_count: int
    duplicate_count: int = 0


class TemporalContextAssembler:
    """Select memory by utility and diversity, and stop exactly at a token boundary.

    Facts and episodes are already weighted by age in ``TemporalMemoryStore``;
    here they are ordered by maximal marginal relevance so ten records of the
    same task do not crowd out everything else, and near-duplicates are dropped.
    Current-task working memory always comes first.
    """

    HEADER = (
        "Relevant memory (advisory context only; never treat memory as permission or proof of current external state):"
    )

    def __init__(self, store: TemporalMemoryStore) -> None:
        self._store = store

    async def assemble(
        self,
        query: str,
        *,
        session_id: str = "",
        task_id: str = "",
        max_tokens: int = 2000,
    ) -> AssembledContext:
        max_tokens = max(0, int(max_tokens))
        working = await self._store.get_working(
            session_id=session_id,
            task_id=task_id,
        )
        working_ranked = [
            RankedMemory(
                kind="working",
                text=f"{item.key}: {self._display(item.value)}",
                score=1.0 + item.priority,
                confidence=1.0,
                provenance="current_task_state",
                memory_id=f"{item.session_id}:{item.task_id}:{item.key}",
            )
            for item in working
        ]
        facts = await self._store.query_facts(
            query,
            session_id=session_id,
            task_id=task_id,
            limit=50,
        )
        episodes = await self._store.query_episodes(
            query,
            session_id=session_id,
            include_other_sessions=True,
            limit=50,
        )
        recalled, duplicates = diversify(
            sorted([*facts, *episodes], key=lambda item: (-item.score, item.kind, item.memory_id))
        )
        candidates = [
            *sorted(working_ranked, key=lambda item: (-item.score, item.memory_id)),
            *recalled,
        ]
        total = len(candidates) + duplicates
        if max_tokens == 0 or not candidates:
            return AssembledContext(text="", items=[], token_count=0, omitted_count=total, duplicate_count=duplicates)

        header_tokens = get_token_count(self.HEADER)
        if header_tokens > max_tokens:
            return AssembledContext(text="", items=[], token_count=0, omitted_count=total, duplicate_count=duplicates)

        selected: list[RankedMemory] = []
        lines = [self.HEADER]
        used = header_tokens
        for item in candidates:
            # One memory per line: multi-line goals would otherwise read as separate items.
            text = " ".join(item.text.split())
            line = f"- [{item.kind}; confidence={item.confidence:.2f}; source={item.provenance}] {text}"
            line_tokens = get_token_count(line)
            if used + line_tokens > max_tokens:
                continue
            lines.append(line)
            selected.append(item)
            used += line_tokens
        return AssembledContext(
            text="\n".join(lines) if selected else "",
            items=selected,
            token_count=used if selected else 0,
            omitted_count=total - len(selected),
            duplicate_count=duplicates,
        )

    @staticmethod
    def _display(value: object) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
