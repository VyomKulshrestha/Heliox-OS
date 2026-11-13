"""How often a follow-up came after talk Heliox left out: four numbers a person can read, and nothing else.

The second-voice rule (pilot.conversation.reach) is off because its threshold could not be measured: synthetic voices
do not mutter, and too few real recordings exist. Real use answers it. Each follow-up Heliox takes adds one to
``taken``, and to ``at_least_1``, ``at_least_2`` and ``at_least_3`` when the listener had left out that many
utterances since Heliox last spoke. Kept as four integers and the day counting began, in the state folder. No words,
no times finer than the day, nothing about who spoke. Shown in the voice diagnostics (voice_listener_stats) and
cleared from there; included in nothing that leaves the machine.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot.conversation.follow_up_counts")

KEYS = ("taken", "at_least_1", "at_least_2", "at_least_3")


class FollowUpCounts:
    def __init__(self, path: Path, *, today: Any = date.today) -> None:
        self._path = Path(path)
        self._today = today

    def read(self) -> dict[str, Any]:
        """``{"since": "YYYY-MM-DD", "taken": n, "at_least_1": n, "at_least_2": n, "at_least_3": n}``."""
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        counts: dict[str, Any] = {key: _count(raw.get(key)) for key in KEYS}
        since = raw.get("since")
        counts["since"] = since if isinstance(since, str) and _is_day(since) else self._today().isoformat()
        return counts

    def record(self, overheard: int) -> None:
        """One follow-up taken, after ``overheard`` utterances the listener left out since Heliox last spoke."""
        counts = self.read()
        counts["taken"] += 1
        for level in (1, 2, 3):
            if overheard >= level:
                counts[f"at_least_{level}"] += 1
        self._write(counts)

    def reset(self) -> dict[str, Any]:
        counts: dict[str, Any] = {key: 0 for key in KEYS}
        counts["since"] = self._today().isoformat()
        self._write(counts)
        return counts

    def _write(self, counts: dict[str, Any]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"since": counts["since"], **{key: int(counts[key]) for key in KEYS}}
            self._path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            logger.warning("Could not keep the follow-up counts", exc_info=True)


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _is_day(value: str) -> bool:
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return len(value) == 10
