"""Activity timeline: a local record of which app and window you had in front, and when.

When the person turns it on, the screen-awareness loop hands each observation (the
active app and its window title) to this timeline, which merges them into spans:
"Code, server.py - Heliox, 14:02-14:37". Only names and titles are kept: never
pixels and never screen text. Password managers and private browsing windows are
skipped, recording can be paused, and spans older than ``retention_days`` are
deleted. The database lives under ``DATA_DIR/timeline``, which actions may not read
or change, so the timeline is reachable only through the person's own questions in
Settings or by voice ("what was I working on yesterday afternoon?").

Answers are composed in code from the stored spans; no model sees the timeline.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Gaps longer than this end a span (screen off, machine asleep, an excluded window).
MAX_GAP_SECONDS = 120.0
# The open span's end is written at most this often; it is always written when the span closes.
WRITE_EVERY_SECONDS = 60.0
ALWAYS_EXCLUDED_MARKERS = ("inprivate", "incognito", "private browsing", "private window")
DEFAULT_EXCLUDED_APPS = ["1password", "bitwarden", "keepass", "lastpass", "dashlane", "keychain access"]

TIMELINE_QUESTION = re.compile(
    r"^(?:hey heliox |heliox )?(?:what (?:was|were) i (?:working on|doing|looking at)|"
    r"what did i (?:work on|do|have open))\b(?P<when>.*)$"
)
LAST_SEEN_QUESTION = re.compile(r"^(?:hey heliox |heliox )?when did i last (?:see|open|use|have open) (?P<what>.+)$")


@dataclass
class _OpenSpan:
    row_id: int
    app: str
    title: str
    start: float
    end: float
    written_end: float


def _fmt_duration(seconds: float) -> str:
    minutes = max(1, round(seconds / 60))
    if minutes < 60:
        return f"{minutes} min"
    hours, rest = divmod(minutes, 60)
    return f"{hours} h {rest} min" if rest else f"{hours} h"


def _fmt_clock(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%I:%M %p").lstrip("0")


def parse_when(text: str, now: datetime) -> tuple[float, float, str]:
    """Turn "yesterday afternoon", "at 3 pm", "an hour ago" and similar into a time window."""
    text = " " + text.lower().strip(" ?.!") + " "
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    label_day = "today"
    if " yesterday " in text:
        day -= timedelta(days=1)
        label_day = "yesterday"

    def window(start_hour: float, end_hour: float, label: str) -> tuple[float, float, str]:
        start = day + timedelta(hours=start_hour)
        end = min(day + timedelta(hours=end_hour), now)
        return start.timestamp(), max(end, start).timestamp(), label

    ago = re.search(r"\b(\d+|an|a|one|two|three)\s+(minute|minutes|hour|hours)\s+ago\b", text)
    if ago:
        words = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3}
        amount = int(ago.group(1)) if ago.group(1).isdigit() else words[ago.group(1)]
        delta = timedelta(minutes=amount) if ago.group(2).startswith("minute") else timedelta(hours=amount)
        center = now - delta
        return (center - timedelta(minutes=15)).timestamp(), (center + timedelta(minutes=15)).timestamp(), ago.group(0)
    if re.search(r"\b(last|past) hour\b", text):
        return (now - timedelta(hours=1)).timestamp(), now.timestamp(), "in the last hour"
    at = re.search(r"\b(?:at|around) (\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\b", text)
    if at:
        hour = int(at.group(1)) % 24
        minute = int(at.group(2) or 0)
        meridiem = (at.group(3) or "").replace(".", "")
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        elif not meridiem and 1 <= hour <= 7:
            hour += 12  # "around 3" during a working day means the afternoon
        center = day + timedelta(hours=hour, minutes=minute)
        label = f"{label_day} around {_fmt_clock(center.timestamp())}"
        return (center - timedelta(minutes=30)).timestamp(), (center + timedelta(minutes=30)).timestamp(), label
    for word, start_hour, end_hour in (
        ("morning", 5, 12),
        ("afternoon", 12, 17),
        ("evening", 17, 24),
        ("tonight", 17, 24),
        ("night", 17, 24),
    ):
        if f" {word} " in text:
            label = (
                "tonight" if word == "tonight" else (f"this {word}" if label_day == "today" else f"yesterday {word}")
            )
            return window(start_hour, end_hour, label)
    if label_day == "yesterday":
        return window(0, 24, "yesterday")
    if " today " in text:
        return window(0, 24, "today")
    return (now - timedelta(hours=2)).timestamp(), now.timestamp(), "in the last two hours"


class ActivityTimeline:
    """Stores app and window-title spans in a local SQLite database."""

    def __init__(self, db_path: Path, *, config: Any = None) -> None:
        self._db_path = Path(db_path)
        # A TimelineConfig, or a function returning the current one, read on every call.
        self._config = config
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._open: _OpenSpan | None = None
        self._paused_until = 0.0

    # ------------------------------------------------------------------ settings

    def _setting(self, name: str, default: Any) -> Any:
        config = self._config() if callable(self._config) else self._config
        return getattr(config, name, default) if config is not None else default

    @property
    def enabled(self) -> bool:
        return bool(self._setting("enabled", False))

    @property
    def paused_until(self) -> float:
        return self._paused_until if self._paused_until > time.time() else 0.0

    def pause(self, minutes: float) -> float:
        """Stop recording for ``minutes`` (0 resumes). Returns when recording resumes."""
        with self._lock:
            self._close_open(time.time())
            self._paused_until = time.time() + max(0.0, float(minutes)) * 60 if minutes else 0.0
        return self.paused_until

    def _excluded(self, app: str, title: str) -> bool:
        text = f"{app} {title}".lower()
        if any(marker in text for marker in ALWAYS_EXCLUDED_MARKERS):
            return True
        apps = self._setting("excluded_apps", DEFAULT_EXCLUDED_APPS) or []
        return any(str(name).strip().lower() and str(name).strip().lower() in text for name in apps)

    # ------------------------------------------------------------------ storage

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS spans ("
                "id INTEGER PRIMARY KEY, start REAL NOT NULL, end REAL NOT NULL, app TEXT NOT NULL, title TEXT NOT NULL)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS spans_end ON spans(end)")
            conn.commit()
            self._conn = conn
        return self._conn

    def _close_open(self, now: float) -> None:
        span = self._open
        self._open = None
        if span is not None and span.end != span.written_end:
            self._db().execute("UPDATE spans SET end = ? WHERE id = ?", (span.end, span.row_id))
            self._db().commit()

    # ------------------------------------------------------------------ recording

    def observe(self, app: str, title: str, when: float | None = None) -> None:
        """Record that ``app`` / ``title`` was in front at ``when``. Cheap; safe to call every few seconds."""
        when = time.time() if when is None else when
        app = (app or "").strip()[:120]
        title = (title or "").strip()[:300]
        with self._lock:
            if not self.enabled or self._paused_until > when or not (app or title) or self._excluded(app, title):
                self._close_open(when)
                return
            span = self._open
            if span is not None and span.app == app and span.title == title and when - span.end <= MAX_GAP_SECONDS:
                span.end = when
                if when - span.written_end >= WRITE_EVERY_SECONDS:
                    self._db().execute("UPDATE spans SET end = ? WHERE id = ?", (when, span.row_id))
                    self._db().commit()
                    span.written_end = when
                return
            self._close_open(when)
            cursor = self._db().execute(
                "INSERT INTO spans (start, end, app, title) VALUES (?, ?, ?, ?)", (when, when, app, title)
            )
            self._db().commit()
            self._open = _OpenSpan(int(cursor.lastrowid or 0), app, title, when, when, when)

    def flush(self) -> None:
        with self._lock:
            span = self._open
            if span is not None and span.end != span.written_end:
                self._db().execute("UPDATE spans SET end = ? WHERE id = ?", (span.end, span.row_id))
                self._db().commit()
                span.written_end = span.end

    # ------------------------------------------------------------------ reading

    def _rows(self, sql: str, args: tuple[Any, ...]) -> list[dict[str, Any]]:
        self.flush()
        with self._lock:
            if not self._db_path.exists():
                return []
            rows = self._db().execute(sql, args).fetchall()
        return [{"start": r[0], "end": r[1], "app": r[2], "title": r[3]} for r in rows]

    def spans(self, start: float, end: float, limit: int = 500) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT start, end, app, title FROM spans WHERE end >= ? AND start <= ? ORDER BY start LIMIT ?",
            (start, end, max(1, min(int(limit), 5000))),
        )

    def search(self, query: str, *, start: float = 0.0, end: float | None = None, limit: int = 50) -> list[dict]:
        pattern = "%" + query.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        return self._rows(
            "SELECT start, end, app, title FROM spans WHERE end >= ? AND start <= ? "
            "AND (title LIKE ? ESCAPE '\\' OR app LIKE ? ESCAPE '\\') ORDER BY start DESC LIMIT ?",
            (start, 1e18 if end is None else end, pattern, pattern, max(1, min(int(limit), 500))),
        )

    def summarize(self, start: float, end: float) -> dict[str, Any]:
        apps: dict[str, dict[str, Any]] = {}
        total = 0.0
        for span in self.spans(start, end, limit=5000):
            seconds = max(0.0, min(span["end"], end) - max(span["start"], start))
            if seconds <= 0:
                continue
            total += seconds
            entry = apps.setdefault(span["app"] or "Unknown app", {"seconds": 0.0, "titles": {}})
            entry["seconds"] += seconds
            entry["titles"][span["title"]] = entry["titles"].get(span["title"], 0.0) + seconds
        ranked = sorted(apps.items(), key=lambda item: item[1]["seconds"], reverse=True)
        return {
            "start": start,
            "end": end,
            "total_seconds": round(total),
            "apps": [
                {
                    "app": name,
                    "seconds": round(data["seconds"]),
                    "top_titles": [
                        title for title, _s in sorted(data["titles"].items(), key=lambda t: t[1], reverse=True)[:3]
                    ],
                }
                for name, data in ranked[:8]
            ],
        }

    # ------------------------------------------------------------------ answers

    def answer(self, question: str, *, now: datetime | None = None) -> str | None:
        """Answer a timeline question in words, or None when the question is not about the timeline."""
        normalized = re.sub(r"\s+", " ", question.lower().strip(" ?.!"))
        last_seen = LAST_SEEN_QUESTION.match(normalized)
        asked = TIMELINE_QUESTION.match(normalized)
        if not last_seen and not asked:
            return None
        if not self.enabled:
            return "The activity timeline is off. You can turn it on in Settings, under Activity timeline."
        now = now or datetime.now()
        if last_seen:
            what = last_seen.group("what").strip()
            what = re.sub(r"^(the|my|that) ", "", what)
            hits = self.search(what, limit=1)
            if not hits:
                return f"I don't see {what} in your timeline."
            hit = hits[0]
            day = datetime.fromtimestamp(hit["end"]).date()
            when_word = (
                "today" if day == now.date() else "yesterday" if day == now.date() - timedelta(days=1) else str(day)
            )
            return f'You last had "{hit["title"] or hit["app"]}" open in {hit["app"]} {when_word} at {_fmt_clock(hit["end"])}.'
        start, end, label = parse_when(asked.group("when"), now)
        summary = self.summarize(start, end)
        if not summary["apps"]:
            return f"Your timeline has nothing recorded {label}."
        parts = []
        for entry in summary["apps"][:3]:
            title = entry["top_titles"][0] if entry["top_titles"] else ""
            detail = f', mostly "{title}"' if title and title != entry["app"] else ""
            parts.append(f"{entry['app']} for {_fmt_duration(entry['seconds'])}{detail}")
        opening = label[0].upper() + label[1:]
        return f"{opening} you used " + "; then ".join(parts) + "."

    # ------------------------------------------------------------------ retention

    def prune(self, *, now: float | None = None) -> int:
        horizon = (time.time() if now is None else now) - float(self._setting("retention_days", 7)) * 86400
        with self._lock:
            if not self._db_path.exists():
                return 0
            cursor = self._db().execute("DELETE FROM spans WHERE end < ?", (horizon,))
            self._db().commit()
            return int(cursor.rowcount or 0)

    def clear(self) -> None:
        with self._lock:
            self._open = None
            if self._db_path.exists():
                self._db().execute("DELETE FROM spans")
                self._db().commit()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._close_open(time.time())
                self._conn.close()
                self._conn = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            count = 0
            if self._db_path.exists():
                count = int(self._db().execute("SELECT COUNT(*) FROM spans").fetchone()[0])
        return {
            "enabled": self.enabled,
            "paused_until": self.paused_until or None,
            "span_count": count,
            "retention_days": int(self._setting("retention_days", 7)),
        }

    async def answer_async(self, question: str) -> str | None:
        return await asyncio.to_thread(self.answer, question)
