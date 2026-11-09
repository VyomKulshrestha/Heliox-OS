"""The live layer: a micro-turn clock that perceives, notices and decides when to speak.

Every ``TICK_SECONDS`` the companion looks at the moment: what the person is typing (the focused field, read
locally), whether they are talking, whether Heliox is talking, whether a focus session is on. The detectors find
specific mistakes; the policy decides, per correction, whether to wait, show a card, speak, or let it go. When
the person fixes something they were told about, the card goes away by itself and that counts as a thank-you.

It never acts on its own. "Fix it" hands a request to the planner through the normal approval path, the same
as if the person had typed it.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import re
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from pilot.live.detectors import KINDS, SPEECH_KINDS, URGENT, Correction, TextContext, detect
from pilot.live.policy import (
    DEFAULT_LEVEL,
    DROP,
    LEVELS,
    SHOW,
    SPEAK,
    Activity,
    Pending,
    Relevance,
    decide,
)

logger = logging.getLogger("pilot.live.engine")

TICK_SECONDS = 0.2
PERCEIVE_EVERY_SECONDS = 0.6
REDETECT_EVERY_SECONDS = 2.0
STALE_AFTER_SECONDS = 1.6  # evidence gone this long from the same window means the person fixed it
SPEECH_TTL_SECONDS = 10.0
PENDING_TTL_SECONDS = 90.0
VISIBLE_TTL_SECONDS = 600.0  # a card left alone this long is withdrawn and counts as mildly unwelcome
RECENT_LIMIT = 20
REPLY_WINDOW_SECONDS = 90.0  # how long "why?", "thanks" or "fix it" refer to the last correction
OCR_KINDS = ("error_on_screen",)  # a whole-screen OCR is too noisy for anything finer


@dataclass(frozen=True)
class Observation:
    """One look at what the person is working on. ``text`` never leaves this process."""

    text: str
    app: str
    title: str
    source: str  # "focused_field" or "screen_ocr"

    @property
    def window(self) -> str:
        return f"{self.app}|{self.title}"


@dataclass(frozen=True)
class LiveReply:
    reply: str
    delegate: str = ""  # a request to hand to the planner, when the person says "fix it"


# Supervision's rule names (pilot.security.risk_patterns) as something worth saying out loud.
_RISK_NOTICES = {
    "destructive_shell_command": (
        "That command could wipe a lot",
        "That command could delete a lot of files, and there's no undo.",
    ),
    "credential_exposure": (
        "That looks like a password or key",
        "That looks like a password or key. Make sure it isn't going somewhere other people can read.",
    ),
    "destructive_sql": ("That SQL deletes data for good", "That SQL drops or deletes data permanently."),
    "unauthorized_admin_command": (
        "That changes admin rights",
        "That command changes administrator accounts or permissions.",
    ),
}

_THANKS = re.compile(r"^(?:thanks|thank you|got it|good catch|nice catch|nice one|okay thanks|ok thanks)$")
_DISMISS = re.compile(r"^(?:not now|later|ignore (?:it|that)|i know|that'?s fine|it'?s fine|never mind|leave it)$")
_WHY = re.compile(r"^(?:why|why is that|what do you mean|explain|explain that|what'?s wrong)$")
_FIX = re.compile(r"^(?:fix it|fix that|do it|go ahead|go ahead and fix it|please fix it)$")


class LiveCompanion:
    def __init__(
        self,
        *,
        settings: Callable[[], Any],
        observe: Callable[[], Awaitable[Observation | None]],
        emit: Callable[[str, dict[str, Any]], Awaitable[None]],
        speak: Callable[[str, bool], Awaitable[None]],
        user_speaking: Callable[[], bool] = lambda: False,
        heliox_speaking: Callable[[], bool] = lambda: False,
        focus_active: Callable[[], bool] = lambda: False,
        people: Callable[[], dict[str, str]] = dict,
        mute_kind: Callable[[str], Awaitable[None]] | None = None,
        on_disable: Callable[[], Awaitable[None]] | None = None,
        relevance_path: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._settings = settings
        self._observe = observe
        self._emit = emit
        self._speak = speak
        self._user_speaking = user_speaking
        self._heliox_speaking = heliox_speaking
        self._focus_active = focus_active
        self._people = people
        self._mute_kind = mute_kind
        self._on_disable = on_disable
        self._was_on = False
        self._clock = clock
        self._today = today
        self.relevance = Relevance(relevance_path)

        self._task: asyncio.Task[None] | None = None
        self._paused = False
        self._pending: dict[str, Pending] = {}
        self._visible: dict[str, Pending] = {}  # delivered and still on screen, by correction id
        self._recent: deque[dict[str, Any]] = deque(maxlen=RECENT_LIMIT)
        self._last_obs: Observation | None = None
        self._last_hash = ""
        self._last_text_change = float("-inf")
        self._last_perceive = float("-inf")
        self._last_detect = float("-inf")
        self._last_spoken_at = float("-inf")
        self._last_shown_at = float("-inf")
        self._state: dict[str, Any] = {}

    # ── lifecycle ──

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if not self.running:
            self._task = asyncio.create_task(self._run(), name="heliox-live")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await self._withdraw_all("off")
        self._pending.clear()

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    async def _run(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Live tick failed", exc_info=True)
            await asyncio.sleep(TICK_SECONDS)

    # ── one micro-turn ──

    def _config(self) -> Any:
        return self._settings()

    def _kinds(self) -> tuple[str, ...]:
        muted = set(getattr(self._config(), "muted_kinds", []) or [])
        return tuple(kind for kind in KINDS if kind not in muted)

    def _context(self, obs: Observation | None, source: str) -> TextContext:
        return TextContext(
            today=self._today(),
            app=obs.app if obs else "",
            title=obs.title if obs else "",
            source=source,
            people=self._people(),
        )

    async def tick(self) -> None:
        cfg = self._config()
        now = self._clock()
        if not getattr(cfg, "enabled", False) or self._paused:
            if self._was_on:
                # Turned off or paused: take the cards down and let the text reader go.
                self._was_on = False
                await self._withdraw_all("off")
                self._pending.clear()
                self._last_obs = None
                if self._on_disable:
                    with contextlib.suppress(Exception):
                        await self._on_disable()
            await self._publish_state(now)
            return
        self._was_on = True

        if now - self._last_perceive >= PERCEIVE_EVERY_SECONDS:
            self._last_perceive = now
            await self._perceive(now)

        self._expire(now)
        await self._deliver_due(now)
        await self._publish_state(now)

    async def _perceive(self, now: float) -> None:
        try:
            obs = await asyncio.wait_for(self._observe(), timeout=2.0)
        except (TimeoutError, Exception):
            obs = None
        if obs is None:
            self._last_obs = None
            return
        digest = hashlib.sha1(obs.text.encode("utf-8", errors="ignore")).hexdigest()
        changed = digest != self._last_hash or (self._last_obs is None or obs.window != self._last_obs.window)
        if changed:
            self._last_hash = digest
            self._last_text_change = now
        self._last_obs = obs
        if changed or now - self._last_detect >= REDETECT_EVERY_SECONDS:
            self._last_detect = now
            kinds = self._kinds()
            if obs.source == "screen_ocr":
                kinds = tuple(kind for kind in kinds if kind in OCR_KINDS)
            found: set[str] = set()
            for correction in detect(obs.text, self._context(obs, obs.source), kinds=kinds):
                found.add(correction.key)
                self._note(correction, obs.source, obs.window, now)
            for item in [*self._pending.values(), *self._visible.values()]:
                if item.source == "speech" or item.window != obs.window:
                    continue
                if item.correction.key in found:
                    item.missing_since = None
                elif item.missing_since is None:
                    item.missing_since = now

    def _note(self, correction: Correction, source: str, window: str, now: float) -> None:
        if any(item.correction.key == correction.key for item in self._visible.values()):
            for item in self._visible.values():
                if item.correction.key == correction.key:
                    item.last_seen = now
            return
        if any(entry.get("key") == correction.key for entry in self._recent):
            return  # already told once; a correction is never repeated for the same text
        existing = self._pending.get(correction.key)
        if existing:
            existing.last_seen = now
            return
        self._pending[correction.key] = Pending(correction, source, now, now, window=window)

    def offer_speech(self, text: str) -> None:
        """Check a sentence the person said. Spoken corrections land as soon as they stop talking."""
        cfg = self._config()
        if not getattr(cfg, "enabled", False) or self._paused or not getattr(cfg, "listen_to_speech", True):
            return
        now = self._clock()
        kinds = tuple(kind for kind in self._kinds() if kind in SPEECH_KINDS)
        for correction in detect(text, self._context(None, "speech"), kinds=kinds):
            self._note(correction, "speech", "speech", now)

    def _gone(self, item: Pending, now: float) -> bool:
        return item.missing_since is not None and now - item.missing_since > STALE_AFTER_SECONDS

    async def notice_risk(self, pattern: str, source: str) -> None:
        """A risk the supervision keyboard hook or OCR caught, by rule name only; delivered as an urgent card."""
        title, message = _RISK_NOTICES.get(pattern, ("That looks risky", "Heads up, that looks risky."))
        now = self._clock()
        correction = Correction(
            kind="secret_in_text" if pattern == "credential_exposure" else "risky_command",
            key=f"notice:{pattern}:{int(now // 60)}",
            title=title,
            message=message,
            evidence="",
            urgency=URGENT,
        )
        self._note(correction, "notice", "notice", now)

    def _expire(self, now: float) -> None:
        for key, item in list(self._pending.items()):
            ttl = SPEECH_TTL_SECONDS if item.source == "speech" else PENDING_TTL_SECONDS
            if self._gone(item, now) or now - item.first_seen > ttl:
                del self._pending[key]
        for correction_id, item in list(self._visible.items()):
            if self._gone(item, now):
                # The evidence left the same window it was found in: they fixed it after being told.
                self.relevance.update(item.correction.kind, "fixed")
                reason = "fixed"
            elif now - item.delivered_at > VISIBLE_TTL_SECONDS:
                self.relevance.update(item.correction.kind, "ignored")
                reason = "expired"
            else:
                continue
            self._visible.pop(correction_id, None)
            asyncio.ensure_future(self._emit("live_interjection_withdrawn", {"id": correction_id, "reason": reason}))

    async def _deliver_due(self, now: float) -> None:
        cfg = self._config()
        level = LEVELS.get(getattr(cfg, "level", DEFAULT_LEVEL), LEVELS[DEFAULT_LEVEL])
        muted = frozenset(getattr(cfg, "muted_kinds", []) or [])
        order = {URGENT: 0}
        said: list[str] = []
        urgent_said = False
        for key, item in sorted(self._pending.items(), key=lambda kv: order.get(kv[1].correction.urgency, 1)):
            act = Activity(
                now=now,
                last_text_change=self._last_text_change,
                user_speaking=self._user_speaking(),
                heliox_speaking=self._heliox_speaking(),
                focus_active=self._focus_active(),
                visible_cards=len(self._visible),
                last_spoken_at=self._last_spoken_at,
                last_shown_at=self._last_shown_at,
            )
            verdict = decide(item, act, level, self.relevance.score(item.correction.kind), muted)
            if verdict == DROP:
                del self._pending[key]
            elif verdict in (SHOW, SPEAK):
                del self._pending[key]
                await self._deliver(item, spoken=verdict == SPEAK, now=now)
                if verdict == SPEAK:
                    said.append(item.correction.message)
                    urgent_said = urgent_said or item.correction.urgency == URGENT
        if said:
            # Several catches in one turn are said as one reply, not as a queue of interruptions.
            self._last_spoken_at = now
            asyncio.ensure_future(self._speak(" ".join(said), urgent_said))

    async def _deliver(self, item: Pending, *, spoken: bool, now: float) -> None:
        correction = item.correction
        item.id = uuid.uuid4().hex[:10]
        item.delivered = True
        item.delivered_at = now
        item.mode = SPEAK if spoken else SHOW
        if item.source != "speech":
            self._visible[item.id] = item
        payload = {
            "id": item.id,
            "kind": correction.kind,
            "title": correction.title,
            "message": correction.message,
            "detail": correction.detail,
            "evidence": correction.evidence,
            "fix": correction.fix,
            "fix_kind": correction.fix_kind,
            "urgency": correction.urgency,
            "spoken": spoken,
            "source": item.source,
            "app": item.window.split("|", 1)[0],
        }
        self._recent.append({**payload, "key": correction.key, "at": now})
        self._last_shown_at = now
        await self._emit("live_interjection", payload)
        if spoken:
            self._last_spoken_at = now

    async def _withdraw_all(self, reason: str) -> None:
        for correction_id in list(self._visible):
            self._visible.pop(correction_id, None)
            with contextlib.suppress(Exception):
                await self._emit("live_interjection_withdrawn", {"id": correction_id, "reason": reason})

    # ── the person's response ──

    def _latest(self, now: float) -> dict[str, Any] | None:
        if not self._recent:
            return None
        entry = self._recent[-1]
        return entry if now - entry["at"] <= REPLY_WINDOW_SECONDS else None

    def _find(self, correction_id: str) -> dict[str, Any] | None:
        return next((entry for entry in self._recent if entry["id"] == correction_id), None)

    async def feedback(self, correction_id: str, verdict: str) -> dict[str, Any]:
        entry = self._find(correction_id)
        if entry is None:
            return {"status": "error", "message": "That correction is no longer available."}
        if verdict == "mute":
            self.relevance.update(entry["kind"], "dismissed")
            if self._mute_kind:
                await self._mute_kind(entry["kind"])
        else:
            self.relevance.update(entry["kind"], verdict)
        if correction_id in self._visible:
            self._visible.pop(correction_id, None)
            await self._emit("live_interjection_withdrawn", {"id": correction_id, "reason": verdict})
        return {"status": "ok", "kind": entry["kind"], "relevance": round(self.relevance.score(entry["kind"]), 2)}

    def explain(self, correction_id: str = "") -> str:
        entry = self._find(correction_id) if correction_id else self._latest(self._clock())
        if entry is None:
            return ""
        parts = [entry["message"]]
        if entry.get("detail"):
            parts.append(entry["detail"])
        if entry.get("fix") and entry.get("fix_kind") in ("replace", "command"):
            parts.append(f"The fix: {entry['fix']}.")
        return " ".join(parts)

    async def handle_reply(self, normalized: str) -> LiveReply | None:
        """ "Thanks", "not now", "why?" and "fix it", when they follow a recent correction."""
        entry = self._latest(self._clock())
        if entry is None:
            return None
        if _WHY.match(normalized):
            return LiveReply(self.explain(entry["id"]))
        if _THANKS.match(normalized):
            await self.feedback(entry["id"], "helpful")
            return LiveReply("Glad it helped.")
        if _DISMISS.match(normalized):
            await self.feedback(entry["id"], "dismissed")
            return LiveReply("Okay, leaving it.")
        if _FIX.match(normalized):
            if entry.get("fix_kind") == "ask" and entry.get("fix"):
                await self.feedback(entry["id"], "used_fix")
                return LiveReply("On it.", delegate=entry["fix"])
            if entry.get("fix"):
                await self.feedback(entry["id"], "used_fix")
                return LiveReply(f"I can't edit your text for you, so here's the fix to use: {entry['fix']}.")
            return LiveReply("There's no automatic fix for that one.")
        return None

    # ── status ──

    def _user_state(self, now: float) -> str:
        if self._user_speaking():
            return "speaking"
        if self._last_obs is None:
            return "away"
        return "typing" if now - self._last_text_change < 1.2 else "paused"

    def status(self) -> dict[str, Any]:
        cfg = self._config()
        now = self._clock()
        return {
            "status": "ok",
            "enabled": bool(getattr(cfg, "enabled", False)),
            "paused": self._paused,
            "running": self.running,
            "level": getattr(cfg, "level", DEFAULT_LEVEL),
            "watching": self._last_obs.app if self._last_obs else "",
            "source": self._last_obs.source if self._last_obs else "",
            "user_state": self._user_state(now),
            "visible": [entry for entry in self._recent if entry["id"] in self._visible],
            "recent": [{k: v for k, v in entry.items() if k not in ("key", "at")} for entry in list(self._recent)[-8:]],
            "relevance": self.relevance.snapshot(),
            "muted_kinds": list(getattr(cfg, "muted_kinds", []) or []),
        }

    async def _publish_state(self, now: float) -> None:
        cfg = self._config()
        state = {
            "enabled": bool(getattr(cfg, "enabled", False)),
            "paused": self._paused,
            "watching": self._last_obs.app if self._last_obs and getattr(cfg, "enabled", False) else "",
            "user_state": self._user_state(now) if getattr(cfg, "enabled", False) else "off",
        }
        if state != self._state:
            self._state = state
            with contextlib.suppress(Exception):
                await self._emit("live_state", state)
