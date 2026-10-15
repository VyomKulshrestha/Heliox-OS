"""Encrypted, consent-gated local store for the co-adaptation study (spec sections 10.6 to 10.9).

``co_adaptation.db`` is separate from every other Heliox database and is created lazily inside the first successful
consent grant. Plaintext columns hold only ids, timestamps, enum values and versions. Every ``*_enc`` column holds
``nonce(12) || AES-256-GCM`` of canonical JSON with the AAD ``"<table>:<primary key>"``, encrypted with the
participant's own key. Payloads are exactly the ``to_dict()`` outputs of ``pilot.coadaptation.core.records`` and are
read back only with ``from_dict()``.

Every write checks, inside its write transaction and before any key is fetched, that the participant holds an active
consent (neither replaced, revoked nor expired) covering the purpose the write serves. Deleting rows needs neither a
key nor a consent.

Erasure is best-effort and resumable (spec 10.7). An export file another program holds open, or a keyring that cannot
be reached, never aborts a revoke, an expiry, an erase-today or a consent update half way: what could not be deleted
now goes on a small pending-erasure list kept next to the database (``co_adaptation_pending_erasure.json``, which
holds pseudonymous participant keys and export file names only, never data) and the service retries it at start, at
every status call and prune, and on a repeated revoke. The list file is deleted as soon as it is empty.

Decrypting and decoding stored rows runs in a worker thread, so a long history never blocks the daemon's event loop.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import os
import re
import tempfile
import uuid
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from pilot.coadaptation.consent import CoAdaptationConsentError, CoAdaptationConsentV1
from pilot.coadaptation.core import SDK_VERSION
from pilot.coadaptation.core.consent import PARTICIPANT_KEY_PATTERN, CoAdaptationPurpose, ConsentPeriod
from pilot.coadaptation.core.engine import EngineEvent, TrialObservation
from pilot.coadaptation.core.outcome_model import MODEL_VERSION
from pilot.coadaptation.core.policy import POLICY_VERSION
from pilot.coadaptation.core.preferences import Preference, preference_from_dict, preference_to_dict
from pilot.coadaptation.core.priors import PRIOR_VERSION
from pilot.coadaptation.core.records import (
    RECORD_SCHEMA_VERSION,
    BurdenReport,
    DecisionRecord,
    SessionRow,
    StoredTrial,
)
from pilot.coadaptation.core.state import FEATURE_SCHEMA
from pilot.coadaptation.core.vocabulary import STRATEGY_VERSION
from pilot.coadaptation.keys import CoAdaptationKeyUnavailable, KeyProvider
from pilot.db.sqlite_pool import AsyncSqlitePool

logger = logging.getLogger("pilot.coadaptation.store")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS participants (
  participant_key TEXT PRIMARY KEY CHECK (participant_key GLOB 'p-[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'),
  created_at TEXT NOT NULL,
  label_enc BLOB);

CREATE TABLE IF NOT EXISTS consents (
  consent_id TEXT PRIMARY KEY, participant_key TEXT NOT NULL REFERENCES participants(participant_key),
  record_json TEXT NOT NULL,
  granted_at TEXT NOT NULL, expires_at TEXT NOT NULL, revoked_at TEXT,
  replaced_at TEXT);

CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY, participant_key TEXT NOT NULL, context TEXT NOT NULL CHECK (context IN ('study')),
  day_index INTEGER NOT NULL, session_number INTEGER NOT NULL, protocol_id TEXT, protocol_sha256 TEXT,
  evidence_kind TEXT NOT NULL CHECK (evidence_kind IN ('live_human','recorded_human','simulated')),
  started_at TEXT NOT NULL, ended_at TEXT, end_reason TEXT,
  payload_enc BLOB NOT NULL);

CREATE TABLE IF NOT EXISTS decisions (
  decision_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, participant_key TEXT NOT NULL, seq INTEGER NOT NULL,
  level TEXT NOT NULL, mode TEXT NOT NULL, behavior TEXT NOT NULL CHECK (behavior IN ('policy','baseline')),
  policy_version TEXT NOT NULL, model_version TEXT NOT NULL, prior_version TEXT NOT NULL, feature_schema TEXT NOT NULL,
  created_at TEXT NOT NULL, superseded_by TEXT,
  payload_enc BLOB NOT NULL);

CREATE TABLE IF NOT EXISTS trials (
  trial_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, participant_key TEXT NOT NULL, seq INTEGER NOT NULL,
  block_index INTEGER NOT NULL, trial_index INTEGER NOT NULL, task TEXT NOT NULL, recorded_at TEXT NOT NULL,
  verification TEXT NOT NULL,
  payload_enc BLOB NOT NULL);

CREATE TABLE IF NOT EXISTS burden (
  report_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, participant_key TEXT NOT NULL, seq INTEGER NOT NULL,
  block_index INTEGER, scope TEXT NOT NULL CHECK (scope IN ('block','session')), recorded_at TEXT NOT NULL,
  payload_enc BLOB NOT NULL);

CREATE TABLE IF NOT EXISTS model_state (
  participant_key TEXT PRIMARY KEY, model_version TEXT NOT NULL, prior_version TEXT NOT NULL,
  rows_hash TEXT NOT NULL, updated_at TEXT NOT NULL, state_enc BLOB NOT NULL);

CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY, participant_key TEXT NOT NULL, kind TEXT NOT NULL,
  session_id TEXT, trial_id TEXT,
  created_at TEXT NOT NULL, payload_enc BLOB NOT NULL);

CREATE TABLE IF NOT EXISTS rules (
  participant_key TEXT NOT NULL, pref_id TEXT NOT NULL,
  created_at TEXT NOT NULL, retracted_at TEXT,
  payload_enc BLOB NOT NULL,
  PRIMARY KEY (participant_key, pref_id));

CREATE INDEX IF NOT EXISTS ix_trials_session ON trials(session_id, seq);
CREATE INDEX IF NOT EXISTS ix_decisions_session ON decisions(session_id, seq);
CREATE INDEX IF NOT EXISTS ix_sessions_participant ON sessions(participant_key, started_at);
CREATE INDEX IF NOT EXISTS ix_events_trial ON events(session_id, trial_id);
"""

ALARM_EVENT_KINDS = frozenset({"drift", "calibration_suspended"})
BOOKKEEPING_EVENT_KINDS = frozenset({"prune", "erase_today", "learning_reset"})
DECISION_CAP = 50_000
DECISION_CAP_KEEP = 49_000
BUNDLE_SCHEMA = 1
PENDING_ERASURE_FILE = "co_adaptation_pending_erasure.json"
PENDING_ERASURE_SCHEMA = 1

_PARTICIPANT_KEY = re.compile(PARTICIPANT_KEY_PATTERN)
_EXPORT_NAME = re.compile(r"^coadapt_(p-[0-9a-f]{8})_(\d{8}T\d{6}Z)(?:_(\d+))?\.enc$")
_PARTICIPANT_TABLES = (
    "decisions",
    "trials",
    "burden",
    "events",
    "model_state",
    "rules",
    "sessions",
    "consents",
    "participants",
)
_SELECTION_STUDY = CoAdaptationPurpose.SELECTION_STUDY

ReplayItem = SessionRow | StoredTrial | BurdenReport | EngineEvent


@dataclass(frozen=True, slots=True)
class ConsentUpdateResult:
    """What a consent update deleted because it removed a purpose (spec 10.3)."""

    purposes: tuple[str, ...]
    export_files: int
    trials_rewritten: int
    sessions_rewritten: int
    export_files_left: int = 0  # export files that could not be deleted now; on the pending-erasure list


@dataclass(frozen=True, slots=True)
class EraseResult:
    rows: int
    key_destroyed: bool  # no keyring key exists for the participant any more
    export_files: int
    export_files_left: int = 0  # export files that could not be deleted now; retried from the pending-erasure list

    @property
    def complete(self) -> bool:
        return self.key_destroyed and self.export_files_left == 0


@dataclass(frozen=True, slots=True)
class EraseDayResult:
    rows: int
    export_files: int
    export_files_left: int = 0


@dataclass(frozen=True, slots=True)
class PendingErasure:
    """Erasure work a revoke, expiry, erase-today or consent update could not finish yet (spec 10.7)."""

    participants: tuple[str, ...] = ()  # rows, export files and keyring key still to be deleted
    rules: tuple[str, ...] = ()  # participant-scoped rules still to be deleted from the preferences file
    files: tuple[str, ...] = ()  # names of single export files (in the exports directory) still to be deleted

    @property
    def empty(self) -> bool:
        return not (self.participants or self.rules or self.files)


@dataclass(frozen=True, slots=True)
class FallbackRelabel:
    """One decision the UI fell back from (spec 9.7), relabelled in the same transaction as its trial."""

    decision_id: str
    applied: str | None
    fallback_reason: str

    def changes(self) -> dict[str, Any]:
        return {"behavior": "baseline", "applied": self.applied, "fallback_reason": self.fallback_reason}


@dataclass(frozen=True, slots=True)
class SessionInfo:
    """Plaintext facts about one stored session; reading them needs no key."""

    session_id: str
    session_number: int
    day_index: int
    started_at: datetime
    ended_at: datetime | None
    end_reason: str | None
    trials: int


@dataclass(frozen=True, slots=True)
class ResultsRows:
    sessions: tuple[SessionRow, ...]
    decisions: tuple[DecisionRecord, ...]
    trials: tuple[StoredTrial, ...]
    burden: tuple[BurdenReport, ...]


@dataclass(frozen=True, slots=True)
class SessionRows:
    session: SessionRow
    decisions: tuple[DecisionRecord, ...]
    trials: tuple[StoredTrial, ...]
    burden: tuple[BurdenReport, ...]


# ── encoding helpers ─────────────────────────────────────────────────────────


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("stored timestamps must carry a UTC offset")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _parse_time(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def _canonical(payload: Any) -> bytes:
    # Python's json writes the shortest round-trip form of every float, so values read back bit-identical.
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode(
        "utf-8"
    )


def _encrypt(key: bytes, aad: str, plaintext: bytes) -> bytes:
    nonce = os.urandom(12)
    return nonce + AESGCM(key).encrypt(nonce, plaintext, aad.encode("utf-8"))


def _decrypt(key: bytes, aad: str, blob: bytes) -> bytes:
    data = bytes(blob)
    return AESGCM(key).decrypt(data[:12], data[12:], aad.encode("utf-8"))


def _encrypt_json(key: bytes, aad: str, payload: Any) -> bytes:
    return _encrypt(key, aad, _canonical(payload))


def _decrypt_json(key: bytes, aad: str, blob: bytes) -> Any:
    return json.loads(_decrypt(key, aad, blob).decode("utf-8"))


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _version(versions: dict[str, str], name: str, default: str) -> str:
    return str(versions.get(name) or versions.get(f"{name}_version") or default)


def _check_participant_key(participant_key: str) -> None:
    if not isinstance(participant_key, str) or not _PARTICIPANT_KEY.fullmatch(participant_key):
        raise ValueError("participant keys must match ^p-[0-9a-f]{8}$")


def _chmod_private(path: Path) -> None:
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)


def _strip_trial_fields(payload: dict[str, Any], *, gaze: bool, hand: bool) -> bool:
    """Null the gaze fields and/or the hand summary of a StoredTrial payload in place; True when anything changed."""
    trial = payload.get("trial")
    if not isinstance(trial, dict):
        return False
    changed = False
    if gaze:
        if trial.get("gaze") is not None:
            trial["gaze"] = None
            changed = True
        for activation in trial.get("activations") or []:
            if isinstance(activation, dict) and activation.get("gaze") is not None:
                activation["gaze"] = None
                changed = True
    if hand and trial.get("hand") is not None:
        trial["hand"] = None
        changed = True
    return changed


def _versions_bundle() -> dict[str, str]:
    return {
        "strategy": STRATEGY_VERSION,
        "feature_schema": FEATURE_SCHEMA,
        "model": MODEL_VERSION,
        "prior": PRIOR_VERSION,
        "policy": POLICY_VERSION,
        "sdk": SDK_VERSION,
    }


class CoAdaptationStore:
    """The participant-keyed encrypted store. Methods match spec section 13.2.1."""

    def __init__(
        self,
        db_path: Path,
        exports_dir: Path,
        key_provider: KeyProvider,
        *,
        clock: Callable[[], datetime] | None = None,
        pending_path: Path | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._exports_dir = Path(exports_dir)
        self._keys = key_provider
        self._clock = clock or (lambda: datetime.now(UTC))
        self._pool: AsyncSqlitePool | None = None
        # Outside the database on purpose: the database file itself is deleted with the last participant.
        self._pending_path = (
            Path(pending_path) if pending_path is not None else self._db_path.with_name(PENDING_ERASURE_FILE)
        )

    # ── lifecycle ────────────────────────────────────────────────────────────

    @property
    def exists(self) -> bool:
        return self._db_path.exists()

    @property
    def is_open(self) -> bool:
        return self._pool is not None

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def exports_dir(self) -> Path:
        return self._exports_dir

    @property
    def pending_path(self) -> Path:
        return self._pending_path

    # ── pending erasure ──────────────────────────────────────────────────────

    def pending_erasure(self) -> PendingErasure:
        """The erasure work still to do. Entries that are not participant keys or export file names are ignored."""
        if not self._pending_path.exists():
            return PendingErasure()
        try:
            document = json.loads(self._pending_path.read_text(encoding="utf-8"))
            if not isinstance(document, dict) or document.get("schema_version") != PENDING_ERASURE_SCHEMA:
                raise ValueError("unknown format")
            participants = tuple(
                key
                for key in document.get("participants", [])
                if isinstance(key, str) and _PARTICIPANT_KEY.fullmatch(key)
            )
            rules = tuple(
                key for key in document.get("rules", []) if isinstance(key, str) and _PARTICIPANT_KEY.fullmatch(key)
            )
            # Only bare export names are ever deleted, so a tampered list cannot point anywhere else.
            files = tuple(
                name for name in document.get("files", []) if isinstance(name, str) and _EXPORT_NAME.fullmatch(name)
            )
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            logger.warning("Co-adaptation pending-erasure list cannot be read (%s)", type(exc).__name__)
            return PendingErasure()
        return PendingErasure(participants=participants, rules=rules, files=files)

    def _pending_change(self, *, add: PendingErasure | None = None, remove: PendingErasure | None = None) -> None:
        """Add and remove entries; never raises, because the bookkeeping must not block an erasure."""
        current = self.pending_erasure()

        def merged(name: str) -> tuple[str, ...]:
            values = list(getattr(current, name))
            for value in getattr(add, name) if add is not None else ():
                if value not in values:
                    values.append(value)
            dropped = set(getattr(remove, name)) if remove is not None else set()
            return tuple(value for value in values if value not in dropped)

        updated = PendingErasure(participants=merged("participants"), rules=merged("rules"), files=merged("files"))
        if updated == current and (not updated.empty or not self._pending_path.exists()):
            return
        try:
            if updated.empty:
                with contextlib.suppress(FileNotFoundError):
                    self._pending_path.unlink()
                return
            document = {
                "schema_version": PENDING_ERASURE_SCHEMA,
                "participants": list(updated.participants),
                "rules": list(updated.rules),
                "files": list(updated.files),
            }
            self._pending_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_private(self._pending_path, (json.dumps(document, indent=2) + "\n").encode("utf-8"))
        except OSError as exc:
            logger.warning("Co-adaptation pending-erasure list cannot be written (%s)", type(exc).__name__)

    def rules_erased(self, participant_key: str) -> None:
        """The service deleted this participant's scoped rules from the preferences file."""
        self._pending_change(remove=PendingErasure(rules=(participant_key,)))

    def retry_pending_files(self) -> int:
        """Delete the single export files an earlier erase-today or consent update could not; returns how many."""
        pending = self.pending_erasure()
        if not pending.files:
            return 0
        deleted, failed = self._delete_files(self._exports_dir / name for name in pending.files)
        left = {path.name for path in failed}
        self._pending_change(remove=PendingErasure(files=tuple(name for name in pending.files if name not in left)))
        return deleted

    async def open(self, *, create: bool) -> None:
        """Open the database; with ``create=False`` this is a no-op when the file does not exist."""
        if self._pool is not None:
            return
        if not create and not self.exists:
            return
        pool = AsyncSqlitePool(self._db_path)
        await pool.start()
        try:
            async with pool.write() as conn:
                # Deleted rows are overwritten in the file (spec 10.6); the setting holds for this connection.
                await conn.execute("PRAGMA secure_delete=ON")
                await conn.executescript(SCHEMA_SQL)
                await conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', '1')")
                await conn.execute(
                    "INSERT OR IGNORE INTO meta(key, value) VALUES ('created_at', ?)", (_iso(self._clock()),)
                )
                await conn.commit()
        except BaseException:
            await pool.close()
            raise
        self._pool = pool
        _chmod_private(self._db_path)

    async def close(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            await pool.close()

    @contextlib.asynccontextmanager
    async def _write_tx(self) -> AsyncIterator[aiosqlite.Connection]:
        if self._pool is None:
            # No database means no consent was ever granted (or every participant is gone).
            raise CoAdaptationConsentError("no_consent")
        async with self._pool.write() as conn:
            await conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                await conn.rollback()
                raise
            else:
                await conn.commit()

    @contextlib.asynccontextmanager
    async def _read(self) -> AsyncIterator[aiosqlite.Connection]:
        if self._pool is None:
            raise RuntimeError("the co-adaptation store is not open")
        async with self._pool.read() as conn:
            yield conn

    async def _fetchall(self, sql: str, args: Iterable[Any] = ()) -> list[tuple[Any, ...]]:
        if self._pool is None:
            return []
        async with self._read() as conn, conn.execute(sql, tuple(args)) as cursor:
            return [tuple(row) for row in await cursor.fetchall()]

    @staticmethod
    async def _conn_fetchall(conn: aiosqlite.Connection, sql: str, args: Iterable[Any] = ()) -> list[tuple[Any, ...]]:
        async with conn.execute(sql, tuple(args)) as cursor:
            return [tuple(row) for row in await cursor.fetchall()]

    async def _key(self, participant_key: str) -> bytes:
        return await asyncio.to_thread(self._keys.get, participant_key)

    # ── consent gate ─────────────────────────────────────────────────────────

    async def _active_consent(self, conn: aiosqlite.Connection, participant_key: str) -> CoAdaptationConsentV1 | None:
        rows = await self._conn_fetchall(
            conn,
            "SELECT record_json FROM consents WHERE participant_key = ? AND revoked_at IS NULL AND replaced_at IS NULL"
            " ORDER BY granted_at DESC",
            (participant_key,),
        )
        if not rows:
            return None
        return CoAdaptationConsentV1.model_validate_json(rows[0][0])

    async def _gate(
        self,
        conn: aiosqlite.Connection,
        participant_key: str,
        purpose: CoAdaptationPurpose = _SELECTION_STUDY,
    ) -> CoAdaptationConsentV1:
        """Raise unless the participant holds an active consent covering ``purpose``. Runs before any key fetch."""
        record = await self._active_consent(conn, participant_key)
        if record is None:
            raise CoAdaptationConsentError("no_consent")
        record.require_active(purpose, now=self._clock())
        return record

    async def _session_owner(self, conn: aiosqlite.Connection, session_id: str) -> str | None:
        rows = await self._conn_fetchall(
            conn, "SELECT participant_key FROM sessions WHERE session_id = ?", (session_id,)
        )
        return rows[0][0] if rows else None

    async def _require_session_of(self, conn: aiosqlite.Connection, session_id: str, participant_key: str) -> None:
        if await self._session_owner(conn, session_id) != participant_key:
            raise ValueError("the session does not belong to this participant")

    # ── participants and consents ────────────────────────────────────────────

    async def add_participant(self, participant_key: str, created_at: datetime, label: str | None) -> None:
        """Create a participant and its key. The only caller of ``KeyProvider.get_or_create``."""
        _check_participant_key(participant_key)
        if label is not None and not 1 <= len(label) <= 32:
            raise ValueError("a display label has 1 to 32 characters")
        async with self._write_tx() as conn:
            existing = await self._conn_fetchall(
                conn, "SELECT 1 FROM participants WHERE participant_key = ?", (participant_key,)
            )
            if existing:
                raise ValueError("this participant already exists")
            key = await asyncio.to_thread(self._keys.get_or_create, participant_key)
            label_enc = (
                None if label is None else _encrypt_json(key, f"participants:{participant_key}", {"label": label})
            )
            await conn.execute(
                "INSERT INTO participants(participant_key, created_at, label_enc) VALUES (?, ?, ?)",
                (participant_key, _iso(created_at), label_enc),
            )

    async def set_participant_label(self, participant_key: str, label: str | None) -> None:
        if label is not None and not 1 <= len(label) <= 32:
            raise ValueError("a display label has 1 to 32 characters")
        async with self._write_tx() as conn:
            await self._gate(conn, participant_key)
            key = await self._key(participant_key)
            label_enc = (
                None if label is None else _encrypt_json(key, f"participants:{participant_key}", {"label": label})
            )
            await conn.execute(
                "UPDATE participants SET label_enc = ? WHERE participant_key = ?", (label_enc, participant_key)
            )

    async def participants(self) -> list[str]:
        rows = await self._fetchall("SELECT participant_key FROM participants ORDER BY created_at, participant_key")
        return [row[0] for row in rows]

    async def participant_labels(self) -> dict[str, str | None]:
        """Decrypted display labels for the status reply only; None when the key cannot be read. Never logged."""
        labels: dict[str, str | None] = {}
        for participant_key, label_enc in await self._fetchall(
            "SELECT participant_key, label_enc FROM participants ORDER BY created_at, participant_key"
        ):
            labels[participant_key] = None
            if label_enc is None:
                continue
            try:
                key = await self._key(participant_key)
                labels[participant_key] = _decrypt_json(key, f"participants:{participant_key}", label_enc)["label"]
            except Exception:
                labels[participant_key] = None
        return labels

    async def put_consent(
        self, record: CoAdaptationConsentV1, *, removed_purposes: frozenset[str] = frozenset()
    ) -> ConsentUpdateResult:
        """Insert a first consent, or replace the active one in one transaction (spec 10.3).

        An update sets ``replaced_at`` on the replaced row, inserts the new record and deletes what each removed
        purpose alone stored: the gaze fields of session and trial rows and the hand summaries are nulled and the rows
        re-encrypted in the same transaction; the export files go right after it commits. Removing gaze_prior or
        hand_pointer deletes the export files too, because every bundle written earlier holds the gaze regions, the
        gaze-check score and the hand summaries (spec 10.3, 10.9). A file that cannot be deleted now is retried from
        the pending-erasure list.
        """
        participant_key = record.participant_key
        exports_to_delete: list[Path] = []
        async with self._write_tx() as conn:
            exists = await self._conn_fetchall(
                conn, "SELECT 1 FROM participants WHERE participant_key = ?", (participant_key,)
            )
            if not exists:
                raise ValueError("unknown participant")
            active = await self._active_consent(conn, participant_key)
            if record.replaces_consent_id is None:
                if active is not None:
                    raise ValueError("this participant already holds an active consent")
                await self._insert_consent(conn, record)
                return ConsentUpdateResult(purposes=(), export_files=0, trials_rewritten=0, sessions_rewritten=0)
            if active is None or active.consent_id != record.replaces_consent_id:
                raise CoAdaptationConsentError("no_consent")
            active.require_active(_SELECTION_STUDY, now=record.granted_at)
            removed = frozenset(purpose for purpose in active.purposes if purpose not in record.purposes)
            if removed_purposes and frozenset(CoAdaptationPurpose(p) for p in removed_purposes) != removed:
                raise ValueError("removed_purposes must equal the purposes the update removes")
            await conn.execute(
                "UPDATE consents SET replaced_at = ? WHERE consent_id = ?",
                (_iso(record.granted_at), str(active.consent_id)),
            )
            await self._insert_consent(conn, record)
            sessions_rewritten = trials_rewritten = 0
            drop_gaze = CoAdaptationPurpose.GAZE_PRIOR in removed
            drop_hand = CoAdaptationPurpose.HAND_POINTER in removed
            if drop_gaze or drop_hand:
                key = await self._key(participant_key)
                if drop_gaze:
                    sessions_rewritten = await self._strip_session_gaze(conn, key, participant_key)
                trials_rewritten = await self._strip_trials(conn, key, participant_key, gaze=drop_gaze, hand=drop_hand)
            if drop_gaze or drop_hand or CoAdaptationPurpose.RESEARCH_EXPORT in removed:
                exports_to_delete = self._export_files(participant_key)
        deleted, left = self._delete_tracked(exports_to_delete)
        return ConsentUpdateResult(
            purposes=tuple(purpose.value for purpose in CoAdaptationPurpose if purpose in removed),
            export_files=deleted,
            trials_rewritten=trials_rewritten,
            sessions_rewritten=sessions_rewritten,
            export_files_left=left,
        )

    async def _insert_consent(self, conn: aiosqlite.Connection, record: CoAdaptationConsentV1) -> None:
        await conn.execute(
            "INSERT INTO consents(consent_id, participant_key, record_json, granted_at, expires_at, revoked_at,"
            " replaced_at) VALUES (?, ?, ?, ?, ?, NULL, NULL)",
            (
                str(record.consent_id),
                record.participant_key,
                json.dumps(record.model_dump(mode="json"), sort_keys=True, separators=(",", ":")),
                _iso(record.granted_at),
                _iso(record.expires_at),
            ),
        )

    async def _strip_session_gaze(self, conn: aiosqlite.Connection, key: bytes, participant_key: str) -> int:
        rewritten = 0
        for session_id, blob in await self._conn_fetchall(
            conn, "SELECT session_id, payload_enc FROM sessions WHERE participant_key = ?", (participant_key,)
        ):
            payload = _decrypt_json(key, f"sessions:{session_id}", blob)
            if payload.get("gaze_check_accuracy") is None:
                continue
            payload["gaze_check_accuracy"] = None
            await conn.execute(
                "UPDATE sessions SET payload_enc = ? WHERE session_id = ?",
                (_encrypt_json(key, f"sessions:{session_id}", payload), session_id),
            )
            rewritten += 1
        return rewritten

    async def _strip_trials(
        self, conn: aiosqlite.Connection, key: bytes, participant_key: str, *, gaze: bool, hand: bool
    ) -> int:
        rewritten = 0
        for trial_id, blob in await self._conn_fetchall(
            conn, "SELECT trial_id, payload_enc FROM trials WHERE participant_key = ?", (participant_key,)
        ):
            payload = _decrypt_json(key, f"trials:{trial_id}", blob)
            if not _strip_trial_fields(payload, gaze=gaze, hand=hand):
                continue
            await conn.execute(
                "UPDATE trials SET payload_enc = ? WHERE trial_id = ?",
                (_encrypt_json(key, f"trials:{trial_id}", payload), trial_id),
            )
            rewritten += 1
        return rewritten

    async def consents(self) -> list[tuple[CoAdaptationConsentV1, datetime | None, datetime | None]]:
        """Every consent record with its revoked_at and replaced_at, oldest first per participant."""
        rows = await self._fetchall(
            "SELECT record_json, revoked_at, replaced_at FROM consents ORDER BY participant_key, granted_at, rowid"
        )
        return [
            (CoAdaptationConsentV1.model_validate_json(record_json), _parse_time(revoked), _parse_time(replaced))
            for record_json, revoked, replaced in rows
        ]

    async def consent_history(self, participant_key: str) -> list[ConsentPeriod]:
        rows = await self._fetchall(
            "SELECT record_json, replaced_at FROM consents WHERE participant_key = ? ORDER BY granted_at, rowid",
            (participant_key,),
        )
        return [
            ConsentPeriod(
                record=CoAdaptationConsentV1.model_validate_json(record_json).to_core(),
                replaced_at=_parse_time(replaced),
            )
            for record_json, replaced in rows
        ]

    # ── sessions ─────────────────────────────────────────────────────────────

    async def put_session(self, row: SessionRow) -> None:
        participant_key = row.participant_key
        async with self._write_tx() as conn:
            consent = await self._gate(conn, participant_key)
            key = await self._key(participant_key)
            payload = row.to_dict()
            if not consent.covers(CoAdaptationPurpose.GAZE_PRIOR):
                payload["gaze_check_accuracy"] = None
            await conn.execute(
                "INSERT INTO sessions(session_id, participant_key, context, day_index, session_number, protocol_id,"
                " protocol_sha256, evidence_kind, started_at, ended_at, end_reason, payload_enc)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row.session_id,
                    participant_key,
                    row.context,
                    int(row.day_index),
                    int(row.session_number),
                    row.protocol_id,
                    row.protocol_sha256,
                    _enum_value(row.evidence_kind),
                    _iso(row.started_at),
                    None if row.ended_at is None else _iso(row.ended_at),
                    row.end_reason,
                    _encrypt_json(key, f"sessions:{row.session_id}", payload),
                ),
            )

    async def close_session(self, session_id: str, ended_at: datetime, reason: str) -> None:
        """Set ended_at and end_reason (plaintext columns); replay reads them with the session payload."""
        async with self._write_tx() as conn:
            owner = await self._session_owner(conn, session_id)
            if owner is None:
                raise ValueError("unknown session")
            await self._gate(conn, owner)
            await conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE session_id = ? AND ended_at IS NULL",
                (_iso(ended_at), reason, session_id),
            )

    async def open_sessions(self) -> list[tuple[str, str]]:
        """(participant_key, session_id) of every session without ended_at. Plaintext columns only, no key."""
        rows = await self._fetchall(
            "SELECT participant_key, session_id FROM sessions WHERE ended_at IS NULL ORDER BY started_at, session_id"
        )
        return [(row[0], row[1]) for row in rows]

    async def sessions_summary(self, participant_key: str) -> list[SessionInfo]:
        """Plaintext facts about the participant's sessions, oldest first. Needs no key."""
        rows = await self._fetchall(
            "SELECT s.session_id, s.session_number, s.day_index, s.started_at, s.ended_at, s.end_reason,"
            " (SELECT COUNT(*) FROM trials t WHERE t.session_id = s.session_id)"
            " FROM sessions s WHERE s.participant_key = ? ORDER BY s.started_at, s.session_id",
            (participant_key,),
        )
        return [
            SessionInfo(
                session_id=row[0],
                session_number=int(row[1]),
                day_index=int(row[2]),
                started_at=_parse_time(row[3]),  # type: ignore[arg-type]
                ended_at=_parse_time(row[4]),
                end_reason=row[5],
                trials=int(row[6]),
            )
            for row in rows
        ]

    # ── rule history ─────────────────────────────────────────────────────────

    async def put_rule(self, participant_key: str, pref: Preference) -> None:
        """Upsert a rule into the participant's rule history (spec 10.6); a retraction updates the row in place."""
        async with self._write_tx() as conn:
            await self._gate(conn, participant_key)
            key = await self._key(participant_key)
            payload_enc = _encrypt_json(key, f"rules:{participant_key}:{pref.pref_id}", preference_to_dict(pref))
            await conn.execute(
                "INSERT INTO rules(participant_key, pref_id, created_at, retracted_at, payload_enc)"
                " VALUES (?, ?, ?, ?, ?) ON CONFLICT(participant_key, pref_id) DO UPDATE SET"
                " created_at = excluded.created_at, retracted_at = excluded.retracted_at,"
                " payload_enc = excluded.payload_enc",
                (
                    participant_key,
                    pref.pref_id,
                    _iso(pref.created_at),
                    None if pref.retracted_at is None else _iso(pref.retracted_at),
                    payload_enc,
                ),
            )

    async def rules(self, participant_key: str) -> list[Preference]:
        rows = await self._fetchall(
            "SELECT pref_id, payload_enc FROM rules WHERE participant_key = ? ORDER BY created_at, pref_id",
            (participant_key,),
        )
        if not rows:
            return []
        key = await self._key(participant_key)
        return [
            preference_from_dict(_decrypt_json(key, f"rules:{participant_key}:{pref_id}", blob))
            for pref_id, blob in rows
        ]

    # ── decisions, trials, burden, events ────────────────────────────────────

    async def decision_exists(self, decision_id: str) -> bool:
        """True when any participant's stored decision has this id (plaintext column, no key)."""
        return bool(await self._fetchall("SELECT 1 FROM decisions WHERE decision_id = ?", (decision_id,)))

    async def put_decision(
        self, record: DecisionRecord, *, participant_key: str, supersedes: str | None = None
    ) -> None:
        """Insert a decision; with ``supersedes``, close that earlier table in the same transaction (spec 9.3)."""
        async with self._write_tx() as conn:
            await self._gate(conn, participant_key)
            await self._require_session_of(conn, record.session_id, participant_key)
            key = await self._key(participant_key)
            versions = dict(record.versions or {})
            await conn.execute(
                "INSERT INTO decisions(decision_id, session_id, participant_key, seq, level, mode, behavior,"
                " policy_version, model_version, prior_version, feature_schema, created_at, superseded_by,"
                " payload_enc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.decision_id,
                    record.session_id,
                    participant_key,
                    int(record.seq),
                    _enum_value(record.level),
                    _enum_value(record.mode),
                    record.behavior,
                    _version(versions, "policy", POLICY_VERSION),
                    _version(versions, "model", MODEL_VERSION),
                    _version(versions, "prior", PRIOR_VERSION),
                    _version(versions, "feature_schema", FEATURE_SCHEMA),
                    _iso(record.created_at),
                    record.superseded_by,
                    _encrypt_json(key, f"decisions:{record.decision_id}", record.to_dict()),
                ),
            )
            if supersedes is not None:
                await self._rewrite_decision_in(
                    conn,
                    key,
                    participant_key,
                    supersedes,
                    {"superseded_by": record.decision_id},
                    {"superseded_by": record.decision_id},
                )

    @staticmethod
    async def _rewrite_decision_in(
        conn: aiosqlite.Connection,
        key: bytes,
        participant_key: str,
        decision_id: str,
        changes: dict[str, Any],
        columns: dict[str, Any],
    ) -> bool:
        """Rewrite one decision of ``participant_key`` inside an open write transaction; False when it is unknown."""
        rows = await CoAdaptationStore._conn_fetchall(
            conn, "SELECT participant_key, payload_enc FROM decisions WHERE decision_id = ?", (decision_id,)
        )
        if not rows:
            return False
        owner, blob = rows[0]
        if owner != participant_key:
            raise ValueError("the decision does not belong to this participant")
        payload = _decrypt_json(key, f"decisions:{decision_id}", blob)
        payload.update(changes)
        # Validate the rewritten payload with the record's own reader before it is stored.
        DecisionRecord.from_dict(payload)
        assignments = ", ".join(f"{column} = ?" for column in columns)
        await conn.execute(
            f"UPDATE decisions SET {assignments}, payload_enc = ? WHERE decision_id = ?",
            (*columns.values(), _encrypt_json(key, f"decisions:{decision_id}", payload), decision_id),
        )
        return True

    async def _rewrite_decision(self, decision_id: str, changes: dict[str, Any], columns: dict[str, Any]) -> bool:
        async with self._write_tx() as conn:
            rows = await self._conn_fetchall(
                conn, "SELECT participant_key FROM decisions WHERE decision_id = ?", (decision_id,)
            )
            if not rows:
                return False
            participant_key = rows[0][0]
            await self._gate(conn, participant_key)
            key = await self._key(participant_key)
            return await self._rewrite_decision_in(conn, key, participant_key, decision_id, changes, columns)

    async def supersede_decision(self, decision_id: str, by: str) -> None:
        await self._rewrite_decision(decision_id, {"superseded_by": by}, {"superseded_by": by})

    async def relabel_decision(self, decision_id: str, *, applied: str | None, fallback_reason: str) -> None:
        """UI fallback relabelling (spec 9.7): behaviour 'baseline', the applied variant (None for a table)."""
        relabel = FallbackRelabel(decision_id=decision_id, applied=applied, fallback_reason=fallback_reason)
        await self._rewrite_decision(decision_id, relabel.changes(), {"behavior": "baseline"})

    async def trial_owner(self, trial_id: str) -> tuple[str, str] | None:
        """(participant_key, session_id) of a stored trial, from plaintext columns; None when it is not stored."""
        rows = await self._fetchall("SELECT participant_key, session_id FROM trials WHERE trial_id = ?", (trial_id,))
        return (rows[0][0], rows[0][1]) if rows else None

    async def read_trial(self, participant_key: str, trial_id: str) -> StoredTrial | None:
        rows = await self._fetchall(
            "SELECT payload_enc FROM trials WHERE trial_id = ? AND participant_key = ?", (trial_id, participant_key)
        )
        if not rows:
            return None
        key = await self._key(participant_key)
        return StoredTrial.from_dict(_decrypt_json(key, f"trials:{trial_id}", rows[0][0]))

    async def trials_since(self, participant_key: str, since: datetime) -> list[StoredTrial]:
        """The participant's trials recorded at or after ``since``, oldest first; decrypted off the event loop."""
        rows = await self._fetchall(
            "SELECT trial_id, payload_enc FROM trials WHERE participant_key = ? AND recorded_at >= ?"
            " ORDER BY recorded_at, seq",
            (participant_key, _iso(since)),
        )
        if not rows:
            return []
        key = await self._key(participant_key)

        def decode() -> list[StoredTrial]:
            return [StoredTrial.from_dict(_decrypt_json(key, f"trials:{trial_id}", blob)) for trial_id, blob in rows]

        return await asyncio.to_thread(decode)

    async def put_trial(
        self,
        observation: TrialObservation,
        *,
        participant_key: str,
        relabels: Iterable[FallbackRelabel] = (),
    ) -> None:
        """Write the StoredTrial, every alarm event it fired and the fallback relabels it carries in one transaction.

        All or nothing: a trial is never stored without the relabelling of the decisions it fell back from (9.7).
        """
        stored = observation.stored
        trial = stored.trial
        relabels = tuple(relabels)
        async with self._write_tx() as conn:
            consent = await self._gate(conn, participant_key)
            await self._require_session_of(conn, trial.session_id, participant_key)
            key = await self._key(participant_key)
            payload = stored.to_dict()
            _strip_trial_fields(
                payload,
                gaze=not consent.covers(CoAdaptationPurpose.GAZE_PRIOR),
                hand=not consent.covers(CoAdaptationPurpose.HAND_POINTER),
            )
            await conn.execute(
                "INSERT INTO trials(trial_id, session_id, participant_key, seq, block_index, trial_index, task,"
                " recorded_at, verification, payload_enc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trial.trial_id,
                    trial.session_id,
                    participant_key,
                    int(trial.seq),
                    int(trial.block_index),
                    int(trial.trial_index),
                    _enum_value(trial.task),
                    _iso(trial.recorded_at),
                    _enum_value(stored.verification),
                    _encrypt_json(key, f"trials:{trial.trial_id}", payload),
                ),
            )
            for event in observation.events:
                if event.kind not in ALARM_EVENT_KINDS:
                    raise ValueError("put_trial stores drift and calibration events only")
                await conn.execute(
                    "INSERT INTO events(event_id, participant_key, kind, session_id, trial_id, created_at,"
                    " payload_enc) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        event.event_id,
                        participant_key,
                        event.kind,
                        event.session_id,
                        event.trial_id,
                        _iso(event.created_at),
                        _encrypt_json(key, f"events:{event.event_id}", event.to_dict()),
                    ),
                )
            for relabel in relabels:
                await self._rewrite_decision_in(
                    conn, key, participant_key, relabel.decision_id, relabel.changes(), {"behavior": "baseline"}
                )

    async def put_burden(self, report: BurdenReport, *, participant_key: str) -> None:
        async with self._write_tx() as conn:
            await self._gate(conn, participant_key)
            await self._require_session_of(conn, report.session_id, participant_key)
            key = await self._key(participant_key)
            await conn.execute(
                "INSERT INTO burden(report_id, session_id, participant_key, seq, block_index, scope, recorded_at,"
                " payload_enc) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    report.report_id,
                    report.session_id,
                    participant_key,
                    int(report.seq),
                    report.block_index,
                    report.scope,
                    _iso(report.recorded_at),
                    _encrypt_json(key, f"burden:{report.report_id}", report.to_dict()),
                ),
            )

    async def put_event(self, participant_key: str, kind: str, payload: dict, created_at: datetime) -> None:
        """Bookkeeping events only (prune, erase_today, learning_reset); alarms are written by put_trial."""
        if kind not in BOOKKEEPING_EVENT_KINDS:
            raise ValueError("put_event stores bookkeeping events only")
        async with self._write_tx() as conn:
            await self._gate(conn, participant_key)
            key = await self._key(participant_key)
            event_id = str(uuid.uuid4())
            await conn.execute(
                "INSERT INTO events(event_id, participant_key, kind, session_id, trial_id, created_at, payload_enc)"
                " VALUES (?, ?, ?, NULL, NULL, ?, ?)",
                (event_id, participant_key, kind, _iso(created_at), _encrypt_json(key, f"events:{event_id}", payload)),
            )

    # ── model snapshot ───────────────────────────────────────────────────────

    async def load_snapshot(self, participant_key: str) -> tuple[bytes, str, str, str] | None:
        rows = await self._fetchall(
            "SELECT state_enc, rows_hash, model_version, prior_version FROM model_state WHERE participant_key = ?",
            (participant_key,),
        )
        if not rows:
            return None
        state_enc, rows_hash, model_version, prior_version = rows[0]
        key = await self._key(participant_key)
        return _decrypt(key, f"model_state:{participant_key}", state_enc), rows_hash, model_version, prior_version

    async def save_snapshot(self, participant_key: str, state: bytes, rows_hash: str, now: datetime) -> None:
        async with self._write_tx() as conn:
            await self._gate(conn, participant_key)
            key = await self._key(participant_key)
            await conn.execute(
                "INSERT INTO model_state(participant_key, model_version, prior_version, rows_hash, updated_at,"
                " state_enc) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(participant_key) DO UPDATE SET"
                " model_version = excluded.model_version, prior_version = excluded.prior_version,"
                " rows_hash = excluded.rows_hash, updated_at = excluded.updated_at, state_enc = excluded.state_enc",
                (
                    participant_key,
                    MODEL_VERSION,
                    PRIOR_VERSION,
                    rows_hash,
                    _iso(now),
                    _encrypt(key, f"model_state:{participant_key}", bytes(state)),
                ),
            )

    async def rows_hash(self, participant_key: str) -> str:
        """sha256 over the sorted ids of the retained rows replay reads, plus the learning-reset marker."""
        ids: list[str] = []
        for table, column in (
            ("sessions", "session_id"),
            ("trials", "trial_id"),
            ("burden", "report_id"),
            ("events", "event_id"),
        ):
            rows = await self._fetchall(f"SELECT {column} FROM {table} WHERE participant_key = ?", (participant_key,))
            ids.extend(f"{table}:{row[0]}" for row in rows)
        marker = await self._learning_reset_marker(participant_key)
        text = "\n".join(sorted(ids)) + f"\nlearning_reset_at:{marker or ''}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def mark_learning_reset(self, participant_key: str, at: datetime) -> None:
        async with self._write_tx() as conn:
            await self._gate(conn, participant_key)
            await conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (f"learning_reset_at:{participant_key}", _iso(at)),
            )

    async def _learning_reset_marker(self, participant_key: str) -> str | None:
        rows = await self._fetchall("SELECT value FROM meta WHERE key = ?", (f"learning_reset_at:{participant_key}",))
        return rows[0][0] if rows else None

    async def learning_reset_at(self, participant_key: str) -> datetime | None:
        return _parse_time(await self._learning_reset_marker(participant_key))

    # ── reading rows back ────────────────────────────────────────────────────

    async def _session_rows_for(self, participant_key: str, *, session_id: str | None = None) -> list[tuple[Any, ...]]:
        if session_id is not None:
            return await self._fetchall(
                "SELECT session_id, started_at, ended_at, end_reason, payload_enc FROM sessions"
                " WHERE participant_key = ? AND session_id = ?",
                (participant_key, session_id),
            )
        return await self._fetchall(
            "SELECT session_id, started_at, ended_at, end_reason, payload_enc FROM sessions"
            " WHERE participant_key = ? ORDER BY started_at, session_id",
            (participant_key,),
        )

    @staticmethod
    def _session_from_row(key: bytes, row: tuple[Any, ...]) -> SessionRow:
        session_id, _started, ended_at, end_reason, blob = row
        payload = _decrypt_json(key, f"sessions:{session_id}", blob)
        # ended_at and end_reason live in plaintext columns, set when the session closes.
        payload["ended_at"] = None if ended_at is None else datetime.fromisoformat(ended_at).isoformat()
        payload["end_reason"] = end_reason
        return SessionRow.from_dict(payload)

    async def _fetch_session_scoped(
        self, sql: str, participant_key: str, session_ids: list[str], *, every_session: bool, order: str = ""
    ) -> list[tuple[Any, ...]]:
        """Rows of ``sql`` (``... WHERE participant_key = ?``) for the given sessions, filtered in SQL."""
        if every_session:
            return await self._fetchall(sql + order, (participant_key,))
        placeholders = ", ".join("?" for _ in session_ids)
        return await self._fetchall(f"{sql} AND session_id IN ({placeholders}){order}", (participant_key, *session_ids))

    async def _rows_of_sessions(
        self,
        participant_key: str,
        key: bytes,
        session_ids: list[str],
        *,
        every_session: bool,
        with_decisions: bool = True,
    ) -> tuple[
        dict[str, list[DecisionRecord]],
        dict[str, list[tuple[int, Any]]],
        dict[str, list[EngineEvent]],
    ]:
        """Decisions, trials and burden reports (merged by seq) and alarm events per session.

        Only the rows of ``session_ids`` are read (``every_session``: the participant's whole history), and they are
        decrypted and decoded in a worker thread.
        """
        if not session_ids:
            return {}, {}, {}
        scoped = self._fetch_session_scoped
        decision_rows = (
            await scoped(
                "SELECT decision_id, session_id, payload_enc FROM decisions WHERE participant_key = ?",
                participant_key,
                session_ids,
                every_session=every_session,
                order=" ORDER BY seq, decision_id",
            )
            if with_decisions
            else []
        )
        trial_rows = await scoped(
            "SELECT trial_id, session_id, seq, payload_enc FROM trials WHERE participant_key = ?",
            participant_key,
            session_ids,
            every_session=every_session,
        )
        burden_rows = await scoped(
            "SELECT report_id, session_id, seq, payload_enc FROM burden WHERE participant_key = ?",
            participant_key,
            session_ids,
            every_session=every_session,
        )
        event_rows = await scoped(
            "SELECT event_id, kind, session_id, trial_id, payload_enc FROM events WHERE participant_key = ?",
            participant_key,
            session_ids,
            every_session=every_session,
            order=" ORDER BY rowid",
        )

        def decode() -> tuple[
            dict[str, list[DecisionRecord]], dict[str, list[tuple[int, Any]]], dict[str, list[EngineEvent]]
        ]:
            decisions: dict[str, list[DecisionRecord]] = {sid: [] for sid in session_ids}
            rows_by_seq: dict[str, list[tuple[int, Any]]] = {sid: [] for sid in session_ids}
            events: dict[str, list[EngineEvent]] = {}
            for decision_id, session_id, blob in decision_rows:
                if session_id in decisions:
                    decisions[session_id].append(
                        DecisionRecord.from_dict(_decrypt_json(key, f"decisions:{decision_id}", blob))
                    )
            for trial_id, session_id, seq, blob in trial_rows:
                if session_id in rows_by_seq:
                    stored = StoredTrial.from_dict(_decrypt_json(key, f"trials:{trial_id}", blob))
                    rows_by_seq[session_id].append((int(seq), stored))
            for report_id, session_id, seq, blob in burden_rows:
                if session_id in rows_by_seq:
                    report = BurdenReport.from_dict(_decrypt_json(key, f"burden:{report_id}", blob))
                    rows_by_seq[session_id].append((int(seq), report))
            for event_id, kind, session_id, trial_id, blob in event_rows:
                if kind in ALARM_EVENT_KINDS and session_id in rows_by_seq and trial_id is not None:
                    event = EngineEvent.from_dict(_decrypt_json(key, f"events:{event_id}", blob))
                    events.setdefault(trial_id, []).append(event)
            for rows in rows_by_seq.values():
                rows.sort(key=lambda item: item[0])
            return decisions, rows_by_seq, events

        return await asyncio.to_thread(decode)

    async def _ordered_rows(
        self,
        participant_key: str,
        *,
        after_reset: bool,
        day_index: int | None = None,
        session_id: str | None = None,
        with_decisions: bool = True,
    ) -> tuple[list[SessionRow], dict[str, list[DecisionRecord]], dict[str, list[Any]]]:
        """Sessions in replay order with their rows: trials and burden reports by seq, each event after its trial.

        ``session_id`` reads that one session only; ``day_index`` keeps the sessions of one day.
        """
        session_rows = await self._session_rows_for(participant_key, session_id=session_id)
        if not session_rows:
            return [], {}, {}
        key = await self._key(participant_key)
        marker = await self.learning_reset_at(participant_key) if after_reset else None

        def decode_sessions() -> list[SessionRow]:
            kept: list[SessionRow] = []
            for row in session_rows:
                session = self._session_from_row(key, row)
                if day_index is not None and session.day_index != day_index:
                    continue
                if marker is not None and (
                    session.started_at < marker or (session.ended_at is not None and session.ended_at <= marker)
                ):
                    # Rows from before a learning reset are kept for the study but are not replayed (spec 10.8).
                    continue
                kept.append(session)
            return kept

        sessions = await asyncio.to_thread(decode_sessions)
        session_ids = [session.session_id for session in sessions]
        every_session = session_id is None and len(sessions) == len(session_rows)
        decisions, rows_by_seq, events = await self._rows_of_sessions(
            participant_key, key, session_ids, every_session=every_session, with_decisions=with_decisions
        )
        ordered: dict[str, list[Any]] = {}
        for sid in session_ids:
            items: list[Any] = []
            for _seq, item in rows_by_seq[sid]:
                items.append(item)
                if isinstance(item, StoredTrial):
                    items.extend(events.get(item.trial.trial_id, ()))
            ordered[sid] = items
        return sessions, decisions, ordered

    async def replay_rows(self, participant_key: str) -> AsyncIterator[ReplayItem]:
        """Every retained row after the learning-reset marker, in the replay order of spec 10.8."""
        sessions, _decisions, ordered = await self._ordered_rows(
            participant_key, after_reset=True, with_decisions=False
        )
        for session in sessions:
            yield session
            for item in ordered[session.session_id]:
                yield item

    async def results_rows(self, participant_key: str, *, day_index: int | None) -> ResultsRows:
        """Stored rows of every retained session (``day_index`` None) or of one day, read with ``from_dict``."""
        sessions, decisions, ordered = await self._ordered_rows(participant_key, after_reset=False, day_index=day_index)
        return ResultsRows(
            sessions=tuple(sessions),
            decisions=tuple(record for session in sessions for record in decisions[session.session_id]),
            trials=tuple(
                item for session in sessions for item in ordered[session.session_id] if isinstance(item, StoredTrial)
            ),
            burden=tuple(
                item for session in sessions for item in ordered[session.session_id] if isinstance(item, BurdenReport)
            ),
        )

    async def session_rows(self, participant_key: str, session_number: int) -> SessionRows | None:
        """One stored session with its decisions, trials and burden reports; None when it no longer exists."""
        candidates = [
            info for info in await self.sessions_summary(participant_key) if info.session_number == session_number
        ]
        if not candidates:
            return None
        # session_number counts only sessions with recorded trials, so prefer the one that has them.
        chosen = max(candidates, key=lambda info: (info.trials, -info.started_at.timestamp()))
        # Only that session's rows are read and decrypted, never the whole history.
        sessions, decisions, ordered = await self._ordered_rows(
            participant_key, after_reset=False, session_id=chosen.session_id
        )
        for session in sessions:
            if session.session_id == chosen.session_id:
                items = ordered[session.session_id]
                return SessionRows(
                    session=session,
                    decisions=tuple(decisions[session.session_id]),
                    trials=tuple(item for item in items if isinstance(item, StoredTrial)),
                    burden=tuple(item for item in items if isinstance(item, BurdenReport)),
                )
        return None

    # ── deletion ─────────────────────────────────────────────────────────────

    def _export_files(self, participant_key: str) -> list[Path]:
        if not self._exports_dir.is_dir():
            return []
        found: list[Path] = []
        for path in self._exports_dir.iterdir():
            match = _EXPORT_NAME.fullmatch(path.name)
            if match and match.group(1) == participant_key:
                found.append(path)
        return sorted(found)

    @staticmethod
    def _delete_files(paths: Iterable[Path]) -> tuple[int, list[Path]]:
        """Delete each file. Returns how many were deleted and the ones that could not be deleted now (a file another
        program holds open on Windows, a read-only attribute); those are never raised, they are retried later."""
        deleted = 0
        failed: list[Path] = []
        for path in paths:
            try:
                path.unlink()
                deleted += 1
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.warning("Co-adaptation could not delete an export file yet (%s)", type(exc).__name__)
                failed.append(path)
        return deleted, failed

    def _delete_tracked(self, paths: Iterable[Path]) -> tuple[int, int]:
        """Delete single export files; the ones left go on the pending-erasure list. Returns (deleted, left)."""
        deleted, failed = self._delete_files(paths)
        if failed:
            self._pending_change(add=PendingErasure(files=tuple(path.name for path in failed)))
        return deleted, len(failed)

    async def _compact(self) -> None:
        if self._pool is None:
            return
        async with self._pool.write() as conn:
            await conn.execute("VACUUM")
        await self._pool.checkpoint("TRUNCATE")

    async def erase_participant(self, participant_key: str) -> EraseResult:
        """Revoke semantics (spec 10.7): every row, every export file, then the key; needs no key or consent.

        Best-effort and resumable. The participant goes on the pending-erasure list before anything is deleted and
        comes off it only when its rows, its export files and its keyring key are all gone, so an export file that
        cannot be deleted now or a keyring that cannot be reached is retried (``PendingErasure.participants``) instead
        of being forgotten once the rows are gone. Its participant-scoped rules stay listed until the service has
        deleted them (``rules_erased``). Running it again for an erased participant completes what is left.
        """
        _check_participant_key(participant_key)
        self._pending_change(add=PendingErasure(participants=(participant_key,), rules=(participant_key,)))
        if self._pool is None and self.exists:
            await self.open(create=False)
        rows = 0
        if self._pool is not None:
            async with self._write_tx() as conn:
                for table in _PARTICIPANT_TABLES:
                    cursor = await conn.execute(f"DELETE FROM {table} WHERE participant_key = ?", (participant_key,))
                    rows += max(cursor.rowcount, 0)
                    await cursor.close()
                await conn.execute("DELETE FROM meta WHERE key = ?", (f"learning_reset_at:{participant_key}",))
        export_files, left = self._delete_files(self._export_files(participant_key))
        try:
            await asyncio.to_thread(self._keys.delete, participant_key)
            key_destroyed = True
        except CoAdaptationKeyUnavailable:
            # Retried from the pending-erasure list until the keyring can be reached again (crypto-shredding).
            key_destroyed = False
        if self._pool is not None:
            try:
                remaining = await self._fetchall("SELECT COUNT(*) FROM participants")
                if remaining and remaining[0][0] == 0:
                    await self.close()
                    for suffix in ("", "-wal", "-shm"):
                        with contextlib.suppress(FileNotFoundError):
                            Path(f"{self._db_path}{suffix}").unlink()
                elif rows:
                    await self._compact()
            except Exception as exc:
                # Deleted rows are already overwritten (secure_delete); compaction only shrinks the file.
                logger.warning("Co-adaptation compaction after an erase failed (%s)", type(exc).__name__)
        result = EraseResult(
            rows=rows, key_destroyed=key_destroyed, export_files=export_files, export_files_left=len(left)
        )
        if result.complete:
            self._pending_change(remove=PendingErasure(participants=(participant_key,)))
        return result

    async def _delete_sessions(self, conn: aiosqlite.Connection, session_ids: Iterable[str]) -> int:
        rows = 0
        for session_id in session_ids:
            for table in ("decisions", "trials", "burden", "events", "sessions"):
                cursor = await conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
                rows += max(cursor.rowcount, 0)
                await cursor.close()
        return rows

    async def erase_day(self, participant_key: str, day_index: int) -> EraseDayResult:
        """Delete one day's sessions with their rows, and the exports that can contain them (spec 10.7).

        The participant's model snapshot goes in the same transaction: it was learned from the erased rows, and
        deleting it needs no key, so it never outlives them even when the keyring is unavailable and the service
        cannot rebuild now; the next engine load rebuilds from the retained rows (spec 10.8).
        """
        if self._pool is None:
            return EraseDayResult(rows=0, export_files=0)
        async with self._write_tx() as conn:
            sessions = await self._conn_fetchall(
                conn,
                "SELECT session_id, started_at FROM sessions WHERE participant_key = ? AND day_index = ?",
                (participant_key, int(day_index)),
            )
            if not sessions:
                return EraseDayResult(rows=0, export_files=0)
            rows = await self._delete_sessions(conn, [session_id for session_id, _ in sessions])
            await conn.execute("DELETE FROM model_state WHERE participant_key = ?", (participant_key,))
        earliest = min(datetime.fromisoformat(started) for _, started in sessions)
        # File names carry whole seconds, so compare at that resolution and err towards deleting.
        threshold = earliest.astimezone(UTC).replace(microsecond=0)
        doomed = []
        for path in self._export_files(participant_key):
            match = _EXPORT_NAME.fullmatch(path.name)
            if match is None:
                continue
            written = datetime.strptime(match.group(2), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
            if written >= threshold:
                doomed.append(path)
        export_files, left = self._delete_tracked(doomed)
        try:
            await self._compact()
        except Exception as exc:
            logger.warning("Co-adaptation compaction after an erase failed (%s)", type(exc).__name__)
        return EraseDayResult(rows=rows, export_files=export_files, export_files_left=left)

    async def prune(self, now: datetime) -> list[str]:
        """Erase every participant without a consent in force at ``now``; one failure never stops the others."""
        if self._pool is None:
            return []
        active = {
            record.participant_key
            for record, revoked_at, replaced_at in await self.consents()
            if revoked_at is None and replaced_at is None and now < record.expires_at
        }
        erased: list[str] = []
        for participant_key in await self.participants():
            if participant_key in active:
                continue
            try:
                await self.erase_participant(participant_key)
            except Exception as exc:
                logger.warning("Co-adaptation prune failed for %s (%s)", participant_key, type(exc).__name__)
                continue
            erased.append(participant_key)
        return erased

    async def apply_decision_cap(self, participant_key: str, *, keep: int | None = None) -> int:
        """Delete the participant's oldest closed sessions while it holds more than ``keep`` decisions (spec 10.6).

        Called only by ``session_begin`` before a new session opens; never deletes an open session. Returns the number
        of sessions deleted and adds it to the ``pruned_sessions`` counter.
        """
        if self._pool is None:
            return 0
        limit = DECISION_CAP_KEEP if keep is None else keep
        deleted = 0
        async with self._write_tx() as conn:
            while True:
                count = (
                    await self._conn_fetchall(
                        conn, "SELECT COUNT(*) FROM decisions WHERE participant_key = ?", (participant_key,)
                    )
                )[0][0]
                if count <= limit:
                    break
                oldest = await self._conn_fetchall(
                    conn,
                    "SELECT session_id FROM sessions WHERE participant_key = ? AND ended_at IS NOT NULL"
                    " ORDER BY started_at, session_id LIMIT 1",
                    (participant_key,),
                )
                if not oldest:
                    break
                await self._delete_sessions(conn, [oldest[0][0]])
                deleted += 1
            if deleted:
                # The snapshot was learned from the deleted sessions; the next engine load rebuilds (spec 10.8).
                await conn.execute("DELETE FROM model_state WHERE participant_key = ?", (participant_key,))
                await conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('pruned_sessions', ?) ON CONFLICT(key) DO UPDATE SET"
                    " value = CAST(CAST(value AS INTEGER) + ? AS TEXT)",
                    (str(deleted), deleted),
                )
        return deleted

    # ── status ───────────────────────────────────────────────────────────────

    async def counts(self) -> dict[str, int | str | None]:
        empty: dict[str, int | str | None] = {
            "exists": False,
            "bytes": 0,
            "participants": 0,
            "sessions": 0,
            "decisions": 0,
            "trials": 0,
            "burden_reports": 0,
            "oldest_recorded_at": None,
            "pruned_sessions": 0,
        }
        if not self.exists:
            return empty
        size = 0
        for suffix in ("", "-wal", "-shm"):
            with contextlib.suppress(OSError):
                size += Path(f"{self._db_path}{suffix}").stat().st_size
        result = {**empty, "exists": True, "bytes": size}
        if self._pool is None:
            return result
        for name, table in (
            ("participants", "participants"),
            ("sessions", "sessions"),
            ("decisions", "decisions"),
            ("trials", "trials"),
            ("burden_reports", "burden"),
        ):
            result[name] = int((await self._fetchall(f"SELECT COUNT(*) FROM {table}"))[0][0])
        oldest = await self._fetchall(
            "SELECT MIN(recorded_at) FROM (SELECT recorded_at FROM trials UNION ALL SELECT recorded_at FROM burden)"
        )
        result["oldest_recorded_at"] = oldest[0][0] if oldest else None
        pruned = await self._fetchall("SELECT value FROM meta WHERE key = 'pruned_sessions'")
        result["pruned_sessions"] = int(pruned[0][0]) if pruned else 0
        return result

    # ── export ───────────────────────────────────────────────────────────────

    async def export(self, participant_key: str, now: datetime) -> tuple[Path, str, int]:
        """Write the encrypted bundle of spec 10.9 and return (path, sha256, size). Requires research_export."""
        if self._pool is None:
            raise CoAdaptationConsentError("no_consent")
        async with self._read() as conn:
            record = await self._active_consent(conn, participant_key)
        if record is None:
            raise CoAdaptationConsentError("no_consent")
        record.require_active(CoAdaptationPurpose.RESEARCH_EXPORT, now=self._clock())
        key = await self._key(participant_key)
        sessions, decisions, ordered = await self._ordered_rows(participant_key, after_reset=False)
        trials: list[dict[str, Any]] = []
        burden: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        for session in sessions:
            for item in ordered[session.session_id]:
                if isinstance(item, StoredTrial):
                    trials.append(item.to_dict())
                elif isinstance(item, BurdenReport):
                    burden.append(item.to_dict())
                else:
                    events.append(item.to_dict())
        marker = await self.learning_reset_at(participant_key)
        bundle = {
            "bundle_schema": BUNDLE_SCHEMA,
            "record_schema": RECORD_SCHEMA_VERSION,
            "consents": [period.to_dict() for period in await self.consent_history(participant_key)],
            "sessions": [session.to_dict() for session in sessions],
            "decisions": [record.to_dict() for session in sessions for record in decisions[session.session_id]],
            "trials": trials,
            "burden": burden,
            "events": events,
            "preferences": [preference_to_dict(pref) for pref in await self.rules(participant_key)],
            "learning_reset_at": None if marker is None else marker.isoformat(),
            "versions": _versions_bundle(),
        }
        nonce = os.urandom(12)
        header = {
            "schema_version": 1,
            "participant_key": participant_key,
            "created_at": _iso(now),
            "cipher": "AES-256-GCM",
            "nonce": base64.b64encode(nonce).decode("ascii"),
        }
        header_line = json.dumps(header, sort_keys=True, separators=(",", ":"))
        ciphertext = AESGCM(key).encrypt(nonce, _canonical(bundle), header_line.encode("utf-8"))
        content = (header_line + "\n" + base64.b64encode(ciphertext).decode("ascii") + "\n").encode("utf-8")
        path = self._next_export_path(participant_key, now)
        self._write_private(path, content)
        return path, hashlib.sha256(content).hexdigest(), len(content)

    def _next_export_path(self, participant_key: str, now: datetime) -> Path:
        self._exports_dir.mkdir(parents=True, exist_ok=True)
        stamp = now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = self._exports_dir / f"coadapt_{participant_key}_{stamp}.enc"
        counter = 1
        while path.exists():
            path = self._exports_dir / f"coadapt_{participant_key}_{stamp}_{counter}.enc"
            counter += 1
        return path

    @staticmethod
    def _write_private(path: Path, content: bytes) -> None:
        fd, tmp_name = tempfile.mkstemp(prefix=".coadapt_export.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            _chmod_private(Path(tmp_name))
            os.replace(tmp_name, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
        _chmod_private(path)
