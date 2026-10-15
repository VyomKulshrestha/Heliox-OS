"""The co-adaptation preferences file (spec section 10.5).

Rules are settings, not behavioural data: the file exists independently of consent, holds participant keys but never
display labels, and is replaced atomically on every write. Device-wide rules survive consent revocation and expiry;
rules scoped to a participant are hard-deleted when that participant revokes or expires. A retracted rule stays in
the file with ``retracted_at`` and is dropped on the first write 30 days after its retraction (each participant's
encrypted store keeps its own rule history, so nothing downstream depends on this tail).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pilot.coadaptation.core.preferences import (
    Preference,
    PreferenceSet,
    preference_from_dict,
    preference_to_dict,
)

FILE_SCHEMA_VERSION = 1
RETRACTED_TAIL = timedelta(days=30)


class PreferenceFileError(ValueError):
    """The preferences file exists but cannot be read; rules are never silently dropped."""


class PreferenceStore:
    """Read and write ``co_adaptation_preferences.json``. Every rule has origin ``explicit_user``."""

    def __init__(self, path: Path, *, clock: Callable[[], datetime] | None = None) -> None:
        self._path = Path(path)
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def path(self) -> Path:
        return self._path

    def entries(self) -> tuple[Preference, ...]:
        """Every rule in the file, active and retracted, in file order."""
        if not self._path.exists():
            return ()
        try:
            document = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PreferenceFileError("the co-adaptation preferences file cannot be read") from exc
        if (
            not isinstance(document, dict)
            or document.get("schema_version") != FILE_SCHEMA_VERSION
            or not isinstance(document.get("preferences"), list)
        ):
            raise PreferenceFileError("the co-adaptation preferences file has an unknown format")
        try:
            prefs = tuple(preference_from_dict(item) for item in document["preferences"])
        except (TypeError, ValueError) as exc:
            raise PreferenceFileError("the co-adaptation preferences file holds an invalid rule") from exc
        if any(pref.origin != "explicit_user" for pref in prefs):
            raise PreferenceFileError("the co-adaptation preferences file holds a rule not set by the person")
        return prefs

    def load(self) -> PreferenceSet:
        return PreferenceSet(self.entries())

    def get(self, pref_id: str) -> Preference | None:
        for pref in self.entries():
            if pref.pref_id == pref_id:
                return pref
        return None

    def add(self, pref_dict: Mapping[str, Any]) -> tuple[Preference, tuple[str, ...]]:
        """Validate a new rule against the active ones and append it.

        Returns the stored rule and the notes of ``PreferenceSet.validate_new``; raises ``PreferenceConflict`` (and
        writes nothing) when the rule conflicts with an active one.
        """
        if pref_dict.get("origin", "explicit_user") != "explicit_user":
            raise ValueError("rules can only be set by the person (origin explicit_user)")
        pref = preference_from_dict({**pref_dict, "origin": "explicit_user"})
        current = self.entries()
        if any(existing.pref_id == pref.pref_id for existing in current):
            raise ValueError("a rule with this id already exists")
        notes = tuple(PreferenceSet(current).validate_new(pref))
        self._write([*current, pref])
        return pref, notes

    def retract(self, pref_id: str, *, at: datetime | None = None) -> bool:
        """Mark an active rule retracted at ``at`` (default: now). False when no active rule has this id.

        The service passes the instant it issues the reply's fallbacks at, so a reconstruction of the rules in force
        at that ``issued_at`` (``PreferenceSet.as_of``) already leaves the retracted rule out.
        """
        retracted_at = self._clock() if at is None else at
        if retracted_at.tzinfo is None:
            raise ValueError("retracted_at must carry a UTC offset")
        current = list(self.entries())
        for index, pref in enumerate(current):
            if pref.pref_id == pref_id and pref.retracted_at is None:
                # A clock that stepped back never yields retracted_at < created_at; such a rule is not in force at
                # ``at`` either way, because it was created after it.
                current[index] = dataclasses.replace(pref, retracted_at=max(retracted_at, pref.created_at))
                self._write(current)
                return True
        return False

    def delete_for_participant(self, participant_key: str) -> int:
        """Hard-delete every rule scoped to this participant (revoke and expiry). Device-wide rules stay."""
        current = list(self.entries())
        kept = [pref for pref in current if pref.scope.participant_key != participant_key]
        removed = len(current) - len(kept)
        if removed:
            self._write(kept)
        return removed

    def _write(self, prefs: list[Preference]) -> None:
        now = self._clock()
        kept = [pref for pref in prefs if pref.retracted_at is None or now - pref.retracted_at < RETRACTED_TAIL]
        document = {
            "schema_version": FILE_SCHEMA_VERSION,
            "preferences": [preference_to_dict(pref) for pref in kept],
        }
        data = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".co_adaptation_preferences.", suffix=".tmp", dir=self._path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            with contextlib.suppress(OSError):
                os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, self._path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
        with contextlib.suppress(OSError):
            os.chmod(self._path, 0o600)
