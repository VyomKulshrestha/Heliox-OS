"""The secrets of a kept plan, held apart from its stored text so the plan can be resumed.

A saved plan (``pilot.workflows.checkpoints``) is stored the way it is shown: every webhook address, URL with a
token, GitHub token, mail password and credential header value (Authorization, Cookie, X-Api-Key and
similar) in it is masked (``pilot.security.secret_urls``). Before this module, a plan
read back from storage held only the masked values and the HTTP layer refused them, so an interrupted plan, or one
approved later from a phone or through MCP, could not be continued.

When a plan with such a value is kept, the real values go into one entry in the operating system's credential store,
``plan_secret:<plan id>``, tied to that plan. Reading the plan back puts each value in place of its masked form,
matched by the masked text itself (the host and last four characters), so a plan whose steps were reordered or cut
down after an approval still gets the right value for each step. A masked value with no match stays masked and is
refused before anything is sent.

How long an entry lives: until its plan completes, is cancelled, superseded or turned down, and never longer than
``KEEP_DAYS`` days from when it was first kept; an expired entry is not restored and is removed when it is next read
and when the daemon starts (``sweep``). The entries' plan ids and times are listed in ``plan_secret:index`` so the
sweep can find them; the index holds no secret.

With no credential store (Linux without a Secret Service, or a store that refuses the write), the values are held in
the daemon's memory only: a plan approved later in the same run still resumes, but after a restart its masked values
cannot be restored. When the store refused the entry (on Windows, one longer than about 1,280 characters), the
index records why, and the refused step's failure says the secret was too large to keep and the request has to
be given again. A step whose secret is a stored connection (Settings > Connections) still runs, because the
sender matches the masked value against the connection; any other such step fails with "give the address again".

Plans sent to a mesh peer carry no secrets at all: only system read-outs may be delegated
(``pilot.network.collab_executor``), so no peer ever needs one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pilot.actions import ActionPlan, ApiRequestParams, EmailParams
from pilot.security.secret_urls import (
    carries_secret,
    is_masked,
    is_secret_header,
    mask_header_value,
    mask_secret,
    mask_url,
    note_unrestored,
)

logger = logging.getLogger("pilot.security.plan_secrets")

PREFIX = "plan_secret:"
INDEX_NAME = "plan_secret:index"
KEEP_DAYS = 7
KEEP_SECONDS = KEEP_DAYS * 24 * 3600


# A credential store entry on Windows holds about 1,280 characters; a longer one is refused.
WINDOWS_ENTRY_CHARS = 1280


@dataclass(frozen=True)
class _Slot:
    """One secret value in a step's parameters: a field, or one header of ``headers``."""

    field: str
    value: str
    mask: Callable[[str], str | None]
    header: str = ""

    def key(self, masked: str) -> str:
        # Header values are matched by header name too: two short tokens both mask to "Bearer ***".
        return f"header:{self.header.casefold()}:{masked}" if self.header else masked


def _slots(params: Any) -> list[_Slot]:
    """Each secret value the parameters hold, masked or not."""
    found: list[_Slot] = []
    if isinstance(params, ApiRequestParams):
        if params.url and (carries_secret(params.url) or is_masked(params.url)):
            found.append(_Slot("url", params.url, mask_url))
        if params.webhook_url:
            found.append(_Slot("webhook_url", params.webhook_url, mask_url))
        if params.token:
            found.append(_Slot("token", params.token, mask_secret))
        for name, value in (params.headers or {}).items():
            if is_secret_header(name) and isinstance(value, str) and value:
                found.append(_Slot("headers", value, mask_header_value, header=str(name)))
    elif isinstance(params, EmailParams) and params.app_password:
        found.append(_Slot("app_password", params.app_password, mask_secret))
    return found


def _masked_slots(plan: ActionPlan) -> list[_Slot]:
    return [slot for action in plan.actions for slot in _slots(action.parameters) if is_masked(slot.value)]


def secrets_of(plan: ActionPlan) -> dict[str, str]:
    """Matching key (the masked form) -> real value, for every secret value the plan holds in memory.

    Two different values with the same key are left out: neither could be put back with certainty, so both steps
    are refused on resume rather than one getting the other's secret.
    """
    values: dict[str, str] = {}
    ambiguous: set[str] = set()
    for action in plan.actions:
        for slot in _slots(action.parameters):
            if is_masked(slot.value):
                continue
            masked = slot.mask(slot.value)
            if not masked or masked == slot.value:
                continue
            key = slot.key(masked)
            if key in values and values[key] != slot.value:
                ambiguous.add(key)
            values[key] = slot.value
    return {key: value for key, value in values.items() if key not in ambiguous}


def restore(plan: ActionPlan, values: dict[str, str]) -> ActionPlan:
    """The plan with each masked secret replaced by its kept value; values with no match stay masked."""
    if not values:
        return plan
    changed = False
    actions = []
    for action in plan.actions:
        update: dict[str, Any] = {}
        headers = dict(getattr(action.parameters, "headers", None) or {})
        for slot in _slots(action.parameters):
            key = slot.key(slot.value)
            if not is_masked(slot.value) or key not in values:
                continue
            if slot.header:
                headers[slot.header] = values[key]
                update["headers"] = headers
            else:
                update[slot.field] = values[key]
        if update:
            changed = True
            action = action.model_copy(update={"parameters": action.parameters.model_copy(update=update)})
        actions.append(action)
    return plan.model_copy(update={"actions": actions}) if changed else plan


def _created(entry: Any) -> float | None:
    value = entry.get("created") if isinstance(entry, dict) else entry
    return float(value) if isinstance(value, int | float) else None


class PlanSecretKeeper:
    """Keeps, restores and forgets the secrets of kept plans (see the module text)."""

    def __init__(self, vault: Any | None, *, clock: Callable[[], float] = time.time) -> None:
        self._vault = vault
        self._clock = clock
        self._memory: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._index: dict[str, Any] | None = None  # read once, then kept in step with every write

    def _store_usable(self) -> bool:
        return self._vault is not None and bool(getattr(self._vault, "available", False))

    async def _read_json(self, name: str) -> dict[str, Any]:
        if not self._store_usable():
            return {}
        try:
            text = await self._vault.get_key(name)
        except Exception as exc:  # noqa: BLE001 - a store that cannot be read is treated as empty, with a log line
            logger.warning("Could not read a kept plan's secrets: %s", type(exc).__name__)
            return {}
        try:
            value = json.loads(text) if text else {}
        except ValueError:
            return {}
        return value if isinstance(value, dict) else {}

    async def _write_json(self, name: str, value: dict[str, Any]) -> bool:
        if not self._store_usable():
            return False
        try:
            await self._vault.store_key(name, json.dumps(value, separators=(",", ":")))
        except Exception as exc:  # noqa: BLE001 - the values stay in memory for this run; the log names the error type
            logger.warning("Could not keep a plan's secrets in the credential store: %s", type(exc).__name__)
            return False
        return True

    async def _delete(self, name: str) -> None:
        if not self._store_usable():
            return
        try:
            await self._vault.delete_key(name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not remove a kept plan's secrets: %s", type(exc).__name__)

    async def _load_index(self) -> dict[str, Any]:
        if self._index is None:
            self._index = await self._read_json(INDEX_NAME)
        return dict(self._index)

    async def _save_index(self, index: dict[str, Any]) -> None:
        if await self._write_json(INDEX_NAME, index):
            self._index = dict(index)

    async def keep(self, plan_id: str, plan: ActionPlan) -> None:
        """Hold the plan's secret values (merged with what is held already) before the plan is stored."""
        values = secrets_of(plan)
        if not values:
            return
        async with self._lock:
            held = self._memory.get(plan_id)
            if held is not None and all(held["values"].get(key) == value for key, value in values.items()):
                return
            created = held["created"] if held is not None else self._clock()
            merged = {**(held["values"] if held is not None else {}), **values}
            entry = {"created": created, "values": merged}
            self._memory[plan_id] = entry
            index = await self._load_index()
            if await self._write_json(PREFIX + plan_id, entry):
                if not isinstance(index.get(plan_id), int | float):
                    index[plan_id] = created
                    await self._save_index(index)
            elif self._store_usable():
                # Held in memory for this run only. The index (no secret) records why, so a resume after a restart
                # can tell the person instead of failing without a reason.
                size = len(json.dumps(entry, separators=(",", ":")))
                why = "too_large" if size > WINDOWS_ENTRY_CHARS else "refused"
                index[plan_id] = {"created": created, "not_kept": why}
                await self._save_index(index)

    async def _lookup(self, plan_id: str) -> tuple[dict[str, str], str]:
        """(kept values, why there are none: a key of secret_urls.UNRESTORED_REASONS, or "")."""
        async with self._lock:
            entry = self._memory.get(plan_id)
            if entry is None:
                entry = await self._read_json(PREFIX + plan_id)
                if entry:
                    self._memory[plan_id] = entry
            listed = (await self._load_index()).get(plan_id)
        if not entry:
            if not self._store_usable():
                return {}, "no_store"
            if isinstance(listed, dict) and listed.get("not_kept"):
                return {}, str(listed["not_kept"])
            return {}, ""
        created = _created(entry)
        if created is None or self._clock() - created > KEEP_SECONDS:
            await self.forget(plan_id)
            return {}, "expired"
        values = entry.get("values")
        return ({str(k): str(v) for k, v in values.items()} if isinstance(values, dict) else {}), ""

    async def values(self, plan_id: str) -> dict[str, str]:
        """The kept values of a plan, or {} when none are kept or they expired."""
        return (await self._lookup(plan_id))[0]

    async def restore(self, plan_id: str, plan: ActionPlan) -> ActionPlan:
        if not _masked_slots(plan):
            return plan
        values, reason = await self._lookup(plan_id)
        restored = restore(plan, values)
        for slot in _masked_slots(restored):
            # The step will be refused before it sends; the refusal says why (pilot.security.secret_urls).
            note_unrestored(slot.value, reason)
        return restored

    async def forget(self, plan_id: str) -> None:
        async with self._lock:
            had = self._memory.pop(plan_id, None) is not None
            index = await self._load_index()
            if plan_id in index or had:
                await self._delete(PREFIX + plan_id)
            if plan_id in index:
                index.pop(plan_id)
                await self._save_index(index)

    async def sweep(self) -> int:
        """Remove every entry older than KEEP_DAYS days. Called when the daemon starts; returns how many went."""
        index = await self._load_index()
        now = self._clock()
        expired = [
            plan_id
            for plan_id, entry in index.items()
            if (created := _created(entry)) is None or now - created > KEEP_SECONDS
        ]
        for plan_id in expired:
            await self.forget(plan_id)
        return len(expired)
