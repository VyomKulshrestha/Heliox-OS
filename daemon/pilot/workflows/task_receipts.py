"""Signed task receipts: a verifiable record of what Heliox did for one task.

When a task that ran at least one action finishes, Heliox writes a receipt: the
request, each action with its target, whether the person approved it, whether it
ran and succeeded, and the task's outcome. Action parameters and outputs are
recorded only as SHA-256 hashes, so a receipt can be shared without exposing file
contents or messages, yet still checked against values someone already has.

Receipts are signed with an Ed25519 key that belongs to this computer and is kept
in the operating-system keyring (or, where no keyring exists, in Heliox's private
configuration folder). Each receipt also carries the hash of the previous one, so
a removed or edited receipt breaks the chain. Anyone can check a receipt with::

    python -m pilot.workflows.task_receipts verify receipt.json

and compare the printed key fingerprint with the one shown in Settings → Task receipts.
Receipts describe what happened; they never grant, approve or run anything.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

logger = logging.getLogger("pilot.workflows.task_receipts")

RECEIPT_FORMAT = "heliox-task-receipt/1"
SIGNING_KEY_NAME = "heliox_task_receipt_signing_key"
KEY_FILE_NAME = "task_receipt_signing_key"
MAX_PENDING_TASKS = 50
MAX_ACTIONS_PER_RECEIPT = 500
CLOCK_SLACK_SECONDS = 1.0
ASKED_QUESTION = re.compile(
    r"^(?:hey heliox |heliox )?when did i (?:last )?(?:ask|tell) you (?P<prep>to|about) (?P<what>.+)$"
)
RECAP_QUESTION = re.compile(
    r"^(?:hey heliox |heliox )?what (?:did you do|have you done|did heliox do)(?: for me)?(?P<when>.*)$"
)


def canonical_json(value: Any) -> bytes:
    """The exact bytes that are hashed and signed: sorted keys, no whitespace, UTF-8."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def key_fingerprint(public_key_raw: bytes) -> str:
    """Short, human-comparable fingerprint of a receipt signing key."""
    digest = _sha256(public_key_raw)[:32]
    return ":".join(digest[i : i + 4] for i in range(0, 32, 4))


def verify_envelope(envelope: Any, *, expected_fingerprint: str | None = None) -> dict[str, Any]:
    """Check a receipt envelope's signature. Never raises on bad input."""
    reasons: list[str] = []
    if not isinstance(envelope, dict) or not isinstance(envelope.get("receipt"), dict):
        return {"valid": False, "signed": False, "fingerprint": None, "reasons": ["This is not a Heliox receipt."]}
    receipt = envelope["receipt"]
    if receipt.get("format") != RECEIPT_FORMAT:
        reasons.append(f"Unknown receipt format: {receipt.get('format')!r}.")
    signature = envelope.get("signature")
    public_key_b64 = envelope.get("public_key")
    if not signature or not public_key_b64:
        reasons.append(envelope.get("unsigned_reason") or "The receipt is not signed.")
        return {"valid": False, "signed": False, "fingerprint": None, "reasons": reasons}
    try:
        public_raw = base64.b64decode(public_key_b64, validate=True)
        Ed25519PublicKey.from_public_bytes(public_raw).verify(
            base64.b64decode(signature, validate=True), canonical_json(receipt)
        )
    except InvalidSignature:
        return {
            "valid": False,
            "signed": True,
            "fingerprint": None,
            "reasons": ["The signature does not match: the receipt was changed after it was signed."],
        }
    except (ValueError, TypeError):
        return {"valid": False, "signed": True, "fingerprint": None, "reasons": ["The signature or key is malformed."]}
    fingerprint = key_fingerprint(public_raw)
    if receipt.get("key_fingerprint") != fingerprint:
        reasons.append("The receipt names a different signing key than the one that signed it.")
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        reasons.append("It was signed by a different computer's key.")
    return {"valid": not reasons, "signed": True, "fingerprint": fingerprint, "reasons": reasons}


class TaskReceipts:
    """Collects each task's actions and writes a signed, hash-chained receipt when it ends."""

    def __init__(self, root: Path, *, config: Any = None, vault: Any = None, key_dir: Path | None = None) -> None:
        self._root = Path(root)
        # A ReceiptsConfig, or a function returning the current one, read on every call.
        self._config = config
        self._vault = vault
        self._key_dir = key_dir
        self._lock = asyncio.Lock()
        self._pending: dict[str, dict[str, Any]] = {}
        self._private_key: Ed25519PrivateKey | None = None
        self._key_problem = ""

    def _setting(self, name: str, default: Any) -> Any:
        config = self._config() if callable(self._config) else self._config
        return getattr(config, name, default) if config is not None else default

    @property
    def enabled(self) -> bool:
        return bool(self._setting("enabled", True))

    def _retention_seconds(self) -> float:
        return float(self._setting("retention_days", 90)) * 86400

    # ------------------------------------------------------------------ signing key

    async def _signing_key(self) -> Ed25519PrivateKey | None:
        if self._private_key is not None:
            return self._private_key
        encoded = None
        vault_ok = self._vault is not None and getattr(self._vault, "available", False)
        if vault_ok:
            try:
                encoded = await self._vault.get_key(SIGNING_KEY_NAME)
            except Exception:
                logger.warning("Receipt key could not be read from the keyring", exc_info=True)
                vault_ok = False
        key_file = self._key_dir / KEY_FILE_NAME if self._key_dir is not None else None
        if encoded is None and not vault_ok and key_file is not None and key_file.is_file():
            encoded = key_file.read_text(encoding="ascii").strip()
        if encoded:
            try:
                self._private_key = Ed25519PrivateKey.from_private_bytes(base64.b64decode(encoded))
                return self._private_key
            except ValueError:
                self._key_problem = "The stored receipt signing key is unreadable."
                logger.error(self._key_problem)
                return None
        key = Ed25519PrivateKey.generate()
        raw = key.private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()
        )
        try:
            if vault_ok:
                await self._vault.store_key(SIGNING_KEY_NAME, _b64(raw))
            elif key_file is not None:
                key_file.parent.mkdir(parents=True, exist_ok=True)
                key_file.write_text(_b64(raw), encoding="ascii")
                try:
                    os.chmod(key_file, 0o600)
                except OSError:
                    pass
            else:
                self._key_problem = "No keyring or private folder is available for the signing key."
                return None
        except Exception as exc:
            self._key_problem = f"The receipt signing key could not be saved: {exc}"
            logger.warning(self._key_problem)
            return None
        self._private_key = key
        return key

    async def key_info(self) -> dict[str, Any]:
        key = await self._signing_key()
        if key is None:
            return {"available": False, "fingerprint": None, "reason": self._key_problem}
        raw = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        return {"available": True, "fingerprint": key_fingerprint(raw), "public_key": _b64(raw)}

    # ------------------------------------------------------------------ collecting

    def record_action(
        self,
        task_key: str,
        plan_id: str,
        action: Any,
        result: Any,
        *,
        user_confirmed: bool = False,
        undo_kind: str = "",
    ) -> None:
        """Add one executed (or refused) action to the task's pending receipt."""
        if not task_key or not self.enabled:
            return
        pending = self._pending.get(task_key)
        if pending is None:
            if len(self._pending) >= MAX_PENDING_TASKS:
                oldest = min(self._pending, key=lambda key: self._pending[key]["started_at"])
                self._pending.pop(oldest, None)
            pending = {"started_at": time.time(), "plan_ids": [], "actions": []}
            self._pending[task_key] = pending
        if plan_id and plan_id not in pending["plan_ids"]:
            pending["plan_ids"].append(plan_id)
        if len(pending["actions"]) >= MAX_ACTIONS_PER_RECEIPT:
            pending["truncated"] = True
            return
        parameters = getattr(action, "parameters", None)
        try:
            params_dump = parameters.model_dump(mode="json") if parameters is not None else {}
        except Exception:
            params_dump = {"unserializable": str(parameters)}
        output = getattr(result, "output", "") or ""
        pending["actions"].append(
            {
                "index": len(pending["actions"]),
                "plan_id": plan_id,
                "action_type": getattr(
                    getattr(action, "action_type", ""), "value", str(getattr(action, "action_type", ""))
                ),
                "target": str(getattr(action, "target", "") or "")[:300],
                "parameters_sha256": _sha256(canonical_json(params_dump)),
                "user_confirmed": bool(user_confirmed),
                "executed": bool(getattr(result, "executed", True)),
                "success": bool(getattr(result, "success", False)),
                "error": str(getattr(result, "error", "") or "")[:300],
                "output_sha256": _sha256(str(output).encode("utf-8")) if output else None,
                "undo": undo_kind,
                "at": _iso(time.time()),
            }
        )

    def discard(self, task_key: str) -> None:
        self._pending.pop(task_key, None)

    # ------------------------------------------------------------------ writing

    def _files(self) -> list[Path]:
        if not self._root.is_dir():
            return []
        return sorted(self._root.glob("*.json"))

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    async def finalize(
        self,
        task_key: str,
        *,
        request: str,
        status: str,
        summary: str,
        source: str,
        undo: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Write the signed receipt for ``task_key``. Returns its short form, or None if nothing ran."""
        pending = self._pending.pop(task_key, None) if task_key else None
        if not pending or not pending["actions"] or not self.enabled:
            return None
        async with self._lock:
            key = await self._signing_key()
            return await asyncio.to_thread(
                self._write_receipt, key, task_key, pending, request, status, summary, source, undo
            )

    def _write_receipt(
        self,
        key: Ed25519PrivateKey | None,
        task_key: str,
        pending: dict[str, Any],
        request: str,
        status: str,
        summary: str,
        source: str,
        undo: dict[str, Any] | None,
    ) -> dict[str, Any]:
        from pilot import __version__

        files = self._files()
        previous = self._read(files[-1]) if files else None
        previous_hash = _sha256(canonical_json(previous)) if previous is not None else None
        finished = time.time()
        public_raw = (
            key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw) if key else None
        )
        receipt = {
            "format": RECEIPT_FORMAT,
            "receipt_id": uuid.uuid4().hex,
            "task_key": task_key,
            "plan_ids": pending["plan_ids"],
            "source": source,
            "request": request[:2000],
            "status": status,
            "summary": summary[:2000],
            "started_at": _iso(pending["started_at"]),
            "finished_at": _iso(finished),
            "actions": pending["actions"],
            "actions_truncated": bool(pending.get("truncated")),
            "undo": undo,
            "heliox_version": __version__,
            "key_fingerprint": key_fingerprint(public_raw) if public_raw else None,
            "previous_receipt_sha256": previous_hash,
        }
        envelope: dict[str, Any] = {"receipt": receipt, "algorithm": "ed25519"}
        if key is not None and public_raw is not None:
            envelope["signature"] = _b64(key.sign(canonical_json(receipt)))
            envelope["public_key"] = _b64(public_raw)
        else:
            envelope["signature"] = None
            envelope["public_key"] = None
            envelope["unsigned_reason"] = self._key_problem or "No signing key was available."
        self._root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.fromtimestamp(finished, UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = self._root / f"{stamp}_{receipt['receipt_id']}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(envelope, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return self._short(envelope)

    @staticmethod
    def _short(envelope: dict[str, Any]) -> dict[str, Any]:
        receipt = envelope["receipt"]
        return {
            "receipt_id": receipt["receipt_id"],
            "task_key": receipt["task_key"],
            "finished_at": receipt["finished_at"],
            "status": receipt["status"],
            "summary": receipt["summary"][:200],
            "request": receipt["request"][:200],
            "source": receipt["source"],
            "action_count": len(receipt["actions"]),
            "signed": bool(envelope.get("signature")),
        }

    # ------------------------------------------------------------------ reading and checking

    def _path_for(self, receipt_id: str) -> Path | None:
        if not receipt_id or not all(ch in "0123456789abcdef" for ch in receipt_id):
            return None
        matches = list(self._root.glob(f"*_{receipt_id}.json")) if self._root.is_dir() else []
        return matches[0] if matches else None

    async def list(self, limit: int = 50) -> list[dict[str, Any]]:
        def run() -> list[dict[str, Any]]:
            rows = []
            for path in reversed(self._files()):
                envelope = self._read(path)
                if envelope and isinstance(envelope.get("receipt"), dict):
                    rows.append(self._short(envelope))
                if len(rows) >= max(1, min(int(limit), 500)):
                    break
            return rows

        return await asyncio.to_thread(run)

    async def get(self, receipt_id: str) -> dict[str, Any] | None:
        path = self._path_for(receipt_id)
        return await asyncio.to_thread(self._read, path) if path else None

    async def verify(self, envelope: Any) -> dict[str, Any]:
        info = await self.key_info()
        result = verify_envelope(envelope)
        result["this_computer"] = bool(info.get("available") and result.get("fingerprint") == info.get("fingerprint"))
        return result

    async def verify_chain(self) -> dict[str, Any]:
        """Check every stored receipt's signature and its link to the one before it."""
        info = await self.key_info()

        def run() -> dict[str, Any]:
            problems = []
            previous_hash = None
            files = self._files()
            for position, path in enumerate(files):
                envelope = self._read(path)
                if envelope is None:
                    problems.append({"file": path.name, "reason": "unreadable"})
                    previous_hash = None
                    continue
                check = verify_envelope(envelope, expected_fingerprint=info.get("fingerprint"))
                if not check["valid"]:
                    problems.append({"file": path.name, "reason": "; ".join(check["reasons"])})
                link = envelope["receipt"].get("previous_receipt_sha256")
                # The oldest kept receipt may point at one removed by retention.
                if position > 0 and link != previous_hash:
                    problems.append(
                        {"file": path.name, "reason": "The receipt before this one was changed or removed."}
                    )
                previous_hash = _sha256(canonical_json(envelope))
            return {"valid": not problems, "count": len(files), "problems": problems}

        return await asyncio.to_thread(run)

    def recap(self, start: float, end: float) -> list[dict[str, Any]]:
        """Short forms of the receipts that finished between ``start`` and ``end``, oldest first."""
        rows = []
        for path in self._files():
            envelope = self._read(path)
            receipt = envelope.get("receipt") if envelope else None
            if not isinstance(receipt, dict):
                continue
            try:
                finished = datetime.fromisoformat(str(receipt["finished_at"])).timestamp()
            except (KeyError, ValueError):
                continue
            if start <= finished <= end:
                rows.append(self._short(envelope))
        return rows

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Short forms of receipts whose request or summary contains every word of ``query``, newest first."""
        words = [word for word in re.sub(r"[^\w\s]", " ", query.casefold()).split() if len(word) > 1]
        if not words:
            return []
        found = []
        for path in reversed(self._files()):
            envelope = self._read(path)
            receipt = envelope.get("receipt") if envelope else None
            if not isinstance(receipt, dict):
                continue
            text = f"{receipt.get('request', '')} {receipt.get('summary', '')}".casefold()
            if all(word in text for word in words):
                found.append(self._short(envelope))
                if len(found) >= max(1, min(int(limit), 200)):
                    break
        return found

    def answer(self, question: str, *, now: datetime | None = None) -> str | None:
        """Answer "what did you do today?" or "when did I ask you to …?" from the receipts; None otherwise."""
        normalized = re.sub(r"\s+", " ", question.lower().strip(" ?.!"))
        when_asked = ASKED_QUESTION.match(normalized)
        if when_asked:
            what = when_asked.group("what")
            hits = self.search(what, limit=1)
            if not hits:
                return f"I have no task where you asked me {when_asked.group('prep')} {what}."
            hit = hits[0]
            moment = datetime.fromisoformat(hit["finished_at"]).astimezone()
            actions = f"{hit['action_count']} action{'' if hit['action_count'] == 1 else 's'}"
            return f'On {moment.strftime("%a %d %b at %I:%M %p")} you asked: "{hit["request"]}" ({actions}, {hit["status"]}).'
        asked = RECAP_QUESTION.match(normalized)
        if not asked:
            return None
        from pilot.workflows.activity_timeline import parse_when

        now = now or datetime.now()
        when = asked.group("when").strip() or "today"
        start, end, label = parse_when(when, now)
        if label == "in the last two hours" and "today" not in when:
            start, end, label = parse_when("today", now)
        if not self.enabled:
            return "Task receipts are off, so I have no record to go on."
        # datetime.now() can trail time.time() by a clock tick on Windows; keep a just-finished task in.
        rows = self.recap(start, end + CLOCK_SLACK_SECONDS)
        if not rows:
            return f"I didn't run any actions {label}."
        failed = sum(1 for row in rows if row["status"] not in ("success", "completed"))
        listed = [
            f'"{row["request"] or row["summary"]}" ({row["action_count"]} action{"" if row["action_count"] == 1 else "s"})'
            for row in rows[-3:]
        ]
        opening = label[0].upper() + label[1:]
        tasks = f"{len(rows)} task{'' if len(rows) == 1 else 's'}"
        text = f"{opening} I ran {tasks}. The latest: " + "; ".join(listed) + "."
        if failed:
            text += f" {failed} did not finish cleanly."
        return text

    async def answer_async(self, question: str) -> str | None:
        return await asyncio.to_thread(self.answer, question)

    async def prune(self, *, now: float | None = None) -> int:
        horizon = (time.time() if now is None else now) - self._retention_seconds()

        def run() -> int:
            removed = 0
            for path in self._files()[:-1]:  # always keep the newest so the chain has a head
                if path.stat().st_mtime < horizon:
                    path.unlink(missing_ok=True)
                    removed += 1
            return removed

        async with self._lock:
            return await asyncio.to_thread(run)


def _main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "verify":
        print("usage: python -m pilot.workflows.task_receipts verify RECEIPT.json")
        return 2
    try:
        envelope = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"Cannot read {argv[1]}: {exc}")
        return 2
    result = verify_envelope(envelope)
    if result["valid"]:
        receipt = envelope["receipt"]
        print(f"Valid receipt {receipt['receipt_id']} ({receipt['status']}, {len(receipt['actions'])} actions)")
        print(f"Signed by key {result['fingerprint']}")
        return 0
    print("NOT valid: " + " ".join(result["reasons"]))
    return 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
