"""Per-task undo: deterministic compensations for the local changes Heliox makes.

Before a supported action runs, the journal records what is needed to reverse it: a
copy of a file that will be overwritten or deleted, where a moved or copied item will
land, the clipboard text that will be replaced. After the action succeeds it records
what Heliox left behind. Undoing a task reverses its entries newest first, and only
when each item is still exactly as Heliox left it. Anything the person changed since
is reported as a conflict and left alone, never overwritten.

The model never writes these compensations; they follow from the action type and its
parameters. Actions whose effects leave the computer (an email, a web request, a
remote command) or that Heliox cannot reverse reliably (shell commands, package
installs, process kills) are classified as not undoable, and the person sees that in
the approval dialog and again in the undo preview.

Undo only restores recorded state on recorded paths. It never plans, never calls the
executor, and refuses paths inside Heliox's own folders. Copies live under
``DATA_DIR/undo``, which actions may not read or change, and are pruned by age and size.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pilot.actions import Action, ActionResult, ActionType, PermissionTier

logger = logging.getLogger("pilot.workflows.undo_journal")

JOURNAL_VERSION = 1
JOURNAL_FILE = "journal.json"
BLOB_DIR = "blobs"
EMPTY_CLIPBOARD = "(clipboard is empty)"  # what clipboard_read returns for an empty clipboard

UNDOABLE_TYPES = frozenset(
    {
        ActionType.FILE_WRITE,
        ActionType.FILE_DELETE,
        ActionType.FILE_MOVE,
        ActionType.FILE_COPY,
        ActionType.CLIPBOARD_WRITE,
    }
)

# Effects that leave the computer: once sent, nothing local can take them back.
EXTERNAL_TYPES = frozenset(
    {
        ActionType.API_SEND_EMAIL,
        ActionType.API_WEBHOOK,
        ActionType.API_SLACK,
        ActionType.API_DISCORD,
        ActionType.API_REQUEST,
        ActionType.API_GITHUB,
        ActionType.EMAIL_REPLY,
        ActionType.SSH_COMMAND,
        ActionType.SSH_SCRIPT,
        ActionType.GIT_PUSH,
        ActionType.CALENDAR_CREATE_EVENT,
        ActionType.CALENDAR_DELETE_EVENT,
        ActionType.CALENDAR_SYNC,
        ActionType.MCP_CALL_TOOL,
        ActionType.BROWSER_CLICK,
        ActionType.BROWSER_CLICK_TEXT,
        ActionType.BROWSER_TYPE,
        ActionType.BROWSER_SELECT,
        ActionType.BROWSER_FILL_FORM,
        ActionType.BROWSER_EXECUTE_JS,
    }
)

REASON_UNDOABLE = "Heliox keeps what it needs to put this back."
REASON_NO_CHANGE = "This only reads; there is nothing to undo."
REASON_EXTERNAL = (
    "This reaches outside your computer (a message, a web page or a remote command); Heliox cannot take it back."
)
REASON_UNSUPPORTED = "Heliox cannot reverse this kind of change automatically."


@dataclass(frozen=True)
class UndoSupport:
    """Whether Heliox can undo an action, decided from its type alone (before it runs)."""

    kind: str  # "undoable" | "no_change" | "not_undoable"
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "reason": self.reason}


def undo_support(action: Action) -> UndoSupport:
    """Classify an action for the approval dialog and the undo preview."""
    if action.action_type in UNDOABLE_TYPES:
        return UndoSupport("undoable", REASON_UNDOABLE)
    if action.action_type in EXTERNAL_TYPES:
        return UndoSupport("not_undoable", REASON_EXTERNAL)
    if action.permission_tier == PermissionTier.READ_ONLY:
        return UndoSupport("no_change", REASON_NO_CHANGE)
    return UndoSupport("not_undoable", REASON_UNSUPPORTED)


@dataclass
class UndoToken:
    """What ``before`` captured for one action, handed back to ``after``."""

    task_key: str
    action: Action
    support: UndoSupport
    entry: dict[str, Any] | None = None  # None when nothing was captured
    clipboard_previous: str | None = None


def _abs(path: str) -> Path:
    # filesystem.py uses Path(path) as given, so resolve it the same way (no ~ expansion).
    return Path(os.path.abspath(path))


def _tree_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:
                pass
    return total


def _signature(path: Path) -> str | None:
    """A content hash for a file, or a structure hash (names, sizes, mtimes) for a folder."""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return "file:" + digest.hexdigest()
    for dirpath, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            item = Path(dirpath) / name
            try:
                stat = item.stat()
            except OSError:
                continue
            rel = item.relative_to(path).as_posix()
            digest.update(f"{rel}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return "dir:" + digest.hexdigest()


def _private(path: str | Path) -> bool:
    from pilot.security.validator import inside_heliox_private_dirs

    return inside_heliox_private_dirs(str(path))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class UndoJournal:
    """Stores per-task compensations and applies them on request."""

    def __init__(self, root: Path, *, config: Any = None) -> None:
        self._root = Path(root)
        self._config = config  # an UndoConfig, read on every call so Settings changes apply at once
        self._lock = asyncio.Lock()
        # Clipboard text may be private, so it is kept in memory only and never written to disk.
        self._clipboard_previous: dict[tuple[str, int], str] = {}

    # ------------------------------------------------------------------ settings

    def _setting(self, name: str, default: Any) -> Any:
        return getattr(self._config, name, default) if self._config is not None else default

    @property
    def enabled(self) -> bool:
        return bool(self._setting("enabled", True))

    def _max_task_bytes(self) -> int:
        return int(self._setting("max_task_mb", 512)) * 1024 * 1024

    def _max_total_bytes(self) -> int:
        return int(self._setting("max_total_mb", 2048)) * 1024 * 1024

    def _retention_seconds(self) -> float:
        return float(self._setting("retention_days", 7)) * 86400

    # ------------------------------------------------------------------ storage

    def _task_dir(self, task_key: str) -> Path:
        safe = "".join(ch for ch in task_key if ch.isalnum() or ch in "-_")[:80] or "task"
        return self._root / safe

    def _blob(self, task_key: str, seq: int) -> Path:
        return self._task_dir(task_key) / BLOB_DIR / str(seq)

    def _read_journal(self, task_key: str) -> dict[str, Any] | None:
        path = self._task_dir(task_key) / JOURNAL_FILE
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Undo journal for %s is unreadable", task_key)
            return None
        if not isinstance(data, dict) or data.get("version") != JOURNAL_VERSION:
            return None
        return data

    def _write_journal(self, task_key: str, journal: dict[str, Any]) -> None:
        path = self._task_dir(task_key) / JOURNAL_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        journal["updated_at"] = time.time()
        tmp = path.with_name(f"{JOURNAL_FILE}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(journal, indent=1), encoding="utf-8")
        os.replace(tmp, path)

    def _load(self, task_key: str) -> dict[str, Any]:
        journal = self._read_journal(task_key)
        if journal is not None:
            return journal
        now = time.time()
        return {
            "version": JOURNAL_VERSION,
            "task_key": task_key,
            "created_at": now,
            "updated_at": now,
            "undone_at": None,
            "next_seq": 1,
            "bytes": 0,
            "plan_ids": [],
            "entries": [],
            "not_undoable": [],
        }

    @staticmethod
    def _take_seq(journal: dict[str, Any]) -> int:
        seq = int(journal.get("next_seq", 1))
        journal["next_seq"] = seq + 1
        return seq

    def _drop_blob(self, task_key: str, seq: int) -> None:
        blob = self._blob(task_key, seq)
        if blob.is_dir():
            shutil.rmtree(blob, ignore_errors=True)
        elif blob.exists():
            blob.unlink(missing_ok=True)

    # ------------------------------------------------------------------ capture

    async def before(self, action: Action, *, task_key: str) -> UndoToken | None:
        """Capture what is needed to reverse ``action``. Never raises and never blocks the action."""
        if not task_key or not self.enabled:
            return None
        token = UndoToken(task_key=task_key, action=action, support=undo_support(action))
        if token.support.kind != "undoable":
            return token
        try:
            if action.action_type == ActionType.CLIPBOARD_WRITE:
                from pilot.system.clipboard import clipboard_read

                try:
                    token.clipboard_previous = await clipboard_read()
                except Exception:
                    token.clipboard_previous = None
                token.entry = {"type": action.action_type.value}
                return token
            async with self._lock:
                token.entry = await asyncio.to_thread(self._capture_file, action, task_key)
        except Exception as exc:  # a failed capture must never stop the action itself
            logger.warning("Undo capture failed for %s: %s", action.action_type.value, exc)
            token.entry = None
            token.support = UndoSupport("not_undoable", f"Heliox could not keep a copy: {exc}")
        return token

    def _capture_file(self, action: Action, task_key: str) -> dict[str, Any]:
        params = action.parameters
        path = _abs(getattr(params, "path", "") or "")
        destination_value = getattr(params, "destination", "") or ""
        journal = self._load(task_key)
        seq = self._take_seq(journal)
        entry: dict[str, Any] = {"seq": seq, "type": action.action_type.value, "copy_bytes": 0}

        def keep_copy(source: Path) -> None:
            if _private(source):
                raise ValueError("it is inside Heliox's own folders")
            size = _tree_bytes(source)
            if journal["bytes"] + size > self._max_task_bytes():
                raise ValueError("it is larger than the undo limit for one task")
            blob = self._blob(task_key, seq)
            blob.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, blob)
            else:
                shutil.copy2(source, blob)
            journal["bytes"] += size
            entry["copy_bytes"] = size

        if action.action_type == ActionType.FILE_WRITE:
            entry["path"] = str(path)
            entry["pre"] = _signature(path)
            if path.is_dir():
                raise ValueError("the target is a folder")
            if path.is_file():
                keep_copy(path)
        elif action.action_type == ActionType.FILE_DELETE:
            entry["path"] = str(path)
            entry["pre"] = _signature(path)
            if path.exists():
                keep_copy(path)
        else:
            destination = _abs(destination_value)
            # Where shutil.move / copy2 put the item: inside an existing folder, or at the path itself.
            into_folder = destination.is_dir() and (action.action_type == ActionType.FILE_MOVE or path.is_file())
            final = destination / path.name if into_folder else destination
            entry["source"] = str(path)
            entry["final"] = str(final)
            entry["pre_final"] = _signature(final)
            if final.is_dir():
                raise ValueError("the destination folder already exists")
            if final.is_file():
                keep_copy(final)  # the file this move or copy will replace
        self._write_journal(task_key, journal)
        return entry

    async def after(self, token: UndoToken | None, result: ActionResult, *, plan_id: str = "") -> None:
        """Record the outcome of the action that ``before`` captured."""
        if token is None:
            return
        try:
            post_clipboard: str | None = None
            succeeded = bool(result.success) and bool(getattr(result, "executed", True))
            if succeeded and token.action.action_type == ActionType.CLIPBOARD_WRITE and token.entry is not None:
                from pilot.system.clipboard import clipboard_read

                try:
                    post_clipboard = await clipboard_read()
                except Exception:
                    post_clipboard = None
            async with self._lock:
                await asyncio.to_thread(self._record_after, token, succeeded, plan_id, post_clipboard)
        except Exception:
            logger.warning("Undo journal update failed", exc_info=True)

    def _record_after(self, token: UndoToken, succeeded: bool, plan_id: str, post_clipboard: str | None) -> None:
        task_key, action, entry = token.task_key, token.action, token.entry
        if not succeeded:
            if entry and "seq" in entry:
                self._drop_blob(task_key, entry["seq"])
                journal = self._read_journal(task_key)
                if journal is not None and entry.get("copy_bytes"):
                    journal["bytes"] = max(0, journal["bytes"] - entry["copy_bytes"])
                    self._write_journal(task_key, journal)
            return
        if token.support.kind == "no_change":
            return
        journal = self._load(task_key)
        if plan_id and plan_id not in journal["plan_ids"]:
            journal["plan_ids"].append(plan_id)
        target = (action.target or getattr(action.parameters, "path", "") or "")[:300]
        if entry is None:
            journal["not_undoable"].append(
                {"type": action.action_type.value, "target": target, "reason": token.support.reason}
            )
            self._write_journal(task_key, journal)
            return
        entry = dict(entry, target=target, status="recorded")
        if action.action_type == ActionType.CLIPBOARD_WRITE:
            entry["seq"] = self._take_seq(journal)
            if token.clipboard_previous is None or post_clipboard is None:
                entry["status"] = "unavailable"
                entry["reason"] = "Heliox could not read the clipboard around this change."
            else:
                entry["post"] = _hash_text(post_clipboard)
                self._clipboard_previous[(task_key, entry["seq"])] = token.clipboard_previous
        elif action.action_type in (ActionType.FILE_WRITE, ActionType.FILE_DELETE):
            entry["post"] = _signature(Path(entry["path"]))
        else:
            entry["post"] = _signature(Path(entry["final"]))
        journal["entries"].append(entry)
        self._write_journal(task_key, journal)

    # ------------------------------------------------------------------ preview

    def _check(
        self, task_key: str, entry: dict[str, Any], virtual: dict[str, str | None]
    ) -> tuple[str, str, dict[str, str | None]]:
        """Decide one entry against the disk as it will be once the newer entries are undone.

        Returns (state, description, effects): state is "ready", "conflict", "unavailable"
        or "done", and effects maps each path to the signature it will have after this
        entry is undone.
        """

        def current(path: str) -> str | None:
            return virtual[path] if path in virtual else _signature(Path(path))

        status = entry.get("status")
        if status == "undone":
            return "done", "Already undone.", {}
        if status == "unavailable":
            return "unavailable", entry.get("reason", "Heliox did not keep what it needs."), {}
        kind = entry["type"]
        if kind == ActionType.CLIPBOARD_WRITE.value:
            if (task_key, entry["seq"]) not in self._clipboard_previous:
                return "unavailable", "The earlier clipboard text is only kept until Heliox restarts.", {}
            return "ready", "Put back the clipboard text you had before.", {}
        paths = [p for p in (entry.get("path"), entry.get("source"), entry.get("final")) if p]
        if any(_private(p) for p in paths):
            return "unavailable", "This is inside Heliox's own folders.", {}
        blob = self._blob(task_key, entry["seq"])
        if kind == ActionType.FILE_WRITE.value:
            path = entry["path"]
            name = Path(path).name
            if current(path) != entry.get("post"):
                return "conflict", f"{name} changed after Heliox wrote it, so it is left as it is.", {}
            if entry.get("pre") is None:
                return "ready", f"Remove {name}, which Heliox created.", {path: None}
            if not blob.exists():
                return "unavailable", f"The saved copy of {name} is missing.", {}
            return "ready", f"Put back the earlier version of {name}.", {path: entry["pre"]}
        if kind == ActionType.FILE_DELETE.value:
            path = entry["path"]
            name = Path(path).name
            if entry.get("pre") is None:
                return "done", "Nothing was there to delete.", {}
            if current(path) is not None:
                return "conflict", f"Something named {name} is there again, so it is left as it is.", {}
            if not blob.exists():
                return "unavailable", f"The saved copy of {name} is missing.", {}
            return "ready", f"Restore {name}.", {path: entry["pre"]}
        source, final = entry["source"], entry["final"]
        final_name = Path(final).name
        if current(final) != entry.get("post"):
            return "conflict", f"{final_name} changed or moved after Heliox put it there, so it is left as it is.", {}
        if entry.get("pre_final") is not None and not blob.exists():
            return "unavailable", f"The saved copy of the file {final_name} replaced is missing.", {}
        effects: dict[str, str | None] = {final: entry.get("pre_final")}
        if kind == ActionType.FILE_MOVE.value:
            if current(source) is not None:
                return "conflict", f"Something named {Path(source).name} is back in the original place.", {}
            effects[source] = entry.get("post")
            description = f"Move {final_name} back to {Path(source).parent}."
        else:
            description = f"Remove the copy {final_name}."
        if entry.get("pre_final") is not None:
            description += " The file it replaced comes back."
        return "ready", description, effects

    def _plan(self, task_key: str, journal: dict[str, Any]) -> dict[str, Any]:
        virtual: dict[str, str | None] = {}
        items = []
        for entry in sorted(journal["entries"], key=lambda e: e["seq"], reverse=True):
            state, description, effects = self._check(task_key, entry, virtual)
            if state == "done":
                continue
            if state == "ready":
                virtual.update(effects)
            items.append(
                {
                    "seq": entry["seq"],
                    "type": entry["type"],
                    "target": entry.get("target", ""),
                    "state": state,
                    "description": description,
                }
            )
        fingerprint = json.dumps([[i["seq"], i["state"]] for i in items])
        return {
            "task_key": task_key,
            "plan_ids": list(journal.get("plan_ids", [])),
            "created_at": journal.get("created_at"),
            "undone": journal.get("undone_at") is not None,
            "items": items,
            "not_undoable": list(journal.get("not_undoable", [])),
            "ready": sum(1 for i in items if i["state"] == "ready"),
            "preview_token": hashlib.sha256(f"{task_key}\0{fingerprint}".encode()).hexdigest()[:32],
        }

    async def preview(self, task_key: str) -> dict[str, Any] | None:
        """What undoing ``task_key`` would do, item by item. None when nothing is recorded."""
        async with self._lock:
            journal = await asyncio.to_thread(self._read_journal, task_key)
            if journal is None:
                return None
            return await asyncio.to_thread(self._plan, task_key, journal)

    async def summary(self, task_key: str) -> dict[str, Any] | None:
        """A small summary for ``task_complete``: how much of this task can be undone."""
        preview = await self.preview(task_key) if task_key else None
        if preview is None:
            return None
        return {
            "task_key": task_key,
            "available": preview["ready"] > 0 and not preview["undone"],
            "undoable": preview["ready"],
            "not_undoable": len(preview["not_undoable"]),
        }

    # ------------------------------------------------------------------ undo

    async def undo(self, task_key: str, preview_token: str) -> dict[str, Any]:
        """Reverse every ready entry, newest first, if the preview the person saw still holds."""
        async with self._lock:
            journal = await asyncio.to_thread(self._read_journal, task_key)
            if journal is None:
                return {"status": "error", "message": "Nothing is recorded to undo for this task."}
            plan = await asyncio.to_thread(self._plan, task_key, journal)
            if plan["preview_token"] != preview_token:
                return {
                    "status": "changed",
                    "message": "Something changed since the preview. Check the new preview before undoing.",
                    "preview": plan,
                }
            results = []
            for entry in sorted(journal["entries"], key=lambda e: e["seq"], reverse=True):
                # Re-check against the real disk: the newer entries are already undone by now.
                state, description, _effects = await asyncio.to_thread(self._check, task_key, entry, {})
                if state == "done":
                    continue
                if state == "ready":
                    try:
                        await self._apply(task_key, entry)
                        entry["status"] = "undone"
                        state = "undone"
                    except Exception as exc:
                        logger.warning("Undo of entry %s failed: %s", entry["seq"], exc)
                        state, description = "failed", f"{description} Failed: {exc}"
                results.append(
                    {"seq": entry["seq"], "target": entry.get("target", ""), "state": state, "description": description}
                )
            journal["undone_at"] = time.time()
            await asyncio.to_thread(self._write_journal, task_key, journal)
            failed = sum(1 for r in results if r["state"] == "failed")
            return {
                "status": "ok" if failed == 0 else "partial",
                "task_key": task_key,
                "undone": sum(1 for r in results if r["state"] == "undone"),
                "skipped": sum(1 for r in results if r["state"] in ("conflict", "unavailable")),
                "failed": failed,
                "results": results,
                "not_undoable": list(journal.get("not_undoable", [])),
            }

    async def _apply(self, task_key: str, entry: dict[str, Any]) -> None:
        if entry["type"] == ActionType.CLIPBOARD_WRITE.value:
            from pilot.system.clipboard import clipboard_read, clipboard_write

            if _hash_text(await clipboard_read()) != entry.get("post"):
                raise ValueError("the clipboard changed after Heliox wrote it")
            previous = self._clipboard_previous.pop((task_key, entry["seq"]))
            await clipboard_write("" if previous == EMPTY_CLIPBOARD else previous)
            return
        await asyncio.to_thread(self._apply_file, task_key, entry)

    @staticmethod
    def _restore(blob: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if blob.is_dir():
            shutil.copytree(blob, target)
            return
        tmp = target.with_name(f".{target.name}.heliox-undo-{uuid.uuid4().hex[:8]}")
        shutil.copy2(blob, tmp)
        os.replace(tmp, target)

    def _apply_file(self, task_key: str, entry: dict[str, Any]) -> None:
        kind = entry["type"]
        blob = self._blob(task_key, entry["seq"])
        if kind in (ActionType.FILE_WRITE.value, ActionType.FILE_DELETE.value):
            path = Path(entry["path"])
            if entry.get("pre") is None:
                path.unlink()
            else:
                self._restore(blob, path)
            return
        source, final = Path(entry["source"]), Path(entry["final"])
        if kind == ActionType.FILE_MOVE.value:
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(final), str(source))
        elif final.is_dir():
            shutil.rmtree(final)
        else:
            final.unlink()
        if entry.get("pre_final") is not None:
            self._restore(blob, final)

    # ------------------------------------------------------------------ listing and retention

    async def list_tasks(self, limit: int = 20) -> list[dict[str, Any]]:
        async with self._lock:
            return await asyncio.to_thread(self._list_tasks, limit)

    def _list_tasks(self, limit: int) -> list[dict[str, Any]]:
        if not self._root.is_dir():
            return []
        rows = []
        for child in self._root.iterdir():
            journal = self._read_journal(child.name) if child.is_dir() else None
            if journal is None or not (journal["entries"] or journal["not_undoable"]):
                continue
            plan = self._plan(child.name, journal)
            rows.append(
                {
                    "task_key": child.name,
                    "plan_ids": plan["plan_ids"],
                    "created_at": journal.get("created_at"),
                    "updated_at": journal.get("updated_at"),
                    "undone": plan["undone"],
                    "undoable": plan["ready"],
                    "not_undoable": len(plan["not_undoable"]),
                    "targets": [i["target"] for i in plan["items"][:3]],
                }
            )
        rows.sort(key=lambda r: r["updated_at"] or 0, reverse=True)
        return rows[: max(1, min(int(limit), 200))]

    async def prune(self, *, now: float | None = None) -> int:
        """Delete journals past retention, then the oldest until the total fits. Returns how many went."""
        async with self._lock:
            return await asyncio.to_thread(self._prune, time.time() if now is None else now)

    def _prune(self, now: float) -> int:
        if not self._root.is_dir():
            return 0
        horizon = now - self._retention_seconds()
        kept = []
        removed = 0
        for child in self._root.iterdir():
            if not child.is_dir():
                continue
            journal = self._read_journal(child.name)
            updated = float(journal.get("updated_at") or 0) if journal else 0.0
            if updated < horizon:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
            else:
                kept.append((updated, child, _tree_bytes(child)))
        total = sum(size for _updated, _child, size in kept)
        for _updated, child, size in sorted(kept, key=lambda row: row[0]):
            if total <= self._max_total_bytes():
                break
            shutil.rmtree(child, ignore_errors=True)
            total -= size
            removed += 1
        for key in [k for k in self._clipboard_previous if not self._task_dir(k[0]).exists()]:
            self._clipboard_previous.pop(key, None)
        return removed

    async def clear(self) -> None:
        """Delete every journal and saved copy."""
        async with self._lock:
            await asyncio.to_thread(shutil.rmtree, self._root, True)
            self._clipboard_previous.clear()
