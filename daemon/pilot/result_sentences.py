"""A sentence for every action that finished: what was done, to what, and the one number that matters.

Each of the action types in ``pilot.actions.ActionType`` has an entry in ``SENTENCES``, a small function from the
step's result to a ``Said``: the shown sentence (or, for a read whose answer is the point, the answer itself) and the
spoken one. They are built from the handler's structured result and the step's own parameters, never from a model,
so they are the same for every provider. A file is named by its name and at most its folder, a repository by its
name and branch, a web address by its host. The handler's full output stays in the result (the window's detail view
and the receipt); the rules of pilot.reply_rules are applied to both forms on the way out.

``step_said`` gives one step's sentence, ``plan_said`` one or two sentences about a whole plan's outcome.
tests/test_reply_contract.py drives a representative result of every action type through the reply path; an action
type added without an entry here fails it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pilot.actions import READ_ONLY_ACTIONS, ActionResult, ActionType, split_machine_lines
from pilot.reply_rules import (
    SHOWN_LIMIT,
    SPOKEN_LIMIT,
    folder_of,
    for_ear,
    for_screen,
    host_of,
    looks_like_json,
    name_of,
    parsed_json,
    plural,
    size_words,
    undoubled,
    words_cut,
)

_CUT_LINE = re.compile(r"^\[Cut: (.*)\]$", re.MULTILINE)
SHORT_ANSWER = 160  # characters: an answer this short is said whole
BEGINNING_WORDS = 12  # a long answer is said as its length and this many words of its beginning


@dataclass(frozen=True)
class Said:
    shown: str
    spoken: str


class Step:
    """One finished step, with what the sentences need: its parameters, its output text and the output's data."""

    def __init__(self, result: ActionResult) -> None:
        self.result = result
        self.action = result.action
        self.type = result.action.action_type
        self.p = result.action.parameters
        words, _machine = split_machine_lines(self.type, result.output or "")
        self.raw = str(words or "")
        self.out = _CUT_LINE.sub("", undoubled(self.raw)).strip()
        self.data = parsed_json(self.raw)

    def param(self, *names: str) -> str:
        for name in names:
            value = getattr(self.p, name, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @property
    def target(self) -> str:
        return str(self.action.target or "").strip()

    def field(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default) if isinstance(self.data, dict) else default


def _said(sentence: str, shown: str = "") -> Said:
    return Said(shown=shown or sentence, spoken=sentence)


def _at(path: str) -> str:
    """note.txt in notes; note.txt; drive C."""
    name, folder = name_of(path), folder_of(path)
    return f"{name} in {folder}" if folder and folder != name else name or "it"


def _clean_line(step: Step, words: int = 25) -> str:
    """The handler's own output when it is one plain line a person can read: no JSON, no list."""
    out = step.out
    if not out or "\n" in out or looks_like_json(out) or len(out.split()) > words:
        return ""
    return out


def _quoted(text: str, words: int = 8) -> str:
    return f'"{words_cut(text, words)}"'


def _answer(step: Step, lead: str, *, empty: str = "", last_line: bool = False) -> Said:
    """A read whose answer is the point: the answer is shown (up to SHOWN_LIMIT) and said, a long one as its length and
    beginning (or its last line, for a log)."""
    text = step.out
    if not text:
        return _said(empty or f"{lead} is empty.")
    lines = [line for line in text.splitlines() if line.strip()]
    flat = " ".join(text.split())
    if looks_like_json(text):
        return Said(shown=for_screen(text, limit=SHOWN_LIMIT), spoken=f"{lead} holds data; it's on screen.")
    if len(flat) <= SHORT_ANSWER and len(lines) <= 2:
        return Said(shown=for_screen(text, limit=SHOWN_LIMIT), spoken=f"{lead}: {flat}")
    if last_line and lines:
        spoken = f"{lead} has {plural(len(lines), 'line')}; the newest is: {words_cut(lines[-1], 16)}"
    else:
        spoken = f"{lead} is {plural(len(flat), 'character')} long, starting: {words_cut(flat, BEGINNING_WORDS)}"
    return Said(shown=for_screen(text, limit=SHOWN_LIMIT), spoken=spoken)


def _plain(sentence: str | Callable[[Step], str]) -> Callable[[Step], Said]:
    """An action with an effect: its own plain line when it has one, else the sentence."""

    def build(step: Step) -> Said:
        own = _clean_line(step)
        made = sentence(step) if callable(sentence) else sentence
        return _said(own or made) if own and not _has_raw(own) else _said(made)

    return build


def _fixed(sentence: str | Callable[[Step], str]) -> Callable[[Step], Said]:
    """An action whose sentence is always built here (its output names paths, ids or raw data)."""

    def build(step: Step) -> Said:
        return _said(sentence(step) if callable(sentence) else sentence)

    return build


def _has_raw(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]:\\|\\\\|[{}\[\]]|\b[0-9a-f]{12,}\b", text))


# ── files ────────────────────────────────────────────────────────────


def _file_read(step: Step) -> Said:
    path = step.param("path") or step.target
    is_log = name_of(path).casefold().endswith((".log", ".out", ".err"))
    return _answer(
        step, name_of(path) or "The file", empty=f"{name_of(path) or 'The file'} is empty.", last_line=is_log
    )


def _file_write(step: Step) -> Said:
    path = step.param("path") or step.target
    found = re.search(r"Written (\d+) bytes", step.out)
    size = int(found.group(1)) if found else len((getattr(step.p, "content", "") or "").encode())
    return _said(f"Wrote {size_words(size)} to {_at(path)}.")


def _file_delete(step: Step) -> Said:
    path = step.param("path") or step.target
    if step.result.skip_reason == "already_absent" or re.search(r"\bno (?:file|folder|directory)\b", step.out, re.I):
        return _said(f"There was no {_at(path)}, so nothing was deleted.")
    what = "the folder" if step.out.startswith("Deleted directory") else ""
    return _said(f"Deleted {what + ' ' if what else ''}{_at(path)}.")


def _move_or_copy(verb: str) -> Callable[[Step], Said]:
    def build(step: Step) -> Said:
        source = step.param("path") or step.target
        destination = step.param("destination")
        found = re.search(r"(?:Moved|Copied) (.+) -> (.+)$", step.out)
        if found:
            destination = found.group(2).strip()
        same_folder = folder_of(source) and folder_of(source) == folder_of(destination)
        if verb == "Moved" and same_folder and name_of(source) != name_of(destination):
            return _said(f"Renamed {name_of(source)} to {name_of(destination)}.")
        if destination and name_of(destination) == name_of(source):
            return _said(f"{verb} {name_of(source)} into {folder_of(destination) or name_of(destination)}.")
        if destination:
            return _said(f"{verb} {name_of(source)} to {_at(destination)}.")
        return _said(f"{verb} {name_of(source)}.")

    return build


def _entries(step: Step) -> tuple[list[str], list[str]]:
    files = re.findall(r"^\[f\] (.+?)\s{2}\(", step.out, re.MULTILINE)
    folders = re.findall(r"^\[d\] (.+?)\s{2}\(", step.out, re.MULTILINE)
    return files, folders


def _names(items: Sequence[str], most: int = 3) -> str:
    names = [name_of(item) for item in items[:most]]
    if len(items) > most:
        return ", ".join(names) + f" and {len(items) - most:,} more"
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}" if names else ""


def _file_list(step: Step) -> Said:
    path = step.param("path") or step.target
    where = name_of(path) or "The folder"
    if step.out.startswith("(empty directory)"):
        return _said(f"{where} is empty.")
    files, folders = _entries(step)
    parts = [plural(len(files), "file")] if files else []
    if folders:
        parts.append(plural(len(folders), "folder"))
    if not parts:
        return _answer(step, where)
    listed = f": {_names(files + folders)}" if files or folders else ""
    return Said(shown=for_screen(step.out), spoken=f"{where} has {' and '.join(parts)}{listed}.")


def _file_search(step: Step) -> Said:
    path = step.param("path") or step.target
    pattern = step.param("pattern") or "*"
    if step.out.startswith("No files matching"):
        return _said(f"No files matching {pattern} in {name_of(path) or 'that folder'}.")
    files, folders = _entries(step)
    found = files + folders
    spoken = f"Found {plural(len(found), 'match', 'matches')} for {pattern} in {name_of(path) or 'that folder'}"
    return Said(shown=for_screen(step.out), spoken=f"{spoken}: {_names(found)}." if found else spoken + ".")


def _directory_summary(step: Step) -> Said:
    path = step.param("path") or step.target
    lines = [line for line in step.out.splitlines()[1:] if line.strip() and not line.startswith("[Cut")]
    folders = sum(1 for line in lines if line.rstrip().endswith("/"))
    files = len(lines) - folders
    spoken = f"{name_of(path) or 'The folder'} holds {plural(files, 'file')} and {plural(folders, 'folder')}."
    return Said(shown=for_screen(step.out), spoken=spoken)


def _directory_size(step: Step) -> Said:
    path = step.param("path") or step.target
    found = re.search(r":\s*([\d.]+ [KMGT]?B) across ([\d,]+) files", step.out)
    if not found:
        return _answer(step, f"The size of {name_of(path)}")
    size = found.group(1).replace("KB", "kilobytes").replace("MB", "megabytes").replace("GB", "gigabytes")
    size = size.replace("TB", "terabytes").replace(" B", " bytes")
    return _said(f"{name_of(path)} takes up {size} across {found.group(2)} files.")


def _file_hash(step: Step) -> Said:
    path = step.param("path") or step.target
    found = re.match(r"(\w+):([0-9a-fA-F]+)", step.out)
    algorithm = (found.group(1) if found else step.param("hash_algorithm") or "sha256").upper().replace("SHA", "SHA-")
    shown = f"The {algorithm} of {_at(path)} is {found.group(2)}." if found else for_screen(step.out)
    return Said(shown=shown, spoken=f"I worked out the {algorithm} of {name_of(path)}; it's on screen.")


def _file_compare(step: Step) -> Said:
    first, second = name_of(step.param("path") or step.target), name_of(step.param("destination"))
    same = step.out.startswith("Identical")
    return _said(f"{first} and {second} are {'identical' if same else 'different'}.")


def _file_permissions(step: Step) -> Said:
    path = step.param("path") or step.target
    mode = step.param("permissions")
    if mode:
        return _said(f"Set the permissions of {_at(path)} to {mode}.")
    found = re.search(r":\s*(\S+)$", step.out)
    return _said(
        f"{name_of(path)} has the permissions {found.group(1)}."
        if found
        else f"Read the permissions of {name_of(path)}."
    )


# ── git ──────────────────────────────────────────────────────────────


def _repo(step: Step) -> str:
    return name_of(step.param("repo_path") or step.target) or "the repository"


def _git_status(step: Step) -> Said:
    branch = step.field("branch") or ""
    staged, unstaged, untracked = (len(step.field(key) or []) for key in ("staged", "unstaged", "untracked"))
    on = f"{_repo(step)} is on {branch}" if branch and branch != "unknown" else _repo(step)
    changed = staged + unstaged + untracked
    if not isinstance(step.data, dict):
        return _answer(step, f"The status of {_repo(step)}")
    if not changed:
        return Said(shown=for_screen(step.field("output", "")), spoken=f"{on}, and there's nothing to commit.")
    parts = [f"{staged:,} staged" if staged else "", f"{unstaged:,} changed" if unstaged else ""]
    parts.append(f"{untracked:,} untracked" if untracked else "")
    detail = ", ".join(part for part in parts if part)
    return Said(shown=for_screen(step.field("output", "")), spoken=f"{on} with {plural(changed, 'file')}: {detail}.")


def _git_diff(step: Step) -> Said:
    diff = str(step.field("diff", "") or step.field("output", "") if isinstance(step.data, dict) else step.out)
    files = re.findall(r"^diff --git a/(\S+)", diff, re.MULTILINE)
    added = len(re.findall(r"^\+(?!\+\+)", diff, re.MULTILINE))
    removed = len(re.findall(r"^-(?!--)", diff, re.MULTILINE))
    shown = for_screen(str(step.field("output", "") or diff))
    if not files:
        return Said(shown=shown or "No changes.", spoken=f"{_repo(step)} has no changes in that diff.")
    where = name_of(files[0]) if len(files) == 1 else plural(len(files), "file")
    return Said(shown=shown, spoken=f"{where} in {_repo(step)}: {plural(added, 'line')} added, {removed:,} removed.")


def _git_log(step: Step) -> Said:
    commits = step.field("commits") or []
    if not isinstance(commits, list) or not commits:
        return Said(shown=for_screen(step.field("output", "") or step.out), spoken=f"{_repo(step)} has no commits yet.")
    newest = commits[0] if isinstance(commits[0], dict) else {}
    message = str(newest.get("message") or "").strip()
    when = str(newest.get("when") or "").strip()
    spoken = f"The newest of {plural(len(commits), 'commit')} in {_repo(step)} is {_quoted(message)}"
    spoken += f", {when}." if when else "."
    return Said(shown=for_screen(step.field("output", "")), spoken=spoken)


def _git_branch(step: Step) -> Said:
    branch = step.field("branch") or step.param("name")
    new = bool(getattr(step.p, "create", False)) and "new branch" in str(step.field("output", "")).lower()
    return _said(f"Switched {_repo(step)} to the {'new ' if new else ''}branch {branch}.")


def _git_stage(step: Step) -> Said:
    files = step.field("staged_files") or getattr(step.p, "files", None) or []
    files = [str(name) for name in files if str(name).strip()]
    if len(files) == 1 and files[0] not in {".", "*"}:
        return _said(f"Staged {name_of(files[0])} in {_repo(step)}.")
    if files and all(name not in {".", "*"} for name in files):
        return _said(f"Staged {plural(len(files), 'file')} in {_repo(step)}.")
    return _said(f"Staged every change in {_repo(step)}.")


def _git_commit(step: Step) -> Said:
    message = str(step.field("commit_message") or step.param("message") or "").strip()
    branch = str(step.field("branch") or "").strip()
    count = step.field("files_committed")
    what = plural(int(count), "file") if isinstance(count, int) and count > 0 else "the staged changes"
    on = f" on {branch}" if branch else ""
    said = f"Committed {what} to {_repo(step)}{on}"
    return _said(f"{said}: {_quoted(message, 10)}." if message else f"{said}.")


def _git_push(step: Step) -> Said:
    branch = str(step.field("branch") or step.param("branch") or "").strip()
    remote = str(step.field("remote") or step.param("remote") or "origin").strip()
    what = branch or "the branch"
    return _said(f"Pushed {what} of {_repo(step)} to {remote}.")


def _git_resolve(step: Step) -> Said:
    return _said(f"Resolved the merge conflict in {_at(step.param('path') or step.target)}.")


# ── system state ─────────────────────────────────────────────────────


def _state(lead: str) -> Callable[[Step], Said]:
    """Battery, disk, memory and processor: the reading in a sentence (pilot.conversation.quick.speakable_state)."""

    def build(step: Step) -> Said:
        from pilot.conversation.quick import speakable_state

        if "No battery detected" in step.out:
            return _said("This computer has no battery.")
        sentence = speakable_state(step.out)
        if sentence:
            return Said(shown=sentence, spoken=sentence)
        return _answer(step, lead)

    return build


def _system_info(step: Step) -> Said:
    from pilot.conversation.quick import speakable_state

    sections = re.findall(r"^=== (.+?) ===", step.out, re.MULTILINE)
    sentence = speakable_state(step.out)
    if sentence and len(sections) <= 1:
        return Said(shown=for_screen(step.out), spoken=sentence)
    if len(sections) > 1:
        listed = ", ".join(section.lower() for section in sections[:-1]) + f" and {sections[-1].lower()}"
        return Said(shown=for_screen(step.out), spoken=f"Here's the system information: {listed}.")
    return _answer(step, "System information")


def _health_review(step: Step) -> Said:
    first = next((line.strip() for line in step.out.splitlines() if line.strip() and "===" not in line), "")
    return Said(shown=for_screen(step.out), spoken=words_cut(first, 24) or "The health review is on screen.")


def _network_info(step: Step) -> Said:
    up = [
        name
        for name in re.findall(r"^\s{2}(\S.*?) \(UP\):", step.out, re.MULTILINE)
        if not re.match(r"^(?:lo\d*|Loopback\b.*)$", name)  # this computer's own loopback is always up
    ]
    if not up:
        return _answer(step, "Network")
    return Said(shown=for_screen(step.out), spoken=f"{plural(len(up), 'network connection')} up: {_names(up)}.")


def _volume(step: Step) -> Said:
    own = _clean_line(step)
    if own:
        return _said(own)
    level = getattr(step.p, "level", None)
    return _said(f"Volume is at {level}." if level is not None else "Done.")


def _brightness_get(step: Step) -> Said:
    found = re.search(r"Brightness:\s*([\d.]+)%?", step.out)
    if found:
        return _said(f"Brightness is at {found.group(1)}%.")
    return _answer(step, "Brightness")


def _clipboard_read(step: Step) -> Said:
    if not step.out or step.out == "(clipboard is empty)":
        return _said("Your clipboard is empty.")
    return _answer(step, "Your clipboard")


def _process_list(step: Step) -> Said:
    rows = re.findall(r"PID=\s*\d+\s+CPU=\s*([\d.]+)%.*?\s{2}(\S.*)$", step.out, re.MULTILINE)
    name = step.param("name")
    if not rows:
        return _said(f"No program called {name} is running." if name else "No matching programs are running.")
    busiest = max(rows, key=lambda row: float(row[0]))[1].strip()
    total = re.search(r"of ([\d,]+) processes", step.out)
    count = int(total.group(1).replace(",", "")) if total else len(rows)
    if name:
        return Said(shown=for_screen(step.out), spoken=f"{plural(count, 'process', 'processes')} match {name}.")
    return Said(
        shown=for_screen(step.out),
        spoken=f"{plural(count, 'process', 'processes')} are running; the busiest is {busiest}.",
    )


def _process_kill(step: Step) -> Said:
    found = re.search(r"to process (.+?) \(PID", step.out) or re.search(r"matching '([^']+)'", step.out)
    name = found.group(1) if found else step.param("name") or "the process"
    return _said(f"Stopped {name}.")


def _process_info(step: Step) -> Said:
    name = re.search(r"^\s*name: (.+)$", step.out, re.MULTILINE)
    status = re.search(r"^\s*status: (.+)$", step.out, re.MULTILINE)
    if name:
        return Said(
            shown=for_screen(step.out), spoken=f"{name.group(1)} is {status.group(1) if status else 'running'}."
        )
    return _answer(step, "Process")


def _env_get(step: Step) -> Said:
    name = step.param("name") or "That variable"
    if "is not set" in step.out:
        return _said(f"{name} is not set.")
    value = step.out.split("=", 1)[1] if "=" in step.out else step.out
    secret = re.search(r"key|token|secret|pass|auth|cred", name, re.I)
    if re.search(r"[\\/]", value) and not secret:
        return Said(shown=for_screen(step.out), spoken=f"{name} is set to a path ending in {name_of(value)}.")
    if secret or len(value) > 60 or not value:
        return Said(shown=for_screen(step.out), spoken=f"{name} is set; its value is on screen.")
    return Said(shown=for_screen(step.out), spoken=f"{name} is set to {value}.")


def _count_lines(lead: str, noun: str, many: str = "", *, empty: str = "") -> Callable[[Step], Said]:
    """A list read: how many, said; the list shown."""

    def build(step: Step) -> Said:
        data = step.data
        if isinstance(data, list):
            count = len(data)
        elif isinstance(data, dict) and isinstance(data.get("items") or data.get("matches"), list):
            count = len(data.get("items") or data.get("matches"))
        else:
            lines = [
                line
                for line in step.out.splitlines()
                if line.strip() and not line.rstrip().endswith(":") and not line.startswith("[Cut")
            ]
            count = len(lines)
        if not step.out or count == 0 or re.match(r"^\(?(?:no |none|empty)", step.out, re.I):
            return _said(empty or f"{lead}: none.")
        shown = step.out if not isinstance(data, (list, dict)) else _listing(data)
        return Said(shown=for_screen(shown), spoken=f"{lead}: {plural(count, noun, many)}.")

    return build


def _listing(data: Any) -> str:
    """JSON data as lines a person reads: one item per line, its name or first values."""
    items = data if isinstance(data, list) else data.get("items") or data.get("matches") or [data]
    lines = []
    for item in items[:50]:
        if isinstance(item, dict):
            label = item.get("name") or item.get("title") or item.get("file") or item.get("path") or item.get("id")
            rest = [f"{key}: {value}" for key, value in item.items() if isinstance(value, (str, int, float))][:3]
            lines.append(str(label) if label else ", ".join(rest))
        else:
            lines.append(str(item))
    return "\n".join(f"- {line}" for line in lines)


def _shell(step: Step) -> Said:
    """A command, a script or code: what it printed is the answer."""
    text = step.out
    if not text or text.strip() in {"(no output)", "Command completed (no output)"}:
        return _said("It ran and printed nothing.")
    if looks_like_json(text):
        data = step.data
        count = len(data) if isinstance(data, (list, dict)) else 0
        return Said(
            shown=for_screen(_listing(data) if isinstance(data, (list, dict)) else text),
            spoken=f"It ran and printed {plural(count, 'item')}." if count else "It ran; the result is on screen.",
        )
    lines = [line for line in text.splitlines() if line.strip()]
    flat = " ".join(text.split())
    if len(flat) <= SHORT_ANSWER and len(lines) <= 2:
        return Said(shown=for_screen(text), spoken=f"It printed: {flat}")
    return Said(
        shown=for_screen(text),
        spoken=f"It printed {plural(len(lines), 'line')}; the last is: {words_cut(lines[-1], 14)}",
    )


# ── the web and messages ─────────────────────────────────────────────


def _title_of(html: str) -> str:
    found = re.search(r"<title[^>]*>(.*?)</title>", html or "", re.I | re.S)
    return " ".join(found.group(1).split()) if found else ""


def _http(step: Step, who: str = "") -> Said:
    data = step.data if isinstance(step.data, dict) else {}
    host = who or host_of(str(data.get("url") or step.param("url", "endpoint") or step.target))
    status = data.get("status")
    body = data.get("body")
    said = f"{host} answered {status}" if status else f"{host} answered"
    if isinstance(body, str):
        title = _title_of(body)
        text = re.sub(r"<(script|style|head)\b.*?</\1\s*>", " ", body, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = " ".join(text.split())
        if title:
            sentence = f"{said}. The page title is {title}."
            shown = f"{said}. The page title is {title}.\n{for_screen(text, limit=600)}"
        elif text:
            sentence = (
                f"{said}: {text}"
                if len(text) <= SHORT_ANSWER
                else f"{said} with {plural(len(text), 'character')}, starting: {words_cut(text, BEGINNING_WORDS)}"
            )
            shown = f"{said}.\n{for_screen(text, limit=600)}"
        else:
            sentence = shown = f"{said} with an empty page."
    elif isinstance(body, (list, dict)):
        noun = "item" if isinstance(body, list) else "field"
        sentence = f"{said} with {plural(len(body), noun)}."
        shown = f"{sentence}\n{_listing(body)}"
    elif data:
        sentence = shown = f"{said}."
    else:
        return _answer(step, f"The answer from {host}")
    return Said(shown=for_screen(shown), spoken=sentence)


def _scrape(step: Step) -> Said:
    host = host_of(step.param("url") or step.target)
    selector = step.param("selector")
    if step.data is not None:
        count = len(step.data) if isinstance(step.data, (list, dict)) else 0
        return Said(shown=for_screen(_listing(step.data)), spoken=f"Read {plural(count, 'item')} from {host}.")
    lines = [line for line in step.out.splitlines() if line.strip()]
    if selector and lines:
        return Said(
            shown=for_screen(step.out),
            spoken=f"Found {plural(len(lines), 'match', 'matches')} for {selector} on {host}; the first is: "
            f"{words_cut(lines[0], 10)}",
        )
    return _answer(step, f"The page on {host}")


def _send(service: str) -> Callable[[Step], Said]:
    def build(step: Step) -> Said:
        channel = step.param("channel")
        return _said(f"Posted to {service}{f' in {channel}' if channel else ''}.")

    return build


def _email_send(step: Step) -> Said:
    to = step.param("to") or "the recipient"
    subject = step.param("subject")
    return _said(f"Sent the email to {to}{f', about {_quoted(subject)}' if subject else ''}.")


def _download(step: Step) -> Said:
    destination = step.param("output_path")
    found = re.search(r"-> (.+?) \(([\d,]+) bytes\)", step.out)
    if found:
        destination = found.group(1)
    size = f", {size_words(int(found.group(2).replace(',', '')))}" if found else ""
    return _said(f"Downloaded {name_of(destination) or 'the file'} from {host_of(step.param('url'))}{size}.")


def _search_content(step: Step) -> Said:
    phrase = step.param("search_text") or "that"
    where = name_of(step.param("directory", "path") or step.target) or "that folder"
    data = step.data if isinstance(step.data, dict) else {}
    matches = data.get("matches") if isinstance(data.get("matches"), list) else None
    if step.out.startswith("No matches") or matches == []:
        return _said(f"No file in {where} mentions {_quoted(phrase)}.")
    if matches is None:
        return _answer(step, f"The search for {_quoted(phrase)}")
    files = []
    for match in matches:
        name = match.get("file") or match.get("path") if isinstance(match, dict) else None
        if name and name not in files:
            files.append(name)
    spoken = f"Found {_quoted(phrase)} {plural(len(matches), 'time')} in {plural(len(files) or 1, 'file')}"
    spoken += f": {_names(files)}." if files else "."
    shown_lines = [spoken]
    for match in matches[:20]:
        if isinstance(match, dict):
            line = match.get("line") or match.get("line_number") or ""
            text = match.get("text") or match.get("content") or match.get("match") or ""
            shown_lines.append(f"- {match.get('file') or match.get('path')}:{line} {text}".rstrip())
    return Said(shown=for_screen("\n".join(shown_lines)), spoken=spoken)


def _log_analyze(step: Step) -> Said:
    data = step.data if isinstance(step.data, dict) else {}
    if not data:
        return _answer(step, name_of(step.param("log_path")) or "The log", last_line=True)
    name = name_of(step.param("log_path") or step.target) or "The log"
    total = data.get("events_total") or 0
    anomalies = data.get("indicators") or []
    timeline = data.get("timeline") or []
    newest = re.sub(r"^\[[^\]]*\]\s*", "", str(timeline[-1])) if timeline else ""
    spoken = f"{name} has {plural(int(total), 'event')}"
    spoken += f" and {plural(len(anomalies), 'anomaly', 'anomalies')}" if anomalies else ", nothing unusual"
    spoken += f"; the newest: {words_cut(newest, 14)}" if newest else "."
    shown = "\n".join([str(data.get("summary") or spoken), *[str(line) for line in timeline[-10:]]])
    return Said(shown=for_screen(shown), spoken=spoken)


# ── triggers ─────────────────────────────────────────────────────────
# Said from what a trigger watches and what it does, never from its generated name or id.


def _watch(kind: str, condition: dict) -> tuple[str, str]:
    """(what is watched, when it fires) in plain words: ("watched.txt", "when it changes")."""
    path = name_of(str(condition.get("path") or ""))
    threshold = condition.get("threshold")
    process = str(condition.get("name") or "").strip()
    above = f" goes above {threshold} percent" if threshold is not None else " runs high"
    table = {
        "file_modified": (path or "that file", "when it changes"),
        "file_change": (path or "that file", "when it changes"),
        "file_created": (path or "that folder", "when a new file appears in it"),
        "file_deleted": (path or "that file", "when it is deleted"),
        "cpu_threshold": ("the processor", f"when it{above}"),
        "memory_threshold": ("the memory", f"when its use{above}"),
        "disk_threshold": (f"the disk{f' {path}' if path else ''}", f"when it fills{above.replace('goes', '')}"),
        "battery_low": (
            "the battery",
            f"when it drops below {threshold if threshold is not None else 20} percent",
        ),
        "process_started": (process or "that program", "when it starts"),
        "process_stopped": (process or "that program", "when it stops"),
        "network_change": ("the network", "when it changes"),
        "custom_condition": ("your condition", "when it holds"),
    }
    return table.get(kind, ("it", "when it fires"))


_ECHO_INTO = re.compile(r"""^\s*echo\s+(?P<text>.+?)\s*>>?\s*(?P<file>\S+)\s*$""", re.IGNORECASE)


def _job(command: str) -> str:
    """What a trigger does when it fires, in plain words: "add "seen a change" to fired.log", "tell you"."""
    command = " ".join(str(command or "").split())
    if not command:
        return "tell you"
    found = _ECHO_INTO.match(command)
    if found:
        text = found.group("text").strip("'\"")
        return f'add "{words_cut(text, 8)}" to {name_of(found.group("file"))}'
    if re.search(r"[|&;<>$`\\/{}=]|^\S+\.(?:exe|sh|ps1|py|bat)\b", command):
        return "run its command"
    return words_cut(command[:1].lower() + command[1:], 10).rstrip(".")


def _step_job(record: Any) -> str:
    """One fixed step of a trigger in plain words: "add "seen" to fired.log", "write notes.txt", "tell you"."""
    from pilot.actions import Action

    try:
        action = Action.model_validate(record)
    except ValueError:
        return "a step that cannot be read"
    params = action.parameters
    command = " ".join(str(getattr(params, "command", "") or "").split())
    if command:
        # An echo into a file is said as adding the line; any other command is named as the command it is.
        return _job(command) if _ECHO_INTO.match(command) else f"run the command {_quoted(command, 12)}"
    if action.action_type is ActionType.NOTIFY:
        return "tell you"
    path = str(getattr(params, "path", "") or "")
    what = name_of(path) if path else name_of(str(action.target or ""))
    verb = action.action_type.value.replace("_", " ")
    if action.action_type is ActionType.FILE_WRITE:
        verb = "write"
    return f"{verb} {what}".strip()


_PLACEHOLDER = "{PREV_OUTPUT}"


def _record_of(record: Any) -> dict:
    return record.model_dump(mode="json") if hasattr(record, "model_dump") else dict(record or {})


def hidden_commands(steps: Sequence[Any]) -> list[str]:
    """The commands among a trigger's fixed steps that ``steps_job`` says in other words (an echo into a file)."""
    found = []
    for record in steps or []:
        command = " ".join(str((_record_of(record).get("parameters") or {}).get("command") or "").split())
        if command and _ECHO_INTO.match(command):
            found.append(command)
    return found


def steps_job(steps: Sequence[Any]) -> str:
    """What a trigger's fixed steps do, in a few words; "" when it has none. A read of a file followed by a write of
    it around the earlier text is an append (the planner's form of one): said as "add "<text>" to <file>"."""
    records = [_record_of(record) for record in steps or []]
    jobs: list[str] = []
    index = 0
    while index < len(records):
        record = records[index]
        following = records[index + 1] if index + 1 < len(records) else {}
        path = str((record.get("parameters") or {}).get("path") or "")
        content = str((following.get("parameters") or {}).get("content") or "")
        if (
            record.get("action_type") == "file_read"
            and following.get("action_type") == "file_write"
            and path
            and str((following.get("parameters") or {}).get("path") or "") == path
            and _PLACEHOLDER in content
        ):
            added = " ".join(content.replace(_PLACEHOLDER, " ").split())
            jobs.append(f"add {_quoted(added)} to {name_of(path)}" if added else f"write {name_of(path)}")
            index += 2
            continue
        jobs.append(_step_job(record))
        index += 1
    if not jobs:
        return ""
    more = f", and {plural(len(jobs) - 3, 'more step')}" if len(jobs) > 3 else ""
    return " and ".join(jobs[:3]) + more


def trigger_entry(trigger: Any) -> dict:
    """A trigger as the store keeps it (pilot.system.triggers), from the stored dict or the engine's object."""
    if isinstance(trigger, dict):
        return trigger
    from dataclasses import asdict, is_dataclass

    entry = asdict(trigger) if is_dataclass(trigger) else dict(getattr(trigger, "__dict__", {}))
    kind = entry.get("trigger_type")
    entry["trigger_type"] = getattr(kind, "value", kind)
    return entry


def _trigger_parts(entry: dict) -> tuple[str, str, str]:
    """(what it watches, when it fires, what it does): the fixed steps when it has them, else its words."""
    condition = entry.get("condition") if isinstance(entry.get("condition"), dict) else {}
    watched, when = _watch(str(entry.get("trigger_type") or ""), condition)
    steps = entry.get("steps") if isinstance(entry.get("steps"), list) else []
    return watched, when, steps_job(steps) or _job(str(entry.get("action_command") or ""))


def trigger_named(trigger: Any) -> str:
    """How every sentence names a trigger: by what it watches, never by its id or a made-up name."""
    entry = trigger_entry(trigger)
    if entry.get("trigger_type") == "time_interval":
        return "the trigger that runs on an interval"
    watched, _when, _job_words = _trigger_parts(entry)
    return f"the trigger that watches {watched}"


def trigger_fire_sentence(trigger: Any, outcome: str, *, error: str = "", seconds: int = 0) -> str:
    """What the person is told about one fire. ``outcome``: ran, failed (``error`` is already a plain sentence),
    timeout, fault, waits (a step needs an approval nobody gave), unreadable, unavailable, job_done, job_failed."""
    entry = trigger_entry(trigger)
    named = trigger_named(entry)
    lead = named[:1].upper() + named[1:] + " fired"
    _watched, _when, job = _trigger_parts(entry)
    sentences = {
        "ran": f"{lead}, and I did what it asks: {job}.",
        "failed": f"{lead}, but a step did not work. {error}".strip(),
        "timeout": f"{lead}, but its steps took longer than {plural(seconds, 'second')}, so I stopped them.",
        "fault": f"{lead}, but something went wrong inside Heliox and its steps stopped.",
        "waits": (
            f"{lead}, but I did not do what it asks ({job}), because that needs your approval and you did not give "
            "it when the trigger was made."
        ),
        "unreadable": f"{lead}, but I could not read its stored steps, so nothing ran.",
        "unavailable": f"{lead}, but I could not run its steps just then.",
        "job_done": f"{lead}, and its job finished.",
        "job_failed": f"{lead}, but its job did not finish.",
    }
    return sentences[outcome]


def _trigger_create(step: Step) -> Said:
    data = step.data if isinstance(step.data, dict) else {}
    entry = data or {
        "trigger_type": step.param("trigger_type"),
        "condition": getattr(step.p, "condition", None) or {},
        "action_command": step.param("action_command"),
    }
    watched, when, job = _trigger_parts(entry)
    if entry.get("trigger_type") == "time_interval":
        seconds = int(getattr(step.p, "cooldown_seconds", 0) or 0)
        every = f"every {plural(seconds // 60, 'minute')}" if seconds >= 60 else "on its interval"
        sentence = f"I'll {job} {every}."
    else:
        sentence = f"I'll watch {watched} and {job} {when}."
    if data.get("note"):
        sentence = f"I was already set to watch {watched}, so nothing changed."
    elif entry.get("steps") and not entry.get("steps_approved"):
        # A step that needs approval waits at a fire for the person (pilot.server._trigger_waits).
        sentence += " If a step needs your approval, I'll ask you when it fires instead of running it."
    return _said(sentence)


def _what_it_watches(entry: dict) -> str:
    watched, when, _job_words = _trigger_parts(entry)
    waiting = ", and a fire is waiting for your approval" if entry.get("pending") else ""
    if entry.get("trigger_type") == "time_interval":
        return f"it runs on an interval{waiting}"
    return f"it watches {watched} {when.replace('when it changes', 'for changes')}{waiting}"


def _trigger_list(step: Step) -> Said:
    if not step.data or step.out.startswith("No triggers"):
        return _said("You have no triggers.")
    items = [item for item in (step.data if isinstance(step.data, list) else []) if isinstance(item, dict)]
    if len(items) == 1:
        spoken = f"You have 1 trigger: {_what_it_watches(items[0])}."
    else:
        parts = [_what_it_watches(item).replace("it ", "one ", 1) for item in items[:3]]
        more = f", and {len(items) - 3:,} more" if len(items) > 3 else ""
        spoken = f"You have {plural(len(items), 'trigger')}: {'; '.join(parts)}{more}."
    lines = []
    for item in items[:50]:
        watched, when, job = _trigger_parts(item)
        lines.append(f"- {watched}, {when}: {job} ({item.get('name') or item.get('id')})")
    return Said(shown=for_screen("\n".join([spoken, *lines])), spoken=spoken)


def _trigger_delete(step: Step) -> Said:
    # The handler answers with the removed entry (pilot.system.triggers.deleted_record): say what it watched.
    removed = step.data if isinstance(step.data, dict) and step.data.get("deleted") is True else None
    if removed is not None:
        if removed.get("trigger_type") == "time_interval":
            return _said("I've removed the trigger that ran on an interval.")
        watched, _when, _job_words = _trigger_parts(removed)
        return _said(f"I've stopped watching {removed.get('watched') or watched}.")
    # An older answer names only the id; what the trigger watched is the step's target when the planner named it.
    target = step.target
    if target and re.search(r"\.\w{1,6}$|[\/]", target):
        return _said(f"I've stopped watching {name_of(target)}.")
    return _said("I've removed that trigger.")


def _trigger_start(step: Step) -> Said:
    found = re.search(r"with (\d+) triggers", step.out)
    count = int(found.group(1)) if found else 0
    return _said(
        f"Your {plural(count, 'trigger')} {'is' if count == 1 else 'are'} on again."
        if count
        else "Triggers are on again."
    )


# ── calendar, mail, tools ────────────────────────────────────────────


def _calendar_create(step: Step) -> Said:
    summary = step.param("summary") or "the event"
    start = step.param("start")
    return _said(f"Added {_quoted(summary)} to your calendar{f' for {start}' if start else ''}.")


def _tool_call(label: Callable[[Step], str]) -> Callable[[Step], Said]:
    """A tool, plugin, skill or MCP call: a short plain answer is said, any other is on screen."""

    def build(step: Step) -> Said:
        name = label(step)
        own = _clean_line(step, words=30)
        if own and not _has_raw(own):
            return _said(f"{name} answered: {own}")
        if not step.out:
            return _said(f"{name} finished.")
        shown = _listing(step.data) if isinstance(step.data, (list, dict)) else step.out
        return Said(shown=for_screen(shown), spoken=f"{name} finished; the result is on screen.")

    return build


def _window(verb: str) -> Callable[[Step], Said]:
    def build(step: Step) -> Said:
        own = _clean_line(step)
        if own and not _has_raw(own):
            return _said(own)  # the handler's own line ("Closed ChatGPT window")
        what = step.param("title", "process_name") or "the window"
        return _said(f"{verb} {what}.")

    return build


def _open_application(step: Step) -> Said:
    from pilot.conversation.handover import _title_app

    found = re.match(r"Launched (.+?)(?: \(|\.$)", step.out)
    own = _clean_line(step)
    if not found and own and not _has_raw(own):
        return _said(own)  # the handler's own sentence ("Discord is open.")
    name = found.group(1) if found else step.param("name") or step.target
    if "already opened in this task" in step.out:
        return _said(f"{_title_app(name)} was already open.")
    return _said(f"{_title_app(name_of(name))} is open.")


def _screenshot(step: Step) -> Said:
    found = re.search(r"saved to (.+)$", step.out, re.I)
    path = found.group(1) if found else step.param("output_path")
    return _said(f"Saved a screenshot as {_at(path)}." if path else "Took a screenshot.")


def _screen_text(step: Step) -> Said:
    return _answer(step, "The screen says", empty="No text was found on the screen.")


def _elements(lead: str) -> Callable[[Step], Said]:
    def build(step: Step) -> Said:
        lines = [line for line in step.out.splitlines() if line.strip()]
        if not lines:
            return _said(f"I found nothing {lead}.")
        return Said(shown=for_screen(step.out), spoken=f"I found {plural(len(lines), 'item')} {lead}.")

    return build


def _page_info(step: Step) -> Said:
    data = step.data if isinstance(step.data, dict) else {}
    title = str(data.get("title") or "").strip()
    host = host_of(str(data.get("url") or ""))
    if title or host:
        return Said(
            shown=for_screen(f"{title}\n{data.get('url', '')}".strip()),
            spoken=f"The page is {_quoted(title or host, 12)}{f' on {host}' if host and title else ''}.",
        )
    found = re.search(r"Title:\s*(.+)", step.out)
    if found:
        return Said(shown=for_screen(step.out), spoken=f"The page is {_quoted(found.group(1), 12)}.")
    return _answer(step, "The page")


def _browser_navigate(step: Step) -> Said:
    return _said(f"Opened {host_of(step.param('url') or step.target) or 'the page'}.")


def _browser_extract(step: Step) -> Said:
    return _answer(step, "The page says", empty="That part of the page is empty.")


def _key_name(step: Step) -> str:
    keys = getattr(step.p, "keys", None) or []
    if keys:
        return " ".join(str(key).capitalize() for key in keys)
    return str(step.param("key") or "the key").capitalize()


SENTENCES: dict[ActionType, Callable[[Step], Said]] = {
    # files
    ActionType.FILE_READ: _file_read,
    ActionType.FILE_WRITE: _file_write,
    ActionType.FILE_DELETE: _file_delete,
    ActionType.FILE_MOVE: _move_or_copy("Moved"),
    ActionType.FILE_COPY: _move_or_copy("Copied"),
    ActionType.FILE_LIST: _file_list,
    ActionType.FILE_SEARCH: _file_search,
    ActionType.DIRECTORY_SUMMARY: _directory_summary,
    ActionType.DIRECTORY_SIZE: _directory_size,
    ActionType.FILE_HASH: _file_hash,
    ActionType.FILE_COMPARE: _file_compare,
    ActionType.FILE_PERMISSIONS: _file_permissions,
    ActionType.FILE_PARSE: lambda step: _answer(step, name_of(step.param("path") or step.target) or "The file"),
    ActionType.FILE_SEARCH_CONTENT: _search_content,
    ActionType.DOWNLOAD_FILE: _download,
    # git
    ActionType.GIT_RESOLVE: _git_resolve,
    ActionType.GIT_STATUS: _git_status,
    ActionType.GIT_DIFF: _git_diff,
    ActionType.GIT_LOG: _git_log,
    ActionType.GIT_BRANCH: _git_branch,
    ActionType.GIT_STAGE: _git_stage,
    ActionType.GIT_COMMIT: _git_commit,
    ActionType.GIT_PUSH: _git_push,
    # packages and services
    ActionType.PACKAGE_INSTALL: _fixed(lambda s: f"Installed {s.param('name') or 'the package'}."),
    ActionType.PACKAGE_REMOVE: _fixed(lambda s: f"Removed {s.param('name') or 'the package'}."),
    ActionType.PACKAGE_UPDATE: _fixed(
        lambda s: f"Updated {plural(n, 'package')}." if (n := s.out.count("[UPGRADE]")) else "Updated your packages."
    ),
    ActionType.PACKAGE_SEARCH: _count_lines("Packages found", "result"),
    ActionType.SERVICE_START: _fixed(lambda s: f"Started the {s.param('name')} service."),
    ActionType.SERVICE_STOP: _fixed(lambda s: f"Stopped the {s.param('name')} service."),
    ActionType.SERVICE_RESTART: _fixed(lambda s: f"Restarted the {s.param('name')} service."),
    ActionType.SERVICE_ENABLE: _fixed(lambda s: f"The {s.param('name')} service will start on its own now."),
    ActionType.SERVICE_DISABLE: _fixed(lambda s: f"The {s.param('name')} service won't start on its own now."),
    ActionType.SERVICE_STATUS: lambda s: _answer(s, f"The {s.param('name')} service"),
    ActionType.GNOME_SETTING_READ: lambda s: _answer(s, f"{s.param('key') or 'The setting'} is"),
    ActionType.GNOME_SETTING_WRITE: _fixed(lambda s: f"Set {s.param('key') or 'the setting'} to {s.param('value')}."),
    ActionType.DBUS_CALL: _tool_call(lambda s: s.param("method") or "The system service"),
    # commands and code
    ActionType.SHELL_COMMAND: _shell,
    ActionType.SHELL_SCRIPT: _shell,
    ActionType.PTY_EXEC: _shell,
    ActionType.SSH_COMMAND: _shell,
    ActionType.SSH_SCRIPT: _shell,
    ActionType.CODE_EXECUTE: _shell,
    ActionType.CODE_GENERATE_AND_RUN: _shell,
    # apps, links, notifications
    ActionType.OPEN_URL: _fixed(
        lambda s: f"Opened {host_of(s.param('url') or s.target) or 'the page'} in your browser."
    ),
    ActionType.OPEN_APPLICATION: _open_application,
    ActionType.NOTIFY: _fixed(lambda s: f"Showed the notification {_quoted(s.param('summary') or 'it')}."),
    ActionType.REQUEST_HUMAN_HELP: _fixed(
        lambda s: (
            f"Over to you: {words_cut(s.param('help_needed') or 'this step needs you', 14)}. I'll carry on "
            "when you hand control back."
        )
    ),
    # processes, clipboard, system state
    ActionType.PROCESS_LIST: _process_list,
    ActionType.PROCESS_KILL: _process_kill,
    ActionType.PROCESS_INFO: _process_info,
    ActionType.CLIPBOARD_READ: _clipboard_read,
    ActionType.CLIPBOARD_WRITE: _fixed(
        lambda s: f"Copied {plural(len(getattr(s.p, 'content', '') or ''), 'character')} to the clipboard."
    ),
    ActionType.SYSTEM_INFO: _system_info,
    ActionType.SYSTEM_HEALTH_REVIEW: _health_review,
    ActionType.DISK_USAGE: _state("Disk"),
    ActionType.MEMORY_USAGE: _state("Memory"),
    ActionType.CPU_USAGE: _state("Processor"),
    ActionType.NETWORK_INFO: _network_info,
    ActionType.BATTERY_INFO: _state("Battery"),
    # power
    ActionType.POWER_SHUTDOWN: _fixed(
        lambda s: (
            f"Shutting down in {plural(d, 'second')}." if (d := getattr(s.p, "delay_seconds", 0)) else "Shutting down."
        )
    ),
    ActionType.POWER_RESTART: _fixed("Restarting the computer."),
    ActionType.POWER_SLEEP: _fixed("Putting the computer to sleep."),
    ActionType.POWER_LOCK: _fixed("Locked the screen."),
    ActionType.POWER_LOGOUT: _fixed("Signing you out."),
    # schedules and environment
    ActionType.SCHEDULE_CREATE: _fixed(
        lambda s: f"Scheduled {s.param('name') or 'the task'}{f' for {sch}' if (sch := s.param('schedule')) else ''}."
    ),
    ActionType.SCHEDULE_LIST: _count_lines("Scheduled tasks", "task", empty="There are no scheduled tasks."),
    ActionType.SCHEDULE_DELETE: _fixed(lambda s: f"Deleted the scheduled task {s.param('name', 'task_id')}."),
    ActionType.ENV_GET: _env_get,
    ActionType.ENV_SET: _fixed(lambda s: f"Set {s.param('name') or 'the variable'}."),
    ActionType.ENV_LIST: _count_lines("Environment variables", "variable"),
    # windows, sound, screen
    ActionType.WINDOW_LIST: _count_lines("Open windows", "window"),
    ActionType.WINDOW_FOCUS: _window("Switched to"),
    ActionType.WINDOW_CLOSE: _window("Closed"),
    ActionType.WINDOW_MINIMIZE: _window("Minimized"),
    ActionType.WINDOW_MAXIMIZE: _window("Maximized"),
    ActionType.VOLUME_GET: _volume,
    ActionType.VOLUME_SET: _volume,
    ActionType.VOLUME_MUTE: _volume,
    ActionType.BRIGHTNESS_GET: _brightness_get,
    ActionType.BRIGHTNESS_SET: _fixed(lambda s: f"Brightness is at {getattr(s.p, 'level', '')}%."),
    ActionType.SCREENSHOT: _screenshot,
    # network, disks, users, registry
    ActionType.WIFI_LIST: _count_lines("Wi-Fi networks in range", "network"),
    ActionType.WIFI_CONNECT: _fixed(lambda s: f"Connected to {s.param('ssid') or 'the network'}."),
    ActionType.WIFI_DISCONNECT: _fixed("Disconnected from Wi-Fi."),
    ActionType.DISK_LIST: _count_lines("Disks", "disk"),
    ActionType.DISK_MOUNT: _fixed(
        lambda s: f"Mounted {name_of(s.param('device'))} at {name_of(s.param('mount_point'))}."
    ),
    ActionType.DISK_UNMOUNT: _fixed(lambda s: f"Unmounted {name_of(s.param('mount_point', 'device')) or 'the disk'}."),
    ActionType.USER_LIST: _count_lines("User accounts", "account"),
    ActionType.USER_INFO: lambda s: _answer(s, "Your account"),
    ActionType.REGISTRY_READ: lambda s: _answer(s, s.param("value_name") or "That registry value"),
    ActionType.REGISTRY_WRITE: _fixed(lambda s: f"Set {s.param('value_name') or 'the value'} in the registry."),
    ActionType.LOG_ANALYZE: _log_analyze,
    # mouse and keyboard
    ActionType.MOUSE_CLICK: _fixed("Clicked."),
    ActionType.MOUSE_DOUBLE_CLICK: _fixed("Double-clicked."),
    ActionType.MOUSE_RIGHT_CLICK: _fixed("Right-clicked."),
    ActionType.MOUSE_MOVE: _fixed("Moved the pointer."),
    ActionType.MOUSE_DRAG: _fixed("Dragged it."),
    ActionType.MOUSE_SCROLL: _plain(
        lambda s: "Scrolled up." if (getattr(s.p, "amount", 0) or 0) > 0 else "Scrolled down."
    ),
    ActionType.MOUSE_POSITION: lambda s: _answer(s, "The pointer is at"),
    ActionType.KEYBOARD_TYPE: _fixed("Typed it."),
    ActionType.KEYBOARD_PRESS: _plain(lambda s: f"Pressed {_key_name(s)}."),
    ActionType.KEYBOARD_HOTKEY: _fixed(lambda s: f"Pressed {_key_name(s)}."),
    ActionType.KEYBOARD_HOLD: _fixed(lambda s: f"Held {_key_name(s)}."),
    # the screen
    ActionType.SCREEN_OCR: _screen_text,
    ActionType.SCREEN_FIND_TEXT: lambda s: _answer(s, f"Looking for {_quoted(s.param('target_text') or 'it')}"),
    ActionType.SCREEN_ANALYZE: lambda s: _answer(s, "On the screen"),
    ActionType.SCREEN_ELEMENT_MAP: _elements("on the screen"),
    ActionType.SCREEN_ACCESSIBILITY_SNAPSHOT: _elements("in the window"),
    ActionType.SCREEN_DETECT_ELEMENTS: _elements("on the screen"),
    # the browser
    ActionType.BROWSER_NAVIGATE: _browser_navigate,
    ActionType.BROWSER_CLICK: _plain("Clicked it."),
    ActionType.BROWSER_CLICK_TEXT: _plain(lambda s: f"Clicked {_quoted(s.param('text') or 'it')}."),
    ActionType.BROWSER_TYPE: _fixed("Typed it into the page."),
    ActionType.BROWSER_SELECT: _fixed(lambda s: f"Chose {_quoted(s.param('value') or 'it')}."),
    ActionType.BROWSER_HOVER: _fixed("Pointed at it."),
    ActionType.BROWSER_SCROLL: _plain(lambda s: f"Scrolled {s.param('direction') or 'down'}."),
    ActionType.BROWSER_EXTRACT: _browser_extract,
    ActionType.BROWSER_EXTRACT_TABLE: _count_lines("Table rows", "row"),
    ActionType.BROWSER_EXTRACT_LINKS: _count_lines("Links on the page", "link"),
    ActionType.BROWSER_EXECUTE_JS: _tool_call(lambda s: "The page script"),
    ActionType.BROWSER_SCREENSHOT: _screenshot,
    ActionType.BROWSER_FILL_FORM: _fixed(
        lambda s: f"Filled in {plural(len(getattr(s.p, 'fields', None) or {}), 'field')}."
    ),
    ActionType.BROWSER_NEW_TAB: _fixed(
        lambda s: f"Opened {host_of(u)} in a new tab." if (u := s.param("url")) else "Opened a new tab."
    ),
    ActionType.BROWSER_CLOSE_TAB: _fixed("Closed the tab."),
    ActionType.BROWSER_LIST_TABS: _count_lines("Open tabs", "tab"),
    ActionType.BROWSER_SWITCH_TAB: _fixed(lambda s: f"Switched to tab {(getattr(s.p, 'tab_index', 0) or 0) + 1}."),
    ActionType.BROWSER_BACK: _fixed("Went back a page."),
    ActionType.BROWSER_FORWARD: _fixed("Went forward a page."),
    ActionType.BROWSER_REFRESH: _fixed("Reloaded the page."),
    ActionType.BROWSER_WAIT: _fixed("The page is ready."),
    ActionType.BROWSER_CLOSE: _fixed("Closed the browser."),
    ActionType.BROWSER_PAGE_INFO: _page_info,
    ActionType.BROWSER_SNAPSHOT: _elements("on the page"),
    # triggers
    ActionType.TRIGGER_CREATE: _trigger_create,
    ActionType.TRIGGER_LIST: _trigger_list,
    ActionType.TRIGGER_DELETE: _trigger_delete,
    ActionType.TRIGGER_START: _trigger_start,
    ActionType.TRIGGER_STOP: _fixed("Your triggers are paused until you turn them on again."),
    # calendar
    ActionType.CALENDAR_PARSE: _count_lines("Events in the file", "event"),
    ActionType.CALENDAR_SYNC: _fixed("Synced your calendar."),
    ActionType.CALENDAR_CREATE_EVENT: _calendar_create,
    ActionType.CALENDAR_LIST_EVENTS: _count_lines("Events", "event", empty="There are no events."),
    ActionType.CALENDAR_DELETE_EVENT: _fixed("Deleted the event."),
    ActionType.CALENDAR_FETCH: _count_lines("Events", "event", empty="There are no events."),
    ActionType.CALENDAR_RECONCILE: lambda s: _answer(s, "Your calendar"),
    # the web, messages and mail
    ActionType.API_REQUEST: _http,
    ActionType.API_GITHUB: lambda s: _http(s, "GitHub"),
    ActionType.API_SEND_EMAIL: _plain(lambda s: _email_send(s).spoken),
    ActionType.API_WEBHOOK: _fixed(lambda s: f"Sent the webhook to {host_of(s.param('url', 'webhook_url'))}."),
    ActionType.API_SLACK: _send("Slack"),
    ActionType.API_DISCORD: _send("Discord"),
    ActionType.API_SCRAPE: _scrape,
    ActionType.EMAIL_FETCH: _count_lines("New emails", "email", empty="There are no new emails."),
    ActionType.EMAIL_SUMMARIZE: lambda s: _answer(s, "Your email"),
    ActionType.EMAIL_REPLY: _fixed("Sent your reply."),
    # workspace and tools
    ActionType.WORKSPACE_INDEX: _fixed(
        lambda s: f"Indexed {name_of(s.param('folder_path') or s.target) or 'the folder'}{_indexed_count(s)}."
    ),
    ActionType.WORKSPACE_SEARCH: _count_lines("Workspace results", "result", empty="Nothing in the workspace matched."),
    ActionType.WASM_CALL: _tool_call(lambda s: f"The {s.param('tool') or 'tool'}"),
    ActionType.PLUGIN_CALL: _tool_call(lambda s: "The plugin"),
    ActionType.SKILL_RUN: _tool_call(lambda s: f"The skill {s.param('skill_id')}".strip()),
    ActionType.MCP_LIST_TOOLS: _count_lines("Tools available", "tool"),
    ActionType.MCP_CALL_TOOL: _tool_call(lambda s: f"The tool {s.param('mcp_tool')}".strip()),
}


def _indexed_count(step: Step) -> str:
    found = re.search(r"([\d,]+) (?:files|documents|chunks)", step.out)
    return f", {found.group(1)} {found.group(0).split()[-1]}" if found else ""


def step_said(result: ActionResult, *, limit: int = SPOKEN_LIMIT) -> Said:
    """One finished step's sentence, with the rules of pilot.reply_rules applied to both forms (``limit``: the most
    characters said; the contract test asks for an uncut copy to check a cut never falls inside a word)."""
    step = Step(result)
    if result.skip_reason in {"already_absent", "already_satisfied"} and step.out:
        # Not run because its state was already there: the handler's sentence says what was found.
        return Said(shown=for_screen(step.out), spoken=for_ear(step.out, limit=limit) or "It was already done.")
    build = SENTENCES.get(step.type)
    said = build(step) if build is not None else _plain("Done.")(step)
    said = _kept_private(step, said)
    shown = for_screen(said.shown) or for_screen(said.spoken)
    if looks_like_json(shown):
        shown = said.spoken
    return Said(shown=shown, spoken=for_ear(said.spoken, limit=limit) or for_ear(shown, limit=limit) or "Done.")


# Reads whose content is the answer, and what the sentence calls that content. Before any of it is said or shown as
# the reply, it goes through the detectors Heliox uses for a secret said or typed (pilot.conversation.withheld).
_CONTENT_LEADS: dict[ActionType, Callable[[Step], str]] = {
    ActionType.CLIPBOARD_READ: lambda s: "Your clipboard",
    ActionType.FILE_READ: lambda s: name_of(s.param("path") or s.target) or "That file",
    ActionType.FILE_PARSE: lambda s: name_of(s.param("path") or s.target) or "That file",
    ActionType.LOG_ANALYZE: lambda s: name_of(s.param("log_path") or s.target) or "That log",
    ActionType.ENV_GET: lambda s: s.param("name") or "That variable",
    ActionType.ENV_LIST: lambda s: "Your environment",
    ActionType.API_REQUEST: lambda s: "The answer from " + (host_of(s.param("url") or s.target) or "that address"),
    ActionType.API_GITHUB: lambda s: "GitHub's answer",
    ActionType.API_SCRAPE: lambda s: "The page",
    ActionType.BROWSER_EXTRACT: lambda s: "The page",
    ActionType.BROWSER_EXECUTE_JS: lambda s: "The page script's answer",
    ActionType.SCREEN_OCR: lambda s: "The screen",
    ActionType.SCREEN_ANALYZE: lambda s: "The screen",
    ActionType.SHELL_COMMAND: lambda s: "The output",
    ActionType.SHELL_SCRIPT: lambda s: "The output",
    ActionType.PTY_EXEC: lambda s: "The output",
    ActionType.SSH_COMMAND: lambda s: "The output",
    ActionType.SSH_SCRIPT: lambda s: "The output",
    ActionType.CODE_EXECUTE: lambda s: "The output",
    ActionType.CODE_GENERATE_AND_RUN: lambda s: "The output",
    ActionType.EMAIL_SUMMARIZE: lambda s: "Your email",
    ActionType.REGISTRY_READ: lambda s: s.param("value_name") or "That registry value",
    ActionType.GNOME_SETTING_READ: lambda s: s.param("key") or "That setting",
    ActionType.MCP_CALL_TOOL: lambda s: "The tool's answer",
    ActionType.SKILL_RUN: lambda s: "The skill's answer",
}


def _content(step: Step) -> str:
    """The content a read returned: an HTTP answer's body, a log's lines, or the output itself."""
    data = step.data if isinstance(step.data, dict) else {}
    if step.type in {ActionType.API_REQUEST, ActionType.API_GITHUB} and "body" in data:
        body = data.get("body")
        return body if isinstance(body, str) else json.dumps(body)
    if step.type is ActionType.LOG_ANALYZE and data:
        return "\n".join(str(line) for line in data.get("timeline") or [])
    return step.out


def private_kind(text: str, *, content: bool = True) -> str:
    """What a text holds that must not be read out ("a password", "a key", "a card number"), or "". The detectors of
    a secret said or typed (pilot.conversation.withheld), and a single token that looks like a key or password."""
    from pilot.conversation.withheld import withhold
    from pilot.workflows.clipboard_history import looks_secret

    found = withhold(text or "")
    if found is not None:
        return found.kind
    token = (text or "").strip()
    # A single token (a clipboard holding only a password) is judged by its shape; a sentence Heliox wrote is not.
    single = content and token and " " not in token and looks_secret(token)
    return "a password or a code" if single else ""


def _masked(text: str) -> str:
    """The content as the window may show it: each secret replaced by "[left out: a password]"."""
    from pilot.conversation.withheld import withhold

    found = withhold(text or "")
    return found.text if found is not None else "[left out: something that looks like a password or a code]"


def _kept_private(step: Step, said: Said) -> Said:
    """A read whose content holds a secret is not said, and is shown masked (the full output stays in the details);
    any other sentence that would say one is replaced the same way."""
    lead = _CONTENT_LEADS.get(step.type)
    if lead is not None:
        kind = private_kind(_content(step))
        if kind:
            sentence = f"{lead(step)} holds something that looks like {kind}, so I haven't read it out."
            return Said(shown=f"{sentence}\n{for_screen(_masked(_content(step)))}", spoken=sentence)
    kind = private_kind(said.spoken, content=False) or private_kind(said.shown, content=False)
    if kind:
        sentence = f"The result holds something that looks like {kind}, so I haven't read it out."
        return Said(shown=f"{sentence}\n{for_screen(_masked(said.shown))}", spoken=sentence)
    return said


# Steps that change files, grouped: "Copied 3 files into big."
_GROUP_VERBS = {
    ActionType.FILE_COPY: ("Copied", "destination"),
    ActionType.FILE_MOVE: ("Moved", "destination"),
    ActionType.FILE_DELETE: ("Deleted", "path"),
    ActionType.FILE_WRITE: ("Wrote", "path"),
}


def _group(results: Sequence[ActionResult]) -> str:
    action_type = results[0].action.action_type
    verb, field = _GROUP_VERBS[action_type]
    folders = {folder_of(str(getattr(r.action.parameters, field, "") or r.action.target)) for r in results}
    into = "into" if verb in {"Copied", "Moved"} else "in"
    where = f" {into} {folders.pop()}" if len(folders) == 1 and "" not in folders else ""
    return f"{verb} {plural(len(results), 'file')}{where}."


# A request that asks for an answer: a question, or "tell me", "what", "show me", "read me". Decided from the request's
# own words, in code, the same for every provider.
_ASKS = re.compile(
    r"\?|\b(?:tell me|show me|read (?:me|it|out|aloud)|what(?:'s| is| are| does| did)?|which|how (?:much|many|"
    r"long|big)|who|when|where|is there|are there|does|do i have)\b",
    re.IGNORECASE,
)


def asks_for_an_answer(request: str) -> bool:
    return bool(_ASKS.search(str(request or "")))


def plan_said(results: Sequence[ActionResult], request: str = "") -> Said:
    """One or two sentences about a plan's outcome: the step the person asked for (the one with the widest reach that
    changed something, else the last read), and how many steps it took when more than one changed something. When
    the request asks for an answer and a read gave one, the answer is said first and the change after it ("read the
    note and tell me what it says, then delete it"), within the spoken limit."""
    done = [result for result in results if result.success]
    if not done:
        return Said(shown="Nothing was done.", spoken="Nothing was done.")
    if len(done) == 1:
        return step_said(done[0])
    types = {result.action.action_type for result in done}
    if len(types) == 1 and next(iter(types)) in _GROUP_VERBS:
        sentence = _group(done)
        return Said(shown=sentence, spoken=for_ear(sentence))
    effects = [result for result in done if result.action.action_type not in READ_ONLY_ACTIONS]
    # The step that matters most: the one with the widest reach (a push over a commit, a payment click over a
    # scroll), the last of them when several reach as far; with no effect at all, the last read.
    weightiest = max((result.action.permission_tier for result in effects), default=None)
    chosen = [result for result in effects if result.action.permission_tier == weightiest]
    outcome = step_said(chosen[-1] if chosen else done[-1])
    if len(effects) > 1:
        tail = f" All {len(done)} steps are done."
        outcome = Said(shown=outcome.shown + tail, spoken=for_ear(outcome.spoken + tail))
    answers = [result for result in done if result.action.action_type in _CONTENT_LEADS and result not in effects]
    if effects and answers and asks_for_an_answer(request):
        answer = step_said(answers[-1])
        return Said(
            shown=f"{answer.shown}\n{outcome.shown}",
            spoken=for_ear(f"{answer.spoken} {outcome.spoken}") or answer.spoken,
        )
    return outcome
