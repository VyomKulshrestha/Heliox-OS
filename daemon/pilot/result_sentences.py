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

    def __init__(self, result: ActionResult, request: str = "") -> None:
        self.result = result
        self.request = str(request or "")  # the person's words, for a read that answers what was asked
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
        # An answer that already names what it is ("Battery: 73%") is not named twice.
        said = flat if flat.casefold().startswith(lead.casefold()) else f"{lead}: {flat}"
        return Said(shown=for_screen(text, limit=SHOWN_LIMIT), spoken=said)
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
    """A file's text is the answer. Only part of it read (its first or last lines) is said as that part: the read does
    not count the whole file, so no sentence claims how long the file is."""
    path = step.param("path") or step.target
    name = name_of(path) or "The file"
    is_log = name.casefold().endswith((".log", ".out", ".err"))
    head = getattr(step.p, "head_lines", None)
    tail = getattr(step.p, "tail_lines", None)
    lines = [line for line in step.out.splitlines() if line.strip()]
    if (head or tail) and lines:
        which = "last" if tail else "first"
        part = f"The {which} {plural(len(lines), 'line')} of {name}"
        pick = lines[-1] if tail else lines[0]
        if len(lines) == 1:
            spoken = f"{part}: {words_cut(pick, 16)}"
        else:
            spoken = f"{part}; the {'newest' if tail and is_log else which} is: {words_cut(pick, 16)}"
        return Said(shown=for_screen(step.out), spoken=spoken)
    return _answer(step, name, empty=f"{name} is empty.", last_line=is_log)


def _file_write(step: Step) -> Said:
    path = step.param("path") or step.target
    if getattr(step.p, "append", False):
        lines = re.search(r"Appended (\d+) lines?", step.out)
        count = int(lines.group(1)) if lines else 1
        what = "a line" if count == 1 else f"{count} lines"
        return _said(f"Added {what} to {_at(path)}.")
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
    """How many files and folders, and the newest few files (the listing's "Newest:" line), else the first few."""
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
    newest = re.search(r"^Newest: (.+)$", step.out, re.MULTILINE)
    if newest:
        names = [name.strip() for name in newest.group(1).split(",") if name.strip()]
        listed = f"; the newest {'is' if len(names) == 1 else 'are'} {_names(names)}"
    else:
        listed = f": {_names(files + folders)}"
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


def _service_status(step: Step) -> Said:
    """Whether the service is running, from what systemctl, launchctl or Windows says."""
    name = step.param("name") or "that"
    text = step.out.casefold()
    if re.search(r"active \(running\)|\bstate\s*:\s*4\s+running|\bstatus\s*:\s*running|\brunning\b", text):
        state = "running"
    elif re.search(r"inactive|\bstopped\b|\bdead\b|\bstate\s*:\s*1\s+stopped", text):
        state = "stopped"
    elif re.search(r"\bfailed\b", text):
        state = "failed"
    else:
        return _answer(step, f"The {name} service")
    return Said(shown=for_screen(step.out), spoken=f"The {name} service is {state}.")


def _git_status(step: Step) -> Said:
    """Which branch, and what is waiting: "repo is on main; 1 file is staged and 2 are changed"."""
    branch = step.field("branch") or ""
    staged, unstaged, untracked = (len(step.field(key) or []) for key in ("staged", "unstaged", "untracked"))
    on = f"{_repo(step)} is on {branch}" if branch and branch != "unknown" else _repo(step)
    if not isinstance(step.data, dict):
        return _answer(step, f"The status of {_repo(step)}")
    if not staged + unstaged + untracked:
        return Said(shown=for_screen(step.field("output", "")), spoken=f"{on}, and there's nothing to commit.")
    parts = []
    if staged:
        parts.append(f"{plural(staged, 'file')} {'is' if staged == 1 else 'are'} staged")
    if unstaged:
        parts.append(f"{plural(unstaged, 'file')} {'has' if unstaged == 1 else 'have'} changes not staged")
    if untracked:
        parts.append(f"{plural(untracked, 'new file')} {'is' if untracked == 1 else 'are'} not tracked yet")
    detail = ", ".join(parts[:-1]) + (f" and {parts[-1]}" if len(parts) > 1 else parts[-1])
    return Said(shown=for_screen(step.field("output", "")), spoken=f"{on}; {detail}.")


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
        total = _total_asked(step)
        if total:
            return _said(total)
        sentence = speakable_state(step.out)
        if sentence:
            return Said(shown=sentence, spoken=sentence)
        return _answer(step, lead)

    return build


# How much there is, as against how much is used or left: "how much RAM does it have", "how big is my disk".
_TOTAL_ASKED = re.compile(
    r"\bhow (?:much|many) (?:ram|memory)\b.*\b(?:have|has|got|installed)\b|\b(?:total|installed) (?:ram|memory)\b|"
    r"\bhow (?:big|large) is\b|\b(?:size|capacity) of\b|\bhow much (?:storage|space) does\b.*\bhave\b",
    re.IGNORECASE,
)


def _total_asked(step: Step) -> str:
    """The total first, then how much is used or free, when the question asks how much there is; "" otherwise."""
    if not _TOTAL_ASKED.search(step.request):
        return ""
    memory = re.search(r"Total:\s*([\d.]+) GB\s+Used:\s*([\d.]+) GB", step.out)
    if memory and step.type is ActionType.MEMORY_USAGE:
        return f"This computer has {memory.group(1)} gigabytes of memory; {memory.group(2)} are in use."
    disk = re.search(r"^\s*(\S+?)\s.*?([\d.]+) GB total, ([\d.]+) GB used", step.out, re.MULTILINE)
    if disk and step.type is ActionType.DISK_USAGE:
        device, total, used = disk.groups()
        label = f"Drive {device[0].upper()}" if re.match(r"[A-Za-z]:", device) else "The disk"
        return f"{label} holds {total} gigabytes; {max(0.0, float(total) - float(used)):.0f} are free."
    return ""


def _facts(text: str) -> list[tuple[str, str, str]]:
    """The facts that matter in system information, as (subject, shown line, spoken words), in reading order."""
    facts: list[tuple[str, str, str]] = []
    system = re.search(r"^\s*system: (.+)$", text, re.MULTILINE)
    release = re.search(r"^\s*release: (.+)$", text, re.MULTILINE)
    cores = re.search(r"^\s*(?:Logical cores|Cores): (\d+)", text, re.MULTILINE)
    if system:
        named = f"{system.group(1).strip()} {release.group(1).strip() if release else ''}".strip()
        facts.append(
            ("os", f"System: {named}" + (f", {cores.group(1)} cores" if cores else ""), f"this computer runs {named}")
        )
    cpu = re.search(r"Average usage: ([\d.]+)%", text)
    if cpu:
        value = round(float(cpu.group(1)))
        facts.append(("cpu", f"Processor: {cpu.group(1)}% in use", f"the processor is at {value} percent"))
    memory = re.search(r"Total:\s*([\d.]+) GB\s+Used:\s*([\d.]+) GB \(([\d.]+)%\)", text)
    if memory:
        total, used, share = memory.groups()
        facts.append(
            (
                "memory",
                f"Memory: {used} of {total} GB in use ({share}%)",
                f"{used} of {total} gigabytes of memory are in use",
            )
        )
    for device, total, used in re.findall(r"^\s*(\S+?)\s.*?([\d.]+) GB total, ([\d.]+) GB used", text, re.MULTILINE)[
        :2
    ]:
        free = max(0.0, float(total) - float(used))
        label = f"drive {device[0].upper()}" if re.match(r"[A-Za-z]:", device) else name_of(device) or "the disk"
        facts.append(
            (
                "disk",
                f"{label[:1].upper() + label[1:]}: {free:.0f} GB free of {total} GB",
                f"{label} has {free:.0f} gigabytes free",
            )
        )
    up = [
        name
        for name in re.findall(r"^\s{2}(\S.*?) \(UP\):", text, re.MULTILINE)
        if not re.match(r"^(?:lo\d*|Loopback\b.*|vEthernet\b.*)$", name)
    ]
    if up:
        facts.append(("network", f"Network: {', '.join(up[:3])} connected", f"you're connected over {_names(up, 2)}"))
    battery = re.search(r"Charge:\s*(\d+)%", text)
    if battery:
        facts.append(("battery", f"Battery: {battery.group(1)}%", f"the battery is at {battery.group(1)} percent"))
    up_for = re.search(r"Up for:\s*(\d+) day\(s\),\s*(\d+) hour\(s\),\s*(\d+) minute\(s\)", text)
    if up_for:
        days, hours, minutes = (int(part) for part in up_for.groups())
        parts = [plural(n, unit) for n, unit in ((days, "day"), (hours, "hour"), (minutes, "minute")) if n]
        said = (", ".join(parts[:-1]) + f" and {parts[-1]}" if len(parts) > 1 else parts[0]) if parts else "a minute"
        facts.append(("uptime", f"Up for: {said}", f"this computer has been up for {said}"))
    clock = re.search(r"Current Local Time:\s*(\w+), (\w+ \d{1,2}), \d{4} (\d{1,2}):(\d{2}):\d{2} ([AP]M)", text)
    if clock:
        day, date, hour, minute, half = clock.groups()
        zone = re.search(r"Timezone:\s*(.+)", text)
        facts.append(
            (
                "time",
                f"Time: {int(hour)}:{minute} {half}, {day} {date}" + (f" ({zone.group(1).strip()})" if zone else ""),
                f"it's {int(hour)}:{minute} {half} on {day}",
            )
        )
    host = re.search(r"^\s*hostname:\s*(\S+)", text, re.MULTILINE)
    if host:
        facts.append(("host", f"Name: {host.group(1)}", f"this computer is called {host.group(1)}"))
    return facts


# The subjects a question can be about, by the words a person uses for them.
_SUBJECTS = {
    "cpu": r"\b(?:cpu|processor|load)\b",
    "memory": r"\b(?:memory|ram)\b",
    "disk": r"\b(?:disk|drive|storage|space)\b",
    "network": r"\b(?:network|wi-?fi|internet|ip|connected|connection)\b",
    "battery": r"\bbattery\b",
    "os": r"\b(?:windows|macos|linux|version|operating system|os)\b",
    "uptime": r"\bhow long\b.*\b(?:up|on|running|been)\b|\buptime\b|\blast (?:boot|restart)|\bbooted\b",
    "time": r"\bwhat time\b|\bthe time\b|\btime zone\b|\btimezone\b",
    "host": r"\b(?:hostname|host name|computer'?s name|name of (?:this|my) computer|called)\b",
}
# Asked about what system information does not hold: said plainly before the facts it does hold.
_NOT_IN_SYSTEM_INFO = {
    r"\btemperature|\bhow hot\b|\bthermal": "the temperature",
    r"\b(?:programs?|apps?|processes)\b|\bwhat(?:'s| is| s) running\b": "which programs are running",
    r"\bfans?\b": "the fan speed",
}


def _section_facts(text: str) -> list[tuple[str, str, str]]:
    """Sections this reader does not know by name: each section's first value, as "Section: value"."""
    facts = []
    for section, body in re.findall(r"^=== (.+?) ===\n((?:(?!===).*\n?)*)", text, re.MULTILINE):
        value = next((line.strip() for line in body.splitlines() if line.strip()), "")
        if value:
            value = value.split(":", 1)[1].strip() if ":" in value else value
            facts.append((section.casefold(), f"{section}: {value}", f"{section.lower()}: {value}"))
    return facts


def _system_info(step: Step) -> Said:
    """The two or three facts that matter, the ones the question was about first; a readable summary on screen (the
    full output stays in the details). Asked for something the reading does not hold, it says so first."""
    facts = _facts(step.out) or _section_facts(step.out)
    if not facts:
        return _answer(step, "System information")
    asked = step.request.casefold()
    first = [fact for fact in facts if re.search(_SUBJECTS.get(fact[0], r"(?!x)x"), asked)]
    ordered = first + [fact for fact in facts if fact not in first]
    missing = next((what for pattern, what in _NOT_IN_SYSTEM_INFO.items() if re.search(pattern, asked)), "")
    picked = [fact[2] for fact in (first or ordered)[:3]]
    said = ", ".join(picked[:-1]) + (f", and {picked[-1]}" if len(picked) > 1 else picked[-1])
    said = said[:1].upper() + said[1:] + "."
    if missing:
        said = f"The system information does not include {missing}. {said}"
    shown = "\n".join(fact[1] for fact in ordered)
    if missing:
        shown = f"The system information does not include {missing}.\n{shown}"
    return Said(shown=shown, spoken=said)


def _health_review(step: Step) -> Said:
    first = next((line.strip() for line in step.out.splitlines() if line.strip() and "===" not in line), "")
    return Said(shown=for_screen(step.out), spoken=words_cut(first, 24) or "The health review is on screen.")


_VIRTUAL_ADAPTER = re.compile(r"^(?:lo\d*|Loopback\b.*|vEthernet\b.*|docker\d*|br-\w+|veth\w*|virbr\d*)$", re.I)


def _network_info(step: Step) -> Said:
    """The connections a person uses (not the loopback or a virtual machine's switch) and their IPv4 addresses; the
    first one's address said, every one shown."""
    up = [
        name
        for name in re.findall(r"^\s{2}(\S.*?) \(UP\):", step.out, re.MULTILINE)
        if not _VIRTUAL_ADAPTER.match(name)
    ]
    if not up:
        return _said("No network connection is up.")
    addresses = {name: _ipv4_of(step.out, name) for name in up}
    first = up[0]
    if addresses[first]:
        spoken = f"Your IP address is {addresses[first]}, on {first}."
    else:
        spoken = f"{plural(len(up), 'network connection')} {'is' if len(up) == 1 else 'are'} up: {_names(up)}."
    if len(up) > 1 and addresses[first]:
        spoken += f" {plural(len(up) - 1, 'other connection')} {'is' if len(up) == 2 else 'are'} up too."
    shown = "\n".join(f"{name}: {addresses[name] or 'no IPv4 address'}" for name in up)
    return Said(shown=f"{spoken}\n{shown}", spoken=spoken)


def _ipv4_of(text: str, adapter: str) -> str:
    """The first IPv4 address listed under ``adapter`` in network_info's output, or ""."""
    block = re.search(rf"^\s{{2}}{re.escape(adapter)} \(UP\):\n((?:\s{{4,}}.*\n?)*)", text, re.MULTILINE)
    found = re.search(r"IPv4:\s*(\d{1,3}(?:\.\d{1,3}){3})", block.group(1)) if block else None
    return found.group(1) if found else ""


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


def _program_name(raw: str) -> str:
    name = raw.strip()
    name = re.sub(r"\.(?:exe|app|bin)$", "", name, flags=re.IGNORECASE)
    return name[:1].upper() + name[1:] if name else name


def _process_list(step: Step) -> Said:
    """How many programs are running and the biggest few by memory, by their names ("84 programs are running; the
    biggest are Chrome, Code and Teams"). Several processes of one program count as one."""
    rows = re.findall(
        r"PID=\s*\d+[ \t]+CPU=\s*([\d.]+)%[ \t]+MEM=\s*([\d.]+)%(?:[ \t]+STATUS=\S*)?[ \t]+(?!STATUS=)(\S.*)$",
        step.out,
        re.MULTILINE,
    )
    asked = step.param("name")
    if not rows:
        return _said(f"No program called {asked} is running." if asked else "No matching programs are running.")
    memory: dict[str, float] = {}
    for _cpu, mem, raw in rows:
        name = _program_name(raw)
        if name:
            memory[name] = memory.get(name, 0.0) + float(mem)
    biggest = sorted(memory, key=lambda name: memory[name], reverse=True)[:3]
    cut = re.search(r"of ([\d,]+) processes", step.out)
    processes = int(cut.group(1).replace(",", "")) if cut else len(rows)
    shown = "\n".join(
        f"- {name}: {memory[name]:.1f}% of memory" for name in sorted(memory, key=memory.get, reverse=True)[:20]
    )
    if asked:
        wanted = _program_name(asked).casefold()
        share = sum(memory[name] for name in memory if wanted in name.casefold()) or sum(memory.values())
        spoken = f"{_program_name(asked)} is running" + (
            f", as {plural(processes, 'process', 'processes')}," if processes > 1 else ","
        )
        spoken += f" using {share:.0f}% of memory." if share >= 1 else f" using {share:.1f}% of memory."
        return Said(shown=f"{spoken}\n{shown}", spoken=spoken)
    asked_about = step.request.casefold()
    if biggest and re.search(r"\b(?:memory|ram)\b", asked_about):
        # "What is using the most memory?": the biggest by name and share first.
        then = f"; then {_names(biggest[1:])}" if len(biggest) > 1 else ""
        spoken = f"{biggest[0]} uses the most memory, {memory[biggest[0]]:.1f}%{then}."
        return Said(shown=f"{spoken}\n{shown}", spoken=spoken)
    if re.search(r"\b(?:cpu|processor)\b", asked_about):
        load: dict[str, float] = {}
        for cpu, _mem, raw in rows:
            name = _program_name(raw)
            if name and not re.match(r"^(?:system idle process|idle)$", name, re.IGNORECASE):
                load[name] = load.get(name, 0.0) + float(cpu)
        busiest = sorted(load, key=load.get, reverse=True)[:3]
        if busiest:
            then = f"; then {_names(busiest[1:])}" if len(busiest) > 1 else ""
            spoken = f"{busiest[0]} is using the most processor time, {load[busiest[0]]:.1f}%{then}."
            return Said(shown=f"{spoken}\n{shown}", spoken=spoken)
    # A list cut short counts processes (its total); a whole list counts programs by name.
    count, words = (processes, ("process", "processes")) if cut else (len(memory), ("program", "programs"))
    spoken = f"{plural(count, *words)} {'is' if count == 1 else 'are'} running"
    spoken += f"; the biggest {'is' if len(biggest) == 1 else 'are'} {_names(biggest)}."
    return Said(shown=f"{spoken}\n{shown}", spoken=spoken)


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
    folders = [part for part in value.split(";" if ";" in value else ":") if part] if _folder_list(value) else []
    if not secret and len(folders) > 1:  # PATH and its kind: said by how many folders (paths are for the eye)
        listed = "\n".join(folders)
        return Said(
            shown=f"{name} holds {plural(len(folders), 'folder')}:\n{for_screen(listed)}",
            spoken=f"{name} holds {plural(len(folders), 'folder')}; the list is on screen.",
        )
    if re.search(r"[\\/]", value) and not secret:
        return Said(shown=for_screen(step.out), spoken=f"{name} is set to a path ending in {name_of(value)}.")
    if secret or len(value) > 60 or not value:
        return Said(shown=for_screen(step.out), spoken=f"{name} is set; its value is on screen.")
    return Said(shown=for_screen(step.out), spoken=f"{name} is set to {value}.")


def _count_lines(
    lead: str, noun: str, many: str = "", *, empty: str = "", names: Callable[[str], str] | None = None
) -> Callable[[Step], Said]:
    """A list read, answered: how many, and the first few by name when ``names`` reads a name from a line (never for
    lists whose lines can be private, such as window titles). ``lead`` holds "{n}" for the count ("You have {n}")."""

    def build(step: Step) -> Said:
        data = step.data
        lines = [
            line
            for line in step.out.splitlines()
            if line.strip() and not line.rstrip().endswith(":") and not line.startswith(("[Cut", "Newest: "))
        ]
        if isinstance(data, list):
            count = len(data)
        elif isinstance(data, dict) and isinstance(data.get("items") or data.get("matches"), list):
            count = len(data.get("items") or data.get("matches"))
        else:
            count = len(lines)
        if not step.out or count == 0 or re.match(r"^\(?(?:no |none|empty)", step.out, re.I):
            return _said(empty or lead.replace("{n}", "no " + (many or noun + "s")) + ".")
        shown = step.out if not isinstance(data, (list, dict)) else _listing(data)
        sentence = lead.replace("{n}", plural(count, noun, many))
        if names is not None and not isinstance(data, (list, dict)):
            named = [name for name in (names(line) for line in lines) if name]
            if named:
                sentence += f": {_names(named)}"
        return Said(shown=for_screen(shown), spoken=sentence + ".")

    return build


def _workspace_name(line: str) -> str:
    found = re.match(r"^\s*\d+\.\s+(.+?)(?:\s+\([\d.]+\))?$", line)
    return name_of(found.group(1)) if found else ""


def _tool_name(line: str) -> str:
    found = re.match(r"^\s*(?:[\w.-]+:\s*)?([\w.-]+)\s+-\s", line)
    return found.group(1) if found else ""


def _disk_list(step: Step) -> Said:
    """How many disks: lsblk's rows of TYPE disk, or Get-Disk's numbered rows; never a header, a partition or a
    separator line."""
    lines = [line for line in step.out.splitlines() if line.strip()]
    header = lines[0].split() if lines else []
    if "TYPE" in header:
        column = header.index("TYPE")
        disks = [line for line in lines[1:] if len(line.split()) > column and line.split()[column] == "disk"]
    else:
        disks = [line for line in lines if re.match(r"^\s*\d+\s+\S", line)]
    if not disks:
        return _answer(step, "Disks")
    return Said(shown=for_screen(step.out), spoken=f"This computer has {plural(len(disks), 'disk')}.")


def _user_info(step: Step) -> Said:
    """The account logged in: id's "uid=1000(me)", or the name under whoami's "User Name" table."""
    found = re.search(r"uid=\d+\(([^)]+)\)", step.out)
    name = found.group(1) if found else ""
    if not name:
        table = re.search(r"^User Name\s+SID\s*\n=+\s+=+\s*\n(\S+)", step.out, re.MULTILINE)
        name = table.group(1).rsplit("\\", 1)[-1] if table else ""
    if not name:
        return _said("I couldn't tell which account is logged in; the details are on screen.")
    return _said(f"You're logged in as {name}.")


def _user_list(step: Step) -> Said:
    """The accounts on this computer: one per line (Linux, macOS), or the names between net user's dashed line and its
    closing "The command completed" line (Windows)."""
    lines = [line.strip() for line in step.out.splitlines() if line.strip()]
    dashes = next((i for i, line in enumerate(lines) if re.fullmatch(r"-{5,}", line)), None)
    if dashes is not None:
        names = [
            name for line in lines[dashes + 1 :] if not line.lower().startswith("the command") for name in line.split()
        ]
    else:
        names = [line.split()[0] for line in lines]
    if not names:
        return _said("This computer has no user accounts listed.")
    return Said(
        shown=for_screen(step.out), spoken=f"This computer has {plural(len(names), 'user account')}: {_names(names)}."
    )


def _first_word(line: str) -> str:
    return line.split()[0].strip(",;") if line.split() else ""


def _before_equals(line: str) -> str:
    return line.split("=", 1)[0].strip() if "=" in line else ""


def _before_columns(line: str) -> str:
    """The first column of a table printed with runs of spaces ("VLC media player  VideoLAN.VLC  3.0.21")."""
    return re.split(r"\s{2,}|\t|\s\|\s", line.strip())[0].strip()


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


# Selectors that name one thing a person asks for by name: "the title of example.com".
_NAMED_PARTS = {"title": "title", "h1": "main heading", "h2": "first subheading", "head > title": "title"}


def _scrape(step: Step) -> Said:
    """What was asked for, as an answer: one named thing ("The title of example.com is Example Domain."), the page's
    text as its length and beginning, or how many matches and the first."""
    host = host_of(step.param("url") or step.target)
    selector = step.param("selector")
    if step.data is not None:
        count = len(step.data) if isinstance(step.data, (list, dict)) else 0
        return Said(shown=for_screen(_listing(step.data)), spoken=f"Read {plural(count, 'item')} from {host}.")
    lines = [line.strip() for line in step.out.splitlines() if line.strip() and not line.startswith("[Cut")]
    if not lines:
        return _said(f"That part of {host} is empty.")
    part = _NAMED_PARTS.get(" ".join(str(selector or "").split()).casefold())
    if part and len(lines) == 1 and len(lines[0]) <= SHORT_ANSWER:
        return _said(f"The {part} of {host} is {lines[0]}.")
    if not selector or str(selector).strip().casefold() in {"body", "html", "main", "article"}:
        return _answer(step, f"The page on {host}")
    if len(lines) == 1:
        return Said(shown=for_screen(step.out), spoken=f"On {host}, {selector} reads: {words_cut(lines[0], 16)}")
    return Said(
        shown=for_screen(step.out),
        spoken=f"Found {plural(len(lines), 'match', 'matches')} for {selector} on {host}; the first is: "
        f"{words_cut(lines[0], 10)}",
    )


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


def _relative(path: str, folder: str) -> str:
    """A path as it sits under the folder that was searched ("sub\\notes.txt"); its name when it is elsewhere."""
    full, base = undoubled(path).replace("\\", "/"), undoubled(folder).replace("\\", "/").rstrip("/")
    if base and full.casefold().startswith(base.casefold() + "/"):
        rest = full[len(base) + 1 :]
        return rest.replace("/", "\\") if "\\" in path else rest
    return name_of(path)


def _search_content(step: Step) -> Said:
    phrase = step.param("search_text") or "that"
    searched = step.param("directory", "path") or step.target
    where = name_of(searched) or "that folder"
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
            found_in = str(match.get("file") or match.get("path") or "")
            shown_lines.append(f"- {_relative(found_in, searched)}:{line} {text}".rstrip())
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
    job = words_cut(command[:1].lower() + command[1:], 10).rstrip(".")
    # The person wrote the job about themselves ("tell me", "remind me of my notes"); Heliox says it back to them.
    return re.sub(r"\b(me|my|mine|myself)\b", lambda m: _TO_YOU[m.group(1).lower()], job, flags=re.IGNORECASE)


_TO_YOU = {"me": "you", "my": "your", "mine": "yours", "myself": "yourself"}


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
    if action.action_type is ActionType.FILE_WRITE and getattr(params, "append", False):
        added = " ".join(str(getattr(params, "content", "") or "").split())
        return f"add {_quoted(added)} to {what}" if added else f"add a line to {what}"
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
    """How every sentence names a trigger: by what it watches, never by its id or a made-up name. When another trigger
    watches the same thing, also by what it does ("the trigger that watches notes.txt and is set to tell you")."""
    entry = trigger_entry(trigger)
    if entry.get("trigger_type") == "time_interval":
        return "the trigger that runs on an interval"
    watched, _when, job_words = _trigger_parts(entry)
    if _shares_its_target(entry):
        return f"the trigger that watches {watched} and is set to {job_words}"
    return f"the trigger that watches {watched}"


def _shares_its_target(entry: dict) -> bool:
    """Whether another stored trigger watches the same thing, so a sentence must say which one this is."""
    from pilot.system.triggers import get_engine

    others = [
        other
        for other in get_engine()._triggers.values()
        if other.id != entry.get("id")
        and other.trigger_type.value == entry.get("trigger_type")
        and other.condition == entry.get("condition")
    ]
    return bool(entry.get("id")) and bool(others)


# What a notice-only trigger says when it fires, by its type: "notes.txt changed."
_NOTICES = {
    "file_modified": "{watched} changed.",
    "file_change": "{watched} changed.",
    "file_created": "A new file appeared in {watched}.",
    "file_deleted": "{watched} was deleted.",
    "cpu_threshold": "The processor went above {threshold} percent.",
    "memory_threshold": "Memory use went above {threshold} percent.",
    "disk_threshold": "The disk went above {threshold} percent full.",
    "battery_low": "The battery is below {threshold} percent.",
    "process_started": "{watched} started.",
    "process_stopped": "{watched} stopped.",
    "network_change": "The network changed.",
    "custom_condition": "Your condition holds now.",
}


def _notice(entry: dict) -> str:
    condition = entry.get("condition") if isinstance(entry.get("condition"), dict) else {}
    watched, _when, _job = _trigger_parts(entry)
    default = {"cpu_threshold": 90, "memory_threshold": 90, "disk_threshold": 95, "battery_low": 20}
    kind = str(entry.get("trigger_type") or "")
    threshold = condition.get("threshold", default.get(kind, ""))
    template = _NOTICES.get(kind, "{watched} changed.")
    sentence = template.format(watched=watched, threshold=threshold)
    if not template.startswith("{watched}"):
        sentence = sentence[:1].upper() + sentence[1:]  # a file's or program's own name keeps its case
    if _shares_its_target(entry):
        sentence = sentence[:-1] + " (the trigger set to tell you)."
    return sentence


def trigger_fire_sentence(trigger: Any, outcome: str, *, error: str = "", seconds: int = 0) -> str:
    """What the person is told about one fire. ``outcome``: ran, failed (``error`` is already a plain sentence),
    timeout, fault, waits (a step needs an approval nobody gave), unreadable, unavailable, job_done, job_failed, notice
    (a notice-only trigger: what happened, by the file's name)."""
    entry = trigger_entry(trigger)
    named = trigger_named(entry)
    lead = named[:1].upper() + named[1:] + " fired"
    _watched, _when, job = _trigger_parts(entry)
    if outcome == "notice":
        return _notice(entry)
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


def _what_it_does(entry: dict) -> str:
    """ "one will add "seen" to fired.log when watched.txt changes": what a trigger does, and when."""
    watched, when, job = _trigger_parts(entry)
    if entry.get("trigger_type") == "time_interval":
        return f"one will {job} on an interval"
    return f"one will {job} {when.replace(' it ', f' {watched} ', 1) if ' it ' in when else when}"


def _trigger_list(step: Step) -> Said:
    if not step.data or step.out.startswith("No triggers"):
        return _said("You have no triggers.")
    items = [item for item in (step.data if isinstance(step.data, list) else []) if isinstance(item, dict)]
    if len(items) == 1:
        spoken = f"You have 1 trigger: {_what_it_watches(items[0])}."
    else:
        alike = len({_trigger_parts(item)[0] for item in items}) < len(items)
        parts = [
            _what_it_does(item) if alike else _what_it_watches(item).replace("it ", "one ", 1) for item in items[:3]
        ]
        more = f", and {len(items) - 3:,} more" if len(items) > 3 else ""
        spoken = f"You have {plural(len(items), 'trigger')}: {'; '.join(parts)}{more}."
    lines = []
    for item in items[:50]:
        watched, when, job = _trigger_parts(item)
        lines.append(f"- {watched}, {when}: {job} ({item.get('name') or item.get('id')})")
    return Said(shown=for_screen("\n".join([spoken, *lines])), spoken=spoken)


def _third_person(job: str) -> str:
    """ "add a line to fired.log" -> "adds a line to fired.log", "tell you" -> "tells you"."""
    first, _space, rest = job.partition(" ")
    if not first:
        return "did its job"
    verb = first + ("es" if first.endswith(("s", "sh", "ch", "x")) else "s")
    return f"{verb} {rest}".strip()


def _trigger_delete(step: Step) -> Said:
    # The handler answers with the removed entry (pilot.system.triggers.deleted_record): say what it watched.
    removed = step.data if isinstance(step.data, dict) and step.data.get("deleted") is True else None
    if removed is not None:
        if removed.get("trigger_type") == "time_interval":
            return _said("I've removed the trigger that ran on an interval.")
        watched, _when, job = _trigger_parts(removed)
        watched = removed.get("watched") or watched
        others = int(removed.get("others_watching") or 0)
        if not others:
            return _said(f"I've stopped watching {watched}.")
        still = "The other one still watches" if others == 1 else f"{others} others still watch"
        return _said(f"I've removed the trigger that {_third_person(job)}. {still} {watched}.")
    # An older answer names only the id; what the trigger watched is the step's target when the planner named it.
    target = step.target
    if target and re.search(r"\.\w{1,6}$|[\\/]", target):
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
    ActionType.PACKAGE_SEARCH: _count_lines("I found {n}", "package", names=_before_columns),
    ActionType.SERVICE_START: _fixed(lambda s: f"Started the {s.param('name')} service."),
    ActionType.SERVICE_STOP: _fixed(lambda s: f"Stopped the {s.param('name')} service."),
    ActionType.SERVICE_RESTART: _fixed(lambda s: f"Restarted the {s.param('name')} service."),
    ActionType.SERVICE_ENABLE: _fixed(lambda s: f"The {s.param('name')} service will start on its own now."),
    ActionType.SERVICE_DISABLE: _fixed(lambda s: f"The {s.param('name')} service won't start on its own now."),
    ActionType.SERVICE_STATUS: lambda s: _service_status(s),
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
    ActionType.SCHEDULE_LIST: _count_lines(
        "You have {n}", "scheduled task", empty="You have no scheduled tasks.", names=_first_word
    ),
    ActionType.SCHEDULE_DELETE: _fixed(lambda s: f"Deleted the scheduled task {s.param('name', 'task_id')}."),
    ActionType.ENV_GET: _env_get,
    ActionType.ENV_SET: _fixed(lambda s: f"Set {s.param('name') or 'the variable'}."),
    ActionType.ENV_LIST: _count_lines(
        "{n} set", "environment variable is", "environment variables are", names=_before_equals
    ),
    # windows, sound, screen
    ActionType.WINDOW_LIST: _count_lines("{n} open", "window is", "windows are"),
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
    ActionType.WIFI_LIST: _count_lines("{n} in range", "Wi-Fi network is", "Wi-Fi networks are", names=_before_columns),
    ActionType.WIFI_CONNECT: _fixed(lambda s: f"Connected to {s.param('ssid') or 'the network'}."),
    ActionType.WIFI_DISCONNECT: _fixed("Disconnected from Wi-Fi."),
    ActionType.DISK_LIST: lambda s: _disk_list(s),
    ActionType.DISK_MOUNT: _fixed(
        lambda s: f"Mounted {name_of(s.param('device'))} at {name_of(s.param('mount_point'))}."
    ),
    ActionType.DISK_UNMOUNT: _fixed(lambda s: f"Unmounted {name_of(s.param('mount_point', 'device')) or 'the disk'}."),
    ActionType.USER_LIST: lambda s: _user_list(s),
    ActionType.USER_INFO: lambda s: _user_info(s),
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
    ActionType.BROWSER_EXTRACT_TABLE: _count_lines("The table has {n}", "row"),
    ActionType.BROWSER_EXTRACT_LINKS: _count_lines("The page has {n}", "link"),
    ActionType.BROWSER_EXECUTE_JS: _tool_call(lambda s: "The page script"),
    ActionType.BROWSER_SCREENSHOT: _screenshot,
    ActionType.BROWSER_FILL_FORM: _fixed(
        lambda s: f"Filled in {plural(len(getattr(s.p, 'fields', None) or {}), 'field')}."
    ),
    ActionType.BROWSER_NEW_TAB: _fixed(
        lambda s: f"Opened {host_of(u)} in a new tab." if (u := s.param("url")) else "Opened a new tab."
    ),
    ActionType.BROWSER_CLOSE_TAB: _fixed("Closed the tab."),
    ActionType.BROWSER_LIST_TABS: _count_lines("{n} open", "tab is", "tabs are"),
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
    ActionType.CALENDAR_PARSE: _count_lines("The file holds {n}", "event", names=_before_columns),
    ActionType.CALENDAR_SYNC: _fixed("Synced your calendar."),
    ActionType.CALENDAR_CREATE_EVENT: _calendar_create,
    ActionType.CALENDAR_LIST_EVENTS: _count_lines(
        "You have {n}", "event", empty="You have no events.", names=_before_columns
    ),
    ActionType.CALENDAR_DELETE_EVENT: _fixed("Deleted the event."),
    ActionType.CALENDAR_FETCH: _count_lines(
        "You have {n}", "event", empty="You have no events.", names=_before_columns
    ),
    ActionType.CALENDAR_RECONCILE: lambda s: _answer(s, "Your calendar"),
    # the web, messages and mail
    ActionType.API_REQUEST: _http,
    ActionType.API_GITHUB: lambda s: _http(s, "GitHub"),
    ActionType.API_SEND_EMAIL: _plain(lambda s: _email_send(s).spoken),
    ActionType.API_WEBHOOK: _fixed(lambda s: f"Sent the webhook to {host_of(s.param('url', 'webhook_url'))}."),
    ActionType.API_SLACK: _send("Slack"),
    ActionType.API_DISCORD: _send("Discord"),
    ActionType.API_SCRAPE: _scrape,
    ActionType.EMAIL_FETCH: _count_lines("You have {n}", "new email", empty="You have no new emails."),
    ActionType.EMAIL_SUMMARIZE: lambda s: _answer(s, "Your email"),
    ActionType.EMAIL_REPLY: _fixed("Sent your reply."),
    # workspace and tools
    ActionType.WORKSPACE_INDEX: _fixed(
        lambda s: f"Indexed {name_of(s.param('folder_path') or s.target) or 'the folder'}{_indexed_count(s)}."
    ),
    ActionType.WORKSPACE_SEARCH: _count_lines(
        "{n} in the workspace",
        "file matches",
        "files match",
        empty="Nothing in the workspace matched.",
        names=_workspace_name,
    ),
    ActionType.WASM_CALL: _tool_call(lambda s: f"The {s.param('tool') or 'tool'}"),
    ActionType.PLUGIN_CALL: _tool_call(lambda s: "The plugin"),
    ActionType.SKILL_RUN: _tool_call(lambda s: f"The skill {s.param('skill_id')}".strip()),
    ActionType.MCP_LIST_TOOLS: _count_lines("{n} available", "tool is", "tools are", names=_tool_name),
    ActionType.MCP_CALL_TOOL: _tool_call(lambda s: f"The tool {s.param('mcp_tool')}".strip()),
}


def _indexed_count(step: Step) -> str:
    found = re.search(r"([\d,]+) (?:files|documents|chunks)", step.out)
    return f", {found.group(1)} {found.group(0).split()[-1]}" if found else ""


def step_said(result: ActionResult, *, limit: int = SPOKEN_LIMIT, request: str = "") -> Said:
    """One finished step's sentence, with the rules of pilot.reply_rules applied to both forms (``limit``: the most
    characters said; the contract test asks for an uncut copy to check a cut never falls inside a word). ``request``:
    the person's words, so a read is said for what was asked about."""
    step = Step(result, request)
    if result.skip_reason in {"already_absent", "already_satisfied"} and step.out:
        if step.type is ActionType.FILE_DELETE:
            said = _file_delete(step)
            return Said(shown=said.shown, spoken=for_ear(said.spoken, limit=limit))
        if step.type is ActionType.TRIGGER_START:
            return Said(shown="Your triggers were already on.", spoken="Your triggers were already on.")
        # Not run because its state was already there: the handler's sentence says what was found, a path in it
        # named like every other sentence names one (its name and folder), without the "Already satisfied:" label,
        # ending as a sentence ends.
        found = re.sub(r"^\s*already satisfied:\s*", "", _names_in(step.out), flags=re.IGNORECASE).strip()
        if found and found[-1] not in ".!?":
            found += "."
        if result.skip_reason == "already_satisfied" and found:
            # The check's own words describe the state it found; the person hears that nothing needed doing.
            article = re.match(r"(?:The|A|An|It|Its|This)\b", found)
            found = f"Nothing needed doing: {found[:1].lower() + found[1:] if article else found}"
        found = found[:1].upper() + found[1:]
        return Said(shown=for_screen(found), spoken=for_ear(found, limit=limit) or "It was already done.")
    build = SENTENCES.get(step.type)
    said = build(step) if build is not None else _plain("Done.")(step)
    said = _kept_private(step, said)
    if step.type in READ_ONLY_ACTIONS:
        said = _answer_first(said)
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
    single = content and token and " " not in token and not _folder_list(token) and looks_secret(token)
    return "a password or a code" if single else ""


def _folder_list(token: str) -> bool:
    """Whether a single token is a list of folders, as PATH holds them ("/usr/bin:/bin", "C:\\Windows;C:\\Tools"),
    with or without its "NAME=" in front."""
    value = token.split("=", 1)[1] if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", token) else token
    parts = [part for part in value.split(";" if ";" in value else ":") if part]
    return bool(parts) and all(re.match(r"^(?:[A-Za-z]:)?[\\/]|^~[\\/]|^%[A-Za-z_]+%", part) for part in parts)


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


def _deleted_triggers(results: Sequence[ActionResult]) -> str:
    """Several triggers removed in one plan: "I've removed 2 triggers; both watched watched.txt." """
    watched = []
    for result in results:
        data = parsed_json(result.output or "")
        entry = data if isinstance(data, dict) else {}
        what = str(entry.get("watched") or (_trigger_parts(entry)[0] if entry.get("trigger_type") else ""))
        watched.append(what or name_of(str(result.action.target or "")))
    known = [what for what in watched if what and what != "it"]
    if known and len(set(known)) == 1 and len(known) == len(results):
        return f"I've removed {plural(len(results), 'trigger')}; {'both' if len(results) == 2 else 'all'} watched {known[0]}."
    if known:
        return f"I've removed {plural(len(results), 'trigger')}, watching {_names(sorted(set(known)))}."
    return f"I've removed {plural(len(results), 'trigger')}."


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


_FILE_NAME_END = re.compile(r"[.][A-Za-z0-9]{1,6}$")


def _names_in(text: str) -> str:
    """Each full path in a sentence as its name and folder ("note.txt in archive")."""
    from pilot.reply_rules import _PATH

    def named(found: re.Match[str]) -> str:
        whole = found.group(0)
        trail = re.search(r"[.:,]+$", whole)
        path = whole[: trail.start()] if trail else whole
        # A file by its name and folder; a folder or a mount point by its name.
        said = _at(path) if _FILE_NAME_END.search(path) else name_of(path)
        return said + (trail.group(0) if trail else "")

    return _PATH.sub(named, undoubled(text))


# What a reader prints for itself: section headers ("=== Network ==="), a table rule ("====== ======"), process rows
# ("PID=  376"), Windows line ends, or JSON. None of it is shown as the reply; the detail view and the receipt keep it.
_MACHINE = re.compile(r"^=== .+ ===$|^=+(?:\s+=+)+$|^PID=\s*\d|\r|^\s*[\[{]\s*$", re.MULTILINE)


def machine_output(text: str) -> bool:
    return bool(_MACHINE.search(text or "")) or looks_like_json(text or "")


def _answer_first(said: Said) -> Said:
    """A read's shown reply: its answer sentence on the first line, the same one that is said; after it, the body only
    when a person reads it as it is (a file's text, a page's text, a list of lines), never a reader's own printout."""
    sentence = said.spoken.strip()
    if not sentence:
        return said
    lines = said.shown.split("\n", 1)
    if for_ear(lines[0]) == for_ear(sentence):
        body = lines[1] if len(lines) > 1 else ""
        first = lines[0]
    else:
        body, first = said.shown, sentence
    if body and (
        machine_output(body)
        or for_ear(body) == for_ear(sentence)
        or " ".join(body.split()).casefold() in " ".join(first.split()).casefold()
    ):
        body = ""
    return Said(shown=f"{first}\n{body}" if body.strip() else first, spoken=said.spoken)


def _skipped(result: ActionResult) -> bool:
    """A step not run because its outcome already held (a trigger engine already running, a file already gone)."""
    return result.skip_reason in {"already_satisfied", "already_absent"}


def plan_said(results: Sequence[ActionResult], request: str = "") -> Said:
    """One or two sentences about a plan's outcome: the step the person asked for (the one with the widest reach that
    changed something, else the last read), and how many steps it took when more than one changed something. When
    the request asks for an answer and a read gave one, the answer is said first and the change after it ("read the
    note and tell me what it says, then delete it"), within the spoken limit."""
    done = [result for result in results if result.success]
    if not done:
        return Said(shown="Nothing was done.", spoken="Nothing was done.")
    ran = [result for result in done if not _skipped(result)]
    if ran and len(ran) < len(done):
        # What already held (the engine was already running) is not what the person asked for: the reply is about
        # the steps that ran, and their count leaves the skipped ones out.
        done = ran
    if len(done) == 1:
        return step_said(done[0], request=request)
    types = {result.action.action_type for result in done}
    if len(types) == 1 and next(iter(types)) in _GROUP_VERBS:
        sentence = _group(done)
        return Said(shown=sentence, spoken=for_ear(sentence))
    if types == {ActionType.TRIGGER_DELETE}:
        sentence = _deleted_triggers(done)
        return Said(shown=sentence, spoken=for_ear(sentence))
    effects = [result for result in done if result.action.action_type not in READ_ONLY_ACTIONS]
    # The step that matters most: the one with the widest reach (a push over a commit, a payment click over a
    # scroll), the last of them when several reach as far; with no effect at all, the last read.
    weightiest = max((result.action.permission_tier for result in effects), default=None)
    chosen = [result for result in effects if result.action.permission_tier == weightiest]
    outcome = step_said(chosen[-1] if chosen else done[-1], request=request)
    if len(effects) > 1:
        tail = f" All {len(done)} steps are done."
        outcome = Said(shown=outcome.shown + tail, spoken=for_ear(outcome.spoken + tail))
    answers = [result for result in done if result.action.action_type in _CONTENT_LEADS and result not in effects]
    if effects and answers and asks_for_an_answer(request):
        answer = step_said(answers[-1], request=request)
        return Said(
            shown=f"{answer.shown}\n{outcome.shown}",
            spoken=for_ear(f"{answer.spoken} {outcome.spoken}") or answer.spoken,
        )
    return outcome
