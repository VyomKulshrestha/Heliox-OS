"""Does a plan do what the request asked? Checked in code before anything runs.

A model can plan the wrong request: in one session it once planned an earlier turn's copy for "Delete
note_backup.txt", and a repair round once copied to the previous turn's destination. A prompt alone does not stop
that on every model, so the planner holds each plan against the request's own words here:

- the things the request names: absolute and relative paths, file names with an extension, quoted text;
- the kind of change its verbs ask for: delete, copy, move, write, send.

A plan with a step that changes something (a file written, deleted, copied or moved, a message sent) is refused when

1. a changing step touches only things the request never named (when it names any): ``off_target``;
2. an absolute path or file name the request names appears nowhere in the plan: ``not_reflected``;
3. its kind of change contradicts every verb of the request (asked to delete, plans a copy): ``wrong_kind``;
4. a message step sends none of the request's quoted text: ``not_reflected``.

Words like "it", "that one" or "again" with no named target take their targets from the turn before (the previous
request and the conversation lines after it), so "delete it" after a copy may delete that copy and nothing else. A
request that names nothing ("tidy my downloads") is not checked for targets; a verb the lists below do not know (an
open verb like "organise", or a request in another language) is not checked for kind. Read-only plans, questions
and code or shell steps are not checked here: code and shell always stop for approval.

A repair round is held tighter (``check_repair``): it may retry or adjust the failed plan's own steps, and it may not
write, delete, copy or move onto a path that neither the request nor the failed plan named, nor onto a path an
earlier request of the session named.

Nothing here grants anything: a plan that passes still goes through every check and approval.
"""

from __future__ import annotations

import json
import ntpath
import os
import re
import sys
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from pilot.actions import READ_ONLY_ACTIONS, Action, ActionType

# The kind of change each changing step makes.
CHANGE_KIND: dict[ActionType, str] = {
    ActionType.FILE_WRITE: "write",
    ActionType.FILE_DELETE: "delete",
    ActionType.FILE_MOVE: "move",
    ActionType.FILE_COPY: "copy",
    ActionType.API_SEND_EMAIL: "send",
    ActionType.API_SLACK: "send",
    ActionType.API_DISCORD: "send",
    ActionType.API_WEBHOOK: "send",
}

# Verbs, in the request's own words, and the kinds of change each allows a plan to make.
_VERBS: dict[str, tuple[str, ...]] = {
    "delete": ("delete", "remove", "erase", "trash", "discard", "purge", "wipe", "get rid of", "empty"),
    "copy": ("copy", "duplicate", "back up", "backup", "clone", "mirror"),
    "move": ("move", "rename", "relocate"),
    "write": (
        "write",
        "create",
        "make",
        "save",
        "append",
        "add",
        "edit",
        "change",
        "replace",
        "insert",
        "prepend",
        "overwrite",
        "generate",
        "draft",
        "store",
        "put",
        "type",
    ),
    "send": ("send", "email", "e-mail", "mail", "post", "message", "notify", "share", "forward", "reply", "dm"),
}
ALLOWED: dict[str, frozenset[str]] = {
    "delete": frozenset({"delete", "move"}),  # deleting may be a move to a trash folder
    "copy": frozenset({"copy", "write"}),  # a copy may be a read and a write
    "move": frozenset({"move", "copy", "delete"}),  # a move may be a copy and a delete
    "write": frozenset({"write", "copy", "move"}),
    "send": frozenset({"send"}),
}
# Verbs whose kind of change cannot be told from the word: a request with one is not checked for kind.
_OPEN_VERBS = (
    "organise",
    "organize",
    "tidy",
    "clean",
    "sort",
    "sync",
    "restore",
    "undo",
    "revert",
    "fix",
    "update",
    "merge",
    "split",
    "convert",
    "compress",
    "zip",
    "unzip",
    "extract",
    "archive",
    "download",
    "install",
    "set up",
    "setup",
    "handle",
    "process",
    "prepare",
    "do",
    "run",
    "redo",
    "repeat",
    "finish",
    "continue",
)
_REFERS_BACK = re.compile(
    r"\b(it|its|that|this|those|these|them|the same|again|too|as well|that one|this one|the file|the folder)\b",
    re.IGNORECASE,
)

# The person-facing sentences start with one of these, so the server knows the refusal is final (no repair loop).
REFUSED_FIRST = "I stopped before doing anything"
REFUSED_REPAIR = "I did not try again"
REFUSAL_PREFIXES = (REFUSED_FIRST, REFUSED_REPAIR)


def is_refusal(error: str | None) -> bool:
    """Whether a plan error is this module's refusal (final: never re-planned by the server)."""
    return bool(error) and str(error).startswith(REFUSAL_PREFIXES)


# ── what a request names ─────────────────────────────────────────────


def _mark_ranges() -> str:
    """Every combining mark (Unicode categories Mn, Mc, Me) as character-class ranges. Python's \\w leaves them
    out, so without them a Hindi file name like रिपोर्ट.txt would break apart at its vowel signs."""
    ranges: list[tuple[int, int]] = []
    for code in range(sys.maxunicode + 1):
        if unicodedata.category(chr(code)).startswith("M"):
            if ranges and ranges[-1][1] == code - 1:
                ranges[-1] = (ranges[-1][0], code)
            else:
                ranges.append((code, code))
    return "".join(f"\\U{a:08x}-\\U{b:08x}" if a != b else f"\\U{a:08x}" for a, b in ranges)


_W = r"\w" + _mark_ranges()  # a letter, digit or underscore of any script, or a combining mark
# Scripts written without spaces between words: a path or file name there can run straight into the next word
# ("C:\\w\\報告書.txtを削除して").
_NO_SPACE = (
    "\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af"
    "\uf900-\ufaff\uff00-\uffef"
)

_URL = re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.IGNORECASE)
_EMAIL = re.compile(rf"(?<![{_W}.+-])[{_W}.+-]+@[{_W}-]+(?:\.[{_W}-]+)+")
_QUOTED = re.compile(
    r"(?:(?<=[\s(:,])|^)(?:'([^'\n]{2,200})'|\"([^\"\n]{2,200})\"|“([^”\n]{2,200})”|‘([^’\n]{2,200})’)"
    r"(?=[\s.,;:!?)]|$)"
)
# A folder name may hold spaces ("C:\My Files\a.txt"); the last part may not, so "C:\a.txt to D:\b" is two paths.
_SEGMENT = r"[^\\/\s\"'<>|*?,;:]+"
_WIN_ABS = re.compile(
    rf"(?<![\w])(?:[A-Za-z]:[\\/]|\\\\[\w.$-]+[\\/])(?:{_SEGMENT}(?: {_SEGMENT})*[\\/])*(?:{_SEGMENT})?"
)
_POSIX_SEGMENT = rf"[{_W}.~+@-]+"
_POSIX_ABS = re.compile(rf"(?:^|(?<=[\s(\"'=]))~?/(?:{_POSIX_SEGMENT}(?: {_POSIX_SEGMENT})*/)*(?:{_POSIX_SEGMENT})?")
_RELATIVE = re.compile(rf"(?<![{_W}:\\/.-])[{_W}.-]+(?:[\\/][{_W}.-]+)+")
_BARE = re.compile(
    rf"(?<![{_W}\\/.-])[{_W}-]+(?:\.[{_W}-]+)*\.[A-Za-z][A-Za-z0-9]{{0,7}}(?=$|[{_NO_SPACE}]|[^{_W}\\/-])"
)
_NOT_FILE_NAMES = frozenset({"e.g", "i.e", "etc", "a.m", "p.m", "vs"})


def _norm(path: str) -> str:
    """One spelling of a path for comparing: home expanded, one separator, no trailing separator, case folded.

    Case is folded on every system: that only ever makes two different Linux paths look alike (an accepted plan),
    never a matching pair look different (a refused one)."""
    # One Unicode spelling (NFC): "café" typed as e plus a combining accent matches the one-character é.
    text = unicodedata.normalize("NFC", str(path or "")).strip().strip("\"'`").rstrip(".,;:!?)")
    if text.startswith("~"):
        text = os.path.expanduser(text)
    text = text.replace("/", "\\")
    unc = text.startswith("\\\\")
    text = re.sub(r"\\+", r"\\", text)
    if unc:
        text = "\\" + text
    if len(text) > 3 or not re.match(r"^[A-Za-z]:\\$", text):
        text = text.rstrip("\\")
    return unicodedata.normalize("NFC", text.casefold())


def _base(path: str) -> str:
    return ntpath.basename(_norm(path))


@dataclass(frozen=True)
class Named:
    """What a request names, compared in ``_norm`` spelling."""

    absolute: tuple[str, ...] = ()
    relative: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    quoted: tuple[str, ...] = ()
    kinds: frozenset[str] = frozenset()
    open_verb: bool = False
    refers_back: bool = False

    @property
    def targets(self) -> bool:
        return bool(self.absolute or self.relative or self.names)

    def union(self, other: Named) -> Named:
        return Named(
            absolute=tuple(dict.fromkeys(self.absolute + other.absolute)),
            relative=tuple(dict.fromkeys(self.relative + other.relative)),
            names=tuple(dict.fromkeys(self.names + other.names)),
            quoted=self.quoted,
            kinds=self.kinds,
            open_verb=self.open_verb,
            refers_back=self.refers_back,
        )


def _verbs(text: str) -> tuple[frozenset[str], bool]:
    words = " " + re.sub(r"[^a-z0-9-]+", " ", text.casefold()) + " "
    kinds = frozenset(
        kind for kind, verbs in _VERBS.items() if any(f" {verb} " in words or f" {verb}s " in words for verb in verbs)
    )
    return kinds, any(f" {verb} " in words for verb in _OPEN_VERBS)


_GLUED_EXTENSION = re.compile(rf"^(.*\.[A-Za-z][A-Za-z0-9]{{0,7}})[{_NO_SPACE}]")


def _glued_tail(path: str) -> str:
    """A path whose last part runs into words of a script written without spaces: cut after its extension, or, with
    no extension to cut at, keep only the folder it is in. Never more than the request said: a looser target can
    only let a plan through, never refuse one because the words around the path were not understood."""
    folder, _, last = path.replace("/", "\\").rpartition("\\")
    if not folder or not re.search(rf"[{_NO_SPACE}]", last):
        return path
    glued = _GLUED_EXTENSION.match(last)
    if glued:
        return f"{folder}\\{glued.group(1)}"
    if re.search(r"\.[A-Za-z][A-Za-z0-9]{0,7}$", last):
        return path
    return folder


def named_in(request: str) -> Named:
    """The targets, quoted text and verbs of one request. The verbs are English words only: a request in another
    language gets the target checks and no kind check, so it is never refused for words that were not understood."""
    text = unicodedata.normalize("NFC", str(request or ""))
    quoted = [next(group for group in match.groups() if group) for match in _QUOTED.finditer(text)]
    # A quoted path is one path, spaces and all.
    rest = _QUOTED.sub(" ", text)
    rest = _URL.sub(" ", rest)
    rest = _EMAIL.sub(" ", rest)
    absolute: list[str] = []
    for chunk in [q for q in quoted if _WIN_ABS.fullmatch(q.strip()) or q.strip().startswith(("/", "~"))]:
        absolute.append(_norm(chunk))
    for pattern in (_WIN_ABS, _POSIX_ABS):
        for match in pattern.finditer(rest):
            found = _glued_tail(match.group(0).strip())
            if len(found) > 1:
                absolute.append(_norm(found))
        rest = pattern.sub(" ", rest)
    relative = [_norm(match.group(0)) for match in _RELATIVE.finditer(rest)]
    rest = _RELATIVE.sub(" ", rest)
    names = [
        name
        for name in (_norm(match.group(0)) for match in _BARE.finditer(rest))
        if name not in _NOT_FILE_NAMES and not re.fullmatch(r"[\d.]+", name)
    ]
    for chunk in quoted:  # a quoted file name or relative path also names a target
        inner = chunk.strip()
        if _BARE.fullmatch(inner):
            names.append(_norm(inner))
        elif _RELATIVE.fullmatch(inner):
            relative.append(_norm(inner))
    # Verbs from the words alone: a folder called "archive" or a file called note_backup.txt is not a verb.
    kinds, open_verb = _verbs(_BARE.sub(" ", rest))
    targets = bool(absolute or relative or names)
    return Named(
        absolute=tuple(dict.fromkeys(a for a in absolute if a)),
        relative=tuple(dict.fromkeys(relative)),
        names=tuple(dict.fromkeys(names)),
        quoted=tuple(dict.fromkeys(" ".join(q.split()).casefold().rstrip(".!?") for q in quoted if q.strip())),
        kinds=kinds,
        open_verb=open_verb,
        # Only a request that names no target of its own takes one from the turn before.
        refers_back=bool(_REFERS_BACK.search(rest)) and not targets,
    )


# What a reply that picks from a list Heliox just offered may hold besides the pick: "delete both of them", "yes, the
# first one", "just the second". A word outside these means the reply is about something else ("delete all my notes").
_PICK_WORDS = frozenset(
    [
        "delete",
        "remove",
        "drop",
        "stop",
        "cancel",
        "get",
        "rid",
        "of",
        "both",
        "all",
        "every",
        "each",
        "them",
        "they",
        "those",
        "these",
        "the",
        "one",
        "ones",
        "of",
        "and",
        "just",
        "only",
        "please",
        "yes",
        "ok",
        "okay",
        "sure",
        "keep",
        "neither",
        "none",
        "no",
        "nor",
        "either",
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "last",
        "two",
        "three",
        "trigger",
        "triggers",
        "watcher",
        "watchers",
        "too",
        "as",
        "well",
    ]
)
_ORDINALS = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}


def picked_from(request: str, count: int) -> list[int] | None:
    """Which of ``count`` things Heliox just listed (in the order it listed them) a short reply picks: their indexes,
    [] for "neither" or "none", None when the reply is not a pick (it names something, or says more than a pick).
    "Both" picks two only when two were offered; "all of them" every one; "the first", "the second", "the last" one."""
    words = re.findall(r"[a-z]+", unicodedata.normalize("NFKC", str(request or "")).casefold())
    if not words or count <= 0 or any(word not in _PICK_WORDS for word in words):
        return None
    named = named_in(request)
    if named.absolute or named.relative or named.names or named.quoted:
        return None
    present = set(words)
    if present & {"neither", "none", "nor"} or (
        present <= {"no", "keep", "both", "them", "all", "of", "ok", "okay"} and present & {"no", "keep"}
    ):
        return []
    if "both" in present:
        return [0, 1] if count == 2 else None
    if present & {"all", "every", "each"}:
        return list(range(count))
    picked = {_ORDINALS[word] for word in present if word in _ORDINALS}
    if "last" in present:
        picked.add(count - 1)
    if not picked or any(index >= count for index in picked):
        return None
    return sorted(picked)


def resolved(named: Named, earlier: Sequence[str]) -> Named:
    """``named`` with the targets "it", "that one" or "again" point at, for a request that names none itself: every
    target named in ``earlier`` (the conversation just before the request and the last request with its reply)."""
    if not named.refers_back:
        return named
    for text in earlier:
        named = named.union(named_in(text))
    return named


# ── what a plan does ─────────────────────────────────────────────────


@dataclass(frozen=True)
class Step:
    index: int
    kind: str
    source: str = ""  # copy and move: what is read (and, for a move, removed)
    target: str = ""  # what is written or deleted; for copy and move the destination
    text: str = ""  # the whole step, normalised, for "is this named thing anywhere in it"


def _params(action: Action) -> dict:
    try:
        return action.parameters.model_dump(mode="json")
    except Exception:  # noqa: BLE001
        return {}


def _step_text(action: Action) -> str:
    raw = json.dumps({"target": action.target, "parameters": _params(action)}, ensure_ascii=False)
    return unicodedata.normalize(
        "NFC", unicodedata.normalize("NFC", raw).replace("\\\\", "\\").replace("/", "\\").casefold()
    )


def changing_steps(actions: Iterable[Action]) -> list[Step]:
    steps = []
    for index, action in enumerate(actions):
        kind = CHANGE_KIND.get(action.action_type)
        if kind is None:
            continue
        params = action.parameters
        path = str(getattr(params, "path", "") or "") or (action.target if kind != "send" else "")
        destination = str(getattr(params, "destination", "") or "")
        if kind in ("copy", "move"):
            steps.append(Step(index, kind, source=_norm(path), target=_norm(destination), text=_step_text(action)))
        elif kind == "send":
            steps.append(Step(index, kind, text=_step_text(action)))
        else:
            steps.append(Step(index, kind, target=_norm(path), text=_step_text(action)))
    return steps


def _inside(path: str, folder: str) -> bool:
    """``path`` is ``folder`` or below it, compared part by part."""
    if not (path and folder):
        return False
    return path == folder or path.startswith(folder if folder.endswith("\\") else folder + "\\")


def _relates(path: str, named: Named) -> bool:
    if not path:
        return False
    if any(_inside(path, folder) for folder in named.absolute):
        return True
    wrapped = "\\" + path + "\\"
    if any(f"\\{rel}\\" in wrapped for rel in named.relative):
        return True
    parts = path.split("\\")
    return any(name in parts for name in named.names)


def _reflected(thing: str, plan_text: str) -> bool:
    return thing in plan_text or (ntpath.basename(thing) or thing) in plan_text


def _short(path: str, original: dict[str, str]) -> str:
    return ntpath.basename(original.get(path, path)) or original.get(path, path)


def describe(step: Step, action: Action | None = None) -> str:
    """The step in a few words, file names only: "copy a.txt to b.txt", "delete note.txt", "send a message"."""

    def name(value: str) -> str:
        return ntpath.basename(value.replace("/", "\\")) or value

    if action is not None:
        params = action.parameters
        path = str(getattr(params, "path", "") or action.target or "")
        destination = str(getattr(params, "destination", "") or "")
    else:
        path, destination = step.source or step.target, step.target
    if step.kind == "copy":
        return f"copy {name(path)} to {name(destination)}"
    if step.kind == "move":
        return f"move {name(path)} to {name(destination)}"
    if step.kind == "delete":
        return f"delete {name(path)}"
    if step.kind == "write":
        return f"write {name(path)}"
    return "send a message"


@dataclass(frozen=True)
class Mismatch:
    """Why a plan does not do what was asked: ``for_model`` goes back to the planner, ``sentence`` to the person."""

    rule: str  # off_target | not_reflected | wrong_kind | repair_new_path | repair_earlier_turn
    for_model: str
    step: str = ""
    details: tuple[str, ...] = field(default_factory=tuple)


# A quoted value that reads as code (a CSS selector, a file pattern, a column name with a dot, a regular expression):
# the step that uses it must carry it as it is. Plain words in quotes ("files that mention 'quarterly budget'") may be
# what a read looks for inside text, so they are held only on plans that change something (rule 4).
# A dot counts only inside a word (".price", "td.amount", "a.txt"): a sentence's full stop is not code.
_CODE_CHARACTERS = re.compile(r"[#\[\]*>:=$^\\/]|\.\w")
_LINE_COUNT = re.compile(r"\b(?:last|first|top|final) (\d{1,6}) lines?\b", re.IGNORECASE)
_BACKTICKED = re.compile(r"`([^`\n]{1,200})`")


def given_values(request: str) -> tuple[str, ...]:
    """The explicit values a request gives that a step must use as they are: quoted or backticked text that reads as
    code (".price-table td.amount", "*.log"), and a line count ("the last 5 lines" gives "5")."""
    text = unicodedata.normalize("NFC", str(request or ""))
    quoted = [next(group for group in match.groups() if group) for match in _QUOTED.finditer(text)]
    quoted += [match.group(1) for match in _BACKTICKED.finditer(text)]
    values = [
        " ".join(value.split())
        for value in quoted
        if _CODE_CHARACTERS.search(value.strip().rstrip(".!?:"))
        and len(value.split()) <= 6
        and not _URL.fullmatch(value.strip())
    ]
    values += [match.group(1) for match in _LINE_COUNT.finditer(text)]
    return tuple(dict.fromkeys(value for value in values if value))


def _carries(value: str, actions: Sequence[Action]) -> bool:
    # The step text is JSON: a quote inside a value is written \" there, and the request has it bare.
    flat = " ".join(" ".join(_step_text(action).replace('\\"', '"').split()) for action in actions)
    wanted = " ".join(unicodedata.normalize("NFC", value).replace("/", "\\").casefold().split())
    if value.isdigit():
        return re.search(rf"(?<![\d.]){value}(?![\d.])", flat) is not None
    return wanted in flat


def dropped_value(request: str, actions: Sequence[Action]) -> Mismatch | None:
    """A value the request gives that no step uses, or None. Checked on every plan that has steps, reads too: a
    scrape that drops the selector the person typed returns the whole page as if it were the answer."""
    if not actions:
        return None
    for value in given_values(request):
        if not _carries(value, actions):
            return Mismatch(
                "value_dropped",
                f"The current request gives {value!r}, but no step uses it. Keep every value the request gives "
                "(a selector, a pattern, a count) in its step, as it is.",
                step=f"leave out {value!r}, which you asked for",
            )
    return None


def check_plan(
    request: str,
    actions: Sequence[Action],
    *,
    earlier: Sequence[str] = (),
    done: Sequence[Action] = (),
) -> Mismatch | None:
    """The first way ``actions`` contradicts ``request``, or None. ``earlier``: the texts "it" or "that one" may
    point back to (see ``resolved``). ``done``: for a repair, the failed plan, whose steps already used some of the
    named things (a repair redoes only the failed copy of three)."""
    dropped = dropped_value(request, actions)
    if dropped is not None:
        return dropped
    steps = changing_steps(actions)
    if not steps:
        return None
    named = named_in(request)
    scope = resolved(named, earlier)
    by_index = {step.index: step for step in steps}

    # 3. the kind of change contradicts every verb of the request.
    if named.kinds and not named.open_verb:
        allowed = frozenset().union(*(ALLOWED[kind] for kind in named.kinds))
        for step in steps:
            if step.kind not in allowed:
                said = " or ".join(sorted(named.kinds))
                return Mismatch(
                    "wrong_kind",
                    f"Step {step.index + 1} would {describe(step, actions[step.index])}, but the current request asks "
                    f"to {said}. Plan exactly what the current request asks.",
                    step=describe(step, actions[step.index]),
                )

    # 1. a changing step touches only things the request never named.
    if scope.targets:
        for step in steps:
            if step.kind == "send":
                continue
            if step.kind in ("copy", "move"):
                ok = _relates(step.target, scope) or _relates(step.source, scope)
            else:
                ok = _relates(step.target, scope)
            if not ok:
                return Mismatch(
                    "off_target",
                    f"Step {step.index + 1} would {describe(step, actions[step.index])}, which the current request "
                    "does not name. Plan only the files and folders the current request names.",
                    step=describe(step, actions[step.index]),
                )

    # 2. a path or file name the request names appears nowhere in the plan.
    plan_text = " ".join(_step_text(action) for action in [*actions, *done])
    for thing in named.absolute + named.names:
        if not _reflected(thing, plan_text):
            first = by_index[min(by_index)]
            return Mismatch(
                "not_reflected",
                f"The current request names {ntpath.basename(thing) or thing}, but no step of the plan uses it. "
                "Plan the current request.",
                step=describe(first, actions[first.index]),
            )

    # 4. a message step sends none of the request's quoted text.
    if named.quoted:
        for step in steps:
            if step.kind == "send" and not any(quote in " ".join(step.text.split()) for quote in named.quoted):
                return Mismatch(
                    "not_reflected",
                    f"Step {step.index + 1} sends a message without the text the current request quotes. Send "
                    "exactly the quoted text.",
                    step=describe(step),
                )
    return None


def check_repair(
    request: str,
    actions: Sequence[Action],
    *,
    failed: Sequence[Action],
    earlier_requests: Sequence[str] = (),
    earlier: Sequence[str] = (),
) -> Mismatch | None:
    """A repair round's plan: the request's own checks, then the failed plan's scope. ``failed``: the plan whose
    step failed; ``earlier_requests``: the session's earlier requests, any order."""
    escalated = repair_escalation(actions, failed)
    if escalated is not None:
        return escalated
    problem = check_plan(request, actions, earlier=earlier, done=failed)
    if problem is not None:
        return problem
    named = resolved(named_in(request), earlier)
    failed_paths: set[str] = set()
    for step in changing_steps(failed):
        failed_paths.update(path for path in (step.source, step.target) if path)
    for action in failed:  # read steps of the failed plan count too (a repair may write what it read)
        path = str(getattr(action.parameters, "path", "") or "")
        if path:
            failed_paths.add(_norm(path))
    before = Named()
    for text in earlier_requests:
        before = before.union(named_in(text))
    for step in changing_steps(actions):
        written = [step.target] + ([step.source] if step.kind == "move" else [])
        for path in [p for p in written if p]:
            if _relates(path, named) or path in failed_paths:
                continue
            what = describe(step, actions[step.index])
            if _relates(path, before):
                return Mismatch(
                    "repair_earlier_turn",
                    f"Step {step.index + 1} would {what}, a file an earlier request named, not this one.",
                    step=what,
                )
            return Mismatch(
                "repair_new_path",
                f"Step {step.index + 1} would {what}, which neither the request nor the failed plan named.",
                step=what,
            )
    return None


# Actions that run anything: a repair may not turn to them to get around a typed step that failed.
RUNS_ANYTHING = frozenset(
    {
        ActionType.SHELL_COMMAND,
        ActionType.SHELL_SCRIPT,
        ActionType.PTY_EXEC,
        ActionType.CODE_EXECUTE,
        ActionType.CODE_GENERATE_AND_RUN,
        ActionType.SSH_COMMAND,
        ActionType.SSH_SCRIPT,
        ActionType.BROWSER_EXECUTE_JS,
    }
)
# Switches that make a step reach further than it did: turning one on in a repair is a different change.
WIDENING_SWITCHES = ("force", "recursive", "overwrite", "allow_private_network", "persistent")


def _plain_type(kind: ActionType) -> str:
    return kind.value.replace("_", " ")


def repair_escalation(actions: Sequence[Action], failed: Sequence[Action]) -> Mismatch | None:
    """A repair that reaches past the failed plan's own actions, or None. It may retry or adjust a step with the same
    action, or add a read. It may not run a shell command, a script or code the failed plan did not (after a rejected
    push the repair once planned "git pull" as a shell command), start an action of another kind that changes
    something (a pull, a reset), or turn on a switch such as "force" that the failed step had off. Holds for every
    action: the push is only the case that was seen."""
    failed_kinds = {action.action_type for action in failed}
    failed_name = next(
        (_plain_type(action.action_type) for action in failed if action.action_type not in READ_ONLY_ACTIONS),
        _plain_type(failed[0].action_type) if failed else "step",
    )
    for index, action in enumerate(actions):
        kind = action.action_type
        if kind in failed_kinds:
            before = [a for a in failed if a.action_type == kind]
            for switch in WIDENING_SWITCHES:
                now_on = bool(getattr(action.parameters, switch, False))
                was_on = any(bool(getattr(a.parameters, switch, False)) for a in before)
                if now_on and not was_on:
                    return Mismatch(
                        "repair_escalation",
                        f"Step {index + 1} turns on {switch!r}, which the failed step did not use.",
                        step=f"run the {failed_name} again with {switch!r} turned on",
                    )
            continue
        if kind in RUNS_ANYTHING:
            return Mismatch(
                "repair_escalation",
                f"Step {index + 1} runs a {_plain_type(kind)} to get around the {failed_name} that failed.",
                step=f"run a {_plain_type(kind)} instead of the {failed_name} that failed",
            )
        if kind not in READ_ONLY_ACTIONS:
            return Mismatch(
                "repair_new_kind",
                f"Step {index + 1} starts a {_plain_type(kind)}, a different kind of change from the {failed_name}.",
                step=f"start a {_plain_type(kind)}, which you did not ask for",
            )
    return None


def first_refusal(mismatch: Mismatch, request: str) -> str:
    """What the person is told when the second plan for a request still does not match it."""
    return (
        f"{REFUSED_FIRST}: the plan I made would {mismatch.step}, which is not what you asked. Nothing was changed. "
        "Please say it again, naming the file or folder."
    )


def repair_refusal(mismatch: Mismatch) -> str:
    """What the person is told when a repair round's plan reaches outside the failed step."""
    if mismatch.rule in ("repair_escalation", "repair_new_kind", "value_dropped"):
        return f"{REFUSED_REPAIR}: the retry would {mismatch.step}."
    return f"{REFUSED_REPAIR}: the retry would {mismatch.step}, which this request did not name."


# ── a read done with code (finding: "show me the last lines of this log" planned as code_execute) ─────────────

_SLICE = re.compile(r"\[\s*(-\s*\d+)\s*:\s*\]|\[\s*:\s*(\d+)\s*\]")
_SHELL_TAIL = re.compile(
    r"\b(?:tail|head)\s+(?:-n\s*)?-?(\d+)\b|Get-Content\b[^|;]*-(?:Tail|Last|Head|First|TotalCount)\s+(\d+)"
    r"|Select-Object\s+-(?:Last|First)\s+(\d+)",
    re.IGNORECASE,
)
_CODE_DOES_MORE = re.compile(r"\b(open|write|remove|unlink|rmtree|subprocess|system|requests|urllib|socket)\b")


def read_done_by_code(actions: Sequence[Action]) -> str:
    """A hint when a step uses code or a shell only to show the first or last lines of a file, which file_read does
    with "head_lines" or "tail_lines" and no approval; "" otherwise."""
    for index, action in enumerate(actions):
        t = action.action_type
        if t == ActionType.CODE_EXECUTE:
            code = str(getattr(action.parameters, "code", "") or "")
            follows_read = index > 0 and actions[index - 1].action_type == ActionType.FILE_READ
            if (
                follows_read
                and "PREV_OUTPUT" in code
                and ("splitlines" in code or "split('\\n')" in code or 'split("\\n")' in code)
                and _SLICE.search(code)
                and not _CODE_DOES_MORE.search(code)
            ):
                return (
                    f"Step {index + 1} runs code only to pick lines of the file step {index} read. Use one file_read "
                    'with "tail_lines": N (the last N lines) or "head_lines": N (the first N lines) instead: it needs '
                    "no approval."
                )
        elif t in (ActionType.SHELL_COMMAND, ActionType.SHELL_SCRIPT):
            params = action.parameters
            line = " ".join(
                [str(getattr(params, "command", "") or ""), *[str(a) for a in getattr(params, "args", []) or []]]
                + [str(getattr(params, "script", "") or "")]
            )
            if _SHELL_TAIL.search(line) and not re.search(r"[>]|\bSet-Content\b|\bOut-File\b|\brm\b|\bdel\b", line):
                return (
                    f"Step {index + 1} runs a shell command only to show lines of a file. Use one file_read with "
                    '"tail_lines": N (the last N lines) or "head_lines": N (the first N lines) instead: it needs no '
                    "approval."
                )
    return ""
