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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from pilot.actions import Action, ActionType

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

_URL = re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_QUOTED = re.compile(
    r"(?:(?<=[\s(:,])|^)(?:'([^'\n]{2,200})'|\"([^\"\n]{2,200})\"|“([^”\n]{2,200})”|‘([^’\n]{2,200})’)"
    r"(?=[\s.,;:!?)]|$)"
)
# A folder name may hold spaces ("C:\My Files\a.txt"); the last part may not, so "C:\a.txt to D:\b" is two paths.
_SEGMENT = r"[^\\/\s\"'<>|*?,;:]+"
_WIN_ABS = re.compile(
    rf"(?<![\w])(?:[A-Za-z]:[\\/]|\\\\[\w.$-]+[\\/])(?:{_SEGMENT}(?: {_SEGMENT})*[\\/])*(?:{_SEGMENT})?"
)
_POSIX_SEGMENT = r"[\w.~+@-]+"
_POSIX_ABS = re.compile(rf"(?:^|(?<=[\s(\"'=]))~?/(?:{_POSIX_SEGMENT}(?: {_POSIX_SEGMENT})*/)*(?:{_POSIX_SEGMENT})?")
_RELATIVE = re.compile(r"(?<![\w:\\/.-])[\w.-]+(?:[\\/][\w.-]+)+")
_BARE = re.compile(r"(?<![\w\\/.-])[\w-]+(?:\.[\w-]+)*\.[A-Za-z][A-Za-z0-9]{0,7}(?![\w\\/-])")
_NOT_FILE_NAMES = frozenset({"e.g", "i.e", "etc", "a.m", "p.m", "vs"})


def _norm(path: str) -> str:
    """One spelling of a path for comparing: home expanded, one separator, no trailing separator, case folded.

    Case is folded on every system: that only ever makes two different Linux paths look alike (an accepted plan),
    never a matching pair look different (a refused one)."""
    text = str(path or "").strip().strip("\"'`").rstrip(".,;:!?)")
    if text.startswith("~"):
        text = os.path.expanduser(text)
    text = text.replace("/", "\\")
    unc = text.startswith("\\\\")
    text = re.sub(r"\\+", r"\\", text)
    if unc:
        text = "\\" + text
    if len(text) > 3 or not re.match(r"^[A-Za-z]:\\$", text):
        text = text.rstrip("\\")
    return text.casefold()


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


def named_in(request: str) -> Named:
    """The targets, quoted text and verbs of one request."""
    text = str(request or "")
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
            found = match.group(0).strip()
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
    return raw.replace("\\\\", "\\").replace("/", "\\").casefold()


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


def first_refusal(mismatch: Mismatch, request: str) -> str:
    """What the person is told when the second plan for a request still does not match it."""
    return (
        f"{REFUSED_FIRST}: the plan I made would {mismatch.step}, which is not what you asked. Nothing was changed. "
        "Please say it again, naming the file or folder."
    )


def repair_refusal(mismatch: Mismatch) -> str:
    """What the person is told when a repair round's plan reaches outside the failed step."""
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
