"""Community catch packs: new things for Heliox Live to notice, as data, admitted only through a precision gate.

The built-in detectors cover a handful of slips. Everyone has their own: a command their team keeps getting wrong,
an idiom people mishear, a phrase that means trouble in their field. A catch pack is a JSON file of rules that
anyone can write and share. It is data, never code, because this layer reads everything the person types.

A rule is a regular expression, where it applies (prose, mail, terminal or code), a message and an optional fix,
and its own evidence: at least three sentences it must catch and three look-alikes it must leave alone. Before a
rule is used it has to pass the gate in ``audit``:

- it compiles, is at most 300 characters, and has no nested repetition such as ``(a+)+``;
- it catches every one of its own examples and none of its own look-alikes;
- it fires on none of the ordinary lines in ``catch_corpus.txt`` (about three thousand lines of prose, code and
  terminal text from this project's own documentation and source) that match where it applies;
- it reads that corpus in under 200 ms, so a slow pattern cannot stall typing;
- only terminal rules may be urgent, since urgent catches interrupt even a focus session.

A rule that fails is held back with the reason, and the rest of its pack still loads. Packs live in
``pilot/live/catches/`` (shipped with Heliox) and in ``live_catches/`` in Heliox's data folder (your own). Check a
pack before sharing it with ``python -m pilot.live.catches check my_pack.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from pilot.live.detectors import LOW, NORMAL, URGENT, Correction, TextContext

logger = logging.getLogger("pilot.live.catches")

KIND = "community_catch"
BUILT_IN_DIR = Path(__file__).with_name("catches")
CORPUS = Path(__file__).with_name("catch_corpus.txt")
PLACES = ("prose", "mail", "terminal", "code", "any")
MAX_PATTERN = 300
MIN_EXAMPLES = 3
MAX_CORPUS_MS = 200.0
SCAN_CHARS = 4000  # only the end of a long field: what was just typed
_ID = re.compile(r"[a-z0-9][a-z0-9-]{1,48}")
# A quantified group that itself ends in a quantifier: the classic catastrophic-backtracking shape.
_NESTED = re.compile(r"\((?:[^()\\]|\\.)*[+*}]\)\s*[+*{]")
_FIELD = re.compile(r"\{(\d|[A-Za-z_]\w{0,30})\}")

# Where each place is, for testing a rule's own examples.
_CONTEXTS = {
    "prose": ("notepad.exe", "notes.txt - Notepad"),
    "mail": ("chrome.exe", "Gmail - Compose"),
    "terminal": ("WindowsTerminal.exe", "PowerShell"),
    "code": ("Code.exe", "main.py - Visual Studio Code"),
    "any": ("notepad.exe", "notes.txt - Notepad"),
}


def place_of(ctx: TextContext) -> set[str]:
    """Which places a piece of text is in. Mail is also prose."""
    if ctx.is_terminal:
        return {"terminal", "any"}
    if ctx.is_code_editor:
        return {"code", "any"}
    places = {"prose", "any"}
    if ctx.is_mail_compose:
        places.add("mail")
    return places


@dataclass
class Rule:
    pack: str
    id: str
    where: tuple[str, ...]
    pattern: re.Pattern[str]
    title: str
    message: str
    fix: str = ""
    urgency: str = LOW
    positives: list[str] = field(default_factory=list)
    decoys: list[str] = field(default_factory=list)

    def applies(self, places: set[str]) -> bool:
        return bool(places.intersection(self.where))

    def find(self, text: str, ctx: TextContext) -> Correction | None:
        if not self.applies(place_of(ctx)):
            return None
        tail = text[-SCAN_CHARS:]
        match = self.pattern.search(tail)
        if not match:
            return None
        evidence = match.group(0).strip()
        return Correction(
            kind=KIND,
            key=f"{KIND}:{self.pack}:{self.id}:{evidence.casefold()}",
            title=_fill(self.title, match),
            message=_fill(self.message, match),
            evidence=evidence[:120],
            detail=f'From the catch pack "{self.pack}".',
            fix=_fill(self.fix, match) if self.fix else "",
            fix_kind="replace" if self.fix else "",
            urgency=self.urgency,
            confidence=0.8,
            trailing=not tail[match.end() :].strip(),
        )


def _fill(template: str, match: re.Match[str]) -> str:
    """{0} is the whole match, {1}… the groups, {name} a named group; nothing else is formatted."""

    def value(field_match: re.Match[str]) -> str:
        name = field_match.group(1)
        try:
            got = match.group(int(name)) if name.isdigit() else match.group(name)
        except IndexError:
            return field_match.group(0)
        return (got or "")[:80]

    return _FIELD.sub(value, template)


# ── the gate ──


@dataclass
class RuleReport:
    id: str
    ok: bool
    problems: list[str] = field(default_factory=list)
    corpus_hits: list[str] = field(default_factory=list)
    corpus_ms: float = 0.0


@dataclass
class PackReport:
    name: str
    path: str
    ok: bool
    problems: list[str] = field(default_factory=list)
    rules: list[RuleReport] = field(default_factory=list)

    @property
    def admitted(self) -> int:
        return sum(rule.ok for rule in self.rules)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "ok": self.ok,
            "problems": self.problems,
            "admitted": self.admitted,
            "held_back": [{"id": r.id, "problems": r.problems} for r in self.rules if not r.ok],
        }


def load_corpus(path: Path = CORPUS) -> dict[str, list[str]]:
    """Ordinary lines by place, from a file of "place<TAB>line" rows."""
    corpus: dict[str, list[str]] = {place: [] for place in ("prose", "terminal", "code")}
    try:
        for row in path.read_text(encoding="utf-8").splitlines():
            place, _, line = row.partition("\t")
            if place in corpus and line.strip():
                corpus[place].append(line)
    except OSError:
        logger.warning("The catch corpus is missing; community rules cannot be checked and stay off")
    return corpus


def _context(place: str) -> TextContext:
    app, title = _CONTEXTS[place]
    return TextContext(today=date.today(), app=app, title=title, source="focused_field")


def _build(pack: str, raw: dict[str, Any], problems: list[str]) -> Rule | None:
    rule_id = str(raw.get("id", ""))
    if not _ID.fullmatch(rule_id):
        problems.append("id must be 2 to 50 lower-case letters, digits or dashes")
        return None
    where = raw.get("where", ["prose"])
    where = tuple(where) if isinstance(where, list) else (where,)
    if not where or any(place not in PLACES for place in where):
        problems.append(f"where must list some of {', '.join(PLACES)}")
    pattern_text = raw.get("pattern")
    if not isinstance(pattern_text, str) or not pattern_text:
        problems.append("pattern is missing")
        return None
    if len(pattern_text) > MAX_PATTERN:
        problems.append(f"pattern is longer than {MAX_PATTERN} characters")
    if _NESTED.search(pattern_text):
        problems.append("pattern repeats a repeated group, which can take forever on some text")
    flags = re.IGNORECASE if "i" in str(raw.get("flags", "")) else 0
    try:
        pattern = re.compile(pattern_text, flags)
    except re.error as error:
        problems.append(f"pattern does not compile: {error}")
        return None
    urgency = str(raw.get("urgency", LOW))
    if urgency not in (LOW, NORMAL, URGENT):
        problems.append("urgency must be low, normal or urgent")
    elif urgency == URGENT and set(where) != {"terminal"}:
        problems.append("only terminal rules may be urgent")
    title, message = str(raw.get("title", "")).strip(), str(raw.get("message", "")).strip()
    if not title or not message or len(title) > 80 or len(message) > 200:
        problems.append("title (up to 80 characters) and message (up to 200) are required")
    positives = [str(x) for x in raw.get("positives", []) if str(x).strip()]
    decoys = [str(x) for x in raw.get("decoys", []) if str(x).strip()]
    if len(positives) < MIN_EXAMPLES or len(decoys) < MIN_EXAMPLES:
        problems.append(f"needs at least {MIN_EXAMPLES} positives and {MIN_EXAMPLES} decoys")
    return Rule(pack, rule_id, where, pattern, title, message, str(raw.get("fix", "")), urgency, positives, decoys)


def audit(pack_data: Any, corpus: dict[str, list[str]], path: str = "") -> tuple[PackReport, list[Rule]]:
    """Check every rule in a pack. Returns the report and the rules that passed."""
    report = PackReport(name="", path=path, ok=False)
    if not isinstance(pack_data, dict):
        report.problems.append("a pack is a JSON object with a name and rules")
        return report, []
    name = str(pack_data.get("name", ""))
    report.name = name
    if not _ID.fullmatch(name):
        report.problems.append("name must be 2 to 50 lower-case letters, digits or dashes")
    rules_raw = pack_data.get("rules")
    if not isinstance(rules_raw, list) or not rules_raw:
        report.problems.append("rules must be a non-empty list")
    if report.problems:
        return report, []
    admitted: list[Rule] = []
    seen: set[str] = set()
    for raw in rules_raw[:200]:
        problems: list[str] = []
        rule = _build(name, raw if isinstance(raw, dict) else {}, problems)
        rule_report = RuleReport(id=str((raw or {}).get("id", "?")) if isinstance(raw, dict) else "?", ok=False)
        if rule is not None and rule.id in seen:
            problems.append("the same id twice")
        if rule is not None and not problems:
            seen.add(rule.id)
            places = [place for place in rule.where if place != "any"] or ["prose"]
            for text in rule.positives:
                if not any(rule.find(text, _context(place)) for place in places):
                    problems.append(f"misses its own example: {text!r}")
            for text in rule.decoys:
                if any(rule.find(text, _context(place)) for place in places):
                    problems.append(f"fires on its own look-alike: {text!r}")
            lines = [line for place in _corpus_places(rule) for line in corpus.get(place, [])]
            if not lines:
                problems.append("no ordinary text to check it against")
            started = time.perf_counter()
            for line in lines:
                if rule.pattern.search(line):
                    rule_report.corpus_hits.append(line[:120])
            rule_report.corpus_ms = round((time.perf_counter() - started) * 1000, 2)
            if rule_report.corpus_hits:
                problems.append(
                    f"fires on {len(rule_report.corpus_hits)} ordinary line(s), e.g. {rule_report.corpus_hits[0]!r}"
                )
            if rule_report.corpus_ms > MAX_CORPUS_MS:
                problems.append(f"too slow: {rule_report.corpus_ms:.0f} ms on the ordinary text")
        rule_report.problems = problems
        rule_report.ok = rule is not None and not problems
        report.rules.append(rule_report)
        if rule_report.ok:
            admitted.append(rule)
    report.ok = bool(report.rules) and all(rule.ok for rule in report.rules)
    return report, admitted


def _corpus_places(rule: Rule) -> list[str]:
    places = set()
    for place in rule.where:
        places |= (
            {"prose"} if place in ("prose", "mail") else {place} if place != "any" else {"prose", "code", "terminal"}
        )
    return sorted(places)


# ── the installed packs ──


class CatchBook:
    """Every admitted rule from the shipped packs and the person's own, re-checked when a file changes."""

    def __init__(self, directories: list[Path] | None = None, corpus: dict[str, list[str]] | None = None) -> None:
        self._dirs = directories if directories is not None else [BUILT_IN_DIR]
        self._corpus = corpus
        self._seen: dict[str, tuple[str, PackReport, list[Rule]]] = {}
        self.rules: list[Rule] = []
        self.reports: list[PackReport] = []
        self._checked_at = float("-inf")

    def refresh(self, force: bool = False) -> None:
        if not force and time.monotonic() - self._checked_at < 30:
            return
        self._checked_at = time.monotonic()
        if self._corpus is None:
            self._corpus = load_corpus()
        rules: list[Rule] = []
        reports: list[PackReport] = []
        names: set[str] = set()
        for directory in self._dirs:
            for path in sorted(Path(directory).glob("*.json")) if Path(directory).is_dir() else []:
                try:
                    raw = path.read_bytes()
                except OSError:
                    continue
                digest = hashlib.sha256(raw).hexdigest()
                cached = self._seen.get(str(path))
                if cached and cached[0] == digest:
                    report, admitted = cached[1], cached[2]
                else:
                    try:
                        data = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, ValueError) as error:
                        report, admitted = PackReport("", str(path), False, [f"not valid JSON: {error}"]), []
                    else:
                        report, admitted = audit(data, self._corpus, str(path))
                    self._seen[str(path)] = (digest, report, admitted)
                if report.name in names:
                    report = PackReport(report.name, str(path), False, ["another pack has the same name"])
                    admitted = []
                names.add(report.name)
                reports.append(report)
                rules.extend(admitted)
        self.rules, self.reports = rules, reports

    def find(self, text: str, ctx: TextContext) -> list[Correction]:
        self.refresh()
        found = []
        for rule in self.rules:
            correction = rule.find(text, ctx)
            if correction is not None:
                found.append(correction)
        return found

    def summary(self) -> dict[str, Any]:
        self.refresh()
        return {
            "packs": [report.as_dict() for report in self.reports],
            "rules": len(self.rules),
            "held_back": sum(len(report.rules) - report.admitted for report in self.reports),
        }


_BOOK: CatchBook | None = None


def install(book: CatchBook) -> None:
    """Set the book the live detectors read (the server adds the person's own folder)."""
    global _BOOK
    _BOOK = book


def book() -> CatchBook:
    global _BOOK
    if _BOOK is None:
        _BOOK = CatchBook()
    return _BOOK


def community_catch(text: str, ctx: TextContext) -> list[Correction]:
    if ctx.source != "focused_field":
        return []
    return book().find(text, ctx)


# ── for pack authors ──


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pilot.live.catches", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="run the precision gate on one or more packs")
    check.add_argument("packs", type=Path, nargs="+")
    args = parser.parse_args(argv)
    corpus = load_corpus()
    failed = False
    for path in args.packs:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            print(f"{path}: cannot read it ({error})")
            failed = True
            continue
        report, admitted = audit(data, corpus, str(path))
        print(f"{report.name or path}: {len(admitted)}/{len(report.rules)} rules admitted")
        for problem in report.problems:
            print(f"  pack: {problem}")
        for rule in report.rules:
            mark = "ok  " if rule.ok else "HELD"
            print(f"  {mark} {rule.id} ({rule.corpus_ms} ms on the ordinary text)")
            for problem in rule.problems:
                print(f"       {problem}")
        failed = failed or not report.ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
