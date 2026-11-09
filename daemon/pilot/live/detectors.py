"""Specific, local mistake detectors for the live layer.

Each detector reads a piece of text the person is writing or saying and returns ``Correction`` objects that
name the exact evidence, say what is wrong in one sentence, and carry a fix when there is one. They are plain
rules over the text: no model call, no network, nothing persisted. A detector that is not sure stays quiet;
a false correction costs more trust than a missed one.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

URGENT, NORMAL, LOW = "urgent", "normal", "low"

KINDS = (
    "risky_command",
    "secret_in_text",
    "weekday_date",
    "arithmetic",
    "profile_fact",
    "error_on_screen",
    "missing_attachment",
    "repeated_word",
    "personal_fact",  # pilot.live.grounded: a local model checks the text against the person's own data
    "community_catch",  # pilot.live.catches: rules from catch packs that passed the precision gate
    "stuck",  # pilot.live.stuck: offers help when the person seems stuck (not a detector over one text)
    "commitment",  # pilot.live.commitments: a promise with a deadline, with an offer to set a reminder
)

KIND_LABELS = {
    "risky_command": "Risky commands before you run them",
    "secret_in_text": "Keys and passwords pasted into messages",
    "weekday_date": "A weekday that doesn't match its date",
    "arithmetic": "Arithmetic slips",
    "profile_fact": "Names that contradict what you told me",
    "error_on_screen": "Errors on screen, with the fix",
    "missing_attachment": "Mentioning an attachment",
    "repeated_word": "Doubled words",
    "personal_fact": "Things that contradict your calendar, to-dos or contacts",
    "community_catch": "Catches from catch packs",
    "stuck": "Offers to help when you seem stuck",
    "commitment": "Promises you make, with a reminder",
}
# Kinds found by a model rather than a rule: still there while their quoted evidence is in the text.
MODEL_KINDS = frozenset({"personal_fact"})


@dataclass(frozen=True)
class Correction:
    kind: str
    key: str
    title: str
    message: str
    evidence: str
    detail: str = ""
    fix: str = ""
    fix_kind: str = ""  # "replace" (corrected text), "command" (safer command) or "ask" (a request for Heliox)
    urgency: str = NORMAL
    confidence: float = 0.9
    # The evidence ends where the text ends: the person may still be typing it, so it needs a longer pause.
    trailing: bool = False


@dataclass
class TextContext:
    """What the detectors know besides the text itself."""

    today: date
    app: str = ""
    title: str = ""
    source: str = "focused_field"  # "focused_field", "screen_ocr" or "speech"
    people: dict[str, str] = field(default_factory=dict)  # relation -> the one name the person gave

    @property
    def where(self) -> str:
        return f"{self.app} {self.title}".casefold()

    @property
    def is_terminal(self) -> bool:
        return any(word in self.where for word in _TERMINAL_WORDS)

    @property
    def is_code_editor(self) -> bool:
        return any(word in self.where for word in _EDITOR_WORDS)

    @property
    def is_mail_compose(self) -> bool:
        where = self.where
        mail_app = any(word in where for word in ("gmail", "outlook", "thunderbird", "mail", "proton"))
        composing = any(word in where for word in ("compose", "new message", "draft", "reply", "re:", "fwd", "message"))
        return mail_app and composing


_TERMINAL_WORDS = (
    "terminal", "powershell", "cmd.exe", "command prompt", "windowsterminal", "bash", "zsh", "wsl", "ubuntu",
    "iterm", "alacritty", "wezterm", "conemu", "warp", "mintty", "git bash",
)  # fmt: skip
_EDITOR_WORDS = (
    "visual studio code", "code.exe", "cursor", "pycharm", "intellij", "webstorm", "sublime", "notepad++",
    "vim", "neovim", "emacs", "zed", "android studio", "rider", "clion",
)  # fmt: skip


def _at_end(text: str, end: int) -> bool:
    return not text[end:].strip()


def _short(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ── Weekday that doesn't match its date ───────────────────────────────────────────────────────────────────────────

_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_DAY_RE = r"(?P<wd>mon(?:day)?|tue(?:s(?:day)?)?|wed(?:nesday)?|thu(?:r(?:s(?:day)?)?)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)"
_MONTH_RE = (
    r"(?P<m>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_DATE_PATTERNS = (
    # "Tuesday, 14 November 2026", "Tue 14th Nov", "Tuesday the 14th of November"
    re.compile(
        rf"\b{_DAY_RE}\.?,?\s+(?:the\s+)?(?P<d>\d{{1,2}})(?:st|nd|rd|th)?(?:\s+of)?\s+{_MONTH_RE}\b\.?"
        rf"(?:,?\s+(?P<y>\d{{4}}))?",
        re.IGNORECASE,
    ),
    # "Tuesday, November 14, 2026", "Tue Nov 14"
    re.compile(
        rf"\b{_DAY_RE}\.?,?\s+{_MONTH_RE}\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(?P<y>\d{{4}}))?",
        re.IGNORECASE,
    ),
)


def _month_number(token: str) -> int:
    token = token[:3].casefold()
    return [m[:3].casefold() for m in calendar.month_name[1:]].index(token) + 1


def _weekday_number(token: str) -> int:
    token = token[:3].casefold()
    return [d[:3] for d in _DAYS].index(token)


def _resolve_year(day: int, month: int, today: date) -> int | None:
    """The year a yearless date most likely means: this year, or next year once it is well past."""
    for year in (today.year, today.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if candidate >= today - timedelta(days=60) or year == today.year + 1:
            return year
    return None


def weekday_date(text: str, ctx: TextContext) -> list[Correction]:
    found: list[Correction] = []
    seen: set[tuple[int, int]] = set()
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            if match.span() in seen:
                continue
            seen.add(match.span())
            stated = _weekday_number(match["wd"])
            day, month = int(match["d"]), _month_number(match["m"])
            month_name = calendar.month_name[month]
            year = int(match["y"]) if match["y"] else _resolve_year(day, month, ctx.today)
            evidence = match.group(0).strip()
            if year is None or day > calendar.monthrange(year, month)[1] or day < 1:
                if 1 <= day <= 31:
                    found.append(
                        Correction(
                            kind="weekday_date",
                            key=f"weekday_date:{evidence.casefold()}",
                            title=f"There's no {day} {month_name}",
                            message=f"{month_name} doesn't have a {day}{_ordinal(day)}.",
                            evidence=evidence,
                            urgency=NORMAL,
                        )
                    )
                continue
            actual = date(year, month, day)
            if actual.weekday() == stated:
                continue
            actual_name, stated_name = _DAYS[actual.weekday()].title(), _DAYS[stated].title()
            # The nearest date that really is the weekday they wrote, in case the date is the slip.
            offset = (stated - actual.weekday()) % 7
            forward = actual + timedelta(days=offset)
            backward = forward - timedelta(days=7)
            nearest = forward if offset <= 3 else backward
            corrected = re.sub(re.escape(match["wd"]), _match_case(actual_name, match["wd"]), evidence, count=1)
            found.append(
                Correction(
                    kind="weekday_date",
                    key=f"weekday_date:{evidence.casefold()}",
                    title=f"{day} {month_name} is a {actual_name}",
                    message=f"{day} {month_name} is a {actual_name}, not a {stated_name}.",
                    evidence=evidence,
                    detail=f"If you meant the {stated_name}, that's {nearest.day} {calendar.month_name[nearest.month]}.",
                    fix=corrected,
                    fix_kind="replace",
                    urgency=NORMAL,
                    confidence=0.97,
                    trailing=_at_end(text, match.end()),
                )
            )
    return found


def _ordinal(n: int) -> str:
    return "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _match_case(word: str, like: str) -> str:
    if like.isupper():
        return word.upper()
    if len(like) <= 4:  # an abbreviation like "Tue" gets "Sat"
        return word[: len(like.rstrip("."))]
    return word if like[:1].isupper() else word.casefold()


# ── Arithmetic slips ──────────────────────────────────────────────────────────────────────────────────────────────

_NUM = r"-?\d+(?:\.\d+)?"
_ARITH = re.compile(
    rf"(?<![\w.,/:-])(?P<a>{_NUM})\s*(?P<op>[+\-*/×÷]|\s[xX]\s)\s*(?P<b>{_NUM})\s*=\s*(?P<c>{_NUM})(?![\w.%/:])"
)
_PERCENT = re.compile(
    rf"(?<![\w.])(?P<p>{_NUM})\s*%\s*of\s*(?P<base>{_NUM})\s*(?:=|is|equals)\s*(?P<c>{_NUM})(?![\w.%])",
    re.IGNORECASE,
)


def _decimals(token: str) -> int:
    return len(token.split(".")[1]) if "." in token else 0


def _fmt(value: float, decimals: int) -> str:
    if decimals == 0 and abs(value - round(value)) < 1e-9:
        return str(round(value))
    return f"{value:.{max(decimals, 2)}f}".rstrip("0").rstrip(".")


def arithmetic(text: str, ctx: TextContext) -> list[Correction]:
    if ctx.is_code_editor or ctx.is_terminal:
        return []  # code is full of things that look like sums and are not claims
    found: list[Correction] = []
    for match in _ARITH.finditer(text):
        a, b, c = float(match["a"]), float(match["b"]), float(match["c"])
        op = match["op"].strip().casefold()
        if op in ("+",):
            value, symbol = a + b, "+"
        elif op in ("-",):
            value, symbol = a - b, "−"
        elif op in ("*", "×", "x"):
            value, symbol = a * b, "×"
        else:
            if b == 0:
                continue
            value, symbol = a / b, "÷"
        places = _decimals(match["c"])
        if round(value, places) == round(c, places):
            continue
        if symbol == "÷" and abs(value - c) < 10 ** (-places) * 0.51:
            continue
        shown = _fmt(value, places)
        evidence = match.group(0).strip()
        found.append(
            Correction(
                kind="arithmetic",
                key=f"arithmetic:{evidence}",
                title=f"That comes to {shown}",
                message=f"{_fmt(a, _decimals(match['a']))} {symbol} {_fmt(b, _decimals(match['b']))} is {shown}, not {match['c']}.",
                evidence=evidence,
                fix=evidence[: match.start("c") - match.start()] + shown,
                fix_kind="replace",
                urgency=NORMAL,
                confidence=0.99,
                trailing=_at_end(text, match.end()),
            )
        )
    for match in _PERCENT.finditer(text):
        p, base, c = float(match["p"]), float(match["base"]), float(match["c"])
        value = p * base / 100
        places = _decimals(match["c"])
        if round(value, places) == round(c, places):
            continue
        shown = _fmt(value, places)
        evidence = match.group(0).strip()
        found.append(
            Correction(
                kind="arithmetic",
                key=f"arithmetic:{evidence}",
                title=f"That comes to {shown}",
                message=f"{match['p']}% of {match['base']} is {shown}, not {match['c']}.",
                evidence=evidence,
                fix=evidence[: match.start("c") - match.start()] + shown,
                fix_kind="replace",
                urgency=NORMAL,
                confidence=0.99,
                trailing=_at_end(text, match.end()),
            )
        )
    return found


# ── Risky commands, caught before Enter ──────────────────────────────────────────────────────────────────────────

_RISKY: tuple[tuple[str, re.Pattern[str], str, str, str], ...] = (
    (
        "rm_root",
        re.compile(r"\brm\s+(?:-[a-z]*r[a-z]*f[a-z]*|-[a-z]*f[a-z]*r[a-z]*)\s+(?:--no-preserve-root\s+)?(?:/|~|/\*|\*)(?=\s|$)", re.I),
        "That deletes everything there",
        "That rm -rf targets the whole drive or your home folder, and there's no undo.",
        "",
    ),
    (
        "force_push",
        re.compile(r"\bgit\s+push\b[^\n]*?(?:\s--force(?!-with-lease)\b|\s-f\b)", re.I),
        "A plain force-push",
        "A plain force-push can overwrite your teammates' commits. --force-with-lease refuses if the remote moved.",
        "force_with_lease",
    ),
    (
        "reset_hard",
        re.compile(r"\bgit\s+reset\s+--hard\b", re.I),
        "That throws away your changes",
        "git reset --hard throws away uncommitted changes. git stash would keep them.",
        "",
    ),
    (
        "clean_fdx",
        re.compile(r"\bgit\s+clean\s+-[a-z]*f[a-z]*d[a-z]*x|\bgit\s+clean\s+-[a-z]*x[a-z]*f", re.I),
        "That deletes ignored files too",
        "git clean -x also deletes ignored files, like your .env and build caches.",
        "",
    ),
    (
        "drop_table",
        re.compile(r"\bdrop\s+(?:table|database|schema)\s+(?!if\s+exists\s+\w+_tmp)\S+", re.I),
        "That drops it for good",
        "DROP removes the table and its data permanently. Make sure you have a backup.",
        "",
    ),
    (
        "delete_all_rows",
        re.compile(r"\bdelete\s+from\s+(?P<t>[\w.\"`\[\]]+)\s*;", re.I),
        "That deletes every row",
        "That DELETE has no WHERE clause, so it removes every row in the table.",
        "",
    ),
    (
        "update_all_rows",
        re.compile(r"\bupdate\s+[\w.\"`\[\]]+\s+set\s+(?:(?!\bwhere\b)[^;])*;", re.I),
        "That updates every row",
        "That UPDATE has no WHERE clause, so it changes every row in the table.",
        "",
    ),
    (
        "chmod_777",
        re.compile(r"\bchmod\s+(?:-R\s+)?777\b", re.I),
        "That makes it writable by anyone",
        "chmod 777 lets every user on the machine change those files.",
        "",
    ),
    (
        "pipe_to_shell",
        re.compile(r"\b(?:curl|wget)\b[^\n|]*\|\s*(?:sudo\s+)?(?:ba|z)?sh\b|\biwr\b[^\n|]*\|\s*iex\b", re.I),
        "That runs a script you haven't read",
        "Piping a download straight into a shell runs it before you can read it.",
        "",
    ),
    (
        "format_drive",
        re.compile(r"\bformat\s+[a-z]:(?=\s|$)|\bmkfs(?:\.\w+)?\s+/dev/|\bdd\s+if=\S+\s+of=/dev/(?:sd|nvme|disk)", re.I),
        "That wipes a whole drive",
        "That command erases an entire drive.",
        "",
    ),
    (
        "remove_drive_root",
        re.compile(
            r"\b(?:remove-item|rm|ri|del|rd|rmdir)\b[^\n]*?(?:-recurse|/s)\b[^\n]*?\s(?:[a-z]:\\?|\$home|~|\$env:userprofile)(?=\s|$)",
            re.I,
        ),
        "That deletes a whole drive or your home folder",
        "That recursive delete targets a drive root or your home folder.",
        "",
    ),
)  # fmt: skip


def _current_command_lines(text: str, ctx: TextContext) -> list[str]:
    """Only what is about to run: the last lines of a terminal, or a short single-line field."""
    lines = [line for line in text.splitlines() if line.strip()]
    if ctx.is_terminal:
        return lines[-2:]
    if len(text) <= 400:
        return lines[-3:]
    return []


def risky_command(text: str, ctx: TextContext) -> list[Correction]:
    if ctx.source == "speech":
        return []
    found: list[Correction] = []
    for line in _current_command_lines(text, ctx):
        for name, pattern, title, message, fix_name in _RISKY:
            match = pattern.search(line)
            if not match:
                continue
            # The command starts where the rule matched; anything before it is the shell prompt.
            command = line[match.start() :].strip()
            fix = ""
            if fix_name == "force_with_lease":
                fix = re.sub(r"\s--force(?!-with-lease)\b|\s-f\b", " --force-with-lease", command, count=1)
            found.append(
                Correction(
                    kind="risky_command",
                    key=f"risky_command:{name}:{command}",
                    title=title,
                    message=message,
                    evidence=_short(command),
                    fix=fix,
                    fix_kind="command" if fix else "",
                    urgency=URGENT,
                    confidence=0.95,
                )
            )
            break
    return found


# ── Keys and passwords pasted where other people will read them ─────────────────────────────────────────────────

_SECRETS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("an OpenAI API key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("an Anthropic API key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("a GitHub token", re.compile(r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{40,}")),
    ("an AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("a Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("a Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("a private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("a password", re.compile(r"\b(?:password|passwd|pwd)\s*[:=]\s*\S{6,}", re.I)),
)
_PASSWORD_MANAGERS = ("1password", "bitwarden", "keepass", "lastpass", "dashlane", "keeper", "enpass", "nordpass")


def _redact(secret: str) -> str:
    return secret[:4] + "…" + secret[-4:] if len(secret) > 12 else "…"


def secret_in_text(text: str, ctx: TextContext) -> list[Correction]:
    where = ctx.where
    if ctx.is_terminal or ctx.is_code_editor or any(name in where for name in _PASSWORD_MANAGERS):
        return []  # an .env file or a password manager is where a key belongs
    for label, pattern in _SECRETS:
        match = pattern.search(text)
        if match:
            secret = match.group(0)
            return [
                Correction(
                    kind="secret_in_text",
                    key=f"secret_in_text:{label}:{_redact(secret)}",
                    title=f"That looks like {label}",
                    message=f"That looks like {label}. Anyone who reads this can use it.",
                    evidence=_redact(secret),
                    detail="If it has been sent already, rotate it where it was issued.",
                    urgency=URGENT,
                    confidence=0.9,
                )
            ]
    return []


# ── Contradicting what the person told Heliox ─────────────────────────────────────────────────────────────────────

# Relations most people have one of at a time; "my friend Sam" says nothing about "my friend Alex".
SINGULAR_RELATIONS = (
    "manager", "boss", "supervisor", "wife", "husband", "spouse", "fiance", "fiancee", "girlfriend", "boyfriend",
    "mum", "mom", "mother", "dad", "father", "landlord", "therapist", "dentist", "doctor", "accountant", "lawyer",
    "mentor", "co-founder", "cofounder",
)  # fmt: skip
_NAME = r"(?P<name>[A-Z][a-z]+(?:\s[A-Z][a-z]+)?)"


def profile_fact(text: str, ctx: TextContext) -> list[Correction]:
    found: list[Correction] = []
    for relation, known in ctx.people.items():
        if relation not in SINGULAR_RELATIONS:
            continue
        pattern = re.compile(rf"\bmy\s+{re.escape(relation)}(?:'s name is|\s+is|,)?\s+{_NAME}\b", re.IGNORECASE)
        for match in pattern.finditer(text):
            said = match["name"]
            first = said.split()[0]
            if not first[:1].isupper() or first.casefold() == known.split()[0].casefold():
                continue
            if first.casefold() in {"and", "said", "says", "will", "was", "is", "has", "had", "can", "the"}:
                continue
            found.append(
                Correction(
                    kind="profile_fact",
                    key=f"profile_fact:{relation}:{first.casefold()}",
                    title=f"Your {relation} is {known}",
                    message=f"You told me your {relation} is {known}, not {first}.",
                    evidence=_short(match.group(0)),
                    detail=f'If that has changed, say "my {relation} is {first}" and I\'ll update it.',
                    fix=match.group(0).replace(said, known),
                    fix_kind="replace",
                    urgency=NORMAL,
                    confidence=0.85,
                    trailing=_at_end(text, match.end()),
                )
            )
    return found


# ── Errors on screen, with the fix ───────────────────────────────────────────────────────────────────────────────

_PIP_NAMES = {
    "cv2": "opencv-python", "sklearn": "scikit-learn", "PIL": "pillow", "yaml": "pyyaml", "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv", "jwt": "pyjwt", "serial": "pyserial", "Crypto": "pycryptodome", "fitz": "pymupdf",
    "dateutil": "python-dateutil", "magic": "python-magic", "win32api": "pywin32", "docx": "python-docx",
}  # fmt: skip

_ERRORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("module_missing", re.compile(r"ModuleNotFoundError: No module named '(?P<mod>[\w.]+)'")),
    ("cannot_find_module", re.compile(r"Error: Cannot find module '(?P<mod>[^']+)'")),
    ("port_in_use", re.compile(r"(?:EADDRINUSE|address already in use)[^\n]*?:(?P<port>\d{2,5})", re.I)),
    ("not_recognized", re.compile(r"'(?P<cmd>[\w.-]+)' is not recognized as an internal or external command", re.I)),
    ("command_not_found", re.compile(r"(?:command not found: (?P<cmd>[\w.-]+)|(?P<cmd2>[\w.-]+): command not found)")),
    ("npm_missing_script", re.compile(r'npm (?:ERR!|error) Missing script: "(?P<script>[\w:.-]+)"')),
    ("not_git_repo", re.compile(r"fatal: not a git repository")),
    (
        "python_syntax",
        re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+)[\s\S]{0,200}?SyntaxError: (?P<msg>[^\n]+)'),
    ),
)


def error_on_screen(text: str, ctx: TextContext) -> list[Correction]:
    if ctx.source == "speech":
        return []
    tail = text[-4000:]
    for name, pattern in _ERRORS:
        matches = list(pattern.finditer(tail))
        if not matches:
            continue
        match = matches[-1]
        evidence = _short(match.group(0))
        if name == "module_missing":
            module = match["mod"].split(".")[0]
            package = _PIP_NAMES.get(module, module)
            return [
                Correction(
                    kind="error_on_screen",
                    key=f"error_on_screen:{evidence}",
                    title=f"{module} isn't installed",
                    message=f"Python can't find {module}. Installing {package} should fix it.",
                    evidence=evidence,
                    fix=f"pip install {package}",
                    fix_kind="command",
                )
            ]
        if name == "cannot_find_module":
            module = match["mod"]
            if module.startswith((".", "/")):
                return [
                    Correction(
                        kind="error_on_screen",
                        key=f"error_on_screen:{evidence}",
                        title="A local import path is wrong",
                        message=f"Node can't find {module}. Check the path and the file's extension.",
                        evidence=evidence,
                    )
                ]
            return [
                Correction(
                    kind="error_on_screen",
                    key=f"error_on_screen:{evidence}",
                    title=f"{module} isn't installed",
                    message=f"Node can't find {module}. Installing it should fix that.",
                    evidence=evidence,
                    fix=f"npm install {module}",
                    fix_kind="command",
                )
            ]
        if name == "port_in_use":
            port = match["port"]
            return [
                Correction(
                    kind="error_on_screen",
                    key=f"error_on_screen:{evidence}",
                    title=f"Port {port} is already taken",
                    message=f"Something else is already using port {port}.",
                    evidence=evidence,
                    detail="Stop that program, or start this one on another port.",
                    fix=f"Show me which program is using port {port}",
                    fix_kind="ask",
                )
            ]
        if name in ("not_recognized", "command_not_found"):
            cmd = match.groupdict().get("cmd") or match.groupdict().get("cmd2") or ""
            return [
                Correction(
                    kind="error_on_screen",
                    key=f"error_on_screen:{evidence}",
                    title=f"{cmd} isn't installed or isn't on PATH",
                    message=f"The shell can't find {cmd}. It isn't installed, or its folder isn't on PATH.",
                    evidence=evidence,
                    fix=f"Check whether {cmd} is installed on this computer",
                    fix_kind="ask",
                )
            ]
        if name == "npm_missing_script":
            return [
                Correction(
                    kind="error_on_screen",
                    key=f"error_on_screen:{evidence}",
                    title=f'There\'s no "{match["script"]}" script',
                    message=f'package.json has no "{match["script"]}" script. npm run lists the ones it has.',
                    evidence=evidence,
                    fix="npm run",
                    fix_kind="command",
                )
            ]
        if name == "not_git_repo":
            return [
                Correction(
                    kind="error_on_screen",
                    key=f"error_on_screen:{evidence}",
                    title="This folder isn't a git repository",
                    message="You're running git outside a repository. cd into the project, or run git init.",
                    evidence=evidence,
                )
            ]
        if name == "python_syntax":
            file_name = re.split(r"[\\/]", match["file"])[-1]
            return [
                Correction(
                    kind="error_on_screen",
                    key=f"error_on_screen:{evidence}",
                    title=f"Syntax error in {file_name}, line {match['line']}",
                    message=f"Python stopped at line {match['line']} of {file_name}: {match['msg'].strip()}.",
                    evidence=evidence,
                )
            ]
    return []


# ── Mentioning an attachment ────────────────────────────────────────────────────────────────────────────────────

_ATTACHMENT = re.compile(
    r"\b(?:i'?ve attached|i have attached|attached (?:is|are|you'll find)|please find attached|see attached|enclosed is)\b",
    re.I,
)


def missing_attachment(text: str, ctx: TextContext) -> list[Correction]:
    if ctx.source != "focused_field" or not ctx.is_mail_compose:
        return []
    match = _ATTACHMENT.search(text)
    if not match:
        return []
    return [
        Correction(
            kind="missing_attachment",
            key=f"missing_attachment:{ctx.title.casefold()}",
            title="You mention an attachment",
            message="You mention an attachment. Make sure it's attached before you send.",
            evidence=_short(match.group(0)),
            urgency=LOW,
            confidence=0.6,
        )
    ]


# ── Doubled words ─────────────────────────────────────────────────────────────────────────────────────────────────

_REPEAT = re.compile(r"\b(?P<w>[A-Za-z]{2,})\s+(?P=w)\b", re.IGNORECASE)
_LEGIT_REPEATS = {"had", "that", "is", "do", "bye", "no", "so", "very", "ha", "go", "knock", "tut", "chop", "yada"}


def repeated_word(text: str, ctx: TextContext) -> list[Correction]:
    if ctx.source != "focused_field" or ctx.is_terminal or ctx.is_code_editor:
        return []
    for match in _REPEAT.finditer(text):
        word = match["w"]
        if word.casefold() in _LEGIT_REPEATS:
            continue
        return [
            Correction(
                kind="repeated_word",
                key=f"repeated_word:{match.group(0).casefold()}",
                title="A doubled word",
                message=f'"{word} {word}" has the word twice.',
                evidence=match.group(0),
                fix=word,
                fix_kind="replace",
                urgency=LOW,
                confidence=0.8,
                trailing=_at_end(text, match.end()),
            )
        ]
    return []


def community_catch(text: str, ctx: TextContext) -> list[Correction]:
    from pilot.live.catches import community_catch as find  # imports this module, so it loads late

    return find(text, ctx)


DETECTORS = {
    "community_catch": community_catch,
    "risky_command": risky_command,
    "secret_in_text": secret_in_text,
    "weekday_date": weekday_date,
    "arithmetic": arithmetic,
    "profile_fact": profile_fact,
    "error_on_screen": error_on_screen,
    "missing_attachment": missing_attachment,
    "repeated_word": repeated_word,
}

# What a spoken sentence can be checked for; screen-only checks (commands, errors, typing) do not apply.
SPEECH_KINDS = ("weekday_date", "arithmetic", "profile_fact")


def detect(text: str, ctx: TextContext, *, kinds: tuple[str, ...] = KINDS) -> list[Correction]:
    """Every correction the enabled detectors find, most urgent first."""
    if not text or not text.strip():
        return []
    allowed = SPEECH_KINDS if ctx.source == "speech" else kinds
    results: list[Correction] = []
    for kind in kinds:
        if kind not in allowed or kind not in DETECTORS:
            continue
        try:
            results.extend(DETECTORS[kind](text, ctx))
        except Exception:  # a detector bug must never take the live layer down
            continue
    order = {URGENT: 0, NORMAL: 1, LOW: 2}
    return sorted(results, key=lambda c: order.get(c.urgency, 3))
