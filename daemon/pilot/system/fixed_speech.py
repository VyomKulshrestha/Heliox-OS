"""The sentences Heliox says word for word, gathered from where they are written, so their audio can be made ahead.

Nothing is typed twice: the conversation layer, the personal layer and Heliox Live write their fixed replies as string
literals (constants such as ``say_back.DROPPED``, tables such as ``handover.MEDIA_ACKS``, and plain ``return "Working
on it."``), and this module reads those source files and keeps every literal shaped like a spoken sentence, and the
sentences of an f-string reply that hold no changing part ("Want me to wrap up the day?"). Anything with a changing
part (an f-string's own sentence, ``.format``, a ``%`` or a brace) is left out: it is synthesised as today.

The result is split into the pieces the local voice speaks (``local_tts_worker.split_for_speech``), since the store of
made audio (``local_tts_worker.SpeechStore``) is looked up piece by piece.
"""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path

PILOT = Path(__file__).resolve().parents[1]
# Where Heliox's own spoken sentences are written.
PACKAGES = ("conversation", "personal", "live")
# Files in those packages that hold text that is not spoken by Heliox: prompts for the model, bench corpora and
# their expected answers, and the person's own words the rules match.
NOT_SPOKEN = frozenset(
    {
        "conversation/bench.py",  # bench prompts and expected answers
        "conversation/style.py",  # the model's style prompt
        "conversation/fast_model.py",  # the model's system prompt
        "conversation/planner_context.py",  # context written for the planner
        "conversation/context.py",  # context written for the model
        "conversation/router.py",  # the person's phrasings the router matches
        "personal/bench.py",
        "personal/context.py",
        "live/bench.py",
        "live/grounded_eval.py",
        "live/policy.py",  # rules text for the model
        "live/focused_text.py",  # why reading the focused field failed: shown in diagnostics, not spoken
        "live/focused_text_linux.py",
        "live/focused_text_macos.py",
        "live/focused_text_policy.py",
        "live/terminal_text.py",
        "live/shell_line.py",
    }
)
# The order the audio is made in: what a person hears most comes first, so the first minute of making ahead already
# covers most of what is said (acknowledgements and holding phrases, the say-back, the layer's own short replies and
# the quick road, then the offers), then every other file.
FIRST = (
    "conversation/handover.py",
    "conversation/say_back.py",
    "conversation/layer.py",
    "conversation/quick.py",
    "conversation/action_talk.py",
    "conversation/followups.py",
    "personal/offers.py",
    "personal/offer_day.py",
    "personal/getting_to_know.py",
    "live/engine.py",
)
MAX_WORDS = 24  # a reply sentence; longer literals are prompts or notes
_SHAPE = re.compile(r"^[\"'“A-Z][^{}%\n\t_/\\<>|`]*[.!?…][\"'”)]?$")
_LOGGING = frozenset({"debug", "info", "warning", "warn", "error", "exception", "critical", "log"})


def spoken_shape(text: str) -> bool:
    """A literal that reads as a sentence Heliox says: capitalised, ending like a sentence, no placeholder, not too
    long, and with letters in it."""
    text = text.strip()
    if not text or text != " ".join(text.split()) or not _SHAPE.match(text):
        return False
    words = text.split()
    return 1 <= len(words) <= MAX_WORDS and any(c.isalpha() for c in text) and not text.isupper()


def _not_spoken_nodes(tree: ast.AST) -> set[int]:
    """Ids of the string nodes that are never speech: docstrings, format templates, log, print and exception messages,
    and dictionary keys (f-strings are handled apart, in sentences_in)."""
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                skip.add(id(body[0].value))
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if isinstance(func, ast.Attribute) and func.attr == "format":
                skip.update(id(part) for part in ast.walk(func.value))
            if name in _LOGGING or name in {"print"}:
                skip.update(id(part) for arg in node.args for part in ast.walk(arg))
        elif isinstance(node, ast.Raise) and node.exc is not None:
            skip.update(id(part) for part in ast.walk(node.exc))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            skip.update(id(part) for part in ast.walk(node.left))
        elif isinstance(node, ast.Dict):
            skip.update(id(key) for key in node.keys if key is not None)
    return skip


def _fixed_pieces_of_fstring(node: ast.JoinedStr) -> list[str]:
    """The sentences written whole in an f-string reply's literal parts ("{sentence} I couldn't check that it worked.",
    "It's nearly {time}, when you stop work. Want me to wrap up the day?"): the local voice speaks each such sentence as
    a piece of its own whatever fills the changing parts. A part's text that is not a whole sentence is left out."""
    from pilot.system.local_tts_worker import split_for_speech

    found = []
    for part in node.values:
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            for piece in split_for_speech(" ".join(part.value.split())):
                if spoken_shape(piece):
                    found.append(piece)
    return found


def sentences_in(source: str) -> list[str]:
    """Every literal in ``source`` shaped like a spoken sentence, and the unchanging sentences of f-string replies, in
    order, without repeats."""
    tree = ast.parse(source)
    skip = _not_spoken_nodes(tree)
    in_fstrings = {id(part) for node in ast.walk(tree) if isinstance(node, ast.JoinedStr) for part in ast.walk(node)}
    found: dict[str, None] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr) and id(node) not in skip:
            for piece in _fixed_pieces_of_fstring(node):
                found.setdefault(piece, None)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in skip or id(node) in in_fstrings:
                continue
            text = node.value.strip()
            if spoken_shape(text):
                found.setdefault(text, None)
    return list(found)


def source_files(root: Path = PILOT) -> list[Path]:
    """The files read, those in FIRST first (in that order), then the rest by name."""
    files = []
    for package in PACKAGES:
        for path in sorted((root / package).rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            if relative not in NOT_SPOKEN:
                files.append(path)

    def rank(path: Path) -> tuple[int, str]:
        relative = path.relative_to(root).as_posix()
        return (FIRST.index(relative) if relative in FIRST else len(FIRST), relative)

    return sorted(files, key=rank)


SHORT_FIRST_WORDS = 4  # "Okay.", "Next track.", "Working on it.", "Still on the plan."


@lru_cache(maxsize=1)
def gather() -> tuple[str, ...]:
    """The pieces the local voice speaks for every fixed sentence, without repeats, in the order to make them: the
    short pieces of the files in FIRST (acknowledgements, holding phrases, the say-back, the layer's short replies),
    then the rest of those files, then every other file. Empty when the source files are not there (nothing is made
    ahead then)."""
    from pilot.system.local_tts_worker import split_for_speech

    ranked: dict[str, tuple[int, int, int]] = {}
    files = source_files()
    for rank, path in enumerate(files):
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        first = path.relative_to(PILOT).as_posix() in FIRST
        for sentence in sentences_in(source):
            for piece in split_for_speech(sentence):
                short = first and len(piece.split()) <= SHORT_FIRST_WORDS
                ranked.setdefault(piece, (0 if short else 1, rank, len(ranked)))
    return tuple(sorted(ranked, key=ranked.__getitem__))
