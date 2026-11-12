"""Which code on the spoken path branches by operating system (pure: reads source text, imports nothing).

The bench runs where the speech models are installed, which today is one Windows machine, so it cannot see how
macOS and Linux behave. This scan lists, in the modules a spoken turn passes through, every function that tests
the platform, so the gaps can be checked one by one. It also collects anything Heliox said or returned during a
run that reads as "not available here".
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SPOKEN_PATH = (
    "pilot/system/voice.py",
    "pilot/system/call_detection.py",
    "pilot/system/local_tts_worker.py",
    "pilot/system/kokoro_tts.py",
    "pilot/system/streaming_asr.py",
    "pilot/system/parakeet_asr.py",
    "pilot/system/turn_detector.py",
    "pilot/system/vad.py",
    "pilot/system/companion_speech.py",
    "pilot/live/engine.py",
    "pilot/live/focused_text.py",
    "pilot/live/focused_text_linux.py",
    "pilot/live/focused_text_macos.py",
    "pilot/live/spoken.py",
    "pilot/system/accessibility.py",
)
_TEST = re.compile(r"sys\.platform|os\.name|platform\.system\(\)|IS_WINDOWS|IS_MAC|IS_LINUX|win32|darwin")
NOT_AVAILABLE = re.compile(
    r"not (?:available|supported)|only (?:on|works on) (?:windows|macos|linux)|unsupported platform|"
    r"isn't available|is not available on",
    re.IGNORECASE,
)


def platform_branches(source: str) -> list[tuple[str, int, str]]:
    """(function, line, the test) for each function whose body tests the platform."""
    tree = ast.parse(source)
    lines = source.splitlines()
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for i in range(node.lineno - 1, getattr(node, "end_lineno", node.lineno)):
                match = _TEST.search(lines[i])
                if match and not lines[i].lstrip().startswith("#"):
                    out.append((node.name, i + 1, lines[i].strip()[:100]))
                    break
    return sorted(set(out), key=lambda x: x[1])


def scan(daemon_dir: Path) -> dict[str, list[tuple[str, int, str]]]:
    found = {}
    for rel in SPOKEN_PATH:
        path = daemon_dir / rel
        if path.exists():
            branches = platform_branches(path.read_text(encoding="utf-8"))
            if branches:
                found[rel] = branches
    return found


def unavailable_messages(texts: list[str]) -> list[str]:
    return sorted({t for t in texts if NOT_AVAILABLE.search(t)})
