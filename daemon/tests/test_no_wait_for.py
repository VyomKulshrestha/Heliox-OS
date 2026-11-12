"""No asyncio.wait_for under daemon/pilot: before Python 3.12 it can lose a cancellation that arrives as the awaited
thing finishes, and Heliox installs on 3.11 (pilot/timeouts.py). Use pilot.timeouts.bounded, or asyncio.timeout.

ALLOWED lists a file and the number of uses it may keep, each with a written reason; it is empty.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PILOT = Path(__file__).resolve().parents[1] / "pilot"
ALLOWED: dict[str, tuple[int, str]] = {}  # "pilot/x.py": (uses, "why this one must stay")


def wait_for_uses(source: str) -> list[int]:
    """Line numbers of asyncio.wait_for calls and of wait_for imported from asyncio, however it is spelled."""
    tree = ast.parse(source)
    asyncio_names = {"asyncio"}
    wait_for_names: set[str] = set()
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            asyncio_names |= {alias.asname or alias.name for alias in node.names if alias.name == "asyncio"}
        elif isinstance(node, ast.ImportFrom) and node.module in ("asyncio", "asyncio.tasks"):
            for alias in node.names:
                if alias.name == "wait_for":
                    wait_for_names.add(alias.asname or alias.name)
                    lines.append(node.lineno)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "wait_for":
            base = func.value
            if (
                isinstance(base, ast.Name)
                and base.id in asyncio_names
                or isinstance(base, ast.Attribute)
                and base.attr == "tasks"
            ):
                lines.append(node.lineno)
        elif isinstance(func, ast.Name) and func.id in wait_for_names:
            lines.append(node.lineno)
    return sorted(set(lines))


def test_the_finder_sees_every_spelling():
    source = (
        "import asyncio\nimport asyncio as aio\nfrom asyncio import wait_for as wf\n"
        "async def f(x):\n    await asyncio.wait_for(x, 1)\n    await aio.wait_for(x, 1)\n    await wf(x, 1)\n"
        "    await asyncio.tasks.wait_for(x, 1)\n    await other.wait_for(x)\n"
    )
    assert wait_for_uses(source) == [3, 5, 6, 7, 8]


def test_no_asyncio_wait_for_under_daemon_pilot():
    found = {}
    for path in sorted(PILOT.rglob("*.py")):
        uses = wait_for_uses(path.read_text(encoding="utf-8"))
        name = path.relative_to(PILOT.parent).as_posix()
        if uses and len(uses) != ALLOWED.get(name, (0, ""))[0]:
            found[name] = uses
    assert not found, f"asyncio.wait_for found (use pilot.timeouts.bounded): {found}"


@pytest.mark.parametrize("name", sorted(ALLOWED))
def test_every_allowed_use_has_a_reason(name):
    uses, reason = ALLOWED[name]
    assert uses > 0 and len(reason) > 20
