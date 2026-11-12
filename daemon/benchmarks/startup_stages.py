"""What each start-up stage does before the first request can be answered, as counts and order, not seconds.

Reads ``PilotServer.initialize`` (the stages are its ``_startup_stage`` calls; see pilot.system.startup) and, for each
stage, lists in order:

- the imports it makes and how many modules each pulls in that were not loaded yet (measured by importing them,
  stage by stage, in a fresh interpreter: the count does not depend on how busy the machine is);
- the awaited calls (stores opened, services started): each is work the first request waits for;
- the blocks that only run for some people, with the setting that decides it.

    .venv\\Scripts\\python.exe benchmarks\\startup_stages.py           # counts, as text
    .venv\\Scripts\\python.exe benchmarks\\startup_stages.py --json    # the same as JSON

Nothing is started: no store is opened, no model or network is touched. Importing is the only thing it runs.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1] / "pilot" / "server.py"
BEFORE_FIRST = "(before the first stage: imports, the key vault, Air Handoff)"


@dataclass
class Stage:
    name: str
    imports: list[str] = field(default_factory=list)
    awaited: list[str] = field(default_factory=list)
    conditional: list[str] = field(default_factory=list)
    new_modules: dict[str, int] = field(default_factory=dict)


def _initialize(tree: ast.Module) -> ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "PilotServer":
            for item in node.body:
                if isinstance(item, ast.AsyncFunctionDef) and item.name == "initialize":
                    return item
    raise LookupError("PilotServer.initialize not found")


def _stage_name(node: ast.stmt) -> str | None:
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        func = node.value.func
        if isinstance(func, ast.Attribute) and func.attr == "_startup_stage" and node.value.args:
            arg = node.value.args[0]
            if isinstance(arg, ast.Constant):
                return str(arg.value)
    return None


def _condition(source: str, node: ast.If) -> str:
    return " ".join((ast.get_source_segment(source, node.test) or "").split())


def _collect(source: str, nodes: list[ast.stmt], stage: Stage, guards: tuple[str, ...] = ()) -> None:
    """Imports and awaited calls in order, each awaited call with the settings that must hold for it to run."""

    def visit(node: ast.AST, guards: tuple[str, ...]) -> None:
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module not in stage.imports:
                stage.imports.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in stage.imports:
                    stage.imports.append(alias.name)
        elif isinstance(node, ast.Await):
            text = " ".join((ast.get_source_segment(source, node.value) or "").split())
            head = text.split("(", 1)[0]
            stage.awaited.append(head + (f"  [only if {' and '.join(guards)}]" if guards else ""))
        if isinstance(node, ast.If):
            condition = _condition(source, node)
            setting = ("config" in condition or "_enabled" in condition) and condition
            if setting and setting not in stage.conditional:
                stage.conditional.append(setting)
            visit(node.test, guards)
            for child in node.body:
                visit(child, (*guards, condition) if setting else guards)
            for child in node.orelse:
                visit(child, (*guards, f"not ({condition})") if setting else guards)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, guards)

    for node in nodes:
        visit(node, guards)


def stages() -> list[Stage]:
    source = SERVER.read_text(encoding="utf-8")
    body = _initialize(ast.parse(source)).body
    found = [Stage(BEFORE_FIRST)]
    for node in body:
        name = _stage_name(node)
        if name is not None:
            found.append(Stage(name))
            continue
        _collect(source, [node], found[-1])
    return found


_COUNT = """
import importlib, json, sys
plan = json.loads(sys.argv[1])
out = {}
for stage, modules in plan:
    counts = {}
    for name in modules:
        before = len(sys.modules)
        try:
            importlib.import_module(name)
        except Exception as error:
            counts[name] = f"failed: {type(error).__name__}"
            continue
        counts[name] = len(sys.modules) - before
    out[stage] = counts
print(json.dumps(out))
"""


def count_modules(found: list[Stage]) -> None:
    """New modules per import, in stage order, in one fresh interpreter that already has pilot.server's own imports."""
    plan = [(stage.name, stage.imports) for stage in found]
    script = "import pilot.server\n" + _COUNT  # the module-level imports happen before start() runs at all
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(plan)],
        capture_output=True,
        text=True,
        cwd=SERVER.parents[1],
        check=True,
    )
    counted = json.loads(result.stdout.strip().splitlines()[-1])
    for stage in found:
        stage.new_modules = counted.get(stage.name, {})


def main() -> None:
    found = stages()
    count_modules(found)
    if "--json" in sys.argv:
        print(json.dumps([stage.__dict__ for stage in found], indent=1))
        return
    for number, stage in enumerate(found):
        modules = sum(value for value in stage.new_modules.values() if isinstance(value, int))
        print(f"{number}. {stage.name}: {modules} new modules, {len(stage.awaited)} awaited calls")
        heavy = sorted(
            ((value, name) for name, value in stage.new_modules.items() if isinstance(value, int) and value >= 25),
            reverse=True,
        )
        for value, name in heavy:
            print(f"     imports {name}: {value} modules")
        for call in stage.awaited:
            print(f"     awaits {call}")
        for condition in stage.conditional:
            print(f"     only when {condition}")


if __name__ == "__main__":
    main()
