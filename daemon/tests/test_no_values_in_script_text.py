"""No value is pasted into the text of code a shell or script interpreter runs.

A value written into PowerShell, AppleScript, cmd or shell text, even escaped, can end the string it sits in and
become code (PowerShell also ends a single-quoted string at U+2018 to U+201B). Values go in as data instead:
``run_powershell(script, values=...)``, ``script_values.osascript_command(body, values)``, or as arguments of a
program run directly. This test reads the daemon's source and fails on any call that hands such an interpreter
code built with an f-string, ``+``, ``%``, ``.format`` or ``.join`` from something that is not a constant (followed
through local names and through helper functions in the same module), or hands ``cmd /c`` any argument that is not
a constant, unless the place is listed below with the reason it is safe.
"""

from __future__ import annotations

import ast
from pathlib import Path

PILOT = Path(__file__).resolve().parents[1] / "pilot"

INTERPRETERS = {"powershell", "pwsh", "osascript", "cmd", "cmd.exe", "bash", "sh", "zsh"}
CODE_RUNNERS = {"run_powershell", "run_shell_script"}  # their first argument is code
PROCESS_CALLS = {"run_command", "run", "Popen", "check_output", "check_call", "call", "create_subprocess_exec"}

# Built from values that cannot carry code. Each says why.
SAFE = {
    "plugins/media/media_plugin.py::volume_set": "an integer volume level",
    "plugins/media/media_plugin.py::media_key": "a key code from a fixed table",
    "system/volume.py::volume_set": "an integer volume level",
    "system/volume.py::volume_mute": "true or false",
    "system/screen.py::brightness_set": "an integer brightness level",
    "system/power.py::power_transition_observed": "a log name and event id from fixed tables",
    "system/processes.py::process_info": "an integer process id",
    "system/voice.py::_record_audio": "an integer duration",
    "agents/screen_vision.py::_sync_fallback_screenshot_hash": "the daemon's own temporary file path",
    "system/accessibility.py::capture_snapshot": "a window handle and a limit, both integers",
    "system/accessibility.py::press_in_background": "a runtime id checked against a digits-and-dots regex first",
    "system/code_exec.py::_execute_cmd": "runs the approved code from the daemon's own temporary .cmd file",
    "system/platform_detect.py::run_shell_script": "the script is the approved shell_script action itself",
}

# Still to be fixed (patches 2 and 3 of the sweep). This list only shrinks; a new place fails the test.
NOT_YET = {
    "system/download.py::download_file",
    "agents/executor.py::_exec_registry_read",
    "agents/executor.py::_exec_registry_write",
    "agents/executor.py::_exec_shell_command",
    "agents/executor.py::_exec_open_url",
    "system/window_mgr.py::window_focus",
    "system/window_mgr.py::window_close",
    "system/window_mgr.py::window_minimize",
    "system/window_mgr.py::window_maximize",
    "system/window_mgr.py::window_exists",
    "system/window_mgr.py::window_read_text",
    "system/window_mgr.py::window_set_text",
    "system/voice.py::_tts_windows",
    "system/voice.py::_transcribe_windows",
    "system/clipboard.py::clipboard_write",
    "system/processes.py::process_exists",
    "system/disks.py::mount_exists",
    "system/applications.py::_resolve_shortcut_target",
    "system/screen.py::screenshot",
}


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def _walk_without_defs(node: ast.AST):
    """ast.walk, but not into nested function, class or lambda bodies (scopes of their own)."""
    yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield from _walk_without_defs(child)


class _Scope:
    """The names assigned in one function (or at module level), with the module's functions, so text built in a
    helper function is followed to where it is built."""

    def __init__(self, statements: list[ast.stmt], functions: dict[str, ast.AST]) -> None:
        self.assigned: dict[str, list[ast.AST]] = {}
        self.functions = functions
        self.nodes = [n for stmt in statements for n in _walk_without_defs(stmt)]
        for node in self.nodes:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.assigned.setdefault(target.id, []).append(node.value)
            elif (
                isinstance(node, ast.AugAssign)
                and isinstance(node.target, ast.Name)
                or isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.value is not None
            ):
                self.assigned.setdefault(node.target.id, []).append(node.value)


def _text_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _constant_text(node: ast.AST) -> bool:
    """Text made only of constants ("a" "b", "a" + "b")."""
    if _text_constant(node):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _constant_text(node.left) and _constant_text(node.right)
    return False


def _built(node: ast.AST, scope: _Scope, seen: set[str]) -> bool:
    """Whether ``node`` builds text from something that is not a constant."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.JoinedStr) and any(isinstance(v, ast.FormattedValue) for v in sub.values):
            return True
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, (ast.Add, ast.Mod)) and not _constant_text(sub):
            sides = (sub.left, sub.right)
            if any(_text_constant(side) or isinstance(side, (ast.JoinedStr, ast.BinOp)) for side in sides):
                return True
        if isinstance(sub, ast.Call) and _call_name(sub) in ("format", "join"):
            receiver = sub.func.value if isinstance(sub.func, ast.Attribute) else None
            if receiver is not None and _text_constant(receiver) and not all(_constant_text(a) for a in sub.args):
                return True
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in scope.functions:
            key = "fn:" + sub.func.id
            if key not in seen:
                seen.add(key)
                inner = _Scope(list(scope.functions[sub.func.id].body), scope.functions)
                returns = [n.value for n in inner.nodes if isinstance(n, ast.Return) and n.value is not None]
                if any(_built(value, inner, seen) for value in returns):
                    return True
        if isinstance(sub, ast.Name) and sub.id in scope.assigned and sub.id not in seen:
            seen.add(sub.id)
            if any(_built(value, scope, seen) for value in scope.assigned[sub.id]):
                return True
    return False


def _list_items(node: ast.AST) -> list[ast.AST] | None:
    """The items of a literal argument list, also when it is ``["cmd", "/c"] + rest``."""
    if isinstance(node, (ast.List, ast.Tuple)):
        return list(node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _list_items(node.left)
        if left is not None:
            return [*left, node.right]
    return None


def _interpreter_code(node: ast.Call, scope: _Scope | None = None) -> tuple[list[ast.AST], bool]:
    """(the arguments of ``node`` that reach an interpreter as code, whether any non-constant one counts: cmd /c
    reads all its arguments as one command line). An argument list held in a local name is followed."""
    name = _call_name(node)
    if name in CODE_RUNNERS:
        return list(node.args[:1]), False
    if name not in PROCESS_CALLS or not node.args:
        return [], False
    first = node.args[0]
    if isinstance(first, ast.Name) and scope is not None and name != "create_subprocess_exec":
        for value in scope.assigned.get(first.id, []):
            items = _list_items(value)
            if items and _text_constant(items[0]) and str(items[0].value).casefold() in INTERPRETERS:
                first = value
                break
    items = list(node.args) if name == "create_subprocess_exec" else _list_items(first)
    if not items or not _text_constant(items[0]):
        return [], False
    program = str(items[0].value).casefold()
    if program not in INTERPRETERS:
        return [], False
    return items[1:], program in ("cmd", "cmd.exe")


def places() -> set[str]:
    found: set[str] = set()
    for path in sorted(PILOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(PILOT).as_posix()
        functions = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        top = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        scopes = [("<module>", top)] + [(name, list(fn.body)) for name, fn in functions.items()]
        for scope_name, statements in scopes:
            scope = _Scope(statements, functions)
            for node in scope.nodes:
                if not isinstance(node, ast.Call):
                    continue
                code, any_value = _interpreter_code(node, scope)
                if (
                    any_value
                    and any(not _constant_text(arg) for arg in code)
                    or any(_built(arg, scope, set()) for arg in code)
                ):
                    found.add(f"{rel}::{scope_name}")
    return found


def test_no_value_is_pasted_into_interpreter_code():
    unexpected = sorted(places() - set(SAFE) - NOT_YET)
    assert not unexpected, (
        "Values pasted into PowerShell, AppleScript, cmd or shell text (pass them as data: run_powershell(values=), "
        f"script_values.osascript_command, or separate arguments of a program run directly): {unexpected}"
    )


def test_the_lists_hold_only_places_that_still_exist():
    present = places()
    assert not (NOT_YET - present), f"fixed, take them off NOT_YET: {sorted(NOT_YET - present)}"
    assert not (set(SAFE) - present), f"no longer built from values, take them off SAFE: {sorted(set(SAFE) - present)}"


def test_the_scan_catches_each_way_of_building_code():
    """The scan itself, on small samples: each must be found, and the constant and data forms must not."""
    import textwrap

    def scan(source: str) -> bool:
        tree = ast.parse(textwrap.dedent(source))
        functions = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        fn = functions["f"]
        scope = _Scope(list(fn.body), functions)
        for node in scope.nodes:
            if isinstance(node, ast.Call):
                code, any_value = _interpreter_code(node, scope)
                if (any_value and any(not _constant_text(a) for a in code)) or any(
                    _built(a, scope, set()) for a in code
                ):
                    return True
        return False

    assert scan("def f(x):\n    run_powershell(f\"Get-Item '{x}'\")")
    assert scan("def f(x):\n    s = 'Get-Item ' + x\n    run_powershell(s)")
    assert scan("def f(x):\n    run_powershell('Get-Item %s' % x)")
    assert scan("def f(x):\n    run_powershell('Get-Item {}'.format(x))")
    assert scan("def g(x):\n    return f\"tell app {x}\"\ndef f(x):\n    run_command(['osascript', '-e', g(x)])")
    assert scan("def f(url):\n    run_command(['cmd', '/c', 'start', '', url])")
    assert scan("def f(cmd):\n    run_command(['cmd', '/c'] + cmd)")
    assert scan("def f(x):\n    subprocess.run(['sh', '-c', 'ls ' + x])")
    assert not scan("def f(x):\n    run_powershell('Get-Item $HelioxArgs[\\'p\\']', values={'p': x})")
    assert not scan("def f(x):\n    run_command(['notify-send', '--', x])")
    assert not scan("def f():\n    run_powershell('Get-Date; ' + 'Get-Item .')")
