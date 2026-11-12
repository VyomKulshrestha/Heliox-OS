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
    "system/processes.py::process_exists": "an integer process id (the name is a $HelioxArgs value)",
    "system/voice.py::_record_audio": "an integer duration",
    "agents/screen_vision.py::_sync_fallback_screenshot_hash": "the daemon's own temporary file path",
    "system/accessibility.py::capture_snapshot": "a window handle and a limit, both integers",
    "system/accessibility.py::press_in_background": "a runtime id checked against a digits-and-dots regex first",
    "system/command_line.py::windows_command_line": "cmd /c only after every part is checked for cmd's code characters",
    "system/command_line.py::batch_file_command": "cmd /c on the daemon's own temporary file, its path checked",
    "system/platform_detect.py::run_shell_script": "the script is the approved shell_script action itself",
}

# Still to be fixed (patches 2 and 3 of the sweep). This list only shrinks; a new place fails the test.
NOT_YET: set[str] = set()


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

    def __init__(self, statements: list[ast.stmt], functions: dict[str, ast.AST], module: _Scope | None = None) -> None:
        self.assigned: dict[str, list[ast.AST]] = {}
        self.functions = functions
        self.module = module
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


def _constant_text(node: ast.AST, scope: _Scope | None = None, seen: frozenset[str] = frozenset()) -> bool:
    """Text made only of constants: literals, "a" + "b", a name every assignment of which is constant text (in the
    function or at module level), and a call to a function of the module that only returns constant text."""
    if _text_constant(node):
        return True
    if isinstance(node, ast.JoinedStr):
        return all(isinstance(v, ast.Constant) for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _constant_text(node.left, scope, seen) and _constant_text(node.right, scope, seen)
    if scope is None:
        return False
    if isinstance(node, ast.Name) and node.id not in seen:
        values = scope.assigned.get(node.id) or (scope.module.assigned.get(node.id) if scope.module else None)
        return bool(values) and all(_constant_text(v, scope, seen | {node.id}) for v in values)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in scope.functions:
        key = "fn:" + node.func.id
        if key in seen or node.args or node.keywords:
            return False
        inner = _Scope(list(scope.functions[node.func.id].body), scope.functions, scope.module)
        returns = [n.value for n in inner.nodes if isinstance(n, ast.Return) and n.value is not None]
        return bool(returns) and all(_constant_text(v, inner, seen | {key}) for v in returns)
    return False


def _built(node: ast.AST, scope: _Scope, seen: set[str]) -> bool:
    """Whether ``node`` builds text from something that is not a constant."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.JoinedStr) and not _constant_text(sub):
            return True
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Mod) and _text_constant(sub.left):
            return True
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Add) and not _constant_text(sub, scope):
            leaves = _add_leaves(sub)
            if any(isinstance(leaf, ast.JoinedStr) or _constant_text(leaf, scope) for leaf in leaves):
                return True
        if isinstance(sub, ast.Call) and _call_name(sub) in ("format", "join"):
            receiver = sub.func.value if isinstance(sub.func, ast.Attribute) else None
            if receiver is not None and _text_constant(receiver):
                if not all(_constant_text(a, scope) for a in sub.args):
                    return True
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in scope.functions:
            key = "fn:" + sub.func.id
            if key not in seen:
                seen.add(key)
                inner = _Scope(list(scope.functions[sub.func.id].body), scope.functions, scope.module)
                returns = [n.value for n in inner.nodes if isinstance(n, ast.Return) and n.value is not None]
                if any(_built(value, inner, seen) for value in returns):
                    return True
        if isinstance(sub, ast.Name) and sub.id in scope.assigned and sub.id not in seen:
            seen.add(sub.id)
            if any(_built(value, scope, seen) for value in scope.assigned[sub.id]):
                return True
    return False


def _add_leaves(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _add_leaves(node.left) + _add_leaves(node.right)
    return [node]


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
        module = _Scope(top, functions)
        scopes = [("<module>", module)] + [
            (name, _Scope(list(fn.body), functions, module)) for name, fn in functions.items()
        ]
        for scope_name, scope in scopes:
            calls = [node for node in scope.nodes if isinstance(node, ast.Call)]
            lists = [node for node in scope.nodes if isinstance(node, (ast.List, ast.Tuple))]
            if any(_pasted(node, scope) for node in calls) or any(_pasted_list(node, scope) for node in lists):
                found.add(f"{rel}::{scope_name}")
    return found


def _pasted(node: ast.Call, scope: _Scope) -> bool:
    code, any_value = _interpreter_code(node, scope)
    if any_value and any(not _constant_text(arg, scope) for arg in code):
        return True
    return any(_built(arg, scope, set()) for arg in code)


def _pasted_list(node: ast.List | ast.Tuple, scope: _Scope) -> bool:
    """An argument list built anywhere (returned, stored, passed on) that starts an interpreter with -c or /c, whose
    code argument is built from a value: the sandbox's old ulimit wrapper pasted a path into "bash -c" this way."""
    items = list(node.elts)
    if len(items) < 3 or not (_text_constant(items[0]) and _text_constant(items[1])):
        return False
    if str(items[0].value).casefold() not in INTERPRETERS or str(items[1].value) not in ("-c", "/c", "-Command"):
        return False
    return _built(items[2], scope, set()) or (
        str(items[0].value).casefold() in ("cmd", "cmd.exe") and not all(_constant_text(i, scope) for i in items[2:])
    )


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
        top = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        scope = _Scope(list(functions["f"].body), functions, _Scope(top, functions))
        return any(_pasted(node, scope) for node in scope.nodes if isinstance(node, ast.Call))

    def scan_lists(source: str) -> bool:
        tree = ast.parse(textwrap.dedent(source))
        functions = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        scope = _Scope(list(functions["f"].body), functions, _Scope([], functions))
        return any(_pasted_list(n, scope) for n in scope.nodes if isinstance(n, (ast.List, ast.Tuple)))

    assert scan("def f(x):\n    run_powershell(f\"Get-Item '{x}'\")")
    assert scan("def f(x):\n    s = 'Get-Item ' + x\n    run_powershell(s)")
    assert scan("def f(x):\n    run_powershell('Get-Item %s' % x)")
    assert scan("def f(x):\n    run_powershell('Get-Item {}'.format(x))")
    assert scan("def g(x):\n    return f\"tell app {x}\"\ndef f(x):\n    run_command(['osascript', '-e', g(x)])")
    assert scan("def f(url):\n    run_command(['cmd', '/c', 'start', '', url])")
    assert scan("def f(cmd):\n    run_command(['cmd', '/c'] + cmd)")
    assert scan("def f(x):\n    subprocess.run(['sh', '-c', 'ls ' + x])")
    assert scan_lists("def f(cmd):\n    return ['bash', '-c', 'ulimit -v 1; exec ' + ' '.join(cmd)]")
    assert not scan_lists("def f(cmd):\n    return ['bash', '-c', 'exec \"$@\"', 'x', *cmd]")
    assert not scan("def f(x):\n    run_powershell('Get-Item $HelioxArgs[\\'p\\']', values={'p': x})")
    assert not scan("def f(x):\n    run_command(['notify-send', '--', x])")
    assert not scan("def f():\n    run_powershell('Get-Date; ' + 'Get-Item .')")
    # Constant pieces held in module names or returned by a helper are constant.
    assert not scan("A = 'Get-Date; '\ndef g():\n    return 'Get-Item .'\ndef f():\n    run_powershell(A + g())")
    assert scan("A = 'Get-Date; '\ndef f(x):\n    run_powershell(A + x)")
