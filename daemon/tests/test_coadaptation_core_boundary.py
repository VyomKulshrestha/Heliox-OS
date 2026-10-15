"""Import boundary of the co-adaptation SDK core (spec 3, 13, 22.1).

``pilot/coadaptation/core/*.py`` and ``pilot/coadaptation/__init__.py`` may import only the standard library, numpy
and ``pilot.coadaptation.core.*`` (relative imports preferred); never ``ActionType``, river, scipy or pydantic.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from pilot.coadaptation.core import state

DAEMON = Path(__file__).resolve().parents[1]
PACKAGE = DAEMON / "pilot" / "coadaptation"
CORE = PACKAGE / "core"
FORBIDDEN_MODULES = ("river", "scipy", "pydantic", "cryptography", "keyring", "aiosqlite")


def _core_files() -> list[Path]:
    return sorted(CORE.glob("*.py")) + [PACKAGE / "__init__.py"]


def _imports(path: Path) -> list[tuple[str, int]]:
    """(module, level) of every import statement in a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((node.module or "", node.level))
    return found


def test_core_imports_only_the_standard_library_numpy_and_the_core() -> None:
    stdlib = set(sys.stdlib_module_names)
    files = _core_files()
    assert len(files) > 10
    for path in files:
        for module, level in _imports(path):
            if level > 0:
                assert path.parent == CORE, f"{path.name}: relative import outside the core"
                continue
            top = module.split(".")[0]
            if top in stdlib or top == "numpy" or top == "__future__":
                continue
            assert module.startswith("pilot.coadaptation.core"), f"{path.name} imports {module}"


def test_core_never_names_forbidden_modules_or_action_types() -> None:
    for path in _core_files():
        modules = [module for module, _ in _imports(path)]
        for module in modules:
            assert module.split(".")[0] not in FORBIDDEN_MODULES, f"{path.name} imports {module}"
            assert not module.startswith("pilot.actions"), path.name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        assert "ActionType" not in names, path.name


def test_package_init_holds_only_a_docstring_and_an_empty_all() -> None:
    tree = ast.parse((PACKAGE / "__init__.py").read_text(encoding="utf-8"))
    body = tree.body
    assert isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
    docstring = body[0].value.value
    assert "presentation" in docstring and "never approves, clicks or runs anything" in docstring
    assert len(body) == 2
    assign = body[1]
    assert isinstance(assign, ast.Assign) and assign.targets[0].id == "__all__"  # type: ignore[attr-defined]
    assert isinstance(assign.value, ast.List) and assign.value.elts == []


def test_importing_the_core_loads_no_authority_or_evaluation_module() -> None:
    probe = (
        "import sys, pilot.coadaptation.core as core\n"
        "from pilot.coadaptation.core import CoAdaptationEngine, PreferenceSet, ConsentRecord\n"
        "bad = [m for m in sys.modules if m.startswith(('pilot.actions', 'pilot.security', 'pilot.agents', "
        "'pilot.neural', 'pilot.config', 'pilot.server'))]\n"
        "wp4 = [m for m in sys.modules if m in ('pilot.coadaptation.core.ope', 'pilot.coadaptation.core.evaluation', "
        "'pilot.coadaptation.core.simulator', 'pilot.coadaptation.core.audit')]\n"
        "third = [m for m in sys.modules if m.split('.')[0] in ('scipy', 'pydantic', 'river')]\n"
        "print(bad, wp4, third, core.SDK_VERSION)\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=DAEMON, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[] [] [] 0.1.0"


def test_core_reexports_resolve() -> None:
    import pilot.coadaptation.core as core

    missing = [name for name in core.__all__ if not hasattr(core, name)]
    assert missing == []


def test_feature_names_and_context_keys_avoid_denied_substrings() -> None:
    denied = state.DENIED_FEATURE_SUBSTRINGS
    names: list[str] = []
    for attribute in ("L1_FEATURE_NAMES", "L2_FEATURE_NAMES", "BURDEN_FEATURE_NAMES"):
        names.extend(getattr(state, attribute, ()))
    for keys in state.ALLOWED_CONTEXT_KEYS.values():
        names.extend(keys)
    assert names
    for name in names:
        assert not any(word in name for word in denied), name
    assert "fatigue" in names  # session time, named fatigue deliberately; the denied entry is fatigue_score
