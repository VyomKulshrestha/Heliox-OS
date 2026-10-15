"""The co-adaptation adapters stay outside Heliox's authority layer (spec 3, 13, tests of 22.2)."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

DAEMON_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = DAEMON_DIR / "pilot" / "coadaptation"
ADAPTER_FILES = sorted(PACKAGE_DIR.glob("*.py"))

ALLOWED_PILOT_IMPORTS = ("pilot.db.sqlite_pool", "pilot.coadaptation")
FORBIDDEN_PILOT_PACKAGES = (
    "pilot.security",
    "pilot.agents",
    "pilot.system",
    "pilot.actions",
    "pilot.memory",
    "pilot.cognitive",
    "pilot.multimodal",
    "pilot.intelligence",
    "pilot.server",
    "pilot.workflows",
    "pilot.config",
    "pilot.neural",
)
EVALUATION_MODULES = (
    "pilot.coadaptation.core.ope",
    "pilot.coadaptation.core.evaluation",
    "pilot.coadaptation.core.simulator",
    "pilot.coadaptation.core.audit",
)
FORBIDDEN_NAMES = ("ExperienceLedger", "MemoryStore", "RiskEvidence", "Executor", "_pending_confirms")
FORBIDDEN_CALLS = ("set_preference", "add_preference", "persona_add_preference")
MEMORY_PREFERENCE_CALL = re.compile(r"(memory|persona|ledger)\w*\s*\.\s*(set|add)_preference\s*\(", re.IGNORECASE)
RUNTIME_FORBIDDEN = ("pilot.actions", "pilot.security", "pilot.agents", "pilot.neural", "pilot.config")


def _absolute(module: str | None, level: int, path: Path) -> str:
    if level == 0:
        return module or ""
    package = path.relative_to(DAEMON_DIR).with_suffix("").parts[:-1]
    base = list(package[: len(package) - (level - 1)])
    return ".".join([*base, module] if module else base)


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute(node.module, node.level, path)
            names.append(base)
            names.extend(f"{base}.{alias.name}" for alias in node.names)
    return names


def _matches(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def test_adapter_files_are_scanned():
    names = {path.name for path in ADAPTER_FILES}
    assert {"consent.py", "schemas.py", "keys.py", "store.py", "preferences_store.py", "service.py"} <= names


def test_adapters_import_only_the_sqlite_pool_and_coadaptation():
    for path in ADAPTER_FILES:
        for name in _imports(path):
            if not _matches(name, "pilot"):
                continue
            assert any(_matches(name, allowed) for allowed in ALLOWED_PILOT_IMPORTS) or _matches(
                "pilot.db.sqlite_pool", name
            ), (path.name, name)
            for forbidden in FORBIDDEN_PILOT_PACKAGES:
                assert not _matches(name, forbidden), (path.name, name)


def test_adapters_never_import_the_evaluation_modules():
    for path in ADAPTER_FILES:
        for name in _imports(path):
            for module in EVALUATION_MODULES:
                assert not _matches(name, module), (path.name, name)


def test_adapters_never_name_the_authority_or_memory_layer():
    for path in ADAPTER_FILES:
        source = path.read_text(encoding="utf-8")
        for name in FORBIDDEN_NAMES:
            assert name not in source, (path.name, name)
        assert "persona_add_preference" not in source, path.name
        assert not MEMORY_PREFERENCE_CALL.search(source), path.name


def test_adapters_never_call_a_preference_writer():
    for path in ADAPTER_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                assert called not in FORBIDDEN_CALLS, (path.name, called, node.lineno)
    # The service's own RPC method of that name is a definition, not a call into memory.
    service = ast.parse((PACKAGE_DIR / "service.py").read_text(encoding="utf-8"))
    defined = {node.name for node in ast.walk(service) if isinstance(node, ast.AsyncFunctionDef)}
    assert "set_preference" in defined


def test_importing_the_service_loads_no_authority_module():
    result = subprocess.run(
        [sys.executable, "-c", "import sys, pilot.coadaptation.service; print(sorted(sys.modules))"],
        cwd=DAEMON_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    modules = ast.literal_eval(result.stdout.strip().splitlines()[-1])
    assert "pilot.coadaptation.service" in modules
    for name in modules:
        for forbidden in RUNTIME_FORBIDDEN:
            assert not _matches(name, forbidden), name
    for module in EVALUATION_MODULES:
        assert module not in modules
