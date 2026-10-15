"""Co-adaptation RPCs stay outside the authority path: source scan, runtime check and import isolation."""

from __future__ import annotations

import ast
import asyncio
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from pilot.config import PilotConfig
from pilot.intelligence.experience import ExperienceEventType, ExperienceLedger
from pilot.server import PendingConfirmation, PilotServer

DAEMON_ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = DAEMON_ROOT / "pilot"

FORBIDDEN_REFERENCES = (
    "_pending_confirms",
    "_executor",
    "_handle_confirm",
    "_permission_checker",
    "_agent_gateway",
    "_memory",
    "_experience_ledger",
    "input_control",
    "cursor",
)
# Every server method that a co-adaptation RPC runs.
CO_ADAPTATION_CODE = sorted(
    name
    for name in dir(PilotServer)
    if name.startswith("_handle_co_adaptation_")
    or name
    in {
        "_co_adaptation_call",
        "_co_adaptation_study_refusal",
        "_co_adaptation_study_call",
        "_set_co_adaptation_enabled",
    }
)
RPC_CALLS: list[tuple[str, dict[str, Any]]] = [
    ("co_adaptation_status", {}),
    ("co_adaptation_config_update", {"ambiguity_threshold": 0.2}),
    ("co_adaptation_consent_grant", {"participant_key": None, "authorized": True}),
    ("co_adaptation_consent_revoke", {"participant_key": "p-3fa29c1e"}),
    ("co_adaptation_preference_set", {"kind": "forbid_variant", "scope": {"context": "study"}, "value": "act"}),
    ("co_adaptation_preference_retract", {"pref_id": "rule-1"}),
    ("co_adaptation_learning_reset", {"participant_key": "p-3fa29c1e"}),
    ("co_adaptation_erase", {"participant_key": "p-3fa29c1e", "scope": "today"}),
    ("co_adaptation_results", {"participant_key": "p-3fa29c1e", "scope": "all"}),
    ("co_adaptation_export", {"participant_key": "p-3fa29c1e"}),
    ("co_adaptation_session_begin", {"participant_key": "p-3fa29c1e", "context": "study"}),
    ("co_adaptation_session_resume", {"session_id": "s-1"}),
    ("co_adaptation_recommend", {"session_id": "s-1", "level": "block_regime", "context": {}}),
    ("co_adaptation_recommend", {"session_id": "s-1", "level": "ambiguity_table", "context": {}}),
    ("co_adaptation_recommend", {"session_id": "s-1", "level": "offer", "context": {}}),
    ("co_adaptation_record_trial", {"session_id": "s-1", "trial": {"trial_id": "t-1"}}),
    ("co_adaptation_report_burden", {"session_id": "s-1", "report": {}}),
    ("co_adaptation_session_end", {"session_id": "s-1", "reason": "stopped"}),
]
SERVICE_METHODS = (
    "status",
    "grant_consent",
    "revoke_consent",
    "set_preference",
    "retract_preference",
    "reset_learning",
    "erase",
    "results",
    "export",
    "begin_session",
    "resume_session",
    "recommend",
    "record_trial",
    "report_burden",
    "end_session",
)


class _Service:
    """A fake service that answers every RPC with ok and keeps an active consent."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        for method in SERVICE_METHODS:
            setattr(self, method, self._rpc(method))

    def _rpc(self, method: str):
        async def call(params: dict) -> dict:
            self.calls.append(method)
            return {"status": "ok"}

        return call

    def has_active_consent(self) -> bool:
        return True

    async def on_config_changed(self, previous: Any) -> None:
        self.calls.append("on_config_changed")


def _source(name: str) -> str:
    return textwrap.dedent(inspect.getsource(getattr(PilotServer, name)))


def test_the_scan_covers_all_sixteen_handlers():
    handlers = [name for name in CO_ADAPTATION_CODE if name.startswith("_handle_co_adaptation_")]
    assert len(handlers) == 16


@pytest.mark.parametrize("name", CO_ADAPTATION_CODE)
def test_co_adaptation_code_never_references_authority_objects(name):
    source = _source(name)
    for forbidden in FORBIDDEN_REFERENCES:
        assert forbidden not in source, f"{name} references {forbidden}"


def test_refusal_check_reads_only_recording_status_task_flag_and_gate_state():
    tree = ast.parse(_source("_co_adaptation_study_refusal"))
    self_attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self"
    }
    assert self_attributes == {"_recorded_workflows", "_interactive_request_active", "_neural_controller"}

    calls: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if isinstance(owner, ast.Attribute) and isinstance(owner.value, ast.Name) and owner.value.id == "self":
            calls.setdefault(owner.attr, set()).add(node.func.attr)
    assert calls == {"_recorded_workflows": {"status"}, "_neural_controller": {"gate_state"}}


@pytest.mark.asyncio
async def test_every_co_adaptation_rpc_leaves_a_pending_approval_and_the_ledger_unchanged(tmp_path, monkeypatch):
    ledger = ExperienceLedger(tmp_path / "experience.db")
    await ledger.initialize()
    try:
        config = PilotConfig()
        monkeypatch.setattr(config, "save", lambda: None)
        server = PilotServer(config)
        server._experience_ledger = ledger
        server._co_adaptation = _Service()
        await ledger.append(ExperienceEventType.APPROVAL_REQUESTED, plan_id="plan-1", idempotency_key="requested")
        pending = PendingConfirmation(plan_id="plan-1", event=asyncio.Event(), required_indices=frozenset({0}))
        server._pending_confirms["plan-1"] = pending
        server._active_plan_id = "plan-1"
        rows_before = len(await ledger.list_events(limit=1000))

        for method, params in RPC_CALLS:
            result = await getattr(server, f"_handle_{method}")(dict(params), None)
            assert result["status"] == "ok", (method, result)

        assert server._pending_confirms == {"plan-1": pending}
        assert pending.resolved is False
        assert pending.event.is_set() is False
        assert pending.confirmed is False
        assert pending.approved_indices is None
        assert pending.input_channel == "unknown"
        assert pending.resolution_path == ""
        assert not pending.decision_lock.locked()
        assert len(await ledger.list_events(limit=1000)) == rows_before
        assert set(server._co_adaptation.calls) >= set(SERVICE_METHODS)
    finally:
        await ledger.close()


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names.add(module)
            names.update(f"{module}.{alias.name}" for alias in node.names)
    return names


def _authority_modules() -> list[Path]:
    return [
        *sorted((PILOT_ROOT / "security").rglob("*.py")),
        PILOT_ROOT / "intelligence" / "world_model.py",
        *sorted((PILOT_ROOT / "agents").rglob("*.py")),
    ]


@pytest.mark.parametrize("path", _authority_modules(), ids=lambda path: path.relative_to(PILOT_ROOT).as_posix())
def test_authority_modules_never_import_co_adaptation(path):
    imported = _imports(path)
    assert not any(name == "pilot.coadaptation" or name.startswith("pilot.coadaptation.") for name in imported)
    # Also no import by string (importlib).
    assert "pilot.coadaptation" not in path.read_text(encoding="utf-8")


def test_pilot_never_imports_the_benchmark_ablations():
    for path in sorted(PILOT_ROOT.rglob("*.py")):
        imported = _imports(path)
        assert not any("coadaptation_ablations" in name for name in imported), path
        assert "coadaptation_ablations" not in path.read_text(encoding="utf-8"), path


def test_server_imports_co_adaptation_only_inside_initialize():
    tree = ast.parse((PILOT_ROOT / "server.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
            assert not str(module).startswith("pilot.coadaptation")
    initialize = next(
        item
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PilotServer"
        for item in node.body
        if isinstance(item, ast.AsyncFunctionDef) and item.name == "initialize"
    )
    lazy = [
        node.module
        for node in ast.walk(initialize)
        if isinstance(node, ast.ImportFrom) and str(node.module).startswith("pilot.coadaptation")
    ]
    assert lazy == ["pilot.coadaptation.service"]


def test_importing_the_server_does_not_load_co_adaptation():
    code = (
        "import sys\n"
        "import pilot.server\n"
        "loaded = sorted(name for name in sys.modules if name.startswith('pilot.coadaptation'))\n"
        "assert not loaded, loaded\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=DAEMON_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
