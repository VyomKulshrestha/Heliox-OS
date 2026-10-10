"""Public evidence must stay generated from current repository authorities."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from pilot.actions import ActionType


def _generate(repo_root: Path, output: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "generate_public_proof.py"),
            "--output",
            str(output),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_public_proof_exposes_evidence_boundaries(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    latest_software_evidence = sorted((repo_root / "docs" / "evidence").glob("software-benchmarks-*.json"))[-1].name
    output = tmp_path / "proof.md"
    _generate(repo_root, output)
    text = output.read_text(encoding="utf-8")

    assert "159" in text
    assert "executor result without an independent post-condition" in text
    assert "not proven live brain control" in text
    assert "result is intentionally linked rather than copied as “green”" in text
    assert "f2df192" in text
    assert "59/59 cases" in text
    assert "25,200 training" in text
    assert "Event-loop responsiveness" in text
    assert latest_software_evidence in text
    assert "Subscription-backed planning" in text
    assert "3/3 fixed planning cases" in text
    assert "side-effect-free planning only; no action was executed" in text
    assert "subscription-planning-codex-2026-08-16.json" in text


def test_committed_public_proof_is_current(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "proof.md"
    _generate(repo_root, output)
    # Line endings depend on the platform and checkout settings, not on the evidence.
    generated = output.read_bytes().replace(b"\r\n", b"\n")
    assert generated == (repo_root / "proof.md").read_bytes().replace(b"\r\n", b"\n")


def test_current_source_documentation_matches_runtime_contracts() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    capabilities = json.loads((repo_root / "capabilities.json").read_text(encoding="utf-8"))
    action_count = len(ActionType)
    independent_count = capabilities["summary"]["independent_postcondition_verifiers"]

    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    daemon_readme = (repo_root / "daemon" / "README.md").read_text(encoding="utf-8")
    architecture = (repo_root / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    neural = (repo_root / "docs" / "NEURAL_INTENT.md").read_text(encoding="utf-8")
    ipc = (repo_root / "IPC_MESSAGE_FORMATS.md").read_text(encoding="utf-8")

    assert f"{action_count} declared action types" in readme
    assert f"The other {action_count - independent_count}" in readme
    assert f"{action_count}-action system interface" in daemon_readme
    assert "system_health_review" in architecture
    assert "OpenRouter" in architecture
    assert "stage up to eight explicit autonomous goals" in neural

    server_tree = ast.parse((repo_root / "daemon" / "pilot" / "server.py").read_text(encoding="utf-8"))
    handler_counts = [
        len(node.value.keys)
        for node in ast.walk(server_tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Dict)
        and any(isinstance(target, ast.Attribute) and target.attr == "_handlers" for target in node.targets)
    ]
    assert len(handler_counts) == 1
    assert f"**{handler_counts[0]} WebSocket RPC methods**" in ipc
    assert f"**{action_count} action" in ipc
