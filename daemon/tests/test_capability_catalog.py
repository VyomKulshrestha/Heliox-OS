"""The public capability catalog must remain derived from runtime authorities."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from pilot.actions import ActionType
from pilot.agents.verifier import POSTCONDITION_VERIFIERS


def _generator_module():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "generate_capability_catalog.py"
    spec = importlib.util.spec_from_file_location("generate_capability_catalog", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_hashes_are_line_ending_independent(tmp_path: Path) -> None:
    module = _generator_module()
    lf = tmp_path / "lf.json"
    crlf = tmp_path / "crlf.json"
    lf.write_bytes(b'{"name": "Heliox"}\n')
    crlf.write_bytes(b'{"name": "Heliox"}\r\n')
    assert module._sha256(lf) == module._sha256(crlf)


def test_generated_capability_catalog_matches_runtime(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "capabilities.json"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "generate_capability_catalog.py"),
            "--output",
            str(output),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    generated = json.loads(output.read_text(encoding="utf-8"))

    assert generated["summary"]["action_types"] == len(ActionType) == 162
    assert generated["summary"]["mesh"]["coverage_complete"] is True
    assert generated["summary"]["mesh"]["uncovered_action_types"] == []
    assert {entry["id"] for entry in generated["actions"]} == {action_type.value for action_type in ActionType}
    assert all(entry["providers"] for entry in generated["actions"])
    assert all(entry["platform_support"]["platforms"] for entry in generated["actions"])
    assert generated["summary"]["independent_postcondition_verifiers"] > 0
    # New verifiers only add to the 22 that existed when the area registry was introduced.
    assert generated["summary"]["independent_postcondition_verifiers"] == len(POSTCONDITION_VERIFIERS) >= 22
    high_risk = {
        ActionType.PROCESS_KILL,
        ActionType.POWER_SHUTDOWN,
        ActionType.POWER_RESTART,
        ActionType.POWER_LOGOUT,
        ActionType.WINDOW_CLOSE,
        ActionType.DISK_UNMOUNT,
        ActionType.SCHEDULE_DELETE,
    }
    generated_actions = {entry["id"]: entry for entry in generated["actions"]}
    assert all(generated_actions[action.value]["verification"]["independent_postcondition"] for action in high_risk)
    assert generated["sources"]["postcondition_verifier"]["path"] == "daemon/pilot/agents/verifier.py"
    area_paths = {entry["path"] for entry in generated["sources"]["postcondition_area_modules"]}
    assert "daemon/pilot/agents/postconditions/__init__.py" in area_paths
    assert "daemon/pilot/agents/postconditions/triggers_schedules.py" in area_paths
    assert generated["plugins"]


def test_committed_capability_catalog_is_current(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    generated_path = tmp_path / "capabilities.json"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "generate_capability_catalog.py"),
            "--output",
            str(generated_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert b"\r\n" not in generated_path.read_bytes()
    assert generated_path.read_bytes() == (repo_root / "capabilities.json").read_bytes()


def test_readme_verifier_counts_come_from_the_committed_catalog() -> None:
    module = _generator_module()
    repo_root = Path(__file__).resolve().parents[2]
    catalog = json.loads((repo_root / "capabilities.json").read_text(encoding="utf-8"))
    readme = (repo_root / "README.md").read_bytes().decode("utf-8")
    summary = catalog["summary"]

    assert (
        module.readme_with_verifier_counts(
            readme, summary["independent_postcondition_verifiers"], summary["action_types"]
        )
        == readme
    )


def test_readme_verifier_row_is_rewritten_in_place_keeping_cell_widths() -> None:
    module = _generator_module()
    row = (
        "| Outcome verification | 22 actions have an independent observed post-condition verifier      "
        "| The other 140 currently rely on the executor result      |"
    )
    text = f"intro\n{row}\nend\n"

    updated = module.readme_with_verifier_counts(text, 31, 162)
    new_row = updated.split("\n")[1]

    assert "| 31 actions have an independent" in new_row
    assert "| The other 131 currently rely" in new_row
    assert [len(cell) for cell in new_row.split("|")] == [len(cell) for cell in row.split("|")]
    assert updated.startswith("intro\n") and updated.endswith("\nend\n")
