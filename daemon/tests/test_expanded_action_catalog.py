"""End-to-end coverage for the expanded action catalog."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionType,
    ElementDetectionParams,
    FileParams,
    GitParams,
    GitResolveParams,
    MouseParams,
    WasmCallParams,
)
from pilot.agents.executor import Executor
from pilot.agents.planner import Planner
from pilot.security.audit import AuditLogger
from pilot.security.permissions import PermissionChecker
from pilot.security.validator import ActionValidator
from pilot.system.filesystem import directory_size, file_compare, file_hash


def _executor(default_config, tmp_path: Path) -> Executor:
    return Executor(
        default_config,
        ActionValidator(default_config),
        PermissionChecker(default_config),
        AuditLogger(audit_file=tmp_path / "audit.log"),
    )


def test_action_catalog_contains_157_real_actions():
    assert len(ActionType) == 159
    assert {
        ActionType.FILE_HASH,
        ActionType.FILE_COMPARE,
        ActionType.DIRECTORY_SIZE,
        ActionType.GIT_STATUS,
        ActionType.GIT_DIFF,
        ActionType.GIT_LOG,
        ActionType.GIT_BRANCH,
        ActionType.GIT_STAGE,
        ActionType.GIT_COMMIT,
        ActionType.GIT_PUSH,
    }.issubset(set(ActionType))


def test_previously_uncovered_actions_have_executor_handlers(default_config, tmp_path: Path):
    executor = _executor(default_config, tmp_path)

    assert {
        ActionType.GIT_RESOLVE,
        ActionType.PLUGIN_CALL,
        ActionType.SCREEN_DETECT_ELEMENTS,
        ActionType.WASM_CALL,
    }.issubset(executor._dispatch_table)


@pytest.mark.parametrize(
    ("action_type", "params", "expected_type"),
    [
        (ActionType.GIT_STATUS, {"repo_path": "."}, GitParams),
        (
            ActionType.GIT_RESOLVE,
            {"path": "conflicted.py", "full_block": "old", "resolved_code": "new"},
            GitResolveParams,
        ),
        (
            ActionType.SCREEN_DETECT_ELEMENTS,
            {"description": "save button", "max_elements": 5},
            ElementDetectionParams,
        ),
        (ActionType.WASM_CALL, {"tool": "wasm_tool", "args": {}}, WasmCallParams),
        (ActionType.PLUGIN_CALL, {"tool": "python_tool", "args": {}}, WasmCallParams),
    ],
)
def test_planner_parses_new_and_previously_uncovered_actions(action_type, params, expected_type):
    parsed = Planner._parse_parameters(action_type, params)
    assert isinstance(parsed, expected_type)


async def test_file_intelligence_actions_execute_real_work(tmp_path: Path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"heliox")
    second.write_bytes(b"heliox")

    digest = await file_hash(str(first))
    comparison = await file_compare(str(first), str(second))
    size = await directory_size(str(tmp_path))

    assert digest.startswith("sha256:")
    assert "Identical:" in comparison
    assert "across 2 files" in size


async def test_git_status_runs_through_executor(default_config, tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "untracked.txt").write_text("hello", encoding="utf-8")
    executor = _executor(default_config, tmp_path)
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.GIT_STATUS,
                parameters=GitParams(repo_path=str(repo)),
            )
        ]
    )

    [result] = await executor.execute(plan)

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["untracked"] == ["untracked.txt"]


async def test_git_resolve_executes_real_conflict_replacement(default_config, tmp_path: Path):
    conflicted = tmp_path / "conflicted.py"
    block = "<<<<<<< HEAD\nold\n=======\nnew\n>>>>>>> branch"
    conflicted.write_text(f"before\n{block}\nafter\n", encoding="utf-8")
    executor = _executor(default_config, tmp_path)
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.GIT_RESOLVE,
                target=str(conflicted),
                parameters=GitResolveParams(
                    path=str(conflicted),
                    full_block=block,
                    resolved_code="resolved",
                ),
            )
        ]
    )

    [result] = await executor.execute(plan)

    assert result.success is True
    assert conflicted.read_text(encoding="utf-8") == "before\nresolved\nafter\n"


async def test_screen_detection_uses_concrete_vision_handler(default_config, tmp_path: Path):
    executor = _executor(default_config, tmp_path)
    action = Action(
        action_type=ActionType.SCREEN_DETECT_ELEMENTS,
        parameters=ElementDetectionParams(description="save button", max_elements=5),
    )
    response = json.dumps(
        {
            "elements": [
                {
                    "label": "Save",
                    "action": "click",
                    "bbox": [10, 20, 30, 40],
                }
            ]
        }
    )
    with patch(
        "pilot.system.vision.screen_detect_elements",
        new=AsyncMock(return_value=response),
    ) as detector:
        output = await executor._exec_screen_detect_elements(action)

    assert output == response
    assert executor._detected_click_target == (25, 40, "Save")
    detector.assert_awaited_once()


async def test_detected_click_target_is_used_once_and_desktop_steps_are_serial(default_config, tmp_path: Path):
    executor = _executor(default_config, tmp_path)
    detect = Action(
        action_type=ActionType.SCREEN_DETECT_ELEMENTS,
        parameters=ElementDetectionParams(description="launch button", region="100,200,800,600"),
    )
    click = Action(
        action_type=ActionType.MOUSE_CLICK,
        parameters=MouseParams(),
    )
    response = json.dumps(
        {
            "elements": [
                {
                    "label": "Launch",
                    "action": "click",
                    "bbox": [10, 20, 30, 40],
                }
            ]
        }
    )

    batches = executor._analyze_dependencies([detect, click])
    assert batches == [[detect], [click]]

    with (
        patch(
            "pilot.system.vision.screen_detect_elements",
            new=AsyncMock(return_value=response),
        ),
        patch(
            "pilot.system.input_control.mouse_click",
            new=AsyncMock(return_value="clicked"),
        ) as mouse_click,
    ):
        await executor._exec_screen_detect_elements(detect)
        result = await executor._exec_mouse_click(click)

    mouse_click.assert_awaited_once_with(125, 240, "left", 1)
    assert result == "clicked (detected target: Launch)"
    assert executor._detected_click_target is None


async def test_zero_coordinate_click_fails_closed_without_detected_target(default_config, tmp_path: Path):
    executor = _executor(default_config, tmp_path)
    action = Action(action_type=ActionType.MOUSE_CLICK, parameters=MouseParams())

    with (
        patch("pilot.system.input_control.mouse_click", new_callable=AsyncMock) as mouse_click,
        pytest.raises(ValueError, match="no grounded coordinates"),
    ):
        await executor._exec_mouse_click(action)

    mouse_click.assert_not_awaited()


async def test_plugin_and_wasm_actions_use_distinct_brokers(default_config, tmp_path: Path):
    executor = _executor(default_config, tmp_path)
    registry = MagicMock()
    registry.call_tool.return_value = {"runtime": "python", "ok": True}
    registry.call_wasm_tool.return_value = {"runtime": "wasm", "ok": True}
    executor.set_plugin_registry(registry)

    plugin_output = await executor._exec_plugin_call(
        Action(
            action_type=ActionType.PLUGIN_CALL,
            parameters=WasmCallParams(tool="python_tool", args={"value": 1}),
        )
    )
    wasm_output = await executor._exec_wasm_call(
        Action(
            action_type=ActionType.WASM_CALL,
            parameters=WasmCallParams(tool="wasm_tool", args={"value": 2}),
        )
    )

    assert json.loads(plugin_output) == {"runtime": "python", "ok": True}
    assert json.loads(wasm_output) == {"runtime": "wasm", "ok": True}
    registry.call_tool.assert_called_once_with(
        "python_tool",
        {"value": 1},
        approved=True,
    )
    registry.call_wasm_tool.assert_called_once_with("wasm_tool", {"value": 2})
